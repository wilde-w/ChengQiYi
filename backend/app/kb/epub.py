"""从 epub（zip 容器）里抽出正文。

知识库导入的入口只认识「一段字节」，而 epub 是**一包** XHTML：正文按
`spine` 顺序散在若干篇里，还混着封面、目录、版权页这些不会有人拿来做
研究的页面。这个模块的职责就一件事——把这一包变成一段纯文本，之后
切分 / 打标 / 嵌入 / 入库全部复用既有链路，与上传一个 .txt 没有区别。

四条踩过才知道的规矩：

1. **顺序看 spine，不看文件名。** 实测三本书里有两本的章节文件名顺序
   与阅读顺序不一致（`Text/part10.xhtml` 排在 `part2` 前面）。参考实现
   靠文件名字典序遍历，对着真书就会把章节打乱。
2. **链接用 posixpath 解。** zip 成员名永远以 `/` 分隔，而这是台 Windows
   机器——用 `os.path.join` 会拼出 `OEBPS\\Text\\cover.xhtml`，然后
   `KeyError`。href 还要 `unquote`（中文书名常被百分号编码）。
3. **纯图片页必须容忍。** 红楼梦 137 篇里 11 篇整篇只有一个 `<img>`。
   它们会抽出空字符串——跳过计数，不报错，也不产生空段。
4. **只做无损抽取，不做编辑。** 章节标题保留成段、脚注标记保留、图片
   丢弃、script/style/ruby 注音丢弃。封面/目录这类页只按**文件名**跳过
   一个极小的白名单（见 `_SKIP_NAMES`）：判定正文与版权页的区别没有
   可靠规则，而猜错的代价（丢掉真实正文）远大于留着两页版权声明。

安全边界：解压总量与篇数都有硬上限（zip 炸弹），超了在**读之前**就报错；
所有失败都是 `EpubError`（用户可修，服务层转成 400 文案）。
"""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import unquote
from xml.etree import ElementTree as ET

from app.kb.chunker import EmptyDocumentError, decode_bytes

#: 解压总量的硬上限。epub 里通常大半是图片，正常电子书远小于此数；
#: 超了基本只有一种可能：这不是电子书，是精心构造的 zip 炸弹。
MAX_UNCOMPRESSED = 64 * 1024 * 1024
#: spine 篇数上限。红楼梦 137 篇，正常电子书到不了 2000 篇。
MAX_DOCS = 2000

CONTAINER_PATH = "META-INF/container.xml"

#: 按名字跳过的「页」。**故意短**：多匹配一个词的代价是丢掉正文，而
#: 少匹配一个的时间代价只是两页目录进库。分隔符后跟后缀的那种（cover-image）
#: 也算，而 coverage 这种同前缀词不算。
_SKIP_NAMES = re.compile(r"^(cover|title|titlepage|copyright|toc|nav)([-_].*)?$", re.I)

#: 块级标签：闭合时补一个空行，正好是切分器认的段落边界。
_BLOCK = frozenset(
    {
        "p",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "tr",
        "td",
        "th",
        "blockquote",
        "section",
        "article",
        "figcaption",
        "dd",
        "dt",
        "pre",
        "hr",
    }
)
#: 整块内容都不算正文的标签。`rt`/`rp` 是 ruby 注音（中文书里是拼音），
#: 留着会让同一句话出现两遍。
_SKIP_TAGS = frozenset({"script", "style", "head", "title", "rt", "rp"})

_SPACES = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")


class EpubError(ValueError):
    """用户可修的 epub 问题。消息要能直接展示。"""


class NotZipError(EpubError):
    """头两个字节是 PK，但根本不是 zip。

    纯文本恰好以 "PK" 开头是可能的，调用方应当**落回文本解码**而不是报错。
    """


@dataclass(slots=True)
class EpubDocument:
    text: str
    #: 电子书自带的元数据，仅当表单里没填时才用。
    title: str = ""
    author: str = ""
    #: 被跳过的篇目名（封面/目录/缺失文件…），用于给用户一句提示。
    skipped: list[str] = field(default_factory=list)
    spine_docs: int = 0
    #: 抽不出文字的篇目（纯图片页）。
    empty_docs: int = 0


def is_zip(blob: bytes) -> bool:
    """是不是 zip 容器。**判据只有魔数**——扩展名不可信，也不该信。"""
    return blob[:2] == b"PK"


def read_epub(blob: bytes) -> EpubDocument:
    """epub 字节 → 正文 + 元数据。读不出正文时抛 `EpubError`。"""
    try:
        zf = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile as exc:
        raise NotZipError("文件头是 PK 但不是一个有效的 zip 压缩包。") from exc

    parts: list[str] = []
    skipped: list[str] = []
    empty = 0
    total = 0

    with zf:
        try:
            container = zf.read(CONTAINER_PATH)
        except KeyError as exc:
            raise EpubError(
                "这是一个 zip 压缩包，但不是 epub：里面没有 META-INF/container.xml。"
                "知识库接受纯文本（.txt/.md）与 epub 电子书；普通压缩包请先解压。"
            ) from exc

        opf_path = _opf_path(container)
        opf = _read_member(zf, opf_path, "epub 的 OPF（内容清单）")
        root = _parse_xml(opf, opf_path)
        title, author = _metadata(root)
        entries = _spine_entries(zf, opf_path, root)

        for idref, name in entries:
            if _skip_name(idref) or _skip_name(posixpath.basename(name)):
                skipped.append(posixpath.basename(name) or idref)
                continue
            try:
                info = zf.getinfo(name)
            except KeyError:
                # spine 指向的文件在包里不存在。计入跳过而不是报错：
                # 缺一篇的电子书仍然可以研究，而整本拒绝会让人一头雾水。
                skipped.append(f"{posixpath.basename(name) or idref}（文件缺失）")
                continue
            total += info.file_size
            if total > MAX_UNCOMPRESSED:
                raise EpubError(
                    f"电子书解压后超过 {MAX_UNCOMPRESSED // 1024 // 1024}MB 上限，"
                    "已停止读取。这个文件可能不是正常的电子书。"
                )
            try:
                raw = zf.read(name)
            except zipfile.BadZipFile:
                skipped.append(f"{name}（已损坏）")
                continue

            try:
                html = decode_bytes(raw)
            except EmptyDocumentError as exc:
                raise EpubError(f"电子书里的 {name} 既不是 UTF-8 也不是 GBK，无法解码。") from exc

            parser = _TextExtractor()
            parser.feed(html)
            parser.close()
            text = parser.text()
            if text:
                parts.append(text)
            else:
                empty += 1

    if not parts:
        raise EpubError(
            "这本电子书里没有抽到任何文字——可能是纯图片的影印版或漫画。"
            "若确定它是有正文的，可以把正文复制出来另存为 .txt 再导入。"
        )

    return EpubDocument(
        text="\n\n".join(parts),
        title=title,
        author=author,
        skipped=skipped,
        spine_docs=len(entries),
        empty_docs=empty,
    )


# ----------------------------------------------------------------------
# 结构：容器 → OPF → spine
# ----------------------------------------------------------------------


def _local(tag: str) -> str:
    """去掉命名空间前缀后的标签名（小写）。

    **必须按本地名匹配。** 各书声明的 OPF 命名空间不一样，用 `opf:title`
    这种带前缀的 selector 在没声明该前缀的书上会一个都找不到。
    """
    return tag.rsplit("}", 1)[-1].lower()


def _find(root: ET.Element, name: str) -> ET.Element | None:
    return next((el for el in root.iter() if _local(el.tag) == name), None)


def _parse_xml(raw: bytes, what: str) -> ET.Element:
    try:
        return ET.fromstring(raw)
    except ET.ParseError as exc:
        raise EpubError(
            f"epub 的结构文件无法解析（不是合法的 XML），文件可能已损坏：{what}"
        ) from exc


def _read_member(zf: zipfile.ZipFile, name: str, what: str) -> bytes:
    try:
        return zf.read(name)
    except KeyError as exc:
        raise EpubError(f"epub 结构不完整：找不到 {what}（{name}）。") from exc


def _opf_path(container: bytes) -> str:
    root = _parse_xml(container, CONTAINER_PATH)
    node = _find(root, "rootfile")
    path = (node.get("full-path") if node is not None else "") or ""
    if not path:
        raise EpubError("epub 的 container.xml 里没有 rootfile，无法定位内容清单。")
    return path


def _metadata(root: ET.Element) -> tuple[str, str]:
    """取 `dc:title` 与 `dc:creator`。取第一个非空值——多书名时那是正题。"""
    meta = _find(root, "metadata")
    if meta is None:
        return "", ""
    title = author = ""
    for el in meta.iter():
        text = (el.text or "").strip()
        if not text:
            continue
        tag = _local(el.tag)
        if tag == "title" and not title:
            title = text
        elif tag == "creator" and not author:
            author = text
    return title, author


def _spine_entries(zf: zipfile.ZipFile, opf_path: str, root: ET.Element) -> list[tuple[str, str]]:
    """按 spine 顺序返回 `(idref, zip 成员路径)`。

    `linear="no"` 的篇目（规范里就是「不属于主线阅读顺序」，目录页常这么标）
    直接不返回——比按文件名猜准得多。
    """
    manifest_el = _find(root, "manifest")
    spine_el = _find(root, "spine")
    if manifest_el is None or spine_el is None:
        raise EpubError("epub 的内容清单里缺 manifest 或 spine，无法确定阅读顺序。")

    media: dict[str, str] = {}
    hrefs: dict[str, str] = {}
    for item in manifest_el:
        if _local(item.tag) != "item":
            continue
        item_id = item.get("id") or ""
        if item_id:
            hrefs[item_id] = item.get("href") or ""
            media[item_id] = item.get("media-type") or ""

    base = posixpath.dirname(opf_path)
    entries: list[tuple[str, str]] = []
    for itemref in spine_el:
        if _local(itemref.tag) != "itemref" or (itemref.get("linear") or "").lower() == "no":
            continue
        idref = itemref.get("idref") or ""
        href = hrefs.get(idref)
        if not href:
            continue
        if not _is_markup(media.get(idref, ""), href):
            continue
        entries.append((idref, _resolve(base, href)))

    if len(entries) > MAX_DOCS:
        raise EpubError(
            f"电子书的 spine 有 {len(entries)} 篇，超过 {MAX_DOCS} 篇上限，已停止读取。"
        )
    return entries


def _is_markup(media_type: str, href: str) -> bool:
    if media_type:
        return "html" in media_type.lower()
    return href.lower().endswith((".xhtml", ".html", ".htm"))


def _resolve(base: str, href: str) -> str:
    """OPF 相对路径 → zip 成员名。片段与查询串要丢掉，百分号编码要还原。"""
    clean = href.split("#", 1)[0].split("?", 1)[0]
    return posixpath.normpath(posixpath.join(base, unquote(clean)))


def _skip_name(raw: str) -> bool:
    stem = posixpath.splitext(posixpath.basename(raw))[0]
    return bool(_SKIP_NAMES.match(raw) or _SKIP_NAMES.match(stem))


# ----------------------------------------------------------------------
# HTML → 文本
# ----------------------------------------------------------------------


class _TextExtractor(HTMLParser):
    """块级边界换行、行内内容直取、脚本与注音丢掉。

    用 `html.parser` 而不是引入依赖：需求只有「读文字」，不是解析 DOM。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag == "br":
            self._out.append("\n")
        elif tag in _BLOCK:
            self._out.append("\n\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # XHTML 里的 <br/>、<hr/> 都走这里，不会经过 handle_starttag。
        if tag == "br":
            self._out.append("\n")
        elif tag in _BLOCK:
            self._out.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK:
            self._out.append("\n\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._out.append(data)

    def text(self) -> str:
        raw = "".join(self._out).replace("\xa0", " ")
        raw = _SPACES.sub(" ", raw)
        lines = (ln.strip() for ln in raw.split("\n"))
        return _BLANKS.sub("\n\n", "\n".join(lines)).strip()
