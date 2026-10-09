"""epub 读取器：从一包 XHTML 到一段纯文本。

全部是纯函数——电子书用 `zipfile` 在内存里拼出来，不碰真书、不碰数据库。
每一条都对着真书上踩过的一个坑：spine 顺序≠文件名顺序、封面/目录混在
spine 里、纯图片页抽不出字、href 带百分号编码、GBK 编码的电子书。

**这里不测「服务层落回文本解码」**，那是 `app/services/kb_import_service.py`
的事，在 `test_kb_import.py` 里。
"""

from __future__ import annotations

import io
import posixpath
import zipfile
from urllib.parse import unquote

import pytest

from app.kb import epub
from app.kb.epub import EpubError, NotZipError, read_epub

CONTENT = "OEBPS/content.opf"

_XHTML = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>页眉不该出现</title></head>'
    "<body>{body}</body></html>"
)

_CONTAINER = (
    '<?xml version="1.0"?>\n'
    '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
    "<rootfiles>"
    f'<rootfile full-path="{CONTENT}" media-type="application/oebps-package+xml"/>'
    "</rootfiles></container>"
)


def opf_xml(
    docs: list[tuple[str, str]],
    *,
    title: str = "测试书",
    creator: str = "某作者",
    linear_no: frozenset[str] = frozenset(),
) -> str:
    """`docs` 是 `(id, href)`，**列表顺序就是 spine 顺序**。"""
    items = "\n".join(
        f'<item id="{i}" href="{h}" media-type="application/xhtml+xml"/>' for i, h in docs
    )

    def ref(i: str) -> str:
        linear = ' linear="no"' if i in linear_no else ""
        return f'<itemref idref="{i}"{linear}/>'

    refs = "\n".join(ref(i) for i, _ in docs)
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"<dc:title>{title}</dc:title><dc:creator>{creator}</dc:creator>"
        "</metadata>"
        f"<manifest>{items}</manifest><spine>{refs}</spine></package>"
    )


def build(
    docs: list[tuple[str, str, str]] | None = None,
    *,
    opf: str | None = None,
    title: str = "测试书",
    creator: str = "某作者",
    linear_no: frozenset[str] = frozenset(),
    extra: dict[str, bytes] | None = None,
    container: bytes | None = None,
) -> bytes:
    """内存造一本 epub。`docs` 是 `(id, href, body_html)`，顺序即 spine 顺序。"""
    docs = docs or []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", _CONTAINER if container is None else container)
        zf.writestr(
            CONTENT,
            opf
            if opf is not None
            else opf_xml(
                [(i, h) for i, h, _ in docs], title=title, creator=creator, linear_no=linear_no
            ),
        )
        for _, href, body in docs:
            # 成员名用的是**解码后**的路径，正是真书里的样子：href 里的
            # 百分号编码只是引用方式，不是文件名。
            name = posixpath.join("OEBPS", unquote(href.split("#", 1)[0]))
            zf.writestr(name, _XHTML.format(body=body))
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return buf.getvalue()


class TestIsZip:
    def test_按魔数判而不是扩展名(self) -> None:
        assert epub.is_zip(b"PK\x03\x04anything")
        assert not epub.is_zip("这是一段中文".encode())
        assert not epub.is_zip(b"Portable text")
        assert not epub.is_zip(b"")


class TestReadEpub:
    def test_按_spine_顺序而不是文件名顺序(self) -> None:
        """真书上两本书的章节文件名顺序与阅读顺序不一致。"""
        doc = read_epub(
            build(
                [
                    ("c1", "z_part.xhtml", "<p>第一篇</p>"),
                    ("c2", "a_part.xhtml", "<p>第二篇</p>"),
                ]
            )
        )
        assert doc.text == "第一篇\n\n第二篇"

    def test_封面目录版权页按名字跳过并记名(self) -> None:
        doc = read_epub(
            build(
                [
                    ("cover", "cover.xhtml", "<img src='c.jpg'/>"),
                    ("nav", "n1.xhtml", "<p>目录</p>"),  # 靠 idref 命中，文件名无关
                    ("c1", "c1.xhtml", "<p>正文</p>"),
                    ("c2", "copyright.xhtml", "<p>版权所有</p>"),
                ]
            )
        )
        assert doc.text == "正文"
        assert doc.skipped == ["cover.xhtml", "n1.xhtml", "copyright.xhtml"]

    def test_相似词不误伤(self) -> None:
        """coverage 里有 cover，但它是一章正文。多跳一个词就是丢正文。"""
        doc = read_epub(build([("c1", "coverage.xhtml", "<p>覆盖率这一章</p>")]))
        assert doc.text == "覆盖率这一章"
        assert doc.skipped == []

    def test_linear_no_的篇目不进正文(self) -> None:
        doc = read_epub(
            build(
                [
                    ("c1", "c1.xhtml", "<p>正文</p>"),
                    ("pop", "pop.xhtml", "<p>弹出注释，不属于阅读顺序</p>"),
                ],
                linear_no=frozenset({"pop"}),
            )
        )
        assert doc.text == "正文"
        # 它不是「被跳过」，本来就不在阅读顺序里——不该占用用户的注意力
        assert doc.skipped == []

    def test_纯图片篇目计入空篇且不留空段(self) -> None:
        """红楼梦 137 篇里 11 篇整篇只有一个 img。"""
        doc = read_epub(
            build(
                [
                    ("c1", "pic.xhtml", "<div><img src='p.jpg' alt=''/></div>"),
                    ("c2", "c2.xhtml", "<p>正文</p>"),
                ]
            )
        )
        assert doc.text == "正文"
        assert doc.empty_docs == 1

    def test_块级标签补空行_行内标签原样(self) -> None:
        doc = read_epub(
            build(
                [
                    (
                        "c1",
                        "c1.xhtml",
                        "<h1>标题</h1><p>第一段</p><p>第二段<br/>换行</p>"
                        "<div><span>内联</span>也内联</div>",
                    )
                ]
            )
        )
        assert doc.text == "标题\n\n第一段\n\n第二段\n换行\n\n内联也内联"

    def test_script_style_与注音被丢掉(self) -> None:
        doc = read_epub(
            build(
                [
                    (
                        "c1",
                        "c1.xhtml",
                        "<p>正文<script>var x=1</script><style>p{color:red}</style>"
                        "<ruby>汉<rt>hàn</rt></ruby>字</p>",
                    )
                ]
            )
        )
        # 页眉的 title 也必须丢掉——它在 head 里，整块不算正文
        assert doc.text == "正文汉字"

    def test_实体与不间断空格(self) -> None:
        doc = read_epub(build([("c1", "c1.xhtml", "<p>a&amp;b&nbsp;c&hellip;</p>")]))
        assert doc.text == "a&b c…"

    def test_GBK_编码的篇目也能读(self) -> None:
        """中文 Windows 上做的电子书真的有 GBK 的。解码复用 decode_bytes。"""
        body = "<p>汉字正文</p>"
        blob = build(
            [],
            opf=opf_xml([("c1", "gbk.xhtml")]),
            extra={"OEBPS/gbk.xhtml": _XHTML.format(body=body).encode("gbk")},
        )
        assert read_epub(blob).text == "汉字正文"

    def test_href_带百分号编码与片段也能定位(self) -> None:
        blob = build([("c1", "%E7%AC%AC1%E7%AB%A0.xhtml#top", "<p>第一章</p>")])
        assert read_epub(blob).text == "第一章"

    def test_书名作者取自电子书元数据(self) -> None:
        doc = read_epub(
            build([("c1", "c1.xhtml", "<p>正文</p>")], title="真书名", creator="真作者")
        )
        assert (doc.title, doc.author) == ("真书名", "真作者")

    def test_元数据没有命名空间前缀也能取到(self) -> None:
        """OPF 的命名空间各家不同，必须按本地名匹配。"""
        opf = (
            '<?xml version="1.0"?><package version="3.0">'
            "<metadata><title>无前缀书名</title><creator>无前缀作者</creator></metadata>"
            '<manifest><item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/></manifest>'
            '<spine><itemref idref="c1"/></spine></package>'
        )
        doc = read_epub(build([("c1", "c1.xhtml", "<p>正文</p>")], opf=opf))
        assert (doc.title, doc.author) == ("无前缀书名", "无前缀作者")

    def test_spine_指向的篇目缺失时跳过而不是整本拒绝(self) -> None:
        opf = (
            '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
            "<metadata/>"
            '<manifest><item id="a" href="missing.xhtml" media-type="application/xhtml+xml"/>'
            '<item id="b" href="c1.xhtml" media-type="application/xhtml+xml"/></manifest>'
            '<spine><itemref idref="a"/><itemref idref="b"/></spine></package>'
        )
        doc = read_epub(build([("c1", "c1.xhtml", "<p>在的那一篇</p>")], opf=opf))
        assert doc.text == "在的那一篇"
        assert doc.skipped == ["missing.xhtml（文件缺失）"]


class TestRejection:
    """失败必须是 `EpubError` 且消息能直接展示——它会被翻成 400 的 detail。"""

    def test_不是_zip(self) -> None:
        with pytest.raises(NotZipError, match="不是一个有效的 zip"):
            read_epub("PK\x03\x04 这其实是一段以 PK 开头的文本".encode())

    def test_zip_但不是_epub(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("readme.txt", "hello")
        with pytest.raises(EpubError, match="不是 epub"):
            read_epub(buf.getvalue())

    def test_结构文件不是合法_XML(self) -> None:
        with pytest.raises(EpubError, match="无法解析"):
            read_epub(build([("c1", "c1.xhtml", "<p>正文</p>")], container=b"<container"))

    def test_全部是图片的电子书(self) -> None:
        with pytest.raises(EpubError, match="没有抽到"):
            read_epub(build([("c1", "c1.xhtml", "<img src='p.jpg'/>")]))

    def test_解压总量超限(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(epub, "MAX_UNCOMPRESSED", 10)
        with pytest.raises(EpubError, match="上限"):
            read_epub(build([("c1", "c1.xhtml", "<p>" + "字" * 100 + "</p>")]))

    def test_篇数超限(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(epub, "MAX_DOCS", 2)
        docs = [(f"c{i}", f"c{i}.xhtml", f"<p>第{i}篇</p>") for i in range(3)]
        with pytest.raises(EpubError, match="上限"):
            read_epub(build(docs))

    def test_篇目编码既不是_UTF8_也不是_GBK(self) -> None:
        body = "<p>正文</p>".encode("utf-16")  # 不是 UTF-8 也不是 GBK
        blob = build([], opf=opf_xml([("c1", "u.xhtml")]), extra={"OEBPS/u.xhtml": body})
        with pytest.raises(EpubError, match="无法解码"):
            read_epub(blob)
