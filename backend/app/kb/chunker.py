"""把一整本书切成 chunk。

**这里唯一要守住的事：不丢字。** 切分器的失败模式是「悄悄少了一段」——
检索时少一条证据，不抛异常、日志正常，也没有人会去数。所以它有一条严格
不变量，由测试守着：**各块 body 拼接后去空白 == 源文本去空白**。

三个刻意的取舍：

1. **不做 NFKC 归一。** `comment_service.normalize()` 会把全角标点「，！？」
   转成半角「,!?」——那是为抖音短评论写的。用在中文长文上会把标点改坏，
   而且和已有语料不一致。这里只去零宽字符、统一换行、折叠空白。
2. **不做常规重叠。** 边界本来就落在句末，没有句子被切断，重叠只是把同一段
   正文存两次、让差分难以解释。只有单句长到必须硬切时才回退一段做重叠，
   并且重叠只加在 `text` 上，`body` 保持干净，不变量因此仍然成立。
3. **chunk_id 只用内容派生**，不用「文件名 + 序号」。序号派生在用户往中间插
   一段之后会整体错位：Qdrant 按 point_id 原地覆盖，点数不变、日志正常，
   而已经写进案例快照的 chunk_id 会静默指到别的段落上。
"""

from __future__ import annotations

import codecs
import hashlib
import re
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from app.constants import Library

# --- 尺寸 -----------------------------------------------------------------
#: 装箱目标。一段完整论证的典型长度：300–500 字。
TARGET_CHARS = 350
#: 硬上限。超过就切——再长的话向量会被稀释，与任何具体查询都只有中等相似度。
MAX_CHARS = 600
#: 低于此值的尾块并进前一块（能并得下的话）。太短的块几乎不含语义。
MIN_CHARS = 60
#: 全文下限，与 loader 对语料 `text` 的要求一致。
MIN_KEEP_CHARS = 8
#: 硬切时给下一块留的重叠（只为不把一个句子拦腰截断到两边都读不通）。
OVERLAP_CHARS = 80

#: 句末标点。连续出现（「……」「？！」）算同一个句末，不切出一字句。
_SENT_END = "。！？；…!?;"
#: 次级切分点。句末切不动时才用。
_COMMA = "，、,"

_NEWLINES = re.compile(r"\r\n|\r")
_ZERO_WIDTH = re.compile("[​-‏  ﻿]")
_SPACES = re.compile("[ \t　]+")
_BLANKS = re.compile(r"\n{3,}")
_CJK = re.compile("[㐀-䶿一-鿿　-〿＀-￯]")
_BRACKETS = re.compile("[《》〈〉\\s]")

_PREFIX: dict[Library, str] = {
    Library.PSYCHOLOGY: "psy",
    Library.LITERATURE: "lit",
    Library.POETRY: "poem",
}


class EmptyDocumentError(RuntimeError):
    """文件里没有可切分的正文。

    这是**不能静默通过**的情况：切出 0 块然后报「导入成功」，在界面上和
    真的成功长得一模一样。
    """


@dataclass(frozen=True, slots=True)
class SplitConfig:
    target_chars: int = TARGET_CHARS
    max_chars: int = MAX_CHARS
    min_chars: int = MIN_CHARS
    overlap_chars: int = OVERLAP_CHARS


@dataclass(slots=True)
class Piece:
    index: int
    #: 落库用。硬切产生的块会带上重叠前缀。
    text: str
    #: 不含重叠前缀的纯净正文——「不丢字」校验比的是它。
    body: str
    paragraph: int
    hard_split: bool = False
    overlap_chars: int = 0


@dataclass(slots=True)
class SplitReport:
    pieces: list[Piece] = field(default_factory=list)
    paragraphs: int = 0
    hard_splits: int = 0
    comma_splits: int = 0
    too_short_pieces: int = 0
    cjk_ratio: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(len(p.body) for p in self.pieces)


@dataclass(slots=True)
class _Unit:
    text: str
    body: str
    hard: bool = False


# ----------------------------------------------------------------------
# 解码
# ----------------------------------------------------------------------

#: 顺序即优先级。`utf-8-sig` 必须先于 `utf-8`（否则 BOM 会变成一个正文字符）。
_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")


def decode_bytes(blob: bytes) -> str:
    """按 UTF-8 → GBK 的顺序试解码。

    中文 Windows 上记事本存的 txt 默认是 GBK，这不是边角情况，是常见情况。

    **不用 `errors="ignore"`。** 静默丢字节会让同一个文件第二次导入切出
    不同的正文——于是「重复导入应该全部跳过」变成「全部新增」，而两次导入
    的差别只在于几个被丢掉的字节，谁也不会想到去看那里。
    """
    if blob.startswith(codecs.BOM_UTF8):
        blob = blob[len(codecs.BOM_UTF8) :]
    for encoding in _ENCODINGS:
        try:
            return blob.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    raise EmptyDocumentError(
        "文件既不是 UTF-8 也不是 GBK/GB18030，无法解码。"
        "请用编辑器另存为 UTF-8 后重试。"
    )


# ----------------------------------------------------------------------
# 切分
# ----------------------------------------------------------------------


def _clean(raw: str) -> str:
    """只做无损清洗：统一换行、去零宽字符、折叠空白。**不做 NFKC。**"""
    text = _NEWLINES.sub("\n", raw)
    text = _ZERO_WIDTH.sub("", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()


def _paragraphs(text: str) -> Iterator[str]:
    """按空行分段，段内把硬折行接回一行。

    **硬折行必须接回去。** 否则一整本每 40 字折一次行、段间才有空行的书，
    会变成一个巨型段落，后面的按句装箱和硬切全线开花，切出来的块全是
    半句话。段内直接拼接是安全的：中文不需要词间空格，而按句装箱本来就会
    在句子边界重新切。
    """
    for block in text.split("\n\n"):
        lines = [ln for ln in block.split("\n") if ln]
        if not lines:
            continue
        yield _join_lines(lines)


def _join_lines(lines: Sequence[str]) -> str:
    """拼接时只在中英文交界处补空格，避免把 "hello" + "world" 粘成 "helloworld"。"""
    out = ""
    for line in lines:
        if out and not (_CJK.search(out[-1]) or _CJK.search(line[0])):
            out += " "
        out += line
    return out


def _split_after(text: str, markers: str) -> list[str]:
    """按一组标点切，标点留在前一段末尾。连续终止符算同一个句末。"""
    parts: list[str] = []
    buf: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        buf.append(ch)
        if ch in markers:
            while i + 1 < n and text[i + 1] in markers:
                i += 1
                buf.append(text[i])
            parts.append("".join(buf))
            buf = []
        i += 1
    if buf:
        parts.append("".join(buf))
    return [p for p in parts if p]


def _pack(units: Sequence[_Unit], cfg: SplitConfig) -> list[_Unit]:
    """贪心装箱：尽量接近 target，绝不超过 max。

    打包用 `body` 计长度——重叠前缀不进长度，否则「接近 target」会随着
    重叠越滚越短。
    """
    out: list[_Unit] = []
    cur: list[_Unit] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if cur:
            out.append(
                _Unit(
                    text="".join(u.text for u in cur),
                    body="".join(u.body for u in cur),
                    # 硬切标记必须跟着一起合并：不传播的话，出厂的每一块
                    # 都是 `hard_split=False`（因为多句段落一律经过这里），
                    # 这个字段就成了一个恒为假的谎话。
                    hard=any(u.hard for u in cur),
                )
            )
            cur, cur_len = [], 0

    for unit in units:
        size = len(unit.body)
        if cur and (cur_len + size > cfg.max_chars or cur_len >= cfg.target_chars):
            flush()
        cur.append(unit)
        cur_len += size
    flush()
    return out


def _hard_cut(text: str, cfg: SplitConfig, report: SplitReport) -> list[_Unit]:
    """无标点长串的最后手段。切点回退一段做重叠，让下一块开头有上下文。

    **计数按切出来的块算，不按「硬切了几次」算。** 一次调用可以切出上百块
    （5 万字无标点就是一整句），记成 1 的话，下面那条「疑似非正文文本」的
    警告永远不会触发——而那正是这条警告存在的唯一理由。
    """
    step = max(1, cfg.max_chars)
    out: list[_Unit] = []
    start = 0
    take_overlap = False
    while start < len(text):
        body = text[start : start + step]
        prefix = ""
        if take_overlap and cfg.overlap_chars > 0:
            prefix = text[max(0, start - cfg.overlap_chars) : start]
        out.append(_Unit(text=prefix + body, body=body, hard=True))
        report.hard_splits += 1
        start += step
        take_overlap = True
    return out


def _split_long(sentence: str, cfg: SplitConfig, report: SplitReport) -> list[_Unit]:
    """一个超长句子：先按逗号拆，拆不动才硬切。"""
    units: list[_Unit] = []
    for part in _split_after(sentence, _COMMA):
        if len(part) <= cfg.max_chars:
            units.append(_Unit(text=part, body=part))
            continue
        units.extend(_hard_cut(part, cfg, report))
    if len(units) > 1:
        report.comma_splits += 1
    return units


def _emit(paragraph: str, index: int, cfg: SplitConfig, report: SplitReport) -> list[Piece]:
    """把一个段落变成若干块。"""
    if len(paragraph) <= cfg.max_chars:
        return [Piece(index=index, text=paragraph, body=paragraph, paragraph=index)]

    units: list[_Unit] = []
    for sentence in _split_after(paragraph, _SENT_END):
        if len(sentence) <= cfg.max_chars:
            units.append(_Unit(text=sentence, body=sentence))
        else:
            units.extend(_split_long(sentence, cfg, report))

    packed = _pack(units, cfg)
    pieces = [
        Piece(
            index=index,
            text=u.text,
            body=u.body,
            paragraph=index,
            hard_split=u.hard,
            # 两者之差**就是**重叠前缀的长度：`text` 比 `body` 多出来的
            # 那一截只可能是硬切回退出来的上下文。
            overlap_chars=len(u.text) - len(u.body),
        )
        for u in packed
    ]
    # 尾块太短就并回去——一句话的开头孤零零一块，几乎不含语义，
    # 却照样占一次嵌入和一次检索候选位。
    if len(pieces) >= 2 and len(pieces[-1].body) < cfg.min_chars:
        tail = pieces.pop()
        head = pieces[-1]
        if len(head.body) + len(tail.body) <= cfg.max_chars:
            pieces[-1] = Piece(
                index=head.index,
                text=head.text + tail.text,
                body=head.body + tail.body,
                paragraph=head.paragraph,
                hard_split=head.hard_split,
                overlap_chars=head.overlap_chars + tail.overlap_chars,
            )
        else:
            report.too_short_pieces += 1
            pieces.append(tail)
    return pieces


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    sample = text[:4000]
    return sum(1 for ch in sample if _CJK.search(ch)) / len(sample)


def split_document(raw: str, cfg: SplitConfig | None = None) -> SplitReport:
    """切分一篇长文。纯函数，同一输入永远得到同一结果。"""
    config = cfg or SplitConfig()
    text = _clean(raw)
    if len(text) < MIN_KEEP_CHARS:
        raise EmptyDocumentError("文件里没有可切分的正文（去空白后不足 8 字）")

    report = SplitReport()
    pieces: list[Piece] = []
    for paragraph in _paragraphs(text):
        for piece in _emit(paragraph, report.paragraphs, config, report):
            pieces.append(piece)
        report.paragraphs += 1

    if not pieces:
        raise EmptyDocumentError("切分后没有得到任何正文片段")

    for i, piece in enumerate(pieces):
        piece.index = i
    report.pieces = pieces
    report.cjk_ratio = _cjk_ratio(text)

    if report.cjk_ratio < 0.3:
        report.warnings.append(
            f"正文里中文字符只占 {report.cjk_ratio:.0%}，可能解码错了，也可能这不是中文文本"
        )
    if report.hard_splits and report.hard_splits > len(pieces) * 0.2:
        report.warnings.append(
            f"{report.hard_splits} 处被硬切（超过块数的 20%），疑似缺少标点的非正文文本"
        )
    if len(text) < 200:
        report.warnings.append(f"全文仅 {len(text)} 字，切出 {len(pieces)} 块——请确认文件是否正确")
    return report


def _norm_work(work: str) -> str:
    """书名归一：NFKC、去《》与空白。

    用户第二次把书名写成「《存在与时间》」而不是「存在与时间」时，
    chunk_id 必须不变，否则同一本书会被当成两份各导一遍。
    """
    return _BRACKETS.sub("", unicodedata.normalize("NFKC", work or ""))


def make_chunk_id(library: Library, work: str, text: str) -> str:
    """内容派生 id：同一段正文 + 同一本书 + 同一个库 → 同一个 id。

    **`work` 必须进哈希**：两本书引用同一段原文时（二手研究引原著），
    不含 work 会撞成同一条，后导入的把前一条的书名作者一并改写，
    原著那条消失且没有任何告警。

    **`library` 必须进哈希**：同一段文字导进两个库会撞 id。跨库查重是
    `loader.load_all()` 的职责，导入路径不经它，撞了不会报错。

    **文件内容哈希绝不能进 id**：那样改一个错别字会让整本书换一批新 id，
    旧的全部变成孤儿、新的全部重嵌。
    """
    material = "\x00".join((str(library), _norm_work(work), text))
    digest = hashlib.blake2b(material.encode("utf-8"), digest_size=16).hexdigest()
    return f"{_PREFIX[library]}:imp:{digest}"
