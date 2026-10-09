"""LLM 打标 + 校验 + 降级。

**这个文件存在的唯一理由是：标签张冠李戴不会报错。**

第 3 块的正文配第 5 块的标签，程序照样跑完、照样入库、日志一行不差；
坏掉的是检索——用户查「此在」查出一段讲落花的东西，而没有人会去查
标签是怎么分配错的。所以这里的校验全部是**机械的、不看语义的**：
子串、闭集、集合相等。这类校验成本接近零，挡掉的却是最贵的错误。

对齐防御按这个顺序失效，每一层都在上一层失效时才起作用：

1. 请求里每项带短整数 `i`，**绝不按数组下标 zip**——模型换个顺序返回是
   完全合法的，而按位置 zip 会把整批标签整体错位一格，且毫无痕迹；
2. `quote` 回显做锚点：如果它命中同批**另一条**的正文，说明模型在乱序
   作答，整批作废；
3. `set(返回的 i) != set(请求的 i)` 或出现越界/重复 → 整批 suspect，
   **不部分接受**；
4. `finish_reason == "length"` → 整批重试。JSON 能解析不等于内容完整，
   被 max_tokens 截断的最后一个对象可能恰好是合法 JSON。

**校验的粒度是「值」而不是「条」。** 一个越界的 emotion 只丢掉那一个词，
不丢整条的 LLM 输出——剩下的 label 仍然是有效的，而整条作废会让降级率
虚高，掩盖真正的问题（prompt 写坏了）。只有 `quote` 对不上才作废整条，
因为那是身份问题，不是质量问题。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from app.constants import Library
from app.kb.chunker import Piece, make_chunk_id
from app.kb.lexical_tags import (
    MAX_EMOTION,
    MAX_IMAGERY,
    MAX_KEYWORDS,
    ChunkTags,
    lexical_tags,
)
from app.kb.schema import KBChunk
from app.logging_conf import get_logger
from app.prompts import tag_chunks as prompt
from app.providers.base import ChatModel, ProviderError
from app.providers.openai_compat import parse_json_response

log = get_logger(__name__)

#: 每批几条。与 prompt 侧保持一致。
BATCH_SIZE = prompt.BATCH_SIZE

#: 连续多少次「不可重试」的 provider 错误就中止整本书。
#: 鉴权失败、模型名写错、额度耗尽都属于这一类：**配置错误不会自愈**，
#: 硬跑下去只是把时间浪费掉，还会在日志里刷出几百行同样的错误。
_NON_RETRYABLE_LIMIT = 3

#: 标签里不许出现的字符。它们会破坏 `text_for_embedding()` 的模板结构——
#: **一个标签就能伪造出「｜情感：x」**，让这段文字凭一个凭空捏造的情感
#: 被检索到。同理不能有换行。
_FORBIDDEN = frozenset("｜《》〈〉：、\n\r\t")

#: `type` 的闭集。库里已有的取值 + 导入场景需要的少数几个。
TYPE_VOCAB: dict[Library, tuple[str, ...]] = {
    Library.PSYCHOLOGY: ("论述", "实证", "综述", "评论"),
    Library.LITERATURE: ("散文", "议论", "史传", "小说", "诗词"),
    Library.POETRY: ("诗", "词", "曲", "赋"),
}

_TYPE_DEFAULT: dict[Library, str] = {
    Library.PSYCHOLOGY: "论述",
    Library.LITERATURE: "散文",
    Library.POETRY: "诗",
}


class TaggingError(RuntimeError):
    """打标大面积失败。**此时必须什么都不写。**

    写进半垃圾的代价远大于不写：600 条只有词法标签的块入库后，会被
    每一次检索当作候选召回成噪声证据，而且回滚要手工按 work 清。
    什么都没写是可恢复的；写进去就是慢性中毒。
    """


class TaggingCancelled(RuntimeError):
    """用户取消了这次导入。"""


@dataclass(slots=True)
class TagReport:
    total: int = 0
    tagged: int = 0
    fallback: int = 0
    dropped: int = 0
    deduped: int = 0
    batches: int = 0
    failed_batches: int = 0
    dropped_tags: Counter[str] = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)

    @property
    def failure_ratio(self) -> float:
        return self.failed_batches / self.batches if self.batches else 0.0

    def summary(self) -> str:
        return (
            f"打标 {self.tagged}/{self.total}"
            f" · 词法降级 {self.fallback}"
            f" · 丢弃 {self.dropped}"
        )


# ----------------------------------------------------------------------
# 校验
# ----------------------------------------------------------------------


def _sanitize_tag(value: object, text: str, report: TagReport) -> str | None:
    """一个标签词要么是正文子串且不含禁用字符，要么不要。

    含禁用字符的**直接拒绝而不是清洗**：把「情感：x」里的冒号删掉会得到
    「情感x」，它既不合法也不像话，而且掩盖了模型正在试图注入结构的事实。
    """
    if not isinstance(value, str):
        report.dropped_tags["非字符串"] += 1
        return None
    tag = value.strip()
    if not tag or len(tag) > 20:
        report.dropped_tags["长度"] += 1
        return None
    if any(ch in _FORBIDDEN for ch in tag):
        report.dropped_tags["禁用字符"] += 1
        return None
    # 词法侧天然满足子串，LLM 侧不满足就是它在顺口编词——
    # 编出来的词会让这段文字被它根本没回答的查询召回。
    if tag not in text:
        report.dropped_tags["非原文子串"] += 1
        return None
    return tag


def _tag_list(
    raw: object, text: str, report: TagReport, *, limit: int, strict_substring: bool = True
) -> list[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for value in raw:
        if not strict_substring:
            # emotion 是**闭集**值，不是原文子串：正文里说的是「心慌」，
            # 该标的是「焦虑」——这两个词本来就不会互为子串。
            tag = value.strip() if isinstance(value, str) else ""
            if tag and not any(ch in _FORBIDDEN for ch in tag) and len(tag) <= 20:
                out.append(tag)
            continue
        tag = _sanitize_tag(value, text, report)
        if tag is not None and tag not in out:
            out.append(tag)
    return out[:limit]


def _validate_item(
    raw: object, text: str, *, library: Library, report: TagReport
) -> ChunkTags | None:
    """校验一条返回项。**返回 None 只有一种情况：身份对不上。**"""
    if not isinstance(raw, dict):
        return None

    quote = raw.get("quote")
    if not isinstance(quote, str) or not quote.strip():
        report.dropped_tags["缺 quote"] += 1
        return None
    quote = quote.strip()
    if quote not in text:
        # 身份锚点对不上。剩下的字段再漂亮也不能信——它们可能是另一段的。
        report.dropped_tags["quote 对不上"] += 1
        return None

    emotion = _tag_list(raw.get("emotion"), text, report, limit=32, strict_substring=False)
    emotion = _match_vocab(emotion, report)[:MAX_EMOTION]

    type_value = raw.get("type")
    type_value = type_value.strip() if isinstance(type_value, str) else ""
    if type_value not in TYPE_VOCAB[library]:
        if type_value:
            report.dropped_tags["type 越界"] += 1
        type_value = ""

    concept = raw.get("concept")
    concept = concept.strip() if isinstance(concept, str) else ""
    if concept and any(ch in _FORBIDDEN for ch in concept):
        report.dropped_tags["禁用字符"] += 1
        concept = ""

    keywords = _tag_list(raw.get("keywords"), text, report, limit=MAX_KEYWORDS)
    imagery = _tag_list(raw.get("imagery"), text, report, limit=MAX_IMAGERY)

    if library is Library.PSYCHOLOGY:
        # **强制空。** `neo4j_index.write_chunks` 对所有库都写 MENTIONS_IMAGERY，
        # 给哲学块打上「大地」「深渊」，`chunks_by_imagery("大地")` 就会从
        # 古典文学那条图路径里把哲学块返回出来。
        if imagery:
            report.dropped_tags["psychology 误标意象"] += len(imagery)
        imagery = []

    return ChunkTags(
        quote=quote,
        concept=concept,
        keywords=keywords,
        imagery=imagery,
        emotion=emotion,
        type=type_value,
        source="llm",
    )


def _match_vocab(values: Sequence[str], report: TagReport) -> list[str]:
    """把 emotion 收敛到闭集。**越界的一律丢掉，不映射到近似词。**"""
    from app.kb.lexical_tags import EMOTION_VOCAB

    allowed = set(EMOTION_VOCAB)
    out: list[str] = []
    for value in values:
        if value in allowed:
            if value not in out:
                out.append(value)
        else:
            report.dropped_tags["emotion 越界"] += 1
    return out


# ----------------------------------------------------------------------
# 一批的来回
# ----------------------------------------------------------------------


async def _call_model(
    model: ChatModel,
    library: Library,
    items: list[tuple[int, str]],
    *,
    work: str,
    author: str,
    discipline: str,
    attempt: int,
) -> tuple[dict[int, dict], str | None]:
    """发一批、收一批。返回 (i → 原始项, finish_reason)。"""
    context = prompt.build_context(
        library, items, work=work, author=author, discipline=discipline
    )
    # **`attempt` 必须进 context。** `CachedChatModel` 的 key 是
    # (task, messages, context) 的哈希——原样重试会命中上一次那条坏缓存，
    # 原样失败，看起来像「重试逻辑没生效」。
    context["attempt"] = attempt

    result = await model.complete(
        prompt.build_messages(library, len(items)),
        task=prompt.TASK,
        context=context,
        temperature=0.1,
        max_tokens=140 * len(items) + 120,
        json_mode=True,
    )
    payload = parse_json_response(result.text)
    rows = payload.get("items")
    if not isinstance(rows, list):
        raise ProviderError("返回里没有 items 数组", retryable=True)

    out: dict[int, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        index = row.get("i")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        out[index] = row
    return out, result.finish_reason


def _is_misaligned(returned: dict[int, dict], texts: dict[int, str]) -> bool:
    """有没有「这一段抄了另一段的原文」的证据。

    这是错位唯一可观测的信号。模型把第 5 段的内容配到第 3 段上时，
    `i` 仍然是 3、字段全都合法，只有 `quote` 会露馅——它抄的是第 5 段的
    原文。命中另一条**而且不在自己这条里**，才算证据；两条正文恰好都含
    同一句话（引用、重复段落）时不作数，否则会把正常的一批判死。
    """
    for index, row in returned.items():
        quote = row.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            continue
        quote = quote.strip()
        own = texts.get(index, "")
        if quote in own:
            continue
        if any(quote in other for i, other in texts.items() if i != index):
            return True
    return False


async def _tag_batch(
    model: ChatModel,
    library: Library,
    items: list[tuple[int, str]],
    *,
    work: str,
    author: str,
    discipline: str,
    report: TagReport,
    attempt: int = 0,
) -> dict[int, ChunkTags]:
    """打一批。失败时**切半重试一次**，再失败返回空字典（调用方走词法降级）。"""
    texts = dict(items)
    try:
        returned, finish_reason = await _call_model(
            model, library, items, work=work, author=author,
            discipline=discipline, attempt=attempt,
        )
    except (ProviderError, ValueError) as exc:
        if isinstance(exc, ProviderError) and not exc.retryable:
            raise
        returned, finish_reason = {}, None

    if returned:
        # 集合校验：**不部分接受**。多一条少一条都说明这一批的对应关系
        # 已经不可信，而「大部分是对的」在检索里的表现是「偶尔查出错的东西」。
        # 截断（`finish_reason == "length"`）同理：解析成功不等于语义完整，
        # 被砍掉的那个 item 往往正好是半截的。
        suspect = (
            set(returned) != set(texts)
            or _is_misaligned(returned, texts)
            or finish_reason == "length"
        )
        if suspect:
            returned = {}

    if returned:
        out: dict[int, ChunkTags] = {}
        for index, row in returned.items():
            tags = _validate_item(row, texts[index], library=library, report=report)
            if tags is not None:
                out[index] = tags
        if out:
            return out

    # --- 重试：切半 ---
    if len(items) > 1:
        report.warnings.append(f"一批 {len(items)} 条打标失败，切半重试")
        half = len(items) // 2
        merged: dict[int, ChunkTags] = {}
        for chunk in (items[:half], items[half:]):
            merged.update(
                await _tag_batch(
                    model, library, chunk, work=work, author=author,
                    discipline=discipline, report=report, attempt=attempt + 1,
                )
            )
        return merged
    return {}


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------


def _chunks(
    pieces: Sequence[Piece],
    tags_by_index: dict[int, ChunkTags],
    *,
    library: Library,
    work: str,
    author: str,
    discipline: str,
    origin: str,
    freq: Counter[str] | None,
    report: TagReport,
    count_fallback: bool = True,
) -> list[KBChunk]:
    out: list[KBChunk] = []
    seen: dict[str, int] = {}
    for piece in pieces:
        tags = tags_by_index.get(piece.index)
        if tags is None:
            tags = lexical_tags(
                piece.body,
                library=library,
                work=work,
                discipline_hint=discipline,
                freq=freq,
            )
            # 用户自己关掉 AI 时不算降级：命令行会打印 `fallback=N`，
            # 而 `--no-tag` 打出「降级 300 段」是在报告一次并不存在的故障。
            if count_fallback:
                report.fallback += 1

        if tags.is_empty_for(library):
            report.dropped += 1
            continue

        # chunk_id 由 `body` 派生而不是 `text`：`text` 带着硬切重叠，
        # 而重叠长度是个可调参数——用 `text` 派生的话，改一次 OVERLAP_CHARS
        # 就会让整本书换一批新 id。
        chunk_id = make_chunk_id(library, work, piece.body)
        if chunk_id in seen:
            # 同一份文件里出现两次完全相同的正文（重复页、刻意重复的格言）。
            # 合并成一条，但**不静默**——`deduped` 计数让「少了一块」可见。
            report.deduped += 1
            continue
        seen[chunk_id] = piece.index

        out.append(
            KBChunk(
                chunk_id=chunk_id,
                library=library,
                text=piece.text,
                type=tags.type or _TYPE_DEFAULT[library],
                imagery=list(tags.imagery),
                emotion=list(tags.emotion),
                keywords=list(tags.keywords),
                author=author or None,
                work=work or None,
                discipline=discipline or None,
                concept=tags.concept or None,
                origin=origin,
            )
        )
    return out


async def tag_pieces(
    pieces: Sequence[Piece],
    *,
    library: Library,
    model: ChatModel | None = None,
    work: str = "",
    author: str = "",
    discipline: str = "",
    origin: str = "",
    freq: Counter[str] | None = None,
    use_llm: bool = True,
    allow_partial: bool = False,
    batch_size: int = BATCH_SIZE,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> tuple[list[KBChunk], TagReport]:
    """把切好的块变成可入库的 `KBChunk`。

    **先打标、后嵌入**，顺序不能反。`text_for_embedding()` 与 `content_hash()`
    都基于带标签的嵌入文本，先嵌入必然触发全部重嵌；而且打标是唯一会失败
    的阶段，先做它意味着硬失败时什么都还没写。

    `allow_partial=False`（默认）时，失败批次过半直接抛 `TaggingError`。
    这是刻意的：**打标大面积失败说明配置坏了**（没额度、模型名错、prompt
    被改坏），此时入库的是一整本只有词法标签的内容，检索质量会静默劣化。

    `use_llm=False` 时不需要 `model`，整本走词法标签——那是用户的显式选择
    （「这次不用 AI」），不是降级，所以不产生任何 fallback 计数。
    """
    report = TagReport(total=len(pieces))
    tags_by_index: dict[int, ChunkTags] = {}

    if use_llm and pieces and model is None:
        raise ValueError("use_llm=True 时必须提供 model")

    if use_llm and pieces and model is not None:
        items = [(p.index, p.body) for p in pieces]
        size = max(1, batch_size)
        batches = [items[i : i + size] for i in range(0, len(items), size)]
        report.batches = len(batches)
        consecutive_fatal = 0

        for position, batch in enumerate(batches):
            if is_cancelled is not None and is_cancelled():
                raise TaggingCancelled("用户取消了这次导入")

            try:
                got = await _tag_batch(
                    model, library, batch, work=work, author=author,
                    discipline=discipline, report=report,
                )
                consecutive_fatal = 0
            except ProviderError as exc:
                if exc.retryable:
                    raise
                consecutive_fatal += 1
                got = {}
                if consecutive_fatal >= _NON_RETRYABLE_LIMIT:
                    raise TaggingError(
                        f"连续 {consecutive_fatal} 次不可重试的模型错误（{exc}）。"
                        "这是配置问题，不会自愈——请检查 API key / 模型名 / 额度。"
                    ) from exc
            tags_by_index.update(got)

            if len(got) < len(batch):
                report.failed_batches += 1
            if on_progress is not None:
                on_progress(min(position + 1, len(batches)), len(batches))

        report.tagged = len(tags_by_index)

        if report.failure_ratio > 0.5 and not allow_partial:
            raise TaggingError(
                f"{report.failed_batches}/{report.batches} 批打标失败"
                f"（{report.failure_ratio:.0%}）。已中止，**未写入任何内容**。"
                "如需强行入库（只有词法标签），请勾选「打标失败也导入」。"
            )

    chunks = _chunks(
        pieces,
        tags_by_index,
        library=library,
        work=work,
        author=author,
        discipline=discipline,
        origin=origin,
        freq=freq,
        report=report,
        count_fallback=use_llm,
    )

    if report.fallback and report.tagged:
        report.warnings.append(
            f"{report.fallback} 段走了词法降级标签——正文仍然入库，"
            "但标签是规则抽取的，检索质量弱于模型标注"
        )
    if report.dropped:
        report.warnings.append(f"{report.dropped} 段因标注全空被丢弃（文学/诗词要求有意象或情感）")
    if report.deduped:
        report.warnings.append(f"{report.deduped} 段与文件里的另一段正文完全相同，已合并")
    if report.dropped_tags:
        top = "、".join(f"{k}×{v}" for k, v in report.dropped_tags.most_common(4))
        report.warnings.append(f"被丢弃的标签值：{top}")
    log.info(
        "kb_tagged",
        total=report.total,
        tagged=report.tagged,
        fallback=report.fallback,
        dropped=report.dropped,
        failed_batches=report.failed_batches,
    )
    return chunks, report


__all__ = [
    "BATCH_SIZE",
    "TYPE_VOCAB",
    "TagReport",
    "TaggingCancelled",
    "TaggingError",
    "tag_pieces",
]
