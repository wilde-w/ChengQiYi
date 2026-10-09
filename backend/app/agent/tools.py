"""故事工坊的工具集：模型能查什么。

六个工具，全部只读。分工是「**先便宜后昂贵**」——`read_input` 与
`kb_sources` 不花钱，`kb_search` / `kb_graph` 要一次 embedding 或一次
图查询，`novel_lookup` 每次都要起一个 MCP 子进程（约 2 秒）。schema 里
把代价写进 description，模型才可能掂量着用。

三条贯穿全篇的约定：

1. **返回文本，不返回 JSON。** 工具结果是喂给模型的上下文，不是给程序
   解析的 API。JSON 要多花一半 token 在括号和引号上，还会让模型把注意力
   放在结构上而不是内容上——这正是隔壁「古典文学 MCP」的取舍，照抄。
2. **`dispatch` 永不抛异常。** 超时、参数坏、工具不存在，一律转成一段
   中文说明回给模型。抛出去的话这一轮就结束了，而模型本来还有机会换个
   问法；何况工具失败是**常态**（MCP 没配就是失败），它必须是一条
   能被对话吸收的信息。
3. **失败信息带可用清单。** `kb_source_text` 失败时列出所有能读的来源，
   工具名写错时列出所有工具名。这段文字就是模型自纠所需的全部信息——
   只说「失败」等于让它下一轮再撞一次。

长度：单条结果正文 ≤ `MAX_RESULT_CHARS`，截断处显式写「（已截断）」。
一轮最多 4 次调用（`loop.MAX_TOOL_CALLS_PER_ROUND`），所以一轮的工具
上下文天然 ≤2400 字，不需要再设一个跨调用的账本。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from app.constants import Library
from app.logging_conf import get_logger
from app.providers.base import ToolCall

log = get_logger(__name__)

#: 单条工具结果的正文上限。给模型看的东西，超过这个数就是噪声。
MAX_RESULT_CHARS = 600

#: 每条摘录的上限。600 字里塞 5 条，每条 100 字左右正好。
EXCERPT_CHARS = 160

#: 模型一次能读到的原文片段上限（`read_input` 不传 limit 时的默认值）。
READ_DEFAULT = 2000

#: 卡片上那行摘要的长度上限。
SUMMARY_CHARS = 40

_TIMEOUT_DEFAULT = 20.0


@dataclass(slots=True)
class ToolSpec:
    """一个工具的全部定义。`handler` 收已解析的参数，返回要回给模型的文本。"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[str]]
    #: 串行执行。起子进程的工具必须为 True——并发就是同时起好几个进程。
    serial: bool = False
    #: 额度名（见 `loop._QUOTA_LIMITS`）。同类工具共享一个每轮上限。
    quota: str | None = None
    timeout: float = _TIMEOUT_DEFAULT

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @property
    def required(self) -> list[str]:
        return [str(x) for x in (self.parameters.get("required") or [])]


class ToolArgumentError(ValueError):
    """handler 自己发现的参数问题（enum 取值不对、必填为空……）。

    单独一个异常类型，是为了让 `dispatch` 把它记成 `bad_arguments` 而不是
    「执行失败」——只有前者会累计到「连错两次就撤掉这个工具」。返回一段
    说明文字是不够的：那看起来像一次**成功**的调用。
    """


class ToolFailure(RuntimeError):
    """handler 自己判定这次调用失败了（背后的服务报错、超时、没有这个文件）。

    消息原样回给模型，卡片是红的。同样不能靠「返回一段说明文字」表达——
    那会让失败在面板上显示成绿色，而用户是照着面板判断这个 agent 靠不靠谱的。
    """


@dataclass(slots=True)
class ToolOutcome:
    """一次调用的结果。`text` 进对话，其余进事件（工具卡片）。"""

    text: str
    ok: bool = True
    summary: str = ""
    preview: str = ""
    elapsed_ms: int = 0
    #: bad_arguments / timeout / failed / unknown / refused。loop 按它决定
    #: 要不要把工具撤掉（连错两次撤）。
    reason: str = ""


def parse_arguments(call: ToolCall) -> tuple[dict[str, Any], str | None]:
    """解析调用参数。返回（参数, 错误说明），错误说明非空时参数必为空字典。

    `arguments` 保持原始 JSON 文本到这一刻才解析，是为了在解析失败时能把
    **模型写的那段原文**回给它看——只说「JSON 不合法」，它下一轮多半会
    再写一遍同样的东西。
    """
    raw = (call.arguments or "").strip() or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"参数不是合法 JSON：{exc}。你给出的是：{_clip(raw, 160)}"
    if not isinstance(data, dict):
        return {}, f"参数必须是一个 JSON 对象，收到的是 {type(data).__name__}。"
    return data, None


class ToolRegistry:
    """本轮可用的工具。`retire()` 是它唯一的可变状态。"""

    def __init__(self, specs: Sequence[ToolSpec]) -> None:
        self._specs: dict[str, ToolSpec] = {s.name: s for s in specs}
        self._order: list[str] = [s.name for s in specs]

    # -- 只读视图 --------------------------------------------------------
    def names(self) -> list[str]:
        """按定义顺序。**不收排序**：顺序即「先便宜后昂贵」的提示。"""
        return [n for n in self._order if n in self._specs]

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [self._specs[n].schema() for n in self._order if n in self._specs]

    def hint(self) -> str:
        """「当前可用的工具：…」。附在失败信息后面，让模型有下一步可走。"""
        names = self.names()
        return f"当前可用的工具：{'、'.join(names)}。" if names else "当前没有可用的工具。"

    def retire(self, name: str) -> None:
        """从本轮撤掉一个工具。下一轮的 `schemas()` 里就没有它了。"""
        self._specs.pop(name, None)

    # -- 执行 ------------------------------------------------------------
    async def dispatch(self, call: ToolCall) -> ToolOutcome:
        """执行一次调用。**永不抛异常**——见模块 docstring 第 2 条。"""
        started = time.perf_counter()
        spec = self.get(call.name)
        if spec is None:
            text = f"没有名为 `{call.name}` 的工具。{self.hint()}"
            return _outcome(text, ok=False, summary="未知工具", reason="unknown", started=started)

        args, err = parse_arguments(call)
        if err is None:
            missing = [k for k in spec.required if k not in args or args[k] in ("", None)]
            if missing:
                err = f"缺少必填参数：{'、'.join(missing)}。"
        if err is not None:
            text = (
                f"`{call.name}` 的参数不合法：{err}\n"
                f"它需要的参数是：{_params_brief(spec)}"
            )
            return _outcome(
                text, ok=False, summary="参数不合法", reason="bad_arguments", started=started
            )

        try:
            text = await asyncio.wait_for(spec.handler(args), timeout=spec.timeout)
        except TimeoutError:
            text = (
                f"`{call.name}` 超过 {spec.timeout:g} 秒没有返回，已放弃这一次调用。"
                "可以换一个工具，或用现有材料继续。"
            )
            return _outcome(text, ok=False, summary="超时", reason="timeout", started=started)
        except ToolArgumentError as exc:
            text = f"`{call.name}` 的参数不合法：{exc}\n它需要的参数是：{_params_brief(spec)}"
            return _outcome(
                text, ok=False, summary="参数不合法", reason="bad_arguments", started=started
            )
        except ToolFailure as exc:
            return _outcome(
                str(exc), ok=False, summary="执行失败", reason="failed", started=started
            )
        except Exception as exc:  # 宽到 base class 是**故意的**，理由见下
            # 这里是「永不抛」的最后一道闸。工具背后是数据库与子进程，
            # 任何一处出问题都不该让整轮对话崩掉。
            log.warning("agent_tool_failed", tool=call.name, error=f"{type(exc).__name__}: {exc}")
            text = (
                f"`{call.name}` 执行失败：{type(exc).__name__}: {_clip(str(exc), 200)}。"
                f"{self.hint()}"
            )
            return _outcome(text, ok=False, summary="执行失败", reason="failed", started=started)

        return _outcome(text, ok=True, started=started)


def _outcome(text: str, *, ok: bool, started: float, summary: str = "", reason: str = "") -> ToolOutcome:
    clipped = _clip(text, MAX_RESULT_CHARS)
    return ToolOutcome(
        text=clipped,
        ok=ok,
        summary=summary or _auto_summary(clipped),
        preview=_clip(clipped, 400),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        reason=reason,
    )


def _auto_summary(text: str) -> str:
    """卡片的默认摘要 = 结果的第一行。各 handler 都把它写成一句人话。"""
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return _clip(first.rstrip("：:"), SUMMARY_CHARS)


def _params_brief(spec: ToolSpec) -> str:
    props = (spec.parameters.get("properties") or {}).keys()
    return "、".join(str(p) for p in props) or "（无参数）"


def _clip(text: Any, limit: int) -> str:
    """截断。**保留换行**——工具结果是 markdown，压成一行就没法读了。"""
    body = str(text)
    return body if len(body) <= limit else body[:limit] + "…（已截断）"


def _one_line(text: Any, limit: int = EXCERPT_CHARS) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _library_of(value: Any) -> Library | None:
    """把模型给的库名换成枚举。写错的库名返回 None（= 不限库），不报错。"""
    if not value:
        return None
    try:
        return Library(str(value).strip().lower())
    except ValueError:
        return None


# ======================================================================
# 各工具的 handler。签名一律 `(args) -> Awaitable[str]`。
# ======================================================================


def build_registry(*, input_text: str, allow_novel: bool) -> ToolRegistry:
    """按这次会话的材料组装工具集。

    `allow_novel=False` 时 `novel_lookup` **根本不进 schemas()**——不是
    「看得见但会失败」：那会让模型把仅有的几轮烧在一个必然失败的调用上，
    而用户以为自己关掉了它。
    """
    specs: list[ToolSpec] = [
        ToolSpec(
            name="read_input",
            description=(
                "读用户提供的评论原文。开头一段已经在系统提示里给过你了，"
                "正文很长时可用来读后面的部分。零成本，随时可调。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "offset": {"type": "integer", "description": "从第几个字开始读，默认 0"},
                    "limit": {"type": "integer", "description": "读多少字，默认 2000"},
                },
            },
            handler=_reader(input_text, READ_DEFAULT),
        ),
        ToolSpec(
            name="kb_search",
            description=(
                "在知识库（心理学 / 古典文学 / 诗词）里做语义检索，找回和某个说法"
                "相近的段落。写故事前用它找到「别人怎么表达同一种情绪」。"
                "参数 query 用一句自然语言，不要用关键词堆砌。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要检索的自然语言说法"},
                    "library": {
                        "type": "string",
                        "enum": [str(x) for x in Library],
                        "description": "限定某一个库；不填表示三个库一起查",
                    },
                    "limit": {"type": "integer", "description": "返回几条，默认 5"},
                },
                "required": ["query"],
            },
            handler=_kb_search,
        ),
        ToolSpec(
            name="kb_graph",
            description=(
                "按「情绪」或「意象」查知识图谱，拿到与它关联的古诗文段落。"
                "比 kb_search 更精准：它走的是人工/模型标注过的情绪—意象—作品关系。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["emotion", "imagery"],
                        "description": "emotion：按情绪查（如「哀伤」）；imagery：按意象查（如「旧毛衣」「雪」）",
                    },
                    "key": {"type": "string", "description": "情绪名或意象名"},
                    "limit": {"type": "integer", "description": "返回几条，默认 6"},
                },
                "required": ["mode", "key"],
            },
            handler=_kb_graph,
        ),
        ToolSpec(
            name="kb_sources",
            description=(
                "列出知识库里已导入的来源（文件名、所属库、段落数）。"
                "零成本。想知道「这个知识库里到底有什么」时先调它。"
            ),
            parameters={"type": "object", "properties": {}},
            handler=_kb_sources,
        ),
        ToolSpec(
            name="kb_source_text",
            description=(
                "读某个导入来源的原文全文（可分段读）。文件名从 kb_sources 的结果里取。"
                "需要逐字引用某一段原文时用它。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source_file": {"type": "string", "description": "来源文件名，见 kb_sources"},
                    "offset": {"type": "integer", "description": "从第几个字开始读，默认 0"},
                    "limit": {"type": "integer", "description": "读多少字，默认 2000"},
                },
                "required": ["source_file"],
            },
            handler=_kb_source_text,
        ),
    ]

    if allow_novel:
        specs.append(_novel_spec())

    return ToolRegistry(specs)


def _reader(text: str, default_limit: int) -> Callable[[dict[str, Any]], Awaitable[str]]:
    """把原文闭包进 handler。原文不进库、不进日志，只在这一次会话里存在。"""

    async def handler(args: dict[str, Any]) -> str:
        total = len(text)
        offset = _as_int(args.get("offset"), 0)
        limit = max(1, _as_int(args.get("limit"), default_limit))
        if offset >= total:
            return f"原文共 {total} 字，起点 {offset} 已经超出末尾。"
        body = text[offset : offset + limit]
        end = offset + len(body)
        tail = "" if end >= total else f"（后面还有 {total - end} 字，可以继续读）"
        return f"原文第 {offset}–{end} 字，共 {total} 字{tail}：\n{body}"

    return handler


async def _kb_search(args: dict[str, Any]) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ToolArgumentError("query 不能为空，它是要检索的那句话。")
    library = _library_of(args.get("library"))
    limit = max(1, min(10, _as_int(args.get("limit"), 5)))

    from app.kb.qdrant_index import search
    from app.providers.factory import get_embedding_model

    vector = (await get_embedding_model().embed([query]))[0]
    # **不设 score_threshold**：本机实测向量分只有 0.06–0.29（见 n5_retrieval），
    # 按常识设个 0.5 会把全部命中切光，而模型只会看到「没有找到」。
    points = await search(vector, library=library, limit=limit)
    if not points:
        where = f"「{library.label}」" if library else "知识库"
        return f"在{where}里没有检索到与「{query}」相近的段落。"

    where = f"「{library.label}」" if library else "知识库"
    lines = [f"在{where}里找到 {len(points)} 条与「{query}」相近的段落："]
    for i, point in enumerate(points, 1):
        payload = dict(getattr(point, "payload", None) or {})
        lines.append(
            f"{i}. {_cite(payload)}（相似度 {float(getattr(point, 'score', 0.0)):.2f}）\n"
            f"   {_one_line(payload.get('text'))}"
        )
    return "\n".join(lines)


async def _kb_graph(args: dict[str, Any]) -> str:
    mode = str(args.get("mode") or "").strip()
    key = str(args.get("key") or "").strip()
    if mode not in ("emotion", "imagery"):
        raise ToolArgumentError("mode 只能是 emotion 或 imagery。")
    if not key:
        raise ToolArgumentError("key 不能为空：按情绪查就写情绪名，按意象查就写意象名。")

    from app.kb.neo4j_index import chunks_by_emotion, chunks_by_imagery
    from app.kb.qdrant_index import fetch_by_ids

    limit = max(1, min(12, _as_int(args.get("limit"), 6)))
    rows = (
        await chunks_by_emotion(key, limit=limit)
        if mode == "emotion"
        else await chunks_by_imagery(key, limit=limit)
    )
    label = "情绪" if mode == "emotion" else "意象"
    if not rows:
        return (
            f"图谱里没有和{label}「{key}」关联的段落。"
            "换一个更常见的词试试（情绪用「哀伤」「孤独」这类基本情绪，意象用名词）。"
        )

    # 图库只存 60 字摘要，正文必须回 Qdrant 取——少了这一跳，卡片上永远
    # 只有半句话（见 `qdrant_index.fetch_by_ids` 的 docstring）。
    ids = [str(r.get("chunk_id") or "") for r in rows if r.get("chunk_id")]
    records = await fetch_by_ids(ids)
    full = {
        str((getattr(rec, "payload", None) or {}).get("chunk_id") or ""): dict(
            getattr(rec, "payload", None) or {}
        )
        for rec in records
    }

    lines = [f"与{label}「{key}」关联的 {len(rows)} 条段落："]
    for i, row in enumerate(rows, 1):
        payload = full.get(str(row.get("chunk_id") or ""), {})
        lines.append(f"{i}. {_cite(payload or row)}\n   {_one_line(payload.get('text') or row.get('snippet'))}")
    return "\n".join(lines)


async def _kb_sources(_args: dict[str, Any]) -> str:
    from app.services.kb_import_service import list_sources

    sources = await list_sources()
    if not sources:
        return "知识库里还没有导入任何来源。"
    lines = [f"知识库里有 {len(sources)} 个来源："]
    for s in sources:
        lines.append(f"- {s.source_file}（{_lib_label(s.library)}）· {s.chunks} 段")
    return "\n".join(lines)


async def _kb_source_text(args: dict[str, Any]) -> str:
    source_file = str(args.get("source_file") or "").strip()
    if not source_file:
        raise ToolArgumentError("source_file 不能为空，文件名从 kb_sources 的结果里取。")
    offset = _as_int(args.get("offset"), 0)
    limit = max(1, _as_int(args.get("limit"), READ_DEFAULT))

    from app.services.kb_import_service import ImportNotFound, source_text

    try:
        got = await source_text(source_file, offset=offset, limit=limit)
    except ImportNotFound as exc:
        # 失败也要给出路：把能读的文件名列出来，模型下一次就能写对。
        raise ToolFailure(
            f"没有名为 `{source_file}` 的来源。\n{await _kb_sources({})}"
        ) from exc

    head = f"《{got.title or got.source_file}》第 {got.offset}–{got.offset + len(got.text)} 字，全文 {got.char_count} 字"
    if got.truncated:
        head += "（后面还有，可继续读）"
    return f"{head}：\n{got.text}"


def _novel_spec() -> ToolSpec:
    """古典文学 MCP：**六合一**，不拆成六个工具。

    每个工具的 schema 都要进**每一轮**的 prompt。拆成六个，光工具描述就
    要占掉几百个 token，而模型一次会话里通常只用得上其中两三个。
    `action` 参数换来的是同一个能力，代价只是模型要多写一个字段。

    代价也如实写进 description：每次约 2 秒（每次都新起一个子进程，
    见 `novel/mcp_client.py`）。不写，它就会一页一页地翻。
    """
    from app.config import get_settings

    return ToolSpec(
        name="novel_lookup",
        description=(
            "查古典文学作品的逐字原文（《红楼梦》等）。六个动作合一，用 action 选：\n"
            "books（找书，拿到 book_id）→ characters（找人，拿到 character_id）→ "
            "dialogues（该人物的对话原文）/ chapters（回目，拿到 chapter_idx）→ "
            "chapter_text（整章原文）/ passage_text（给 global_idx 取该段前后文）。\n"
            "返回值是渲染好的 markdown 原文，可直接引用。"
            "**每次调用约 2 秒**（要起一个子进程），想清楚再调，不要连着翻页。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(_NOVEL_REQUIRED),
                    "description": "要做的动作",
                },
                "query": {"type": "string", "description": "书名 / 回目 / 对话的关键词过滤"},
                "name": {"type": "string", "description": "人物名或别名（characters 用）"},
                "book_id": {"type": "integer", "description": "来自 books 的结果"},
                "character_id": {"type": "integer", "description": "来自 characters 的结果"},
                "chapter_idx": {"type": "integer", "description": "库内章序号，来自 chapters 的结果"},
                "global_idx": {"type": "integer", "description": "全书段序，来自对话或整章原文"},
                "chapter_from": {"type": "integer", "description": "起始章号，与 chapter_to 配对使用"},
                "chapter_to": {"type": "integer", "description": "结束章号，与 chapter_from 配对使用"},
                "cursor": {"type": "integer", "description": "翻页游标：传上一页返回里的段号"},
                "offset": {"type": "integer", "description": "跳过前 N 条，用于翻页"},
                "limit": {"type": "integer", "description": "返回条数上限，1-100"},
                "before": {"type": "integer", "description": "passage_text 向前多取几段，0-20"},
                "after": {"type": "integer", "description": "passage_text 向后多取几段，0-20"},
            },
            "required": ["action"],
        },
        handler=_novel_lookup,
        # 每次调用都新起一个 stdio 子进程——并发就是同时起好几个。
        serial=True,
        quota="mcp",
        # 比 MCP 自己的超时多 2 秒：让子进程那层先报错，拿到的是它的
        # 中文说明而不是我们这边一句干巴巴的 TimeoutError。
        timeout=float(get_settings().NOVEL_MCP_TIMEOUT) + 2.0,
    )


#: 每个动作的必填参数，以及**缺了它去哪拿**——模型手里只有 id 是数字，
#: 不告诉它上一步是哪个工具，它就只能瞎猜。
_NOVEL_REQUIRED: dict[str, tuple[str, ...]] = {
    "books": (),
    "characters": ("name",),
    "dialogues": ("character_id",),
    "chapters": ("book_id",),
    "chapter_text": ("book_id", "chapter_idx"),
    "passage_text": ("book_id", "global_idx"),
}

_NOVEL_WHERE: dict[str, str] = {
    "name": "用 action=characters 时要给人物名或别名，如「宝玉」「宝二爷」。",
    "character_id": "character_id 要先调 action=characters 拿到。",
    "book_id": "book_id 要先调 action=books 拿到。",
    "chapter_idx": "chapter_idx 要先调 action=chapters 拿到（它是库内序号，不是回目号）。",
    "global_idx": "global_idx 是全书段序，从上一次对话或整章原文的返回里取。",
}

_NOVEL_DEFAULT_LIMIT: dict[str, int] = {
    "books": 20,
    "characters": 20,
    "dialogues": 20,
    "chapters": 100,
    "chapter_text": 80,
}


async def _novel_lookup(args: dict[str, Any]) -> str:
    action = str(args.get("action") or "").strip()
    if action not in _NOVEL_REQUIRED:
        raise ToolArgumentError(f"action 只能是：{'、'.join(_NOVEL_REQUIRED)}。")

    missing = [k for k in _NOVEL_REQUIRED[action] if args.get(k) in (None, "")]
    if missing:
        raise ToolArgumentError(
            f"action={action} 还缺少参数：{'、'.join(missing)}。{_NOVEL_WHERE[missing[0]]}"
        )

    from app.novel import service as novel_service
    from app.novel.mcp_client import NovelMcpError

    limit = _clamp(args.get("limit"), _NOVEL_DEFAULT_LIMIT.get(action, 20), 1, 100)
    try:
        if action == "books":
            got = await novel_service.books(
                query=_opt_str(args.get("query")), limit=limit, offset=_as_int(args.get("offset"), 0)
            )
        elif action == "characters":
            got = await novel_service.characters(
                name=str(args["name"]), book_id=_opt_int(args.get("book_id")), limit=limit
            )
        elif action == "dialogues":
            pair = _opt_int(args.get("chapter_from")) is not None and _opt_int(args.get("chapter_to")) is not None
            # 起止章号**成对**才有意义：只给一个，对方会静默忽略（它的契约如此）。
            got = await novel_service.dialogues(
                character_id=int(args["character_id"]),
                book_id=_opt_int(args.get("book_id")),
                chapter_from=_opt_int(args.get("chapter_from")) if pair else None,
                chapter_to=_opt_int(args.get("chapter_to")) if pair else None,
                query=_opt_str(args.get("query")),
                limit=limit,
                cursor=_opt_int(args.get("cursor")),
            )
        elif action == "chapters":
            got = await novel_service.chapters(
                book_id=int(args["book_id"]),
                query=_opt_str(args.get("query")),
                limit=limit,
                offset=_as_int(args.get("offset"), 0),
            )
        elif action == "chapter_text":
            got = await novel_service.chapter_text(
                book_id=int(args["book_id"]),
                chapter_idx=int(args["chapter_idx"]),
                cursor=_opt_int(args.get("cursor")),
                limit=limit,
            )
        else:
            got = await novel_service.passage_text(
                book_id=int(args["book_id"]),
                global_idx=int(args["global_idx"]),
                before=_clamp(args.get("before"), 2, 0, 20),
                after=_clamp(args.get("after"), 2, 0, 20),
            )
    except NovelMcpError as exc:
        raise ToolFailure(_novel_error(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise ToolArgumentError(f"参数类型不对：{exc}。id 与序号都要给整数。") from exc

    # 对方渲染好的 markdown **原样透出**：在这里解析回结构再重排，等于把
    # 对方的展示层当成 API 用（见 `app/novel/mcp_client.py` 的模块注释）。
    return got.markdown


def _novel_error(exc: Any) -> str:
    """`NovelMcpError` → 模型看得懂的一段话。三档 `kind` 分开说。"""
    kind = str(getattr(exc, "kind", "failed"))
    tail = {
        "unavailable": "这是配置问题，不是参数问题：换用知识库工具，本轮不要再调它。",
        "timeout": "可以重试一次；如果再超时就换知识库工具。",
        "failed": "换一个参数再试一次，或改用知识库工具。",
    }.get(kind, "换用知识库工具继续。")
    return f"novel_lookup 调用失败（{kind}）：{_clip(str(exc), 200)}{tail}"


def _opt_str(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _opt_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: Any, default: int, low: int, high: int) -> int:
    got = _as_int(value, default)
    return max(low, min(high, got))


def _cite(payload: dict[str, Any]) -> str:
    """一行出处。字段名按 `KBChunk.payload()`：作品名在 `work`，来源在 `origin`。"""
    work = str(payload.get("work") or "").strip()
    author = str(payload.get("author") or "").strip()
    origin = str(payload.get("origin") or "").strip()
    label = _lib_label(payload.get("library"))
    bits = [f"《{work}》" if work else (origin or "未命名来源")]
    if author:
        bits.append(author)
    bits.append(label)
    chunk_id = str(payload.get("chunk_id") or "")
    if chunk_id:
        bits.append(f"#{chunk_id[:8]}")
    return "·".join(bits)


def _lib_label(value: Any) -> str:
    lib = _library_of(value)
    return lib.label if lib else str(value or "未知库")


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "EXCERPT_CHARS",
    "MAX_RESULT_CHARS",
    "READ_DEFAULT",
    "ToolArgumentError",
    "ToolFailure",
    "ToolOutcome",
    "ToolRegistry",
    "ToolSpec",
    "build_registry",
    "parse_arguments",
]
