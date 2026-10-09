"""流水线节点调模型的共用件。

存在理由只有一个：**`attempt` 必须进 context**。

`CachedChatModel` 的缓存键是 `(task, messages, context)` 的哈希
（见 `providers/cache.py`）。原样重试等于拿同一把钥匙再开一次同一个柜子——
拿回来的还是上次那份坏输出（截断的 JSON、空内容），重试逻辑看起来像没生效，
而日志里两次调用的参数一模一样，排查时最容易往「模型不稳定」上归因。

所以这里的 `call_json` 每次尝试都改写 `context["attempt"]`，强制换一把钥匙。

`kb/tagging.py` 里有一份等价的实现。**故意没有抽成一份共用**：那边的重试
粒度是「一批 8 条，失败切半」，这边是「一次调用失败就整体降级」，两者的
退避与降级策略不同，硬合并会把两套参数塞进同一个函数签名里。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.providers.base import ChatMessage, ChatModel, ProviderError
from app.providers.openai_compat import parse_json_response

#: 与 `kb/tagging.py:_FORBIDDEN` 同一份。标签里的这些字符会破坏
#: 「关键词：a、b｜正文…」这类拼接模板，一个标签就能伪造出结构。
FORBIDDEN = frozenset("｜《》〈〉：、\n\r\t")

#: 单个标签的长度上限。超过的基本是模型把一句话塞进来了。
MAX_TAG_CHARS = 20


async def call_json(
    model: ChatModel,
    *,
    task: str,
    messages: Sequence[ChatMessage],
    context: dict[str, Any],
    temperature: float = 0.2,
    max_tokens: int = 1200,
    attempts: int = 2,
) -> dict[str, Any]:
    """调一次模型并解析 JSON 对象。失败按 `attempts` 重试。

    重试的判据是**任何**异常，不只是 ProviderError：解析失败抛的也是
    ProviderError，但 `json.JSONDecodeError` 之类的东西一样不该让整个节点倒下
    ——节点层的降级由调用方决定，这里只负责「试满次数再放弃」。

    最后一次失败会把异常抛出去，调用方据此走降级链。
    """
    last: Exception | None = None
    for attempt in range(attempts):
        ctx = dict(context)
        # 见模块 docstring：不改 context 就等于不重试。
        ctx["attempt"] = attempt
        try:
            result = await model.complete(
                messages,
                task=task,
                context=ctx,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
            )
            return parse_json_response(result.text)
        except Exception as exc:  # noqa: BLE001 — 交给下方统一收口
            last = exc

    raise ProviderError(
        f"{task} 调用失败（已试 {attempts} 次）：{type(last).__name__}: {last}",
        retryable=False,
    )


def clean_tags(
    value: Any,
    *,
    allowed: frozenset[str] | set[str] | None = None,
    limit: int = 3,
) -> list[str]:
    """把模型给的标签列表收成干净的短列表。

    `allowed` 非空时按**闭集**过滤（emotion 用）。这一步不能省：模型对
    「从表里选」的遵守率不是 100%，而自创词在图谱里会长成孤立节点——
    `chunks_by_emotion("焦虑")` 只命中三分之一，且没人会想到去查这里。

    超出 `limit` 的部分直接截断而不是报错：多给一个标签不是错误，
    只是不如少给的那几个重要。
    """
    if value is None:
        return []
    if isinstance(value, str):
        raw: list[Any] = [value]
    elif isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        return []

    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        tag = item.strip()
        if not tag or len(tag) > MAX_TAG_CHARS:
            continue
        if any(ch in FORBIDDEN for ch in tag):
            continue
        if allowed is not None and tag not in allowed:
            continue
        if tag in out:
            continue
        out.append(tag)
        if len(out) >= limit:
            break
    return out


def clean_text(value: Any, *, limit: int = 400) -> str:
    """把模型给的一段话收干净。空/非字符串一律返回空串，不返回 None。

    调用方拿空串就能判断「这段没产出」，不需要再区分 None 与 ""。
    """
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if len(text) > limit:
        text = text[:limit].rstrip()
    return text


def evidence_brief(items: Any, *, limit: int = 400) -> list[dict[str, Any]]:
    """把证据列表收成提示词里读的那种形态（n6 与 n7 共用）。

    **`id` 就是 `chunk_id`**，不是数据库主键。主键是 UUID、要 flush 之后才
    存在，而节点写正文时手里根本没有它；`chunk_id` 跨库唯一、跨重跑稳定，
    也是 RRF 去重用的那把键。三者共用一个键，模型引用的、正文里写的、
    检索集里存的才是同一个东西——用主键就得在三个地方做映射，而其中任何
    一处忘了转，症状都是「引用全都对不上」。

    `text` 截断到 `limit`：提示词里十条各 400 字已经是一屏，再长就只是
    把真正要对照的句子挤出上下文窗口。
    """
    out: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id:
            continue
        out.append(
            {
                "id": chunk_id,
                "kind": str(item.get("kind") or ""),
                "title": item.get("title"),
                "source": item.get("source"),
                "author": item.get("author"),
                "text": clean_text(item.get("text"), limit=limit),
            }
        )
    return out


__all__ = [
    "FORBIDDEN",
    "MAX_TAG_CHARS",
    "call_json",
    "clean_tags",
    "clean_text",
    "evidence_brief",
]
