"""故事工坊的事件契约。

**协议复用，通道不复用。** 信封还是 `RunEvent`（`seq` + `type` + `data`），
SSE 帧格式、断线续传语义一字不改；但 Redis stream 用 `agent` 前缀、
落库落进 `agent_event`，与流水线各走各的通道。理由只有一条：`seq` 是
「这条流里第几条」，两条流混在一起，前端的 `?since=` 游标就没有意义了。

这里只定义**载荷的形状**（键名、含义），不碰 Redis / PG——所以本模块
零依赖，`loop.py` 与 `agent_service.py` 都读它这一份，键名不会漂。

事件分两族：

- **对话**（`agent_message` 表）：system / user / assistant / tool，
  下一轮要原样重发给模型；
- **过程**（本模块）：工具卡片、逐字增量、轮次心跳，只给 UI 看。

分表而不是一张，是因为组装历史时得跳过一堆 UI 事件——混在一起的代价是
每加一种事件就要在多处补一个 `if`。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.constants import EventType

#: Redis stream key 前缀与 SSE 端点的命名空间。流水线是 `run`。
NAMESPACE = "agent"

#: 一轮的终态。前端收到这几个之后不再追加任何增量。
TERMINAL: frozenset[EventType] = frozenset(
    {EventType.AGENT_COMPLETED, EventType.AGENT_CANCELLED, EventType.ERROR}
)

#: **一场会话**的终态。判断「这条 SSE 流能不能收工」用它，不用上面那个：
#: `agent_completed` 只说明这一稿写完了，会话还在待命，用户随时会说
#: 「再暗一点」。照轮关流的话，下一轮的事件会被送进一条已经断掉的连接，而
#: 两边都不吭声——前端以为连接一直开着（`isStreaming()` 看的是 handle 在不在，
#: 连接断了 handle 照样在），`onSend` 因此只发 POST 不重开流。症状是「说了话
#: 面板纹丝不动，刷新一下第二版早就在库里了」。
#:
#: 另外两种确实让会话进终态（与 `AgentStatus.is_terminal` 同步），它们到了就
#: 该立刻收工：再等心跳那次探测的话要多等一个周期，而且探测那条路会给一帧
#: 已经送到的终态**再补一条合成的 error 帧**，把一次正常取消显示成「出错」。
SESSION_TERMINAL: frozenset[EventType] = frozenset(
    {EventType.AGENT_CANCELLED, EventType.ERROR}
)

#: 发事件的回调。由 `agent_service` 注入（它才知道 Redis / PG 的事），
#: `loop` 只负责按契约拼载荷。
Emit = Callable[[EventType, dict[str, Any]], Awaitable[None]]


def started(turn: int) -> dict[str, Any]:
    """新一轮开始。前端在这一刻把上一版正文收进对话流（它已不是「当前稿」）。"""
    return {"turn": turn}


def thinking(*, round_no: int, tools: list[str]) -> dict[str, Any]:
    """一次模型调用开始。

    `tools` 是这一轮**真正递上去**的工具名——收尾轮是空列表，这一点对
    排查「模型为什么没查」很关键：是它不想查，还是我们没给它。
    """
    return {"round": round_no, "tools": tools}


def tool_call(
    *, call_id: str, name: str, args: dict[str, Any], raw_args: str, round_no: int
) -> dict[str, Any]:
    """模型要求调一次工具。

    `args` 是解析成功的参数（解析失败的给空字典），`raw_args` 是原文——
    参数坏掉时，卡片上要能看到模型**到底写了什么**。
    """
    return {
        "call_id": call_id,
        "name": name,
        "args": args,
        "raw_args": raw_args,
        "round": round_no,
    }


def tool_result(
    *,
    call_id: str,
    name: str,
    ok: bool,
    summary: str,
    preview: str,
    elapsed_ms: int,
    chars: int,
    reason: str = "",
) -> dict[str, Any]:
    """一次工具调用的结果。

    `summary` 是卡片上那行字（「3 条」「失败：超时」），`preview` 是展开后
    看得到的正文片段。二者都由 `tools.dispatch` 产出——**面板上显示的
    必须就是喂给模型的那段文字**，另起一套渲染就等于让用户看不见真相。
    """
    data: dict[str, Any] = {
        "call_id": call_id,
        "name": name,
        "ok": ok,
        "summary": summary,
        "preview": preview,
        "elapsed_ms": elapsed_ms,
        "chars": chars,
    }
    if reason:
        data["reason"] = reason
    return data


def delta(text: str) -> dict[str, Any]:
    """正文增量。复用流水线的 `delta` 事件类型，靠 `field` 区分落点。"""
    return {"field": "story", "text": text}


def completed(*, turn: int, rounds: int, tool_calls: int, chars: int) -> dict[str, Any]:
    """交稿。`rounds` / `tool_calls` 常驻显示在面板上——它们是这个 agent
    唯一可被核查的「自主程度」证据。"""
    return {"turn": turn, "rounds": rounds, "tool_calls": tool_calls, "chars": chars}


def cancelled(*, turn: int) -> dict[str, Any]:
    return {"turn": turn}


def error(*, message: str, code: str = "agent_failed") -> dict[str, Any]:
    return {"message": message, "code": code}


__all__ = [
    "NAMESPACE",
    "SESSION_TERMINAL",
    "TERMINAL",
    "Emit",
    "cancelled",
    "completed",
    "delta",
    "error",
    "started",
    "thinking",
    "tool_call",
    "tool_result",
]
