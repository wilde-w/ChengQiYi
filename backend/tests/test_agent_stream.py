"""故事工坊的事件流：这条连接该活多久。

这里守的是一条**只在这两处接缝上才成立**的规矩：**一轮跑完 ≠ 会话结束**。

后端按轮关流的话，用户说「再暗一点」之后那一轮的事件会被送进一条已经断掉的
连接，而两边都不吭声：前端 `onSend` 只发 POST 不重开流（它以为连接一直
开着——`isStreaming()` 看的是 handle 在不在，连接断了 handle 照样在），
控制台干干净净。症状是「面板纹丝不动，刷新一下第二版早就在库里了」。

所以这里**不测 SSE 那台帧机器**。`sse_stream.tail` 是通用的，流水线要的恰恰
是「收到终态就收工」，它没错。测的是 agent 这条线**怎么接线**：`agent.py`
递给它的那个终态集合，按轮看是对的，按流看是错的。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from app.agent import events as ev
from app.api.v1 import agent as agent_api
from app.constants import AgentStatus, EventType
from app.graph.bus import RunEvent

SESSION = "s-stream"


class _Bus:
    """给完手头这几条就挂着。

    **不能给完就结束**：真总线后面还有第二轮。`subscribe` 在这里永不返回，
    于是「这条流还开着吗」这个问题，只有 `_event_stream` 自己的选择能回答。
    """

    def __init__(self, events: list[RunEvent], *, replay: list[RunEvent] | None = None) -> None:
        self._events = events
        self._replay = replay or []

    async def replay(self, run_id: str, since: int) -> list[RunEvent]:
        """**默认一条都不给，这是故意的。**

        面板是在按下「开始写」的那一刻开流的——那时一条事件都还没有，回放
        当然是空的，这一轮的所有事件都得从订阅这条路来。让假总线在 `replay`
        里把事件全给出去，就绕开了 `tail`，而按轮关流那个 bug 正好在 `tail`
        上：这样写出来的测试**在旧的（错的）代码上照样绿**。
        """
        return [e for e in self._replay if e.seq > since]

    async def subscribe(self, run_id: str, since: int) -> AsyncIterator[RunEvent]:
        for event in self._events:
            if event.seq > since:
                yield event
        await asyncio.Event().wait()


class _Request:
    """心跳那条分支会问「客户端还在吗」——它是这条流活着的唯一证据。"""

    async def is_disconnected(self) -> bool:
        return False


def _bus_of(events: list[RunEvent]):
    async def _get_bus(namespace: str = "") -> Any:
        return _Bus(events)

    return _get_bus


def _ev(seq: int, type_: EventType, data: dict[str, Any]) -> RunEvent:
    return RunEvent(seq=seq, run_id=SESSION, type=type_, data=data)


class _Settings:
    """心跳压到下限。`_event_stream` 里是 `max(5, …)`，那 5 秒是地板，
    一条用例要干等 15 秒（生产值）就没人愿意跑了。"""

    SSE_HEARTBEAT_SECONDS = 0


def _patch(monkeypatch: pytest.MonkeyPatch, events: list[RunEvent], *, status: str) -> None:
    async def _no_events(run_id: str, cursor: int) -> list[RunEvent]:
        return []

    async def _status(session_id: str) -> str:
        return status

    monkeypatch.setattr(agent_api, "get_bus", _bus_of(events))
    monkeypatch.setattr(agent_api, "get_settings", _Settings)
    # PG 是回放的兜底源，这里没有库。
    monkeypatch.setattr(agent_api.agent_service, "load_events", _no_events)
    monkeypatch.setattr(agent_api.agent_service, "session_status", _status)


def _stream() -> AsyncIterator[str]:
    return agent_api._event_stream(
        SESSION, since=0, status=AgentStatus.RUNNING.value, request=_Request()
    )


async def test_一轮的终态事件不该关掉这条流(monkeypatch: pytest.MonkeyPatch) -> None:
    """交稿帧之后还要能继续等——用户随时会说「再暗一点」。"""
    _patch(
        monkeypatch,
        [
            _ev(1, EventType.AGENT_STARTED, ev.started(turn=1)),
            _ev(2, EventType.DELTA, ev.delta("第一版")),
            _ev(3, EventType.AGENT_COMPLETED, ev.completed(turn=1, rounds=2, tool_calls=2, chars=3)),
        ],
        status=AgentStatus.RUNNING.value,
    )

    stream = _stream()
    got: list[str] = []
    async for frame in stream:
        got.append(frame)
        if EventType.AGENT_COMPLETED.value in frame:
            break
    assert len(got) == 3, "前三帧应当原样透出"

    # **命门在这一句。** 再去要一帧：
    #   超时 → 它还在等第二轮（对）
    #   StopAsyncIteration → 它已经收工了（错，而且正是线上那个 bug）
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(anext(stream), timeout=0.5)

    await stream.aclose()


async def test_会话真的结束了流要收掉(monkeypatch: pytest.MonkeyPatch) -> None:
    """上一条的另一半：跨回合常驻不是「永不收工」。

    会话级终态由心跳那次探测收掉（`_probe` 查的一直是**会话**状态，它才是
    唯一作数的判据）。事件表里一条终态帧都没有的情形——进程被杀留下的僵尸，
    被 `reap_stale_agent_sessions` 直接改库收掉——也只有它接得住。
    """
    _patch(monkeypatch, [], status=AgentStatus.FAILED.value)

    frames = [frame async for frame in _stream()]

    assert frames, "至少应当有一条合成的终态帧"
    assert "stream_ended_without_terminal_event" in frames[-1], (
        f"最后一条应当是合成的终态帧，实际是 {frames[-1]!r}"
    )


async def test_取消之后流立刻收掉且不补假的错误帧(monkeypatch: pytest.MonkeyPatch) -> None:
    """取消走的是**事件**这条路，不是探测那条路——两条路的收尾不一样。

    这里守的是 `SESSION_TERMINAL` 到底该是哪几个。曾经图省事给 `tail` 传了
    空集合（「交给探测兜底」），那有个看不见的后坐力：心跳分支是「先抽干
    队列、再用同一个集合判断这一帧算不算终态」，空集合让它**永远判 false**，
    于是一次正常取消会在转发完 `agent_cancelled` 之后再补一条合成的 error
    帧，而前端把带 message 的 error 当真失败——用户点了「停止」，界面上
    写着「出错」。

    所以既不能传按轮的 `TERMINAL`（上一条），也不能传空集合（这一条）。
    """
    _patch(
        monkeypatch,
        [
            _ev(1, EventType.AGENT_STARTED, ev.started(turn=1)),
            _ev(2, EventType.AGENT_CANCELLED, ev.cancelled(turn=1)),
        ],
        status=AgentStatus.CANCELLED.value,
    )

    # 不设超时：**读完**这条流本身就是断言的一半。传空集合时它不会挂住，
    # 会在一个心跳周期后多吐一条 error 帧再收工——所以失败是「多了一条」，
    # 而不是「超时」，下面两句都指得出来。
    frames = [frame async for frame in _stream()]

    assert "event: agent_cancelled" in frames[-1], f"末帧应当是取消帧，实际是 {frames[-1]!r}"
    assert not any("stream_ended_without_terminal_event" in frame for frame in frames), (
        "取消不是「没收到结束事件」——合成的 error 帧会把正常取消显示成出错"
    )
