"""SSE 帧机器：回放、心跳、断开检测、终态收尾。

**这里住的是「怎么把事件变成帧」，不是「事件从哪来」。** 流水线（`events.py`）
与故事工坊（`agent.py`）共用这一份——两套 SSE 实现分头演进的结果一定是其中
一套悄悄坏掉，而坏掉的表现是「刷新之后内容对不上」，不会有任何报错。

所以下面这些细节**一个字都不能改**，它们各自对应一个已经踩过的坑：

- **错误帧必须带 `seq` 和 `run_id`。** 前端把它们当普通事件读并推进自己的
  游标；缺了 `seq` 游标会被写成 `undefined`，下次重连拼出 `?since=undefined`，
  服务端按整数解析直接 422——表现为「网络一抖，流就再也接不上了」。
- **心跳是注释帧。** `: ping` 不占 seq、不进事件表、不触发前端任何状态变化。
  用真事件保活会把「没有进展」伪装成「有进展」。
- **断开检测挂在心跳那次写入上。** 只写不读时，客户端消失不会让协程收到
  `CancelledError`，连接会一直挂着直到下一次写失败——心跳恰好是那个写入点。
- **终态回放完要补一条终态帧，不能只是关掉连接。** 关连接对前端是歧义的：
  它分不清「跑完了」和「连接断了」，前者要拉快照对账，后者要重连。不分清的
  代价是页面永久卡在「进行中…」。

## 与具体存储的分工

回放有两个源（Redis Stream + PG 兜底），但 PG 那张表是**各自的**：
流水线查 `run_event`，故事工坊查 `agent_event`。所以这里把「怎么从 PG 读」
作为 `load_pg` 回调注入，本模块只管合并、排序、去重与去重后的边界情况。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import Any, NamedTuple

from fastapi import Request

from app.constants import EventType
from app.graph.bus import RunEvent
from app.logging_conf import get_logger

log = get_logger(__name__)

Frame = str


class Probe(NamedTuple):
    """一次状态探测的结果。

    `terminal` 是判据，`label` 只进错误文案——两者分开是因为库里存的是
    `succeeded` / `idle` 这种机器词，而那句话是给用户看的。
    """

    label: str
    terminal: bool


#: 从 PG 补历史。第一个参数是「谁的」（run_id 或 session_id），第二个是起点。
LoadPaged = Callable[[str, int], Awaitable[list[RunEvent]]]
#: 查状态。`None` → 这个 id 根本不存在。
StatusOf = Callable[[], Awaitable[Probe | None]]

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    # nginx 默认会缓冲响应，那样 SSE 就退化成「一次性吐完」。显式关掉。
    "X-Accel-Buffering": "no",
}


async def replay(bus: Any, run_id: str, since: int, *, load_pg: LoadPaged) -> list[RunEvent]:
    """补齐 seq > since 的历史。总线与 PG 各查一次，按 seq 去重合并。

    两处都查是必要的，且顺序不能反：总线（Streams）是主源、最新；
    PG 是兜底，只在**它比总线拿到更多**时才补进去——否则会把已被
    MAXLEN 裁剪掉头部的流重新拼出一个缺了一段的序列。
    """
    events: dict[int, RunEvent] = {}

    replay_fn = getattr(bus, "replay", None)
    if replay_fn is not None:
        with suppress(Exception):
            for event in await replay_fn(run_id, since):
                if event.seq > since:
                    events[event.seq] = event

    # PG 是兜底，不是主源。但两种情况必须查它：
    #   - Stream 被 MAXLEN 裁掉了头部（events 里没有最小那几段）
    #   - 总线降级成了 InProcessBus 且进程重启过（一条都补不出来）
    missing = await load_pg(run_id, since)
    if len(missing) > len(events):
        for event in missing:
            events.setdefault(event.seq, event)

    return [events[seq] for seq in sorted(events)]


async def subscribe(
    bus: Any,
    run_id: str,
    since: int,
    *,
    load_pg: LoadPaged,
    terminal: frozenset[EventType],
) -> AsyncIterator[RunEvent]:
    """订阅增量。总线不支持 subscribe 时退化为长轮询 PG。"""
    subscribe_fn = getattr(bus, "subscribe", None)
    if subscribe_fn is None:
        async for event in _poll_pg(run_id, since, load_pg=load_pg, terminal=terminal):
            yield event
        return
    async for event in subscribe_fn(run_id, since):
        yield event


async def tail(
    source: AsyncIterator[RunEvent],
    *,
    heartbeat: int,
    request: Request,
    status_of: StatusOf,
    run_id: str,
    start_seq: int,
    terminal: frozenset[EventType],
) -> AsyncIterator[Frame]:
    """给事件流加心跳、断开检测与「僵尸任务」兜底。

    心跳时顺带查一次状态。这一查是为了**进程重启**：任务死在 `RUNNING` 上，
    库里永远不会再更新，如果只等事件，用户的进度条会卡在 43% 直到他放弃。
    发现状态已终态就主动收工——前端会拉快照并显示真实结局。

    自己记 `last_seq`（从回放终点 `start_seq` 起）是为了合成帧也有合法的 seq：
    游标停在回放终点不动，合成帧就只能从现在这条流已送达的最大值往上加。
    """
    last_seq = start_seq
    queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()
    pump_task = asyncio.create_task(pump(source, queue))
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=heartbeat)
            except TimeoutError:
                if await request.is_disconnected():
                    return
                yield comment("ping")
                probe = await status_of()
                if probe is not None and probe.terminal:
                    # 库里说它结束了，但流里没送来终态事件。把排空的机会让给
                    # 队列（可能刚好有事件在路上），实在没有就发一条合成的。
                    with suppress(TimeoutError):
                        queued = await asyncio.wait_for(queue.get(), timeout=0.5)
                        if queued is not None:
                            last_seq = max(last_seq, queued.seq)
                            frame = queued.to_sse()
                            yield frame
                            if is_terminal_frame(frame, terminal):
                                return
                    yield error_frame(
                        "stream_ended_without_terminal_event",
                        f"状态已是 {probe.label}，但未收到结束事件。请拉取快照。",
                        seq=last_seq + 1,
                        run_id=run_id,
                    )
                    return
                continue
            if event is None:
                return
            last_seq = max(last_seq, event.seq)
            frame = event.to_sse()
            yield frame
            if is_terminal_frame(frame, terminal):
                return
    finally:
        pump_task.cancel()
        with suppress(asyncio.CancelledError):
            await pump_task


async def pump(
    source: AsyncIterator[RunEvent], queue: asyncio.Queue[RunEvent | None]
) -> None:
    try:
        async for event in source:
            await queue.put(event)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.warning("event_source_failed", error=str(exc))
    finally:
        # 哨兵：告诉消费者源已结束（正常结束或出错，都不该让连接悬着）
        with suppress(asyncio.QueueFull):
            queue.put_nowait(None)


async def _poll_pg(
    run_id: str,
    since: int,
    *,
    load_pg: LoadPaged,
    terminal: frozenset[EventType],
    interval: float = 1.0,
) -> AsyncIterator[RunEvent]:
    """无订阅能力时的退路：按 seq 增量轮询事件表。

    只在总线降级且 PG 可用时走到。轮询间隔取 1s——比 SSE 差，但比卡住好。
    """
    cursor = since
    while True:
        events = await load_pg(run_id, cursor)
        for event in events:
            cursor = event.seq
            yield event
        if events and events[-1].type in terminal:
            return
        await asyncio.sleep(interval)


def comment(text: str) -> Frame:
    """SSE 注释帧。客户端会忽略，但能保活并探测断开。"""
    return f": {text}\n\n"


def error_frame(code: str, message: str, *, seq: int, run_id: str) -> Frame:
    """连接建立之后的失败通知。

    **必须带 `seq` 和 `run_id`。** 前端把它们当作普通事件读，收到后推进
    自己的游标（`sse.ts`）；缺了 `seq` 游标会被写成 `undefined`，
    下一次重连就拼出 `?since=undefined`，服务端按整数解析直接 422——
    表现为「网络一抖，流就再也接不上了」，而日志里只有一条 422。
    """
    payload = json.dumps(
        {
            "seq": seq,
            "run_id": run_id,
            "type": EventType.ERROR.value,
            "code": code,
            "message": message,
            "node": None,
            "progress": None,
            "data": {},
            "ts": time.time(),
        },
        ensure_ascii=False,
    )
    return f"event: {EventType.ERROR.value}\ndata: {payload}\n\n"


def is_terminal_frame(frame: Frame, terminal: frozenset[EventType]) -> bool:
    """从已序列化的帧里判终态。

    避免为此把 RunEvent 对象一路带到 `tail` 之外——这里只做一次字符串前缀
    判断，比给生成器加一层返回类型更省事，也不会漏掉 ERROR（它同样意味着
    这个 id 不会再产出新事件）。
    """
    return any(f"event: {t.value}\n" in frame for t in terminal)


__all__ = [
    "SSE_HEADERS",
    "Probe",
    "comment",
    "error_frame",
    "is_terminal_frame",
    "replay",
    "subscribe",
    "tail",
]
