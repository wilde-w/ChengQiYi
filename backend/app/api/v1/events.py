"""流水线的 SSE 事件流。

`GET /runs/{id}/events?since=<seq>` 的语义：**把 seq 之后发生的一切按序补齐，
然后继续跟着发新的**。前端刷新页面时带上自己见过的最大 seq，就能无缺口续上。

帧的机器（回放合并、心跳、断开检测、终态收尾）在 `sse_stream.py` 里，与故事
工坊共用——那边的坑都是同一个坑。**本模块只剩「这个域特有的两件事」**：

1. **回放的第二源是 `run_event` 表。** Redis Stream 会被 MAXLEN 裁剪，或整个
   Redis 被清空；两种情况下都得能从头把日志补出来。
2. **终态运行要能立刻结束。** 已完成的运行不能再挂着一个永不关闭的连接，
   否则每次刷新页面都泄漏一个订阅。所以先查一次状态：终态 → 回放完就 `return`。

**为什么一个连接只服务一个运行。** 跨运行复用连接会让 `since` 变成
每个运行一个游标，而前端只有一个 `lastEventId`。宁可多开几条 HTTP 连接。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.api.v1 import sse_stream
from app.config import get_settings
from app.constants import EventType, RunStatus
from app.db.session import session_scope
from app.graph.bus import RunEvent, get_bus
from app.logging_conf import get_logger
from app.models.library import RunEvent as RunEventRow
from app.models.run import AnalysisRun

log = get_logger(__name__)
router = APIRouter(tags=["events"])

# 终态事件：收到它就该主动关闭连接。前端也会关，但服务端不该依赖客户端守规矩。
TERMINAL_EVENTS = frozenset({EventType.RUN_COMPLETED, EventType.RUN_CANCELLED, EventType.ERROR})

#: 命名空间。总线按它分键，两边的 seq 各自从 1 开始。
NAMESPACE = "run"


@router.get("/runs/{run_id}/events")
async def stream_events(
    run_id: str,
    request: Request,
    since: int = Query(default=0, ge=0, description="已收到的最大 seq，回放起点（不含）"),
) -> StreamingResponse:
    status = await _run_status(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail="运行不存在")

    return StreamingResponse(
        _event_stream(run_id, since=since, status=status, request=request),
        media_type="text/event-stream",
        headers=sse_stream.SSE_HEADERS,
    )


async def _run_status(run_id: str) -> str | None:
    async with session_scope() as session:
        run = await session.get(AnalysisRun, run_id)
        return None if run is None else run.status


async def _event_stream(
    run_id: str, *, since: int, status: str, request: Request
) -> AsyncIterator[str]:
    """帧的生产者。

    整体结构：先补历史（可能来自两处存储），再决定是订阅还是收工。
    把「补历史」与「订阅」分开写，是因为终态运行只需要前者。
    """
    heartbeat = max(5, get_settings().SSE_HEARTBEAT_SECONDS)
    terminal = RunStatus(status).is_terminal
    last_seq = since

    try:
        bus = await get_bus(NAMESPACE)
        replayed = await sse_stream.replay(bus, run_id, since, load_pg=_load_pg)
        for event in replayed:
            last_seq = max(last_seq, event.seq)
            yield event.to_sse()

        if terminal:
            # **终态运行也必须以一条终态帧收尾，不能只是「关掉连接」。**
            # 关连接对前端是歧义的：它分不清「跑完了」和「连接断了」——
            # 前者要拉快照对账，后者要重连。不分清的代价是页面永久卡在
            # 「分析中…」上（重连逻辑不会启动，因为连接是"正常"结束的）。
            # 事件表里没有终态帧时（进程被杀留下的运行），这里补一条。
            if not any(event.type in TERMINAL_EVENTS for event in replayed):
                yield sse_stream.error_frame(
                    "stream_ended_without_terminal_event",
                    f"运行状态已是 {status}，但未收到结束事件。请拉取快照。",
                    seq=last_seq + 1,
                    run_id=run_id,
                )
            else:
                yield sse_stream.comment(f"end replay seq={last_seq}")
            return

        # 从 last_seq 继续订阅。注意起点用 last_seq 而非 since：
        # 回放里最后一条的 seq 才是真正的断点。
        source = sse_stream.subscribe(
            bus,
            run_id,
            last_seq if last_seq else since,
            load_pg=_load_pg,
            terminal=TERMINAL_EVENTS,
        )
        async for frame in sse_stream.tail(
            source,
            heartbeat=heartbeat,
            request=request,
            status_of=lambda: _probe(run_id),
            run_id=run_id,
            start_seq=last_seq,
            terminal=TERMINAL_EVENTS,
        ):
            yield frame
    except Exception as exc:
        # 客户端断开（刷新页面、关标签）会以 CancelledError 的形式进来，
        # 它不是 Exception 的子类（3.8+），所以不会走到这里，无需记录。
        log.exception("sse_stream_failed", run_id=run_id, since=since)
        # 连接已经建立，改不了 HTTP 状态码，只能用事件把失败告诉前端。
        # 前端收到后应回退到轮询 /runs/{id} 快照。
        yield sse_stream.error_frame(
            "stream_failed", f"{type(exc).__name__}: {exc}", seq=last_seq + 1, run_id=run_id
        )


async def _probe(run_id: str) -> sse_stream.Probe | None:
    status = await _run_status(run_id)
    return None if status is None else sse_stream.Probe(status, RunStatus(status).is_terminal)


async def _load_pg(run_id: str, since: int) -> list[RunEvent]:
    """回放的第二个源：`run_event` 表。"""
    try:
        async with session_scope() as session:
            rows = (
                await session.execute(
                    select(RunEventRow)
                    .where(RunEventRow.run_id == run_id, RunEventRow.seq > since)
                    .order_by(RunEventRow.seq)
                )
            ).scalars()
            return [
                RunEvent(
                    seq=row.seq,
                    run_id=row.run_id,
                    type=EventType(row.type),
                    node=row.node,
                    progress=row.progress,
                    message=row.message,
                    data=row.data or {},
                )
                for row in rows
            ]
    except Exception as exc:
        log.warning("event_replay_from_pg_failed", run_id=run_id, error=str(exc))
        return []
