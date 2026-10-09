"""流水线执行器：把图跑起来，把进展变成事件，把产物落库。

这是唯一一处同时接触「图 / 事件总线 / 数据库」的地方。三者都集中在此，
是为了让节点保持纯净（不碰 DB、不碰 seq），也让 SSE 的重连语义有唯一的实现点。

三个必须做对的地方：
  1. **多模式流**：`stream_mode=["updates","custom"]` 会 yield `(mode, chunk)` 元组。
     自定义事件里带 kind，updates 里是 {节点名: patch}。两者语义完全不同，别混。
  2. **每个事件都落 run_event 表**。Redis Stream 会被 MAXLEN 裁剪，
     重连时若 `since` 已不在流里，唯一的补救就是这个表。
  3. **节奏控制**。mock 模式下节点几乎瞬时完成，进度会四次跳变（2→15→35…），
     完全看不出流式感。MOCK_MIN_STEP_DELAY_MS 给每个里程碑之间加一点停顿。
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from app.constants import EventType, NodeKey, RunStatus, SourceKind
from app.db.session import session_scope
from app.graph import persist
from app.graph.builder import graph_context
from app.graph.bus import RunBus, RunEmitter, get_bus
from app.graph.emitting import (
    KIND_DELTA,
    KIND_MILESTONE,
    KIND_NODE_START,
    KIND_PARTIAL,
    KIND_PROGRESS,
    KIND_WARNING,
)
from app.graph.progress import ProgressTracker
from app.graph.registry import END_PROGRESS, done_template_for, entering_for
from app.logging_conf import get_logger
from app.models.library import RunEvent as RunEventRow
from app.models.run import AnalysisRun

log = get_logger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


class RunRunner:
    """一次分析运行。"""

    def __init__(self, run_id: str, *, bus: RunBus | None = None) -> None:
        self.run_id = run_id
        self._bus = bus
        self.emitter: RunEmitter | None = None
        self.progress = ProgressTracker()
        self.revision = 1
        self._last_emit_at = 0.0
        self._min_step_delay = 0.0
        # n1/n2 的进度文案按数据源取（见 registry.TEXT_SOURCE_COPY）：
        # 「正在解析视频链接…」对一次文本分析是假话。默认抖音，run 开始后由 state 覆盖。
        self._source_kind = SourceKind.DOUYIN.value

    # ------------------------------------------------------------------
    async def execute(self, state: dict[str, Any]) -> RunStatus:
        from app.config import get_settings

        settings = get_settings()
        # 只在演示模式下做节奏控制：真实 provider 本来就慢，再加延迟是雪上加霜
        if settings.is_demo:
            self._min_step_delay = max(0, settings.MOCK_MIN_STEP_DELAY_MS) / 1000.0
        self._source_kind = str(state.get("source_kind") or SourceKind.DOUYIN.value)

        # 总线是异步选出来的（要探一次 Redis），所以不能放在 __init__ 里
        self._bus = self._bus or await get_bus()
        self.emitter = RunEmitter(self._bus, self.run_id, on_emit=self._record)

        started = time.perf_counter()
        await self._mark_running()

        try:
            async with graph_context() as handle:
                if handle.degraded:
                    await self._warn(
                        "checkpointer_degraded",
                        f"检查点降级为内存模式（{handle.reason}），重启后干预与恢复能力会丢失。",
                    )
                await self._emit(
                    EventType.RUN_STARTED,
                    progress=0.0,
                    message="开始分析",
                    data={"revision": self.revision},
                )
                status = await self._stream(handle.graph, state)
                if status == RunStatus.CANCELLED:
                    # 用户在节点边界上点的中止。这是个**正常结局**，不是故障：
                    # 不发这一条的话，流只能靠 SSE 的心跳兜底收尾，而那条合成帧
                    # 是 error 类型——前端会把一次主动取消显示成「分析失败」。
                    with suppress(Exception):
                        await self._emit(EventType.RUN_CANCELLED, message="分析已取消")
        except asyncio.CancelledError:
            await self._finish(RunStatus.CANCELLED, started)
            with suppress(Exception):
                await self._emit(EventType.RUN_CANCELLED, message="分析已取消")
            raise
        except Exception as exc:
            log.exception("run_failed", run_id=self.run_id)
            await self._fail(exc)
            await self._finish(RunStatus.FAILED, started)
            return RunStatus.FAILED

        await self._finish(status, started)
        return status

    # ------------------------------------------------------------------
    async def _emit(self, event_type: EventType, **kwargs: Any) -> None:
        """发事件。emitter 在 execute() 里才建好，这里是唯一的转发点。"""
        if self.emitter is None:
            return
        await self.emitter.emit(event_type, **kwargs)

    async def _stream(self, graph: Any, state: dict[str, Any]) -> RunStatus:
        config = {
            "configurable": {"thread_id": self.run_id},
            "recursion_limit": 50,
        }

        async for mode, chunk in graph.astream(state, config, stream_mode=["updates", "custom"]):
            if await self._cancel_requested():
                return RunStatus.CANCELLED

            if mode == "custom":
                await self._on_custom(chunk)
            elif mode == "updates":
                await self._on_updates(chunk)

        if self.progress.current < END_PROGRESS:
            # 图正常结束却没走到 100，说明有节点没被访问到。宁可显式补上，
            # 也不要让前端卡在 9x% 上反复轮询。
            self.progress.finish(NodeKey.N7_LITERARY)
        await self._emit(
            EventType.RUN_COMPLETED,
            progress=END_PROGRESS,
            message="分析完成",
            data={"revision": self.revision},
        )
        return RunStatus.SUCCEEDED

    # ------------------------------------------------------------------
    async def _on_custom(self, chunk: Any) -> None:
        if not isinstance(chunk, dict):
            return
        kind = chunk.get("kind")
        node = chunk.get("node")

        if kind == KIND_NODE_START:
            value = self.progress.begin(node)
            await self._emit(
                EventType.NODE_STARTED,
                node=node,
                progress=value,
                message=entering_for(node, source_kind=self._source_kind),
            )
            return

        if kind == KIND_MILESTONE:
            await self._pace()
            await self._emit(
                EventType.PROGRESS,
                node=node,
                progress=self.progress.current,
                message=chunk.get("message"),
            )
            return

        if kind == KIND_PROGRESS:
            value = self.progress.set(float(chunk.get("fraction") or 0.0))
            await self._emit(EventType.PROGRESS, node=node, progress=value)
            return

        if kind == KIND_PARTIAL:
            await self._emit(
                EventType.PARTIAL,
                node=node,
                progress=self.progress.current,
                message=chunk.get("message"),
                # **展开在前、判别键在后**，顺序不能反。payload 是节点自由填的，
                # 证据项恰好自带一个 `kind`（psychology/literature/poetry）；
                # 写成 {"kind": ..., **payload} 的话它会把判别键覆盖掉，
                # 前端按 data.kind 分派时一个都认不出来——表现是证据卡一张都不出现，
                # 而事件流里条条都在，排查时最容易往「没发出去」的方向找。
                data={**(chunk.get("data") or {}), "kind": chunk.get("data_kind")},
            )
            return

        if kind == KIND_DELTA:
            await self._emit(
                EventType.DELTA,
                node=node,
                data={"section": chunk.get("section"), "text": chunk.get("text") or ""},
            )
            return

        if kind == KIND_WARNING:
            await self._emit(
                EventType.WARNING,
                node=node,
                message=chunk.get("message"),
                data={"code": chunk.get("code")},
            )

    async def _on_updates(self, chunk: Any) -> None:
        if not isinstance(chunk, dict):
            return
        for node_name, patch in chunk.items():
            if not isinstance(patch, dict):
                continue
            key = NodeKey(node_name) if node_name in set(NodeKey) else None
            if key is None:
                continue

            value = self.progress.finish(key)
            await self._commit(node_name, patch)

            message = _format_done(key, patch, source_kind=self._source_kind)
            await self._pace()
            await self._emit(
                EventType.NODE_COMPLETED,
                node=node_name,
                progress=value,
                message=message,
                data={"revision": self.revision},
            )

    # ------------------------------------------------------------------
    async def _commit(self, node: str, patch: dict[str, Any]) -> None:
        """落库。失败不能中断分析——产物已经通过事件发给前端了，
        库写不进去是「历史查不到」，不是「这次分析没结果」。"""
        try:
            async with session_scope() as session:
                await persist.apply_patch(
                    session, run_id=self.run_id, node=node, patch=patch, revision=self.revision
                )
                run = await session.get(AnalysisRun, self.run_id)
                if run is not None:
                    run.current_node = node
                    run.progress = int(self.progress.current)
                    # warnings 由状态 reducer 累积，这里只把本次新增的并进去
                    extra = patch.get("warnings")
                    if extra:
                        run.warnings = [*(run.warnings or []), *extra]
        except Exception as exc:
            log.warning("persist_failed", run_id=self.run_id, node=node, error=str(exc))

    async def _record(self, event: Any) -> None:
        """把事件写进 run_event 表，作为 Streams 被裁剪后的回放兜底。"""
        try:
            async with session_scope() as session:
                session.add(
                    RunEventRow(
                        run_id=self.run_id,
                        seq=event.seq,
                        type=event.type.value,
                        node=event.node,
                        progress=int(event.progress) if event.progress is not None else None,
                        message=event.message,
                        data=event.data or {},
                    )
                )
        except Exception as exc:
            log.warning("event_persist_failed", seq=event.seq, error=str(exc))

    # ------------------------------------------------------------------
    async def _pace(self) -> None:
        """里程碑之间的最小间隔。只在演示模式下生效，见 execute() 的说明。"""
        if self._min_step_delay <= 0:
            return
        elapsed = time.perf_counter() - self._last_emit_at
        if elapsed < self._min_step_delay:
            await asyncio.sleep(self._min_step_delay - elapsed)
        self._last_emit_at = time.perf_counter()

    async def _cancel_requested(self) -> bool:
        try:
            async with session_scope() as session:
                run = await session.get(AnalysisRun, self.run_id)
                return bool(run and run.cancel_requested)
        except Exception:
            return False

    async def _mark_running(self) -> None:
        try:
            async with session_scope() as session:
                run = await session.get(AnalysisRun, self.run_id)
                if run is not None:
                    run.status = RunStatus.RUNNING.value
                    run.started_at = _now()
                    self.revision = int(run.revision or 1)
        except Exception as exc:
            log.warning("mark_running_failed", run_id=self.run_id, error=str(exc))

    async def _finish(self, status: RunStatus, started: float) -> None:
        try:
            async with session_scope() as session:
                run = await session.get(AnalysisRun, self.run_id)
                if run is not None:
                    run.status = status.value
                    run.finished_at = _now()
                    run.duration_ms = int((time.perf_counter() - started) * 1000)
                    run.progress = (
                        int(END_PROGRESS) if status == RunStatus.SUCCEEDED else run.progress
                    )
                    if status == RunStatus.SUCCEEDED:
                        run.current_node = None
        except Exception as exc:
            log.warning("finish_failed", run_id=self.run_id, error=str(exc))

    async def _fail(self, exc: Exception) -> None:
        payload = {"code": "run_failed", "message": f"{type(exc).__name__}: {exc}"}
        with suppress(Exception):
            await self._emit(EventType.ERROR, message=payload["message"], data=payload)
        try:
            async with session_scope() as session:
                run = await session.get(AnalysisRun, self.run_id)
                if run is not None:
                    run.error = payload
        except Exception as exc2:
            log.warning("fail_persist_failed", error=str(exc2))

    async def _warn(self, code: str, message: str) -> None:
        with suppress(Exception):
            await self._emit(EventType.WARNING, message=message, data={"code": code})


def _format_done(key: NodeKey, patch: dict[str, Any], *, source_kind: str = "") -> str:
    """节点完成文案：注册表模板 + 节点返回值里数出来的真实数字。

    数字从 patch 里现算，而不是让节点自己发一句「已获取 N 条评论」——
    节点忘了发就永远缺一条，而这里是从节点实际交出的产物推出来的，
    发不发得出来不取决于节点的自觉。
    """
    template = done_template_for(key, source_kind=source_kind or SourceKind.DOUYIN.value)
    fields: dict[str, Any] = {}
    if key == NodeKey.N2_COMMENTS:
        fields["count"] = (patch.get("comment_stats") or {}).get("total", 0)
    elif key == NodeKey.N3_CLUSTER:
        fields["k"] = len(patch.get("clusters") or [])
    elif key == NodeKey.N5_RETRIEVAL:
        evidence = patch.get("evidence") or []
        fields["psychology"] = sum(1 for e in evidence if e.get("kind") == "psychology")
        fields["literature"] = sum(1 for e in evidence if e.get("kind") in ("literature", "poetry"))
    elif key == NodeKey.N6_REASONING:
        fields["steps"] = len(patch.get("reasoning") or [])
    try:
        return template.format(**fields) if fields else template
    except (KeyError, IndexError):
        return template
