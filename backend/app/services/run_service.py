"""运行的生命周期：创建、执行、查询、取消。

任务跑在**进程内的 asyncio.Task**，不是外部队列。这是 V1 的刻意选择：
单 worker 部署（README 里说明了原因），省掉一套消息中间件。
代价是进程重启会丢掉正在跑的任务——README 里也说明了迁移到
任务队列（arq / celery）的路径，以及为什么 V1 不需要它。

链接解析放在这里而不是节点里：解析失败应该让 POST /runs 直接返回 400，
而不是创建一个注定失败、只能去读日志的运行记录。
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import desc, func, select

from app.constants import EventType, RunStatus, SourceKind, TextMode
from app.db.session import session_scope
from app.douyin.base import DouyinError
from app.graph.bus import RunEmitter, RunEvent, get_bus
from app.graph.registry import START_PROGRESS
from app.graph.runner import RunRunner
from app.graph.state import initial_state
from app.logging_conf import get_logger
from app.models.base import new_id
from app.models.library import RunEvent as RunEventRow
from app.models.run import AnalysisRun
from app.services.text_source import split_text

log = get_logger(__name__)

# 进程被杀时留下的运行，收尾时写进 error 的文案。前端会把这句话原样显示，
# 所以它得说清「发生了什么」，而不是一句「运行失败」。
INTERRUPTED_MESSAGE = "服务重启，运行已中断"

# run_id → 正在执行的 asyncio.Task。用于取消与「同一运行不重复启动」。
_tasks: dict[str, asyncio.Task[Any]] = {}
# 限制同时进行的分析数。每个运行会开多个外部连接，不设上限会在演示时
# 因为并发请求把自己拖垮。
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        from app.config import get_settings

        _semaphore = asyncio.Semaphore(max(1, get_settings().RUN_MAX_CONCURRENCY))
    return _semaphore


class RunNotFound(LookupError):
    pass


class ResolveFailed(ValueError):
    """输入无法解析成视频。属于用户错误，API 应回 400。"""


# ----------------------------------------------------------------------
# 创建
# ----------------------------------------------------------------------


async def create_run(payload: Any) -> tuple[AnalysisRun, dict[str, Any]]:
    """创建运行记录并**立刻**开始后台执行。返回 (记录, 解析信息)。"""
    from app.config import get_settings

    settings = get_settings()
    kind = SourceKind(str(payload.source))
    ref, resolved = await _resolve_input(payload, kind)

    providers = settings.provider_summary()
    # 判别式落在 providers 这个 JSON 列上，不新增数据库列（没有 alembic，
    # 加列要手动 ALTER 现有库）。它随 RunSummary 一起下发，前端因此不必
    # 自己去猜「aweme_id 为空就是文本源」这条隐式规则。
    providers["source_kind"] = kind.value
    if kind == SourceKind.TEXT:
        providers["text_mode"] = str(payload.text_mode)

    # id 在这里生成而不是交给 ORM 的 default：thread_id 必须等于 id，
    # 而 default 要到 flush 时才求值——那时 thread_id 早就按 None 写进去了。
    run_id = new_id()
    run = AnalysisRun(
        id=run_id,
        thread_id=run_id,
        input_raw=payload.input,
        # 文本源没有视频，这里就是 NULL——它也是老行的判别回落（见 source_kind_of）
        aweme_id=ref.aweme_id if ref is not None else None,
        depth=payload.depth,
        kb=payload.kb.model_dump(),
        comment_limit=payload.comment_limit,
        cluster_params=payload.cluster_params.model_dump(exclude_none=True),
        status=RunStatus.QUEUED.value,
        progress=int(START_PROGRESS),
        revision=1,
        warnings=[],
        providers=providers,
    )

    async with session_scope() as session:
        session.add(run)

    _spawn(run.id)
    return run, resolved


async def _resolve_input(payload: Any, kind: SourceKind) -> tuple[Any | None, dict[str, Any]]:
    """把请求里的 input 变成一次运行的身份标识：视频引用，或一段文本。

    放在这里而不是节点里，理由与抖音链接一样：输入不合法应当让 POST /runs
    当场 400，而不是先建一条注定失败、只能去读日志的运行记录。

    文本源的这一步只是**探针**——切不出条目就报错。真正的切分在 n2，
    它从 state 里读到的是原文；切分规则只有 `text_source.split_text` 一处。
    """
    if kind == SourceKind.TEXT:
        items, _ = split_text(
            str(payload.input), mode=str(payload.text_mode), limit=payload.comment_limit
        )
        if not items:
            raise ResolveFailed("文本里没有可分析的条目：内容为空或全是空行")
        # resolved_via 用 "text" 而不去扩 ResolvedVia：那个类型的语义是
        # 「抖音链接是怎么解析出来的」，文本源与它无关。
        return None, {"aweme_id": None, "canonical_url": "", "resolved_via": "text"}

    from app.douyin.factory import get_douyin_provider

    try:
        ref = await get_douyin_provider().resolve_link(payload.input)
    except DouyinError as exc:
        raise ResolveFailed(str(exc)) from exc
    return ref, {
        "aweme_id": ref.aweme_id,
        "canonical_url": ref.canonical_url,
        "resolved_via": ref.resolved_via,
    }


def source_kind_of(run: AnalysisRun) -> SourceKind:
    """这次运行的数据源。**判别式只有这一处**，别处不要再自己判断。

    首选 create_run 显式写下的 `providers["source_kind"]`；本轮之前建的老行
    没有这个键，退回 `aweme_id`——create_run 保证抖音源必有 id，所以
    「既没有标记、也没有 id」只可能是文本源。
    """
    raw = (run.providers or {}).get("source_kind")
    if raw is not None:
        try:
            return SourceKind(str(raw))
        except ValueError:
            log.warning("unknown_source_kind", run_id=run.id, value=str(raw))
    return SourceKind.TEXT if not run.aweme_id else SourceKind.DOUYIN


def text_mode_of(run: AnalysisRun) -> TextMode:
    """文本源的切分方式。缺失时取按行——它是弹窗里的默认值。"""
    try:
        return TextMode(str((run.providers or {}).get("text_mode")))
    except ValueError:
        return TextMode.LINE


def _spawn(run_id: str) -> None:
    existing = _tasks.get(run_id)
    if existing is not None and not existing.done():
        return
    task = asyncio.create_task(_run_guarded(run_id), name=f"run-{run_id}")
    _tasks[run_id] = task
    task.add_done_callback(lambda _t: _tasks.pop(run_id, None))


async def _run_guarded(run_id: str) -> None:
    async with _get_semaphore():
        try:
            await execute_run(run_id)
        except asyncio.CancelledError:
            log.info("run_task_cancelled", run_id=run_id)
            raise
        except Exception:
            # execute() 内部已经把失败写进库并发过事件了，这里只兜底日志，
            # 避免一个未捕获异常把事件循环的默认处理器刷屏
            log.exception("run_task_crashed", run_id=run_id)


async def execute_run(run_id: str, *, state: dict[str, Any] | None = None) -> RunStatus:
    """执行一次运行。`state` 传入时用于干预后的重跑（从既有状态续跑）。"""
    if state is None:
        async with session_scope() as session:
            run = await session.get(AnalysisRun, run_id)
            if run is None:
                raise RunNotFound(run_id)
            kind = source_kind_of(run)
            is_text = kind == SourceKind.TEXT
            state = initial_state(
                run_id=run_id,
                aweme_id=run.aweme_id or "",
                # 正文与链接各归各的字段。让 source_url 去装正文，那个名字
                # 就成了一句谎话，而它会一直骗到下一个读它的人。
                source_url="" if is_text else run.input_raw,
                source_kind=kind.value,
                source_text=run.input_raw if is_text else "",
                text_mode=text_mode_of(run).value,
                depth=run.depth,  # type: ignore[arg-type]
                kb=run.kb or {},
                comment_limit=run.comment_limit,
                cluster_params=run.cluster_params or {},
            )

    runner = RunRunner(run_id)
    return await runner.execute(state)


# ----------------------------------------------------------------------
# 查询
# ----------------------------------------------------------------------


async def get_run(run_id: str) -> AnalysisRun:
    async with session_scope() as session:
        run = await session.get(AnalysisRun, run_id)
        if run is None:
            raise RunNotFound(run_id)
        session.expunge(run)
        return run


async def list_runs(*, limit: int = 20, offset: int = 0) -> list[AnalysisRun]:
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(AnalysisRun)
                .order_by(desc(AnalysisRun.created_at))
                .limit(limit)
                .offset(offset)
            )
        ).scalars()
        return list(rows)


async def run_snapshot(run_id: str) -> dict[str, Any]:
    """一次运行的全部产物。前端在 run_completed 后用它做对账。"""
    from app.models.insight import Evidence, InsightSection, ReasoningStep
    from app.models.run import Cluster, Comment, PsychProfile, Video

    async with session_scope() as session:
        run = await session.get(AnalysisRun, run_id)
        if run is None:
            raise RunNotFound(run_id)

        video = (
            await session.execute(select(Video).where(Video.run_id == run_id))
        ).scalar_one_or_none()
        comments = list(
            (
                await session.execute(
                    select(Comment)
                    .where(Comment.run_id == run_id)
                    # 次级键不是装饰：文本源的点赞数全是 0，只按 like_count 排的话
                    # 并列顺序由数据库决定——表现是「刷新一下，左栏顺序变了」。
                    # 而粘贴的行的先后是有意义的（`txt-0001` 起，零填充保证字典序
                    # 等于序号序）。抖音源同赞数时按 id 排也是确定行为，无副作用。
                    .order_by(desc(Comment.like_count), Comment.douyin_comment_id)
                )
            ).scalars()
        )
        clusters = list(
            (
                await session.execute(
                    select(Cluster).where(Cluster.run_id == run_id).order_by(Cluster.order_index)
                )
            ).scalars()
        )
        profile = (
            await session.execute(select(PsychProfile).where(PsychProfile.run_id == run_id))
        ).scalar_one_or_none()
        evidence = list(
            (
                await session.execute(
                    select(Evidence).where(Evidence.run_id == run_id).order_by(Evidence.rank)
                )
            ).scalars()
        )
        reasoning = list(
            (
                await session.execute(
                    select(ReasoningStep)
                    .where(ReasoningStep.run_id == run_id)
                    .order_by(ReasoningStep.step_index)
                )
            ).scalars()
        )
        sections = list(
            (
                await session.execute(select(InsightSection).where(InsightSection.run_id == run_id))
            ).scalars()
        )

        cluster_key_by_id = {c.id: c.cluster_key for c in clusters}

        return {
            "run": run,
            "video": video,
            "comments": comments,
            "clusters": clusters,
            "cluster_meta": (clusters[0].params if clusters else {}) or {},
            "profile": profile,
            "evidence": evidence,
            "reasoning": reasoning,
            "sections": sections,
            "cluster_key_by_id": cluster_key_by_id,
        }


# ----------------------------------------------------------------------
# 取消
# ----------------------------------------------------------------------


async def request_cancel(run_id: str) -> bool:
    """请求取消。

    **有活跃任务时只置标志位**，由 runner 在节点边界检查后自行退出——
    强行 cancel() asyncio 任务会跳过事务提交与事件发送，
    前端会卡在一个永远等不到结束事件的流上。

    **没有活跃任务时就地收尾。** 这一行是上一次进程留下的僵尸：它的
    asyncio.Task 随进程一起没了，标志位永远不会有谁来读，前端点了中止
    也还是卡着——用户看到的「按钮点了没用」就是这么来的。
    """
    async with session_scope() as session:
        run = await session.get(AnalysisRun, run_id)
        if run is None:
            raise RunNotFound(run_id)
        if RunStatus(run.status).is_terminal:
            return False
        run.cancel_requested = True

    task = _tasks.get(run_id)
    if task is not None and not task.done():
        return True

    log.warning("cancel_without_task", run_id=run_id, hint="没有活跃任务，就地收尾")
    await _close_stale_run(run_id, status=RunStatus.CANCELLED)
    return True


async def reconcile_orphans() -> int:
    """把上一次进程死掉时留在 queued/running 的运行收成终态。**启动时调一次。**

    进程被杀不会写终态（`shutdown_tasks` 只覆盖优雅关闭），那一行就会永远
    停在 running：前端顶栏的开始按钮一直是灰的、进度条不动，而且没有任何
    线索指向「服务重启过」。SSE 的心跳兜底（`events.py` 的 `_tail`）只在
    **库里已是终态**时才收尾，所以它救不了这种行——必须在这里补上那一刀。

    与知识库导入的 `reap_stale_imports` 有一处不同：那**不碰 queued**
    （等用户确认的暂停是合法状态），这里**连 queued 一起收**。运行的
    queued 只是「任务还没被调度」，而那个 asyncio.Task 已经随进程消失，
    不存在任何一个进程会来推进它。

    **前提是单进程部署**（`server.py` 里 `workers=1`）。两个后端连同一个库
    时，后起的那个会把先起那个正在跑的运行收掉——它确实无从判断那些运行的
    任务还活着。要支持多实例得先有任务队列，那是 V1 之外的事。
    """
    async with session_scope() as session:
        rows = list(
            (
                await session.execute(
                    select(AnalysisRun).where(
                        AnalysisRun.status.in_([RunStatus.QUEUED.value, RunStatus.RUNNING.value])
                    )
                )
            ).scalars()
        )
        if not rows:
            return 0
        targets = [(run.id, await _last_seq(session, run.id)) for run in rows]
        for run in rows:
            run.status = RunStatus.FAILED.value
            run.error = {"code": "interrupted", "message": INTERRUPTED_MESSAGE}
            run.finished_at = datetime.now(UTC)
            run.current_node = None

    for run_id, last_seq in targets:
        await _emit_terminal(
            run_id,
            last_seq,
            EventType.ERROR,
            INTERRUPTED_MESSAGE,
            {"code": "interrupted", "message": INTERRUPTED_MESSAGE},
        )
    log.warning("runs_reconciled", count=len(targets), hint="上次运行遗留的未完成运行已标记为失败")
    return len(targets)


async def _last_seq(session: Any, run_id: str) -> int:
    """这条运行已落库的最大 seq。合成事件要接在它后面，不能重号。"""
    value = (
        await session.execute(select(func.max(RunEventRow.seq)).where(RunEventRow.run_id == run_id))
    ).scalar_one_or_none()
    return int(value or 0)


async def _close_stale_run(run_id: str, *, status: RunStatus) -> None:
    """把一条**没有活跃任务**的非终态运行就地收到终态。

    顺序是「先写库、再发事件」，不能反：反过来的话，事件先到而快照还是
    running，前端对账会看到「流说结束了、快照说还在跑」，两边打架。
    """
    async with session_scope() as session:
        run = await session.get(AnalysisRun, run_id)
        if run is None or RunStatus(run.status).is_terminal:
            return
        last_seq = await _last_seq(session, run_id)
        run.status = status.value
        run.finished_at = datetime.now(UTC)
        run.current_node = None

    await _emit_terminal(run_id, last_seq, EventType.RUN_CANCELLED, "分析已取消", None)


async def _emit_terminal(
    run_id: str,
    start_seq: int,
    event_type: EventType,
    message: str,
    data: dict[str, Any] | None,
) -> None:
    """补发一条终态事件。走 RunEmitter 是为了同时落总线与 PG：

    总线让**当前挂着的**流立刻收到并关闭；PG 让**刷新后重连**的流回放得到
    它（`since` 大于历史最大 seq 时，只有落库的这条能补上）。缺任何一半，
    都有一半的时间线会卡到心跳兜底——而那兜底发的是 error 帧，会把一次
    主动取消显示成「分析失败」。
    """
    with suppress(Exception):
        bus = await get_bus()
        emitter = RunEmitter(bus, run_id, start_seq=start_seq, on_emit=_persist_event)
        await emitter.emit(event_type, message=message, data=data or {})


async def _persist_event(event: RunEvent) -> None:
    """事件的落库回放源。runner 里那份是方法，这份给没有 runner 的收尾路径用。"""
    with suppress(Exception):
        async with session_scope() as session:
            session.add(
                RunEventRow(
                    run_id=event.run_id,
                    seq=event.seq,
                    type=event.type.value,
                    node=event.node,
                    progress=int(event.progress) if event.progress is not None else None,
                    message=event.message,
                    data=event.data or {},
                )
            )


def active_task_count() -> int:
    return sum(1 for t in _tasks.values() if not t.done())


async def shutdown_tasks() -> None:
    """应用关闭时收敛在跑的任务。"""
    pending = [t for t in _tasks.values() if not t.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    _tasks.clear()


__all__ = [
    "INTERRUPTED_MESSAGE",
    "ResolveFailed",
    "RunNotFound",
    "active_task_count",
    "create_run",
    "execute_run",
    "get_run",
    "list_runs",
    "reconcile_orphans",
    "request_cancel",
    "run_snapshot",
    "shutdown_tasks",
    "source_kind_of",
    "text_mode_of",
]
