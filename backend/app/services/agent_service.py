"""故事工坊的服务层：会话、消息、事件、一轮的执行。

`app/agent/` 里那个循环**不碰数据库**——它只收一串消息、吐一串消息与事件。
落库、发总线、更新状态全在这里，理由和流水线那边一样：控制流与持久化分开，
「模型决定了什么」与「我们把它记成了什么」才是两件能分别检查的事。

## 四条约束，每一条都对应一个已经想清楚的失败

1. **一条 assistant 说了几次调用，就要有几条 `role="tool"` 回复，且逐字落库。**
   库里这串就是下一轮喂给模型的那串。落库时漏一条 tool 回复，症状是第三轮
   突然 400——离现场很远。
2. **先写库、再发终态事件。** 反过来的话，前端收到「写完了」去拉快照，却看到
   还在 `running`，两边打架，界面不知道该信谁。
3. **每个出口都要发一条终态事件。** 前端靠它把「正在写」关掉。漏一条，面板
   永远转圈，而用户等的是一个不会再来的结果。
4. **取消只在轮边界生效。** 强行 `task.cancel()` 会跳过事务提交与事件发送，
   前端会挂在一个永远等不到结束的流上——这与流水线的取舍一致。
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select

from app.agent import events as ev
from app.agent.events import Emit
from app.agent.loop import TurnRequest, TurnResult, run_turn
from app.agent.tools import build_registry
from app.config import get_settings
from app.constants import AgentStage, AgentStatus, EventType
from app.db.session import session_scope
from app.graph.bus import RunEmitter, RunEvent, get_bus
from app.logging_conf import get_logger
from app.models.agent import AgentEvent, AgentMessage, AgentSession
from app.prompts import story as story_prompt
from app.providers.base import ChatMessage, ToolCall
from app.providers.factory import get_chat_model

log = get_logger(__name__)

#: 同时进行的轮数上限。V1 是单人工位，正常永远只有 1；留着是因为
#: 「一次写故事要跑 10–60 秒」，而面板是可以被疯狂点的。超出就排队，
#: 不拒绝——拒绝对用户没有任何好处，他要的只是别把机器占满。
MAX_CONCURRENT_TURNS = 2

INTERRUPTED_MESSAGE = "服务重启过，这一轮没有跑完。请重新开始。"

#: 标题取原文开头这么多字。只用于列表与面板标题，不参与任何判据。
TITLE_CHARS = 18

_tasks: dict[str, asyncio.Task[Any]] = {}
_semaphore: asyncio.Semaphore | None = None
#: 取消标志的快速通道。库里那一列是给「重启之后还能看出被取消过」用的，
#: 而轮边界上的检查是**每个模型调用前都会跑一次**的热路径，不该查库。
_cancel_requested: set[str] = set()


class AgentNotFound(LookupError):
    """会话不存在。API 回 404。"""


class AgentRejected(ValueError):
    """用户可修的输入问题（会话已结束、上一轮还没写完）。API 回 409。"""


@dataclass(slots=True)
class SessionDetail:
    """打开面板要的全部东西。`demo` 由 API 层填（它才知道 provider 摘要）。"""

    session: AgentSession
    messages: list[AgentMessage]


# ----------------------------------------------------------------------
# 会话
# ----------------------------------------------------------------------


async def create_session(
    *,
    input_text: str,
    allow_novel: bool = True,
    target_chars: int | None = None,
) -> AgentSession:
    """开一个会话。**只落一行**，不调模型——第一句话由用户说。"""
    settings = get_settings()
    options = {
        "allow_novel": bool(allow_novel),
        "target_chars": int(target_chars or story_prompt.TARGET_CHARS),
    }
    row = AgentSession(
        title=_title_of(input_text),
        input_text=input_text,
        input_chars=len(input_text),
        status=AgentStatus.IDLE.value,
        #: 演示模式必须落库：面板上那个徽章要在刷新之后还在——
        #: 「这几百字是模板拼的」是用户判断值不值得读的唯一依据。
        providers=settings.provider_summary(),
        model=model_name(),
        options=options,
    )
    async with session_scope() as session:
        session.add(row)
    log.info(
        "agent_session_created",
        session_id=row.id,
        chars=row.input_chars,
        allow_novel=options["allow_novel"],
        target_chars=options["target_chars"],
        demo=settings.is_demo,
    )
    return row


async def get_session(session_id: str) -> AgentSession:
    async with session_scope() as session:
        row = await session.get(AgentSession, session_id)
        if row is None:
            raise AgentNotFound(session_id)
        session.expunge(row)
        return row


async def session_detail(session_id: str) -> SessionDetail:
    """会话 + 全部对话。刷新页面时前端重建界面就靠这一份。"""
    async with session_scope() as session:
        row = await session.get(AgentSession, session_id)
        if row is None:
            raise AgentNotFound(session_id)
        messages = await _messages_in(session, session_id)
        session.expunge(row)
        for message in messages:
            session.expunge(message)
    return SessionDetail(session=row, messages=messages)


async def session_status(session_id: str) -> str | None:
    """SSE 的心跳兜底要读它。`None` 表示会话不存在。"""
    async with session_scope() as session:
        row = await session.get(AgentSession, session_id)
        return None if row is None else row.status


# ----------------------------------------------------------------------
# 一轮
# ----------------------------------------------------------------------


async def send_message(session_id: str, text: str) -> AgentSession:
    """接着写一句（或改一稿），并立刻把这一轮排上。

    **user 消息先落库、状态先翻成 running，然后才 spawn。** 反过来的话，
    任务可能在状态还没写之前就发出第一条事件，前端拉快照会看到「流里已经在
    思考、快照说闲着」。
    """
    async with session_scope() as session:
        row = await session.get(AgentSession, session_id)
        if row is None:
            raise AgentNotFound(session_id)
        status = AgentStatus(row.status)
        if status.is_terminal:
            # label 自带「已」字（已中断 / 已停止），模板里别再补一个——
            # 「已经已停止了」这种句子一眼就能看出是拼出来的。
            raise AgentRejected(f"这个会话{status.label}，请重新开始一个。")
        if status is AgentStatus.RUNNING:
            raise AgentRejected("上一轮还在写。等它写完，或者先点停止。")

        turn = row.turn + 1
        # seq 接着会话里的最大值走。它是 `(session_id, seq)` 唯一约束的那一半，
        # 也是前端「刷新后接着聊」时判断「哪些是新的」的依据。
        seq = await _last_message_seq(session, session_id) + 1
        session.add(
            AgentMessage(session_id=session_id, seq=seq, turn=turn, role="user", content=text)
        )
        row.turn = turn
        row.status = AgentStatus.RUNNING.value
        row.stage = AgentStage.PREPARING.value
        row.cancel_requested = False
        row.error = None
        row.finished_at = None
        if row.started_at is None:
            row.started_at = datetime.now(UTC)
        # **这里不能 `expunge`。** 别处（`get_session` / `get_run`）用它把行交出去
        # 是安全的，因为那些行是干净的；`expunge()` 会把**还没 flush 的改动丢掉**
        # ——内存里 `row.status` 已经是 running，库里那一行还是 idle。症状是
        # 「任务已经在跑，快照说它闲着」，或者更隐蔽：任务按库里那个没更新过的
        # `turn` 去比对消息，直接判成序列不完整。
        #
        # 不能 `expunge` 就得处理另一件事：`updated_at` 是 `onupdate` 生成的，
        # flush 之后那一列处于**已过期**状态，要在会话还开着的时候读回来，
        # 否则出了会话再访问就是 DetachedInstanceError（路由的 DTO 里就有它）。
        await session.flush()
        await session.refresh(row)

    _cancel_requested.discard(session_id)
    _spawn(session_id)
    log.info("agent_turn_queued", session_id=session_id, turn=turn, chars=len(text))
    return row


async def execute_turn(session_id: str, turn: int) -> None:
    """跑完一轮并落库。**每个出口都必须发一条终态事件。**"""
    async with session_scope() as session:
        row = await session.get(AgentSession, session_id)
        if row is None:
            log.warning("agent_turn_missing_session", session_id=session_id)
            return
        options = dict(row.options or {})
        input_text = row.input_text
        messages = await _messages_in(session, session_id)
        event_seq = await _last_event_seq(session, session_id)

    if not messages or messages[-1].role != "user" or messages[-1].turn != turn:
        # 不该发生：user 消息是 send_message 在同一处落的。真发生了说明有人
        # 手改了库，此时**不能猜**——猜错的代价是把两轮的消息混着发出去。
        await _finish(session_id, turn, stopped="error", error="会话消息序列不完整，这一轮没有开始。")
        return

    history = [_to_chat_message(m) for m in messages[:-1]]
    user_message = messages[-1].content

    registry = build_registry(
        input_text=input_text,
        allow_novel=bool(options.get("allow_novel", True)),
    )
    bus = await get_bus(ev.NAMESPACE)
    emitter = RunEmitter(bus, session_id, start_seq=event_seq, on_emit=_persist_event)
    emit = _make_emit(emitter, session_id)

    await emit(EventType.AGENT_STARTED, ev.started(turn=turn))
    result = await run_turn(
        TurnRequest(
            system_prompt=story_prompt.build_system_prompt(
                input_text,
                target_chars=int(options.get("target_chars") or story_prompt.TARGET_CHARS),
                allow_novel=bool(options.get("allow_novel", True)),
            ),
            history=history,
            user_message=user_message,
            turn=turn,
        ),
        model=get_chat_model(),
        registry=registry,
        emit=emit,
        should_cancel=_canceller(session_id),
    )

    await _persist_turn(session_id, turn, result)
    # **先写库、再发事件。** 前端收到终态后的第一件事是拉快照对账，那一刻库里
    # 必须已经是终态，否则它会看到「流说结束了、快照说还在写」——取消这条路上
    # 曾经就是这样（循环自己发终态，抢在落库前面），表现是点了停止之后面板
    # 还亮着「写作中」。所以三种终态**全都在这里发**，循环一个都不发。
    await _emit_terminal_event(emitter, result, turn=turn)
    log.info(
        "agent_turn_done",
        session_id=session_id,
        turn=turn,
        stopped=result.stopped,
        rounds=result.rounds,
        tool_calls=result.tool_calls,
        chars=len(result.text),
    )


async def request_cancel(session_id: str) -> bool:
    """请求停止。**有活跃任务时只置标志位**，由循环在轮边界检查后自行退出。

    与流水线同样的理由：`task.cancel()` 会跳过事务提交与事件发送，前端会挂在
    一个永远等不到结束事件的流上。没有活跃任务时（上一次进程留下的僵尸）就地
    收尾——那一轮的标志位永远不会有谁来读。
    """
    async with session_scope() as session:
        row = await session.get(AgentSession, session_id)
        if row is None:
            raise AgentNotFound(session_id)
        if AgentStatus(row.status) is not AgentStatus.RUNNING:
            return False
        row.cancel_requested = True
        turn = row.turn

    _cancel_requested.add(session_id)
    task = _tasks.get(session_id)
    if task is not None and not task.done():
        return True

    log.warning("agent_cancel_without_task", session_id=session_id, hint="没有活跃任务，就地收尾")
    await _close_session(session_id, turn=turn, status=AgentStatus.CANCELLED)
    return True


# ----------------------------------------------------------------------
# 生命周期
# ----------------------------------------------------------------------


async def reap_stale_agent_sessions() -> int:
    """把上一次进程死掉时留在 running 的会话收成 failed。**启动时调一次。**

    与 `reconcile_orphans` 是同一件事：进程被杀不会写终态，那一行会永远停在
    running，前端打开它只会看到一个转不完的圈。SSE 的心跳兜底也救不了——
    它只在库里已是终态时才收尾。

    **前提是单进程部署**（`server.py` 里 `workers=1`）。
    """
    async with session_scope() as session:
        rows = list(
            (
                await session.execute(
                    select(AgentSession).where(AgentSession.status == AgentStatus.RUNNING.value)
                )
            ).scalars()
        )
        if not rows:
            return 0
        targets = [(row.id, await _last_event_seq(session, row.id)) for row in rows]
        for row in rows:
            row.status = AgentStatus.FAILED.value
            row.stage = None
            row.error = INTERRUPTED_MESSAGE
            row.cancel_requested = False
            row.finished_at = datetime.now(UTC)

    for session_id, seq in targets:
        with suppress(Exception):
            bus = await get_bus(ev.NAMESPACE)
            emitter = RunEmitter(bus, session_id, start_seq=seq, on_emit=_persist_event)
            await emitter.emit(
                EventType.ERROR,
                message=INTERRUPTED_MESSAGE,
                data=ev.error(message=INTERRUPTED_MESSAGE),
            )
    log.warning("agent_sessions_reaped", count=len(targets), hint="服务重启过，把它们收成 failed")
    return len(targets)


async def shutdown_agent_tasks() -> None:
    """应用关闭时收敛在跑的任务。"""
    pending = [t for t in _tasks.values() if not t.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    _tasks.clear()
    _cancel_requested.clear()


def active_task_count() -> int:
    return sum(1 for t in _tasks.values() if not t.done())


# ----------------------------------------------------------------------
# 事件回放（SSE 用）
# ----------------------------------------------------------------------


async def load_events(session_id: str, since: int) -> list[RunEvent]:
    """`agent_event` 表里 seq > since 的事件。SSE 回放的第二源。

    第一源是 Redis Stream，它会被 MAXLEN 裁剪；这张表是「刷新后回放没有
    缺口」的兜底。`run_id` 装的是 session_id——信封沿用 `RunEvent` 是刻意的，
    前端的帧解析因此一行都不用改（见 `agent/events.py` 的模块注释）。
    """
    try:
        async with session_scope() as session:
            rows = (
                await session.execute(
                    select(AgentEvent)
                    .where(AgentEvent.session_id == session_id, AgentEvent.seq > since)
                    .order_by(AgentEvent.seq)
                )
            ).scalars()
            return [
                RunEvent(
                    seq=row.seq,
                    run_id=row.session_id,
                    type=EventType(row.type),
                    message=row.message,
                    data=row.data or {},
                )
                for row in rows
            ]
    except Exception as exc:
        log.warning("agent_event_replay_failed", session_id=session_id, error=str(exc))
        return []


# ----------------------------------------------------------------------
# 内部：任务与事件
# ----------------------------------------------------------------------


def _spawn(session_id: str) -> None:
    existing = _tasks.get(session_id)
    if existing is not None and not existing.done():
        return
    task = asyncio.create_task(_guarded(session_id), name=f"agent-{session_id}")
    _tasks[session_id] = task
    task.add_done_callback(lambda _t: _tasks.pop(session_id, None))


async def _guarded(session_id: str) -> None:
    async with _get_semaphore():
        async with session_scope() as session:
            row = await session.get(AgentSession, session_id)
            turn = 0 if row is None else row.turn
        try:
            await execute_turn(session_id, turn)
        except asyncio.CancelledError:
            log.info("agent_task_cancelled", session_id=session_id)
            raise
        except Exception as exc:
            # 未预期的异常也必须变成一次干净的收尾：面板上永远转圈是更坏的结果。
            log.exception("agent_task_crashed", session_id=session_id)
            await _finish(
                session_id, turn, stopped="error", error=f"{type(exc).__name__}: {exc}"
            )


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_TURNS)
    return _semaphore


def _canceller(session_id: str):
    async def should_cancel() -> bool:
        return session_id in _cancel_requested

    return should_cancel


def _make_emit(emitter: RunEmitter, session_id: str) -> Emit:
    """做这一轮的 `emit`，顺带把会话的 `stage` 维护成实话。

    `stage` 是给**刷新后**看的那一句（「正在查资料…」）。工具卡片那些细节由
    事件回放重建，所以它只是个粗粒度的提示，不值得每个事件写一次库——
    这里只在阶段真的变了才写。
    """
    current: list[AgentStage | None] = [AgentStage.PREPARING]

    async def emit(event_type: EventType, data: dict[str, Any]) -> None:
        await emitter.emit(event_type, data=data)
        stage = _stage_of(event_type)
        if stage == current[0]:
            return
        current[0] = stage
        with suppress(Exception):
            async with session_scope() as session:
                row = await session.get(AgentSession, session_id)
                if row is not None:
                    row.stage = None if stage is None else stage.value

    return emit


def _stage_of(event_type: EventType) -> AgentStage | None:
    """事件 → 阶段。**只有落表的那几种算数**，其余（工具结果、终态）不改阶段：
    工具结果是「刚才那件事的结果」而不是「现在在做什么」。"""
    if event_type is EventType.AGENT_THINKING:
        return AgentStage.THINKING
    if event_type is EventType.AGENT_TOOL_CALL:
        return AgentStage.TOOL
    if event_type is EventType.DELTA:
        return AgentStage.WRITING
    return None


async def _persist_event(event: RunEvent) -> None:
    """事件的落库回放源。与流水线那份同构，只是表不同。"""
    with suppress(Exception):
        async with session_scope() as session:
            session.add(
                AgentEvent(
                    session_id=event.run_id,
                    seq=event.seq,
                    type=event.type.value,
                    message=event.message,
                    data=event.data or {},
                )
            )


async def _persist_turn(session_id: str, turn: int, result: TurnResult) -> None:
    """把这一轮新增的消息按序落库，并更新会话行。

    `result.messages` 的顺序**就是**喂给模型的顺序，一条都不能省：
    带 tool_calls 的 assistant 与它的 `role="tool"` 回复必须成对出现，
    下一轮重发时才不会被服务端拒掉。
    """
    async with session_scope() as session:
        seq = await _last_message_seq(session, session_id)
        for message in result.messages:
            seq += 1
            session.add(
                AgentMessage(
                    session_id=session_id,
                    seq=seq,
                    turn=turn,
                    role=message.role,
                    content=message.content or "",
                    tool_calls=[
                        {"id": c.id, "name": c.name, "arguments": c.arguments}
                        for c in message.tool_calls
                    ],
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                )
            )

        row = await session.get(AgentSession, session_id)
        if row is None:
            return
        row.rounds += int(result.rounds)
        row.tool_calls += int(result.tool_calls)
        row.stage = None
        row.finished_at = datetime.now(UTC)
        if result.stopped == "final":
            row.status = AgentStatus.IDLE.value
            # 只有交稿才动 story。取消/出错时保住上一版——**不产半截稿**，
            # 也不该因为一次失败把用户已经看过的那一版擦掉。
            if result.text:
                row.story = result.text
        elif result.stopped == "cancelled":
            row.status = AgentStatus.CANCELLED.value
            row.error = None
        else:
            row.status = AgentStatus.FAILED.value
            row.error = result.error or "这一轮没有完成。"


async def _finish(session_id: str, turn: int, *, stopped: str, error: str) -> None:
    """不经过循环的收尾（任务崩溃、消息序列坏掉、就地的取消）。"""
    async with session_scope() as session:
        row = await session.get(AgentSession, session_id)
        if row is None:
            return
        row.stage = None
        row.finished_at = datetime.now(UTC)
        row.status = (
            AgentStatus.CANCELLED.value if stopped == "cancelled" else AgentStatus.FAILED.value
        )
        row.error = error or None
    log.warning("agent_turn_finished_early", session_id=session_id, stopped=stopped, error=error)
    await _emit_terminal(session_id, turn=turn, stopped=stopped, message=error)


async def _close_session(session_id: str, *, turn: int, status: AgentStatus) -> None:
    """把一条**没有活跃任务**的会话就地收成终态。先写库、再发事件。"""
    async with session_scope() as session:
        row = await session.get(AgentSession, session_id)
        if row is None or AgentStatus(row.status).is_terminal:
            return
        row.status = status.value
        row.stage = None
        row.finished_at = datetime.now(UTC)
        row.cancel_requested = False

    await _emit_terminal(session_id, turn=turn, stopped="cancelled", message="已经停止。")


async def _emit_terminal_event(emitter: RunEmitter, result: TurnResult, *, turn: int) -> None:
    """一轮的终态事件。**三种结局在这里各发一条，循环一条都不发。**

    错误文案与 `_persist_turn` 写进那一列的是同一句——界面上「红卡片」和
    「快照里的 error」说的必须是同一件事。
    """
    if result.stopped == "final":
        await emitter.emit(
            EventType.AGENT_COMPLETED,
            data=ev.completed(
                turn=turn,
                rounds=result.rounds,
                tool_calls=result.tool_calls,
                chars=len(result.text),
            ),
        )
    elif result.stopped == "cancelled":
        await emitter.emit(EventType.AGENT_CANCELLED, data=ev.cancelled(turn=turn))
    else:
        message = result.error or "这一轮没有完成。"
        await emitter.emit(EventType.ERROR, message=message, data=ev.error(message=message))


async def _emit_terminal(session_id: str, *, turn: int, stopped: str, message: str) -> None:
    """补发一条终态事件，总线与 PG 都要落到。

    总线让**当前挂着的**流立刻收到并关闭；PG 让**刷新后重连**的流回放得到它。
    缺任何一半，都有一半的时间线要卡到心跳兜底，而兜底发的是 error 帧——
    会把一次主动取消显示成「生成失败」。
    """
    with suppress(Exception):
        async with session_scope() as session:
            seq = await _last_event_seq(session, session_id)
        bus = await get_bus(ev.NAMESPACE)
        emitter = RunEmitter(bus, session_id, start_seq=seq, on_emit=_persist_event)
        if stopped == "cancelled":
            await emitter.emit(EventType.AGENT_CANCELLED, data=ev.cancelled(turn=turn))
        else:
            await emitter.emit(
                EventType.ERROR, message=message, data=ev.error(message=message)
            )


# ----------------------------------------------------------------------
# 内部：读写与转换
# ----------------------------------------------------------------------


async def _messages_in(session: Any, session_id: str) -> list[AgentMessage]:
    rows = (
        await session.execute(
            select(AgentMessage)
            .where(AgentMessage.session_id == session_id)
            .order_by(AgentMessage.seq)
        )
    ).scalars()
    return list(rows)


async def _last_message_seq(session: Any, session_id: str) -> int:
    value = (
        await session.execute(
            select(func.max(AgentMessage.seq)).where(AgentMessage.session_id == session_id)
        )
    ).scalar_one_or_none()
    return int(value or 0)


async def _last_event_seq(session: Any, session_id: str) -> int:
    value = (
        await session.execute(
            select(func.max(AgentEvent.seq)).where(AgentEvent.session_id == session_id)
        )
    ).scalar_one_or_none()
    return int(value or 0)


def _to_chat_message(row: AgentMessage) -> ChatMessage:
    """库里的行 → 喂给模型的消息。**逐字**，包括工具调用的原文。"""
    return ChatMessage(
        role=row.role,  # type: ignore[arg-type]
        content=row.content or "",
        tool_calls=[
            ToolCall(
                id=str(item.get("id") or ""),
                name=str(item.get("name") or ""),
                arguments=str(item.get("arguments") or "{}"),
            )
            for item in (row.tool_calls or [])
        ],
        tool_call_id=row.tool_call_id,
        name=row.name,
    )


def _title_of(input_text: str) -> str:
    """标题只用来让人认出这是哪一段。取第一行开头，不做任何美化。"""
    first = " ".join(input_text.strip().splitlines()[:1]).strip()
    return first[:TITLE_CHARS] + ("…" if len(first) > TITLE_CHARS else "")


def model_name() -> str:
    """当前模型名，只用于界面显示。

    **失败时返回空串而不是抛。** 建会话这个动作本身不调模型；密钥没配的话，
    该报错的地方是「开始写」那一刻（那里会变成一张红卡片，写着缺什么），
    而不是让一个与模型无关的 POST 直接 500。
    """
    with suppress(Exception):
        return str(getattr(get_chat_model(), "name", ""))
    return ""


__all__ = [
    "INTERRUPTED_MESSAGE",
    "MAX_CONCURRENT_TURNS",
    "AgentNotFound",
    "AgentRejected",
    "SessionDetail",
    "active_task_count",
    "create_session",
    "execute_turn",
    "get_session",
    "load_events",
    "model_name",
    "reap_stale_agent_sessions",
    "request_cancel",
    "send_message",
    "session_detail",
    "session_status",
    "shutdown_agent_tasks",
]
