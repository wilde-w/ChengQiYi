"""故事工坊的接口：开会话、接着说一句、停、看过程。

**这个模块不进流水线。** `agent_session` 与 `analysis_run` 之间没有外键、没有
共享状态、不占 `NodeKey`、不出现在进度带上。「根据一段评论写个故事」与「分析
这条视频」是两件事，合并成一件之后每个问题都会变成两个问题的叠加：进度该按谁
的算、取消该停哪个、刷新该恢复谁。

**SSE 与流水线共用 `sse_stream.py` 的帧机器**（回放合并、心跳、断开检测、
终态收尾），但换命名空间（`agent`）、换兜底表（`agent_event`）。一个连接仍然
只服务一个会话——跨会话复用会让 `since` 变成每个会话一个游标，而前端只有一个。

**待命的会话也要挂着连接**（下面 `_event_stream` 里的 `terminal` 分支只管
已中断/已停止）。这不是疏忽：前端只有「收到终态帧」和「连接断了」两种结束
信号，而后一种会触发重连——对着一个待命的会话关掉连接，等于让页面每 5 秒重连
一次，永远如此。留着这条连接，下一轮的事件就直接从它上面流过来。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.agent import events as ev
from app.api.v1 import sse_stream
from app.config import get_settings
from app.constants import AgentStage, AgentStatus, Library
from app.graph.bus import get_bus
from app.logging_conf import get_logger
from app.novel import service as novel_service
from app.prompts import story as story_prompt
from app.schemas.agent import (
    AgentCapabilitiesOut,
    AgentSessionDetail,
    AgentSessionOut,
    CreateAgentSessionRequest,
    SendAgentMessageRequest,
)
from app.services import agent_service

log = get_logger(__name__)
router = APIRouter(tags=["agent"])


# ----------------------------------------------------------------------
# 会话
# ----------------------------------------------------------------------


@router.post("/agent/sessions", response_model=AgentSessionDetail, status_code=201)
async def create_agent_session(payload: CreateAgentSessionRequest) -> Any:
    """开一次写故事，并把第一轮排上。

    材料在这里就校验（空串、超长），而不是等第一次生成时才失败：那时用户已经
    等了十几秒，而错误其实一开就能看出来。

    **第一句话由后端说**（`payload.instruction` 为空时用
    `story_prompt.FIRST_INSTRUCTION`）。曾经把它留给调用方，结果是按了「开始写」
    之后会话静静地停在 idle——前端把它接成了一次 `POST /messages` 才补上，
    但那条路要求前端手里有一份开场白，也就是在第二处维护同一句提示词。默认值
    放回后端，两条路就只剩一条：建会话 = 开始写。

    返回的**不是**建会话那一刻的快照，而是排上第一轮之后的：`status=running`、
    `turn=1`。差别看着小，但前者会让面板在头几百毫秒里显示「待命」——输入框
    是能打的，「停止」是藏着的，而模型已经在跑了。第一轮的过程与正文仍然只从
    事件流里来（与流水线的 `POST /runs` 同一条规矩）。
    """
    created = await agent_service.create_session(
        input_text=payload.input,
        allow_novel=payload.allow_novel,
        target_chars=payload.target_chars,
    )
    row = await agent_service.send_message(
        created.id,
        payload.instruction or story_prompt.FIRST_INSTRUCTION,
    )
    return _detail(row, messages=[])


@router.get("/agent/sessions/{session_id}", response_model=AgentSessionDetail)
async def get_agent_session(session_id: str) -> Any:
    """会话 + 原文 + 全部对话。**刷新页面靠它重建界面。**

    流负责实时性，这个接口负责真实性：正文、轮次、工具调用次数都以它为准。
    事件流里的 `delta` 只是把正文「写」出来给人看，一个字都不进这里。
    """
    try:
        detail = await agent_service.session_detail(session_id)
    except agent_service.AgentNotFound as exc:
        raise HTTPException(status_code=404, detail="会话不存在") from exc
    return _detail(detail.session, messages=detail.messages)


@router.get("/agent/capabilities", response_model=AgentCapabilitiesOut)
async def agent_capabilities() -> Any:
    """面板能做什么。

    **不探 MCP。** 探一次要起子进程、等两秒，而这是打开面板就会调的接口——
    把首屏卡在一个与本机配置无关的等待上，是最没必要的一处等待。这里只做配置层
    自检（`prerequisites()` 只查文件与开关），真连不上时那一轮工具调用会失败，
    界面上是一张红卡片——**那是特性**，它演示了 agent 处理失败工具的能力。
    """
    settings = get_settings()
    return {
        "allow_novel": bool(settings.NOVEL_MCP_ENABLED),
        "novel_hint": _novel_hint(),
        "model": agent_service.model_name(),
        "demo": settings.is_demo,
        "libraries": [{"value": lib.value, "label": lib.label} for lib in Library],
    }


# ----------------------------------------------------------------------
# 一轮
# ----------------------------------------------------------------------


@router.post(
    "/agent/sessions/{session_id}/messages", response_model=AgentSessionOut, status_code=202
)
async def send_agent_message(session_id: str, payload: SendAgentMessageRequest) -> Any:
    """接着说一句（或改一稿），并立刻把这一轮排上。

    202 而不是 200：返回的是「已收下」，正文要等事件流里那串 `delta` 敲完。
    这一轮跑完之前，同一个会话再发是 409——不是拒绝服务，是循环里那条
    「每个 tool_call 必须被一对一应答」的协议不允许两条消息挤进同一轮。
    """
    try:
        row = await agent_service.send_message(session_id, payload.text)
    except agent_service.AgentNotFound as exc:
        raise HTTPException(status_code=404, detail="会话不存在") from exc
    except agent_service.AgentRejected as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _session(row)


@router.post("/agent/sessions/{session_id}/cancel", status_code=202)
async def cancel_agent_session(session_id: str) -> dict[str, Any]:
    """请求停止。**置标志位而非 kill 任务**，让循环在轮边界收尾。

    立刻强杀会跳过落库与终态事件，前端就挂在一条永远不结束的流上。没有活跃
    任务时（进程重启留下的僵尸）就地收尾——那个标志位永远不会有谁来读。
    """
    try:
        accepted = await agent_service.request_cancel(session_id)
    except agent_service.AgentNotFound as exc:
        raise HTTPException(status_code=404, detail="会话不存在") from exc
    return {
        "accepted": accepted,
        "message": "已请求停止，这一轮会在边界上收尾" if accepted else "现在没有在写",
    }


# ----------------------------------------------------------------------
# 事件流
# ----------------------------------------------------------------------


@router.get("/agent/sessions/{session_id}/events")
async def stream_agent_events(
    session_id: str,
    request: Request,
    since: int = Query(default=0, ge=0, description="已收到的最大 seq，回放起点（不含）"),
) -> StreamingResponse:
    status = await agent_service.session_status(session_id)
    if status is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    return StreamingResponse(
        _event_stream(session_id, since=since, status=status, request=request),
        media_type="text/event-stream",
        headers=sse_stream.SSE_HEADERS,
    )


async def _event_stream(
    session_id: str, *, since: int, status: str, request: Request
) -> AsyncIterator[str]:
    """帧的生产者。结构同流水线那条：先补历史，再决定是订阅还是收工。"""
    heartbeat = max(5, get_settings().SSE_HEARTBEAT_SECONDS)
    terminal = AgentStatus(status).is_terminal
    last_seq = since

    try:
        bus = await get_bus(ev.NAMESPACE)
        replayed = await sse_stream.replay(
            bus, session_id, since, load_pg=agent_service.load_events
        )
        for event in replayed:
            last_seq = max(last_seq, event.seq)
            yield event.to_sse()

        if terminal:
            # 与流水线同一条规矩：**必须以终态帧收尾，不能只是关掉连接。**
            # 关连接对前端是歧义的——它分不清「写完了」和「连接断了」，
            # 前者要拉快照对账，后者要重连。不分清的代价是页面永久停在
            # 「写作中」。会话是被 `reap_stale_agent_sessions` 收掉的
            # （进程被杀，事件表里没有终态帧）时，这里补一条。
            #
            # 这里问的是「这个客户端**已经收到过一个结局**了吗」，所以用按轮的
            # 那个 `TERMINAL`：`agent_completed` 同样是前端认得的一种结局
            # （它到了，前端就收工、拉快照），收到过就不该再补一条错误帧。
            # 下面递给 `tail` 的是「这条流什么时候可以收工」，那要的是会话级。
            if not any(event.type in ev.TERMINAL for event in replayed):
                yield sse_stream.error_frame(
                    "stream_ended_without_terminal_event",
                    f"会话已经{AgentStatus(status).label}，但没有收到结束事件。请拉取快照。",
                    seq=last_seq + 1,
                    run_id=session_id,
                )
            else:
                yield sse_stream.comment(f"end replay seq={last_seq}")
            return

        # 起点用 last_seq 而非 since：回放里最后一条的 seq 才是真正的断点。
        source = sse_stream.subscribe(
            bus,
            session_id,
            last_seq if last_seq else since,
            load_pg=agent_service.load_events,
            # **`ev.SESSION_TERMINAL`，不是 `ev.TERMINAL`。** 这里传的是「收到
            # 哪几种事件就可以收工了」，而 `TERMINAL` 是**按轮**的：里头的
            # `agent_completed` 只说明这一稿写完了，会话还在待命（用户随时会说
            # 「再暗一点」）。照轮关流的话，第二轮的事件会被送进一条已经断掉的
            # 连接——前端不报错、面板永远停在上一版正文（`useAgentStream` 那边
            # 是照着「连接会一直开着」写的，`onSend` 因此只发 POST 不重开流）。
            #
            # 也不能传空集合：`tail` 的心跳分支会「先抽干队列、再用同一个集合
            # 判断这一帧算不算终态」，空集合会让它在转发完 `agent_cancelled`
            # 之后**再补一条合成的 error 帧**，把一次正常取消显示成「出错」。
            #
            # 会话真的结束了由 `_probe` 兜底：它本来就是**会话级**的，每次心跳
            # 查一遍，最慢一个心跳周期就把流收掉。事件表里缺终态帧的情形
            # （进程被杀、`reap_stale_agent_sessions` 收的僵尸）也是它管的。
            terminal=ev.SESSION_TERMINAL,
        )
        async for frame in sse_stream.tail(
            source,
            heartbeat=heartbeat,
            request=request,
            status_of=lambda: _probe(session_id),
            run_id=session_id,
            start_seq=last_seq,
            terminal=ev.SESSION_TERMINAL,
        ):
            yield frame
    except Exception as exc:
        # 客户端断开（关面板、刷新）以 CancelledError 的形式进来，它不是
        # Exception 的子类（3.8+），所以不会走到这里，无需记录。
        log.exception("agent_sse_stream_failed", session_id=session_id, since=since)
        # 连接已经建立，改不了 HTTP 状态码，只能用事件把失败告诉前端。
        yield sse_stream.error_frame(
            "stream_failed", f"{type(exc).__name__}: {exc}", seq=last_seq + 1, run_id=session_id
        )


async def _probe(session_id: str) -> sse_stream.Probe | None:
    status = await agent_service.session_status(session_id)
    if status is None:
        return None
    state = AgentStatus(status)
    # label 只进那句「状态已是 …，但没有收到结束事件」——加引号是为了让
    # 「状态已是已中断」这种人机词粘连读起来还是一句话。
    return sse_stream.Probe(f"「{state.label}」", state.is_terminal)


def _novel_hint() -> str | None:
    """古典文学 MCP 为什么起不来。`None` 表示看起来没问题。"""
    try:
        return novel_service.get_client().prerequisites()
    except Exception as exc:
        return f"古典文学 MCP 配置读取失败：{exc}"


# ----------------------------------------------------------------------
# ORM → DTO
# ----------------------------------------------------------------------


def _detail(row: Any, *, messages: list[Any]) -> dict[str, Any]:
    return {
        **_session(row),
        "input_text": row.input_text,
        "messages": [_message(m) for m in messages],
        "demo": get_settings().is_demo,
    }


def _session(row: Any) -> dict[str, Any]:
    status = AgentStatus(row.status)
    return {
        "id": row.id,
        "title": row.title or "",
        "input_chars": int(row.input_chars or 0),
        "status": row.status,
        # 落库的是机器词（preparing/thinking/…），出去的是给人看的那句。
        # 前端因此不必维护一张状态映射表——那种表的症状是「改了后端文案，
        # 界面上还是旧说法」。
        "stage": AgentStage(row.stage).label if row.stage else None,
        "turn": int(row.turn or 0),
        "story": row.story,
        "rounds": int(row.rounds or 0),
        "tool_calls": int(row.tool_calls or 0),
        "model": row.model,
        "providers": row.providers or {},
        "options": row.options or {},
        "error": row.error,
        # 终态集合只在这里判一次。让前端拿 status 字符串自己推的话，
        # 以后加一个状态就要两个仓库各改一次，而漏改的表现是「永远在转圈」。
        "is_running": status is AgentStatus.RUNNING,
        "is_terminal": status.is_terminal,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def _message(row: Any) -> dict[str, Any]:
    return {
        "seq": int(row.seq),
        "turn": int(row.turn or 0),
        "role": row.role,
        "content": row.content or "",
        # 参数是**原始 JSON 文本**，不是解析后的对象：坏参数时界面上要能看到
        # 模型到底写了什么，而那是排查「它为什么老失败」的唯一线索。
        "tool_calls": [
            {
                "id": str(item.get("id") or ""),
                "name": str(item.get("name") or ""),
                "arguments": str(item.get("arguments") or ""),
            }
            for item in (row.tool_calls or [])
        ],
        "tool_call_id": row.tool_call_id,
        "name": row.name,
    }
