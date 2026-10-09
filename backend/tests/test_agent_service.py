"""故事工坊的会话服务：落库顺序、seq、并发门、僵尸与取消。

服务层是「模型 / 工具 / 数据库」三者的接缝，这里守的四条都属于**接缝上的错**：
循环的单测看不到（它不碰库），工具的测试也看不到（它不经过服务）。其中代价最高
的是「库里那一串 == 喂给模型的那一串」——两边只要差一个字符，症状就是「第三轮
突然被服务端 400 拒掉」，而现场在两轮之前。

需要 PG（`docker compose up -d`）。起不来就**跳过并说明原因**——在一种
「其实什么都没测」的状态下变绿，比红更糟。

模型路径走 `MOCK_MODE=always`（conftest 钉死）：这里的 mock 不是退而求其次，
它按轮次脚本化地真调工具，所以「一轮跑完」这件事本身就是被验证的。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

import pytest
from sqlalchemy import delete, text

from app.agent import events as ev
from app.agent.loop import TurnResult
from app.constants import AgentStage, AgentStatus, EventType
from app.db.session import session_scope
from app.graph.bus import get_bus
from app.models.agent import AgentSession
from app.providers.base import ChatMessage
from app.services import agent_service as svc
from tests.test_agent_loop import assert_balanced

SAMPLE = (
    "妈妈走了以后我才发现，她留下的那件毛衣我一直没舍得洗。"
    "想她的时候我就抱着坐一会儿，好像这样她就还在。"
    "后来我把它叠好收进柜子，可每次开柜子都要先看一眼。"
)


@pytest.fixture
async def made() -> AsyncIterator[list[str]]:
    """地基：库连不上就跳过；跑完按 session_id 收回（子表靠 FK 级联）。"""
    try:
        async with session_scope() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"需要 docker compose 起的 PG：{type(exc).__name__}: {exc}")

    # 连得上却没有表是**另一回事**：那是忘了 `init-db`，不该悄悄跳过。
    # 跳过在这里格外危险——八条用例全绿，而一行代码都没跑过。
    async with session_scope() as session:
        exists = (await session.execute(text("SELECT to_regclass('agent_session')"))).scalar()
    assert exists, "缺 agent_session 表，先跑 `python -m app.cli init-db`"

    ids: list[str] = []
    yield ids

    # 先收任务再删数据：反过来会有个任务在删完之后继续往库里写。
    await svc.shutdown_agent_tasks()
    if ids:
        async with session_scope() as session:
            await session.execute(delete(AgentSession).where(AgentSession.id.in_(ids)))
        with suppress(Exception):
            bus = await get_bus(ev.NAMESPACE)
            for session_id in ids:
                await bus.clear(session_id)


async def _new(made: list[str], text: str = SAMPLE) -> Any:
    row = await svc.create_session(input_text=text)
    made.append(row.id)
    return row


async def _wait_done(session_id: str, timeout: float = 60.0) -> str:
    """轮询到不转圈为止。**刻意不用内部 task 对象**——那正是前端要走的路。"""
    now = asyncio.get_running_loop().time
    deadline = now() + timeout
    while now() < deadline:
        status = await svc.session_status(session_id)
        if status != AgentStatus.RUNNING.value:
            return str(status)
        await asyncio.sleep(0.05)
    raise AssertionError(f"会话 {session_id} 在 {timeout}s 内没有跑完")


# ----------------------------------------------------------------------
# 一轮跑完的样子
# ----------------------------------------------------------------------


async def test_一轮跑完_消息与事件都从1起且无缺口(made: list[str]) -> None:
    row = await _new(made)
    assert row.status == AgentStatus.IDLE.value  # 开会话不调模型

    await svc.send_message(row.id, "请根据这段材料写一个故事。")
    assert await _wait_done(row.id) == AgentStatus.IDLE.value

    detail = await svc.session_detail(row.id)
    assert detail.messages[0].role == "user"
    assert detail.session.story, "交稿了却没有正文"
    # 首次生成至少要自己查两次资料——这是「它是 agent 而不是模板」的可核查证据。
    assert detail.session.tool_calls >= 2, detail.session.tool_calls
    assert detail.session.rounds >= 2, detail.session.rounds
    assert detail.session.stage is None, "跑完了还留着「正在…」"

    # 消息 seq：从 1 起、严格递增、无重复
    mseq = [m.seq for m in detail.messages]
    assert mseq == list(range(1, len(mseq) + 1))

    events = await svc.load_events(row.id, 0)
    kind = [e.type for e in events]
    assert events[0].seq == 1
    assert events[0].type is EventType.AGENT_STARTED
    assert kind[-1] is EventType.AGENT_COMPLETED
    assert EventType.AGENT_TOOL_CALL in kind, "过程里没有工具卡片"
    assert EventType.AGENT_TOOL_RESULT in kind
    assert events == sorted(events, key=lambda e: e.seq)
    assert [e.seq for e in events] == list(range(1, len(events) + 1))


async def test_每个工具调用都有配对的应答且内容不为空(made: list[str]) -> None:
    """配平不变量跑在**库里那一串**上。

    循环里的配平（`test_agent_loop.py`）验的是它自己刚拼出来的消息；
    这里验的是绕了一圈 JSON 列之后还是不是一个样——少一条 tool 回复、
    或者 `tool_call_id` 在往返里丢了，下一轮就是 400，而现场在两轮之前。
    """
    row = await _new(made)
    await svc.send_message(row.id, "请根据这段材料写一个故事。")
    await _wait_done(row.id)

    messages = (await svc.session_detail(row.id)).messages
    # 这里**刻意**调私有的 `_to_chat_message`：要验的就是「这一串库里的行，
    # 经下一轮真正会用的那个转换之后，是不是还配平」，换一条路径就没有意义了。
    assert_balanced([svc._to_chat_message(m) for m in messages])

    tools = [m for m in messages if m.role == "tool"]
    assert tools, "mock 第一轮就该调工具"
    for m in tools:
        assert m.content, f"{m.name} 的结果是空的——卡片上会是一片空白"
        assert m.name and m.tool_call_id


# ----------------------------------------------------------------------
# 接缝：库里那一串就是喂给模型的那一串
# ----------------------------------------------------------------------


async def test_第二轮把第一轮的消息原样递给模型(
    made: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    real = svc.run_turn

    async def spy(req: Any, **kw: Any) -> Any:
        seen.append(req)
        return await real(req, **kw)

    monkeypatch.setattr(svc, "run_turn", spy)

    row = await _new(made)
    await svc.send_message(row.id, "请根据这段材料写一个故事。")
    await _wait_done(row.id)
    detail = await svc.session_detail(row.id)
    first_turn = detail.messages
    first_story = detail.session.story

    await svc.send_message(row.id, "再暗一点。")
    await _wait_done(row.id)
    assert len(seen) == 2

    history = list(seen[1].history)
    assert len(history) == len(first_turn), "上一轮的消息没有原样带上来"
    for got, db in zip(history, first_turn, strict=True):
        assert got.role == db.role
        assert got.content == (db.content or "")
        assert got.tool_call_id == db.tool_call_id
        assert got.name == db.name
        assert [(c.id, c.name, c.arguments or "{}") for c in got.tool_calls] == [
            (c.get("id"), c.get("name"), c.get("arguments") or "{}")
            for c in (db.tool_calls or [])
        ], "工具调用的原文在往返里被改过了"

    # 收尾那条是这一轮的新的 user 消息，不在 history 里
    assert seen[1].user_message == "再暗一点。"
    assert seen[1].turn == 2

    rev = await svc.session_detail(row.id)
    assert rev.session.turn == 2
    assert rev.session.story != first_story, "改稿轮的新版与上一版一模一样"
    assert len(rev.messages) > len(first_turn), "第二轮的对话没有被加到后面"


async def test_改稿轮也留下工具卡片(made: list[str]) -> None:
    """「能连续改稿」的可核查证据：第二轮的对话里也有 tool 往返。"""
    row = await _new(made)
    await svc.send_message(row.id, "请根据这段材料写一个故事。")
    await _wait_done(row.id)
    first = (await svc.session_detail(row.id)).messages

    await svc.send_message(row.id, "短一些，换成第二人称。")
    await _wait_done(row.id)
    second = (await svc.session_detail(row.id)).messages

    new = second[len(first) :]
    assert [m.role for m in new].count("tool") >= 1
    assert new[-1].role == "assistant"
    assert new[-1].content.strip()


# ----------------------------------------------------------------------
# 状态机
# ----------------------------------------------------------------------


async def test_上一轮没写完时第二条消息被拒(
    made: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """卡住 `run_turn`，让「正在写」这个窗口是确定的，而不是抢出来的。"""
    gate = asyncio.Event()
    real = svc.run_turn

    async def blocked(req: Any, **kw: Any) -> Any:
        await gate.wait()
        return await real(req, **kw)

    monkeypatch.setattr(svc, "run_turn", blocked)

    row = await _new(made)
    await svc.send_message(row.id, "第一句")
    assert await svc.session_status(row.id) == AgentStatus.RUNNING.value

    with pytest.raises(svc.AgentRejected) as err:
        await svc.send_message(row.id, "第二句")
    assert "还在写" in str(err.value)

    # 被拒的那句一个字都没落库，轮次也没动
    detail = await svc.session_detail(row.id)
    assert [m.content for m in detail.messages] == ["第一句"]
    assert detail.session.turn == 1

    gate.set()
    assert await _wait_done(row.id) == AgentStatus.IDLE.value


async def test_终态会话不能再被喂消息(made: list[str]) -> None:
    row = await _new(made)
    async with session_scope() as session:
        db_row = await session.get(AgentSession, row.id)
        db_row.status = AgentStatus.CANCELLED.value

    with pytest.raises(svc.AgentRejected) as err:
        await svc.send_message(row.id, "接着写")
    assert "已停止" in str(err.value)


async def test_reap_把进程留下的僵尸收成已中断(made: list[str]) -> None:
    """进程被杀不会写终态。不收的话，那一行永远停在 running，前端转圈到天荒地老。"""
    row = await _new(made)
    async with session_scope() as session:
        db_row = await session.get(AgentSession, row.id)
        db_row.status = AgentStatus.RUNNING.value
        db_row.stage = AgentStage.TOOL.value

    assert await svc.reap_stale_agent_sessions() >= 1

    detail = await svc.session_detail(row.id)
    assert detail.session.status == AgentStatus.FAILED.value
    assert detail.session.error == svc.INTERRUPTED_MESSAGE
    assert detail.session.stage is None
    assert detail.session.finished_at is not None

    # 终态必须落进事件表：刷新后重连的那条流靠回放它才知道已经结束
    events = await svc.load_events(row.id, 0)
    assert events[-1].type is EventType.ERROR


@pytest.mark.parametrize(
    "stopped,event,status",
    [
        ("final", EventType.AGENT_COMPLETED, "idle"),
        ("cancelled", EventType.AGENT_CANCELLED, "cancelled"),
        ("error", EventType.ERROR, "failed"),
    ],
)
async def test_终态事件发出时库里已经是终态(
    made: list[str],
    monkeypatch: pytest.MonkeyPatch,
    stopped: str,
    event: EventType,
    status: str,
) -> None:
    """**先落库、再发事件。** 前端收到终态的第一件事是拉快照对账，
    事件早于落库它就会看到「流说结束了、快照说还在写」——真机上按停止时
    正是这一幕（循环那时候自己发终态，抢在了落库前面）。

    三种结局各查一遍：它们在循环里是三条不同的出口，在服务里是同一条。
    """
    seen: list[tuple[str | None, str]] = []
    real = svc._emit_terminal_event
    box: dict[str, str] = {}

    async def spy(emitter: Any, result: Any, *, turn: int) -> None:
        seen.append((await svc.session_status(box["sid"]), str(event)))
        await real(emitter, result, turn=turn)

    async def fake_turn(req: Any, **kw: Any) -> Any:
        return TurnResult(
            messages=[ChatMessage("assistant", "稿子" if stopped == "final" else "")],
            text="稿子" if stopped == "final" else "",
            rounds=1,
            tool_calls=0,
            stopped=stopped,
            error="" if stopped == "final" else "炸了",
        )

    monkeypatch.setattr(svc, "_emit_terminal_event", spy)
    monkeypatch.setattr(svc, "run_turn", fake_turn)

    row = await _new(made)
    box["sid"] = row.id
    await svc.send_message(row.id, "写一个")
    assert await _wait_done(row.id) == status
    assert seen == [(status, str(event))], seen


async def test_没有活跃任务时_cancel_就地收尾(made: list[str]) -> None:
    row = await _new(made)
    assert await svc.request_cancel(row.id) is False, "待命的会话没什么可停的"

    async with session_scope() as session:
        db_row = await session.get(AgentSession, row.id)
        db_row.status = AgentStatus.RUNNING.value

    assert await svc.request_cancel(row.id) is True
    detail = await svc.session_detail(row.id)
    assert detail.session.status == AgentStatus.CANCELLED.value

    events = await svc.load_events(row.id, 0)
    assert events[-1].type is EventType.AGENT_CANCELLED


# ----------------------------------------------------------------------
# 建会话就得开始写
# ----------------------------------------------------------------------


async def test_建会话就把第一轮排上(made: list[str]) -> None:
    """「开始写」按下去之后，会话**不能**停在待命。

    这一条钉的是一个真出现过的缺口：`create_session` 只落一行、不动模型，而
    把开场白留给调用方的那条路上，前端只调了建会话——于是模型一次都没跑，
    面板上「第 0 稿 · 0 轮 · 0 次工具调用」静静地挂着，像在等人，而按钮已经
    按过了。所以现在开场白由路由补上，建会话 = 开始写。

    测在路由这一层而不是服务层：缺的正是**这两步的编排**，服务层各自都是对的。
    """
    from app.api.v1.agent import create_agent_session
    from app.prompts.story import FIRST_INSTRUCTION
    from app.schemas.agent import CreateAgentSessionRequest

    # 路由返回的是 DTO（dict），不是 ORM 行
    created = await create_agent_session(CreateAgentSessionRequest(input=SAMPLE))
    made.append(created["id"])
    assert created["status"] == AgentStatus.RUNNING.value
    assert created["turn"] == 1

    await _wait_done(created["id"])
    stored = await svc.session_detail(created["id"])
    assert stored.messages[0].role == "user"
    assert stored.messages[0].content == FIRST_INSTRUCTION
    assert stored.session.story, "跑完一轮还应当有正文"
