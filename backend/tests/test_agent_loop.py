"""故事工坊的循环：上限、配平、失败吸收。

这些用例守的是**真模型才会犯的错**，而它们在 MOCK_MODE 下一次都不会走到——
所以每一条都必须自己造出那个场景（脚本化的假模型），不能指望跑一遍 mock
就能覆盖。

最要紧的一条是**配平**：assistant 说了几次调用，就必须有几条 `role="tool"`
回复。少一条，下一轮请求被服务端 400 拒掉，而症状是「第三轮突然报错」。
这里的断言直接跑在「发给模型的那串消息」上（`model.seen[-1]`），而不是
跑在返回的 `TurnResult` 上——前者才是真的会撞上 400 的那一份。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from app.agent import loop
from app.agent.tools import ToolRegistry, ToolSpec
from app.constants import EventType
from app.providers.base import ChatMessage, ChatResult, ToolCall


@dataclass
class Scripted:
    """按轮次脚本化的假模型。

    `replies[i]` 是第 i+1 轮的回复（`calls` 为空表示交稿）。脚本用完之后
    一直用最后一条——「永远在调工具」这种用例因此不用写 6 条。
    """

    replies: list[dict[str, Any]]
    seen: list[list[ChatMessage]] = field(default_factory=list)
    tools_seen: list[list[str]] = field(default_factory=list)

    async def complete(
        self, messages, *, task=None, context=None, tools=None, **kw
    ) -> ChatResult:
        self.seen.append(list(messages))
        self.tools_seen.append([] if tools is None else [t["function"]["name"] for t in tools])
        script = self.replies[min(len(self.seen) - 1, len(self.replies) - 1)]
        calls = [
            ToolCall(id=f"c{len(self.seen)}_{j}", name=name, arguments=args)
            for j, (name, args) in enumerate(script.get("calls") or [])
        ]
        return ChatResult(
            text=script.get("text") or "",
            model="scripted",
            finish_reason="tool_calls" if calls else "stop",
            tool_calls=calls,
        )


def fake_registry(*, slow: bool = False) -> Any:
    """只依赖 `ToolRegistry` 本身，不碰 Qdrant / PG / MCP。"""

    async def ok(args: dict[str, Any]) -> str:
        return f"结果：{json.dumps(args, ensure_ascii=False)}"

    async def hits(args: dict[str, Any]) -> str:
        # 带《书名》——mock 的稿子会把工具结果里的书名写进正文，
        # 这正是「查过的东西真的进了稿子」的证据。
        return f"找到 1 条与「{args.get('query')}」相近的段落：\n1. 《饮水词》·纳兰性德·诗词"

    async def boom(_args: dict[str, Any]) -> str:
        raise RuntimeError("背后的服务炸了")

    async def sleepy(_args: dict[str, Any]) -> str:
        await asyncio.sleep(5)
        return "太慢了"

    def obj(*props: str) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {p: {"type": "string"} for p in props},
            "required": [props[0]] if props else [],
        }

    specs = [
        ToolSpec("kb_search", "查知识库", obj("query"), hits),
        ToolSpec("kb_graph", "查图谱", obj("mode"), ok),
        ToolSpec("kb_sources", "列来源", {"type": "object", "properties": {}}, ok),
        ToolSpec("kb_source_text", "读原文", obj("source_file"), ok),
        ToolSpec("read_input", "读材料", {"type": "object", "properties": {}}, ok),
        ToolSpec("novel_lookup", "查古籍", obj("action"), ok, serial=True, quota="mcp"),
    ]
    if slow:
        specs.append(
            ToolSpec("sleepy", "很慢", {"type": "object", "properties": {}}, sleepy, timeout=0.05)
        )
    specs.append(ToolSpec("boom", "会炸", {"type": "object", "properties": {}}, boom))
    return ToolRegistry(specs)


async def run(
    replies: list[dict[str, Any]],
    *,
    registry: Any = None,
    cancel: bool = False,
    cancel_after: int | None = None,
    slow: bool = False,
) -> tuple[loop.TurnResult, Scripted, list[tuple[str, dict[str, Any]]]]:
    model = Scripted(replies)
    events: list[tuple[str, dict[str, Any]]] = []
    checks = 0

    async def emit(kind: EventType, data: dict[str, Any]) -> None:
        events.append((str(kind), data))

    async def should_cancel() -> bool:
        nonlocal checks
        checks += 1
        if cancel_after is not None:
            return checks > cancel_after
        return cancel

    result = await loop.run_turn(
        loop.TurnRequest(
            system_prompt="你是故事工坊。",
            history=[],
            user_message="请根据材料写一段故事。",
            turn=1,
        ),
        model=model,
        registry=registry or fake_registry(slow=slow),
        emit=emit,
        should_cancel=should_cancel,
    )
    return result, model, events


def assert_balanced(messages: list[ChatMessage]) -> None:
    """配平不变量：每个 tool_call 恰好一条回复，且没有孤儿 tool 消息。"""
    pending: dict[str, str] = {}
    for m in messages:
        if m.role == "assistant":
            for c in m.tool_calls:
                pending[c.id] = c.name
        elif m.role == "tool":
            assert m.tool_call_id in pending, f"孤儿 tool 消息：{m.tool_call_id}"
            assert pending.pop(m.tool_call_id) == m.name, "回复挂到了别的调用上"
    assert not pending, f"没有回复的调用：{pending}"


def kinds(events: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [k for k, _ in events]


TERMINALS = {
    str(EventType.AGENT_COMPLETED),
    str(EventType.AGENT_CANCELLED),
    str(EventType.ERROR),
}


def assert_no_terminal(events: list[tuple[str, dict[str, Any]]]) -> None:
    """**循环一个终态事件都不发。**

    它发了就会抢在落库前面抵达前端，而前端收到终态的第一件事是拉快照对账——
    那一瞬看到的是「流说结束了、快照说还在写」（真机上按停止时实测到的那一幕）。
    三种终态一律由 `agent_service` 在 `_persist_turn` 之后补，见那边
    `_emit_terminal_event` 的注释。
    """
    assert not [k for k in kinds(events) if k in TERMINALS], kinds(events)


# ---------------------------------------------------------------- 配平


async def test_每个工具调用都有配对的回复():
    result, model, events = await run(
        [
            {"calls": [("kb_search", '{"query": "想念"}'), ("kb_graph", '{"mode": "emotion"}')]},
            {"text": "这是终稿。"},
        ]
    )

    assert result.stopped == "final"
    assert result.tool_calls == 2
    assert_balanced(model.seen[-1])
    assert_balanced(result.messages)
    assert kinds(events).count(str(EventType.AGENT_TOOL_CALL)) == 2
    assert kinds(events).count(str(EventType.AGENT_TOOL_RESULT)) == 2


async def test_发多了也要每条都回_但只执行上限内的():
    """一轮发 5 个、上限 4 个：4 个真执行，第 5 个回一段说明。

    **不能只回 4 条**——少一条，下一轮 400。
    """
    five = [("kb_search", '{"query": "a"}'), ("kb_graph", '{"mode": "b"}'),
            ("kb_sources", "{}"), ("kb_source_text", '{"source_file": "c.txt"}'),
            ("read_input", "{}")]
    result, model, _ = await run([{"calls": five}, {"text": "终稿。"}])

    assert result.tool_calls == loop.MAX_TOOL_CALLS_PER_ROUND  # 只有 4 次真执行
    assert_balanced(model.seen[-1])
    assert_balanced(result.messages)

    replies = [m for m in result.messages if m.role == "tool"]
    assert len(replies) == 5
    assert "一轮最多执行" in replies[-1].content
    assert replies[-1].name == "read_input"


async def test_未知工具名也要回一条():
    result, model, _ = await run(
        [{"calls": [("不存在的工具", "{}")]}, {"text": "终稿。"}]
    )

    reply = next(m for m in result.messages if m.role == "tool")
    assert "没有名为" in reply.content
    assert "当前可用的工具" in reply.content  # 失败信息里带可用清单
    assert_balanced(model.seen[-1])


async def test_工具抛异常转成文本不炸整轮():
    result, _, _ = await run([{"calls": [("boom", "{}")]}, {"text": "终稿。"}])

    reply = next(m for m in result.messages if m.role == "tool")
    assert "执行失败" in reply.content
    assert result.stopped == "final"


async def test_工具超时转成文本():
    result, _, _ = await run([{"calls": [("sleepy", "{}")]}, {"text": "终稿。"}], slow=True)

    reply = next(m for m in result.messages if m.role == "tool")
    assert "没有返回" in reply.content
    assert result.stopped == "final"


# ---------------------------------------------------------------- 上限


async def test_轮数用尽时最后一轮不给工具且带强制作答指令():
    result, model, _ = await run(
        [{"calls": [("kb_search", '{"query": "再查"}')], "text": "还在查"}]
    )

    assert result.rounds == loop.MAX_ROUNDS
    # 最后一轮的 tools 必须是空列表：模型看不见工具，才可能交稿。
    assert model.tools_seen[-1] == []
    assert model.tools_seen[0] != []
    assert model.seen[-1][-1].name == "force_final"
    assert_balanced(model.seen[-1])


async def test_收尾轮仍然调工具时不落库带_tool_calls_的助手消息():
    """那条消息会落库并在下一轮重发，而它永远等不到配对的回复——必须拆掉。"""
    call = {"calls": [("kb_search", '{"query": "x"}')], "text": "还在查。"}
    result, model, _ = await run([call, call])

    assert model.tools_seen[-1] == []
    # 最后一轮那句「还在查。」退化成纯文本交稿，tool_calls 已被拆掉。
    tail = [m for m in result.messages if m.role == "assistant"][-1]
    assert tail.tool_calls == []
    assert tail.content == "还在查。"
    assert result.text == "还在查。"
    assert_balanced(model.seen[-1])
    assert_balanced(result.messages)


async def test_坏参数连错两次后撤掉工具():
    bad = {"calls": [("kb_search", "query=想念，limit=3")]}  # 不是 JSON
    result, model, _ = await run([bad, bad, bad, {"text": "终稿。"}])

    replies = [m for m in result.messages if m.role == "tool"]
    assert sum("参数不合法" in r.content for r in replies) == loop.BAD_ARGS_LIMIT
    # 第三次调用时它已经不在了——loop 报「没有名为」，而不是再数一次坏参数。
    assert "没有名为" in replies[-1].content
    assert_balanced(model.seen[-1])


async def test_坏参数之后模型改用别的工具仍能交稿():
    result, _, _ = await run(
        [
            {"calls": [("kb_search", "{不是JSON")]},
            {"calls": [("kb_search", "还是不是")]},
            {"calls": [("kb_sources", "{}")]},
            {"text": "终稿。"},
        ]
    )
    assert result.stopped == "final"
    assert result.text == "终稿。"


async def test_mcp_额度用完后再调只回说明():
    calls = [("novel_lookup", '{"action": "books"}')] * 3
    result, _, _ = await run([{"calls": calls}, {"text": "终稿。"}])

    replies = [m for m in result.messages if m.role == "tool"]
    assert "额度已用完" in replies[-1].content
    assert result.tool_calls == loop.MAX_MCP_CALLS_PER_TURN


# ---------------------------------------------------------------- 收尾与中断


async def test_取消在轮边界退出且不产正文():
    result, _, events = await run(
        [{"calls": [("kb_search", '{"query": "x"}')]}, {"text": "不该出现"}], cancel=True
    )

    assert result.stopped == "cancelled"
    assert result.text == ""
    assert result.messages == []  # 第一轮就没往下走
    assert_no_terminal(events)
    assert not [k for k in kinds(events) if k == str(EventType.DELTA)]


async def test_第二轮才取消时保留过程消息但不产正文():
    """第一轮已经查了工具，用户在等第二轮时按了取消。

    过程消息要留着（面板上已经出现了那张卡片），但**不能有正文**——
    半截稿比没有稿更误导人。
    """
    result, _, events = await run(
        [{"calls": [("kb_search", '{"query": "x"}')]}, {"text": "不该出现"}], cancel_after=1
    )

    assert result.stopped == "cancelled"
    assert result.text == ""
    assert result.rounds == 2
    assert [m.role for m in result.messages] == ["assistant", "tool"]
    assert not [k for k in kinds(events) if k == str(EventType.DELTA)]
    assert_no_terminal(events)


async def test_交稿那一刻挂着的取消也算取消():
    """模型已经把正文写出来了，但这一刻取消已经挂着——按停止的人不该拿到新版本。

    这一条是接口走查时发现的洞：循环顶部的检查要**下一轮**才跑到，而交稿这一轮
    之后没有下一轮。漏掉它，真模型（正文是最长的一段等待）上「停止」基本无效。
    """
    result, _, events = await run([{"text": "不该出现"}], cancel_after=1)

    assert result.stopped == "cancelled"
    assert result.text == ""
    assert result.messages == []
    assert not [k for k in kinds(events) if k == str(EventType.DELTA)]
    assert_no_terminal(events)


async def test_空正文判为失败():
    result, _, events = await run([{"text": "   "}])

    assert result.stopped == "error"
    assert result.text == ""
    assert_no_terminal(events)
    assert "没有给出正文" in result.error


async def test_模型抛错时干净收尾():
    class Boom:
        async def complete(self, *a, **kw):
            raise RuntimeError("连接被重置")

    events: list[tuple[str, dict[str, Any]]] = []

    async def emit(kind, data):
        events.append((str(kind), data))

    result = await loop.run_turn(
        loop.TurnRequest(system_prompt="s", history=[], user_message="u", turn=1),
        model=Boom(),
        registry=fake_registry(),
        emit=emit,
    )
    assert result.stopped == "error"
    assert "连接被重置" in result.error
    assert_no_terminal(events)


# ---------------------------------------------------------------- 逐字


async def test_切块拼回去逐字等于正文():
    text = "第一句到此为止。第二句也到此为止，还带个逗号。第三句短。\n换行之后再来一句，凑够长度。"
    result, _, events = await run([{"text": text}])

    deltas = [d["text"] for k, d in events if k == str(EventType.DELTA)]
    assert "".join(deltas) == text
    assert deltas == [d["text"] for k, d in events if k == str(EventType.DELTA)]
    assert result.text == text


async def test_历史与新材料一起进上下文():
    model = Scripted([{"text": "终稿。"}])

    async def emit(*_a):
        return None

    history = [
        ChatMessage("user", "第一轮要求"),
        ChatMessage("assistant", "第一版"),
    ]
    await loop.run_turn(
        loop.TurnRequest(system_prompt="S", history=history, user_message="再暗一点", turn=2),
        model=model,
        registry=fake_registry(),
        emit=emit,
    )
    sent = model.seen[0]
    assert [m.role for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[-1].content == "再暗一点"


# ---------------------------------------------------------------- mock 脚本
#
# 默认运行模式是 MOCK_MODE=always，所以「演示路径」必须真跑通。这里是
# **真 loop × 真 mock 模型**，只把工具背后的服务换成假的——协议、轮次
# 计数、消息落库全是真的。


MATERIAL = (
    "妈妈走了以后我才发现，她留下的那件毛衣我一直没舍得洗。"
    "想她的时候我就抱着坐一会儿，可是每次想起来最后一次没好好陪她，就特别后悔。"
)


async def run_mock(
    *, input_text: str = MATERIAL, script: str = "default", history: list[ChatMessage] | None = None,
    user_message: str = "请根据材料写一段故事。", allow_novel: bool = True,
) -> tuple[loop.TurnResult, list[tuple[str, dict[str, Any]]]]:
    from app.providers.mock_llm import MockChatModel

    events: list[tuple[str, dict[str, Any]]] = []

    async def emit(kind: EventType, data: dict[str, Any]) -> None:
        events.append((str(kind), data))

    specs = [s for s in fake_registry()._specs.values() if allow_novel or s.name != "novel_lookup"]
    result = await loop.run_turn(
        loop.TurnRequest(
            system_prompt=f"你是故事工坊。原文：\n{input_text}",
            history=history or [],
            user_message=user_message,
            turn=1 if not history else 2,
        ),
        model=MockChatModel(latency_ms=0, agent_script=script),
        registry=ToolRegistry(specs),
        emit=emit,
    )
    return result, events


async def test_mock_三轮脚本_两次以上工具调用后交稿():
    result, events = await run_mock()

    assert result.stopped == "final"
    assert result.rounds == 3
    assert result.tool_calls == 3  # kb_search + kb_graph + novel_lookup
    calls = [d for k, d in events if k == str(EventType.AGENT_TOOL_CALL)]
    assert [c["name"] for c in calls] == ["kb_search", "kb_graph", "novel_lookup"]
    assert_balanced(result.messages)


async def test_mock_参数从输入派生():
    """换一段评论，检索词跟着变——这是「读了材料」的唯一证据。"""
    _sad, sad_events = await run_mock(input_text="妈妈走了以后我一直在想她，眼泪止不住。")
    _work, work_events = await run_mock(
        input_text="又是加班到凌晨，累得干不动了，领导还在催绩效，真不想上班。"
    )

    first = [d for k, d in sad_events if k == str(EventType.AGENT_TOOL_CALL)]
    second = [d for k, d in work_events if k == str(EventType.AGENT_TOOL_CALL)]

    assert first[0]["args"]["query"] != second[0]["args"]["query"]  # 检索词跟着材料变
    assert first[1]["args"] == {"mode": "emotion", "key": "哀伤", "limit": 3}
    assert second[1]["args"] == {"mode": "emotion", "key": "倦怠", "limit": 3}


async def test_mock_关掉古典文学时不调它():
    result, events = await run_mock(allow_novel=False)

    assert result.stopped == "final"
    assert result.rounds == 2
    names = [d["name"] for k, d in events if k == str(EventType.AGENT_TOOL_CALL)]
    assert "novel_lookup" not in names


async def test_mock_正文里出现查到的书名():
    result, _ = await run_mock()
    assert "《饮水词》" in result.text


async def test_mock_说再暗一点会真的改():
    first, _ = await run_mock()
    history = [
        ChatMessage("user", "请根据材料写一段故事。"),
        ChatMessage("assistant", first.text),
    ]
    second, events = await run_mock(history=history, user_message="再暗一点")

    assert second.stopped == "final"
    assert second.text != first.text
    assert "雨" in second.text or "灯" in second.text
    assert second.text.startswith("《")  # 还是整篇，不是 diff
    # 改稿也要留下一组新的工具卡片。
    assert [d["name"] for k, d in events if k == str(EventType.AGENT_TOOL_CALL)] == ["kb_search"]


async def test_mock_从不调工具也能交稿():
    result, _ = await run_mock(script="never_calls_tools")
    assert result.stopped == "final" and result.tool_calls == 0 and result.rounds == 1


async def test_mock_永远调工具时由收尾轮兜住():
    result, _ = await run_mock(script="loops_forever")
    assert result.rounds == loop.MAX_ROUNDS
    assert result.stopped == "final"  # 收尾轮还是交了稿
    assert result.text


async def test_mock_参数一直写坏时工具被撤掉():
    result, _ = await run_mock(script="bad_arguments")
    assert result.stopped == "final"
    assert result.rounds == 3  # 坏两次 → 撤掉 → 交稿


async def test_mock_空答复判失败():
    result, _ = await run_mock(script="empty_answer")
    assert result.stopped == "error" and result.text == ""
