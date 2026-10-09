"""故事工坊的真模型冒烟：跑一轮，把**发出去的东西**与**收回来的东西**原样打出来。

## 为什么必须单独有这么一个脚本

`pytest` 全绿在这里给的是**虚假的安全感**。默认跑测试时模型是 mock，而 mock
走的是一条我们自己写的确定性假循环：它当然会规规矩矩地返回 `tool_calls`，当然
会带上配对的 `role="tool"` 应答，参数当然是合法 JSON。真模型会坏的那几种方式
（用文字说要查而不是真的调、`arguments` 是跨 chunk 拼坏的 JSON、返回了一个
不存在的工具名、把 `content` 留成 `null`）在 mock 下**一次都不会走到**。
所以这个脚本不测已经被单测钉住的东西，只测单测碰不到的那一层。

## 它比「跑通就行」多做的三件事

1. **把 wire format 打出来。** 每一轮 request 里有哪些工具 schema、response
   里 callback 了哪些 `tool_calls`（名字 + 原始 arguments 串）。这是出问题时
   唯一能看的东西。
2. **核对协议配平**：把下一轮 request 里 `role="tool"` 消息的 `tool_call_id`
   收上来，断言上一轮每一个 `tool_calls[i].id` 都在里面。漏一条 = 真模型那边
   下一轮必 400，而错误信息只会说「messages 格式不对」。
3. **数工具调用次数**：验收标准是「一段典型评论的首次生成，工具调用 ≥ 2」。
   这个数字同时也是风险 #3（模型只用文字说要查）唯一会露头的地方。

用法：

    .venv/Scripts/python.exe agent_smoke.py                 # 内置的典型评论
    .venv/Scripts/python.exe agent_smoke.py "自己的文本"
    .venv/Scripts/python.exe agent_smoke.py --allow-mock    # 明知是 mock 也跑，只看链路

退出码：0 通过 / 1 没通过 / 2 环境不对（模型是 mock、依赖连不上）。
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

from app.agent import events as ev
from app.clients.neo4j_client import init_neo4j
from app.clients.qdrant_client import init_qdrant
from app.clients.redis_client import init_redis
from app.config import get_settings
from app.constants import EventType
from app.db.session import dispose_engine, init_engine
from app.graph.bus import get_bus
from app.logging_conf import configure_logging
from app.prompts import story as story_prompt
from app.services import agent_service

#: 一段典型的抖音评论——有情绪、有冲突、有话没说完。不是随便找的：它要能
#: 让模型有理由去查（情绪指向了心理机制，措辞里有可以被追问的意象）。
DEFAULT_INPUT = (
    "我妈昨天给我打电话，说家里那盆养了十年的君子兰终于开花了，"
    "她讲得很高兴，讲了快十分钟。我一边嗯嗯地应着，一边在想明天要交的报表。"
    "挂掉之后我才反应过来——她其实是想我了，从过年到现在我一次都没回去。"
    "我给花浇水的次数，比她见到我的次数多。"
)

TERMINAL = {"agent_completed", "agent_cancelled", "error"}


class Recorder:
    """包住真模型，把每次 `complete()` 的请求与响应留下来。

    **必须包在模型这一层，不能去解 HTTP。** 想在 `messages` 进 `openai_compat`
    之前看到它，只能在调用点上截——一旦进了那一层，角色分支、`tool_calls`
    序列化都已经做完了，看到的是 OpenAI 的形状而不是「我们发出去的东西」。
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.name = getattr(inner, "name", "?")
        self.rounds: list[dict[str, Any]] = []

    async def complete(self, messages: list[Any], *, task: str, tools: Any = None, **kw: Any):
        sent = time.monotonic()
        res = await self._inner.complete(messages, task=task, tools=tools, **kw)
        self.rounds.append(
            {
                "task": task,
                "messages": messages,
                "tools": [t.get("function", {}).get("name") for t in (tools or [])],
                "text": res.text or "",
                "calls": list(res.tool_calls),
                "elapsed_ms": int((time.monotonic() - sent) * 1000),
            }
        )
        return res


def _preview(text: str, n: int = 160) -> str:
    flat = " ".join(text.split())
    return flat[:n] + ("…" if len(flat) > n else "")


def dump_rounds(rec: Recorder) -> None:
    print("\n" + "=" * 78)
    print(f"发给模型的每一轮（模型 {rec.name}，共 {len(rec.rounds)} 轮）")
    print("=" * 78)
    for i, r in enumerate(rec.rounds, 1):
        print(f"\n--- 第 {i} 轮  task={r['task']}  工具 schema: {r['tools'] or '（无，收尾轮）'}")
        # 消息序列的**结构**比内容重要：一眼能看出有没有配对的 tool 应答
        for m in r["messages"]:
            role = getattr(m, "role", "?")
            calls = getattr(m, "tool_calls", None) or []
            tail = ""
            if calls:
                tail = f"   → tool_calls: {[c.name for c in calls]}"
            elif getattr(m, "tool_call_id", None):
                tail = f"   ← 应答 {m.tool_call_id} ({getattr(m, 'name', '')})"
            elif getattr(m, "name", None):
                tail = f"   [{m.name}]"
            print(f"    [{role:9}] {_preview(getattr(m, 'content', '') or '', 90)}{tail}")
        print(f"  ← 文字 {len(r['text'])} 字，{r['elapsed_ms']}ms：{_preview(r['text'], 100)}")
        for c in r["calls"]:
            print(f"  ← 调用 {c.name}  arguments={c.arguments[:120]!r}")


def check_pairing(rec: Recorder) -> list[str]:
    """核对协议配平：每个 `tool_calls[i].id` 都要在下一轮被一对一应答。

    这是本项目最容易犯、也最难查的一个错——漏一条的后果是**真模型那边下一轮
    400**，而 mock 永远规规矩矩。这里在真实往返上验一遍。
    """
    problems: list[str] = []
    for i, r in enumerate(rec.rounds[:-1], 1):
        want = [c.id for c in r["calls"]]
        if not want:
            continue
        nxt = rec.rounds[i]
        got = [getattr(m, "tool_call_id", None) for m in nxt["messages"] if getattr(m, "role", "") == "tool"]
        missing = [cid for cid in want if cid not in got]
        extra = [cid for cid in got if cid and cid not in want]
        if missing:
            problems.append(f"第 {i} 轮的 {missing} 没有配对的 tool 应答（下一轮会 400）")
        if extra:
            problems.append(f"第 {i} 轮后出现了无主的 tool 应答 {extra}")
    return problems


async def run(text: str, allow_mock: bool) -> int:
    settings = get_settings()
    summary = settings.provider_summary()
    if summary.get("llm") == "mock" and not allow_mock:
        print(f"模型是 mock（MOCK_MODE={summary.get('mock_mode')}），这个脚本测不到任何东西。")
        print("补上 LLM 的 key，或加 --allow-mock 只看链路。")
        return 2
    print(f"providers: {summary}")

    init_engine()
    init_redis()
    init_qdrant()
    init_neo4j()

    rec = Recorder(agent_service.get_chat_model())

    def _stub(*_a: Any, **_k: Any) -> Recorder:
        return rec

    # `execute_turn` 里写的是 `model=get_chat_model()`——模块级名字，运行时查表。
    # 所以在这里换掉它就能截到每一次往返，且**不用改被测代码一行**。
    agent_service.get_chat_model = _stub  # type: ignore[assignment]

    session = await agent_service.create_session(input_text=text, allow_novel=True)
    print(f"会话 {session.id}，材料 {len(text)} 字")
    print(f"（跑完可以在面板上打开它：刷新页面会自动接回来，或直接查 agent_session {session.id}）")

    bus = await get_bus(ev.NAMESPACE)
    events: list[Any] = []

    async def consume() -> None:
        # 订阅先于发消息，省得「已经跑完了才连上」——回放能兜住，但那样看不到
        # 逐条到达的节奏，而这正是这个脚本想让人看见的东西之一。
        async for event in bus.subscribe(session.id, 0):
            events.append(event)
            if event.type == EventType.AGENT_TOOL_CALL:
                print(f"  · 第 {event.data.get('round')} 轮调用 {event.data.get('name')} "
                      f"{str(event.data.get('raw_args'))[:110]}")
            elif event.type == EventType.AGENT_TOOL_RESULT:
                mark = "✓" if event.data.get("ok") else "✗"
                print(f"  · {mark} {event.data.get('name')} {event.data.get('summary')} "
                      f"（{event.data.get('chars')} 字 · {event.data.get('elapsed_ms')}ms）")
            if event.type.value in TERMINAL:
                return

    started = time.monotonic()
    watcher = asyncio.create_task(consume())
    await agent_service.send_message(session.id, story_prompt.FIRST_INSTRUCTION)
    await asyncio.wait_for(watcher, timeout=240)
    wall = time.monotonic() - started

    dump_rounds(rec)

    detail = await agent_service.session_detail(session.id)
    tools_used = [e for e in events if e.type == EventType.AGENT_TOOL_RESULT]
    executed = [e for e in tools_used if e.data.get("reason") not in {"refused", "bad_arguments"}]

    print("\n" + "=" * 78)
    print(f"结果：{detail.session.status} · 第 {detail.session.turn} 稿 · "
          f"{detail.session.rounds} 轮 · {detail.session.tool_calls} 次工具调用 · "
          f"正文 {len(detail.session.story or '')} 字 · 用时 {wall:.1f}s")
    print("=" * 78)
    print(f"正文：{_preview(detail.session.story or '（空）', 700)}")

    failures: list[str] = []
    failures += check_pairing(rec)
    if len(executed) < 2:
        # 验收标准。**不达标多半不是 bug 而是提示词的问题**：模型在只用文字说
        # 「我应该去查一下」而不是真的调，而它看起来像交稿，日志完全正常。
        failures.append(
            f"首次生成只真正执行了 {len(executed)} 次工具调用（验收标准 ≥ 2）——"
            "如果模型在回答里用文字说「我需要查一下」，那是风险 #3：它没有真的调工具。"
        )
    if not (detail.session.story or "").strip():
        failures.append("正文是空的")
    if detail.session.error:
        failures.append(f"会话报错：{detail.session.error}")

    if failures:
        print("\n未通过：")
        for f in failures:
            print(f"  ✗ {f}")
    else:
        print("\n通过：工具调用 ≥ 2、正文非空、每个 tool_call 都有配对的应答。")
    return 1 if failures else 0


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    allow_mock = "--allow-mock" in sys.argv
    text = argv[0] if argv else DEFAULT_INPUT
    configure_logging(get_settings().LOG_LEVEL, False)
    try:
        return asyncio.run(_main(text, allow_mock))
    finally:
        asyncio.run(dispose_engine())


async def _main(text: str, allow_mock: bool) -> int:
    try:
        return await run(text, allow_mock)
    except Exception as exc:
        # 依赖连不上（PG / Redis 没起）也走这条。**不能让用户对着一串堆栈猜**
        # 到底是被测的东西坏了还是环境没准备好——这是两种完全不同的动作。
        print(f"\n环境不对，没能跑起来：{type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
