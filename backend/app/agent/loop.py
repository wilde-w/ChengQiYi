"""故事工坊的一轮：模型 ↔ 工具的循环。

这是全项目第一处**由模型决定下一步**的控制流。流水线的 7 个节点是写死的
线性拓扑；这里模型自己决定查什么、查几次、什么时候交稿。代价是不可复现
的运行，所以每一处上限都必须显式：

- `MAX_ROUNDS`：模型调用次数（**含**最后那次强制作答轮）；
- `MAX_TOOL_CALLS_PER_ROUND`：一轮里最多真执行几次调用；
- `MAX_MCP_CALLS_PER_TURN`：起子进程的工具（古典文学 MCP）每次最多几次；
- `TOTAL_TIMEOUT` + `FINAL_TIMEOUT`：整轮预算与收尾预算。

## 三条会让服务端 400 的硬约束，都在这一个文件里兜住

1. **assistant 说了几次调用，就要有幾条 `role="tool"` 回复。** 少一条，
   下一轮请求被拒——症状是「第三轮突然报错」，最难查。所以超限的、
   工具名不存在的、参数坏掉的，**都要回一条**（内容是中文说明）。
   同理，收尾轮收到 tool_calls 时不能把它们塞进 assistant 消息里——那条
   消息会落库并在下一轮重发，而它永远等不到配对回复。
2. **落库的那串消息必须逐字等于喂给模型的那串。** 包括强制作答轮那条
   `force_final` 的 user 消息——它由代码生成、不是用户说的，但模型见过它，
   不存就会让两条路分叉。
3. **正文不做真流式**：`stream()` 不支持工具调用，而流式下 `arguments`
   是跨 chunk 的增量 JSON，要另写状态机。agent 的等待时间几乎全花在工具
   上，正文到手后再按 `chunk_text` 切块重放，用户看到的效果完全一样。

## 为什么坏参数要撤工具

真模型最常犯的错是参数写坏。不撤的话它会原地重试到把 6 轮烧光，用户最后
看到一句「模型没能完成任务」——而它其实只需要有人告诉它「这个工具没了，
用手上的材料写」。撤销是**唯一**能让它换路的做法（重发同样的 schema
等于重发同一个诱惑）。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.agent import events as ev
from app.agent.events import Emit
from app.agent.tools import ToolOutcome, ToolRegistry, ToolSpec, parse_arguments
from app.config import get_settings
from app.constants import DELTA_INTERVAL_MS, EventType
from app.logging_conf import get_logger
from app.providers.base import ChatMessage, ProviderError
from app.text_chunks import chunk_text

log = get_logger(__name__)

#: 缓存键与日志里的 task 名。**必须是新的名字**——沿用 `literary` 会让
#: 故事与洞察报告共用缓存，而两者的提示词与输出契约完全不同。
TASK = "agent_turn"

MAX_ROUNDS = 6
MAX_TOOL_CALLS_PER_ROUND = 4
MAX_MCP_CALLS_PER_TURN = 2
TOTAL_TIMEOUT = 180.0
#: 收尾轮的预算。单独给，是因为「轮数到了」和「时间到了」都不该让用户
#: 白等一场——宁可多给 10 秒换一份稿子。
FINAL_TIMEOUT = 60.0

#: 同名工具连续几次参数不合法就撤掉它。
BAD_ARGS_LIMIT = 2

#: 额度名 → 上限。`ToolSpec.quota` 里写的名字在这里查上限。
_QUOTA_LIMITS = {"mcp": MAX_MCP_CALLS_PER_TURN}

#: 强制作答轮追加的指令。`name="force_final"` 让落库的那条能自证来历
#: （前端据此不把它画成用户说的话），而 wire format 里它就是个普通 user 消息。
FORCE_FINAL_NAME = "force_final"
FORCE_FINAL_TEXT = (
    "时间或轮次已经用完。现在直接交稿：只输出正文，不要再调用任何工具。"
)


@dataclass(slots=True)
class TurnRequest:
    """一轮的输入。`history` 是库里那一串（**不含**本轮 user），system 由代码重建。"""

    system_prompt: str
    history: Sequence[ChatMessage]
    user_message: str
    turn: int


@dataclass(slots=True)
class TurnResult:
    #: 本轮新增的消息，按序落库。含中间轮带 tool_calls 的 assistant 与工具回复。
    messages: list[ChatMessage] = field(default_factory=list)
    #: 交稿正文。取消 / 出错时为空——**不产半截稿**。
    text: str = ""
    rounds: int = 0
    tool_calls: int = 0
    #: final / cancelled / error
    stopped: str = "final"
    error: str = ""


async def run_turn(
    req: TurnRequest,
    *,
    model: Any,
    registry: ToolRegistry,
    emit: Emit,
    should_cancel: Callable[[], Awaitable[bool]] | None = None,
) -> TurnResult:
    """跑完一轮。**只返回结果：不碰数据库，也不发终态事件。**

    `AGENT_STARTED` 之前的准备与三种终态（完成 / 取消 / 错误）都由
    `agent_service` 在**落库之后**发出。循环自己发就会抢在落库前面抵达前端，
    而前端收到终态的第一件事是拉快照对账——那一瞬它会看到「流说结束了、
    快照说还在写」（引擎走查时实测到的那一幕）。所以循环只发过程事件：
    轮次、工具卡片、正文增量。
    """
    messages: list[ChatMessage] = [
        ChatMessage("system", req.system_prompt),
        *req.history,
        ChatMessage("user", req.user_message),
    ]
    new: list[ChatMessage] = []
    deadline = time.monotonic() + TOTAL_TIMEOUT
    result = TurnResult()
    bad_args: dict[str, int] = {}
    quota: dict[str, int] = {}

    async def stop_now() -> TurnResult | None:
        """取消已经挂上了就收工。**已经发生的工具调用要落库**：面板上那些卡片
        是用户看过的，刷新之后必须还在。丢掉的只有正文——半截稿比没有稿更误导人。
        """
        if should_cancel is None or not await should_cancel():
            return None
        result.stopped = "cancelled"
        result.messages = new
        return result

    for round_no in range(1, MAX_ROUNDS + 1):
        result.rounds = round_no
        if (stopped := await stop_now()) is not None:
            return stopped

        forced = round_no == MAX_ROUNDS or time.monotonic() >= deadline
        if forced:
            note = ChatMessage("user", FORCE_FINAL_TEXT, name=FORCE_FINAL_NAME)
            messages.append(note)
            new.append(note)
        schemas = [] if forced else registry.schemas()
        await emit(
            EventType.AGENT_THINKING,
            ev.thinking(round_no=round_no, tools=[] if forced else registry.names()),
        )

        try:
            res = await asyncio.wait_for(
                model.complete(messages, task=TASK, tools=schemas),
                timeout=_round_budget(deadline, forced),
            )
        except TimeoutError:
            return _fail(result, new, "模型在限时内没有返回。")
        except ProviderError as exc:
            return _fail(result, new, f"模型调用失败：{exc}")
        except Exception as exc:  # 宽到 base class 是**故意的**，理由见下
            # 未预期的异常也要变成一次干净的收尾：面板上永远转圈是更坏的结果。
            log.exception("agent_round_failed", round=round_no)
            return _fail(result, new, f"生成过程中出错：{type(exc).__name__}: {exc}")

        calls = list(res.tool_calls)
        if forced and calls:
            # 收尾轮明说了不许再调工具。**不能把它们写进 assistant 消息**：
            # 那条消息会落库、下一轮重发，而它永远等不到配对的 tool 回复。
            log.warning("agent_tool_call_on_forced_round", round=round_no, calls=[c.name for c in calls])
            calls = []

        if not calls:
            # **交稿这一刻要再查一次取消。** 循环顶部那次只在**下一轮**才跑到，
            # 而交稿之后没有下一轮了——漏掉这一次，表现就是「点了停止，新版本
            # 照样覆盖上来」。而且这一轮恰恰是最可能被按停止的：正文是整轮里
            # 最长的一段等待（引擎走查时就是这么发现的，两处检查缺一不可）。
            if (stopped := await stop_now()) is not None:
                return stopped
            text = (res.text or "").strip()
            if not text:
                return _fail(result, new, "模型这一轮既没有调用工具，也没有给出正文。")
            assistant = ChatMessage("assistant", res.text)
            messages.append(assistant)
            new.append(assistant)
            await _emit_text(emit, res.text)
            result.messages = new
            result.text = res.text
            result.stopped = "final"
            return result

        assistant = ChatMessage("assistant", res.text, tool_calls=calls)
        messages.append(assistant)
        new.append(assistant)

        # 1) 所有调用先上卡片，顺序 = 模型给出的顺序。
        for call in calls:
            args, _ = parse_arguments(call)
            await emit(
                EventType.AGENT_TOOL_CALL,
                ev.tool_call(
                    call_id=call.id,
                    name=call.name,
                    args=args,
                    raw_args=call.arguments,
                    round_no=round_no,
                ),
            )

        # 2) 分派：能并发的先并发跑（网络等待占大头），串行的单独排队。
        planned, refusals = _plan(registry, calls, quota)
        outcomes: dict[str, ToolOutcome] = {}

        parallel = [(c, s) for c, s in planned if not s.serial]
        serial = [(c, s) for c, s in planned if s.serial]
        if parallel:
            done = await asyncio.gather(*(registry.dispatch(c) for c, _ in parallel))
            # strict=True：gather 保证等长，短了就是有人动了上面那行生成器
            for (call, _spec), outcome in zip(parallel, done, strict=True):
                outcomes[call.id] = outcome
                await _emit_outcome(emit, call, outcome)
        for call, _spec in serial:
            outcome = await registry.dispatch(call)
            outcomes[call.id] = outcome
            await _emit_outcome(emit, call, outcome)
        result.tool_calls += len(planned)

        # 3) 按原序组装回复。**每个 call_id 都要有一条**，漏一条下一轮 400。
        retired: set[str] = set()
        for call in calls:
            outcome = outcomes.get(call.id)
            if outcome is None:
                reason = refusals.get(call.id, "工具没有返回结果（内部错误）。")
                outcome = ToolOutcome(
                    text=reason, ok=False, summary="未执行", preview=reason, reason="refused"
                )
                await _emit_outcome(emit, call, outcome)
            elif outcome.reason == "bad_arguments":
                bad_args[call.name] = bad_args.get(call.name, 0) + 1
                if bad_args[call.name] >= BAD_ARGS_LIMIT and registry.get(call.name) is not None:
                    registry.retire(call.name)
                    retired.add(call.name)
                    log.info("agent_tool_retired", tool=call.name, round=round_no)
            elif outcome.ok:
                bad_args[call.name] = 0

            text = outcome.text
            if call.name in retired:
                text += f"\n（`{call.name}` 连续 {BAD_ARGS_LIMIT} 次参数不合法，本轮已停用。）"
            msg = ChatMessage("tool", text, tool_call_id=call.id, name=call.name)
            messages.append(msg)
            new.append(msg)

    # 只有 range 走完才会到这里，而最后一轮就是强制作答轮——它要么交了稿
    # （在循环里 return），要么已经按上面的错误分支返回。
    return _fail(result, new, "模型在收尾轮之后仍然没有给出正文。")


def _round_budget(deadline: float, forced: bool) -> float:
    """这一轮给模型多久。

    收尾轮单独给 `FINAL_TIMEOUT`；其它轮受 `TOTAL_TIMEOUT` 约束，但至少
    留 5 秒——预算只剩 0.2 秒时发出去的请求等于必然超时。
    """
    if forced:
        return FINAL_TIMEOUT
    return max(5.0, min(FINAL_TIMEOUT, deadline - time.monotonic()))


def _plan(
    registry: ToolRegistry, calls: Sequence[Any], quota: dict[str, int]
) -> tuple[list[tuple[Any, ToolSpec]], dict[str, str]]:
    """决定哪些调用真的执行，其余给一段中文说明。

    三种拒绝理由都要**说清楚原因**：模型看不到 schema 也看不到配额表，
    只看到「不行」的话，它下一轮还会做同样的事。
    """
    planned: list[tuple[Any, ToolSpec]] = []
    refusals: dict[str, str] = {}
    for i, call in enumerate(calls):
        spec = registry.get(call.name)
        if i >= MAX_TOOL_CALLS_PER_ROUND:
            refusals[call.id] = (
                f"一轮最多执行 {MAX_TOOL_CALLS_PER_ROUND} 次工具调用，这一次没有执行。"
                "请先用手上的材料判断下一步。"
            )
            continue
        if spec is None:
            refusals[call.id] = (
                f"没有名为 `{call.name}` 的工具：名字可能写错了，或它已在本轮停用。"
                + registry.hint()
            )
            continue
        if spec.quota:
            limit = _QUOTA_LIMITS.get(spec.quota, 0)
            if quota.get(spec.quota, 0) >= limit:
                refusals[call.id] = (
                    f"`{call.name}` 这一类工具每次最多调用 {limit} 次，额度已用完。"
                    "请用已有材料继续，或直接交稿。"
                )
                continue
            quota[spec.quota] = quota.get(spec.quota, 0) + 1
        planned.append((call, spec))
    return planned, refusals


async def _emit_outcome(emit: Emit, call: Any, outcome: ToolOutcome) -> None:
    await emit(
        EventType.AGENT_TOOL_RESULT,
        ev.tool_result(
            call_id=call.id,
            name=call.name,
            ok=outcome.ok,
            summary=outcome.summary,
            preview=outcome.preview,
            elapsed_ms=outcome.elapsed_ms,
            chars=len(outcome.text),
            reason=outcome.reason,
        ),
    )


async def _emit_text(emit: Emit, text: str) -> None:
    """正文到手后按块重放。`chunk_text` 只切不删，拼回去必须逐字等于正文。"""
    chunks = chunk_text(text)
    delay = DELTA_INTERVAL_MS / 1000 if get_settings().is_demo else 0.0
    for i, chunk in enumerate(chunks):
        await emit(EventType.DELTA, ev.delta(chunk))
        if delay and i < len(chunks) - 1:
            await asyncio.sleep(delay)


def _fail(result: TurnResult, new: list[ChatMessage], message: str) -> TurnResult:
    """干净地失败：**不产正文**，但保住已经发生的工具调用。

    错误事件不在这里发——`agent_service` 会在落库之后补上（理由见 `run_turn`）。
    """
    log.warning("agent_turn_failed", error=message, rounds=result.rounds)
    result.stopped = "error"
    result.error = message
    result.text = ""
    result.messages = new
    return result


__all__ = [
    "BAD_ARGS_LIMIT",
    "FORCE_FINAL_TEXT",
    "MAX_MCP_CALLS_PER_TURN",
    "MAX_ROUNDS",
    "MAX_TOOL_CALLS_PER_ROUND",
    "TASK",
    "TOTAL_TIMEOUT",
    "TurnRequest",
    "TurnResult",
    "run_turn",
]
