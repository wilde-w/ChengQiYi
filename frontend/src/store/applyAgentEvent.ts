/**
 * 故事工坊的状态归约。**纯函数**——流式正确性全在这里被单测覆盖，store 只剩
 * 「存值 + 通知」。这一条与流水线的 `applyEvent` 是同一条经验：浏览器里手动
 * 复现「卡在某一帧断网」几乎不可能，而 reducer 一行就能断言。
 *
 * 与流水线那套刻意不共用：那边是「七个节点的进度」，这边是「一场对话」，
 * 硬凑成一个 reducer 的结果是两边都要带一堆 `if (isAgent)`。
 *
 * 三条不变量（对应 test/applyAgentEvent.test.ts）：
 *
 * 1. **seq 只进不退。** 重连会重放边界帧，同一条事件再来一次不能改变任何
 *    东西（引用相等，不触发重渲染）。
 * 2. **终态单向门。** 收到 `agent_completed` / `agent_cancelled` / `error`
 *    之后，任何 `delta` 都不再追加。晚到的正文会让「已交稿」的稿子继续变长，
 *    而用户会以为那是模型又写了一段。
 * 3. **快照是权威。** 流是「边跑边报」，完整性依赖每一帧都送达；快照不是。
 *    两者对不上时以快照为准——用户看到的数字必须和库里的一致。
 */

import type {
  AgentMessage,
  AgentSessionDetail,
  AgentStatus,
  RunEvent,
} from '../api/types'

/** 一次工具调用的结果。字段名与后端 `agent/events.py` 的 `tool_result` 对齐。 */
export interface ToolResult {
  ok: boolean
  /** 卡片上那行字（「6 条」「失败：超时」）。 */
  summary: string
  /** 展开后看得到的正文片段。**就是喂给模型的那段文字**，不另起一套渲染。 */
  preview: string
  elapsedMs: number
  chars: number
  /** bad_arguments / timeout / refused …失败时才有。 */
  reason?: string
}

/** 面板上的一张工具卡片。`tool_call` 先到、`tool_result` 后到，所以结果可空。 */
export interface ToolCard {
  callId: string
  name: string
  /** 解析成功的参数（模型写坏了就是空对象）。 */
  args: Record<string, unknown>
  /** 模型写的**原始 JSON 文本**。参数坏掉时，这里是唯一的线索。 */
  rawArgs: string
  /** 这一轮里的第几次模型调用。 */
  round: number
  /** 开始时间（客户端时钟）。用来在等待期显示「已等 2.3s」。 */
  startedAt: number
  result?: ToolResult
}

/** 前端特有的 `none`：还没开过会话。服务端永远不会下发它。 */
export type AgentLocalStatus = AgentStatus | 'none'

export interface AgentState {
  sessionId: string | null
  status: AgentLocalStatus
  /** 后端下发的阶段名（快照里的那份，中文）。 */
  stage: string | null
  /** 事件流推出来的、比 `stage` 更细的那句。终态时清空。 */
  liveStage: string | null

  title: string
  inputText: string
  inputChars: number

  /** 对话。**库里那一串的镜像**，刷新后由快照重建。 */
  messages: AgentMessage[]
  /** 当前这一轮的工具卡片。`agent_started` 时清空——上一轮的卡片由对话重建。 */
  cards: ToolCard[]

  /**
   * 当前这一版的正文（逐字贴上来）。
   *
   * 它与 `messages` 里最后一条 assistant 是**同一份内容的两个阶段**：流结束
   * 时快照落地，`reconcile` 会把两处对齐。故意留两份是因为正文在写的过程中
   * 还没落库，而右栏要在那一刻就长出来。
   */
  story: string
  /** 正文正在逐字到来（右栏画光标用）。终态一票否决。 */
  streaming: boolean

  turn: number
  /** 最近一轮的模型调用次数与工具调用次数——常驻面板，是自主程度的证据。 */
  rounds: number
  toolCalls: number
  model: string | null
  error: string | null
  demo: boolean

  lastSeq: number
  /** 客户端侧的问题（流断了等）。**不占 seq**，与流水线同一条规矩。 */
  warnings: string[]
}

export function agentInitialState(): AgentState {
  return {
    sessionId: null,
    status: 'none',
    stage: null,
    liveStage: null,
    title: '',
    inputText: '',
    inputChars: 0,
    messages: [],
    cards: [],
    story: '',
    streaming: false,
    turn: 0,
    rounds: 0,
    toolCalls: 0,
    model: null,
    error: null,
    demo: false,
    lastSeq: 0,
    warnings: [],
  }
}

export function isAgentTerminal(status: AgentLocalStatus): boolean {
  return status === 'failed' || status === 'cancelled'
}


/**
 * 开一条新会话。`detail` 是刷新后的重建路径：直接吃快照，不用等重放。
 *
 * **换会话必须重来一遍**，不能沿用上一个会话的 `lastSeq`：两条会话的 seq
 * 各自从 1 开始，沿用会让新会话的前 N 条事件全被当成重复丢掉——界面一片空白
 * 而日志里什么事都没有。
 */
export function resetForSession(
  sessionId: string,
  detail?: AgentSessionDetail | null,
): AgentState {
  const base = { ...agentInitialState(), sessionId }
  return detail ? mergeAgentSnapshot(base, detail) : base
}

export function applyAgentEvent(state: AgentState, event: RunEvent): AgentState {
  // 还没开会话：事件无处可落。**不能顺手建一条**——那样一条不属于任何会话的
  // 迟到帧会让面板凭空长出半场对话。
  if (!state.sessionId) return state
  // 切会话时的迟到帧：上一条会话的事件不能落到这一条的头上
  if (event.run_id && event.run_id !== state.sessionId) return state

  // 不变量 1：seq 只进不退。重放边界帧与乱序都走这一条。
  if (typeof event.seq !== 'number' || !Number.isFinite(event.seq)) return state
  if (event.seq <= state.lastSeq) return state

  const data = (event.data ?? {}) as Record<string, unknown>
  const next: AgentState = { ...state, lastSeq: event.seq }

  switch (event.type) {
    case 'agent_started': {
      // 新一轮。上一版正文已经落在 `messages` 里（上一轮结束时对账过），
      // 所以这里把「当前稿」腾空，右栏从零开始长新的一版。
      return {
        ...next,
        status: 'running',
        liveStage: '正在准备材料…',
        story: '',
        streaming: false,
        cards: [],
        error: null,
        // 上一轮那些客户端警告（流断过、连接重试过）到这里就过期了：
        // 新一轮已经开始跑，把它继续挂在面板上只会让人以为现在还有问题。
        warnings: [],
        turn: num(data.turn, state.turn),
      }
    }

    case 'agent_thinking': {
      const round = num(data.round, state.rounds)
      const tools = Array.isArray(data.tools) ? (data.tools as string[]).length : 0
      return {
        ...next,
        rounds: Math.max(state.rounds, round),
        liveStage: tools > 0 ? `第 ${round} 轮：模型正在决定下一步…` : `第 ${round} 轮：正在写正文…`,
      }
    }

    case 'agent_tool_call': {
      const callId = String(data.call_id ?? '')
      if (!callId) return next
      const card: ToolCard = {
        callId,
        name: String(data.name ?? ''),
        args: (data.args as Record<string, unknown>) ?? {},
        rawArgs: String(data.raw_args ?? ''),
        round: num(data.round, state.rounds),
        startedAt: num(data.at, Date.now()),
      }
      // 同 call_id 不重复建卡：重连重放时第一条 `tool_call` 会被再送一次，
      // 而卡片已经在那儿了。**已有的那张一个字都不能改**——重放发生在每一
      // 次重连上，用重放帧去覆盖它会把已经跑完的卡片打回「查询中…」，
      // 顺带把 startedAt 换成此刻（耗时从头算）。第一次到达的那一份才是
      // 权威的：它带着真实的 round 与客户端时钟。
      if (state.cards.some((c) => c.callId === callId)) {
        return { ...next, liveStage: `正在调用 ${card.name}…` }
      }
      return {
        ...next,
        cards: [...state.cards, card],
        liveStage: `正在调用 ${card.name}…`,
      }
    }

    case 'agent_tool_result': {
      const callId = String(data.call_id ?? '')
      if (!callId) return next
      const result: ToolResult = {
        ok: Boolean(data.ok),
        summary: String(data.summary ?? ''),
        preview: String(data.preview ?? ''),
        elapsedMs: num(data.elapsed_ms, 0),
        chars: num(data.chars, 0),
        reason: data.reason ? String(data.reason) : undefined,
      }
      // 被拒的调用（超限 / 未知工具 / 额度用完）也有一条 tool_result，
      // 但它们**没被执行**——后端记的 `tool_calls` 只数真正跑了的，
      // 这里跟着它数，免得面板上的数字比库里的大。
      const executed = result.reason !== 'refused' && result.reason !== 'bad_arguments'
      const cards = state.cards.some((c) => c.callId === callId)
        ? state.cards.map((c) => (c.callId === callId ? { ...c, result } : c))
        : [
            ...state.cards,
            {
              callId,
              name: String(data.name ?? ''),
              args: {},
              rawArgs: '',
              round: state.rounds,
              startedAt: Date.now(),
              result,
            },
          ]
      return {
        ...next,
        cards,
        toolCalls: executed ? state.toolCalls + 1 : state.toolCalls,
        liveStage: result.ok ? '正在读查到的材料…' : `工具 ${String(data.name ?? '')} 没有成功…`,
      }
    }

    case 'delta': {
      // 不变量 2：终态单向门。正文只在「这一轮还在跑」的时候往上贴——
      // `agent_completed` 把状态置回 idle，`agent_cancelled` / `error` 置成
      // 终态，于是这三种之后晚到的 `delta` 天然被挡在门外。挡不住的代价很具体：
      // 已经交稿的正文会在用户眼前继续变长，看起来像模型又写了一段。
      //
      // **丢掉的是正文，不是游标**：这里必须返回 `next` 而不是 `state`。
      // 游标停在原地的话，下一次重连会从这一帧起重新要一遍——把后面的
      // `agent_tool_result` 也一起重放，而它是**自增**的（`toolCalls + 1`），
      // 重放一次面板上的次数就比库里的多一次。
      if (state.status !== 'running') return next
      const text = String(data.text ?? '')
      if (!text) return next
      return { ...next, story: state.story + text, streaming: true }
    }

    case 'agent_completed':
      // 会话回到待命：还能接着说「再暗一点」，所以**不是**终态。
      return {
        ...next,
        status: 'idle',
        streaming: false,
        liveStage: null,
        rounds: num(data.rounds, state.rounds),
        toolCalls: num(data.tool_calls, state.toolCalls),
        turn: num(data.turn, state.turn),
      }

    case 'agent_cancelled':
      return { ...next, status: 'cancelled', streaming: false, liveStage: null }

    case 'error': {
      // 只有**带 `data.message` 的** error 才算「这一轮失败了」——那是服务层
      // 在落库之后发的。`sse_stream.error_frame` 合成的那些（流断了、回放到
      // 终态却没等到终态事件）`data` 是空的，它们说的是「这条管道出了问题」，
      // 而会话本身可能还好好的（比如正在待命）。把会话判死会让一个还能接着
      // 说「再暗一点」的面板变成只能重开。
      if (!data.message) {
        return noteAgentWarning(next, String(event.message ?? '事件流中断'))
      }
      return { ...next, status: 'failed', error: String(data.message), streaming: false, liveStage: null }
    }

    default:
      return next
  }
}

/**
 * 快照对账。**这是权威版本**：流负责实时性，快照负责真实性。
 *
 * `cards` 不动：快照里没有过程事件（那是 `agent_event` 的事），把它清掉只会
 * 让用户正在看的工具卡片在跑完的一瞬间消失。它不该丢的另一个理由——`elapsed_ms`
 * 只存在于事件里，清掉就再也找不回来了。
 */
export function mergeAgentSnapshot(state: AgentState, detail: AgentSessionDetail): AgentState {
  return {
    ...state,
    sessionId: detail.id,
    status: detail.status,
    stage: detail.stage ?? null,
    liveStage: null,
    title: detail.title,
    inputText: detail.input_text,
    inputChars: detail.input_chars,
    messages: detail.messages,
    story: detail.story ?? '',
    streaming: false,
    turn: detail.turn,
    rounds: detail.rounds,
    toolCalls: detail.tool_calls,
    model: detail.model ?? null,
    error: detail.error ?? null,
    demo: detail.demo,
  }
}

export function noteAgentWarning(state: AgentState, message: string): AgentState {
  return { ...state, warnings: [...state.warnings, message].slice(-50) }
}

// ----------------------------------------------------------------------
// 对话的组装
// ----------------------------------------------------------------------

export type TranscriptItem =
  | { kind: 'user'; id: string; turn: number; text: string; generated: boolean }
  /**
   * `draft` 区分同一种角色的两种东西。**带 `tool_calls` 的那条 assistant 消息
   * 不是草稿**——它是模型在决定查什么之前说的话（「我先去查查古诗怎么写这种
   * 迟来的领会」），而 `turn` 与真正的第一版正文相同，所以按 turn 标注的话
   * 两句话都会写成「第 1 稿」。模型偶尔还用英文写这句（在中文提示词下），
   * 标成草稿会让人以为正文串了英文——实际上正文是干净的。
   */
  | { kind: 'assistant'; id: string; turn: number; text: string; draft: boolean }
  | { kind: 'tool'; id: string; turn: number; card: ToolCard }

/**
 * 把「对话」与「过程」拼成中栏那一条时间线。
 *
 * 两者的来源不同：对话来自 `agent_message`（权威、可刷新重建），过程来自
 * 事件流（有耗时、有预览）。拼的时候**以对话为准**——每一条带 `tool_calls`
 * 的 assistant 消息生成一张卡，卡片上的结果优先用事件里那份（它更全），
 * 事件还没到就用紧随其后的 `role="tool"` 消息正文（刷新后只有它）。
 *
 * 于是同一次工具调用永远只画一张卡，无论它来自实时流还是回放。
 */
export function buildTranscript(state: Pick<AgentState, 'messages' | 'cards'>): TranscriptItem[] {
  const live = new Map(state.cards.map((c) => [c.callId, c]))
  const toolText = new Map<string, string>()
  for (const m of state.messages) {
    if (m.role === 'tool' && m.tool_call_id) toolText.set(m.tool_call_id, m.content)
  }

  const out: TranscriptItem[] = []
  for (const m of state.messages) {
    if (m.role === 'user') {
      out.push({
        kind: 'user',
        id: `m-${m.seq}`,
        turn: m.turn,
        text: m.content,
        // 收尾指令是代码生成的（时间/轮数用完了），不是用户说的话。
        generated: m.name === 'force_final',
      })
      continue
    }
    if (m.role === 'assistant') {
      if (m.content.trim()) {
        out.push({
          kind: 'assistant',
          id: `m-${m.seq}`,
          turn: m.turn,
          text: m.content,
          draft: (m.tool_calls?.length ?? 0) === 0,
        })
      }
      for (const call of m.tool_calls ?? []) {
        out.push({ kind: 'tool', id: `c-${call.id}`, turn: m.turn, card: cardOf(call, live, toolText) })
      }
    }
  }
  return out
}

function cardOf(
  call: { id: string; name: string; arguments: string },
  live: Map<string, ToolCard>,
  toolText: Map<string, string>,
): ToolCard {
  const found = live.get(call.id)
  if (found) return found
  // 事件流还没到（或者刚刷新、流还没重放完）。用消息里的那几样先画出来：
  // 名字、参数、结果正文——比「什么都没有」强，而且它们都是真的。
  const text = toolText.get(call.id)
  return {
    callId: call.id,
    name: call.name,
    args: safeParse(call.arguments),
    rawArgs: call.arguments,
    round: 0,
    startedAt: 0,
    result: text
      ? { ok: true, summary: '', preview: text, elapsedMs: 0, chars: text.length }
      : undefined,
  }
}

function safeParse(raw: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(raw || '{}')
    return parsed && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : {}
  } catch {
    // 模型写坏的 JSON。参数原文还在 `rawArgs` 里，卡片照画。
    return {}
  }
}

function num(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}
