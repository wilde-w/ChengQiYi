/**
 * 故事工坊归约器的黄金测试。
 *
 * 与 `applyEvent.test.ts` 同一条理由：这里钉的是**不变量**，不是实现细节。
 * 流式界面真正会坏的那几种方式——重连重放让同一张工具卡画两遍、终态之后
 * 正文还在长、切会话时上一条的帧落到这一条头上——在浏览器里手动复现要精确
 * 卡在某一帧断网，而在 reducer 里一行就能断言。
 *
 * 故意**不**测样式与组件：那是 Playwright 的活（Phase 4）。这一层只管状态。
 */

import { describe, expect, it } from 'vitest'

import type { AgentSessionDetail, RunEvent } from '../api/types'
import {
  agentInitialState,
  applyAgentEvent,
  buildTranscript,
  mergeAgentSnapshot,
  resetForSession,
  type AgentState,
  type ToolCard,
  type TranscriptItem,
} from '../store/applyAgentEvent'

const SID = 's1'

function ev(
  seq: number,
  type: RunEvent['type'],
  data?: Record<string, unknown>,
  runId = SID,
): RunEvent {
  return { seq, run_id: runId, type, data }
}

/** 从空会话起，按顺序喂一串事件。省得每个用例都写一遍 `let s = ...`。 */
function feed(events: RunEvent[], from?: AgentState): AgentState {
  return events.reduce(applyAgentEvent, from ?? resetForSession(SID))
}

function detail(partial: Partial<AgentSessionDetail> = {}): AgentSessionDetail {
  return {
    id: SID,
    title: '标题',
    input_chars: 20,
    input_text: '一段评论',
    status: 'idle',
    stage: null,
    turn: 1,
    story: '第一版正文',
    rounds: 2,
    tool_calls: 3,
    model: 'deepseek-chat',
    providers: {},
    options: {},
    error: null,
    is_running: false,
    is_terminal: false,
    demo: false,
    created_at: '2026-01-01T00:00:00',
    updated_at: '2026-01-01T00:00:00',
    messages: [],
    ...partial,
  }
}

/** 取第一张工具卡。省得每个断言都写一遍窄化（`noUncheckedIndexedAccess` 下
 *  `items[0].card` 通不过，而一串 `?.` 会把断言写得没法读）。 */
function firstToolCard(items: TranscriptItem[]): ToolCard {
  const item = items.find((i) => i.kind === 'tool')
  if (!item || item.kind !== 'tool') throw new Error('时间线里没有工具卡')
  return item.card
}

function firstItem(items: TranscriptItem[]): TranscriptItem {
  const item = items[0]
  if (!item) throw new Error('时间线是空的')
  return item
}

describe('seq 只进不退（不变量 1）', () => {
  it('丢弃 seq 小于等于游标的事件', () => {
    const s = feed([ev(1, 'agent_started', { turn: 1 }), ev(2, 'delta', { text: '甲' })])
    const again = applyAgentEvent(s, ev(2, 'delta', { text: '甲' }))
    // 引用相等：重连重放边界帧不该触发一次重渲染
    expect(again).toBe(s)

    const stale = applyAgentEvent(s, ev(1, 'agent_started', { turn: 1 }))
    expect(stale).toBe(s)
    expect(stale.lastSeq).toBe(2)
  })

  it('乱序到达的迟到帧不改变状态', () => {
    const s = feed([
      ev(1, 'agent_started', { turn: 1 }),
      ev(3, 'delta', { text: '甲' }),
      ev(5, 'agent_completed', { turn: 1, rounds: 2, tool_calls: 1 }),
    ])
    const late = applyAgentEvent(s, ev(4, 'delta', { text: '乙' }))
    expect(late).toBe(s)
    expect(late.story).toBe('甲')
  })

  it('没有会话时事件无处可落，不会被顺手建出一条会话', () => {
    const empty = agentInitialState()
    expect(applyAgentEvent(empty, ev(1, 'agent_started', { turn: 1 }))).toBe(empty)
  })

  it('切会话后的迟到帧落到新会话头上会被挡住', () => {
    const s = resetForSession('s2')
    const other = applyAgentEvent(s, ev(1, 'agent_started', { turn: 1 }, 's1'))
    expect(other).toBe(s)
    // 反向确认：换成自己的 run_id 就进得去（否则上面那条断言可能是别的原因过的）
    const mine = applyAgentEvent(s, ev(1, 'agent_started', { turn: 1 }, 's2'))
    expect(mine.status).toBe('running')
  })
})

describe('工具卡片：同一次调用只画一张', () => {
  it('tool_call 与 tool_result 按 call_id 配对', () => {
    const s = feed([
      ev(1, 'agent_started', { turn: 1 }),
      ev(2, 'agent_tool_call', {
        call_id: 'c1',
        name: 'kb_search',
        args: { query: '想念' },
        raw_args: '{"query":"想念"}',
        round: 1,
        at: 1000,
      }),
      ev(3, 'agent_tool_result', {
        call_id: 'c1',
        name: 'kb_search',
        ok: true,
        summary: '6 条',
        preview: '……',
        elapsed_ms: 320,
        chars: 480,
      }),
    ])
    expect(s.cards).toHaveLength(1)
    const card = s.cards[0]!
    expect(card.callId).toBe('c1')
    expect(card.args.query).toBe('想念')
    // 卡片身份（args/rawArgs/startedAt）来自 tool_call，结果来自 tool_result
    expect(card.startedAt).toBe(1000)
    expect(card.result?.summary).toBe('6 条')
    expect(card.result?.elapsedMs).toBe(320)
    expect(s.toolCalls).toBe(1)
  })

  it('重连重放 tool_call 不会建出第二张卡', () => {
    const s = feed([
      ev(1, 'agent_tool_call', { call_id: 'c1', name: 'kb_search', round: 1 }),
      ev(2, 'agent_tool_result', { call_id: 'c1', name: 'kb_search', ok: true, summary: '6 条' }),
    ])
    // 重连后从 seq 1 回放：这一条会带着新的 seq 再来一次
    const replay = applyAgentEvent(s, ev(9, 'agent_tool_call', { call_id: 'c1', name: 'kb_search' }))
    expect(replay.cards).toHaveLength(1)
    // 结果不能被重放的 tool_call 抹掉
    expect(replay.cards[0]!.result?.summary).toBe('6 条')
  })

  it('没有配对 tool_call 的结果也会长出一张卡（结果先到不至于丢）', () => {
    const s = feed([
      ev(1, 'agent_tool_result', { call_id: 'c9', name: 'kb_graph', ok: false, reason: 'timeout' }),
    ])
    expect(s.cards).toHaveLength(1)
    expect(s.cards[0]!.result?.ok).toBe(false)
  })

  it('被拒的调用不计数——面板上的数字必须和库里的一致', () => {
    const s = feed([
      ev(1, 'agent_tool_result', { call_id: 'c1', name: 'kb_search', ok: false, reason: 'refused' }),
      ev(2, 'agent_tool_result', { call_id: 'c2', name: 'kb_search', ok: false, reason: 'bad_arguments' }),
      ev(3, 'agent_tool_result', { call_id: 'c3', name: 'kb_search', ok: false, reason: 'timeout' }),
      ev(4, 'agent_tool_result', { call_id: 'c4', name: 'kb_search', ok: true, summary: '6 条' }),
    ])
    // timeout 是**真的跑过**了（只是没跑成），后端也数它
    expect(s.toolCalls).toBe(2)
    expect(s.cards).toHaveLength(4)
  })

  it('agent_started 清空上一轮的卡片，但不动对话里的历史', () => {
    const first = feed([
      ev(1, 'agent_started', { turn: 1 }),
      ev(2, 'agent_tool_call', { call_id: 'c1', name: 'kb_search' }),
      ev(3, 'agent_completed', { turn: 1, rounds: 1, tool_calls: 1 }),
    ])
    expect(first.cards).toHaveLength(1)

    const second = applyAgentEvent(first, ev(4, 'agent_started', { turn: 2 }))
    expect(second.cards).toEqual([])
    expect(second.turn).toBe(2)
    // 新一版从零长起：右栏不该还挂着上一版的正文
    expect(second.story).toBe('')
    expect(second.status).toBe('running')
  })
})

describe('终态单向门（不变量 2）', () => {
  it('agent_completed 之后晚到的 delta 一律不追加', () => {
    const s = feed([
      ev(1, 'agent_started', { turn: 1 }),
      ev(2, 'delta', { text: '甲' }),
      ev(3, 'agent_completed', { turn: 1, rounds: 2, tool_calls: 4 }),
    ])
    const late = applyAgentEvent(s, ev(4, 'delta', { text: '乙' }))
    expect(late.story).toBe('甲')
    expect(late.streaming).toBe(false)
    expect(late.lastSeq).toBe(4) // 游标还是要推进，否则后面每一帧都会重来一遍
  })

  it('agent_cancelled 之后同样挡住 delta', () => {
    const s = feed([
      ev(1, 'agent_started', { turn: 1 }),
      ev(2, 'delta', { text: '甲' }),
      ev(3, 'agent_cancelled', { turn: 1 }),
      ev(4, 'delta', { text: '乙' }),
    ])
    expect(s.story).toBe('甲')
    expect(s.status).toBe('cancelled')
    expect(s.streaming).toBe(false)
  })

  it('agent_completed 只是「这一轮结束」，不是会话终态', () => {
    const s = feed([ev(1, 'agent_started', { turn: 1 }), ev(2, 'agent_completed', { turn: 1 })])
    expect(s.status).toBe('idle')
    // 还能接着说「再暗一点」：门是关上了，但会话没死
    expect(s.status).not.toBe('failed')
    expect(s.status).not.toBe('cancelled')
  })
})

describe('error 帧：合成的与真的分开', () => {
  it('带 data.message 的才算这一轮失败', () => {
    const s = feed([
      ev(1, 'agent_started', { turn: 1 }),
      ev(2, 'error', { message: '模型调用失败：超时' }),
    ])
    expect(s.status).toBe('failed')
    expect(s.error).toBe('模型调用失败：超时')
  })

  it('不带 data.message 的合成帧只记警告，不把会话判死', () => {
    // sse_stream 合成的那些（流断了、回放到终态却没等到终态事件）data 是空的。
    // 把会话判死会让一个还能接着改稿的面板变成只能重开。
    const s = feed([
      ev(1, 'agent_started', { turn: 1 }),
      ev(2, 'agent_completed', { turn: 1 }),
      { seq: 3, run_id: SID, type: 'error', message: '事件流中断' },
    ])
    expect(s.status).toBe('idle')
    expect(s.error).toBeNull()
    expect(s.warnings).toEqual(['事件流中断'])
  })

  it('新一轮开始作废上一轮的客户端警告', () => {
    let s = feed([ev(1, 'agent_started', { turn: 1 }), ev(2, 'agent_completed', { turn: 1 })])
    s = applyAgentEvent(s, { seq: 3, run_id: SID, type: 'error', message: '连接重试中' })
    expect(s.warnings).toHaveLength(1)
    s = applyAgentEvent(s, ev(4, 'agent_started', { turn: 2 }))
    expect(s.warnings).toEqual([])
  })
})

describe('截图与正文', () => {
  it('agent_thinking 的轮数不回退', () => {
    const s = feed([
      ev(1, 'agent_thinking', { round: 3, tools: ['kb_search'] }),
      ev(2, 'agent_thinking', { round: 2, tools: [] }),
    ])
    expect(s.rounds).toBe(3)
  })

  it('agent_completed 带回权威的轮数与工具次数', () => {
    const s = feed([
      ev(1, 'agent_started', { turn: 1 }),
      ev(2, 'agent_completed', { turn: 1, rounds: 4, tool_calls: 6 }),
    ])
    expect(s.rounds).toBe(4)
    expect(s.toolCalls).toBe(6)
  })

  it('空 delta 不把正文推成 streaming', () => {
    const s = feed([ev(1, 'agent_started', { turn: 1 }), ev(2, 'delta', { text: '' })])
    expect(s.streaming).toBe(false)
  })
})

describe('快照是权威（不变量 3）', () => {
  it('快照覆盖流推出来的那一份数字', () => {
    const streamed = feed([
      ev(1, 'agent_started', { turn: 1 }),
      ev(2, 'agent_tool_call', { call_id: 'c1', name: 'kb_search' }),
      ev(3, 'delta', { text: '半截' }),
    ])
    // 流漏了终态帧：面板上是 running，库里其实已经写完了
    expect(streamed.status).toBe('running')

    const snap = mergeAgentSnapshot(
      streamed,
      detail({ status: 'idle', story: '完整的正文', rounds: 3, tool_calls: 5, turn: 1 }),
    )
    expect(snap.status).toBe('idle')
    expect(snap.story).toBe('完整的正文')
    expect(snap.rounds).toBe(3)
    expect(snap.toolCalls).toBe(5)
    expect(snap.streaming).toBe(false)
    // 卡片不动：elapsed_ms 只存在于事件里，清掉就再也找不回来了
    expect(snap.cards).toHaveLength(1)
  })

  it('options 里的旧会话不会覆盖当前会话', () => {
    const s = mergeAgentSnapshot(resetForSession('s2'), detail({ id: 's1' }))
    // mergeAgentSnapshot 是纯赋值（不判 id），判 id 是 store 的事——
    // 这里钉住它的行为，免得以后有人以为它会自己挡
    expect(s.sessionId).toBe('s1')
  })
})

describe('刷新后的对话重建', () => {
  it('上一版正文落在 messages 里，agent_started 把它腾出当前稿', () => {
    // 第二轮开跑前 store 已对过账：第一轮的稿子进了 messages
    let s = resetForSession(SID)
    s = mergeAgentSnapshot(
      s,
      detail({
        turn: 1,
        story: '第一版正文',
        messages: [{ seq: 1, turn: 1, role: 'assistant', content: '第一版正文', tool_calls: [] }],
      }),
    )
    s = applyAgentEvent(s, ev(1, 'agent_started', { turn: 2 }))
    expect(s.story).toBe('')
    expect(s.messages).toHaveLength(1)
    expect(buildTranscript(s).map((i) => i.kind)).toEqual(['assistant'])
  })

  it('刷新重建：工具调用来自 messages，卡片不重复', () => {
    // 实时流已经建过卡
    const live = feed([
      ev(1, 'agent_started', { turn: 1 }),
      ev(2, 'agent_tool_call', { call_id: 'c1', name: 'kb_search', args: { query: '想念' } }),
      ev(3, 'agent_tool_result', { call_id: 'c1', ok: true, summary: '6 条', elapsed_ms: 320 }),
    ])
    // 快照带来同一串对话（含 tool 消息）
    const snap = mergeAgentSnapshot(
      live,
      detail({
        turn: 1,
        messages: [
          { seq: 1, turn: 1, role: 'user', content: '一段评论', tool_calls: [] },
          {
            seq: 2,
            turn: 1,
            role: 'assistant',
            content: '',
            tool_calls: [{ id: 'c1', name: 'kb_search', arguments: '{"query":"想念"}' }],
          },
          { seq: 3, turn: 1, role: 'tool', content: '（6 条结果）', tool_calls: [], tool_call_id: 'c1', name: 'kb_search' },
          { seq: 4, turn: 1, role: 'assistant', content: '第一版正文', tool_calls: [] },
        ],
      }),
    )
    const items = buildTranscript(snap)
    expect(items.map((i) => i.kind)).toEqual(['user', 'tool', 'assistant'])
    const card = firstToolCard(items)
    // 卡片身份来自实时流，结果优先用事件里那份（它有耗时）
    expect(card.result?.elapsedMs).toBe(320)
    expect(card.args.query).toBe('想念')
  })

  it('从零刷新（没有实时事件）也能画出工具卡', () => {
    const snap = mergeAgentSnapshot(
      resetForSession(SID),
      detail({
        messages: [
          {
            seq: 1,
            turn: 1,
            role: 'assistant',
            content: '',
            tool_calls: [{ id: 'c1', name: 'kb_search', arguments: '{"query":"想念"}' }],
          },
          { seq: 2, turn: 1, role: 'tool', content: '查到的正文', tool_calls: [], tool_call_id: 'c1' },
        ],
      }),
    )
    const items = buildTranscript(snap)
    expect(items).toHaveLength(1)
    const card = firstToolCard(items)
    expect(card.name).toBe('kb_search')
    expect(card.result?.preview).toBe('查到的正文')
  })

  it('参数写坏的 JSON 不抛，卡片照画', () => {
    const snap = mergeAgentSnapshot(
      resetForSession(SID),
      detail({
        messages: [
          {
            seq: 1,
            turn: 1,
            role: 'assistant',
            content: '',
            tool_calls: [{ id: 'c1', name: 'kb_search', arguments: '{"query": "想念"' }],
          },
        ],
      }),
    )
    const card = firstToolCard(buildTranscript(snap))
    expect(card.args).toEqual({})
    // 原文还在，这是唯一的线索
    expect(card.rawArgs).toBe('{"query": "想念"')
  })

  it('代码生成的收尾指令不算用户说的话', () => {
    const snap = mergeAgentSnapshot(
      resetForSession(SID),
      detail({
        messages: [
          { seq: 1, turn: 2, role: 'user', content: '现在直接交稿', tool_calls: [], name: 'force_final' },
        ],
      }),
    )
    const item = firstItem(buildTranscript(snap))
    expect(item.kind === 'user' && item.generated).toBe(true)
  })
})

describe('换会话', () => {
  it('seq 从零重来，不沿用上一条会话的游标', () => {
    // 沿用游标的话，新会话的前 N 条事件会被当成重复全丢掉——界面一片空白
    // 而日志里什么事都没有。
    const old = feed([ev(1, 'agent_started', { turn: 1 }), ev(2, 'delta', { text: '甲' })])
    expect(old.lastSeq).toBe(2)

    const fresh = resetForSession('s2')
    expect(fresh.lastSeq).toBe(0)
    const started = applyAgentEvent(fresh, ev(1, 'agent_started', { turn: 1 }, 's2'))
    expect(started.status).toBe('running')
  })

  it('resetForSession 带快照时直接吃快照，不用等重放', () => {
    const s = resetForSession('s2', detail({ id: 's2', status: 'running', story: '正在写的' }))
    expect(s.sessionId).toBe('s2')
    expect(s.status).toBe('running')
    expect(s.story).toBe('正在写的')
  })
})
