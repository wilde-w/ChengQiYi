/**
 * 故事工坊的状态。
 *
 * 与 `runStore` 同一个模子：store **不含任何业务逻辑**，全部委托给
 * `applyAgentEvent` 里的纯函数——流式正确性因此可以被单测覆盖，而这里只剩
 * 「存值 + 通知」。
 *
 * 单独一个 store 而不是往 `runStore` 里塞一个字段：两者会被同时打开（一边
 * 跑着分析、一边让写手改稿），共用一个 store 的结果是任何一次 `reset`
 * 都会顺手把对方清空，而这种 bug 只在「同时用」时才出现。
 */

import { create } from 'zustand'

import type { AgentCapabilities, AgentSessionDetail, RunEvent } from '../api/types'
import {
  agentInitialState,
  applyAgentEvent,
  mergeAgentSnapshot,
  noteAgentWarning,
  resetForSession,
  type AgentState,
} from './applyAgentEvent'

export interface AgentStore extends AgentState {
  /** 打开面板时抓一次。中文档位与库清单都来自后端，前端不硬编码。 */
  capabilities: AgentCapabilities | null

  setCapabilities: (capabilities: AgentCapabilities) => void
  /** 开一条新会话（或刷新后重建某一条）。**必须走这里**，它会重置 seq 游标。 */
  openSession: (sessionId: string, detail?: AgentSessionDetail | null) => void
  ingest: (event: RunEvent) => void
  /** 快照对账。拿到的总是**权威版本**。 */
  reconcile: (detail: AgentSessionDetail) => void
  noteWarning: (message: string) => void
  reset: () => void
}

export const useAgentStore = create<AgentStore>((set, get) => ({
  ...agentInitialState(),
  capabilities: null,

  setCapabilities: (capabilities) => set({ capabilities }),

  openSession: (sessionId, detail) => set(resetForSession(sessionId, detail)),

  ingest: (event) => {
    const state = get()
    // 上一条会话的迟到事件（切换竞态）不能污染这一条：两条会话的 seq 各自
    // 从 1 起，混进来的代价不只是多一帧——它会推进游标，把真正属于这条会话的
    // 前几条事件当成重复丢掉。
    if (state.sessionId && event.run_id && event.run_id !== state.sessionId) return
    set(applyAgentEvent(state, event))
  },

  reconcile: (detail) => {
    const state = get()
    if (state.sessionId && detail.id !== state.sessionId) return
    // 快照先到（刷新重建走的就是这条路）：先按它的 id 重置游标，再吃它的内容。
    const base = state.sessionId ? state : resetForSession(detail.id)
    set(mergeAgentSnapshot(base, detail))
  },

  noteWarning: (message) => set((s) => noteAgentWarning(s, message)),

  reset: () => set((s) => ({ ...agentInitialState(), capabilities: s.capabilities })),
}))

// ----------------------------------------------------------------------
// 选择器
// ----------------------------------------------------------------------

export const selectAgentRunning = (s: AgentStore) => s.status === 'running'
export const selectAgentStory = (s: AgentStore) => s.story
export const selectAgentMessages = (s: AgentStore) => s.messages
export const selectAgentCards = (s: AgentStore) => s.cards
