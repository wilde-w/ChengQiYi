/**
 * 把故事工坊的 SSE 接到 store 上。`useRunStream` 的同款，两处**故意不同**：
 *
 * 1. **终态后不关流**（`keepOpenOnTerminal`）。流水线跑完就结束了；一场对话
 *    写完一稿还在待命，用户随时会说「再暗一点」。关掉连接的话，`sse.ts` 会
 *    按「服务端关流但没送终态」处理并退避重连——于是页面每 5 秒重连一次，
 *    永远如此。后端那边让待命会话保持连接，正是为了让这条连接一直有用。
 * 2. **每一轮的终态都要对账**（不只是会话结束）。正文是逐字拼出来的，而
 *    轮次/工具次数只有快照里才有权威值——面板上那个「3 轮 · 4 次工具调用」
 *    必须和库里一致。
 */

import { useCallback, useEffect, useMemo, useRef } from 'react'

import { api } from '../api/endpoints'
import { openRunStream, type StreamHandle } from '../api/sse'
import { useAgentStore } from '../store/agentStore'

/** 一轮的三种结局。收到之后拉一次快照。 */
const TURN_END = new Set(['agent_completed', 'agent_cancelled', 'error'])

export interface AgentStreamControls {
  /** `since` 默认 0：刷新重建靠**重放全部历史**，而不是从半路接上。 */
  start: (sessionId: string, opts?: { since?: number }) => void
  stop: () => void
  isStreaming: () => boolean
}

export function useAgentStream(): AgentStreamControls {
  const handleRef = useRef<StreamHandle | null>(null)
  // 对账是异步的，而期间用户可能已经换了一条会话——用代次号丢弃过期响应，
  // 否则「上一条会话的快照」会盖住「这一条」。
  const genRef = useRef(0)

  const stop = useCallback(() => {
    handleRef.current?.close()
    handleRef.current = null
    genRef.current += 1
  }, [])

  const reconcile = useCallback(async (sessionId: string, generation: number) => {
    try {
      const detail = await api.getAgentSession(sessionId)
      if (genRef.current !== generation) return
      useAgentStore.getState().reconcile(detail)
    } catch {
      // 对账失败不影响已经渲染出来的内容——流已经把结果给全了。
      // 不弹提示：这不是用户需要处理的问题。
    }
  }, [])

  const start = useCallback(
    (sessionId: string, opts?: { since?: number }) => {
      stop()
      const generation = genRef.current
      // 直接开流（没有先拉快照）也要有会话 id，否则第一帧事件无处可落。
      if (!useAgentStore.getState().sessionId) {
        useAgentStore.getState().openSession(sessionId)
      }

      handleRef.current = openRunStream({
        runId: sessionId,
        path: '/api/v1/agent/sessions/{id}/events',
        since: opts?.since ?? 0,
        keepOpenOnTerminal: true,
        onEvent: (event) => {
          if (genRef.current !== generation) return
          useAgentStore.getState().ingest(event)
          if (TURN_END.has(event.type)) void reconcile(sessionId, generation)
        },
        onClose: (reason) => {
          if (genRef.current !== generation) return
          // `terminal`：这一轮/这条会话结束时 onEvent 已经对过账了，不必再来一次。
          if (reason === 'terminal') return
          // `ended`：服务端关流但没送终态。多数时候意味着会话在服务端已经是
          // 终态了——进程被杀留下的僵尸被 `reap_stale_agent_sessions` 收掉，
          // 而那条终态事件根本没进过事件表。**只在快照是终态时采用**：
          // 把一条还在跑的会话对账进来，会把正在写的稿子标成失败。
          void api
            .getAgentSession(sessionId)
            .then((detail) => {
              if (genRef.current !== generation) return
              if (detail.is_terminal) useAgentStore.getState().reconcile(detail)
            })
            .catch(() => undefined)
        },
        onError: (err) => {
          if (genRef.current !== generation) return
          // 走 noteWarning 而不是 ingest：伪造一个 seq 会让重连后真正的
          // 那一帧被当成重复丢弃，流就永久缺一帧。
          useAgentStore.getState().noteWarning(`事件流中断：${err.message}`)
        },
      })
    },
    [reconcile, stop],
  )

  // 面板关掉时断开——否则连接会一直挂着，而后端会一直为它发心跳
  useEffect(() => stop, [stop])

  const isStreaming = useCallback(() => handleRef.current !== null, [])

  // **必须是稳定引用。** 不包 memo 的话每次渲染都是一个新对象，而调用方
  // 理所当然地会把它写进依赖数组（`useEffect(..., [open, stream])`）——
  // 于是那个 effect 每次渲染都重跑，它的清理函数把 `alive` 置成 false，
  // 正在 await 的「刷新后接回上一场」刚好走到 `if (!alive) return` 就悄悄
  // 退出了，而 `resumedRef` 已经置真、不会重试。症状是刷新之后面板一片空白，
  // 控制台干干净净。（`useRunStream` 有同样的形状，它没炸只是因为那边的
  // 调用方没有 alive 守卫——不是它做对了。）
  return useMemo(() => ({ start, stop, isStreaming }), [start, stop, isStreaming])
}
