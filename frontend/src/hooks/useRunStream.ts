/**
 * 把 SSE 流接到 store 上。
 *
 * 职责边界：hook 管「连接的生命周期」，store 管「事件变成什么状态」。
 * hook 里不做任何状态推导。
 *
 * 一件事值得说明——**为什么终态后要再拉一次快照**：
 * 流是「边跑边报」的，它的完整性依赖每一帧都送达。快照是权威版本。
 * 两者对不上时以快照为准。这不是防御性编程，是必要的：
 * 重连边界、进程重启、事件表回放与 Stream 的差异都会造成微小偏差，
 * 而用户看到的数字必须和数据库里的一致。
 */

import { useCallback, useEffect, useRef } from 'react'

import { api } from '../api/endpoints'
import { openRunStream, type StreamHandle } from '../api/sse'
import { clearActiveRun } from '../store/activeRun'
import { isTerminal } from '../store/applyEvent'
import { useRunStore } from '../store/runStore'
import type { SourceKind } from '../api/types'

export interface RunStreamControls {
  /**
   * `since` 默认 0 = 重放这条运行的全部历史。
   *
   * 刷新后恢复**故意用 0 而不是上次的 seq**：重放历史能让节点清单、
   * 过程记录、评论分页全部按原样重建，而「从半路接上」会留下列表里
   * 从未出现过前半段的事件流残缺。重放的开销就是跑几百次 reducer，
   * 而它换来的是「刷新后看到的和没刷新时完全一致」。
   */
  start: (
    runId: string,
    opts?: { since?: number; progress?: number; sourceKind?: SourceKind },
  ) => void
  stop: () => void
  isStreaming: () => boolean
  lastSeq: () => number
}

export function useRunStream(): RunStreamControls {
  const handleRef = useRef<StreamHandle | null>(null)
  const runIdRef = useRef<string | null>(null)
  // 快照对账是异步的，而流可能已经关闭——用一个代次号丢弃过期响应，
  // 否则「上一次运行的快照」会覆盖「这一次运行的状态」。
  const genRef = useRef(0)

  const stop = useCallback(() => {
    handleRef.current?.close()
    handleRef.current = null
    runIdRef.current = null
    genRef.current += 1
  }, [])

  const start = useCallback(
    (runId: string, opts?: { since?: number; progress?: number; sourceKind?: SourceKind }) => {
      stop()
      runIdRef.current = runId
      const generation = genRef.current

      const store = useRunStore.getState()
      store.startRun(runId, opts?.progress ?? 0, opts?.sourceKind ?? 'douyin')

    handleRef.current = openRunStream({
      runId,
      since: opts?.since ?? 0,
      onEvent: (event) => {
        if (genRef.current !== generation) return
        useRunStore.getState().ingest(event)
      },
      onOpen: (reconnected) => {
        if (genRef.current !== generation) return
        if (reconnected) useRunStore.getState().markReconnected()
      },
      onClose: (reason) => {
        if (genRef.current !== generation) return
        if (reason !== 'terminal') {
          // `ended`：服务端关流但没送终态事件。多数时候意味着这次运行在
          // 服务端已经是终态了——进程重启把任务丢了、或者那条终态事件刚好
          // 没送出来。拉一次快照看看，**只在快照是终态时采用**：把一条还在
          // 跑的运行对账进来，会把正在跑的节点标成 failed。
          void api
            .getRun(runId)
            .then((detail) => {
              if (genRef.current !== generation) return
              if (isTerminal(detail.status)) useRunStore.getState().reconcile(detail)
            })
            .catch(() => undefined)
          return
        }
        // 跑完了就不再需要「刷新后恢复这条运行」——快照会成为权威来源，
        // 页面重开时也不该自动把用户拽回上一次的旧结果。
        clearActiveRun()
        // 终态：拉一次权威快照对账
        void api
          .getRun(runId)
          .then((detail) => {
            if (genRef.current !== generation) return
            useRunStore.getState().reconcile(detail)
          })
          .catch(() => {
            // 对账失败不影响已渲染的内容——流已经给出了完整结果。
            // 不弹错误提示：这不是用户需要处理的问题。
          })
      },
      onError: (err) => {
        if (genRef.current !== generation) return
        // 走 noteWarning 而不是 ingest：伪造一个 seq 会让重连后真正的
        // 那一帧被当成重复丢弃，流就永久缺一帧。
        useRunStore.getState().noteWarning('stream_error', `事件流中断：${err.message}`)
      },
    })
    },
    [stop],
  )

  // 组件卸载时断开——否则切走页面后连接会一直挂着，
  // 而且 onEvent 还会往一个已卸载的 store 里写
  useEffect(() => stop, [stop])

  return {
    start,
    stop,
    isStreaming: () => handleRef.current !== null,
    lastSeq: () => handleRef.current?.lastSeq() ?? 0,
  }
}
