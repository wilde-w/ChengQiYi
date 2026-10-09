/**
 * 运行状态。
 *
 * **为什么是 Zustand 而不是 Context。** 这个工作台在流式期间每秒会推
 * 几十个事件；Context 的值一变，所有消费者都重渲染——包括正在画
 * ECharts 的环形图和推理图。Zustand 的选择器订阅让每个组件只在自己
 * 关心的切片变化时重渲染：正文逐字输出时，图表面板纹丝不动。
 *
 * store 本身**不含任何业务逻辑**——全部委托给 applyEvent 里的纯函数。
 * 这样流式正确性可以被单测覆盖，而 store 只剩「存值 + 通知」。
 */

import { create } from 'zustand'

import type { NodeKey, NodeSpec, RunDetail, RunEvent, SourceKind } from '../api/types'
import {
  applyEvent,
  initialState,
  mergeSnapshot,
  resetForRun,
  type NodeState,
  type RunState,
} from './applyEvent'

export interface RunStore extends RunState {
  /** 后端下发的进度分带。中文文案的唯一来源，启动时抓一次。 */
  specs: NodeSpec[]

  setSpecs: (specs: NodeSpec[]) => void
  /**
   * `seedProgress` 只有「刷新后恢复」才用得上：重放历史时进度会从 0 重新爬，
   * 而用户刷新前看到的已经是 45%，先落到快照给的值再让重放往上顶，
   * 进度条就不会当着用户的面倒退一截。
   */
  startRun: (runId: string, seedProgress?: number, sourceKind?: SourceKind) => void
  ingest: (event: RunEvent) => void
  /** 客户端侧的问题（连接断开等）。**不占用 seq**——占用了会让真正的
   *  那一帧在重连后被当成重复而丢弃。 */
  noteWarning: (code: string, message: string) => void
  reconcile: (detail: RunDetail) => void
  markReconnected: () => void
  reset: () => void
}

const NODE_KEYS: NodeKey[] = [
  'n1_video',
  'n2_comments',
  'n3_cluster',
  'n4_psych',
  'n5_retrieval',
  'n6_reasoning',
  'n7_literary',
]

/**
 * 初始与复位共用的节点表：全部 pending，让中栏在开跑前就把七步渲染出来。
 *
 * 注意**不能**用 `resetForRun('')` 当初始值——它会把 status 置成 `running`，
 * 于是空白页面的顶栏显示「分析中…」、按钮置灰，而三栏都写着「等待中」。
 */
const PENDING_NODES: Record<string, NodeState> = Object.fromEntries(
  NODE_KEYS.map((k) => [k, 'pending']),
)

export const useRunStore = create<RunStore>((set, get) => ({
  ...initialState(),
  nodes: { ...PENDING_NODES },
  specs: [],

  setSpecs: (specs) => set({ specs }),

  startRun: (runId, seedProgress = 0, sourceKind = 'douyin') =>
    set({
      ...resetForRun(runId, NODE_KEYS, sourceKind),
      progress: Number.isFinite(seedProgress)
        ? Math.min(100, Math.max(0, Math.round(seedProgress)))
        : 0,
    }),

  ingest: (event) => {
    const state = get()
    // 上一个运行的迟到事件（重连竞态）不能污染新运行的状态
    if (state.runId && event.run_id && event.run_id !== state.runId) return
    set(applyEvent(state, event, state.specs))
  },

  noteWarning: (code, message) =>
    set((s) => ({
      warnings: [...s.warnings, { code, message }],
      timeline: [
        ...s.timeline,
        { id: -Date.now(), kind: 'warning' as const, node: null, text: message, at: Date.now() },
      ].slice(-200),
    })),

  reconcile: (detail) => set(mergeSnapshot(get(), detail)),

  markReconnected: () => set({ reconnected: true }),

  reset: () => set((s) => ({ ...initialState(), nodes: { ...PENDING_NODES }, specs: s.specs })),
}))

// ----------------------------------------------------------------------
// 选择器
// ----------------------------------------------------------------------

/**
 * 中栏各节点的状态。放在这里而不是组件里，是为了让「哪些节点已完成」
 * 只算一次——三个栏都要用它来判断自己的完成度。
 */
export const selectNodeStates = (s: RunStore) => s.nodes
export const selectProgress = (s: RunStore) => s.progress
export const selectVideo = (s: RunStore) => s.video
export const selectComments = (s: RunStore) => s.comments
export const selectCommentStats = (s: RunStore) => s.commentStats
export const selectClusters = (s: RunStore) => s.clusters
export const selectClusterMeta = (s: RunStore) => s.clusterMeta
export const selectProfile = (s: RunStore) => s.profile
export const selectEvidence = (s: RunStore) => s.evidence
export const selectReasoning = (s: RunStore) => s.reasoning
export const selectSections = (s: RunStore) => s.sections
export const selectTimeline = (s: RunStore) => s.timeline
export const selectSourceKind = (s: RunStore) => s.sourceKind
export const selectSpecs = (s: RunStore) => s.specs

/**
 * 三栏各自的「进行中 / 完成 / 失败」。
 *
 * 由节点状态派生而非各栏自己维护——避免出现「左栏说完成、顶栏还在转」
 * 这种自相矛盾的界面。左栏只关心 n1/n2，中栏 n3–n6，右栏 n7。
 */
export function columnState(
  spec: { nodes: Record<string, NodeState | undefined>; status: RunState['status'] },
  key: 'left' | 'middle' | 'right',
): { running: boolean; failed: boolean; done: boolean } {
  const owned: NodeKey[] =
    key === 'left'
      ? ['n1_video', 'n2_comments']
      : key === 'middle'
        ? ['n3_cluster', 'n4_psych', 'n5_retrieval', 'n6_reasoning']
        : ['n7_literary']
  const states = owned.map((k) => spec.nodes[k])
  const anyRunning = states.some((s) => s === 'running')
  const allDone = states.every((s) => s === 'done')
  return {
    running: anyRunning,
    failed: spec.status === 'failed',
    done: allDone && spec.status !== 'failed',
  }
}
