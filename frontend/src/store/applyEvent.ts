/**
 * 事件 → 状态。**纯函数，无 React、无副作用、无时间依赖。**
 *
 * 所有流式正确性都集中在这一个文件里，因此它可以被单独测（见 test/applyEvent.test.ts）。
 * store 只负责「把返回值塞回去」，组件只负责渲染——中间没有任何隐藏状态。
 *
 * 三条不变量，任何改动都必须保住：
 *   1. **进度单调不减。** 后端已经保证了，但重连重放、乱序到达都可能打破它，
 *      前端再钳一次——进度条倒退是用户最先注意到的 bug。
 *   2. **seq 严格递增。** 重放/重连会送来已处理过的事件，必须按 seq 丢弃，
 *      否则评论会重复追加、正文会重复打印。
 *   3. **不修改入参。** 返回新对象，让 Zustand 的引用比较生效。
 */

import type {
  CleanStats,
  ClusterOut,
  CommentItem,
  EvidenceOut,
  LocalRunStatus,
  NodeKey,
  NodeSpec,
  ProfileOut,
  ReasoningStepOut,
  RunEvent,
  RunStatus,
  SectionKey,
  SectionOut,
  SourceKind,
  VideoOut,
} from '../api/types'

export type NodeState = 'pending' | 'running' | 'done' | 'failed'

/** 中栏时间线上的一条。`progress` / `progress` 类事件不产生条目（太吵）。 */
export interface TimelineItem {
  id: number
  node?: NodeKey | null
  kind: 'node' | 'milestone' | 'warning' | 'error'
  text: string
  at: number
  /** 里程碑带真实数字，用于「已获取 40 条评论」这类文案的高亮 */
  highlight?: string | null
}

export interface RunState {
  runId: string | null
  /**
   * 数据源种类。决定左栏怎么说话（「N 条评论」还是「手动文本 · N 条」）、
   * 条目右侧显示点赞数还是行号。
   *
   * 由 `startRun` 传入、由快照对账确认——事件流里**没有**这个信息，
   * 它是 per-run 的属性而不是逐帧的产物，塞进每一帧只是重复自己。
   */
  sourceKind: SourceKind
  status: LocalRunStatus
  /** 0–100 的整数。后端下发浮点，由 clampProgress 收口取整。 */
  progress: number
  lastSeq: number
  revision: number

  currentNode: NodeKey | null
  currentMessage: string | null
  /** 每个节点的状态。初始全 pending，由 node_started / node_completed 驱动。 */
  nodes: Record<string, NodeState>

  video: VideoOut | null
  comments: CommentItem[]
  /** 抓取期间的真实条数，与 comments.length 不同——后者可能被 limit 截断。 */
  commentsReceived: number
  commentStats: CleanStats | null
  clusters: ClusterOut[]
  clustersFinal: boolean
  /** n3 写进每个簇的 params 副本。算法、参数、是否退化为 KMeans 都在这里，
   *  中栏用它显示「这一簇是怎么分出来的」。缺省空对象，不是 null。 */
  clusterMeta: Record<string, unknown>
  profile: ProfileOut | null
  evidence: EvidenceOut[]
  reasoning: ReasoningStepOut[]
  sections: Partial<Record<SectionKey, SectionOut>>

  timeline: TimelineItem[]
  warnings: { code: string; message: string }[]
  error: { code: string; message: string } | null
  durationMs: number | null
  finishedAt: number | null

  /** 是否曾经断线重连过——用于在界面上标注「已恢复」而不是假装无事发生 */
  reconnected: boolean
}

export function initialState(): RunState {
  return {
    runId: null,
    sourceKind: 'douyin',
    // 「还没开始」而不是「排队中」。排队中意味着已有一条运行在等资源，
    // 用户应当看到一个会动的界面；空白页面不是那种状态。
    status: 'idle',
    progress: 0,
    lastSeq: 0,
    revision: 1,
    currentNode: null,
    currentMessage: null,
    nodes: {},
    video: null,
    comments: [],
    commentsReceived: 0,
    commentStats: null,
    clusters: [],
    clustersFinal: false,
    clusterMeta: {},
    profile: null,
    evidence: [],
    reasoning: [],
    sections: {},
    timeline: [],
    warnings: [],
    error: null,
    durationMs: null,
    finishedAt: null,
    reconnected: false,
  }
}

/** 新一次运行：清空产物但保留节点分带定义。 */
export function resetForRun(
  runId: string,
  nodeKeys: NodeKey[],
  sourceKind: SourceKind = 'douyin',
): RunState {
  const base = initialState()
  base.runId = runId
  // 必须在开跑前就定下来：左栏的副标题与空态文案在第一帧就要说对话，
  // 而快照对账要等到运行结束才发生。
  base.sourceKind = sourceKind
  base.status = 'running'
  for (const key of nodeKeys) base.nodes[key] = 'pending'
  base.timeline = [
    // id 用负数，避开事件 seq 的取值空间（从 1 开始）。用 1 的话，
    // 第一条真实事件（seq=1）会和时间线首项撞 key。
    { id: -1, kind: 'milestone', text: '已提交分析请求，正在排队…', at: Date.now() },
  ]
  return base
}

/**
 * 应用一个事件。
 *
 * `specs` 用于把节点完成文案里的占位符填上——文案模板由后端
 * `/meta/pipeline` 下发，前端不硬编码任何一句中文。
 */
export function applyEvent(state: RunState, event: RunEvent, specs?: NodeSpec[]): RunState {
  // 不变量 2：已处理过的事件直接丢弃。
  // 用 >= 而非 >：同一 seq 重复到达也要丢。
  if (event.seq <= state.lastSeq) return state

  const next: RunState = { ...state, lastSeq: event.seq }
  if (event.progress != null) next.progress = clampProgress(state.progress, event.progress)

  switch (event.type) {
    case 'run_started':
      next.status = 'running'
      next.revision = num(event.data?.revision, state.revision)
      return pushTimeline(next, {
        kind: 'milestone',
        node: null,
        text: event.message || '开始分析',
        at: tsOf(event),
      })

    case 'node_started':
      if (event.node) next.nodes = { ...state.nodes, [event.node]: 'running' }
      next.currentNode = event.node ?? state.currentNode
      next.currentMessage = event.message ?? null
      return pushTimeline(next, {
        kind: 'node',
        node: event.node,
        text: event.message || labelOf(specs, event.node),
        at: tsOf(event),
      })

    case 'node_completed':
      if (event.node) next.nodes = { ...state.nodes, [event.node]: 'done' }
      // 节点完成即「离开」它：currentNode 只在 node_started → node_completed
      // 之间有意义。不清掉的话，中栏会一直停在最后一个节点上。
      if (next.currentNode === event.node) {
        next.currentNode = null
        next.currentMessage = null
      }
      return pushTimeline(next, {
        kind: 'milestone',
        node: event.node,
        text: event.message || doneLabelOf(specs, event.node),
        at: tsOf(event),
      })

    case 'progress':
      // 纯进度事件不进时间线——每 2% 一条会把关键文案冲走。
      // 但它带 message 时是里程碑（后端用 progress 事件发带文案的节点内节点）。
      if (!event.message) return next
      next.currentMessage = event.message
      return pushTimeline(next, {
        kind: 'milestone',
        node: event.node,
        text: event.message,
        at: tsOf(event),
      })

    case 'partial':
      return applyPartial(next, event)

    case 'delta':
      return applyDelta(next, event)

    case 'section_updated':
      return applySectionUpdated(next, event)

    case 'warning': {
      const entry = { code: str(event.data?.code) ?? 'warning', message: event.message ?? '' }
      next.warnings = [...state.warnings, entry]
      return pushTimeline(next, {
        kind: 'warning',
        node: event.node,
        text: entry.message,
        at: tsOf(event),
      })
    }

    case 'error': {
      next.error = {
        code: str(event.data?.code) ?? 'error',
        message: event.message ?? str(event.data?.message) ?? '未知错误',
      }
      next.status = 'failed'
      return pushTimeline(next, {
        kind: 'error',
        node: event.node,
        text: next.error.message,
        at: tsOf(event),
      })
    }

    case 'run_completed':
      next.status = 'succeeded'
      next.progress = 100
      next.finishedAt = tsOf(event)
      next.durationMs = numOrNull(event.data?.duration_ms)
      return pushTimeline(next, {
        kind: 'milestone',
        node: null,
        text: event.message || '分析完成',
        at: tsOf(event),
      })

    case 'run_cancelled':
      next.status = 'cancelled'
      next.finishedAt = tsOf(event)
      return pushTimeline(next, {
        kind: 'milestone',
        node: null,
        text: event.message || '分析已取消',
        at: tsOf(event),
      })

    default:
      return next
  }
}

// ----------------------------------------------------------------------
// partial：结构化产物的增量
// ----------------------------------------------------------------------

function applyPartial(state: RunState, event: RunEvent): RunState {
  const data = event.data ?? {}
  const kind = str(data.kind)
  // 后端把产物字段平铺在 data 上，只额外加一个 `kind` 做判别
  // （见 graph/runner.py 的 _on_custom）。这里剥掉它，让下面的
  // 类型断言说的是真话——否则每个产物对象里都会多出一个 kind 字段。
  const payload = { ...data }
  delete payload.kind
  const next: RunState = { ...state }

  switch (kind) {
    case 'video':
      // 视频信息是整体到达的（一次请求拿全），直接替换
      next.video = payload as unknown as VideoOut
      return next

    case 'comment_page': {
      const page = (payload.items as CommentItem[] | undefined) ?? []
      // 去重而非直接拼接：分页游标可能重叠，且重连后的边界帧也会带来重复项。
      // 重复的评论会让左栏计数虚高，是那种「看起来没问题但数字不对」的 bug。
      const seen = new Set(state.comments.map((c) => c.comment_id))
      const fresh = page.filter((c) => c.comment_id && !seen.has(c.comment_id))
      next.comments = fresh.length ? [...state.comments, ...fresh] : state.comments
      next.commentsReceived = num(payload.received, state.commentsReceived + page.length)
      return next
    }

    case 'comment_stats':
      next.commentStats = payload as unknown as CleanStats
      return next

    case 'clusters':
      next.clusters = (payload.clusters as ClusterOut[] | undefined) ?? state.clusters
      // 元信息与簇一起到达（n3 在同一条 partial 里发），单独放行会让
      // 「算法/参数」在中栏缺一段——而「这簇是 KMeans 兜底出来的」正是
      // 用户判断结果可信度时最需要知道的一句。
      if (payload.cluster_meta && typeof payload.cluster_meta === 'object') {
        next.clusterMeta = payload.cluster_meta as Record<string, unknown>
      }
      next.clustersFinal = true
      return next

    case 'profile':
      // 整体替换而非合并：侧写是一次算完的（分布、张力、摘要互相关联），
      // 半个旧侧写配半个新侧写比没有更糟。
      next.profile = payload as unknown as ProfileOut
      return next

    case 'evidence': {
      // 两条契约都容忍：逐条到达（payload 就是证据本身）与整批到达
      // （payload.evidence 是数组）。M6 的逐条错峰发送用前者。
      const batch = payload.evidence as EvidenceOut[] | undefined
      next.evidence = batch ? [...state.evidence, ...batch] : [...state.evidence, payload as unknown as EvidenceOut]
      return next
    }

    case 'reasoning': {
      // 同 evidence：n6 一条一条发推理步，重连快照则整批给。
      const batch = payload.reasoning as ReasoningStepOut[] | undefined
      next.reasoning = batch
        ? [...state.reasoning, ...batch]
        : [...state.reasoning, payload as unknown as ReasoningStepOut]
      return next
    }

    case 'sections': {
      const sections = (payload.sections as SectionOut[] | undefined) ?? []
      next.sections = mergeSections(state.sections, sections)
      return next
    }

    default:
      return next
  }
}

// ----------------------------------------------------------------------
// delta：文本流式增量
// ----------------------------------------------------------------------

/**
 * 正文逐字输出。
 *
 * 这里**只做追加**，不做 markdown 解析或防抖——防抖放在 hook 层
 * （100ms flush）而不是 reducer 里，否则 reducer 就不再是纯函数了。
 */
function applyDelta(state: RunState, event: RunEvent): RunState {
  const data = event.data ?? {}
  const key = str(data.section) as SectionKey | null
  const text = str(data.text) ?? ''
  if (!key || !text) return state

  const existing = state.sections[key]
  return {
    ...state,
    sections: {
      ...state.sections,
      [key]: {
        key,
        title: existing?.title ?? '',
        content_md: (existing?.content_md ?? '') + text,
        citations: existing?.citations ?? [],
        version: existing?.version ?? 1,
        stale: existing?.stale ?? false,
        based_on_revision: existing?.based_on_revision ?? state.revision,
        edited_by_user: existing?.edited_by_user ?? false,
        citation_coverage: existing?.citation_coverage ?? 0,
        model: existing?.model ?? null,
      },
    },
  }
}

function applySectionUpdated(state: RunState, event: RunEvent): RunState {
  const section = event.data?.section as SectionOut | undefined
  if (!section?.key) return state
  return {
    ...state,
    sections: { ...state.sections, [section.key]: section },
  }
}

function mergeSections(
  prev: RunState['sections'],
  incoming: SectionOut[],
): RunState['sections'] {
  const out = { ...prev }
  for (const s of incoming) {
    if (s?.key) out[s.key] = s
  }
  return out
}

// ----------------------------------------------------------------------
// 快照对账
// ----------------------------------------------------------------------

/**
 * 用服务端快照校正流式期间累积的状态。
 *
 * **流负责实时性，快照负责真实性。** 流式过程中我们依赖事件的完整性；
 * 快照则是权威版本。收到 `run_completed` 后必须拉一次并覆盖，因为：
 *   - 重连时可能丢过几帧（服务端会补，但补不上的情况是存在的）；
 *   - 「流说 40 条评论，实际入库 38 条」这种偏差只有对账才能发现。
 *
 * 对账**不覆盖** progress 与 lastSeq——前者已经到 100 了，后者是游标。
 * 时间线也不覆盖：它是过程记录，快照里没有对应物。
 */
export function mergeSnapshot(state: RunState, detail: RunSnapshotLike): RunState {
  const sections: RunState['sections'] = { ...state.sections }
  for (const s of detail.sections ?? []) {
    if (s?.key) sections[s.key] = s
  }

  // **终态是单向门。**
  // 后端先发 `run_completed` 事件、再往库里写终态，中间隔着一次数据库往返。
  // 而前端正是在收到那个事件后才去拉快照对账——于是快照很容易落在这个窗口里，
  // 报回一个还没来得及更新的 `running`，把刚确立的 succeeded 打回去。
  // 症状是：进度 100%、过程记录写着「分析完成」，顶栏却还显示「分析中…」。
  const status = isTerminal(state.status) ? state.status : detail.status

  const nodes = { ...state.nodes }
  // 对账只发生在运行结束之后（终态流关闭，或刷新后打开一条已完成的运行）。
  // 因此绝大多数情况下七个节点应当整体收敛：还停在 pending 的说明是
  // 刷新后从快照恢复的，而快照没有节点级信息。
  const finished = status === 'succeeded' || status === 'cancelled'
  for (const key of Object.keys(nodes)) {
    if (finished) nodes[key] = 'done'
    else if (nodes[key] === 'running') nodes[key] = 'failed'
  }

  return {
    ...state,
    status,
    // 快照是权威版本：`?? state.sourceKind` 让老运行（字段缺席）保持原值，
    // 而不是被 undefined 冲掉。
    sourceKind: detail.source_kind ?? state.sourceKind,
    revision: detail.revision,
    // 从快照恢复到进度：刷新后如果只靠流重放，进度条会先归零再爬回来。
    // 这里先落到权威值，再由 clampProgress 保证它只增不减。
    progress:
      detail.progress == null ? state.progress : clampProgress(state.progress, detail.progress),
    nodes,
    currentNode: null,
    currentMessage: null,
    video: detail.video ?? state.video,
    comments: detail.comments ?? state.comments,
    commentsReceived: detail.comments?.length ?? state.commentsReceived,
    commentStats: detail.comment_stats ?? state.commentStats,
    clusters: detail.clusters ?? state.clusters,
    clustersFinal: true,
    clusterMeta: detail.cluster_meta ?? state.clusterMeta,
    // `?? state.x` 而不是直接赋值：快照里这个字段可能缺席（老运行、
    // 或产物根本没生成），缺席时把流里已经拿到的东西抹掉是纯倒退。
    profile: detail.profile ?? state.profile,
    evidence: detail.evidence ?? state.evidence,
    reasoning: detail.reasoning ?? state.reasoning,
    sections,
    warnings: detail.warnings?.length ? detail.warnings : state.warnings,
    error: detail.error ?? state.error,
    durationMs: detail.duration_ms ?? state.durationMs,
    finishedAt: state.finishedAt ?? Date.now(),
  }
}

/** `mergeSnapshot` 需要的字段子集——用结构类型而不是直接依赖 RunDetail，
 *  这样单测可以只构造关心的字段。 */
export interface RunSnapshotLike {
  status: RunStatus
  source_kind?: SourceKind | null
  revision: number
  progress?: number | null
  duration_ms?: number | null
  video?: VideoOut | null
  comments?: CommentItem[]
  comment_stats?: CleanStats
  clusters?: ClusterOut[]
  cluster_meta?: Record<string, unknown>
  profile?: ProfileOut | null
  evidence?: EvidenceOut[]
  reasoning?: ReasoningStepOut[]
  sections?: SectionOut[]
  warnings?: { code: string; message: string }[]
  error?: { code: string; message: string } | null
}

// ----------------------------------------------------------------------
// 工具
// ----------------------------------------------------------------------

/**
 * 不变量 1：进度绝不回退。
 *
 * 后端已经钳过一次，这里再钳一次不是冗余——重连重放、网络乱序、
 * 以及「快照对账后把 progress 从 100 改回 87」都属于后端的保证覆盖不到的路径。
 */
function clampProgress(current: number, incoming: number): number {
  const value = Number.isFinite(incoming) ? incoming : current
  // 在 reducer 里取整一次，而不是让每个展示点各取各的：
  // 后端下发的是 3.8 这种浮点，顶栏直接渲染就成了「3.8%」。
  // 收口在这里，`progress` 在全局就是 0–100 的整数。
  return Math.round(Math.max(current, Math.min(100, value)))
}

const TERMINAL_STATUSES: ReadonlySet<LocalRunStatus> = new Set([
  'succeeded',
  'failed',
  'cancelled',
])

export function isTerminal(status: LocalRunStatus): boolean {
  return TERMINAL_STATUSES.has(status)
}

const MAX_TIMELINE = 200

function pushTimeline(state: RunState, item: Omit<TimelineItem, 'id'>): RunState {
  const id = state.lastSeq
  const timeline = [...state.timeline, { ...item, id }]
  // 上限而非无限增长：中栏是「过程」，不是可回滚的历史
  return { ...state, timeline: timeline.slice(-MAX_TIMELINE) }
}

function labelOf(specs: NodeSpec[] | undefined, node: NodeKey | null | undefined): string {
  if (!node) return ''
  return specs?.find((s) => s.key === node)?.entering ?? node
}

function doneLabelOf(specs: NodeSpec[] | undefined, node: NodeKey | null | undefined): string {
  if (!node) return '完成'
  // done 文案可能含 `{count}` 这类占位符——它由节点自己发的里程碑
  // 带真实数字填充。这里拿不到数字时把占位符去掉，而不是把 `{count}`
  // 原样显示给用户。
  const raw = specs?.find((s) => s.key === node)?.done ?? '完成'
  return raw.replace(/\{[^}]+\}/g, '').replace(/\s{2,}/g, ' ').trim()
}

function num(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback
}

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

function str(v: unknown): string | null {
  return typeof v === 'string' && v ? v : null
}

function tsOf(event: RunEvent): number {
  return event.ts ? event.ts * 1000 : Date.now()
}
