/**
 * 与后端 DTO 一一对应的类型。
 *
 * 这些类型是**手写镜像**而非从 OpenAPI 生成：生成器在处理
 * `RunDetail` 这种「一个大对象装下全部产物」的响应时会产出难用的
 * 交叉类型，而手写的这一份顺便承担了「前端到底依赖哪些字段」的文档职责。
 * 代价是后端改字段时要同步改这里——所以字段名一律照抄，不做重命名。
 */

export type Depth = 'quick' | 'standard' | 'deep'
export type LibraryKey = 'psychology' | 'literature' | 'poetry'
export type NodeKey =
  | 'n1_video'
  | 'n2_comments'
  | 'n3_cluster'
  | 'n4_psych'
  | 'n5_retrieval'
  | 'n6_reasoning'
  | 'n7_literary'
export type SectionKey = 'profile' | 'mechanism' | 'allusion' | 'insight'
export type RunStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'
/**
 * 这次分析的数据源：抖音视频，还是用户粘进来的一段文本。
 *
 * 由后端下发（`RunSummary.source_kind`），前端**不要**自己去推
 * `aweme_id == null`——那条隐式规则只存在于后端一处。
 */
export type SourceKind = 'douyin' | 'text'
export type TextMode = 'line' | 'paragraph'

/**
 * 客户端状态 = 服务端状态 + `idle`。
 *
 * `idle` 表示「还没提交过任何运行」，是前端专有状态：服务端永远不会下发它。
 * 这个区分是必要的——早先拿 `running` 当初始值，结果空白页面的顶栏显示
 * 「分析中…」且按钮置灰，而三栏都写着「等待中」。
 */
export type LocalRunStatus = RunStatus | 'idle'

export type EventType =
  | 'run_started'
  | 'node_started'
  | 'node_completed'
  | 'progress'
  | 'partial'
  | 'delta'
  | 'section_updated'
  | 'intervention'
  | 'warning'
  | 'error'
  | 'run_completed'
  | 'run_cancelled'
  | 'heartbeat'
  /* 故事工坊。与上面共用同一个信封，但**走另一条通道**（命名空间 `agent`、
     另一张事件表），两边的 seq 各自计数。见 api/sse.ts 的 TERMINAL。 */
  | 'agent_started'
  | 'agent_thinking'
  | 'agent_tool_call'
  | 'agent_tool_result'
  | 'agent_completed'
  | 'agent_cancelled'

/** `partial` 事件的 data.kind——指明这次增量是哪种产物。 */
export type PartialKind =
  | 'video'
  | 'comment_page'
  | 'comment_stats'
  | 'clusters'
  | 'profile'
  | 'evidence'
  | 'reasoning'
  | 'sections'

/** SSE 事件的统一信封。与后端 graph/bus.py 的 RunEvent 对齐。 */
export interface RunEvent {
  seq: number
  run_id: string
  type: EventType
  node?: NodeKey | null
  progress?: number | null
  message?: string | null
  data?: Record<string, unknown>
  ts?: number
}

export interface Providers {
  llm?: string
  embedding?: string
  douyin?: string
  mock_mode?: string
  is_demo?: boolean
}

export interface RunSummary {
  id: string
  status: RunStatus
  input_raw: string
  aweme_id?: string | null
  /** 缺失时按 douyin 处理（老运行）：这是后端 `source_kind_of()` 的同一套回落。 */
  source_kind?: SourceKind
  depth: Depth
  progress: number
  current_node?: NodeKey | null
  revision: number
  providers: Providers
  warnings: { code: string; message: string }[]
  error?: { code: string; message: string } | null
  duration_ms?: number | null
  created_at: string
  started_at?: string | null
  finished_at?: string | null
}

export interface VideoOut {
  aweme_id: string
  title?: string | null
  caption?: string | null
  author_name?: string | null
  author_avatar?: string | null
  publish_time?: string | null
  duration_ms?: number | null
  cover_url?: string | null
  share_url?: string | null
  stats: Record<string, number>
  transcript?: { text?: string; source?: string } | null
  transcript_source?: string | null
}

export interface CommentItem {
  /** 抓取期间来自 douyin_comment_id，落库后是行 id。两者都可作为 key。 */
  comment_id: string
  text: string
  author_name?: string | null
  like_count: number
  reply_count: number
  publish_time?: string | null
  is_ad?: boolean
  is_spam?: boolean
  is_duplicate?: boolean
  filter_reason?: string | null
  /** 快照接口会带上（来自 cluster 关联），流式阶段还没有。 */
  cluster_key?: string | null
}

export interface CleanStats {
  total: number
  kept: number
  ads: number
  spam: number
  duplicates: number
  empty: number
  reasons: Record<string, number>
}

export interface ClusterOut {
  cluster_key: string
  label: string
  summary?: string | null
  size: number
  raw_size: number
  is_noise: boolean
  emotion_tags: string[]
  topic_tags: string[]
  need_tags: string[]
  keywords: string[]
  color: string
  order_index: number
}

export interface EvidenceOut {
  id: string
  kind: LibraryKey
  library: string
  chunk_id: string
  title?: string | null
  source?: string | null
  author?: string | null
  text: string
  score: number
  retrieval_path: 'vector' | 'graph'
  match_reason?: string | null
  cluster_ids: string[]
  pinned: boolean
  excluded: boolean
  rank: number
  payload: Record<string, unknown>
}

export interface ReasoningStepOut {
  step_index: number
  phenomenon: string
  mechanism: string
  insight: string
  evidence_ids: string[]
  allusion_ids: string[]
  confidence: number
}

/** 正文里一枚脚注记号对应的一条引用。`text` 是 hover 时显示的出处。 */
export interface CitationOut {
  evidence_id: string
  /** 规范化后的脚注记号，与 `content_md` 里出现的那串逐字相同。 */
  marker: string
  /** `《标题》· 作者`，由后端拼好——chunk_id 不是给人读的。 */
  text?: string
  library?: string
}

export interface SectionOut {
  key: SectionKey
  title: string
  content_md: string
  citations: CitationOut[]
  version: number
  stale: boolean
  based_on_revision: number
  edited_by_user: boolean
  citation_coverage: number
  model?: string | null
}

export interface RunDetail extends RunSummary {
  video?: VideoOut | null
  comments: CommentItem[]
  comment_stats: CleanStats
  clusters: ClusterOut[]
  cluster_meta: Record<string, unknown>
  profile?: ProfileOut | null
  evidence: EvidenceOut[]
  reasoning: ReasoningStepOut[]
  sections: SectionOut[]
  stale: Record<string, boolean>
}

export interface ProfileOut {
  emotions: { label: string; value: number; cluster_key?: string }[]
  topics: { label: string; value: number; cluster_key?: string }[]
  needs: { label: string; value: number; cluster_key?: string }[]
  /** 按维度分组：emotion / topic / need 三键，可能缺键。 */
  global_tags: Record<string, string[]>
  cluster_tags: Record<string, unknown>
  core_tension?: string | null
  summary?: string | null
  /**
   * 产出这份侧写的模型名。`fallback` = 模型没参与，这份是各主题标签的汇总；
   * `none` = 没有簇所以没有侧写。前端据此挂「由标签汇总」的牌子——降级出来
   * 的侧写和正常侧写长得一模一样，不说的话没人分得出来。
   */
  model?: string | null
}

/** `GET /meta/pipeline` 下发的进度分带。中文文案的唯一来源在后端。 */
export interface NodeSpec {
  key: NodeKey
  label: string
  start: number
  end: number
  entering: string
  done: string
}

export interface CreateRunResponse {
  run: RunSummary
  resolved: { aweme_id: string; canonical_url: string; resolved_via: string }
  demo: boolean
}

export interface Capabilities {
  is_demo: boolean
  providers: Providers
  douyin_modes: string[]
  depth_levels: Depth[]
}

/* ------------------------------------------------------------------ *
 * 知识库导入
 *
 * 与后端 app/schemas/kb.py 一一对应。字段名照抄，不做重命名。
 * ------------------------------------------------------------------ */

/** 导入作业的状态复用运行状态那套枚举——两边是同一组语义。 */
export type ImportStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'

export interface KbImportJob {
  id: string
  library: LibraryKey
  library_label: string
  filename: string
  file_size: number
  status: ImportStatus
  stage?: string | null
  /** 阶段的中文名由后端下发。前端不维护第二份映射——两处文案迟早会漂移。 */
  stage_label?: string | null
  progress: number
  message?: string | null
  chunk_total: number
  chunk_tagged: number
  chunk_indexed: number
  warnings: string[]
  error?: string | null
  /** 终态判定由后端给。前端自己判就多了一处「状态枚举改了但这里没改」。 */
  is_terminal: boolean
  /** 停在 queued 等用户确认（文件太大）。 */
  needs_confirm: boolean
  created_at: string
  started_at?: string | null
  finished_at?: string | null
}

export interface KbImportCreated {
  job: KbImportJob
  auto_started: boolean
  warnings: string[]
  demo: boolean
}

export interface KbSource {
  source_file: string
  library: LibraryKey
  library_label: string
  chunks: number
  updated_at?: string | null
}

export interface KbSourceDocument {
  chunk_id: string
  library: string
  source_file?: string | null
  text_preview: string
  updated_at?: string | null
}

export interface KbDeleteResult {
  source_file: string
  removed: number
  message: string
}

/**
 * 某个来源的**原文**（不是切块预览）。
 *
 * 一次不一定给全：整本红楼梦上百万字，一次性塞进 DOM 会卡，所以按字符分页。
 * 「还有多少没加载」用 `char_count - (offset + text.length)` 算，
 * **不要**用 `truncated` 反推位置——它只说「后面还有」。
 */
export interface KbSourceText {
  source_file: string
  title?: string | null
  author?: string | null
  library: LibraryKey
  /** 全文长度（不是本段的） */
  char_count: number
  offset: number
  truncated: boolean
  text: string
}

/* ------------------------------------------------------------------ *
 * 古典文学 MCP（另一个仓的知识库）
 *
 * 与「观心知识库导入」是两回事：那边是我们导入的内容，这边是
 * `ClassicalNovelProject` 那个仓的红楼梦库，经 MCP 只读查阅。
 * 两边都不混合——不写进观心的 Qdrant/Neo4j，也不进分析流水线。
 * ------------------------------------------------------------------ */

export interface NovelText {
  /** **对方**的工具名（`novel_*`）。透出来是为了排错：同样一句
   *  「取不到原文」，调 `novel_get_chapter_text` 和调 `novel_list_chapters`
   *  是两件完全不同的事。 */
  tool: string
  /** 对方渲染好的 Markdown。**原样显示，前端不做解析**——解析它等于把对方的
   *  展示层当 API 用，它加个「⚠️」前缀我们就全线崩。 */
  markdown: string
}

export interface NovelHealth {
  enabled: boolean
  ok: boolean
  error?: string | null
  /** 连不上时附带的「怎么修」。用户看到的是这句，不是堆栈。 */
  hint?: string | null
  tools: string[]
  missing: string[]
}

/* ----------------------------------------------------------------------
 * 故事工坊（`/agent/*`）
 *
 * 这三个类型名与后端 `app/schemas/agent.py` 一一对应。**字段名照抄**：
 * 后端改了名字前端就该编译不过，而不是静默地少显示一块。
 * -------------------------------------------------------------------- */

/**
 * 会话状态。比 `RunStatus` 多一个 `idle`——两次改稿之间它就是闲着的，
 * 而「没事发生」和「跑完了」在界面上是两句话。
 *
 * `failed` / `cancelled` 是**会话的**终态：进程被杀留下的半截会话不能再喂
 * 消息（它下一轮会带着一串谁也说不清的上下文去找模型）。界面上给的是
 * 「重新开始」，不是「接着聊」。
 */
export type AgentStatus = 'idle' | 'running' | 'failed' | 'cancelled'

/** assistant 消息里的一次工具调用。`arguments` 是**原始 JSON 文本**。 */
export interface AgentToolCall {
  id: string
  name: string
  arguments: string
}

/** 一条对话消息。库里存的这一串就是下一轮喂给模型的那一串。 */
export interface AgentMessage {
  seq: number
  turn: number
  role: string
  content: string
  tool_calls: AgentToolCall[]
  tool_call_id?: string | null
  /** `role="tool"` 时是工具名；`role="user"` 且为 `force_final` 时表示这条
   *  是**代码生成的**收尾指令，不是用户说的话。 */
  name?: string | null
}

export interface AgentSession {
  id: string
  title: string
  input_chars: number
  status: AgentStatus
  /** 中文阶段名，直接显示。**前端不维护状态映射表**——那种表的表现是
   *  「改了后端文案，界面上还是旧说法」。 */
  stage?: string | null
  turn: number
  /** 最近一版的正文。 */
  story?: string | null
  /** 常驻显示：这是这个 agent「自主程度」唯一可核查的证据。 */
  rounds: number
  tool_calls: number
  model?: string | null
  providers: Record<string, unknown>
  options: Record<string, unknown>
  error?: string | null
  /** 终态集合由**后端**判定——前端拿 status 字符串自己推的话，以后加一个
   *  状态就要两个仓各改一次，而漏改的表现是「永远在转圈」。 */
  is_running: boolean
  is_terminal: boolean
  created_at: string
  updated_at: string
}

export interface AgentSessionDetail extends AgentSession {
  input_text: string
  messages: AgentMessage[]
  /** 演示模式徽章（模型是 mock 时必须显示）。 */
  demo: boolean
}

export interface AgentCapabilities {
  input_max: number
  instruction_max: number
  target_chars: number
  target_chars_choices: number[]
  allow_novel: boolean
  /** 古典文学 MCP 的**配置**自检：null 表示看起来没问题，字符串是「为什么
   *  起不来」。注意它不探 MCP 本身（探一次要起子进程等两秒，而这是打开面板
   *  就会调的接口）。 */
  novel_hint?: string | null
  model: string
  demo: boolean
  libraries: { value: string; label: string }[]
}

/** 开会话时能带的那几个选项。 */
export interface AgentOptions {
  allow_novel?: boolean
  target_chars?: number
}
