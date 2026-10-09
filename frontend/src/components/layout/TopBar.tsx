
import { MockBadge } from '../common/MockBadge'

export type Depth = 'quick' | 'standard' | 'deep'

export const DEPTH_OPTIONS: { value: Depth; label: string; hint: string }[] = [
  { value: 'quick', label: '快速', hint: '检索 3/2/1，约 30 秒' },
  { value: 'standard', label: '标准', hint: '检索 6/4/3，约 60 秒' },
  { value: 'deep', label: '深度', hint: '检索 9/6/5，约 120 秒' },
]

export const KB_OPTIONS = [
  { key: 'psychology', label: '心理学·神经科学' },
  { key: 'literature', label: '古典文学' },
  { key: 'poetry', label: '诗词' },
] as const

export type KbKey = (typeof KB_OPTIONS)[number]['key']
export type KbSelection = Record<KbKey, boolean>

export function TopBar({
  input,
  onInputChange,
  depth,
  onDepthChange,
  kb,
  onKbChange,
  onStart,
  onAbort,
  running,
  aborting,
  canStart,
  providers,
  progress,
  progressMessage,
  onOpenKb,
  onOpenKbText,
  onOpenAgent,
  onOpenScene,
}: {
  input: string
  onInputChange: (v: string) => void
  depth: Depth
  onDepthChange: (d: Depth) => void
  kb: KbSelection
  onKbChange: (kb: KbSelection) => void
  onStart: () => void
  onAbort: () => void
  running: boolean
  aborting?: boolean
  canStart: boolean
  providers?: { llm?: string; embedding?: string; douyin?: string; is_demo?: boolean } | null
  progress?: number
  progressMessage?: string | null
  onOpenKb: () => void
  onOpenKbText: () => void
  onOpenAgent: () => void
  onOpenScene: () => void
}) {
  return (
    <header className="border-line bg-surface shrink-0 border-b">
      <div className="flex flex-wrap items-center gap-3 px-5 py-3">
        <div className="flex items-baseline gap-2.5">
          <span className="text-ink font-serif text-[19px] leading-none tracking-[0.14em]">观心</span>
          <span className="text-ink-faint hidden text-[11px] leading-none sm:inline">
            抖音内容心理洞察工作台
          </span>
        </div>

        <div className="bg-line mx-1 hidden h-5 w-px sm:block" aria-hidden />

        <input
          value={input}
          onChange={(e) => onInputChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && canStart && !running) onStart()
          }}
          placeholder="粘贴抖音分享链接、短链或 aweme_id"
          spellCheck={false}
          className="border-line bg-paper focus:border-accent-line focus:bg-surface min-w-[220px] flex-1 rounded-md border px-3 py-[7px] text-[13px] outline-none transition-colors placeholder:text-ink-ghost"
        />

        <SegmentedDepth value={depth} onChange={onDepthChange} disabled={running} />

        <KbToggles kb={kb} onChange={onKbChange} disabled={running} />

        <button
          type="button"
          onClick={onStart}
          disabled={!canStart || running}
          className="bg-accent hover:bg-accent/90 disabled:bg-line disabled:text-ink-ghost rounded-md px-4 py-[7px] text-[13px] font-medium text-white transition-colors disabled:cursor-not-allowed"
        >
          {running ? '分析中…' : '开始分析'}
        </button>

        {/* 分析中才出现。**必须有这个出口**：没有它，一次卡住的分析会把
            整个工作台锁死——开始按钮被运行态占着，而运行态自己不会结束
            （节点卡在外部调用上、或上次进程留下的僵尸运行）。 */}
        {running ? (
          <button
            type="button"
            onClick={onAbort}
            disabled={aborting}
            title="在当前节点结束后停止，已产出的结果会保留"
            className="border-danger text-danger hover:bg-danger-soft disabled:border-line disabled:text-ink-ghost rounded-md border px-3 py-[7px] text-[13px] transition-colors disabled:cursor-default"
          >
            {aborting ? '中止中…' : '中止'}
          </button>
        ) : null}

        {/* 故事工坊。**与流水线并列的第二条链路**，同样不受 running 影响：
            它有自己的会话、自己的表和自己的事件流，跑着分析的同时让写手改一稿
            是安全的。把入口和「＋ 导入」「📖 原文」摆在同一排，就是在说这件事：
            它们是并列的功能，不是一个流程里的下一步。 */}
        <button
          type="button"
          onClick={onOpenAgent}
          title="根据一段评论（或任意文本）写故事：它自己查资料，写完还能接着改"
          className="border-line text-ink-soft hover:bg-paper-sunk hover:text-ink rounded-md border px-3 py-[7px] text-[13px] transition-colors"
        >
          ✍ 写故事
        </button>

        {/* 对话工坊。**与「写故事」并列的第三条链路**：那边是一个写手从头写到尾，
            这边是一桌人各说各的、演完还能逐句改。同样不受 running 影响——它有
            自己的会话、自己的表和自己的事件流。 */}
        <button
          type="button"
          onClick={onOpenScene}
          title="填一段评论和几张人物卡，让这桌人自己把戏演出来；演完可以逐句提意见"
          className="border-line text-ink-soft hover:bg-paper-sunk hover:text-ink rounded-md border px-3 py-[7px] text-[13px] transition-colors"
        >
          🎭 写对话
        </button>

        {/* 导入不受 running 影响：分析在跑的时候往知识库里加东西是安全的，
            下一次检索才看得见它——而那正是用户的预期。 */}
        <button
          type="button"
          onClick={onOpenKb}
          title="导入本机文本文件或 epub 电子书到知识库"
          className="border-line text-ink-soft hover:bg-paper-sunk hover:text-ink rounded-md border px-3 py-[7px] text-[13px] transition-colors"
        >
          ＋ 导入
        </button>

        {/* 与「＋ 导入」并列：一个是往里存，一个是往外查。同样不受 running
            影响——分析在跑的时候查原文只是看，不会改变任何东西。 */}
        <button
          type="button"
          onClick={onOpenKbText}
          title="查阅古典文学知识库与观心知识库的原文"
          className="border-line text-ink-soft hover:bg-paper-sunk hover:text-ink rounded-md border px-3 py-[7px] text-[13px] transition-colors"
        >
          📖 原文
        </button>

        <MockBadge providers={providers} />
      </div>

      {running || (progress ?? 0) > 0 ? (
        <div className="border-line flex items-center gap-3 border-t px-5 py-2">
          <div className="bg-paper-sunk h-1 flex-1 overflow-hidden rounded-full">
            <div
              className="bg-accent h-full rounded-full transition-[width] duration-500 ease-out"
              style={{ width: `${progress ?? 0}%` }}
            />
          </div>
          <span className="text-ink-soft w-[42px] shrink-0 text-right font-mono text-[11px] tabular-nums">
            {progress ?? 0}%
          </span>
          <span className="text-ink-faint w-[280px] shrink-0 truncate text-[11px]">
            {progressMessage ?? ''}
          </span>
        </div>
      ) : null}
    </header>
  )
}

function SegmentedDepth({
  value,
  onChange,
  disabled,
}: {
  value: Depth
  onChange: (d: Depth) => void
  disabled?: boolean
}) {
  return (
    <div
      className={`border-line bg-paper inline-flex rounded-md border p-[2px] ${disabled ? 'opacity-60' : ''}`}
      role="group"
      aria-label="分析深度"
    >
      {DEPTH_OPTIONS.map((o) => (
        <button
          key={o.value}
          type="button"
          title={o.hint}
          disabled={disabled}
          onClick={() => onChange(o.value)}
          className={`rounded px-2.5 py-[5px] text-[12px] transition-colors disabled:cursor-not-allowed ${
            value === o.value ? 'bg-surface text-ink shadow-sm' : 'text-ink-faint hover:text-ink-soft'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

function KbToggles({
  kb,
  onChange,
  disabled,
}: {
  kb: KbSelection
  onChange: (kb: KbSelection) => void
  disabled?: boolean
}) {
  const enabledCount = KB_OPTIONS.filter((o) => kb[o.key]).length
  return (
    <div
      className={`border-line bg-paper inline-flex items-center gap-1 rounded-md border p-[2px] pr-2 ${disabled ? 'opacity-60' : ''}`}
      role="group"
      aria-label="知识库"
    >
      <span className="text-ink-ghost pl-1.5 text-[11px]">知识库</span>
      {KB_OPTIONS.map((o) => (
        <button
          key={o.key}
          type="button"
          disabled={disabled}
          aria-pressed={kb[o.key]}
          title={o.label}
          onClick={() =>
            onChange({ ...kb, [o.key]: !kb[o.key] } as KbSelection)
          }
          className={`rounded px-2 py-[5px] text-[12px] transition-colors disabled:cursor-not-allowed ${
            kb[o.key]
              ? 'bg-accent-soft text-accent'
              : 'text-ink-ghost hover:text-ink-faint line-through decoration-1'
          }`}
        >
          {o.label.replace('心理学·神经科学', '心理学').replace('古典文学', '文学')}
        </button>
      ))}
      {enabledCount === 0 ? (
        <span className="text-warn text-[11px]" title="至少启用一个知识库才能检索到证据">
          未选
        </span>
      ) : null}
    </div>
  )
}
