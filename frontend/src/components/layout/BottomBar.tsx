import type { ReactNode } from 'react'

export function BottomBar({
  onRerun,
  onSaveCase,
  canAct,
  canRerun,
  runId,
  durationMs,
  extra,
}: {
  onRerun: () => void
  onSaveCase: () => void
  canAct: boolean
  /**
   * 能否重跑。文本源为 false：重跑要拿原始输入再提交一次，而顶栏那个
   * 输入框是**链接**输入框，不会回填粘贴的文本。留一个点了没反应的按钮，
   * 比把它置灰并说明原因要糟。
   */
  canRerun?: boolean
  runId?: string | null
  durationMs?: number | null
  extra?: ReactNode
}) {
  const rerun = canRerun ?? canAct
  return (
    <footer className="border-line bg-surface flex shrink-0 items-center gap-2 border-t px-5 py-2.5">
      <BarButton
        onClick={onRerun}
        disabled={!canAct || !rerun}
        title={rerun ? undefined : '手动文本的运行请重新粘贴文本发起'}
      >
        重新分析
      </BarButton>

      {/* 导出报告在 V1 范围外。置灰并说明，比藏起来诚实。 */}
      <BarButton disabled title="V1.1 提供">
        导出报告
        <span className="text-ink-ghost ml-1 text-[10px]">V1.1</span>
      </BarButton>

      <BarButton onClick={onSaveCase} disabled={!canAct} emphasis>
        保存到案例库
      </BarButton>

      {extra}

      <div className="ml-auto flex items-center gap-3">
        {durationMs ? (
          <span className="text-ink-faint font-mono text-[11px] tabular-nums">
            用时 {(durationMs / 1000).toFixed(1)}s
          </span>
        ) : null}
        {runId ? (
          <span className="text-ink-ghost font-mono text-[11px]" title={runId}>
            run {runId.slice(0, 8)}
          </span>
        ) : null}
      </div>
    </footer>
  )
}

function BarButton({
  children,
  onClick,
  disabled,
  emphasis,
  title,
}: {
  children: ReactNode
  onClick?: () => void
  disabled?: boolean
  emphasis?: boolean
  title?: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={[
        'inline-flex items-center rounded-md border px-3 py-[6px] text-[12.5px] transition-colors',
        'disabled:cursor-not-allowed disabled:opacity-50',
        emphasis
          ? 'border-accent-line bg-accent-soft text-accent hover:bg-accent/10'
          : 'border-line text-ink-soft hover:border-line-strong hover:text-ink',
      ].join(' ')}
    >
      {children}
    </button>
  )
}
