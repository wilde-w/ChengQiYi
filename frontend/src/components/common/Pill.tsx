import type { ReactNode } from 'react'

export type ColumnStatus = 'idle' | 'running' | 'done' | 'failed'

const STATUS_STYLE: Record<ColumnStatus, { label: string; cls: string }> = {
  idle: { label: '等待中', cls: 'text-ink-faint bg-paper-sunk border-line' },
  running: { label: '进行中', cls: 'text-accent bg-accent-soft border-accent-line' },
  done: { label: '完成', cls: 'text-ok bg-[#f0f9f3] border-[#cfe7d8]' },
  failed: { label: '失败', cls: 'text-danger bg-danger-soft border-[#f3d3d3]' },
}

/** 列状态胶囊。运行中带一个呼吸点，让「正在跑」在静默期也能被感知到。 */
export function StatusPill({ status, hint }: { status: ColumnStatus; hint?: ReactNode }) {
  const s = STATUS_STYLE[status]
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-[3px] text-[11px] leading-none ${s.cls}`}
    >
      {status === 'running' && (
        <span className="animate-pulse-soft size-1.5 rounded-full bg-current" aria-hidden />
      )}
      {s.label}
      {hint ? <span className="opacity-70">{hint}</span> : null}
    </span>
  )
}
