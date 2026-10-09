import type { ReactNode } from 'react'
import { StatusPill, type ColumnStatus } from '../common/Pill'

/**
 * 三栏统一的容器。
 *
 * 标题栏固定、正文独立滚动——这是工作台的关键：中栏进度在滚动时，
 * 左栏评论列表不该跟着动。
 */
export function ColumnShell({
  title,
  subtitle,
  status,
  statusHint,
  width,
  children,
  headerExtra,
  footer,
}: {
  title: string
  subtitle?: string
  status: ColumnStatus
  statusHint?: ReactNode
  width?: number
  children: ReactNode
  headerExtra?: ReactNode
  footer?: ReactNode
}) {
  return (
    <section
      className="bg-paper flex min-h-0 flex-col"
      style={width !== undefined ? { width, flexShrink: 0 } : { flex: 1, minWidth: 0 }}
      aria-label={title}
    >
      <header className="border-line bg-paper/85 sticky top-0 z-[5] border-b px-4 py-3 backdrop-blur-sm">
        <div className="flex items-center justify-between gap-2">
          <div className="flex min-w-0 items-baseline gap-2">
            <h2 className="text-ink text-[13px] font-semibold tracking-wide">{title}</h2>
            {subtitle ? (
              <span className="text-ink-faint truncate text-[11px]">{subtitle}</span>
            ) : null}
          </div>
          <StatusPill status={status} hint={statusHint} />
        </div>
        {headerExtra ? <div className="mt-2.5">{headerExtra}</div> : null}
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>

      {footer ? (
        <div className="border-line bg-paper/85 sticky bottom-0 border-t px-4 py-2.5 backdrop-blur-sm">
          {footer}
        </div>
      ) : null}
    </section>
  )
}
