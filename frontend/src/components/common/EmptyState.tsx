import type { ReactNode } from 'react'

/**
 * 空状态。
 *
 * 每个面板都必须有设计过的空状态，并且**说清这里将来会出现什么**。
 * 空白面板会让人以为功能坏了——尤其在流式场景下，用户无法区分
 * 「还没轮到」和「出错了」。
 */
export function EmptyState({
  icon,
  title,
  hint,
  children,
}: {
  icon?: ReactNode
  title: string
  hint?: string
  children?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-10 text-center">
      {icon ? <div className="text-ink-ghost mb-0.5 text-2xl leading-none">{icon}</div> : null}
      <div className="text-ink-soft text-[13px]">{title}</div>
      {hint ? <div className="text-ink-faint max-w-[30ch] text-xs leading-relaxed">{hint}</div> : null}
      {children}
    </div>
  )
}
