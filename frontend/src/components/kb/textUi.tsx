import type { ReactNode } from 'react'

/**
 * 「查阅原文」这个弹窗共用的小控件。
 *
 * 项目没有公共 Button（两个既有 Dialog 各自在文件底部带一份），但这次是
 * 一个弹窗拆成三份文件：再各带一份，很快就会出现「同一个按钮三种 padding」。
 */

export function Button({
  children,
  onClick,
  primary,
  disabled,
}: {
  children: ReactNode
  onClick: () => void
  primary?: boolean
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`rounded-md px-3.5 py-[6px] text-[12.5px] transition-colors disabled:cursor-not-allowed ${
        primary
          ? 'bg-accent hover:bg-accent/90 text-white disabled:bg-line disabled:text-ink-ghost'
          : 'border-line text-ink-soft hover:bg-surface border'
      }`}
    >
      {children}
    </button>
  )
}

export function ErrorLine({ text }: { text: string }) {
  return (
    <div className="border-danger/25 bg-danger-soft text-danger rounded-md border px-3 py-2 text-[12px] leading-relaxed whitespace-pre-wrap">
      {text}
    </div>
  )
}

export function Hint({ children }: { children: ReactNode }) {
  return <span className="text-ink-faint mt-0.5 text-[11px] leading-relaxed">{children}</span>
}

/**
 * 一个参数输入框。**窄屏下自己收缩**（`min-w-0` + `w-full`）：
 * 这些输入框都放在固定列数的网格里，一个长 placeholder 就能把整行撑爆。
 */
export function Param({
  label,
  value,
  onChange,
  placeholder,
  onEnter,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  placeholder?: string
  onEnter?: () => void
}) {
  return (
    <label className="flex min-w-0 flex-col gap-1">
      <span className="text-ink-soft text-[11.5px]">{label}</span>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && onEnter) onEnter()
        }}
        placeholder={placeholder}
        spellCheck={false}
        className="border-line bg-paper focus:border-accent-line focus:bg-surface w-full min-w-0 rounded-md border px-2 py-[5px] text-[12.5px] outline-none transition-colors placeholder:text-ink-ghost"
      />
    </label>
  )
}

/**
 * 结果区。
 *
 * **原文不被重新排版**：`whitespace-pre-wrap` 保留对方给的换行，长段落在
 * 窄屏上折行而不是横向滚动。
 *
 * 字体显式写 `font-sans`：Tailwind 的 preflight 给 `pre` 配的是等宽字体，
 * 中文正文用等宽难看得很。
 */
export function ResultBlock({ text }: { text: string }) {
  return (
    <pre className="border-line bg-paper text-ink font-sans mt-2 max-h-[46vh] overflow-y-auto rounded-md border px-3.5 py-3 text-[13px] leading-relaxed whitespace-pre-wrap">
      {text}
    </pre>
  )
}

/** 整数输入框的值。空 = 不传（后端按「不限定」处理），非数字按没填算。 */
export function num(raw: string | undefined): number | undefined {
  const text = (raw ?? '').trim()
  if (!text) return undefined
  const value = Number(text)
  return Number.isFinite(value) ? Math.trunc(value) : undefined
}
