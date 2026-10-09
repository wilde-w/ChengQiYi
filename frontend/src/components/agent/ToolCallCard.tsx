/**
 * 一张工具卡片。
 *
 * **这张卡是这个面板里最该被看见的东西。** 它让「agent 自己决定了什么」成为
 * 可核查的事实，而不是一个说法：查了什么、参数是什么、回来了多少、花了多久。
 * 所以默认就显示「工具名 + 参数 + 结果摘要 + 耗时」，展开才看细节——
 * 收起时的那一行仍然是有信息量的。
 *
 * 参数显示的是**模型写下的原文**（`rawArgs`）而不是我们解析后的对象：
 * 参数写坏时，那是唯一能看出「它到底想干什么」的线索，而解析失败的对象
 * 恰好会把这个线索抹掉。
 */

import { useState } from 'react'

import type { ToolCard } from '../../store/applyAgentEvent'

export function ToolCallCard({ card }: { card: ToolCard }) {
  const [open, setOpen] = useState(false)
  const failed = card.result ? !card.result.ok : false
  // 后端把「没执行」的几种情况（超限、未知工具、额度用完）标成 reason=refused，
  // 它们不是失败而是**被拒**——卡片上要说得出这个区别，否则用户会以为是
  // 知识库坏了。
  const refused = card.result?.reason === 'refused'

  return (
    <div
      className={`animate-rise overflow-hidden rounded-md border ${
        failed
          ? refused
            ? 'border-warn/40 bg-warn-soft/40'
            : 'border-danger/40 bg-danger-soft/40'
          : 'border-line bg-paper'
      }`}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full flex-col gap-1 px-2.5 py-1.5 text-left"
        title={open ? '收起' : '展开参数与结果'}
      >
        <span className="flex w-full items-start gap-1.5">
          <span className="text-ink-ghost mt-[1px] shrink-0 text-[10px] leading-none" aria-hidden>
            {open ? '▾' : '▸'}
          </span>
          <span className="min-w-0 flex-1">
            <span className="text-accent font-mono text-[11.5px] font-medium">{card.name}</span>
            <span className="text-ink-soft ml-1.5 font-mono text-[11px] break-all">
              {argsSummary(card)}
            </span>
          </span>
          {/* 这一格只放**短的标签**（条数/字数 + 耗时）。曾被塞进 `result.summary`
              过：后端那边 `SUMMARY_CHARS = 40`，摘要是「一句人话」，40 个中文字约
              420px，而这里是 `shrink-0` + `whitespace-nowrap`——不收缩、不换行，
              于是它把左边 `flex-1` 的宽度吃到 0，参数被压成一行一个字、卡片长到
              700px，而这张卡存在的意义（显示了什么参数）正好被卡片的
              `overflow-hidden` 切掉。`min-w-0 max-w-[45%] truncate` 是兜底：
              短标签按内容走，真长起来也不会再把左半边挤没。 */}
          <span className="min-w-0 max-w-[45%] shrink truncate text-right font-mono text-[10.5px] tabular-nums">
            {card.result ? (
              <span className={failed ? 'text-danger' : 'text-ink-faint'}>
                {failed ? (refused ? '未执行' : '失败') : summaryOf(card)}
                {card.result.elapsedMs > 0 ? ` · ${formatMs(card.result.elapsedMs)}` : ''}
              </span>
            ) : (
              <span className="text-ink-ghost animate-pulse-soft">查询中…</span>
            )}
          </span>
        </span>

        {/* 工具自己那句话单独一行：它可能有 40 个字，一行放得下就一行，
            放不下就换行——这一行有整格的宽度，不必和参数抢。 */}
        {card.result?.summary ? (
          <span className="text-ink-faint ml-[14px] text-[10.5px] leading-snug break-words">
            {card.result.summary}
          </span>
        ) : null}
      </button>

      {open ? (
        <div className="border-line min-w-0 border-t px-2.5 py-2">
          <Section label="参数（模型写的原文）">{card.rawArgs || JSON.stringify(card.args)}</Section>
          {card.result ? <Section label="结果">{card.result.preview || '（空）'}</Section> : null}
        </div>
      ) : null}
    </div>
  )
}

function Section({ label, children }: { label: string; children: string }) {
  return (
    <div className="mb-2 last:mb-0">
      <div className="text-ink-ghost mb-0.5 text-[10px]">{label}</div>
      <pre className="text-ink-soft max-h-52 overflow-auto font-mono text-[11px] leading-relaxed break-words whitespace-pre-wrap">
        {children}
      </pre>
    </div>
  )
}

/** 一句话说清它查了什么。取第一个短字符串参数当代表，没有就退回键值对。 */
function argsSummary(card: ToolCard): string {
  const prefer = ['query', 'name', 'action', 'key', 'mode', 'source_file']
  for (const key of prefer) {
    const value = card.args[key]
    if (typeof value === 'string' && value) return `${key}=${truncate(value, 28)}`
  }
  const entries = Object.entries(card.args).filter(([, v]) => v !== null && v !== undefined)
  if (entries.length === 0) return card.rawArgs ? '{…}' : '()'
  return entries
    .slice(0, 3)
    .map(([k, v]) => `${k}=${truncate(typeof v === 'string' ? v : JSON.stringify(v), 20)}`)
    .join(' ')
}

function summaryOf(card: ToolCard): string {
  return card.result?.chars ? `${card.result.chars} 字` : '已返回'
}

function truncate(text: string, limit: number): string {
  return text.length > limit ? `${text.slice(0, limit)}…` : text
}

function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`
}
