/**
 * 右栏：正文。
 *
 * 这里只放**最新那一版**。旧版留在中栏的对话里（它们已经是「他说过的话」），
 * 而不是在这里堆成一摞——用户要读的是当下的这篇，比较版本是对话该干的事。
 *
 * 逐字到来时留一个方块光标：没有它，「模型还在写」和「卡住了」看起来一模一样。
 */

import { useEffect, useState } from 'react'

import type { AgentLocalStatus } from '../../store/applyAgentEvent'

export function StoryPane({
  story,
  status,
  streaming,
  turn,
  rounds,
  toolCalls,
  model,
  error,
  hasSession,
}: {
  story: string
  status: AgentLocalStatus
  streaming: boolean
  turn: number
  rounds: number
  toolCalls: number
  model: string | null
  error: string | null
  hasSession: boolean
}) {
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    if (!copied) return
    const timer = setTimeout(() => setCopied(false), 1500)
    return () => clearTimeout(timer)
  }, [copied])

  const running = status === 'running'
  const chars = story.replace(/\s/g, '').length

  const copy = () => {
    void navigator.clipboard
      ?.writeText(story)
      .then(() => setCopied(true))
      .catch(() => undefined)
  }

  return (
    <div className="border-line flex min-h-0 w-[38%] min-w-[300px] shrink-0 flex-col border-l">
      <div className="border-line flex shrink-0 items-baseline justify-between gap-2 border-b px-4 py-2">
        <span className="text-ink text-[12px] font-medium">正文</span>
        {/* 这三个数字是「自主程度」唯一可核查的证据——常驻，不折叠。 */}
        <span className="text-ink-faint font-mono text-[10.5px] tabular-nums">
          {hasSession ? `第 ${turn} 稿 · ${rounds} 轮 · ${toolCalls} 次工具` : ''}
        </span>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        {error && status === 'failed' ? (
          <div className="border-danger/40 bg-danger-soft/40 text-danger rounded-md border px-3 py-2 text-[12px] leading-relaxed">
            {error}
          </div>
        ) : story ? (
          <div className="text-ink font-serif text-[14.5px] leading-[1.9] whitespace-pre-wrap">
            {renderStory(story)}
            {streaming ? (
              <span className="bg-accent ml-0.5 inline-block h-[14px] w-[7px] animate-pulse-soft align-text-bottom" />
            ) : null}
          </div>
        ) : (
          <p className="text-ink-faint px-1 py-6 text-center text-[12px] leading-relaxed">
            {running ? '正在写…' : hasSession ? '这一版还没写出来。' : '还没有稿子。'}
          </p>
        )}
      </div>

      <div className="border-line flex shrink-0 items-center justify-between gap-2 border-t px-4 py-2">
        <span className="text-ink-ghost font-mono text-[10.5px] tabular-nums">
          {story ? `${chars} 字` : ''}
          {model ? ` · ${model}` : ''}
        </span>
        <button
          type="button"
          onClick={copy}
          disabled={!story}
          className="border-line text-ink-soft hover:border-line-strong hover:text-ink shrink-0 rounded-md border px-2 py-[3px] text-[11px] transition-colors disabled:cursor-not-allowed disabled:opacity-50"
        >
          {copied ? '已复制' : '复制'}
        </button>
      </div>
    </div>
  )
}

/**
 * 把正文按空行切成段。
 *
 * 用 `<p>` 而不是一个大 `white-space: pre-wrap` 块：段间距交给排版而不是让
 * 模型恰好多敲一个换行——它写出来的空行数量每次都不一样，而段落之间该有
 * 多少距离是**版式**的事。
 */
function renderStory(story: string) {
  return story
    .split(/\n{2,}/)
    .filter((para) => para.trim())
    .map((para, i) => (
      <p key={i} className="mb-3 last:mb-0">
        {para.trim()}
      </p>
    ))
}
