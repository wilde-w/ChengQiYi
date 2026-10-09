/**
 * 中栏：对话与过程。
 *
 * 一条时间线里混着三种东西——用户说的、模型说的、模型查的。它们合在一起才是
 * 「这场对话」，而工具卡片是其中最该被看见的：**「agent 自己决定了什么」在这
 * 里变成可核查的事实**。所以卡片与气泡同级，不是折叠在某个「详情」里。
 *
 * 输入框放在这一栏的底部而不是右栏：用户接下来产生的是**对话**（「再暗一点」
 * 「换成第二人称」），正文只是它的产物。
 */

import { useEffect, useMemo, useRef } from 'react'

import type { AgentLocalStatus, AgentState } from '../../store/applyAgentEvent'
import { buildTranscript } from '../../store/applyAgentEvent'
import { ToolCallCard } from './ToolCallCard'

export function AgentTranscript({
  messages,
  cards,
  story,
  status,
  liveStage,
  stage,
  instruction,
  onInstructionChange,
  onSubmit,
  onStop,
  stopping,
  note,
  disabled,
  instructionMax,
}: {
  messages: AgentState['messages']
  cards: AgentState['cards']
  /** 正在逐字写的那一版。还没落库，所以它不在 messages 里。 */
  story: string
  status: AgentLocalStatus
  liveStage: string | null
  stage: string | null
  instruction: string
  onInstructionChange: (value: string) => void
  onSubmit: () => void
  onStop: () => void
  stopping: boolean
  /** 上一条指令的错误（409 等），就地显示在输入框上方。 */
  note: string | null
  disabled: boolean
  instructionMax: number
}) {
  const items = useMemo(() => buildTranscript({ messages, cards }), [messages, cards])
  const running = status === 'running'
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const pinnedRef = useRef(true)

  // 自动跟到底部，但**用户往上翻看过就不打扰他**——模型正写着的时候被强行
  // 拽回底部，是这类界面里最招人烦的一个行为。
  useEffect(() => {
    const el = scrollRef.current
    if (!el || !pinnedRef.current) return
    el.scrollTop = el.scrollHeight
  }, [items.length, story.length, running])

  const onScroll = () => {
    const el = scrollRef.current
    if (!el) return
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
  }

  return (
    // `min-w-0` 不是保险，是必需的：`flex-1` 的默认 `min-width: auto` 会把这一栏
    // 撑到内容的 min-content 宽度。中栏装的是**模型与用户说的话**——一段不带空格
    // 的长串（工具参数里的一长串 id、模型写坏的一行）就能把这一栏顶宽，挤掉右边
    // 的正文栏；正文栏是 `shrink-0`，被挤出去的那部分会被面板的 `overflow-hidden`
    // 直接切掉，一个字都不报错。
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div ref={scrollRef} onScroll={onScroll} className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        {items.length === 0 ? (
          <p className="text-ink-faint px-1 py-6 text-center text-[12px] leading-relaxed">
            左边粘一段文字，点「开始写」。
            <br />
            模型会自己决定查什么资料、查几次，写完还能接着改。
          </p>
        ) : null}

        <div className="flex flex-col gap-2.5">
          {items.map((item) => {
            if (item.kind === 'user') {
              return item.generated ? (
                <div key={item.id} className="text-ink-ghost py-0.5 text-center text-[10.5px]">
                  ※ 系统：{item.text}
                </div>
              ) : (
                <div key={item.id} className="flex justify-end">
                  <div className="bg-accent-soft text-ink max-w-[85%] rounded-lg rounded-br-sm px-2.5 py-1.5 text-[12.5px] leading-relaxed break-words whitespace-pre-wrap">
                    {item.text}
                  </div>
                </div>
              )
            }
            if (item.kind === 'assistant') {
              // 不是草稿的那条（见 `TranscriptItem.draft`）：模型在调工具之前说的
              // 那句话。它不该长成气泡的样子——气泡意味着「这是一版正文」。
              return item.draft ? (
                <div key={item.id} className="flex justify-start">
                  <div className="border-line bg-surface text-ink-soft max-w-[92%] rounded-lg rounded-bl-sm border px-2.5 py-1.5 text-[12px] leading-relaxed break-words whitespace-pre-wrap">
                    <span className="text-ink-ghost mr-1.5 text-[10px]">第 {item.turn} 稿</span>
                    {truncate(item.text, 160)}
                  </div>
                </div>
              ) : (
                <div key={item.id} className="text-ink-ghost pl-1 text-[10.5px] leading-relaxed">
                  ※ 查资料前说的：{truncate(item.text, 160)}
                </div>
              )
            }
            return <ToolCallCard key={item.id} card={item.card} />
          })}

          {/* 正在写的这一版：它还没落库，所以不进对话，只在这里露出尾巴。 */}
          {running && story ? (
            <div className="flex justify-start">
              <div className="border-accent-line bg-surface text-ink max-w-[92%] rounded-lg rounded-bl-sm border px-2.5 py-1.5 text-[12px] leading-relaxed break-words whitespace-pre-wrap">
                <span className="text-accent mr-1.5 text-[10px]">正在写</span>
                {truncate(story, 160)}
              </div>
            </div>
          ) : null}
        </div>
      </div>

      <div className="border-line shrink-0 border-t px-3 py-2">
        <div className="text-ink-faint mb-1 flex items-center gap-1.5 text-[11px]">
          {running ? (
            <>
              <span className="bg-accent animate-pulse-soft size-1.5 rounded-full" aria-hidden />
              <span>{liveStage ?? stage ?? '正在写…'}</span>
            </>
          ) : (
            <span className="text-ink-ghost">
              {status === 'idle' ? '写完了。可以继续说要求，比如「再暗一点」「短一些」。' : ''}
              {status === 'cancelled' ? '这场对话已停止，只能重新开始一场。' : ''}
              {status === 'failed' ? '这一场出错了，只能重新开始一场。' : ''}
            </span>
          )}
        </div>

        {note ? <div className="text-danger mb-1 text-[11px]">{note}</div> : null}

        <div className="flex items-end gap-2">
          <textarea
            value={instruction}
            onChange={(e) => onInstructionChange(e.target.value)}
            onKeyDown={(e) => {
              // Enter 发送、Shift+Enter 换行——与市面上的对话界面一致
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault()
                if (!running && instruction.trim()) onSubmit()
              }
            }}
            rows={2}
            maxLength={instructionMax}
            disabled={disabled}
            placeholder={running ? '正在写，等它写完…' : '接着说要求，比如「再暗一点，短一些」'}
            className="border-line bg-paper focus:border-accent-line focus:bg-surface min-h-[38px] flex-1 resize-none rounded-md border px-2.5 py-1.5 text-[12.5px] outline-none transition-colors placeholder:text-ink-ghost disabled:opacity-60"
          />
          {running ? (
            <button
              type="button"
              onClick={onStop}
              disabled={stopping}
              title="在当前这一步结束后停下，已写出的正文不会被覆盖"
              className="border-danger text-danger hover:bg-danger-soft disabled:border-line disabled:text-ink-ghost shrink-0 rounded-md border px-3 py-[7px] text-[12.5px] transition-colors disabled:cursor-default"
            >
              {stopping ? '停止中…' : '停止'}
            </button>
          ) : (
            <button
              type="button"
              onClick={onSubmit}
              disabled={disabled || !instruction.trim()}
              className="bg-accent hover:bg-accent/90 disabled:bg-line disabled:text-ink-ghost shrink-0 rounded-md px-3 py-[7px] text-[12.5px] font-medium text-white transition-colors disabled:cursor-not-allowed"
            >
              发送
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function truncate(text: string, limit: number): string {
  return text.length > limit ? `…${text.slice(-limit)}` : text
}
