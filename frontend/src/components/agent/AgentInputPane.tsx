/**
 * 左栏：材料。
 *
 * 同一栏两种形态——还没开始写时是可编辑的输入框，开始写之后变成只读的原文。
 * 不另开一个「新建」弹窗：材料就是这场对话的起点，把它放在对话的左边，
 * 「模型为什么写成了这样」随时可以回头看。
 *
 * 字数档位与库清单都来自 `/agent/capabilities`（后端是唯一来源）。前端硬编码
 * 一份的下场是「后端加了 3000 字档，界面上选不到」，而两边都不会报错。
 */

import type { AgentCapabilities } from '../../api/types'

export function AgentInputPane({
  hasSession,
  inputText,
  draft,
  onDraftChange,
  allowNovel,
  onAllowNovelChange,
  targetChars,
  onTargetCharsChange,
  capabilities,
  starting,
  error,
  onStart,
  onRestart,
  disabled,
}: {
  hasSession: boolean
  inputText: string
  draft: string
  onDraftChange: (value: string) => void
  allowNovel: boolean
  onAllowNovelChange: (value: boolean) => void
  targetChars: number
  onTargetCharsChange: (value: number) => void
  capabilities: AgentCapabilities | null
  starting: boolean
  error: string | null
  onStart: () => void
  onRestart: () => void
  disabled: boolean
}) {
  const max = capabilities?.input_max ?? 20000
  const choices = capabilities?.target_chars_choices ?? [300, 600, 900, 1500, 2400]

  return (
    <div className="border-line flex min-h-0 w-[26%] min-w-[240px] shrink-0 flex-col border-r">
      <div className="border-line flex shrink-0 items-baseline justify-between border-b px-4 py-2">
        <span className="text-ink text-[12px] font-medium">材料</span>
        <span className="text-ink-faint font-mono text-[10.5px] tabular-nums">
          {hasSession ? `${inputText.length} 字` : draft ? `${draft.length} 字` : ''}
        </span>
      </div>

      {hasSession ? (
        <>
          <div className="text-ink-soft min-h-0 flex-1 overflow-y-auto px-4 py-3 text-[12px] leading-relaxed whitespace-pre-wrap">
            {inputText}
          </div>
          <div className="border-line shrink-0 border-t px-3 py-2">
            <button
              type="button"
              onClick={onRestart}
              disabled={disabled}
              title="换一段材料，重新开一场（这一场会留在库里，但界面上不再显示）"
              className="border-line text-ink-soft hover:border-line-strong hover:text-ink w-full rounded-md border px-2 py-[5px] text-[11.5px] transition-colors disabled:cursor-not-allowed disabled:opacity-50"
            >
              换一段材料
            </button>
          </div>
        </>
      ) : (
        <>
          <textarea
            value={draft}
            onChange={(e) => onDraftChange(e.target.value)}
            maxLength={max}
            placeholder="把一段网友评论粘进来。也可以粘任何一段你想让它写成故事的文字。"
            className="bg-paper focus:bg-surface text-ink min-h-0 flex-1 resize-none px-4 py-3 text-[12.5px] leading-relaxed outline-none placeholder:text-ink-ghost"
          />

          <div className="border-line shrink-0 border-t px-3 py-2">
            <div className="mb-2 flex items-center gap-2">
              <label className="text-ink-faint text-[11px]" htmlFor="agent-target">
                篇幅
              </label>
              <select
                id="agent-target"
                value={targetChars}
                onChange={(e) => onTargetCharsChange(Number(e.target.value))}
                className="border-line bg-paper text-ink-soft rounded border px-1.5 py-[3px] text-[11px]"
              >
                {choices.map((n) => (
                  <option key={n} value={n}>
                    约 {n} 字
                  </option>
                ))}
              </select>

              <label className="text-ink-faint ml-auto flex cursor-pointer items-center gap-1 text-[11px]">
                <input
                  type="checkbox"
                  checked={allowNovel}
                  onChange={(e) => onAllowNovelChange(e.target.checked)}
                  className="accent-accent"
                />
                古典文学
              </label>
            </div>

            {/* MCP 连不上时先说清楚：那几张红色卡片是**配置**问题，
                不是这个 agent 不会用工具。 */}
            {allowNovel && capabilities?.novel_hint ? (
              <div className="text-warn mb-2 text-[10.5px] leading-snug">
                {capabilities.novel_hint}
              </div>
            ) : null}

            {error ? <div className="text-danger mb-2 text-[11px]">{error}</div> : null}

            <button
              type="button"
              onClick={onStart}
              disabled={starting || !draft.trim()}
              className="bg-accent hover:bg-accent/90 disabled:bg-line disabled:text-ink-ghost w-full rounded-md px-3 py-[7px] text-[12.5px] font-medium text-white transition-colors disabled:cursor-not-allowed"
            >
              {starting ? '正在开…' : '开始写'}
            </button>

            <p className="text-ink-ghost mt-2 text-[10.5px] leading-snug">
              它会自己决定查哪些资料、查几次，写完可以继续提要求让它改。
            </p>
          </div>
        </>
      )}
    </div>
  )
}
