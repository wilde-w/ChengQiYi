import { useCallback, useMemo, useState, type ReactNode } from 'react'

import type { CreateRunPayload } from '../../api/endpoints'
import type { TextMode } from '../../api/types'
import {
  MIN_COMMENTS_FOR_CLUSTERING,
  TEXT_INPUT_MAX,
  commentLimitFor,
  previewSplit,
  splitParts,
} from '../../lib/splitText'
import { Modal } from '../common/Modal'
import { DEPTH_OPTIONS, KB_OPTIONS, type Depth, type KbSelection } from '../layout/TopBar'

/**
 * 「粘贴文本」对话框。
 *
 * 这是左栏数据源的第二个入口：用户手上没有视频链接，只有一段现成的文字
 * （从别处复制的评论、访谈记录、笔记），想让它走同一条研究与解析流水线。
 *
 * 三件事值得说明：
 *
 * 1. **切分方式是个开关，不是一个猜测。** 「一行一条」对复制来的评论区
 *    是对的，「按空行分段」对一篇长文是对的，而**没有任何办法从文本本身
 *    可靠地分辨这两者**——一篇没有空行的长文按行切会碎成几百条半句话。
 *    所以让用户自己说是哪一种，并在旁边实时显示这个选择切出来多少条：
 *    三十条和三百条是完全不同的两次分析，他应该在按下按钮之前就知道。
 *
 * 2. **关闭不清空。** 误触 Esc 或点一下遮罩就丢掉一次长粘贴是不可接受的，
 *    与 KbImportDialog 保留作业的做法一致。文本留在这里，重新打开还在。
 *
 * 3. **提交后清空。** 这次文本已经变成一条运行了，留在框里只会让用户
 *    以为还没提交、再点一次，从而白跑一遍。
 */
export function TextSourceDialog({
  open,
  onClose,
  depth,
  kb,
  running,
  onLaunch,
}: {
  open: boolean
  onClose: () => void
  depth: Depth
  kb: KbSelection
  running: boolean
  /** 与顶栏共用同一个提交路径。返回错误文案，成功则返回 null。 */
  onLaunch: (payload: CreateRunPayload) => Promise<string | null>
}) {
  const [text, setText] = useState('')
  const [mode, setMode] = useState<TextMode>('line')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const overLimit = text.length > TEXT_INPUT_MAX

  const preview = useMemo(() => {
    const total = splitParts(text, mode).length
    const limit = commentLimitFor(total)
    return { limit, ...previewSplit(text, mode, limit) }
  }, [text, mode])

  const submit = useCallback(async () => {
    if (submitting || running || preview.total === 0 || overLimit) return
    setSubmitting(true)
    setError(null)
    const message = await onLaunch({
      input: text,
      source: 'text',
      text_mode: mode,
      depth,
      kb,
      comment_limit: commentLimitFor(preview.total),
    })
    setSubmitting(false)
    if (message) {
      // 错误显示在**弹窗内**：此刻顶栏的链接输入框是空的，把「文本为空」
      // 这类报错显示在那里，用户会对着一个跟他刚才做的事无关的地方发愣。
      setError(message)
      return
    }
    setText('')
    onClose()
  }, [submitting, running, preview.total, overLimit, onLaunch, text, mode, depth, kb, onClose])

  const busy = submitting || running

  return (
    <Modal
      open={open}
      onClose={onClose}
      width="max-w-[680px]"
      title="粘贴文本"
      subtitle="把一段文字切分成条目，走与抖音视频相同的解析流程：聚类 → 心理侧写 → 知识库检索 → 推理 → 洞察。"
      footer={
        <div className="flex w-full items-center justify-between gap-3">
          <span className="text-ink-faint text-[11.5px]">
            {preview.total === 0
              ? '还没有内容'
              : `将分析 ${Math.min(preview.limit, preview.total)} 条`}
          </span>
          <div className="flex items-center gap-2">
            <Button onClick={onClose}>取消</Button>
            <Button
              primary
              disabled={busy || preview.total === 0 || overLimit}
              onClick={() => void submit()}
            >
              {submitting ? '提交中…' : running ? '分析进行中' : '开始分析'}
            </Button>
          </div>
        </div>
      }
    >
      <div className="flex flex-col gap-3.5">
        <Field label="文本内容" hint={`${text.length} / ${TEXT_INPUT_MAX}`}>
          <textarea
            value={text}
            onChange={(e) => {
              setText(e.target.value)
              if (error) setError(null)
            }}
            spellCheck={false}
            placeholder={'把评论、访谈记录或一段笔记粘到这里\n\n第一行\n第二行\n第三行'}
            className="border-line bg-paper focus:border-accent-line focus:bg-surface min-h-[220px] resize-y rounded-md border px-3 py-2.5 font-mono text-[12.5px] leading-relaxed outline-none transition-colors placeholder:text-ink-ghost"
          />
        </Field>

        <Field label="切分方式">
          <div className="flex flex-wrap gap-1.5">
            <ModeButton
              active={mode === 'line'}
              onClick={() => setMode('line')}
              label="每行一条"
              hint="适合从评论区复制来的多行短句"
            />
            <ModeButton
              active={mode === 'paragraph'}
              onClick={() => setMode('paragraph')}
              label="按空行分段"
              hint="适合一整篇文章，一段算一条"
            />
          </div>
          <Hint>
            {mode === 'line'
              ? '每一行作为一条，空行会被忽略。原文里的换行因此不能保留。'
              : '用一个空行分段，一段作为一条；段内的换行会在分析时被合并成空格。没有空行的长文会整体算作一条。'}
          </Hint>
        </Field>

        <div className="border-line bg-paper-sunk rounded-md border px-3.5 py-3">
          <div className="flex items-baseline justify-between gap-3">
            <span className="text-ink text-[13px]">
              识别到 <b className="font-mono tabular-nums">{preview.total}</b> 条
            </span>
            <span className="text-ink-faint font-mono text-[11px] tabular-nums">
              上限 {preview.limit}
            </span>
          </div>
          <ul className="text-ink-faint mt-1.5 flex flex-col gap-0.5 text-[11.5px] leading-relaxed">
            {preview.total > 0 && preview.total < MIN_COMMENTS_FOR_CLUSTERING ? (
              <li>
                · 少于 {MIN_COMMENTS_FOR_CLUSTERING} 条时不做聚类，整体作为一个主题分析。
              </li>
            ) : null}
            {preview.dropped > 0 ? (
              <li className="text-warn">
                · 超出上限的 {preview.dropped} 条不会被分析（前 {preview.limit} 条以内为准）。
              </li>
            ) : null}
            {preview.truncated > 0 ? (
              <li className="text-warn">· {preview.truncated} 条过长，会被截断到 2000 字。</li>
            ) : null}
            {overLimit ? (
              <li className="text-danger">· 超过 {TEXT_INPUT_MAX} 字上限，请分批分析。</li>
            ) : null}
          </ul>
        </div>

        <div className="border-line text-ink-faint flex flex-wrap items-center gap-x-3 gap-y-1 border-t pt-3 text-[11.5px]">
          <span>
            深度：
            <span className="text-ink-soft">
              {DEPTH_OPTIONS.find((o) => o.value === depth)?.label ?? depth}
            </span>
          </span>
          <span>
            知识库：
            <span className="text-ink-soft">
              {KB_OPTIONS.filter((o) => kb[o.key]).map((o) => o.label).join(' / ') || '未选择'}
            </span>
          </span>
          <span className="text-ink-ghost">在顶栏调整</span>
        </div>

        {error ? <ErrorLine text={error} /> : null}
      </div>
    </Modal>
  )
}

/* --------------------------- 小组件 --------------------------- */

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-ink-soft flex items-baseline justify-between gap-2 text-[12px]">
        <span>{label}</span>
        {hint ? <span className="text-ink-ghost font-mono text-[11px]">{hint}</span> : null}
      </span>
      {children}
    </label>
  )
}

function Hint({ children }: { children: ReactNode }) {
  return <span className="text-ink-faint mt-0.5 text-[11px] leading-relaxed">{children}</span>
}

function ModeButton({
  active,
  onClick,
  label,
  hint,
}: {
  active: boolean
  onClick: () => void
  label: string
  hint: string
}) {
  return (
    <button
      type="button"
      aria-pressed={active}
      title={hint}
      onClick={onClick}
      className={`rounded-md border px-2.5 py-[5px] text-[12px] transition-colors ${
        active
          ? 'border-accent-line bg-accent-soft text-accent'
          : 'border-line text-ink-faint hover:text-ink-soft'
      }`}
    >
      {label}
    </button>
  )
}

function Button({
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

function ErrorLine({ text }: { text: string }) {
  return (
    <div className="border-danger/25 bg-danger-soft text-danger rounded-md border px-3 py-2 text-[12px] leading-relaxed">
      {text}
    </div>
  )
}
