import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'

import { api, ApiError } from '../../api/endpoints'
import type { KbImportJob, LibraryKey } from '../../api/types'
import { KB_OPTIONS } from '../layout/TopBar'
import { Modal } from '../common/Modal'
import { KbSourceList } from './KbSourceList'

/**
 * 后端两个体积上限的镜像。**只用来提前拦一下**，真正的判据在后端。
 *
 * 为什么 epub 单独一把尺子：电子书里大半体积是图片，正文反而只有几十万字。
 * 按扩展名选而不是按 zip 头——前端手里只有 File 元数据，读完头两个字节再判
 * 会让这段逻辑比它在后端的那一份更啰嗦，而这里拦错的后果只是多传一次。
 */
const TEXT_MAX_BYTES = 5 * 1024 * 1024
const EPUB_MAX_BYTES = 20 * 1024 * 1024

const isEpub = (f: File) => f.name.toLowerCase().endsWith('.epub')
const maxBytesFor = (f: File) => (isEpub(f) ? EPUB_MAX_BYTES : TEXT_MAX_BYTES)

type Phase = 'form' | 'confirm' | 'running' | 'done'

/**
 * 导入对话框：选文件 → 打标 → 入库，全程进度可见。
 *
 * 四种形态由作业对象自己决定，不额外维护一份状态机：
 * 没有作业 = 填表；作业停在 queued 且 needs_confirm = 等确认；
 * 非终态 = 跑着；终态 = 结果。这样「刷新页面前后看到的一致」是白拿的。
 */
export function KbImportDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [file, setFile] = useState<File | null>(null)
  const [library, setLibrary] = useState<LibraryKey>('psychology')
  const [title, setTitle] = useState('')
  const [author, setAuthor] = useState('')
  const [discipline, setDiscipline] = useState('')
  const [useLlm, setUseLlm] = useState(true)
  const [allowPartial, setAllowPartial] = useState(false)

  const [job, setJob] = useState<KbImportJob | null>(null)
  const [demo, setDemo] = useState(false)
  const [formError, setFormError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  /** 来源列表在导入成功后要重抓，用这个计数当触发器。 */
  const [sourcesVersion, setSourcesVersion] = useState(0)

  const phase: Phase = !job
    ? 'form'
    : job.is_terminal
      ? 'done'
      : job.needs_confirm
        ? 'confirm'
        : 'running'

  // 轮询。**只要作业还活着就轮**，不按界面形态分情况——这里踩过一次：
  // 早先只在「运行中」形态轮询，于是点下「继续导入」之后，服务端返回的
  // 还是那一条 `queued` 快照（真正的执行是后台任务，此刻还没改库），
  // 界面停在确认态、轮询又没启动，两边互等，永远卡住。
  // 依赖只取 id 与终态标志：依赖整个对象会让每拍都重建一次定时器。
  const jobId = job?.id
  const jobDone = job?.is_terminal ?? true
  useEffect(() => {
    if (!jobId || jobDone) return
    const timer = window.setInterval(() => {
      void api
        .getKbImport(jobId)
        .then((next) => {
          setJob(next)
          if (next.is_terminal && next.status === 'succeeded') {
            setSourcesVersion((v) => v + 1)
          }
        })
        .catch(() => {
          // 轮询失败（后端重启、网络抖动）不改状态：下一拍会重试。
          // 把界面切成错误态反而会让一次瞬时抖动看起来像导入失败。
        })
    }, 1000)
    return () => window.clearInterval(timer)
  }, [jobId, jobDone])

  const pickFile = useCallback((f: File | null) => {
    setFormError(null)
    if (!f) return
    const cap = maxBytesFor(f)
    if (f.size > cap) {
      // 本地先拦一道：让用户不必等传完才被告知超限。
      setFormError(
        `文件 ${(f.size / 1024 / 1024).toFixed(1)}MB 超过 ${cap / 1024 / 1024}MB 上限`,
      )
      setFile(null)
      return
    }
    setFile(f)
    // 书名预填。**epub 不预填**：它的文件名是「书名 (作者) (来源站...).epub」
    // 这种一长串，而书里自带一个干净得多的书名——留空让后端用它。
    if (!title.trim() && !isEpub(f)) {
      // 文件名的扩展名去掉——它是给操作系统看的，不是标题的一部分。
      setTitle(f.name.replace(/\.[^.]+$/, ''))
    }
  }, [title])

  const submit = useCallback(async () => {
    if (!file || uploading) return
    setUploading(true)
    setFormError(null)
    try {
      const resp = await api.createKbImport({
        file, library, title, author, discipline,
        use_llm: useLlm, allow_partial: allowPartial,
      })
      setDemo(resp.demo)
      setJob(resp.job)
    } catch (err) {
      setFormError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setUploading(false)
    }
  }, [file, uploading, library, title, author, discipline, useLlm, allowPartial])

  const confirmStart = useCallback(async () => {
    if (!job) return
    try {
      const next = await api.startKbImport(job.id)
      // 服务端返回的是**开跑之前**的那一行（真正执行的是后台任务），
      // 所以 status 还是 queued。原样塞回去会让用户又看到一次确认页，
      // 而这次没有任何按钮能再点。本地把 needs_confirm 抹掉，
      // 界面立刻切到进度态；一拍之后轮询拿到的是真状态。
      setJob({ ...next, needs_confirm: false })
    } catch (err) {
      setFormError(err instanceof ApiError ? err.message : String(err))
    }
  }, [job])

  const cancel = useCallback(async () => {
    if (!job) return
    try {
      await api.cancelKbImport(job.id)
      setJob(await api.getKbImport(job.id))
    } catch {
      // 取消失败不值得打断用户——作业要么已经在跑，要么已经结束。
    }
  }, [job])

  const reset = useCallback(() => {
    setJob(null)
    setFile(null)
    setTitle('')
    setAuthor('')
    setDiscipline('')
    setFormError(null)
    setUseLlm(true)
    setAllowPartial(false)
  }, [])

  return (
    <Modal
      open={open}
      onClose={onClose}
      width="max-w-[640px]"
      title="知识库"
      subtitle="导入本机的一个文本文件或 epub 电子书，切分、打标、向量化后成为可检索的内容。"
      footer={<Footer
        phase={phase}
        uploading={uploading}
        canSubmit={Boolean(file)}
        onClose={onClose}
        onSubmit={() => void submit()}
        onConfirm={() => void confirmStart()}
        onCancel={() => void cancel()}
        onReset={reset}
      />}
    >
      {phase === 'form' ? (
        <FormFields
          file={file}
          library={library}
          title={title}
          author={author}
          discipline={discipline}
          useLlm={useLlm}
          allowPartial={allowPartial}
          error={formError}
          onPickFile={pickFile}
          onLibrary={setLibrary}
          onTitle={setTitle}
          onAuthor={setAuthor}
          onDiscipline={setDiscipline}
          onUseLlm={setUseLlm}
          onAllowPartial={setAllowPartial}
        />
      ) : (
        <JobView job={job!} demo={demo} error={formError} />
      )}

      <KbSourceList version={sourcesVersion} active={open} />
    </Modal>
  )
}

/* ------------------------------------------------------------------ */

function FormFields(props: {
  file: File | null
  library: LibraryKey
  title: string
  author: string
  discipline: string
  useLlm: boolean
  allowPartial: boolean
  error: string | null
  onPickFile: (f: File | null) => void
  onLibrary: (v: LibraryKey) => void
  onTitle: (v: string) => void
  onAuthor: (v: string) => void
  onDiscipline: (v: string) => void
  onUseLlm: (v: boolean) => void
  onAllowPartial: (v: boolean) => void
}) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  return (
    <div className="flex flex-col gap-3.5">
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          props.onPickFile(e.dataTransfer.files?.[0] ?? null)
        }}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer rounded-lg border border-dashed px-4 py-5 text-center transition-colors ${
          dragging ? 'border-accent-line bg-accent-soft' : 'border-line-strong bg-paper hover:bg-paper-sunk'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".txt,.md,.markdown,.epub,text/plain,application/epub+zip"
          className="hidden"
          onChange={(e) => props.onPickFile(e.target.files?.[0] ?? null)}
        />
        {props.file ? (
          <>
            <div className="text-ink text-[13px] break-all">{props.file.name}</div>
            <div className="text-ink-faint mt-0.5 text-[11px]">
              {(props.file.size / 1024).toFixed(0)} KB · 点击可重新选择
            </div>
          </>
        ) : (
          <>
            <div className="text-ink-soft text-[13px]">点击选择文件，或把文本文件 / 电子书拖到这里</div>
            <div className="text-ink-faint mt-0.5 text-[11px]">
              纯文本（.txt / .md，UTF-8 或 GBK，最大 5MB）或电子书（.epub，最大 20MB）
            </div>
          </>
        )}
      </div>

      <Field label="归入">
        <div className="flex flex-wrap gap-1.5">
          {KB_OPTIONS.map((o) => (
            <button
              key={o.key}
              type="button"
              aria-pressed={props.library === o.key}
              onClick={() => props.onLibrary(o.key)}
              className={`rounded-md border px-2.5 py-[5px] text-[12px] transition-colors ${
                props.library === o.key
                  ? 'border-accent-line bg-accent-soft text-accent'
                  : 'border-line text-ink-faint hover:text-ink-soft'
              }`}
            >
              {o.label}
            </button>
          ))}
        </div>
        <Hint>
          {props.library === 'psychology'
            ? '抽象论述类文本选这个：标签抽的是「概念 + 关键词」，不要求正文里有意象。'
            : '文学与诗词要求每段能抽出意象或情感，抽不到的段落会被丢弃——哲学、社科类文本在这里可能只剩一小部分。'}
        </Hint>
      </Field>

      <div className="grid grid-cols-2 gap-3">
        <Field label="书名 / 篇名" hint="epub 留空则用电子书自带的书名 / 作者">
          <TextInput value={props.title} onChange={props.onTitle} placeholder="如：存在与时间" />
        </Field>
        <Field label="作者">
          <TextInput value={props.author} onChange={props.onAuthor} placeholder="如：海德格尔" />
        </Field>
      </div>

      <Field label="领域" hint="仅心理学库使用，会写进检索文本">
        <TextInput
          value={props.discipline}
          onChange={props.onDiscipline}
          placeholder="如：存在主义哲学"
        />
      </Field>

      <div className="border-line flex flex-col gap-2 border-t pt-3">
        <Toggle
          checked={props.useLlm}
          onChange={props.onUseLlm}
          label="用 AI 抽取标签"
          hint="关掉则只做规则抽取。正文照样入库，但标签质量弱，检索命中率会低一截。"
        />
        <Toggle
          checked={props.allowPartial}
          onChange={props.onAllowPartial}
          label="打标失败时仍然导入"
          hint="默认关闭。开着的话，大面积打标失败的书会带着粗糙标签入库，之后每次检索都可能把它当成证据召回。"
        />
      </div>

      {props.error ? <ErrorLine text={props.error} /> : null}
    </div>
  )
}

function JobView({
  job,
  demo,
  error,
}: {
  job: KbImportJob
  demo: boolean
  error: string | null
}) {
  if (job.needs_confirm && !job.is_terminal) {
    return (
      <div className="flex flex-col gap-3">
        <div className="border-warn/30 bg-warn-soft text-warn rounded-md border px-3.5 py-3 text-[12.5px]">
          这个文件切出了 <b>{job.chunk_total}</b> 段，超过确认阈值。
        </div>
        <ul className="text-ink-soft flex flex-col gap-1 text-[12.5px]">
          <li>· 每段都要调一次模型打标，预计需要 {estimate(job.chunk_total)}。</li>
          <li>· 期间可以关闭这个窗口，进度会留在服务器上；重新打开仍然看得到。</li>
          <li>· 不点继续的话什么都不会发生，不会消耗任何模型调用。</li>
        </ul>
        {error ? <ErrorLine text={error} /> : null}
      </div>
    )
  }

  const pct = Math.min(100, Math.max(0, job.progress))
  const done = job.is_terminal
  const ok = job.status === 'succeeded'

  return (
    <div className="flex flex-col gap-3">
      <div>
        <div className="mb-1.5 flex items-baseline justify-between gap-3">
          <span className="text-ink text-[13px]">
            {done ? (ok ? '导入完成' : job.status === 'cancelled' ? '已取消' : '导入失败') : job.stage_label || '处理中'}
          </span>
          <span className="text-ink-faint font-mono text-[11px] tabular-nums">{pct}%</span>
        </div>
        <div className="bg-paper-sunk h-1.5 overflow-hidden rounded-full">
          <div
            className={`h-full rounded-full transition-[width] duration-500 ease-out ${
              ok || !done ? 'bg-accent' : done ? 'bg-danger' : 'bg-accent'
            }`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      <div className="text-ink-soft flex items-center gap-4 text-[12px]">
        <Counter label="已切分" value={job.chunk_total} />
        <Counter label="已打标" value={job.chunk_tagged} />
        <Counter label="已入库" value={job.chunk_indexed} />
        {!done ? <span className="text-ink-faint truncate">{job.message ?? ''}</span> : null}
      </div>

      {(job.error || error) && !ok ? <ErrorLine text={job.error || error || ''} /> : null}

      {done && ok && job.chunk_indexed === 0 ? (
        // 「成功但库里是空的」是这条链路上最像成功的一种失败，必须显式说出来。
        <ErrorLine text="这次导入没有写入任何内容。若是文学/诗词库，通常是这段文字里找不到可标注的意象或情感——换成心理学库再试。" />
      ) : null}

      {job.warnings.length > 0 ? (
        <ul className="border-line text-ink-soft flex flex-col gap-1 border-t pt-2.5 text-[12px]">
          {job.warnings.map((w, i) => (
            <li key={i}>· {w}</li>
          ))}
        </ul>
      ) : null}

      {demo && ok ? (
        <div className="text-warn text-[11.5px]">
          演示模式：标签由规则抽取而非真实模型生成，检索效果仅供展示。
        </div>
      ) : null}

      {!done ? (
        <div className="text-ink-faint text-[11.5px]">
          可以关掉这个窗口，导入会在后台继续；刷新页面后进度也不会丢。
        </div>
      ) : null}
    </div>
  )
}

function Footer({
  phase, uploading, canSubmit, onClose, onSubmit, onConfirm, onCancel, onReset,
}: {
  phase: Phase
  uploading: boolean
  canSubmit: boolean
  onClose: () => void
  onSubmit: () => void
  onConfirm: () => void
  onCancel: () => void
  onReset: () => void
}) {
  if (phase === 'form') {
    return (
      <>
        <Button onClick={onClose}>取消</Button>
        <Button primary disabled={!canSubmit || uploading} onClick={onSubmit}>
          {uploading ? '上传中…' : '开始导入'}
        </Button>
      </>
    )
  }
  if (phase === 'confirm') {
    return (
      <>
        <Button onClick={onCancel}>放弃</Button>
        <Button primary onClick={onConfirm}>继续导入</Button>
      </>
    )
  }
  if (phase === 'running') {
    return (
      <>
        <Button onClick={onClose}>后台运行</Button>
        <Button onClick={onCancel}>取消导入</Button>
      </>
    )
  }
  return (
    <>
      <Button onClick={onReset}>再导一个</Button>
      <Button primary onClick={onClose}>完成</Button>
    </>
  )
}

/* --------------------------- 小组件 --------------------------- */

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-ink-soft text-[12px]">
        {label}
        {hint ? <span className="text-ink-ghost ml-1.5 text-[11px]">{hint}</span> : null}
      </span>
      {children}
    </label>
  )
}

function Hint({ children }: { children: ReactNode }) {
  return <span className="text-ink-faint mt-0.5 text-[11px] leading-relaxed">{children}</span>
}

function TextInput({
  value, onChange, placeholder,
}: {
  value: string
  onChange: (v: string) => void
  placeholder?: string
}) {
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      spellCheck={false}
      className="border-line bg-paper focus:border-accent-line focus:bg-surface rounded-md border px-2.5 py-[6px] text-[12.5px] outline-none transition-colors placeholder:text-ink-ghost"
    />
  )
}

function Toggle({
  checked, onChange, label, hint,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  label: string
  hint?: string
}) {
  return (
    <label className="flex cursor-pointer items-start gap-2">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="accent-accent mt-[3px] size-3.5 shrink-0"
      />
      <span className="min-w-0">
        <span className="text-ink text-[12.5px]">{label}</span>
        {hint ? (
          <span className="text-ink-faint mt-0.5 block text-[11px] leading-relaxed">{hint}</span>
        ) : null}
      </span>
    </label>
  )
}

function Button({
  children, onClick, primary, disabled,
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

function Counter({ label, value }: { label: string; value: number }) {
  return (
    <span className="text-ink-faint">
      {label} <span className="text-ink font-mono tabular-nums">{value}</span>
    </span>
  )
}

function ErrorLine({ text }: { text: string }) {
  return (
    <div className="border-danger/25 bg-danger-soft text-danger rounded-md border px-3 py-2 text-[12px] leading-relaxed">
      {text}
    </div>
  )
}

/**
 * 段数 → 粗略耗时。
 *
 * 刻意给的是区间而不是一个数：真实速度取决于模型与限流，一个精确到分钟的
 * 估计值在实际情况里错得很离谱，反而让人以为卡住了。段数是确定的信息，
 * 把它放在前面。
 */
function estimate(chunks: number): string {
  const lo = Math.max(1, Math.round((chunks * 0.1) / 60))
  const hi = Math.max(lo + 1, Math.round((chunks * 0.3) / 60))
  return `${lo}–${hi} 分钟`
}
