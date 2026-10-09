import { useCallback, useEffect, useState } from 'react'

import { api, ApiError } from '../../api/endpoints'
import type { KbSource, KbSourceText } from '../../api/types'
import { Button, ErrorLine, Hint, ResultBlock } from './textUi'

/** 一次取多少字。后端上限 500000；6 万字大约是 60–80KB，DOM 毫无压力。 */
const PAGE = 60_000

/**
 * 观心自己的知识库：导入过的来源 → 原文全文。
 *
 * 与「＋ 导入」里的片段预览分工不同：那边答的是「RAG 切出了什么」（120 字
 * 预览 + chunk_id，用来判断切分质量），这里答的是「我导入的到底是本书吗」。
 * 两条都要留着，它们回答的不是同一个问题。
 *
 * 原文读的是**导入时落盘的正文**，不是把 Qdrant 里的 chunk 拼起来——
 * chunk 之间没有任何序号字段，拼出来的顺序是乱的。
 */
export function KbSourceTextPane({ active }: { active: boolean }) {
  const [sources, setSources] = useState<KbSource[] | null>(null)
  const [listError, setListError] = useState<string | null>(null)

  const [picked, setPicked] = useState<string | null>(null)
  const [doc, setDoc] = useState<KbSourceText | null>(null)
  /** 已加载的**字符数**（用后端那套口径，见 `codePoints`）。 */
  const [loaded, setLoaded] = useState(0)
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadList = useCallback(async () => {
    try {
      setSources(await api.listKbSources())
      setListError(null)
    } catch (err) {
      // 后端没起来是常态（前端可以独立启动），失败就说一句，不弹错。
      setListError(err instanceof ApiError ? err.message : String(err))
    }
  }, [])

  useEffect(() => {
    if (!active) return
    void loadList()
  }, [active, loadList])

  const fetchChunk = useCallback(async (sourceFile: string, offset: number, replace: boolean) => {
    setBusy(true)
    setError(null)
    try {
      const next = await api.kbSourceText(sourceFile, offset, PAGE)
      setDoc(next)
      setLoaded(next.offset + codePoints(next.text))
      setText((prev) => (replace ? next.text : prev + next.text))
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }, [])

  const open = useCallback(
    (sourceFile: string) => {
      setPicked(sourceFile)
      setDoc(null)
      setText('')
      setLoaded(0)
      void fetchChunk(sourceFile, 0, true)
    },
    [fetchChunk],
  )

  return (
    <div className="flex flex-col gap-3">
      <Hint>
        这里只列**从界面导入**的内容。随仓库交付的语料文件不在内——它们会在下一次
        <span className="font-mono"> ingest-kb </span>时被重新装回来，列出来会让人以为可以删。
      </Hint>

      {listError ? <ErrorLine text={listError} /> : null}

      {sources && sources.length === 0 ? (
        <div className="text-ink-faint text-[12px]">还没有导入过内容。用顶栏的「＋ 导入」加一个。</div>
      ) : null}

      <ul className="divide-line border-line max-h-[180px] divide-y overflow-y-auto rounded-md border">
        {(sources ?? []).map((s) => (
          <li key={`${s.source_file}:${s.library}`}>
            <button
              type="button"
              onClick={() => open(s.source_file)}
              aria-pressed={picked === s.source_file}
              className={`w-full px-3 py-2 text-left transition-colors ${
                picked === s.source_file ? 'bg-accent-soft' : 'hover:bg-paper-sunk'
              }`}
            >
              <div className="text-ink truncate text-[12.5px]">{s.source_file}</div>
              <div className="text-ink-faint text-[11px]">
                {s.library_label} · 入库 {s.chunks} 条
              </div>
            </button>
          </li>
        ))}
      </ul>

      {error ? <ErrorLine text={error} /> : null}

      {picked && text ? (
        <div>
          <div className="flex items-baseline justify-between gap-3">
            <div className="text-ink-soft min-w-0 truncate text-[12px]">
              {doc?.title || picked}
              {doc?.author ? <span className="text-ink-faint"> · {doc.author}</span> : null}
            </div>
            <span className="text-ink-ghost shrink-0 text-[11px] tabular-nums">
              已加载 {loaded.toLocaleString()} / {(doc?.char_count ?? 0).toLocaleString()} 字
            </span>
          </div>

          <ResultBlock text={text} />

          {doc?.truncated ? (
            <div className="mt-2 flex items-center gap-2">
              <Button disabled={busy} onClick={() => void fetchChunk(picked, loaded, false)}>
                {busy ? '读取中…' : '继续加载'}
              </Button>
              <Hint>整本红楼梦上百万字，一次性塞进页面会卡，所以按字符分页。</Hint>
            </div>
          ) : null}
        </div>
      ) : null}

      {picked && !text && busy ? (
        <div className="text-ink-faint text-[12px]">读取原文…</div>
      ) : null}
    </div>
  )
}

/**
 * 字符串的**码点数**，与后端 Python 的 `len()` 同口径。
 *
 * 不能直接用 `text.length`：那是 UTF-16 码元数，一个 emoji 在 JS 里算 2、
 * 在 Python 里算 1。用错了会让翻页的 offset 多走一格，表现为「继续加载」
 * 之后看到一小段重复的正文——一个不会报错、只能靠眼力发现的 bug。
 */
function codePoints(text: string): number {
  return [...text].length
}
