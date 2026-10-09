import { useCallback, useEffect, useState } from 'react'

import { api, ApiError } from '../../api/endpoints'
import type { KbSource, KbSourceDocument } from '../../api/types'

/**
 * 已导入的来源，以及每个来源切出来的片段预览。
 *
 * **只列导入的内容。** 三个随仓库交付的语料文件不在内——列在这里会让人
 * 以为可以删，而删掉之后下一次 `ingest-kb` 又会把它们装回来，看起来像
 * 删除按钮坏了。
 *
 * 展开看片段是这个面板存在的主要理由：它是用户唯一能直接看到
 * 「RAG 到底切出了什么」的地方。没有它，切分质量只能靠猜。
 */
export function KbSourceList({ version, active }: { version: number; active: boolean }) {
  const [sources, setSources] = useState<KbSource[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState<string | null>(null)

  const reload = useCallback(async () => {
    try {
      setSources(await api.listKbSources())
      setError(null)
    } catch (err) {
      // 后端没起来是常态（前端可以独立启动），失败就说一句，不弹错。
      setError(err instanceof ApiError ? err.message : String(err))
    }
  }, [])

  useEffect(() => {
    if (!active) return
    void reload()
  }, [active, reload, version])

  const onDelete = useCallback(
    async (name: string) => {
      try {
        await api.deleteKbSource(name)
        if (open === name) setOpen(null)
        await reload()
      } catch (err) {
        setError(err instanceof ApiError ? err.message : String(err))
      }
    },
    [open, reload],
  )

  return (
    <section className="border-line mt-4 border-t pt-3.5">
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-ink-soft text-[12.5px]">已导入的内容</h3>
        <span className="text-ink-ghost text-[11px]">
          {sources ? `${sources.length} 个来源` : ''}
        </span>
      </div>

      {error ? <div className="text-danger text-[12px]">{error}</div> : null}

      {sources && sources.length === 0 ? (
        <div className="text-ink-faint text-[12px]">
          还没有导入过内容。上面选一个文件就能加进来。
        </div>
      ) : null}

      <ul className="divide-line max-h-[260px] divide-y overflow-y-auto">
        {(sources ?? []).map((s) => (
          <SourceRow
            key={`${s.source_file}:${s.library}`}
            source={s}
            expanded={open === s.source_file}
            onToggle={() => setOpen(open === s.source_file ? null : s.source_file)}
            onDelete={() => void onDelete(s.source_file)}
          />
        ))}
      </ul>
    </section>
  )
}

function SourceRow({
  source, expanded, onToggle, onDelete,
}: {
  source: KbSource
  expanded: boolean
  onToggle: () => void
  onDelete: () => void
}) {
  const [confirming, setConfirming] = useState(false)

  return (
    <li className="py-2">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={expanded}
          className="text-ink-ghost hover:text-ink-soft w-3 shrink-0 text-[10px] transition-colors"
        >
          {expanded ? '▾' : '▸'}
        </button>
        <button type="button" onClick={onToggle} className="min-w-0 flex-1 text-left">
          <div className="text-ink truncate text-[12.5px]">{source.source_file}</div>
          <div className="text-ink-faint text-[11px]">
            {source.library_label} · {source.chunks} 条 · {formatTime(source.updated_at)}
          </div>
        </button>

        {confirming ? (
          // 二次确认做成行内的，不用 window.confirm：原生弹窗会阻塞整个
          // 页面，而且它长得不像这个产品的一部分。
          <span className="flex shrink-0 items-center gap-1.5 text-[11.5px]">
            <span className="text-danger">删除 {source.chunks} 条？</span>
            <button
              type="button"
              onClick={onDelete}
              className="border-danger/30 text-danger rounded border px-1.5 py-[2px]"
            >
              删除
            </button>
            <button
              type="button"
              onClick={() => setConfirming(false)}
              className="border-line text-ink-faint rounded border px-1.5 py-[2px]"
            >
              取消
            </button>
          </span>
        ) : (
          <button
            type="button"
            onClick={() => setConfirming(true)}
            className="text-ink-ghost hover:text-danger shrink-0 text-[11.5px] transition-colors"
          >
            删除
          </button>
        )}
      </div>

      {expanded ? <DocumentPreview sourceFile={source.source_file} /> : null}
    </li>
  )
}

function DocumentPreview({ sourceFile }: { sourceFile: string }) {
  const [docs, setDocs] = useState<KbSourceDocument[] | null>(null)

  useEffect(() => {
    let alive = true
    void api
      .kbSourceDocuments(sourceFile)
      .then((rows) => alive && setDocs(rows))
      .catch(() => alive && setDocs([]))
    return () => {
      alive = false
    }
  }, [sourceFile])

  if (!docs) return <div className="text-ink-ghost mt-1.5 pl-5 text-[11.5px]">读取中…</div>
  if (docs.length === 0) {
    return <div className="text-ink-ghost mt-1.5 pl-5 text-[11.5px]">没有片段</div>
  }

  return (
    <ul className="border-line mt-1.5 ml-5 flex flex-col gap-1 border-l pl-3">
      {docs.map((d) => (
        <li key={d.chunk_id} className="text-[11.5px] leading-relaxed">
          <span className="text-ink-ghost font-mono">{d.chunk_id.slice(-8)}</span>{' '}
          <span className="text-ink-soft">{d.text_preview}</span>
        </li>
      ))}
    </ul>
  )
}

function formatTime(iso?: string | null): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getMonth() + 1}月${d.getDate()}日 ${pad(d.getHours())}:${pad(d.getMinutes())}`
}
