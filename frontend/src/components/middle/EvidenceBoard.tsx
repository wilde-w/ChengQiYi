/**
 * 中栏的证据卡：检索回来的心理学机制与文学典故。
 *
 * 每张卡回答四件事：**是什么**（标题/作者）、**原文说了什么**、**凭什么被选中**
 * （检索路径与相似度）、**它服务哪个主题**。第三件是这个产品与「让模型随便
 * 引一句诗」的分界线——`match_reason` 与展开后的 trace 全部来自后端检索时
 * 写下的实际参数，不是模型事后编的说明。
 *
 * 展开区展示的 9 个 trace 键（`payload` 里平铺的那些）是同一份数据的另一种
 * 粒度。折叠起来时它们一个都不显示：用户读结论时不需要知道自己被 `q3` 的第
 * 2 名命中，只有当他开始怀疑这条证据时，这些数字才有意义。
 */

import { useState } from 'react'

import type { EvidenceOut } from '../../api/types'
import { KB_OPTIONS } from '../layout/TopBar'
import { selectEvidence, useRunStore } from '../../store/runStore'

const LIBRARY_ORDER = ['psychology', 'literature', 'poetry'] as const

const LIBRARY_LABEL: Record<string, string> = Object.fromEntries(
  KB_OPTIONS.map((o) => [o.key, o.label]),
)

/** 后端 `RetrievalPath` 的中文说法。两条路含义不同，不该都叫「检索」。 */
const PATH_LABEL: Record<string, string> = {
  vector: '语义相似',
  graph: '图谱关联',
}

export function EvidenceBoard() {
  const evidence = useRunStore(selectEvidence)
  if (!evidence.length) return null

  const groups: { key: string; label: string; items: EvidenceOut[] }[] = LIBRARY_ORDER.map(
    (key) => ({
      key,
      label: LIBRARY_LABEL[key] ?? key,
      items: evidence.filter((e) => e.library === key),
    }),
  ).filter((g) => g.items.length > 0)

  // 后端可能返回三个库之外的 library（换语料时会）。丢掉它们等于悄悄少几张卡，
  // 所以单独兜一组，让「界面上的条数」和「后端的条数」永远对得上。
  const known = new Set(LIBRARY_ORDER as readonly string[])
  const others = evidence.filter((e) => !known.has(e.library))
  if (others.length) groups.push({ key: 'other', label: '其他', items: others })

  const graphCount = evidence.filter((e) => e.retrieval_path === 'graph').length

  return (
    <section className="animate-rise px-4 py-3">
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <span className="text-ink-soft text-[12px] font-medium">检索到的证据</span>
        <span className="text-ink-faint text-[11.5px]">
          共 {evidence.length} 条
          {graphCount > 0 && `，其中 ${graphCount} 条来自图谱关联`}
        </span>
      </div>

      <div className="mt-2.5 space-y-3">
        {groups.map((group) => (
          <div key={group.key}>
            <div className="text-ink-ghost mb-1.5 text-[11px]">
              {group.label} · {group.items.length}
            </div>
            <div className="space-y-2">
              {group.items.map((item) => (
                <EvidenceCard key={item.chunk_id} evidence={item} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

function EvidenceCard({ evidence }: { evidence: EvidenceOut }) {
  const [open, setOpen] = useState(false)
  const trace = evidence.payload as Record<string, unknown>
  const hasText = Boolean(evidence.text)

  return (
    // 推理卡的「典故」那一行会滚到这里。id 用 chunk_id 而不是 rank——
    // rank 会随重跑变化，而跳转链接是在渲染那一刻生成的。
    <article
      id={anchorId(evidence.chunk_id)}
      className="border-line bg-surface rounded-lg border px-3 py-2.5"
    >
      <header className="flex items-start gap-2">
        <span className="text-ink-ghost mt-[1px] shrink-0 font-mono text-[11px] tabular-nums">
          {String(trace.ev_id ?? '')}
        </span>
        <h3 className="text-ink min-w-0 flex-1 text-[12.5px] leading-snug font-medium">
          {evidence.title || evidence.chunk_id}
        </h3>
        <span className="text-ink-ghost shrink-0 text-[10.5px]">
          {PATH_LABEL[evidence.retrieval_path] ?? evidence.retrieval_path}
        </span>
      </header>

      {(evidence.author || evidence.source) && (
        <div className="text-ink-faint mt-0.5 pl-[22px] text-[11px]">
          {[evidence.author, evidence.source].filter(Boolean).join(' · ')}
        </div>
      )}

      {hasText ? (
        <p
          className={`text-ink-soft mt-1.5 pl-[22px] text-[12px] leading-relaxed break-words ${
            open ? '' : 'line-clamp-3'
          }`}
        >
          {evidence.text}
        </p>
      ) : (
        <p className="text-ink-ghost mt-1.5 pl-[22px] text-[11.5px]">这条语料没有正文。</p>
      )}

      {evidence.match_reason && (
        <div className="text-ink-ghost mt-1.5 pl-[22px] text-[11px]">{evidence.match_reason}</div>
      )}

      <div className="mt-1.5 flex items-center gap-2 pl-[22px]">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="text-ink-faint hover:text-ink-soft text-[11px] underline-offset-2 hover:underline"
          aria-expanded={open}
        >
          {open ? '收起' : '怎么选中的'}
        </button>
        {evidence.rank > 0 && (
          <span className="text-ink-ghost font-mono text-[10.5px] tabular-nums">
            融合分 {evidence.score.toFixed(3)}
          </span>
        )}
      </div>

      {open && <Trace payload={trace} evidence={evidence} />}
    </article>
  )
}

/** 展开区。九项全部来自后端检索时写下的实际参数。 */
function Trace({ payload, evidence }: { payload: Record<string, unknown>; evidence: EvidenceOut }) {
  const rows: [string, string][] = [
    ['出自查询', fmt(payload.query)],
    ['检索路径', PATH_LABEL[evidence.retrieval_path] ?? evidence.retrieval_path],
    ['向量相似', fmt(payload.vector_score)],
    ['图谱权重', fmt(payload.graph_weight)],
    ['融合得分', fmt(payload.rrf_score)],
    ['库内名次', fmt(payload.rank_in_library)],
    ['配额位次', payload.quota_slot == null ? '回填补位' : fmt(payload.quota_slot)],
    ['概念桥', fmt(payload.concept_bridge)],
    ['语料 id', evidence.chunk_id],
  ]
  return (
    <dl className="border-line mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 border-t pt-2 pl-[22px]">
      {rows.map(([label, value]) => (
        <div key={label} className="col-span-2 grid grid-cols-subgrid">
          <dt className="text-ink-ghost text-[11px]">{label}</dt>
          <dd className="text-ink-faint truncate font-mono text-[11px]" title={value}>
            {value}
          </dd>
        </div>
      ))}
    </dl>
  )
}

/** `null` / `undefined` 在这里是**有信息量的**：它说明这条证据不是走那条路来的。 */
function fmt(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'number') return String(value)
  return String(value)
}

/** 跳到某条证据卡。两个组件都在中栏同一个滚动容器里。 */
export function anchorId(chunkId: string): string {
  return `ev-${chunkId}`
}

export function scrollToEvidence(chunkId: string): void {
  const node = document.getElementById(anchorId(chunkId))
  node?.scrollIntoView({ block: 'center', behavior: 'smooth' })
}
