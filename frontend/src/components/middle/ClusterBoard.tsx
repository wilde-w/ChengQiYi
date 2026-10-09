/**
 * 中栏的主题卡片。
 *
 * 每张卡回答三件事：**这一簇在说什么**（标签）、**有多少人在说**（条数）、
 * **凭什么这么分**（代表评论选出的关键词）。第三件是刻意的——模型的标签
 * 是整个界面上最好看也最不可核查的部分，关键词则是这一簇真实的字，
 * 用户扫一眼就知道标签有没有跑偏。
 *
 * 噪声与长尾用差别明显的样式：它们**不是主题**，是「没被归进任何主题的人」。
 * 画成一样的卡片会让用户以为视频底下有五个主题，其中一个是「边缘声音」。
 */

import type { ClusterOut } from '../../api/types'
import { NOISE_COLOR } from '../../lib/theme'
import { selectClusterMeta, selectClusters, useRunStore } from '../../store/runStore'
import { TAG_COLORS } from '../../lib/theme'

export function ClusterBoard() {
  const clusters = useRunStore(selectClusters)
  const meta = useRunStore(selectClusterMeta)

  if (!clusters.length) return null

  return (
    <section className="animate-rise px-4 py-3">
      <BoardHeader meta={meta} count={clusters.length} />
      <div className="mt-2.5 space-y-2.5">
        {clusters.map((cluster) => (
          <ClusterCard key={cluster.cluster_key} cluster={cluster} />
        ))}
      </div>
    </section>
  )
}

/**
 * 表头说清**这一批簇是怎么分出来的**。
 *
 * 走 KMeans 兜底时必须明说：等分出来的簇，形状没有语义聚类的可信度高，
 * 而卡片本身长得一模一样。用户有权知道自己看的哪一种。
 */
function BoardHeader({ meta, count }: { meta: Record<string, unknown>; count: number }) {
  const method = String(meta.method ?? '')
  const deduped = Number(meta.deduped ?? 0)
  const k = Number(meta.k ?? 0)

  let how: string
  if (method === 'hdbscan') how = `按语义密度聚类，找到 ${k} 个主题`
  else if (method === 'kmeans') how = '语义密度不足以分簇，改用等分聚类'
  else if (meta.reason === 'too_few_comments') how = '评论太少，未做分簇'
  else if (meta.reason) how = '未做分簇'
  else how = `找到 ${count} 组`

  return (
    <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
      <span className="text-ink-soft text-[12px] font-medium">主题分布</span>
      <span className="text-ink-faint text-[11.5px]">{how}</span>
      {deduped > 0 && (
        <span className="text-ink-ghost text-[11px]">· 合并了 {deduped} 条重复表达</span>
      )}
      {method === 'kmeans' && (
        <span
          className="text-warn border-warn/30 bg-warn-soft rounded-full border px-1.5 py-[1px] text-[10.5px]"
          title="等分聚类只保证每组条数接近，不保证每组说的是一件事。配好 embedding key 后语义聚类会接管。"
        >
          兜底
        </span>
      )}
    </div>
  )
}

function ClusterCard({ cluster }: { cluster: ClusterOut }) {
  const isNoise = cluster.is_noise
  const isTail = cluster.cluster_key === 'tail'
  const duplicates = cluster.raw_size - cluster.size

  return (
    <article
      className={`rounded-lg border px-3 py-2.5 ${
        isNoise ? 'border-dashed border-line bg-paper-sunk' : 'border-line bg-surface'
      }`}
    >
      <header className="flex items-start gap-2">
        <span
          className="mt-[6px] size-2.5 shrink-0 rounded-full"
          style={{ background: isNoise ? NOISE_COLOR : cluster.color }}
          aria-hidden
        />
        <h3
          className={`min-w-0 flex-1 text-[13px] leading-snug ${isNoise ? 'text-ink-faint' : 'text-ink font-medium'}`}
        >
          {cluster.label || '（未命名）'}
        </h3>
        <span
          className={`shrink-0 font-mono text-[11px] tabular-nums ${isNoise ? 'text-ink-ghost' : 'text-ink-faint'}`}
        >
          {cluster.size} 条
        </span>
      </header>

      {/* 去重前后不一致时说明一声，否则「30 条」里的 12 条复读看不见 */}
      {duplicates > 0 && (
        <div className="text-ink-ghost mt-1 pl-[18px] text-[11px]">
          另有 {duplicates} 条是它的重复表达
        </div>
      )}

      {isNoise && (
        <p className="text-ink-faint mt-1.5 pl-[18px] text-[11.5px] leading-relaxed">
          这些声音彼此不相似，也没有归入任何主题。它们仍然算在下面的分布里，
          只是不被当成一个「主题」。
        </p>
      )}
      {isTail && (
        <p className="text-ink-faint mt-1.5 pl-[18px] text-[11.5px] leading-relaxed">
          主题数超过卡片上限，排在后面的几组合并在这里——不是被丢掉了。
        </p>
      )}

      <div className="mt-2 space-y-1.5 pl-[18px]">
        <TagRow kind="emotion" values={cluster.emotion_tags} />
        <TagRow kind="topic" values={cluster.topic_tags} />
        <TagRow kind="need" values={cluster.need_tags} />
      </div>

      {cluster.keywords.length > 0 && (
        <div className="mt-2 pl-[18px]">
          {/* 关键词是这一簇真实的字，可核查；模型的标签不是。 */}
          <span className="text-ink-ghost mr-1.5 text-[11px]">原话</span>
          <span className="text-ink-faint font-mono text-[11px]">
            {cluster.keywords.map((k) => `「${k}」`).join(' ')}
          </span>
        </div>
      )}

      {cluster.summary && (
        <p className="text-ink-soft border-line mt-2 border-t pt-2 pl-[18px] text-[12px] leading-relaxed">
          {cluster.summary}
        </p>
      )}
    </article>
  )
}

const TAG_LABEL: Record<keyof typeof TAG_COLORS, string> = {
  emotion: '情绪',
  topic: '主题',
  need: '需求',
}

function TagRow({ kind, values }: { kind: keyof typeof TAG_COLORS; values: string[] }) {
  if (!values.length) return null
  const tone = TAG_COLORS[kind]
  return (
    <div className="flex flex-wrap items-center gap-1">
      <span className="text-ink-ghost mr-0.5 text-[11px]">{TAG_LABEL[kind]}</span>
      {values.map((v) => (
        <span
          key={v}
          className="rounded-full border px-1.5 py-[1px] text-[11px] leading-tight"
          style={{ background: tone.bg, color: tone.fg, borderColor: tone.border }}
        >
          {v}
        </span>
      ))}
    </div>
  )
}
