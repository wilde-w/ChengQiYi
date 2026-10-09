/**
 * 中栏的心理侧写卡。
 *
 * 卡片上有**两份**主导情绪，并且刻意并排放：
 *   - 分布条：Python 按簇的**人数**加权算出来的，可复算；
 *   - 模型判断：模型读完代表评论后给的一组标签。
 *
 * 两者不一致是常态，也是信息——代表评论按点赞挑，本身就偏向最响的声音。
 * 「模型认为的主导情绪」和「按人数算的主导情绪」指向不同的东西时，
 * 那说明嗓门最大的人和人数最多的人关心的事不一样。只留一份的话，
 * 这个差别会被悄悄抹掉，而用户以为看到的是全部人的情绪。
 *
 * 降级时（`model === 'fallback'`）必须挂个牌子：那份侧写是簇标签拼的，
 * 没有张力、没有解读，长得却和正常侧写一模一样。
 */

import type { ProfileOut } from '../../api/types'
import { TAG_COLORS } from '../../lib/theme'
import { selectProfile, useRunStore } from '../../store/runStore'

type Dimension = 'emotions' | 'topics' | 'needs'

const DIMENSION_LABEL: Record<Dimension, string> = {
  emotions: '情绪',
  topics: '主题',
  needs: '深层需求',
}

export function ProfileCard() {
  const profile = useRunStore(selectProfile)
  if (!profile) return null

  const empty =
    !profile.emotions.length && !profile.topics.length && !profile.needs.length

  return (
    <section className="animate-rise border-line mx-4 my-3 rounded-lg border bg-surface px-3.5 py-3">
      <header className="flex items-baseline gap-2">
        <span className="text-ink-soft text-[12px] font-medium">心理侧写</span>
        {profile.model === 'fallback' && (
          <span
            className="text-warn border-warn/30 bg-warn-soft rounded-full border px-1.5 py-[1px] text-[10.5px]"
            title="模型未能参与，这一份由各主题的标签汇总而成：没有核心张力，也没有深层解读。"
          >
            由标签汇总
          </span>
        )}
        {profile.model === 'none' && (
          <span className="text-ink-ghost text-[10.5px]">（未生成）</span>
        )}
      </header>

      {profile.core_tension && (
        <p className="font-literary text-ink mt-2 text-[14px] leading-relaxed">
          {profile.core_tension}
        </p>
      )}

      {empty ? (
        <p className="text-ink-faint mt-2 text-[12px] leading-relaxed">
          {profile.core_tension
            ? '没有可统计的主题标签。'
            : '这一条视频还没有生成侧写。'}
        </p>
      ) : (
        <div className="mt-2.5 space-y-3">
          {(['emotions', 'topics', 'needs'] as Dimension[]).map((dim) => (
            <Distribution key={dim} dimension={dim} items={profile[dim]} />
          ))}
        </div>
      )}

      <ModelTags tags={profile.global_tags} />

      {profile.summary && (
        <p className="text-ink-soft border-line mt-2.5 border-t pt-2 text-[12px] leading-relaxed">
          {profile.summary}
        </p>
      )}
    </section>
  )
}

function Distribution({
  dimension,
  items,
}: {
  dimension: Dimension
  items: ProfileOut['emotions']
}) {
  if (!items.length) return null
  // 宽度按本组最大值归一，各组之间不横比——三条分布的量纲不同。
  const max = Math.max(...items.map((i) => i.value), 1)
  const tone = TAG_COLORS[
    dimension === 'emotions' ? 'emotion' : dimension === 'topics' ? 'topic' : 'need'
  ]

  return (
    <div>
      <div className="text-ink-faint mb-1 text-[11px]">
        {DIMENSION_LABEL[dimension]}
        <span className="text-ink-ghost ml-1">按人数加权</span>
      </div>
      <ul className="space-y-[3px]">
        {items.map((item) => (
          <li key={item.label} className="flex items-center gap-2">
            <span className="text-ink-soft w-[7.5em] shrink-0 truncate text-[11.5px]">
              {item.label}
            </span>
            <span
              className="h-[7px] min-w-[2px] rounded-full transition-[width] duration-500 ease-out"
              style={{ width: `${(item.value / max) * 100}%`, background: tone.fg, opacity: 0.75 }}
              aria-hidden
            />
            <span className="text-ink-ghost ml-auto shrink-0 font-mono text-[10.5px] tabular-nums">
              {item.value}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/**
 * 模型给的那一组标签。空的维度不占位——留一行「（无）」只是噪音。
 */
function ModelTags({ tags }: { tags: Record<string, string[]> }) {
  const rows = (
    [
      ['emotion', '情绪', tags.emotion],
      ['topic', '主题', tags.topic],
      ['need', '需求', tags.need],
    ] as const
  )
    .map(([key, label, values]) => ({ key, label, values: values ?? [] }))
    .filter((r) => r.values.length > 0)

  if (!rows.length) return null

  return (
    <div className="border-line mt-2.5 border-t pt-2">
      <div className="text-ink-ghost mb-1 text-[11px]">模型判断</div>
      <div className="space-y-1">
        {rows.map((row) => (
          <div key={row.key} className="flex flex-wrap items-center gap-1">
            <span className="text-ink-ghost mr-0.5 text-[11px]">{row.label}</span>
            {row.values.map((v) => (
              <span
                key={v}
                className="rounded-full border px-1.5 py-[1px] text-[11px] leading-tight"
                style={{
                  background: TAG_COLORS[row.key].bg,
                  color: TAG_COLORS[row.key].fg,
                  borderColor: TAG_COLORS[row.key].border,
                }}
              >
                {v}
              </span>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}
