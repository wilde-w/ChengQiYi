/**
 * 右栏：四段洞察。
 *
 * 段落的顺序**由前端固定**（`SECTION_ORDER`），不由到达顺序决定。n7 逐段
 * 发正文，网络一抖动到达顺序就可能变；让顺序跟着到达走的话，同一份结果
 * 刷新两次会排成两个样子，而读者会以为内容变了。
 *
 * 段落没内容时**如实说「本段未生成」**，不留空白。空白在读者眼里和
 * 「还在加载」没有区别，而这两件事的处置完全不同——前者要去看 warning，
 * 后者只要等。
 */

import type { SectionKey, SectionOut } from '../../api/types'
import { selectSections, selectEvidence, useRunStore } from '../../store/runStore'
import { Markdown } from './Markdown'

/** 与后端 `SectionKey` 的声明顺序一致。 */
const SECTION_ORDER: { key: SectionKey; title: string }[] = [
  { key: 'profile', title: '心理侧写' },
  { key: 'mechanism', title: '科学机制' },
  { key: 'allusion', title: '文学类比' },
  { key: 'insight', title: '最终洞察' },
]

export function SectionBoard() {
  const sections = useRunStore(selectSections)
  const evidence = useRunStore(selectEvidence)

  const present = SECTION_ORDER.filter((s) => sections[s.key])
  const loaded = new Set(evidence.map((e) => e.chunk_id))

  return (
    <div className="divide-line divide-y">
      {present.map(({ key, title }) => (
        <SectionBlock
          key={key}
          title={sections[key]?.title || title}
          section={sections[key]!}
          loaded={loaded}
        />
      ))}
    </div>
  )
}

function SectionBlock({
  title,
  section,
  loaded,
}: {
  title: string
  section: SectionOut
  loaded: Set<string>
}) {
  const citations = section.citations ?? []
  const body = section.content_md ?? ''

  return (
    <section className="animate-rise px-4 py-3">
      <header className="mb-2 flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <h3 className="text-ink text-[13px] font-medium">{title}</h3>
        <Traceability citations={citations} coverage={section.citation_coverage} loaded={loaded} />
        {section.version > 1 && (
          <span className="text-ink-ghost text-[10.5px]">v{section.version}</span>
        )}
      </header>

      {body ? (
        <Markdown text={body} citations={citations} />
      ) : (
        <p className="text-ink-ghost text-[12px] italic">本段未生成</p>
      )}
    </section>
  )
}

/**
 * 「引用可追溯」徽章。
 *
 * `能用 / 总共` 而不是一个百分比：读者想确认的是「我在这段里看到的每一枚
 * chip，点下去是不是都有东西」，那是一个数得出来的比值，不是一个概率。
 *
 * 判据是**这条引用指向的证据还在不在当前这一版的证据列表里**。它能真的
 * 失败——重新检索过证据之后再回看旧段落，就会有几枚 chip 指向已经不在
 * 列表里的语料，点下去什么也不会发生。那种时候这个数字变小、颜色变警告，
 * 比让用户自己点几下才发现要好。
 */
function Traceability({
  citations,
  coverage,
  loaded,
}: {
  citations: SectionOut['citations']
  coverage: number
  loaded: Set<string>
}) {
  if (!citations.length) {
    return <span className="text-ink-ghost text-[10.5px]">本段无引用</span>
  }
  const resolvable = citations.filter((c) => loaded.has(c.evidence_id)).length
  const broken = resolvable < citations.length

  return (
    <span
      className={`text-[10.5px] ${broken ? 'text-warn' : 'text-ink-ghost'}`}
      title={
        broken
          ? '有引用在这一版的证据列表里找不到——可能是证据被重新检索过'
          : `正文里 ${citations.length} 处引用全部落到了证据上（覆盖率 ${(coverage * 100).toFixed(0)}%）`
      }
    >
      引用可追溯 {resolvable}/{citations.length}
    </span>
  )
}
