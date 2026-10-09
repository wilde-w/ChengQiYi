/**
 * 中栏的推理链：现象 → 机制 → 典故 → 洞察。
 *
 * 四层颜色来自 `theme.ts` 的 `REASONING_LAYERS`——同一种颜色在整条链上
 * 只代表一件事，用户扫一眼就知道哪几行是「观察到的事实」、哪几行是
 * 「解释」。这是推理卡与散文的区别所在，颜色不是装饰。
 *
 * **典故这一层没有自己的文字。** `ReasoningStep` 只有 phenomenon /
 * mechanism / insight 三段，典故靠 `allusion_ids` 指向证据卡，点一下滚过去。
 * 在这里把典故原文再抄一遍的话，同一段文字就有两份副本，而用户看到的会是
 * 「推理里引的那句」和「证据卡上的原文」对不上——文学段落的原文、出处、
 * 检索路径只该有一份。
 */

import { useMemo } from 'react'

import type { EvidenceOut, ReasoningStepOut } from '../../api/types'
import { REASONING_LAYERS } from '../../lib/theme'
import { scrollToEvidence } from './EvidenceBoard'
import { selectEvidence, selectReasoning, useRunStore } from '../../store/runStore'

export function ReasoningChain() {
  const reasoning = useRunStore(selectReasoning)
  const evidence = useRunStore(selectEvidence)

  const byChunk = useMemo(() => {
    const map = new Map<string, EvidenceOut>()
    for (const item of evidence) map.set(item.chunk_id, item)
    return map
  }, [evidence])

  if (!reasoning.length) return null

  return (
    <section className="animate-rise px-4 py-3">
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <span className="text-ink-soft text-[12px] font-medium">推理链</span>
        <span className="text-ink-faint text-[11.5px]">{reasoning.length} 步</span>
      </div>

      <div className="mt-2.5 space-y-2.5">
        {reasoning.map((step) => (
          <StepCard key={step.step_index} step={step} byChunk={byChunk} />
        ))}
      </div>
    </section>
  )
}

function StepCard({
  step,
  byChunk,
}: {
  step: ReasoningStepOut
  byChunk: Map<string, EvidenceOut>
}) {
  const degraded = !step.mechanism && !step.insight

  return (
    <article className="border-line bg-surface rounded-lg border px-3 py-2.5">
      <header className="mb-2 flex items-center gap-2">
        <span className="text-ink-ghost text-[11px]">第 {step.step_index} 步</span>
        <Confidence value={step.confidence} />
        {degraded && (
          // 降级步不是错误，但它必须看起来和一步正常推理不一样：
          // 否则用户会以为「只列了现象」就是这次的结论。
          <span className="text-warn text-[10.5px]">模型未给出解释</span>
        )}
      </header>

      <div className="space-y-2">
        <Layer layerKey="phenomenon">
          <Text value={step.phenomenon} fallback="—" />
        </Layer>

        <Layer layerKey="mechanism">
          <Text value={step.mechanism} fallback="尚未解释" />
          <Chips ids={step.evidence_ids} byChunk={byChunk} compact />
        </Layer>

        <Layer layerKey="allusion">
          <Allusions ids={step.allusion_ids} byChunk={byChunk} />
        </Layer>

        <Layer layerKey="insight">
          <Text value={step.insight} fallback="尚未形成洞察" />
        </Layer>
      </div>
    </article>
  )
}

function Layer({ layerKey, children }: { layerKey: string; children: React.ReactNode }) {
  const layer = REASONING_LAYERS.find((l) => l.key === layerKey)
  if (!layer) return null
  return (
    <div className="grid grid-cols-[38px_1fr] items-start gap-x-2">
      <div className="flex items-center gap-1.5 pt-[3px]">
        <span
          className="h-3 w-[2px] shrink-0 rounded-full"
          style={{ backgroundColor: layer.color }}
          aria-hidden
        />
        <span className="text-[11px]" style={{ color: layer.color }}>
          {layer.label}
        </span>
      </div>
      <div className="min-w-0">{children}</div>
    </div>
  )
}

function Text({ value, fallback }: { value: string; fallback: string }) {
  if (!value) {
    return <p className="text-ink-ghost text-[12px] italic">{fallback}</p>
  }
  return <p className="text-ink-soft text-[12.5px] leading-relaxed break-words">{value}</p>
}

/**
 * 典故层：把引用的证据作为卡片列出来。这一层**没有自己的正文**，所以
 * 证据卡就是内容本身，而不是挂在文字后面的注脚。
 *
 * 找不到对应证据时不隐藏：那种情况下「引了一个不存在的东西」正是用户
 * 最该看见的事——静默丢掉会让这一层看起来只是「这次没引典故」。
 */
function Allusions({ ids, byChunk }: { ids: string[]; byChunk: Map<string, EvidenceOut> }) {
  if (!ids.length) {
    return <p className="text-ink-ghost text-[12px] italic">这一步没有引用典故</p>
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {ids.map((id) => {
        const item = byChunk.get(id)
        if (!item) {
          return (
            <span
              key={id}
              className="border-warn/40 text-warn rounded-md border px-2 py-[3px] text-[11.5px]"
              title="这条引用在检索结果里找不到"
            >
              {id}（未找到）
            </span>
          )
        }
        return (
          <button
            key={id}
            type="button"
            onClick={() => scrollToEvidence(id)}
            title={item.text}
            className="border-line hover:border-accent-line hover:bg-paper-sunk max-w-full truncate rounded-md border px-2 py-[3px] text-left text-[11.5px] transition-colors"
          >
            <span className="text-ink">《{item.title || id}》</span>
            {item.author && <span className="text-ink-faint"> · {item.author}</span>}
          </button>
        )
      })}
    </div>
  )
}

/** 机制层后面的小引注：一行装得下，只报「引了什么」。 */
function Chips({
  ids,
  byChunk,
  compact,
}: {
  ids: string[]
  byChunk: Map<string, EvidenceOut>
  compact?: boolean
}) {
  if (!ids.length) return null
  return (
    <div className={`flex flex-wrap gap-1.5 ${compact ? 'mt-1' : ''}`}>
      {ids.map((id) => {
        const item = byChunk.get(id)
        return (
          <button
            key={id}
            type="button"
            onClick={() => scrollToEvidence(id)}
            title={item?.text ?? id}
            className="bg-paper-sunk text-ink-faint hover:text-ink-soft max-w-[220px] truncate rounded px-1.5 py-[2px] text-[10.5px] transition-colors"
          >
            {item?.title ? `《${item.title}》` : id}
          </button>
        )
      })}
    </div>
  )
}

/**
 * 可信度。**显示的是重算值，不是模型的自评**——公式三项都能对着证据卡
 * 数出来，这是这个数字敢显示的前提（见 n6 模块 docstring）。
 *
 * 用百分比而不是 0.725 这样的原始值：读的人要判断的是「这条能信多少」，
 * 百分比不需要先在脑子里换算一遍。除零不设防是因为 `confidence` 恒为
 * 0..1 的有限数，模型给的任何东西都不参与计算。
 */
function Confidence({ value }: { value: number }) {
  const tone = value >= 0.6 ? 'text-ink-faint' : value >= 0.3 ? 'text-ink-soft' : 'text-warn'
  return (
    <span
      className={`ml-auto font-mono text-[10.5px] tabular-nums ${tone}`}
      title="证据支撑 55% + 机制完整 30% + 引用典故 15%，三项都能对着下方证据卡复算"
    >
      可信度 {(value * 100).toFixed(1)}%
    </span>
  )
}
