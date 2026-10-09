/**
 * 中栏的流水线进度。
 *
 * 这个面板回答用户的唯一问题是「现在跑到哪一步了，还要多久」。
 * 因此它有两层：上面的**节点清单**（全景，一眼看出还剩几步）和下面的
 * **事件时间线**（细节，带真实数字的里程碑）。
 *
 * 节点清单在运行开始时就全部渲染成灰色——让用户提前知道总共七步，
 * 比让他盯着一个慢慢长出来的列表更有耐心。
 */

import { useEffect, useRef } from 'react'

import type { NodeSpec } from '../../api/types'
import { useRunStore } from '../../store/runStore'
import type { TimelineItem } from '../../store/applyEvent'
import { EmptyState } from '../common/EmptyState'

export function ProgressStream() {
  const specs = useRunStore((s) => s.specs)
  const nodes = useRunStore((s) => s.nodes)
  const timeline = useRunStore((s) => s.timeline)
  const status = useRunStore((s) => s.status)
  const progress = useRunStore((s) => s.progress)
  const reconnected = useRunStore((s) => s.reconnected)

  if (!specs.length) {
    return (
      <EmptyState
        icon="◈"
        title="正在加载流水线定义"
        hint="节点分带与文案由后端 /meta/pipeline 下发，不在前端硬编码。"
      />
    )
  }

  return (
    <div className="px-4 py-3">
      {reconnected && (
        <div className="border-line bg-paper-sunk text-ink-faint mb-3 rounded-md border px-3 py-2 text-[11px]">
          连接曾中断，已按事件序号续上——上方进度是连续的，没有跳段。
        </div>
      )}

      <NodeChecklist specs={specs} nodes={nodes} />

      <div className="border-line mt-4 border-t pt-3">
        <div className="text-ink-ghost mb-2 text-[11px] tracking-wide">过程记录</div>
        <Timeline items={timeline} />
      </div>

      {status === 'running' && progress >= 99 && (
        <div className="text-ink-faint mt-3 text-[11px]">正在收尾…</div>
      )}
    </div>
  )
}

function NodeChecklist({
  specs,
  nodes,
}: {
  specs: NodeSpec[]
  nodes: Record<string, string>
}) {
  return (
    <ol className="space-y-[3px]">
      {specs.map((spec, i) => {
        const state = nodes[spec.key] ?? 'pending'
        return (
          <li
            key={spec.key}
            className={`flex items-center gap-2.5 rounded-md px-2 py-[6px] text-[12.5px] transition-colors ${
              state === 'running' ? 'bg-accent-soft' : ''
            }`}
          >
            <NodeMark state={state} index={i + 1} />
            <span
              className={
                state === 'pending'
                  ? 'text-ink-ghost'
                  : state === 'running'
                    ? 'text-accent font-medium'
                    : 'text-ink-soft'
              }
            >
              {spec.label}
            </span>
            <span className="text-ink-ghost ml-auto font-mono text-[10.5px] tabular-nums">
              {spec.start}–{spec.end}
            </span>
          </li>
        )
      })}
    </ol>
  )
}

function NodeMark({ state, index }: { state: string; index: number }) {
  if (state === 'done') {
    return (
      <span className="bg-ok/10 text-ok flex size-[18px] shrink-0 items-center justify-center rounded-full text-[10px]">
        ✓
      </span>
    )
  }
  if (state === 'running') {
    return (
      <span className="border-accent text-accent flex size-[18px] shrink-0 items-center justify-center rounded-full border text-[10px]">
        <span className="animate-pulse-soft font-mono">{index}</span>
      </span>
    )
  }
  if (state === 'failed') {
    return (
      <span className="bg-danger-soft text-danger flex size-[18px] shrink-0 items-center justify-center rounded-full text-[10px]">
        ✕
      </span>
    )
  }
  return (
    <span className="border-line text-ink-ghost flex size-[18px] shrink-0 items-center justify-center rounded-full border text-[10px]">
      {index}
    </span>
  )
}

/** 自动滚到底部——运行中用户不需要手动跟。 */
function Timeline({ items }: { items: TimelineItem[] }) {
  const ref = useRef<HTMLDivElement>(null)
  const pinnedRef = useRef(true)

  useEffect(() => {
    const el = ref.current
    if (!el || !pinnedRef.current) return
    el.scrollTop = el.scrollHeight
  }, [items.length])

  if (!items.length) {
    return <div className="text-ink-ghost py-2 text-[11.5px]">尚无事件</div>
  }

  return (
    <div
      ref={ref}
      onScroll={(e) => {
        const el = e.currentTarget
        // 用户往上翻时不再自动滚动——否则他读不到任何东西。
        // 留 24px 容差，免得「差一像素到底」被判定成手动滚动。
        pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24
      }}
      className="max-h-[46vh] space-y-1 overflow-y-auto pr-1"
    >
      {items.map((item) => (
        <TimelineRow key={item.id} item={item} />
      ))}
    </div>
  )
}

function TimelineRow({ item }: { item: TimelineItem }) {
  const tone =
    item.kind === 'error'
      ? 'text-danger'
      : item.kind === 'warning'
        ? 'text-warn'
        : item.kind === 'node'
          ? 'text-ink-faint'
          : 'text-ink-soft'

  const glyph =
    item.kind === 'error' ? '✕' : item.kind === 'warning' ? '!' : item.kind === 'node' ? '▸' : '·'

  return (
    <div className={`animate-rise flex gap-2 text-[11.5px] leading-relaxed ${tone}`}>
      <span className="text-ink-ghost w-3 shrink-0 text-center font-mono">{glyph}</span>
      <span className="min-w-0 flex-1">{item.text}</span>
    </div>
  )
}
