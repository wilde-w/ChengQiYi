import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * 可拖拽列宽，宽度持久化到 localStorage。
 *
 * 三栏工作台的宽度需求随任务变化很大（读长评论 vs 看推理图），
 * 固定宽度会让某一栏永远不够用；而每次刷新都重置又等于没有这个功能。
 */

type Opts = {
  storageKey: string
  initial: number
  min?: number
  max?: number
  /** 1 = 拖拽时向右变宽（左栏）；-1 = 向左变宽（右栏） */
  direction?: 1 | -1
}

export function useResizable({
  storageKey,
  initial,
  min = 240,
  max = 760,
  direction = 1,
}: Opts) {
  const [width, setWidth] = useState<number>(() => {
    if (typeof window === 'undefined') return initial
    const raw = window.localStorage.getItem(`guanxin.col.${storageKey}`)
    const parsed = raw ? Number.parseInt(raw, 10) : Number.NaN
    return Number.isFinite(parsed) ? clamp(parsed, min, max) : initial
  })

  const [dragging, setDragging] = useState(false)
  const drag = useRef<{ startX: number; startWidth: number } | null>(null)

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      e.preventDefault()
      ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
      drag.current = { startX: e.clientX, startWidth: width }
      setDragging(true)
    },
    [width],
  )

  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      const d = drag.current
      if (!d) return
      const next = clamp(d.startWidth + (e.clientX - d.startX) * direction, min, max)
      setWidth(next)
    },
    [direction, min, max],
  )

  const onPointerUp = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!drag.current) return
      ;(e.target as HTMLElement).releasePointerCapture(e.pointerId)
      drag.current = null
      setDragging(false)
    },
    [],
  )

  // 拖动结束后才落盘，避免每帧写 localStorage
  useEffect(() => {
    if (dragging) return
    window.localStorage.setItem(`guanxin.col.${storageKey}`, String(width))
  }, [dragging, storageKey, width])

  const reset = useCallback(() => setWidth(initial), [initial])

  return { width, dragging, reset, handleProps: { onPointerDown, onPointerMove, onPointerUp } }
}

function clamp(v: number, min: number, max: number) {
  return Math.min(Math.max(v, min), max)
}

/** 拖拽手柄本体。视觉上是一条细线，hover/拖动时变宽变深。 */
export function ResizeHandle({
  dragging,
  ...props
}: { dragging: boolean } & React.ComponentProps<'div'>) {
  return (
    <div
      role="separator"
      aria-orientation="vertical"
      className={[
        'group relative z-10 w-px shrink-0 cursor-col-resize bg-line transition-colors',
        'before:absolute before:-inset-x-1 before:inset-y-0 before:content-[""]',
        dragging ? 'bg-accent' : 'hover:bg-accent-line',
      ].join(' ')}
      {...props}
    />
  )
}
