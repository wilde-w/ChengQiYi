import { useEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

/** 按下标题栏那一刻记住的东西。`left/top` 是**居中时**的盒子（已减掉偏移）。 */
type Grip = {
  x0: number
  y0: number
  dx: number
  dy: number
  left: number
  top: number
  w: number
}

/** 面板至少留这么多在视口里，否则它就抓不回来了。 */
const KEEP_X = 120
const KEEP_Y = 56

/** 夹在区间里。**区间可能是倒的**（视口比面板还窄时），那时取中点。 */
function within(v: number, lo: number, hi: number): number {
  return lo > hi ? (lo + hi) / 2 : Math.min(Math.max(v, lo), hi)
}

function bounded(grip: Grip, dx: number, dy: number): { dx: number; dy: number } {
  const vw = window.innerWidth
  const vh = window.innerHeight
  // 水平：整块不许滑出左边，也不许滑出右边。
  const x0 = KEEP_X - grip.w - grip.left
  const x1 = vw - KEEP_X - grip.left
  // 垂直：标题栏（也就是抓取区）始终留在视口里——往上顶出去就再也够不着了。
  const y0 = -grip.top
  const y1 = vh - KEEP_Y - grip.top
  return { dx: within(dx, x0, x1), dy: within(dy, y0, y1) }
}

/**
 * 模态框。项目里的第一个。
 *
 * **必须用 `createPortal` 挂到 body。** 三栏布局里的栏头是 `sticky z-[5]`，
 * 它自己创建了一个层叠上下文——模态框若渲染在树的深处，会被后出现的栏头
 * 压在下面，表现为「遮罩盖住了对话框但没盖住栏头」这种看起来像 bug 的渲染。
 * 挂到 body 就完全绕开了这个问题，不必再靠调 z-index 互相攀比。
 *
 * 关闭的两条路：Esc 与点击遮罩。**点击面板本身不关**——用户在里面填了
 * 书名、选好了文件，一次误触就全没了。
 *
 * `draggable` 让面板能被挪到别处（故事工坊那种占满一屏的才需要：挪开一点
 * 才看得见底下的工作台）。**只有标题栏是抓取区**——面板内部要选字、要滚动、
 * 要拖滑块，整块可拖等于把这些全废掉。
 */
export function Modal({
  open,
  onClose,
  title,
  subtitle,
  children,
  footer,
  width = 'max-w-[560px]',
  height = 'max-h-[calc(100vh-3rem)]',
  bodyClassName = 'overflow-y-auto px-5 py-4',
  draggable = false,
}: {
  open: boolean
  onClose: () => void
  title: string
  subtitle?: ReactNode
  children: ReactNode
  footer?: ReactNode
  width?: string
  /** 面板的高度。故事工坊要一个**固定**高度——三栏各自滚，而不是整块滚。 */
  height?: string
  /** 内容区的类。默认带内边距且自己滚；三栏面板要把这两样都换掉。 */
  bodyClassName?: string
  /** 按住标题栏可拖动。位置**不持久化**：关掉再打开回到居中。 */
  draggable?: boolean
}) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    // 打开期间锁住背景滚动：三栏工作台是可以滚的，不锁的话滚轮会
    // 穿透遮罩滚到底下的列表上，看起来像是模态框自己在乱动。
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prev
    }
  }, [open, onClose])

  // ---- 拖动 --------------------------------------------------------------
  // 位置存成**相对居中位置的偏移**，用 `transform: translate` 而不是
  // `left/top`：transform 不参与布局，面板挪到哪儿都不会改变遮罩的滚动区域。
  // 用 left/top 的话，往右下挪一点，这个 `overflow-y-auto` 的遮罩就长出可滚
  // 内容（`overflow-x` 会跟着变成 auto），面板一动、整层跟着滚，手感是坏的。
  const [drag, setDrag] = useState<{ dx: number; dy: number } | null>(null)
  const panelRef = useRef<HTMLDivElement | null>(null)
  const gripRef = useRef<Grip | null>(null)

  useEffect(() => {
    // 关掉就归位。**不持久化**：位置被记住之后，换个窗口尺寸再打开，
    // 面板可能只剩一个角露在屏幕外——而用户此刻没有任何办法把它拉回来。
    if (!open) {
      setDrag(null)
      gripRef.current = null
    }
  }, [open])

  const onGripDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return
    // 标题栏右边就是关闭按钮。它得能点，不能被拖动吃掉。
    if ((e.target as HTMLElement).closest('button, a, input, select, textarea')) return
    const panel = panelRef.current
    if (!panel) return
    const rect = panel.getBoundingClientRect()
    const dx = drag?.dx ?? 0
    const dy = drag?.dy ?? 0
    // **减掉当前偏移才是「居中时的位置」。** `getBoundingClientRect` 已经把
    // translate 算进去了；直接拿它当基准，每拖一次基准就漂一点，夹取范围
    // 于是越来越窄，表现为「拖到第三次就拖不动了」。
    gripRef.current = {
      x0: e.clientX,
      y0: e.clientY,
      dx,
      dy,
      left: rect.left - dx,
      top: rect.top - dy,
      w: rect.width,
    }
    // 不阻止默认的话，拖动会顺手把标题里的字选中，整块看着像文本不像窗口。
    e.preventDefault()
    e.currentTarget.setPointerCapture(e.pointerId)
    setDrag({ dx, dy })
  }

  const onGripMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const grip = gripRef.current
    if (!grip) return
    setDrag(bounded(grip, grip.dx + e.clientX - grip.x0, grip.dy + e.clientY - grip.y0))
  }

  const onGripUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!gripRef.current) return
    gripRef.current = null
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId)
    }
  }

  if (!open) return null

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-ink/25 p-4 sm:p-8"
      onMouseDown={onClose}
      role="presentation"
    >
      {/* 面板限高 + 分三段：头尾固定，中间自己滚。
          不限高的后果很具体——内容一多，主操作按钮就被推到视口之外，
          而用户看到的是一屏填得好好的表单，完全不知道下面还有按钮。 */}
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onMouseDown={(e) => e.stopPropagation()}
        // **拖过之后就把 `animate-rise` 摘掉。** 它是 `animation: rise … both`，
        // 动画的填充值在层叠里高于普通声明——包括元素上的
        // `style="transform: …"`，留着这个类，内联的 translate 永远不生效。
        className={`bg-surface border-line-strong my-auto flex ${height} w-full ${width} flex-col rounded-lg border shadow-xl ${
          drag ? '' : 'animate-rise'
        }`}
        style={drag ? { transform: `translate(${drag.dx}px, ${drag.dy}px)` } : undefined}
      >
        <div
          className={`border-line flex shrink-0 items-start justify-between gap-4 border-b px-5 py-3.5 ${
            draggable ? 'cursor-grab touch-none active:cursor-grabbing' : ''
          }`}
          onPointerDown={draggable ? onGripDown : undefined}
          onPointerMove={draggable ? onGripMove : undefined}
          onPointerUp={draggable ? onGripUp : undefined}
          onPointerCancel={draggable ? onGripUp : undefined}
        >
          <div className="min-w-0">
            <h2 className="text-ink text-[15px] font-medium">{title}</h2>
            {subtitle ? <div className="text-ink-faint mt-0.5 text-[11.5px]">{subtitle}</div> : null}
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="关闭"
            className="text-ink-ghost hover:text-ink-soft -mr-1 -mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-[18px] leading-none transition-colors"
          >
            ×
          </button>
        </div>

        <div className={`min-h-0 flex-1 ${bodyClassName}`}>{children}</div>

        {footer ? (
          <div className="border-line bg-paper-sunk flex shrink-0 items-center justify-end gap-2 rounded-b-lg border-t px-5 py-3">
            {footer}
          </div>
        ) : null}
      </div>
    </div>,
    document.body,
  )
}
