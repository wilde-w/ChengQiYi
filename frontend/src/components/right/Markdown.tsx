/**
 * 一个手写的行级 markdown 渲染器。**刻意不装库。**
 *
 * 三个理由，任何一个单独都不够，合起来才成立：
 *
 * 1. 无论如何都要做「脚注记号 → 引用 chip」的替换，而这一步必须在
 *    **文本节点**上做。装了 `react-markdown` 也得上 `components={{...}}`
 *    去改它的 AST，等于同时维护两套渲染规则。
 * 2. `react-markdown` 默认走 `dangerouslySetInnerHTML` 那条路，为了安全
 *    还得再引 `rehype-sanitize`——依赖从零变成三个，而每个都是供应链上
 *    的一段。
 * 3. 内容是**我们自己的提示词规定的子集**（标题 / 段落 / 粗体 / 引用块 /
 *    列表 / 脚注记号）。子集外的构造按纯文本原样显示：模型写 `<script>`
 *    就真的显示成这八个字符。宁可朴素，不可把模型的输出当 HTML 用。
 *
 * 这一条是安全边界，不是审美偏好——所以整个文件里没有一处
 * `dangerouslySetInnerHTML`，输出全是 React 文本节点，转义由 React 保证。
 *
 * 渲染器是纯函数，能被 vitest 直接测（见 `src/test/markdown.test.ts`）。
 */

import type { ReactNode } from 'react'

import type { CitationOut } from '../../api/types'
import { CitationChip, DanglingMarker } from './CitationChip'

type Block =
  | { kind: 'heading'; level: number; text: string }
  | { kind: 'quote'; lines: string[] }
  | { kind: 'list'; items: string[] }
  | { kind: 'para'; lines: string[] }

/** 正文里的两种行内构造：粗体与脚注记号。其余一律是纯文本。 */
const INLINE_RE = /(\*\*[^*\n]+\*\*|\[\^\d+\])/g

/** 引用记号的形态。与后端 `validators.MARKER_TEMPLATE` 必须一致。 */
const MARKER_RE = /^\[\^(\d+)\]$/

export function Markdown({
  text,
  citations,
}: {
  text: string
  /** 按 `marker` 索引，正文里的记号靠它找回 evidence_id。 */
  citations: CitationOut[]
}) {
  const byMarker = new Map(citations.map((c) => [c.marker, c]))
  return (
    <div className="space-y-2.5">
      {parseBlocks(text).map((block, i) => (
        <BlockView key={i} block={block} byMarker={byMarker} />
      ))}
    </div>
  )
}

function BlockView({
  block,
  byMarker,
}: {
  block: Block
  byMarker: Map<string, CitationOut>
}) {
  switch (block.kind) {
    case 'heading':
      return (
        <h4
          className={
            block.level <= 2
              ? 'text-ink pt-1 text-[13px] font-medium'
              : 'text-ink-soft pt-0.5 text-[12.5px] font-medium'
          }
        >
          {inline(block.text, byMarker)}
        </h4>
      )
    case 'quote':
      return (
        <blockquote className="border-line-strong text-ink-soft border-l-2 pl-3 text-[12.5px] leading-relaxed whitespace-pre-line">
          {block.lines.map((line, i) => (
            <p key={i}>{inline(line, byMarker)}</p>
          ))}
        </blockquote>
      )
    case 'list':
      return (
        <ul className="space-y-1">
          {block.items.map((item, i) => (
            <li key={i} className="flex gap-2 text-[12.5px] leading-relaxed">
              <span className="text-ink-ghost shrink-0 pt-[2px] text-[9px]">●</span>
              <span className="min-w-0">{inline(item, byMarker)}</span>
            </li>
          ))}
        </ul>
      )
    case 'para':
      return (
        // `whitespace-pre-line`：模型偶尔把一个段落写成连续几行，
        // 那些换行是它有意打的，合成一行会把它的断句抹掉。
        <p className="text-ink-soft text-[12.5px] leading-relaxed whitespace-pre-line">
          {block.lines.map((line, i) => (
            <span key={i}>
              {i > 0 && '\n'}
              {inline(line, byMarker)}
            </span>
          ))}
        </p>
      )
  }
}

/** 行内切分。输出全是 React 文本节点——没有一处把字符串当 HTML 用。 */
function inline(text: string, byMarker: Map<string, CitationOut>): ReactNode[] {
  const out: ReactNode[] = []
  let last = 0
  for (const match of text.matchAll(INLINE_RE)) {
    const token = match[0]
    const at = match.index ?? 0
    if (at > last) out.push(text.slice(last, at))
    last = at + token.length

    if (token.startsWith('**')) {
      out.push(
        <strong key={at} className="text-ink font-medium">
          {token.slice(2, -2)}
        </strong>,
      )
      continue
    }
    const marker = MARKER_RE.exec(token)?.[0]
    const citation = marker ? byMarker.get(marker) : undefined
    if (marker && citation) {
      out.push(<CitationChip key={at} citation={citation} index={Number(marker.slice(2, -1))} />)
    } else {
      out.push(<DanglingMarker key={at} marker={token} />)
    }
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}

/**
 * 块级切分。规则只有四条，都是「提示词允许的写法」的镜像：
 * `#` 标题、`>` 引用、`-`/`*` 列表、其余是段落。
 *
 * 段落**合并连续的非空行**：模型把一个自然段写成几行是常事，逐行成段会
 * 让一段话被拆成好几个间距很大的小段。
 */
export function parseBlocks(text: string): Block[] {
  const blocks: Block[] = []
  let para: string[] = []
  let quote: string[] = []
  let list: string[] = []

  const flush = () => {
    if (para.length) blocks.push({ kind: 'para', lines: para })
    if (quote.length) blocks.push({ kind: 'quote', lines: quote })
    if (list.length) blocks.push({ kind: 'list', items: list })
    para = []
    quote = []
    list = []
  }

  for (const raw of (text ?? '').split('\n')) {
    const line = raw.replace(/\s+$/, '')
    const trimmed = line.trim()

    if (!trimmed) {
      flush()
      continue
    }

    const heading = /^(#{1,6})\s+(.*)$/.exec(trimmed)
    if (heading) {
      flush()
      blocks.push({ kind: 'heading', level: (heading[1] ?? '#').length, text: heading[2] ?? '' })
      continue
    }

    if (trimmed.startsWith('>')) {
      // 引用与段落一样合并：连续几行 `>` 是一个引用块，不是几个。
      if (para.length || list.length) flush()
      quote.push(trimmed.replace(/^>\s?/, ''))
      continue
    }

    const item = /^[-*+]\s+(.*)$/.exec(trimmed)
    if (item) {
      if (para.length || quote.length) flush()
      list.push(item[1] ?? '')
      continue
    }

    if (quote.length || list.length) flush()
    para.push(trimmed)
  }
  flush()
  return blocks
}
