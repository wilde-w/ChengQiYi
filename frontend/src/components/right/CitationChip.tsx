/**
 * 正文里的引用 chip。
 *
 * 它是这个产品的立身之本的可见形式：正文里每一处「这里有依据」都能点开
 * 落到一张证据卡上。所以它必须**看起来就能点**——一个不可点的上标数字
 * 会让整段正文的引用变成装饰。
 *
 * 配色刻意保持中性：`theme.ts` 里的三组标签色在别处有固定含义
 * （情绪 / 主题 / 需求），拿它们代表知识库分库会让读者以为 chip 上的
 * 颜色说的是情绪。分库信息放在 hover 的出处里，那才是它该待的地方。
 *
 * 找不到对应证据时不隐藏、不静默降级成一个数字：那种情况下「正文引了
 * 一个不存在的东西」正是用户最该看见的事。后端的 `sanitize` 保证这种情况
 * 不会落库，但前端不该依赖「上游保证」来假装它不存在——保证失效的那天，
 * 这里就是唯一会说话的地方。
 */

import type { CitationOut } from '../../api/types'
import { scrollToEvidence } from '../middle/EvidenceBoard'

export function CitationChip({ citation, index }: { citation: CitationOut; index: number }) {
  return (
    <button
      type="button"
      title={citation.text || citation.evidence_id}
      onClick={() => scrollToEvidence(citation.evidence_id)}
      className="bg-paper-sunk text-ink-faint hover:bg-accent-soft hover:text-accent mx-[1px] inline-flex h-[15px] min-w-[15px] cursor-pointer items-center justify-center rounded px-[3px] align-super font-mono text-[9.5px] leading-none tabular-nums transition-colors"
    >
      {index}
    </button>
  )
}

/** 记号在正文里出现，但没有任何一条引用认领它。 */
export function DanglingMarker({ marker }: { marker: string }) {
  return (
    <span
      title="正文里的这处引用没有落到证据上"
      className="border-warn/40 text-warn mx-[1px] inline-flex h-[15px] items-center rounded border px-[3px] align-super font-mono text-[9.5px] leading-none"
    >
      {marker}
    </span>
  )
}
