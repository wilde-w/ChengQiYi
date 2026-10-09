/**
 * 切分规则的**前端镜像**，只用于弹窗里的实时条数预览。
 *
 * 这里的返回值不参与任何请求——真正的切分在后端 `services/text_source.py`，
 * 它才是权威。之所以要抄一份，是因为用户粘进去 300 行、再点开始，
 * 不该等到分析跑起来才知道「原来要按行切」；他需要在下手之前就看到
 * 「识别到 300 条」。
 *
 * 规则必须与后端逐条对齐（行模式丢空行并 strip、段落模式以空行分段、
 * 段内换行保留），否则预览的条数和左栏最终显示的条数会对不上，
 * 而用户没有任何办法解释这个差异。改这里时同步改后端。
 */

export type TextMode = 'line' | 'paragraph'

/** 后端 `text_source.TEXT_ITEM_MAX_CHARS` 的镜像——仅用于提示「会被截断」。 */
export const TEXT_ITEM_MAX_CHARS = 2000

/** 与后端 `schemas/run.py` 的 `comment_limit` 上限一致。 */
export const COMMENT_LIMIT_MAX = 2000

/** 与 `constants.MIN_COMMENTS_FOR_CLUSTERING` 一致：低于它不做聚类。 */
export const MIN_COMMENTS_FOR_CLUSTERING = 12

/** 段落分隔：一个空行（允许带空格/制表符），连续多个等价于一个。 */
const BLANK_LINE = /\n[ \t]*\n+/

export interface SplitPreview {
  /** 切出的总条数（上限之前） */
  total: number
  /** 因超出条数上限而不会被分析的条数 */
  dropped: number
  /** 因单条过长会被截断的条数 */
  truncated: number
}

export function splitParts(text: string, mode: TextMode): string[] {
  const normalized = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n')
  const raw = mode === 'paragraph' ? normalized.split(BLANK_LINE) : normalized.split('\n')
  return raw.map((part) => part.trim()).filter((part) => part.length > 0)
}

/**
 * `limit` 传当前将要用到的条数上限（见 `commentLimitFor`）。
 * 只返回计数而不是条目：预览不需要正文，几百条文本在打字时反复切片是浪费。
 */
export function previewSplit(text: string, mode: TextMode, limit: number): SplitPreview {
  const parts = splitParts(text, mode)
  const kept = Math.max(0, Math.min(parts.length, Math.max(0, limit)))
  return {
    total: parts.length,
    dropped: parts.length - kept,
    truncated: parts.slice(0, kept).filter((p) => p.length > TEXT_ITEM_MAX_CHARS).length,
  }
}

/**
 * 这次运行的条数上限。
 *
 * 不是固定 100：用户粘了 500 行却只分析前 100 行是**数据丢失**，
 * 而他并没有要求抽样。上限只用来兜住「粘了一整本书」这种量级，
 * 再往上（2000）就由后端 schema 拒绝。
 */
export function commentLimitFor(total: number): number {
  return Math.min(COMMENT_LIMIT_MAX, Math.max(100, total))
}

/** 后端的 `TEXT_INPUT_MAX`。本地先拦一道，不让他把 6 万字传完才被 422 拒掉。 */
export const TEXT_INPUT_MAX = 50_000
