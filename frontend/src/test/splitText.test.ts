/**
 * 切分预览。
 *
 * 这些用例是**后端 `tests/test_text_source.py` 的镜像**：同一批样例，
 * 同一批期望值。两边的规则一旦漂移，弹窗会说「识别到 30 条」而左栏出现
 * 27 条，而用户完全无从解释。所以样例要成对维护。
 */

import { describe, expect, it } from 'vitest'

import {
  COMMENT_LIMIT_MAX,
  TEXT_ITEM_MAX_CHARS,
  commentLimitFor,
  previewSplit,
  splitParts,
} from '../lib/splitText'

describe('splitParts', () => {
  it('按行切并丢掉空行与首尾空白', () => {
    expect(splitParts('  第一条  \n\n第二条\n   \n第三条\n', 'line')).toEqual([
      '第一条',
      '第二条',
      '第三条',
    ])
  })

  it('CRLF 与 CR 都当换行', () => {
    expect(splitParts('一\r\n二\r三', 'line')).toEqual(['一', '二', '三'])
  })

  it('段落模式以空行为界，段内换行原样保留', () => {
    expect(splitParts('第一段第一行\n第一段第二行\n\n\n第二段', 'paragraph')).toEqual([
      '第一段第一行\n第一段第二行',
      '第二段',
    ])
  })

  it('段落模式里没有空行的整篇是一条', () => {
    expect(splitParts('一\n二\n三', 'paragraph')).toHaveLength(1)
  })

  it('空或纯空白切不出东西', () => {
    for (const text of ['', '   ', '\n\n\n', '\r\n \t\r\n']) {
      expect(splitParts(text, 'line')).toEqual([])
      expect(splitParts(text, 'paragraph')).toEqual([])
    }
  })
})

describe('previewSplit', () => {
  it('统计条数并按上限标记丢弃', () => {
    const text = Array.from({ length: 10 }, (_, i) => `第${i + 1}条`).join('\n')
    expect(previewSplit(text, 'line', 4)).toEqual({ total: 10, dropped: 6, truncated: 0 })
  })

  it('上限足够时不丢东西', () => {
    expect(previewSplit('一\n二', 'line', 100)).toEqual({ total: 2, dropped: 0, truncated: 0 })
  })

  it('超长条目计入 truncated，恰好等于上限不计', () => {
    expect(previewSplit('啊'.repeat(TEXT_ITEM_MAX_CHARS + 1), 'line', 100).truncated).toBe(1)
    expect(previewSplit('啊'.repeat(TEXT_ITEM_MAX_CHARS), 'line', 100).truncated).toBe(0)
  })

  it('被上限丢掉的超长条目不算 truncated——它根本没被分析', () => {
    const long = '啊'.repeat(TEXT_ITEM_MAX_CHARS + 1)
    expect(previewSplit(`${long}\n第二条`, 'line', 1)).toEqual({
      total: 2,
      dropped: 1,
      truncated: 1,
    })
  })
})

describe('commentLimitFor', () => {
  it('少于下限时取下限', () => {
    expect(commentLimitFor(0)).toBe(100)
    expect(commentLimitFor(30)).toBe(100)
  })

  it('不抽样：条数超过下限就按实际条数取', () => {
    expect(commentLimitFor(500)).toBe(500)
  })

  it('封顶在 schema 允许的最大值', () => {
    expect(commentLimitFor(99_999)).toBe(COMMENT_LIMIT_MAX)
  })
})
