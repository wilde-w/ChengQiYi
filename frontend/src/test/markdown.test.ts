/**
 * 右栏渲染器的测试。
 *
 * 这里守的是两条**安全与一致性**的边界，不是排版细节：
 *
 * 1. **模型的输出永远不被当作 HTML。** 正文里出现 `<script>` 时，渲染
 *    结果里必须是转义过的文本。这一条一旦破了，整条流水线就是一个把
 *    模型输出直接注入页面的通道。
 * 2. **正文里的记号与 chip 一一对应。** 读者点一枚 chip 就该落到一张
 *    证据卡上；落不到的时候必须看得见（警告色），而不是静默消失。
 */

import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { CitationOut } from '../api/types'
import { Markdown, parseBlocks } from '../components/right/Markdown'

const CITATIONS: CitationOut[] = [
  {
    evidence_id: 'psy:klass:001',
    marker: '[^1]',
    text: '《持续性联结》· Klass',
    library: 'psychology',
  },
]

function html(text: string, citations: CitationOut[] = CITATIONS): string {
  return renderToStaticMarkup(Markdown({ text, citations }))
}

describe('parseBlocks（块级切分）', () => {
  it('连续的非空行合成一个段落', () => {
    const blocks = parseBlocks('第一行\n第二行\n\n隔一段')
    expect(blocks).toEqual([
      { kind: 'para', lines: ['第一行', '第二行'] },
      { kind: 'para', lines: ['隔一段'] },
    ])
  })

  it('标题、引用、列表各自成块', () => {
    const blocks = parseBlocks('## 小标题\n> 引一句\n> 再引一句\n- 甲\n- 乙')
    expect(blocks).toEqual([
      { kind: 'heading', level: 2, text: '小标题' },
      { kind: 'quote', lines: ['引一句', '再引一句'] },
      { kind: 'list', items: ['甲', '乙'] },
    ])
  })

  it('块之间按出现顺序排列', () => {
    const blocks = parseBlocks('段落一\n\n> 引用\n\n段落二')
    expect(blocks.map((b) => b.kind)).toEqual(['para', 'quote', 'para'])
  })

  it('空文本不产出任何块', () => {
    expect(parseBlocks('')).toEqual([])
    expect(parseBlocks('\n\n  \n')).toEqual([])
  })

  it('子集之外的写法当纯文本', () => {
    // 不认识的构造要原样落在段落里，而不是被当成语法吃掉。
    const blocks = parseBlocks('| 表头 | 表头 |\n```\ncode\n```')
    expect(blocks).toEqual([{ kind: 'para', lines: ['| 表头 | 表头 |', '```', 'code', '```'] }])
  })
})

describe('行内渲染', () => {
  it('粗体变成 strong', () => {
    expect(html('这是**重点**。')).toContain('<strong')
    expect(html('这是**重点**。')).toContain('重点')
  })

  it('记号变成可点的 chip，出处进 title', () => {
    const out = html('机制说明 [^1]')

    expect(out).toContain('<button')
    // title 是读者 hover 时唯一能看到的东西——chunk_id 不是给人读的，
    // 所以它必须显示成《标题》· 作者，而不是一串内部 id。
    expect(out).toContain('title="《持续性联结》· Klass"')
    expect(out).not.toContain('title="[^1]"')
  })

  it('认领不了的记号显示成警告而不是消失', () => {
    // 后端的 sanitize 保证这种情况不会落库。但「保证」失效的那天，
    // 这里必须是唯一会说话的地方——静默丢掉会让读者以为这句本来就没引用。
    const out = html('一句没有依据的话 [^9]')

    expect(out).toContain('[^9]')
    expect(out).toContain('text-warn')
    expect(out).not.toContain('<button')
  })

  it('未闭合的方括号原样保留', () => {
    expect(html('半截 [^ 记号')).toContain('[^ 记号')
  })
})

describe('安全边界：模型的输出不是 HTML', () => {
  it('script 标签被转义成文本', () => {
    const out = html('<script>alert(1)</script>')

    expect(out).toContain('&lt;script&gt;')
    expect(out).not.toContain('<script>')
  })

  it('img 的 onerror 也只是文本', () => {
    const out = html('<img src=x onerror="alert(1)">')

    expect(out).not.toContain('<img')
    expect(out).toContain('&lt;img')
  })

  it('行内构造里夹带的标签同样被转义', () => {
    const out = html('**<b>粗</b>**')

    expect(out).not.toContain('<b>')
    expect(out).toContain('&lt;b&gt;')
  })
})

describe('整段渲染', () => {
  it('四段常见的写法都能出来', () => {
    const out = html(
      ['## 机制', '', '这不只是一个情绪问题。[^1]', '', '> 引一句原文', '', '- 第一点', '- 第二点'].join(
        '\n',
      ),
    )

    expect(out).toContain('机制')
    expect(out).toContain('<blockquote')
    expect(out).toContain('<li')
    expect(out.match(/<li/g)).toHaveLength(2)
  })

  it('段内换行保留', () => {
    // 模型把一个自然段写成几行是有意断句，合成一行会把断句抹掉。
    // 断言取**文本内容**而不是 HTML：换行在两个 `<span>` 之间，
    // 直接找 `上句\n下句` 会被标签隔开，那是断言的写法问题，不是行为问题。
    const text = html('上句\n下句').replace(/<[^>]+>/g, '')
    expect(text).toContain('上句\n下句')
  })
})
