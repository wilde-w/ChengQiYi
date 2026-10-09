/**
 * applyEvent 的黄金测试。
 *
 * 这里断言的是**不变量**，不是实现细节：进度不回退、seq 去重、
 * 增量合并的幂等性。流式 UI 的 bug 几乎都出在这三条上，
 * 而它们又极难在浏览器里手动复现（要精确卡在某一帧断网）。
 */

import { describe, expect, it } from 'vitest'

import type { RunEvent } from '../api/types'
import { applyEvent, initialState, mergeSnapshot, resetForRun } from '../store/applyEvent'

function ev(partial: Partial<RunEvent> & { seq: number; type: RunEvent['type'] }): RunEvent {
  return { run_id: 'r1', ...partial } as RunEvent
}

describe('seq 去重（不变量 2）', () => {
  it('丢弃 seq 小于等于游标的事件', () => {
    let s = initialState()
    s = applyEvent(s, ev({ seq: 5, type: 'progress', progress: 30 }))
    expect(s.lastSeq).toBe(5)

    // 重连重放：同样的 seq 再来一次不能改变任何东西
    const again = applyEvent(s, ev({ seq: 5, type: 'progress', progress: 30 }))
    expect(again).toBe(s) // 引用相等——没触发任何重渲染

    const older = applyEvent(s, ev({ seq: 3, type: 'progress', progress: 10 }))
    expect(older).toBe(s)
  })

  it('乱序到达的旧事件不会推进游标', () => {
    const s = applyEvent(initialState(), ev({ seq: 9, type: 'progress', progress: 50 }))
    const out = applyEvent(s, ev({ seq: 7, type: 'run_completed', progress: 100 }))
    expect(out.lastSeq).toBe(9)
    expect(out.status).not.toBe('succeeded')
  })
})

describe('进度单调（不变量 1）', () => {
  it('回退的 progress 被钳住', () => {
    let s = applyEvent(initialState(), ev({ seq: 1, type: 'progress', progress: 42 }))
    s = applyEvent(s, ev({ seq: 2, type: 'progress', progress: 17 }))
    expect(s.progress).toBe(42)
  })

  it('超过 100 的值被钳到 100', () => {
    const s = applyEvent(initialState(), ev({ seq: 1, type: 'progress', progress: 130 }))
    expect(s.progress).toBe(100)
  })

  it('NaN / 非数值不改变进度', () => {
    let s = applyEvent(initialState(), ev({ seq: 1, type: 'progress', progress: 20 }))
    s = applyEvent(s, ev({ seq: 2, type: 'progress', progress: Number.NaN }))
    s = applyEvent(s, ev({ seq: 3, type: 'progress', progress: null }))
    expect(s.progress).toBe(20)
  })
})

describe('partial: comment_page', () => {
  const page = (ids: string[]) =>
    ev({
      seq: 0,
      type: 'partial',
      node: 'n2_comments',
      data: {
        kind: 'comment_page',
        items: ids.map((id) => ({ comment_id: id, text: id, like_count: 1, reply_count: 0 })),
        received: ids.length,
      },
    })

  it('逐页追加', () => {
    let s = initialState()
    s = applyEvent(s, { ...page(['a', 'b']), seq: 1 })
    s = applyEvent(s, { ...page(['c']), seq: 2 })
    expect(s.comments.map((c) => c.comment_id)).toEqual(['a', 'b', 'c'])
  })

  it('重叠分页不会产生重复项', () => {
    let s = initialState()
    s = applyEvent(s, { ...page(['a', 'b']), seq: 1 })
    // 服务端游标重叠，第二页把 b 又发了一次
    s = applyEvent(s, { ...page(['b', 'c']), seq: 2 })
    expect(s.comments.map((c) => c.comment_id)).toEqual(['a', 'b', 'c'])
  })

  it('data.kind 不会被混进评论对象', () => {
    const s = applyEvent(initialState(), { ...page(['a']), seq: 1 })
    expect(s.comments[0]).toBeDefined()
    expect('kind' in s.comments[0]!).toBe(false)
  })
})

describe('partial: clusters', () => {
  const clusters = (keys: string[]) =>
    ev({
      seq: 0,
      type: 'partial',
      node: 'n3_cluster',
      data: {
        kind: 'clusters',
        clusters: keys.map((k, i) => ({
          cluster_key: k,
          label: k,
          size: 5,
          raw_size: 5,
          is_noise: false,
          emotion_tags: [],
          topic_tags: [],
          need_tags: [],
          keywords: [],
          color: '#000',
          order_index: i,
        })),
        cluster_meta: { method: 'kmeans', k: keys.length, deduped: 3 },
      },
    })

  it('整体替换而非追加：n3 每标完一簇就重发完整列表', () => {
    let s = applyEvent(initialState(), { ...clusters(['c0']), seq: 1 })
    s = applyEvent(s, { ...clusters(['c0', 'c1']), seq: 2 })
    expect(s.clusters.map((c) => c.cluster_key)).toEqual(['c0', 'c1'])
  })

  it('cluster_meta 与簇同帧到达时一并落进 state', () => {
    // 分开收的话，中栏会在簇卡出现之后才敢说「这是 KMeans 兜底的」——
    // 而「兜底」正是用户判断这批卡片可信度时最需要的一句。
    const s = applyEvent(initialState(), { ...clusters(['c0']), seq: 1 })
    expect(s.clusterMeta).toEqual({ method: 'kmeans', k: 1, deduped: 3 })
  })

  it('data.kind 不会被混进簇对象', () => {
    const s = applyEvent(initialState(), { ...clusters(['c0']), seq: 1 })
    expect('kind' in s.clusters[0]!).toBe(false)
  })
})

describe('partial: profile', () => {
  it('整包替换：半个旧侧写配半个新侧写比没有更糟', () => {
    const first = ev({
      seq: 1,
      type: 'partial',
      node: 'n4_psych',
      data: {
        kind: 'profile',
        emotions: [{ label: '哀伤', value: 40 }],
        topics: [],
        needs: [],
        global_tags: { emotion: ['哀伤'], topic: [], need: [] },
        cluster_tags: {},
      },
    })
    const second = ev({
      seq: 2,
      type: 'partial',
      node: 'n4_psych',
      data: {
        kind: 'profile',
        emotions: [{ label: '倦怠', value: 9 }],
        topics: [],
        needs: [],
        global_tags: { emotion: ['倦怠'], topic: [], need: [] },
        cluster_tags: {},
        model: 'fallback',
      },
    })

    let s = applyEvent(initialState(), first)
    s = applyEvent(s, second)

    expect(s.profile?.emotions.map((e) => e.label)).toEqual(['倦怠'])
    expect(s.profile?.model).toBe('fallback')
  })

  it('global_tags 按维度分组，不是一个大平表', () => {
    // 声明成 list[str] 时右栏那段情绪列表恒为空——接口不报错，只是永远没内容。
    const s = applyEvent(
      initialState(),
      ev({
        seq: 1,
        type: 'partial',
        data: { kind: 'profile', global_tags: { emotion: ['哀伤'], topic: ['亲子关系'], need: ['被看见'] } },
      }),
    )
    expect(s.profile?.global_tags.emotion).toEqual(['哀伤'])
    expect(s.profile?.global_tags.need).toEqual(['被看见'])
  })
})

describe('partial: evidence / reasoning 的两种契约', () => {
  it('evidence 整批到达时按数组展开', () => {
    const s = applyEvent(
      initialState(),
      ev({
        seq: 1,
        type: 'partial',
        data: { kind: 'evidence', evidence: [{ chunk_id: 'a' }, { chunk_id: 'b' }] },
      }),
    )
    expect(s.evidence.map((e) => e.chunk_id)).toEqual(['a', 'b'])
  })

  it('evidence 逐条到达时追加——**判别键不能覆盖证据自己的 kind**', () => {
    // 证据项自带 kind（psychology/literature/poetry）。runner 拼 data 时
    // 若把 kind 写在 payload 之后，它会被 data_kind 盖掉，前端按 kind 分派
    // 时一条都认不出来：表现是证据卡一张都不出现，而事件流里条条都在。
    const s = applyEvent(
      initialState(),
      ev({
        seq: 1,
        type: 'partial',
        data: { kind: 'evidence', evidence: [{ chunk_id: 'a', kind: 'psychology' }] },
      }),
    )
    expect(s.evidence[0]!.kind).toBe('psychology')
  })

  it('reasoning 逐条与整批都接受', () => {
    let s = applyEvent(initialState(), ev({ seq: 1, type: 'partial', data: { kind: 'reasoning', step: 1 } }))
    expect(s.reasoning).toHaveLength(1)
    s = applyEvent(s, ev({ seq: 2, type: 'partial', data: { kind: 'reasoning', reasoning: [{ step: 2 }] } }))
    expect(s.reasoning).toHaveLength(2)
  })
})

describe('delta: 正文流式', () => {
  it('同一段落的增量按序拼接', () => {
    let s = initialState()
    s = applyEvent(s, ev({ seq: 1, type: 'delta', data: { section: 'insight', text: '人们' } }))
    s = applyEvent(s, ev({ seq: 2, type: 'delta', data: { section: 'insight', text: '并不' } }))
    s = applyEvent(s, ev({ seq: 3, type: 'delta', data: { section: 'insight', text: '孤独。' } }))
    expect(s.sections.insight?.content_md).toBe('人们并不孤独。')
  })

  it('不同段落各自独立累积', () => {
    let s = initialState()
    s = applyEvent(s, ev({ seq: 1, type: 'delta', data: { section: 'profile', text: '甲' } }))
    s = applyEvent(s, ev({ seq: 2, type: 'delta', data: { section: 'mechanism', text: '乙' } }))
    s = applyEvent(s, ev({ seq: 3, type: 'delta', data: { section: 'profile', text: '丙' } }))
    expect(s.sections.profile?.content_md).toBe('甲丙')
    expect(s.sections.mechanism?.content_md).toBe('乙')
  })

  it('元信息先到、正文后到：标题与引用不会被增量吃掉', () => {
    // n7 的发帧顺序：先一条 sections partial（正文留空，只带标题与引用），
    // 再一串 delta 填正文。顺序反过来的话，元信息里的 content_md=""
    // 会把已经打出来的正文整个覆盖掉——用户看到正文写完又消失。
    let s = initialState()
    s = applyEvent(
      s,
      ev({
        seq: 1,
        type: 'partial',
        data: {
          kind: 'sections',
          sections: [
            {
              key: 'mechanism',
              title: '科学机制',
              content_md: '',
              citations: [{ evidence_id: 'psy:1', marker: '[^1]', text: '《持续性联结》· Klass' }],
              version: 1,
              stale: false,
              based_on_revision: 1,
              edited_by_user: false,
              citation_coverage: 1,
            },
          ],
        },
      }),
    )
    expect(s.sections.mechanism?.title).toBe('科学机制')

    s = applyEvent(s, ev({ seq: 2, type: 'delta', data: { section: 'mechanism', text: '机制' } }))
    s = applyEvent(s, ev({ seq: 3, type: 'delta', data: { section: 'mechanism', text: '说明[^1]' } }))

    expect(s.sections.mechanism?.content_md).toBe('机制说明[^1]')
    // 增量落在元信息搭好的壳上，而不是另起一个标题为空的段落。
    expect(s.sections.mechanism?.title).toBe('科学机制')
    expect(s.sections.mechanism?.citations).toHaveLength(1)
  })

  it('缺 section 或空 text 的帧不产生段落', () => {
    const s0 = initialState()
    // 注意断言的是「没有副作用」而非「返回同一个对象」：
    // 这一帧消耗掉了一个 seq，游标必须前进，否则重连后会重复处理它。
    const a = applyEvent(s0, ev({ seq: 1, type: 'delta', data: { text: '无段落' } }))
    expect(a.sections).toEqual({})
    expect(a.lastSeq).toBe(1)

    const b = applyEvent(a, ev({ seq: 2, type: 'delta', data: { section: 'insight' } }))
    expect(b.sections.insight).toBeUndefined()
  })
})

describe('节点状态', () => {
  it('node_started → running，node_completed → done', () => {
    let s = resetForRun('r1', ['n1_video', 'n2_comments'])
    s = applyEvent(s, ev({ seq: 1, type: 'node_started', node: 'n1_video' }))
    expect(s.nodes.n1_video).toBe('running')
    expect(s.currentNode).toBe('n1_video')

    s = applyEvent(s, ev({ seq: 2, type: 'node_completed', node: 'n1_video' }))
    expect(s.nodes.n1_video).toBe('done')
    // 完成后必须离开该节点，否则中栏会一直停在最后一个节点上
    expect(s.currentNode).toBeNull()
  })
})

describe('终态', () => {
  it('run_completed 把进度钉到 100', () => {
    let s = applyEvent(initialState(), ev({ seq: 1, type: 'progress', progress: 99 }))
    s = applyEvent(s, ev({ seq: 2, type: 'run_completed' }))
    expect(s.status).toBe('succeeded')
    expect(s.progress).toBe(100)
  })

  it('error 事件把状态置为 failed 并保留信息', () => {
    const s = applyEvent(
      initialState(),
      ev({ seq: 1, type: 'error', message: '评论接口不可用', data: { code: 'run_failed' } }),
    )
    expect(s.status).toBe('failed')
    expect(s.error).toEqual({ code: 'run_failed', message: '评论接口不可用' })
  })

  it('warning 不改变状态，只累积记录', () => {
    const s = applyEvent(
      initialState(),
      ev({ seq: 1, type: 'warning', message: '降级', data: { code: 'bus_degraded' } }),
    )
    expect(s.status).toBe('idle')
    expect(s.warnings).toEqual([{ code: 'bus_degraded', message: '降级' }])
  })
})

describe('快照对账', () => {
  it('覆盖产物但不动 progress 与 lastSeq', () => {
    let s = applyEvent(initialState(), ev({ seq: 7, type: 'progress', progress: 100 }))
    s = mergeSnapshot(s, {
      status: 'succeeded',
      revision: 2,
      comments: [
        { comment_id: 'x', text: '权威版本', like_count: 1, reply_count: 0 },
      ] as never,
      comment_stats: { total: 1, kept: 1, ads: 0, spam: 0, duplicates: 0, empty: 0, reasons: {} },
    })
    expect(s.lastSeq).toBe(7)
    expect(s.progress).toBe(100)
    expect(s.comments).toHaveLength(1)
    expect(s.revision).toBe(2)
  })

  it('快照带回中栏的产物：簇、元信息、侧写、推理链', () => {
    // 刷新后流式产物全靠这一步恢复。少合并一项，症状是「刷新一下簇卡还在、
    // 侧写没了」——而用户会以为分析结果本来就不完整。
    const s = mergeSnapshot(resetForRun('r1', ['n3_cluster']), {
      status: 'succeeded',
      revision: 3,
      clusters: [{ cluster_key: 'c0', label: '想念', size: 5, raw_size: 6, is_noise: false }] as never,
      cluster_meta: { method: 'hdbscan', k: 1 },
      profile: { emotions: [], topics: [], needs: [], global_tags: {}, cluster_tags: {} } as never,
      reasoning: [{ step: 1 }] as never,
    })

    expect(s.clusters).toHaveLength(1)
    expect(s.clusterMeta).toEqual({ method: 'hdbscan', k: 1 })
    expect(s.profile).not.toBeNull()
    expect(s.reasoning).toHaveLength(1)
    expect(s.revision).toBe(3)
  })

  it('把仍是 running 的节点收尾，避免中栏永远停在「进行中」', () => {
    let s = resetForRun('r1', ['n1_video', 'n7_literary'])
    s = applyEvent(s, ev({ seq: 1, type: 'node_started', node: 'n7_literary' }))
    s = mergeSnapshot(s, { status: 'succeeded', revision: 1 })
    expect(s.nodes.n7_literary).toBe('done')
    expect(s.currentNode).toBeNull()
  })

  it('刷新后恢复一条已完成的运行：整张节点清单收敛为完成', () => {
    // 快照没有节点级信息，恢复时七个节点都还是 pending。
    // 若只把 running 翻成 done，中栏会显示七个灰着的节点配一条跑完的运行。
    const s = mergeSnapshot(resetForRun('r1', ['n1_video', 'n2_comments']), {
      status: 'succeeded',
      revision: 1,
      progress: 100,
    })
    expect(s.nodes).toEqual({ n1_video: 'done', n2_comments: 'done' })
  })

  it('失败的运行把半路的节点标成 failed 而不是 done', () => {
    let s = resetForRun('r1', ['n1_video', 'n2_comments'])
    s = applyEvent(s, ev({ seq: 1, type: 'node_started', node: 'n2_comments' }))
    s = mergeSnapshot(s, { status: 'failed', revision: 1 })
    expect(s.nodes.n2_comments).toBe('failed')
    expect(s.nodes.n1_video).toBe('pending')
  })

  it('终态是单向门：迟到的 running 快照不能把 succeeded 打回去', () => {
    // 后端先发 run_completed、再写库，中间隔着一次数据库往返。
    // 前端正是在收到事件后才拉快照，于是快照会报回尚未更新的 running。
    let s = applyEvent(resetForRun('r1', ['n1_video']), ev({ seq: 1, type: 'run_completed' }))
    expect(s.status).toBe('succeeded')

    s = mergeSnapshot(s, { status: 'running', revision: 1, progress: 100 })
    expect(s.status).toBe('succeeded')
    expect(s.nodes.n1_video).toBe('done')
  })

  it('非终态时快照仍然可以改写状态', () => {
    const s = mergeSnapshot(initialState(), { status: 'failed', revision: 1 })
    expect(s.status).toBe('failed')
  })

  it('恢复进度时沿用单调钳制，不会把已是 100 的进度拉回来', () => {
    let s = applyEvent(initialState(), ev({ seq: 5, type: 'progress', progress: 100 }))
    s = mergeSnapshot(s, { status: 'succeeded', revision: 1, progress: 62 })
    expect(s.progress).toBe(100)

    // 反向：空状态下从快照拿到 62，不必等下一帧事件
    const fresh = mergeSnapshot(initialState(), { status: 'running', revision: 1, progress: 62 })
    expect(fresh.progress).toBe(62)
  })
})

describe('不修改入参（不变量 3）', () => {
  it('reducer 返回新对象且原状态未被改动', () => {
    const before = initialState()
    const snapshot = JSON.stringify(before)
    const after = applyEvent(
      before,
      ev({ seq: 1, type: 'partial', data: { kind: 'comment_page', items: [], received: 0 } }),
    )
    expect(JSON.stringify(before)).toBe(snapshot)
    expect(after).not.toBe(before)
  })
})

describe('数据源种类（sourceKind）', () => {
  it('默认是抖音——老快照没有这个字段时不能被冲掉', () => {
    expect(initialState().sourceKind).toBe('douyin')

    // 文本运行跑到一半来了个不带字段的快照：保持原值，而不是退回 douyin
    const text = resetForRun('r1', [], 'text')
    expect(mergeSnapshot(text, { status: 'running', revision: 1 }).sourceKind).toBe('text')
    expect(mergeSnapshot(text, { status: 'running', revision: 1, source_kind: null }).sourceKind).toBe(
      'text',
    )
  })

  it('快照是权威版本：刷新恢复时以它为准', () => {
    const s = mergeSnapshot(initialState(), {
      status: 'succeeded',
      revision: 1,
      source_kind: 'text',
    })
    expect(s.sourceKind).toBe('text')
  })

  it('resetForRun 不带参数时回到抖音', () => {
    expect(resetForRun('r1', []).sourceKind).toBe('douyin')
  })
})
