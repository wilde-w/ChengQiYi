/**
 * 图表与标签的配色。
 *
 * 这些值必须同时被 ECharts（canvas 绘制，只吃字面量）和 DOM 使用，
 * 所以放在 TS 里而不是 CSS 变量里——CSS 变量在 canvas 里读不到。
 *
 * **簇色不在这里。** 它跟着 `Cluster.color` 从后端来（见 backend 的
 * `constants.CLUSTER_PALETTE`）：簇色是流水线的产物，落在库里，前端只负责
 * 画。两边各存一份的话，症状是同一个簇在卡片和环形图里显示成两个颜色，
 * 而这种不一致几乎不会被归因到配色表上。
 */

/** 噪声簇的固定色。刻意用灰——一眼看出「这不是一个真实主题」。 */
export const NOISE_COLOR = '#9a958c'

/** 推理链四层：现象 → 机制 → 典故 → 洞察。 */
export const REASONING_LAYERS = [
  { key: 'phenomenon', label: '现象', color: '#8a8580' },
  { key: 'mechanism', label: '机制', color: '#5b6abf' },
  { key: 'allusion', label: '典故', color: '#c4703a' },
  { key: 'insight', label: '洞察', color: '#4a8c7e' },
] as const

export type ReasoningLayerKey = (typeof REASONING_LAYERS)[number]['key']

/** 情绪/主题/需求标签的配色——同一维度内保持一致，跨维度可区分。 */
export const TAG_COLORS = {
  emotion: { bg: '#fdf2f4', fg: '#9f3d5c', border: '#f2d5dd' },
  topic: { bg: '#eef2fb', fg: '#3f5599', border: '#d3ddf2' },
  need: { bg: '#eff7f3', fg: '#2f6b55', border: '#d3e9dd' },
} as const

export type TagKind = keyof typeof TAG_COLORS
