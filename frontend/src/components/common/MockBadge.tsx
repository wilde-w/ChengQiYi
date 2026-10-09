/**
 * 演示模式徽章。
 *
 * 只要任一子系统跑在 mock 上就必须可见，且要能说清**是哪一部分**。
 * 一个「看起来很像真的」的假分析比一个明显的错误危险得多。
 */
export function MockBadge({
  providers,
  compact = false,
}: {
  providers?: { llm?: string; embedding?: string; douyin?: string; is_demo?: boolean } | null
  compact?: boolean
}) {
  if (!providers?.is_demo) return null

  const mockParts = (['llm', 'embedding', 'douyin'] as const)
    .filter((k) => providers[k] === 'mock')
    .map((k) => ({ llm: 'LLM', embedding: '向量', douyin: '抖音数据' })[k])

  return (
    <span
      title={`以下部分使用 mock 数据：${mockParts.join('、')}。配置对应 API key 后自动切换为真实数据。`}
      className="border-warn/30 bg-warn-soft text-warn inline-flex items-center gap-1 rounded-full border px-2 py-[3px] text-[11px] leading-none"
    >
      <span aria-hidden>◐</span>
      演示模式
      {!compact && <span className="opacity-70">· {mockParts.join('/')}</span>}
    </span>
  )
}
