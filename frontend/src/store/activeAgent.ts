/**
 * 「这个标签页正开着哪条写故事会话」的落点。`activeRun.ts` 的同款。
 *
 * 只存 `sessionId`，**不存事件、不存正文、不存对话**——刷新后的恢复路线是把
 * id 交还给事件流，由服务端回放历史、再由快照对齐，界面自然重建出来。
 * 在这里存一份正文等于给同一份状态造第二个真源，两边一旦对不上，
 * 就会冒出「右栏有正文、中栏没有那句要求」这类谁也解释不清的偏差。
 *
 * 与流水线**用两个 key**：一条分析运行和一场对话是可以同时存在的，
 * 共用一个槽位意味着打开面板会把当前运行从「刷新后恢复」的名单里挤掉。
 *
 * 所有访问都包了 try/catch——隐私模式、被策略禁用、配额满都会让 setItem
 * 抛异常，而存不下一个 id 远不该让整个面板崩掉。
 */

const KEY = 'guanxin.active_agent'

export interface ActiveAgent {
  sessionId: string
  savedAt: number
}

export function saveActiveAgent(sessionId: string): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify({ sessionId, savedAt: Date.now() }))
  } catch {
    // 存不下只影响刷新恢复，不影响当前这场对话
  }
}

export function loadActiveAgent(): ActiveAgent | null {
  try {
    const raw = sessionStorage.getItem(KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<ActiveAgent>
    if (typeof parsed?.sessionId !== 'string' || !parsed.sessionId) return null
    return {
      sessionId: parsed.sessionId,
      savedAt: typeof parsed.savedAt === 'number' ? parsed.savedAt : 0,
    }
  } catch {
    // 旧版本写下的格式对不上时当没有，而不是让 JSON 解析异常冒到渲染层
    return null
  }
}

export function clearActiveAgent(): void {
  try {
    sessionStorage.removeItem(KEY)
  } catch {
    // 同上
  }
}
