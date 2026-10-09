/**
 * 「这个标签页正在看哪条运行」的落点。
 *
 * 只存 `runId` 和用户填的原始输入，**不存事件、不存产物、不存进度**——
 * 刷新后的恢复路线是把 runId 交还给事件流，由服务端重放历史把状态重建出来。
 * 存一份产物快照就等于给同一份状态造了第二个真源，两边一旦对不上，
 * 就会冒出「界面显示 40 条评论、库里 38 条」这类谁也解释不清的偏差。
 *
 * 用 sessionStorage 而非 localStorage：这是「本标签页在看什么」，
 * 不是「本用户上次看了什么」。同时开两个标签页对比两条运行是正常用法，
 * 用 localStorage 会互相覆盖。
 *
 * 所有访问都包了 try/catch——隐私模式、被策略禁用、配额满都会让
 * setItem 抛异常，而存不下一个 runId 远不该让整个工作台崩掉。
 */

const KEY = 'guanxin.active_run'

export interface ActiveRun {
  runId: string
  /** 恢复时填回输入框，让用户一眼看出刷新后在看的是哪条链接 */
  input: string
  savedAt: number
}

export function saveActiveRun(runId: string, input: string): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify({ runId, input, savedAt: Date.now() }))
  } catch {
    // 存不下只影响刷新恢复，不影响当前这次运行
  }
}

export function loadActiveRun(): ActiveRun | null {
  try {
    const raw = sessionStorage.getItem(KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<ActiveRun>
    if (typeof parsed?.runId !== 'string' || !parsed.runId) return null
    return {
      runId: parsed.runId,
      input: typeof parsed.input === 'string' ? parsed.input : '',
      savedAt: typeof parsed.savedAt === 'number' ? parsed.savedAt : 0,
    }
  } catch {
    // 旧版本写下的格式对不上时直接当没有，而不是让 JSON 解析异常冒到渲染层
    return null
  }
}

export function clearActiveRun(): void {
  try {
    sessionStorage.removeItem(KEY)
  } catch {
    // 同上
  }
}
