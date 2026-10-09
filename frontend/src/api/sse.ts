/**
 * SSE 订阅。
 *
 * **不用 `EventSource`。** 三个理由：
 *   1. 它带一个自己的重连逻辑，重连时不带 `since`——刷新页面后要么全量重放
 *      （几百帧、正文重新逐字打印），要么丢一段。我们要的是「从断点续上」。
 *   2. 它不支持自定义头，也没法看响应状态码（404 和「运行中」长得一样）。
 *   3. 我们要在 `run_completed` 后主动关闭，它的语义是「永远重连」。
 *
 * 所以用 `fetch` + 手写 SSE 解析。代价是要自己处理分块边界——
 * 一个 `\n\n` 帧可能被切成两个 chunk。见 `parse()` 里的缓冲处理。
 *
 * ——「流负责实时性，快照负责真实性」——
 * 这个模块只负责把事件按序交出去；收到终态事件后**不**自动重连，
 * 由调用方决定是拉快照还是继续。
 */

import type { RunEvent } from './types'

export interface StreamOptions {
  /** 事件流地址里的那个 id。流水线是 run_id，故事工坊是 session_id。 */
  runId: string
  /**
   * 事件流的路径模板，`{id}` 会被替换成 `runId`。
   *
   * 只有故事工坊需要传它（`/api/v1/agent/sessions/{id}/events`）。其余一切
   * ——手写帧解析、40 秒静默看门狗、退避重连、「seq 缺失不推进游标」——
   * 两条流**一个字都不能分叉**：那些细节各自对应一个已经踩过的坑，抄一份
   * 出来就等于让其中一份慢慢烂掉。
   */
  path?: string
  /** 已经收到的最大 seq。重连时从这里续，0 表示从头。 */
  since?: number
  onEvent: (event: RunEvent) => void
  /** 连接建立（或重连成功）时回调，参数是本次是否为重连。 */
  onOpen?: (reconnected: boolean) => void
  /** 一次连接正常结束（服务端关闭 / 终态）。不会重连。 */
  onClose?: (reason: 'terminal' | 'ended') => void
  onError?: (error: Error) => void
  /** 收到终态事件后是否继续等待。默认 false（即关闭流）。 */
  keepOpenOnTerminal?: boolean
}

export interface StreamHandle {
  close: () => void
  /** 当前已知的最大 seq。用于重连或切换运行。 */
  lastSeq: () => number
}

/**
 * 「这条运行已经结束了」的事件。**`error` 也在内**：一次失败同样是结局，
 * 后面不会再有新事件。把它排除在外，失败就会走「连接正常结束但没看到
 * 终态」那条歧义路径——而上面刚说过，那条路径以前不会重连。
 */
const TERMINAL = new Set([
  'run_completed',
  'run_cancelled',
  // 故事工坊一轮的三种结局。**注意 `agent_completed` 不是「会话结束」**：
  // 写完一稿之后会话还在待命，用户随时可以说「再暗一点」。所以调用方要传
  // `keepOpenOnTerminal` —— 关掉连接会触发退避重连，而后端那边这正是
  // 「待命的会话也要挂着连接」的理由（见 api/v1/agent.py 的模块注释）。
  'agent_completed',
  'agent_cancelled',
  'error',
])

/**
 * 静默看门狗的超时。服务端每 15 秒（`SSE_HEARTBEAT_SECONDS`）发一条
 * 注释帧保活，所以「连着 40 秒一个字节都没收到」不是安静，是死了。
 */
const IDLE_TIMEOUT_MS = 40_000

/**
 * 建立 SSE 连接，返回一个可关闭的句柄。
 *
 * 断线由内部指数退避重连：500ms → 1s → 2s → 4s → 5s（上限），带 ±30% 抖动。
 * 抖动是必要的——多个标签页同时重连会形成同步脉冲。
 */
export function openRunStream(options: StreamOptions): StreamHandle {
  const { runId, onEvent, onOpen, onClose, onError, keepOpenOnTerminal } = options
  const path = options.path ?? '/api/v1/runs/{id}/events'

  let lastSeq = options.since ?? 0
  let closed = false
  let attempt = 0
  // 退避重置与「是否算重连」是两件事：退避每次连上就归零（下次断线
  // 仍从 500ms 起步），但 reconnected 必须记住「曾经连上过」，
  // 否则界面会把每次重连都当成首次连接，已渲染的产物被当成新数据。
  let hasOpened = false
  let controller: AbortController | null = null
  let retryTimer: ReturnType<typeof setTimeout> | null = null
  let idleTimer: ReturnType<typeof setTimeout> | null = null

  const clearIdle = () => {
    if (idleTimer) clearTimeout(idleTimer)
    idleTimer = null
  }

  const abort = () => {
    clearIdle()
    controller?.abort()
    controller = null
  }

  /**
   * 静默看门狗。
   *
   * **必须有它，因为服务端死亡对浏览器是不可见的。** 后端进程被杀时，
   * dev 代理（vite）收到上游 ECONNRESET 只记一行日志，并不关闭下游连接
   * ——这条 fetch 流于是「活着但永远静默」：`read()` 不返回，`catch`
   * 不触发，重连逻辑没有起点。表现就是页面停在「分析中…」上直到刷新。
   * 只有把「多久没收到字节」当成判据，客户端才能自己发现这件事。
   */
  function onIdle() {
    if (closed) return
    abort()
    onError?.(new Error('事件流静默超时（后端可能已重启）'))
    scheduleRetry()
  }

  const bumpIdle = () => {
    if (idleTimer) clearTimeout(idleTimer)
    idleTimer = setTimeout(onIdle, IDLE_TIMEOUT_MS)
  }

  const scheduleRetry = () => {
    if (closed) return
    const base = Math.min(500 * 2 ** attempt, 5000)
    const jitter = base * 0.3 * (Math.random() * 2 - 1)
    attempt += 1
    retryTimer = setTimeout(connect, Math.max(200, base + jitter))
  }

  async function connect() {
    if (closed) return
    abort()
    controller = new AbortController()
    const reconnected = hasOpened

    let resp: Response
    try {
      resp = await fetch(
        `${path.replace('{id}', encodeURIComponent(runId))}?since=${lastSeq}`,
        {
          signal: controller.signal,
          headers: { accept: 'text/event-stream' },
          // 关掉浏览器层缓存，否则某些代理会缓存住整个响应体
          cache: 'no-store',
        },
      )
    } catch (err) {
      if (closed || (err as Error).name === 'AbortError') return
      onError?.(toError(err))
      scheduleRetry()
      return
    }

    if (!resp.ok || !resp.body) {
      // 404 意味着运行不存在——重连没有意义，直接报错收工
      if (resp.status === 404) {
        closed = true
        onError?.(new Error('运行记录不存在'))
        onClose?.('ended')
        return
      }
      onError?.(new Error(`事件流连接失败（${resp.status}）`))
      scheduleRetry()
      return
    }

    attempt = 0
    hasOpened = true
    bumpIdle()
    onOpen?.(reconnected)

    const reader = resp.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    let sawTerminal = false

    try {
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        // 心跳注释帧也会走到这里——它不产生事件，但证明连接是活的
        bumpIdle()
        buffer += decoder.decode(value, { stream: true })

        // 按空行切帧。最后一段可能不完整，留在 buffer 里等下一个 chunk。
        let sep: number
        while ((sep = buffer.indexOf('\n\n')) !== -1) {
          const raw = buffer.slice(0, sep)
          buffer = buffer.slice(sep + 2)
          const event = parseFrame(raw)
          if (!event) continue
          // 服务端的合成帧（如 error）曾经不带 seq。一旦拿它推进游标，
          // lastSeq 会变成 undefined，重连时拼出 ?since=undefined → 422，
          // 而且此后再也接不上。**没有合法 seq 的帧不推进游标**，
          // 但仍然交给上层——错误信息本身是要展示的。
          if (typeof event.seq !== 'number' || !Number.isFinite(event.seq)) {
            onError?.(new Error(`事件缺少 seq：${event.type}`))
            onEvent(event)
            continue
          }
          // 断线重连时服务端会重放边界帧（since 是排他的，但服务端
          // 与前端对「已收到」的认知可能差一帧）。在这里统一去重，
          // 让下游的 reducer 可以假设事件严格递增。
          if (event.seq <= lastSeq) continue
          lastSeq = event.seq
          onEvent(event)
          if (TERMINAL.has(event.type)) {
            sawTerminal = true
            if (!keepOpenOnTerminal) {
              closed = true
              abort()
              onClose?.('terminal')
              return
            }
          }
        }
      }
    } catch (err) {
      if (closed || (err as Error).name === 'AbortError') return
      onError?.(toError(err))
      scheduleRetry()
      return
    }

    clearIdle()
    if (closed) return
    // 服务端关流了。两种可能，必须分开处理：
    //   - 送过终态帧：运行真的结束了。
    //   - 一个终态帧都没有：**连接是在半路断的**（进程被杀、服务重启、
    //     网络抖动），而不是运行结束。这时唯一正确的动作是重连——
    //     早先这里直接放弃，于是页面停在「分析中…」上再也不会动，
    //     用户只能刷新。
    if (sawTerminal) {
      closed = true
      onClose?.('terminal')
    } else {
      onClose?.('ended')
      scheduleRetry()
    }
  }

  void connect()

  return {
    close: () => {
      closed = true
      if (retryTimer) clearTimeout(retryTimer)
      abort()
    },
    lastSeq: () => lastSeq,
  }
}

/** 解析一个 SSE 帧（已去掉结尾的空行）。注释帧与无 data 的帧返回 null。 */
export function parseFrame(raw: string): RunEvent | null {
  let data = ''
  for (const line of raw.split('\n')) {
    if (line.startsWith(':')) continue // 心跳注释帧
    if (line.startsWith('data:')) {
      // 按规范只去掉紧跟冒号的一个空格，不改动其余内容
      const chunk = line.slice(5)
      data += chunk.startsWith(' ') ? chunk.slice(1) : chunk
    }
  }
  if (!data) return null
  try {
    return JSON.parse(data) as RunEvent
  } catch {
    // 半个 JSON —— 只可能是服务端 bug，不值得让整个流崩掉
    return null
  }
}

function toError(err: unknown): Error {
  return err instanceof Error ? err : new Error(String(err))
}
