/**
 * REST 端点。路径统一带 `/api/v1` 前缀，由 vite 代理转发到后端。
 *
 * 一律用相对路径：SSE 需要同源（EventSource 不支持自定义头，跨域要靠
 * 一堆额外配置），而相对路径让代理配置成为唯一的地址来源。
 */

import type {
  AgentCapabilities,
  AgentOptions,
  AgentSession,
  AgentSessionDetail,
  Capabilities,
  CreateRunResponse,
  Depth,
  KbDeleteResult,
  KbImportCreated,
  KbImportJob,
  KbSource,
  KbSourceDocument,
  KbSourceText,
  LibraryKey,
  NodeSpec,
  NovelHealth,
  NovelText,
  RunDetail,
  RunSummary,
  SourceKind,
  TextMode,
} from './types'

const BASE = '/api/v1'

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let resp: Response
  try {
    resp = await fetch(`${BASE}${path}`, {
      headers: { 'content-type': 'application/json' },
      ...init,
    })
  } catch (cause) {
    // 网络层失败（后端没起来、代理断开）。与 HTTP 错误分开报，
    // 因为「后端没连上」需要给出完全不同的提示。
    throw new ApiError('无法连接后端服务，请确认服务已启动', 0, cause)
  }

  if (!resp.ok) {
    let detail: unknown
    try {
      detail = await resp.json()
    } catch {
      detail = await resp.text().catch(() => undefined)
    }
    throw new ApiError(extractMessage(detail) ?? `请求失败（${resp.status}）`, resp.status, detail)
  }
  return (await resp.json()) as T
}

function extractMessage(detail: unknown): string | null {
  if (typeof detail === 'string' && detail.trim()) return detail
  if (detail && typeof detail === 'object' && 'detail' in detail) {
    const d = (detail as { detail: unknown }).detail
    // FastAPI 的校验错误是数组，取第一条的 msg 而不是把整坨 JSON 丢给用户
    if (typeof d === 'string') return d
    if (Array.isArray(d) && d.length > 0) {
      const first = d[0] as { msg?: string; loc?: unknown[] }
      if (first?.msg) {
        const field = Array.isArray(first.loc) ? first.loc[first.loc.length - 1] : undefined
        return field ? `${String(field)}：${first.msg}` : first.msg
      }
    }
  }
  return null
}

export interface CreateRunPayload {
  input: string
  /** 省略即抖音。文本源必须显式传 `text`，后端据此走 `_resolve_input` 的另一条分支。 */
  source?: SourceKind
  /** 仅 `source=text` 时有意义；抖音路径会接受并忽略它。 */
  text_mode?: TextMode
  depth: Depth
  kb: Record<LibraryKey, boolean>
  comment_limit?: number
}

export const api = {
  createRun: (payload: CreateRunPayload) =>
    request<CreateRunResponse>('/runs', { method: 'POST', body: JSON.stringify(payload) }),

  getRun: (runId: string) => request<RunDetail>(`/runs/${runId}`),

  listRuns: (limit = 20, offset = 0) =>
    request<RunSummary[]>(`/runs?limit=${limit}&offset=${offset}`),

  cancelRun: (runId: string) =>
    request<{ accepted: boolean; message: string }>(`/runs/${runId}/cancel`, { method: 'POST' }),

  pipeline: () => request<{ nodes: NodeSpec[] }>('/meta/pipeline'),

  capabilities: () => request<Capabilities>('/meta/capabilities'),

  /* ---------------- 知识库导入 ---------------- */

  createKbImport: (payload: KbImportPayload) => {
    const form = new FormData()
    form.append('file', payload.file)
    form.append('library', payload.library)
    form.append('title', payload.title ?? '')
    form.append('author', payload.author ?? '')
    form.append('discipline', payload.discipline ?? '')
    form.append('use_llm', String(payload.use_llm ?? true))
    form.append('allow_partial', String(payload.allow_partial ?? false))
    // `headers: {}` 必须显式传：request() 的默认头是 application/json，
    // 而它在展开顺序上会把整个 headers 对象替换掉。不覆盖的话，
    // 浏览器不会补 multipart 的 boundary，后端收到的是一个无法解析的 body。
    return request<KbImportCreated>('/kb/imports', { method: 'POST', body: form, headers: {} })
  },

  getKbImport: (jobId: string) => request<KbImportJob>(`/kb/imports/${jobId}`),

  listKbImports: (limit = 20) => request<KbImportJob[]>(`/kb/imports?limit=${limit}`),

  startKbImport: (jobId: string) =>
    request<KbImportJob>(`/kb/imports/${jobId}/start`, { method: 'POST' }),

  cancelKbImport: (jobId: string) =>
    request<{ accepted: boolean; message: string }>(`/kb/imports/${jobId}/cancel`, {
      method: 'POST',
    }),

  listKbSources: () => request<KbSource[]>('/kb/sources'),

  kbSourceDocuments: (sourceFile: string, limit = 200) =>
    request<KbSourceDocument[]>(
      `/kb/sources/${encodeURIComponent(sourceFile)}/documents?limit=${limit}`,
    ),

  deleteKbSource: (sourceFile: string) =>
    request<KbDeleteResult>(`/kb/sources/${encodeURIComponent(sourceFile)}`, {
      method: 'DELETE',
    }),

  kbSourceText: (sourceFile: string, offset = 0, limit = 60_000) =>
    request<KbSourceText>(
      `/kb/sources/${encodeURIComponent(sourceFile)}/text?offset=${offset}&limit=${limit}`,
    ),

  /* ---------------- 古典文学（另一个仓的 MCP） ----------------
   * 全部只读 GET，返回的都是 {tool, markdown}。
   * 入参用对象而不是位置参数：六个工具的参数各不相同，位置参数在调用点
   * 读不出谁是谁（`novelChapterText(2, 8)` 是哪两个数？）。 */

  novelHealth: () => request<NovelHealth>('/novel/health'),

  novelBooks: (p: { query?: string; limit?: number; offset?: number } = {}) =>
    request<NovelText>(`/novel/books?${qs(p)}`),

  novelCharacters: (p: { name: string; book_id?: number; limit?: number }) =>
    request<NovelText>(`/novel/characters?${qs(p)}`),

  novelDialogues: (p: {
    character_id: number
    book_id?: number
    chapter_from?: number
    chapter_to?: number
    query?: string
    limit?: number
    cursor?: number
  }) => request<NovelText>(`/novel/dialogues?${qs(p)}`),

  novelChapters: (p: { book_id: number; query?: string; limit?: number; offset?: number }) =>
    request<NovelText>(`/novel/chapters?${qs(p)}`),

  novelChapterText: (p: { book_id: number; chapter_idx: number; cursor?: number; limit?: number }) =>
    request<NovelText>(`/novel/chapter-text?${qs(p)}`),

  novelPassageText: (p: { book_id: number; global_idx: number; before?: number; after?: number }) =>
    request<NovelText>(`/novel/passage-text?${qs(p)}`),

  /* ---------------- 故事工坊 ----------------
   * 与流水线并列的一条独立链路：不占节点、不动进度带、与当前分析运行无关。
   * 事件流走 `openRunStream` 的 `path` 选项，不另写一份 SSE。 */

  agentCapabilities: () => request<AgentCapabilities>('/agent/capabilities'),

  createAgentSession: (p: { input: string } & AgentOptions) =>
    request<AgentSessionDetail>('/agent/sessions', {
      method: 'POST',
      body: JSON.stringify(p),
    }),

  getAgentSession: (sessionId: string) =>
    request<AgentSessionDetail>(`/agent/sessions/${sessionId}`),

  /** 202：返回的是「已收下」，正文要等事件流里那串 `delta` 敲完。 */
  sendAgentMessage: (sessionId: string, text: string) =>
    request<AgentSession>(`/agent/sessions/${sessionId}/messages`, {
      method: 'POST',
      body: JSON.stringify({ text }),
    }),

  cancelAgentSession: (sessionId: string) =>
    request<{ accepted: boolean; message: string }>(`/agent/sessions/${sessionId}/cancel`, {
      method: 'POST',
    }),
}

/**
 * 组装查询串。**空值一律不发**：`query=` 与不传 `query` 在后端是两回事
 *（前者会去匹配空关键词），而 `cursor=0` 这种合法值不能被当成「没填」。
 */
function qs(params: Record<string, string | number | undefined>): string {
  const sp = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === '') continue
    sp.set(key, String(value))
  }
  return sp.toString()
}

export interface KbImportPayload {
  file: File
  library: LibraryKey
  title?: string
  author?: string
  discipline?: string
  use_llm?: boolean
  allow_partial?: boolean
}
