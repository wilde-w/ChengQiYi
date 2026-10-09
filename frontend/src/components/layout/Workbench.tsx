import { useCallback, useEffect, useRef, useState } from 'react'

import { api, ApiError, type CreateRunPayload } from '../../api/endpoints'
import type { Providers } from '../../api/types'
import { useRunStream } from '../../hooks/useRunStream'
import { ResizeHandle, useResizable } from '../../hooks/useResizable'
import { clearActiveRun, loadActiveRun, saveActiveRun } from '../../store/activeRun'
import { selectCommentStats, useRunStore } from '../../store/runStore'
import { StoryAgentDialog } from '../agent/StoryAgentDialog'
import { EmptyState } from '../common/EmptyState'
import { KbImportDialog } from '../kb/KbImportDialog'
import { KbTextDialog } from '../kb/KbTextDialog'
import { ClusterBoard } from '../middle/ClusterBoard'
import { EvidenceBoard } from '../middle/EvidenceBoard'
import { ProfileCard } from '../middle/ProfileCard'
import { ProgressStream } from '../middle/ProgressStream'
import { ReasoningChain } from '../middle/ReasoningChain'
import { SectionBoard } from '../right/SectionBoard'
import { TextSourceDialog } from '../source/TextSourceDialog'
import { ColumnShell } from './ColumnShell'
import { BottomBar } from './BottomBar'
import { TopBar, KB_OPTIONS, type Depth, type KbSelection } from './TopBar'

const INITIAL_KB = Object.fromEntries(KB_OPTIONS.map((o) => [o.key, true])) as KbSelection

/**
 * 三栏工作台。
 *
 * 本文件的职责只有「布局 + 把用户操作转成 store/网络调用」。
 * 业务状态一律来自 store，业务规则一律在 store/applyEvent 里。
 */
export function Workbench() {
  const [input, setInput] = useState('')
  const [depth, setDepth] = useState<Depth>('standard')
  const [kb, setKb] = useState<KbSelection>(INITIAL_KB)
  const [providers, setProviders] = useState<Providers | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [aborting, setAborting] = useState(false)
  const [inputError, setInputError] = useState<string | null>(null)
  const [kbOpen, setKbOpen] = useState(false)
  const [kbTextOpen, setKbTextOpen] = useState(false)
  const [agentOpen, setAgentOpen] = useState(false)
  const [textOpen, setTextOpen] = useState(false)

  const stream = useRunStream()

  const setSpecs = useRunStore((s) => s.setSpecs)
  const status = useRunStore((s) => s.status)
  const progress = useRunStore((s) => s.progress)
  const currentMessage = useRunStore((s) => s.currentMessage)
  const error = useRunStore((s) => s.error)
  const runId = useRunStore((s) => s.runId)
  const durationMs = useRunStore((s) => s.durationMs)
  const stats = useRunStore(selectCommentStats)
  const comments = useRunStore((s) => s.comments)
  const sourceKind = useRunStore((s) => s.sourceKind)
  const sectionCount = useRunStore((s) => Object.keys(s.sections).length)
  const isText = sourceKind === 'text'

  const left = useResizable({ storageKey: 'left', initial: 340, min: 260, max: 560, direction: 1 })
  const right = useResizable({ storageKey: 'right', initial: 440, min: 320, max: 720, direction: -1 })

  // 启动时抓一次：分带文案与 provider 模式都不该硬编码在前端
  useEffect(() => {
    let alive = true
    void api
      .pipeline()
      .then((r) => alive && setSpecs(r.nodes))
      .catch(() => undefined)
    void api
      .capabilities()
      .then((r) => alive && setProviders(r.providers))
      .catch(() => undefined)
    return () => {
      alive = false
    }
  }, [setSpecs])

  const running = status === 'running' || submitting

  /**
   * 刷新后恢复上一次运行。
   *
   * 分两种情况，因为「重放」只在还没跑完时才有意义：
   *   - 仍在跑：把 runId 交还给事件流，服务端重放历史把整个界面重建出来。
   *   - 已结束：快照本身就是完整产物，直接对账即可，不必再开流。
   */
  const resumedRef = useRef(false)
  useEffect(() => {
    if (resumedRef.current) return
    resumedRef.current = true

    const saved = loadActiveRun()
    if (!saved) return
    setInput(saved.input)

    void (async () => {
      try {
        const detail = await api.getRun(saved.runId)
        // **在这段 await 期间用户可能已经点了「开始分析」。**
        // 恢复流程这时必须让步：继续往下走会把 runId 换回旧的那条，再用
        // 旧快照的产物覆盖新运行的界面，而新流的事件会被 store 的
        // runId 守卫全部丢掉。表现是「点了按钮，界面停在上一轮的结果上，
        // 进度条也不动」——一个看起来完全没反应的按钮。
        // 顺带一提，这里也不能 clearActiveRun()：那删掉的是**新运行**的存档。
        if (useRunStore.getState().runId) return

        setProviders(detail.providers)
        // 数据源也以快照为准：老运行没有这个字段，按抖音算。
        // 没有它，刷新后回到一条文本分析会显示成「N 条评论」。
        const kind = detail.source_kind ?? 'douyin'
        if (detail.status === 'succeeded' || detail.status === 'failed' || detail.status === 'cancelled') {
          useRunStore.getState().startRun(saved.runId, 0, kind)
          useRunStore.getState().reconcile(detail)
          clearActiveRun()
          return
        }
        stream.start(saved.runId, { progress: detail.progress, sourceKind: kind })
      } catch {
        // 运行已被清理（换库、清空数据）或后端还没起来。
        // 这不是用户能处理的问题，静默退回空工作台即可。
        clearActiveRun()
      }
    })()
  }, [stream])

  /**
   * 提交一次分析。顶栏（链接）与左栏弹窗（粘贴文本）共用这一条路径。
   *
   * 返回错误文案、成功返回 null，而不是自己往某个输入框里塞错误：
   * 两个入口的错误要显示在**各自**的地方——文本源的失败出现在顶栏那个
   * 空着的链接框下面，用户会对着一个跟他刚做的事无关的位置发愣。
   */
  const launch = useCallback(
    async (payload: CreateRunPayload): Promise<string | null> => {
      try {
        const resp = await api.createRun(payload)
        setProviders(resp.run.providers)
        // 先落盘再开流：反过来的话，在两者之间刷新会丢掉这次运行。
        // 文本源存空串：sessionStorage 里塞几千字没有意义，恢复时也不该
        // 把那坨文本灌回顶栏的链接输入框。
        saveActiveRun(resp.run.id, payload.source === 'text' ? '' : payload.input)
        stream.start(resp.run.id, { sourceKind: resp.run.source_kind ?? payload.source ?? 'douyin' })
        return null
      } catch (err) {
        // 输入不合法是**用户错误**，要当场说清哪里不对，
        // 而不是先建一条注定失败的运行再让他去读日志
        return err instanceof ApiError ? err.message : String(err)
      }
    },
    [stream],
  )

  const onStart = useCallback(async () => {
    const value = input.trim()
    if (!value || running) return
    setSubmitting(true)
    setInputError(null)
    const message = await launch({ input: value, depth, kb, comment_limit: 100 })
    if (message) setInputError(message)
    setSubmitting(false)
  }, [input, depth, kb, running, launch])

  /**
   * 中止当前分析。
   *
   * 服务端是「置标志位、节点边界退出」，所以请求成功后运行还会再跑一小会儿
   * ——按钮因此停在「中止中…」而不是立刻复位。复位交给 `running` 变 false
   * （流收到 run_cancelled 时）的那个副作用。
   */
  const onAbort = useCallback(async () => {
    const id = useRunStore.getState().runId
    if (!id || aborting) return
    setAborting(true)
    try {
      const resp = await api.cancelRun(id)
      // accepted=false 说明服务端认为它已经结束了——多半是流漏掉了终态事件
      // （断线、进程重启）。这时拉一次快照补上，否则按钮会一直停在「中止中…」。
      if (!resp.accepted) {
        const detail = await api.getRun(id)
        useRunStore.getState().reconcile(detail)
      }
    } catch (err) {
      setAborting(false)
      setInputError(err instanceof ApiError ? err.message : String(err))
    }
  }, [aborting])

  // 运行结束（含被中止）就复位按钮。放在副作用里而不是上面的 await 之后：
  // 标志位置下之后运行还可能活着，此刻复位会让按钮闪回「中止」、
  // 用户可以再点一次，而第二次点击没有任何意义。
  useEffect(() => {
    if (!running) setAborting(false)
  }, [running])

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <TopBar
        input={input}
        onInputChange={(v) => {
          setInput(v)
          if (inputError) setInputError(null)
        }}
        depth={depth}
        onDepthChange={setDepth}
        kb={kb}
        onKbChange={setKb}
        onStart={() => void onStart()}
        onAbort={() => void onAbort()}
        running={running}
        aborting={aborting}
        canStart={input.trim().length > 0}
        providers={providers}
        progress={progress}
        progressMessage={inputError ?? currentMessage}
        onOpenKb={() => setKbOpen(true)}
        onOpenKbText={() => setKbTextOpen(true)}
        onOpenAgent={() => setAgentOpen(true)}
      />

      <main className="flex min-h-0 flex-1">
        <ColumnShell
          title="数据源"
          status={columnStatus(runId, status)}
          subtitle={stats ? (isText ? `手动文本 · ${stats.total} 条` : `${stats.total} 条评论`) : undefined}
          width={left.width}
          footer={
            <div className="flex items-center justify-between gap-2">
              <span className="text-ink-ghost truncate text-[11px]">
                {stats
                  ? `保留 ${stats.kept} · 广告 ${stats.ads} · 灌水 ${stats.spam} · 重复 ${stats.duplicates}`
                  : isText
                    ? '手动文本 · 每行一条'
                    : '视频信息 · 评论 · 口播文案'}
              </span>
              {/* 第二个数据源入口。放在栏脚而不是顶栏：它和链接输入框是**并列的
                  两种数据源**，而不是链路上的一步；同时顶栏那一行已经没有位置了。 */}
              <button
                type="button"
                onClick={() => setTextOpen(true)}
                disabled={running}
                title={
                  running ? '分析进行中，等这次跑完再开下一次' : '粘贴一段文字，走同一条分析流程'
                }
                className="border-line text-ink-soft hover:border-line-strong hover:text-ink shrink-0 rounded-md border px-2 py-[3px] text-[11px] transition-colors disabled:cursor-not-allowed disabled:opacity-50"
              >
                ＋ 粘贴文本分析
              </button>
            </div>
          }
        >
          {!runId ? (
            <EmptyState
              icon="▤"
              title="尚无数据"
              hint="开始分析后，这里会依次出现视频信息、评论区样本与口播文案。没有链接的话，点下面的「粘贴文本分析」，粘一段文字同样可以分析。"
            />
          ) : comments.length === 0 ? (
            <EmptyState
              icon="▤"
              title={isText ? '正在切分文本' : '正在获取视频与评论'}
              hint={isText ? '切分是瞬时的，条目马上就会出现。' : '评论会分页到达，这里将逐页增长。'}
            />
          ) : (
            <CommentPreview isText={isText} />
          )}
        </ColumnShell>

        <ResizeHandle dragging={left.dragging} {...left.handleProps} />

        <ColumnShell
          title="分析过程"
          status={columnStatus(runId, status)}
          headerExtra={
            <div className="bg-paper-sunk h-[3px] overflow-hidden rounded-full">
              <div
                className="bg-accent h-full rounded-full transition-[width] duration-500 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
          }
        >
          {!runId ? (
            <EmptyState
              icon="◈"
              title="等待开始"
              hint="评论聚类、心理语义抽取、证据检索与推理链会在这里逐步展开。"
            />
          ) : (
            <>
              {/* 产物在上、过程在下：用户要读的是主题、侧写、证据和推理，
                  「跑到第几步」只在等的过程中看一眼。
                  顺序即阅读顺序——先看评论被分成了什么，再看证据，
                  最后看这些证据被串成了什么。 */}
              <ClusterBoard />
              <ProfileCard />
              <EvidenceBoard />
              <ReasoningChain />
              <ProgressStream />
            </>
          )}
        </ColumnShell>

        <ResizeHandle dragging={right.dragging} {...right.handleProps} />

        <ColumnShell
          title="洞察输出"
          status={columnStatus(runId, status)}
          width={right.width}
          footer={
            <span className="text-ink-ghost text-[11px]">心理侧写 · 机制 · 类比 · 洞察</span>
          }
        >
          {error ? (
            <EmptyState icon="✕" title="分析失败" hint={error.message} />
          ) : sectionCount > 0 ? (
            <SectionBoard />
          ) : (
            <EmptyState
              icon="◉"
              title={runId ? '等待生成' : '尚无洞察'}
              hint="分析完成后，这里输出心理侧写、科学机制、文学类比与最终洞察，每段结论都可追溯到证据来源。"
            />
          )}
        </ColumnShell>
      </main>

      <BottomBar
        onRerun={() => void onStart()}
        onSaveCase={() => {}}
        canAct={Boolean(runId) && !running}
        // 文本源没有可回填的输入（顶栏是链接框），重跑得重新粘一次
        canRerun={!isText}
        runId={runId}
        durationMs={durationMs}
      />

      <KbImportDialog open={kbOpen} onClose={() => setKbOpen(false)} />

      <KbTextDialog open={kbTextOpen} onClose={() => setKbTextOpen(false)} />

      {/* 故事工坊。挂在这里而不是塞进某一栏：它有自己的会话与事件流，
          与三栏里的任何一栏都不是同一个东西。 */}
      <StoryAgentDialog open={agentOpen} onClose={() => setAgentOpen(false)} />

      <TextSourceDialog
        open={textOpen}
        onClose={() => setTextOpen(false)}
        depth={depth}
        kb={kb}
        running={running}
        onLaunch={launch}
      />
    </div>
  )
}

/**
 * 三栏状态只由**整体运行状态**决定，不按栏细分。
 *
 * 刻意如此：早期设计里每栏各自判断自己那几个节点，结果是
 * 「左栏说完成、顶栏还在转」这类自相矛盾的界面。
 * 节点级的进度由中栏的清单精确表达，栏头只需要一个粗粒度信号。
 */
function columnStatus(
  runId: string | null,
  status: string,
): 'idle' | 'running' | 'done' | 'failed' {
  if (!runId) return 'idle'
  if (status === 'failed') return 'failed'
  if (status === 'succeeded' || status === 'cancelled') return 'done'
  return 'running'
}

/** M4 阶段的评论占位列表。M5 会替换成带排序/过滤/分页的完整组件。 */
function CommentPreview({ isText = false }: { isText?: boolean }) {
  const comments = useRunStore((s) => s.comments)
  return (
    <ul className="divide-line divide-y">
      {comments.slice(0, 60).map((c, index) => (
        <li key={c.comment_id} className="animate-rise px-4 py-2.5">
          <div className="text-ink-soft flex items-baseline justify-between gap-2 text-[11px]">
            {/* 文本源没有作者，左边这栏留空会显得像渲染坏了 */}
            <span className="truncate">{isText ? '手动文本' : c.author_name || '匿名'}</span>
            <span className="text-ink-ghost shrink-0 font-mono tabular-nums">
              {/* 文本源的点赞数恒为 0，写成「♥ 0」是三十行一模一样的信息；
                  行号才有信息量——它就是用户粘贴时的第几行。 */}
              {isText ? `#${sequenceOf(c.comment_id, index)}` : `♥ ${formatCount(c.like_count)}`}
            </span>
          </div>
          <p className="text-ink mt-1 text-[12.5px] leading-relaxed break-words">{c.text}</p>
        </li>
      ))}
    </ul>
  )
}

/**
 * 文本源的条目序号。
 *
 * 优先从合成 id（`txt-0007`）反解而不是用列表下标：将来加上「隐藏被过滤
 * 条目」这类开关后下标会重新编号，而 id 是这次运行里固定的身份。
 *
 * 但流式阶段与快照对账后的 id 不是同一个东西（见 types.ts 的 `comment_id`）：
 * 流里发的是切分时合成的 `txt-N`，落库后取回的是行 id。所以下标这条兜底
 * **必须留着**——两条路径算出的编号一致，因为文本源的排序就是粘贴顺序。
 */
function sequenceOf(commentId: string, fallbackIndex: number): number {
  const matched = /^txt-(\d+)$/.exec(commentId)
  return matched ? Number(matched[1]) : fallbackIndex + 1
}

function formatCount(n: number): string {
  if (n >= 10000) return `${(n / 10000).toFixed(1)}w`
  return String(n)
}
