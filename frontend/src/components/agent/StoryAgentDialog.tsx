/**
 * 故事工坊：三栏面板（材料 / 对话与过程 / 正文）。
 *
 * 与三栏工作台**完全隔离**：不占节点、不动进度带、不碰当前分析运行。同一个
 * 标签页里可以一边跑着分析、一边让写手改稿——这两件事没有共享状态，唯一的
 * 交点是顶栏那两个入口按钮。
 *
 * 本组件只做「把用户操作转成 API 调用」与拼装三栏；状态一律在 agentStore 里，
 * 业务规则一律在 applyAgentEvent 里。与 Workbench 的分工一模一样。
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { api, ApiError } from '../../api/endpoints'
import { useAgentStream } from '../../hooks/useAgentStream'
import { clearActiveAgent, loadActiveAgent, saveActiveAgent } from '../../store/activeAgent'
import { useAgentStore } from '../../store/agentStore'
import { Modal } from '../common/Modal'
import { AgentInputPane } from './AgentInputPane'
import { AgentTranscript } from './AgentTranscript'
import { StoryPane } from './StoryPane'

export function StoryAgentDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [draft, setDraft] = useState('')
  const [instruction, setInstruction] = useState('')
  const [allowNovel, setAllowNovel] = useState(true)
  const [targetChars, setTargetChars] = useState(900)
  const [starting, setStarting] = useState(false)
  const [stopping, setStopping] = useState(false)
  // 两个错误分开存：它们产生在两块不同的地方，就该显示在各自的地方。
  // 合成一个的话，一次「开始写」失败会显示在对话栏里——而用户刚才按的按钮
  // 在左边，他会对着一个跟他刚做的事无关的位置发愣。
  const [startError, setStartError] = useState<string | null>(null)
  const [sendError, setSendError] = useState<string | null>(null)

  const stream = useAgentStream()

  const sessionId = useAgentStore((s) => s.sessionId)
  const status = useAgentStore((s) => s.status)
  const stage = useAgentStore((s) => s.stage)
  const liveStage = useAgentStore((s) => s.liveStage)
  const title = useAgentStore((s) => s.title)
  const inputText = useAgentStore((s) => s.inputText)
  const messages = useAgentStore((s) => s.messages)
  const cards = useAgentStore((s) => s.cards)
  const story = useAgentStore((s) => s.story)
  const streaming = useAgentStore((s) => s.streaming)
  const turn = useAgentStore((s) => s.turn)
  const rounds = useAgentStore((s) => s.rounds)
  const toolCalls = useAgentStore((s) => s.toolCalls)
  const model = useAgentStore((s) => s.model)
  const error = useAgentStore((s) => s.error)
  const demo = useAgentStore((s) => s.demo)
  const warnings = useAgentStore((s) => s.warnings)
  const capabilities = useAgentStore((s) => s.capabilities)

  const running = status === 'running'
  const terminal = status === 'failed' || status === 'cancelled'

  // 打开时抓一次能力，并把上一次的会话接回来。**只做一次**——关闭再打开
  // 不该把用户正在看的那一场丢掉。
  const resumedRef = useRef(false)
  useEffect(() => {
    if (!open) return
    let alive = true

    if (!useAgentStore.getState().capabilities) {
      void api
        .agentCapabilities()
        .then((cap) => {
          if (!alive) return
          useAgentStore.getState().setCapabilities(cap)
          setTargetChars(cap.target_chars)
          setAllowNovel(cap.allow_novel)
        })
        .catch(() => undefined)
    }

    if (resumedRef.current) return
    resumedRef.current = true
    const saved = loadActiveAgent()
    if (!saved || useAgentStore.getState().sessionId) return

    void (async () => {
      try {
        const detail = await api.getAgentSession(saved.sessionId)
        if (!alive) return
        // 快照先把对话与正文重建出来，再开流把**过程**（工具卡片）重放回来。
        // 两条缺一不可：快照里没有事件，流里没有正文。
        useAgentStore.getState().openSession(saved.sessionId, detail)
        stream.start(saved.sessionId, { since: 0 })
      } catch {
        // 会话已被清理（换库、清空数据）或后端还没起来。静默退回空面板。
        clearActiveAgent()
      }
    })()

    return () => {
      alive = false
    }
  }, [open, stream])

  // 关面板就断开流（后端那边会继续把这一轮跑完，重新打开时回放补上）
  useEffect(() => {
    if (!open) stream.stop()
  }, [open, stream])

  // 一轮结束后把「停止中…」复位。放在副作用里而不是 await 之后：标志位
  // 置下之后这一轮还可能活着，此刻复位会让按钮闪回「停止」而用户可以再点一次。
  useEffect(() => {
    if (!running) setStopping(false)
  }, [running])

  const onStart = useCallback(async () => {
    const value = draft.trim()
    if (!value || starting) return
    setStarting(true)
    setStartError(null)
    try {
      const detail = await api.createAgentSession({
        input: value,
        allow_novel: allowNovel,
        target_chars: targetChars,
      })
      // 先落盘再开流：中间刷新才不会丢掉这一场（与 Workbench 的 launch 同一条规矩）
      saveActiveAgent(detail.id)
      useAgentStore.getState().openSession(detail.id, detail)
      stream.start(detail.id)
      setDraft('')
    } catch (err) {
      setStartError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setStarting(false)
    }
  }, [allowNovel, draft, starting, stream, targetChars])

  const onSend = useCallback(async () => {
    const id = useAgentStore.getState().sessionId
    const text = instruction.trim()
    if (!id || !text || running) return
    setSendError(null)
    // 先清空输入框（乐观）：事件流马上会推 agent_started，用户的这句话会以
    // 气泡的形式回来。失败时再把这句还给他——比留在框里、看起来没发出去强。
    setInstruction('')
    try {
      await api.sendAgentMessage(id, text)
      // **发完必须确认那头有人接。** 连接死掉时我们是不知道的：`isStreaming()`
      // 看的是 handle 在不在，而连接断了 handle 照样在。曾经的症状是第二个
      // 202 被送进一条已经断掉的连接——界面停在上一版正文、控制台干干净净，
      // 刷新一下才发现库里早就写好第二版了。
      //
      // 重开一次最省事，也不必去判连接是否还活着。`since` 用 store 里的游标：
      // 这段空档里漏掉的帧由服务端回放补上（这正是 `since` 存在的理由），
      // 而已经收到的会被 seq 挡掉，不会重画。
      stream.start(id, { since: useAgentStore.getState().lastSeq })
    } catch (err) {
      setInstruction(text)
      setSendError(err instanceof ApiError ? err.message : String(err))
    }
  }, [instruction, running, stream])

  const onStop = useCallback(async () => {
    const id = useAgentStore.getState().sessionId
    if (!id || stopping) return
    setStopping(true)
    try {
      const resp = await api.cancelAgentSession(id)
      // accepted=false：服务端认为它已经结束了（多半是流漏掉了终态事件）。
      // 拉一次快照补上，否则界面会一直停在「正在写」。
      if (!resp.accepted) {
        const detail = await api.getAgentSession(id)
        useAgentStore.getState().reconcile(detail)
      }
    } catch (err) {
      setStopping(false)
      setSendError(err instanceof ApiError ? err.message : String(err))
    }
  }, [stopping])

  const onRestart = useCallback(() => {
    stream.stop()
    clearActiveAgent()
    useAgentStore.getState().reset()
    setInstruction('')
    setStartError(null)
    setSendError(null)
  }, [stream])

  const hasSession = Boolean(sessionId)

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="✍ 写故事"
      subtitle={
        <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
          {hasSession ? (
            <>
              <span className="truncate" title={title}>
                {title || '（无标题）'}
              </span>
              <span className="text-ink-ghost">
                第 {turn} 稿 · {rounds} 轮 · {toolCalls} 次工具调用
              </span>
              {model ? <span className="text-ink-ghost font-mono">{model}</span> : null}
            </>
          ) : (
            <span>粘一段评论，让一个会自己查资料、写完还能改的写手来写</span>
          )}
          {demo ? (
            <span
              title="模型跑在 mock 上，正文是模板拼出来的"
              className="border-warn/30 bg-warn-soft text-warn inline-flex items-center gap-1 rounded-full border px-2 py-[2px] text-[10.5px] leading-none"
            >
              <span aria-hidden>◐</span>演示模式
            </span>
          ) : null}
        </span>
      }
      width="max-w-[1180px]"
      height="h-[86vh]"
      // 三栏各自滚：内容区交给子栏管滚动，自己不留内边距也不要滚动条
      bodyClassName="flex min-h-0 overflow-hidden"
      // 面板占了大半个屏，能按住标题栏挪开一点才看得见底下的工作台。
      draggable
    >
      <AgentInputPane
        hasSession={hasSession}
        inputText={inputText}
        draft={draft}
        onDraftChange={(v) => {
          setDraft(v)
          if (startError) setStartError(null)
        }}
        allowNovel={allowNovel}
        onAllowNovelChange={setAllowNovel}
        targetChars={targetChars}
        onTargetCharsChange={setTargetChars}
        capabilities={capabilities}
        starting={starting}
        error={startError}
        onStart={() => void onStart()}
        onRestart={onRestart}
        disabled={running}
      />

      <AgentTranscript
        messages={messages}
        cards={cards}
        story={story}
        status={status}
        liveStage={liveStage}
        stage={stage}
        instruction={instruction}
        onInstructionChange={(v) => {
          setInstruction(v)
          if (sendError) setSendError(null)
        }}
        onSubmit={() => void onSend()}
        onStop={() => void onStop()}
        stopping={stopping}
        note={sendError ?? warnings.at(-1) ?? null}
        disabled={!hasSession || terminal}
        instructionMax={capabilities?.instruction_max ?? 2000}
      />

      <StoryPane
        story={story}
        status={status}
        streaming={streaming}
        turn={turn}
        rounds={rounds}
        toolCalls={toolCalls}
        model={model}
        error={error}
        hasSession={hasSession}
      />
    </Modal>
  )
}
