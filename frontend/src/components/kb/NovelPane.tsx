import { useCallback, useEffect, useRef, useState } from 'react'

import { api, ApiError } from '../../api/endpoints'
import type { NovelHealth, NovelText } from '../../api/types'
import { Button, ErrorLine, Hint, Param, ResultBlock, num } from './textUi'

/**
 * 古典文学知识库（另一个仓 `ClassicalNovelProject`）的查阅面板。
 *
 * **只读、不进分析流水线**：这里没有 run_id、不发事件、不落库，
 * 一次点击 = 一次 MCP 调用 = 一段 Markdown。
 *
 * 五个模式对应对方的五个工具，其中一个（章节原文）是两步：
 * 先列回目拿 `chapter_idx`，再读整章。之所以不做成「点回目直接读」——
 * 那需要解析对方返回的 Markdown，而那份 Markdown 是人读的展示层，
 * 解析它等于把它当 API 用。
 */

type ModeKey = 'books' | 'characters' | 'dialogues' | 'chapters' | 'passage_text'

const MODES: { key: ModeKey; label: string; hint: string }[] = [
  { key: 'books', label: '书目', hint: 'book_id 只从这里来，其它模式都要它。' },
  { key: 'characters', label: '人物', hint: '按名字找人，顺带给别名与 character_id。' },
  { key: 'dialogues', label: '对话原文', hint: '某人的对话。每条的「段N」可以喂给段落原文。' },
  { key: 'chapters', label: '章节原文', hint: '先列回目拿 chapter_idx，再读整章原文。' },
  { key: 'passage_text', label: '段落原文', hint: '按全书段序取一段及其前后文。' },
]

/** 输入框全用字符串存：空串 = 不传，由后端按「不限定」处理。 */
const INITIAL: Record<string, string> = {
  limit: '20',
  chapter_limit: '80',
  before: '2',
  after: '2',
}

export function NovelPane({ active }: { active: boolean }) {
  const [mode, setMode] = useState<ModeKey>('books')
  const [fields, setFields] = useState(INITIAL)
  const [result, setResult] = useState<NovelText | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [health, setHealth] = useState<NovelHealth | null>(null)

  const set = (key: string) => (v: string) => setFields((prev) => ({ ...prev, [key]: v }))
  const field = (key: string) => ({ value: fields[key] ?? '', onChange: set(key) })

  const run = useCallback(async (call: () => Promise<NovelText>) => {
    setBusy(true)
    setError(null)
    try {
      setResult(await call())
    } catch (err) {
      // 503/504/502／422 在后端都给了一句能直接显示的中文，原样透出来，
      // 不在这里改成「请求失败」这种丢掉全部信息的文案。
      setResult(null)
      setError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }, [])

  // 打开弹窗先看一眼「这个入口现在能不能用」，顺手把书目拉出来——
  // 它是所有其它模式的前置（book_id 只能从这儿取），空着手进来必然要再点一次。
  const booted = useRef(false)
  useEffect(() => {
    if (!active || booted.current) return
    booted.current = true
    void api
      .novelHealth()
      .then(setHealth)
      .catch(() => undefined)
    void run(() => api.novelBooks({ limit: 20 }))
  }, [active, run])

  const fail = (message: string) => setError(message)

  const submit = () => {
    const bookId = num(fields.book_id)
    switch (mode) {
      case 'books':
        void run(() =>
          api.novelBooks({ query: trimmed(fields.query), limit: num(fields.limit) ?? 20 }),
        )
        return
      case 'characters': {
        const name = trimmed(fields.name)
        if (!name) return fail('人名必填，如「宝玉」。')
        return void run(() =>
          api.novelCharacters({ name, book_id: bookId, limit: num(fields.limit) ?? 20 }),
        )
      }
      case 'dialogues': {
        const characterId = num(fields.character_id)
        if (characterId === undefined) {
          return fail('character_id 必填：先用「人物」查一个（结果里有 `character_id=…`）。')
        }
        return void run(() =>
          api.novelDialogues({
            character_id: characterId,
            book_id: bookId,
            // 起止章号在后端是**成对**生效的，这里照原样传，不做补默认值
            chapter_from: num(fields.chapter_from),
            chapter_to: num(fields.chapter_to),
            limit: num(fields.limit) ?? 20,
            cursor: num(fields.cursor),
          }),
        )
      }
      case 'chapters': {
        if (bookId === undefined) return fail(missingBook())
        return void run(() =>
          api.novelChapters({
            book_id: bookId,
            query: trimmed(fields.query),
            limit: num(fields.limit) ?? 100,
            offset: num(fields.offset),
          }),
        )
      }
      case 'passage_text': {
        if (bookId === undefined) return fail(missingBook())
        const globalIdx = num(fields.global_idx)
        if (globalIdx === undefined) {
          return fail('段号必填：对话原文里的「段N」或整章原文里的「[N]」。')
        }
        return void run(() =>
          api.novelPassageText({
            book_id: bookId,
            global_idx: globalIdx,
            before: num(fields.before) ?? 2,
            after: num(fields.after) ?? 2,
          }),
        )
      }
    }
  }

  /** 「章节原文」的第二步。单独一个按钮，因为它和「列回目」是两个工具。 */
  const loadChapter = () => {
    const bookId = num(fields.book_id)
    if (bookId === undefined) return fail(missingBook())
    const chapterIdx = num(fields.chapter_idx)
    if (chapterIdx === undefined) {
      return fail('chapter_idx 必填：先点「列出回目」，从结果里取一个。')
    }
    void run(() =>
      api.novelChapterText({
        book_id: bookId,
        chapter_idx: chapterIdx,
        cursor: num(fields.cursor),
        limit: num(fields.chapter_limit) ?? 80,
      }),
    )
  }

  const current = MODES.find((m) => m.key === mode)!

  return (
    <div className="flex flex-col gap-3">
      {health && !health.ok ? (
        <ErrorLine
          text={[health.error, health.hint].filter(Boolean).join('\n')}
        />
      ) : null}

      <div className="flex flex-wrap gap-1.5" role="group" aria-label="查阅模式">
        {MODES.map((m) => (
          <button
            key={m.key}
            type="button"
            aria-pressed={mode === m.key}
            onClick={() => setMode(m.key)}
            className={`rounded-md border px-2.5 py-[5px] text-[12px] transition-colors ${
              mode === m.key
                ? 'border-accent-line bg-accent-soft text-accent'
                : 'border-line text-ink-faint hover:text-ink-soft'
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      <Hint>{current.hint}</Hint>

      <div className="flex flex-col gap-2.5">
        {mode === 'books' ? (
          <div className="grid grid-cols-[1fr_88px] gap-2">
            <Param label="书名关键词（可空）" placeholder="如：红楼" {...field('query')} onEnter={submit} />
            <Param label="本页条数" {...field('limit')} onEnter={submit} />
          </div>
        ) : null}

        {mode === 'characters' ? (
          <div className="grid grid-cols-[1fr_88px_88px] gap-2">
            <Param label="人名或别名" placeholder="如：宝玉" {...field('name')} onEnter={submit} />
            <Param label="book_id（可空）" {...field('book_id')} onEnter={submit} />
            <Param label="本页条数" {...field('limit')} onEnter={submit} />
          </div>
        ) : null}

        {mode === 'dialogues' ? (
          <>
            <div className="grid grid-cols-[1fr_88px_88px_88px] gap-2">
              <Param
                label="character_id"
                placeholder="来自「人物」"
                {...field('character_id')}
                onEnter={submit}
              />
              <Param label="book_id（可空）" {...field('book_id')} onEnter={submit} />
              <Param label="起始章" {...field('chapter_from')} onEnter={submit} />
              <Param label="结束章" {...field('chapter_to')} onEnter={submit} />
            </div>
            <div className="grid grid-cols-[1fr_88px] gap-2">
              <Param
                label="游标 cursor（翻页用，可空）"
                placeholder="把上一次结果末尾的 cursor 填进来"
                {...field('cursor')}
                onEnter={submit}
              />
              <Param label="本页条数" {...field('limit')} onEnter={submit} />
            </div>
          </>
        ) : null}

        {mode === 'chapters' ? (
          <>
            <div className="grid grid-cols-[88px_1fr_88px_88px] gap-2">
              <Param label="book_id" {...field('book_id')} onEnter={submit} />
              <Param label="回目关键词（可空）" {...field('query')} onEnter={submit} />
              <Param label="本页条数" {...field('limit')} onEnter={submit} />
              <Param label="offset" {...field('offset')} onEnter={submit} />
            </div>
            <div className="border-line grid grid-cols-[88px_88px_88px_1fr] gap-2 border-t pt-2.5">
              <Param label="chapter_idx" {...field('chapter_idx')} onEnter={loadChapter} />
              <Param label="游标 cursor" {...field('cursor')} onEnter={loadChapter} />
              <Param label="每页段数" {...field('chapter_limit')} onEnter={loadChapter} />
              <span className="text-ink-faint self-end pb-[6px] text-[11px] leading-relaxed">
                第二步：读这一回的整章原文，段号是全书段序。
              </span>
            </div>
          </>
        ) : null}

        {mode === 'passage_text' ? (
          <div className="grid grid-cols-[88px_1fr_72px_72px] gap-2">
            <Param label="book_id" {...field('book_id')} onEnter={submit} />
            <Param
              label="段号 global_idx"
              placeholder="对话结果里的「段N」或整章原文里的「[N]」"
              {...field('global_idx')}
              onEnter={submit}
            />
            <Param label="前面几段" {...field('before')} onEnter={submit} />
            <Param label="后面几段" {...field('after')} onEnter={submit} />
          </div>
        ) : null}

        <div className="flex items-center gap-2">
          <Button primary disabled={busy} onClick={submit}>
            {busy ? '查询中…' : mode === 'chapters' ? '列出回目' : '查询'}
          </Button>
          {mode === 'chapters' ? (
            <Button disabled={busy} onClick={loadChapter}>
              读取整章原文
            </Button>
          ) : null}
          <Hint>
            {busy
              ? '每次调用要在对方那边起一个子进程，第一次约 2 秒。'
              : '内容由另一个仓（红楼梦库）渲染成 Markdown 后原样返回，这里不做改写。'}
          </Hint>
        </div>
      </div>

      {error ? <ErrorLine text={error} /> : null}

      {result ? (
        <div>
          <div className="text-ink-ghost flex items-baseline justify-between text-[11px]">
            <span className="font-mono">{result.tool}</span>
            <span className="tabular-nums">{result.markdown.length} 字</span>
          </div>
          <ResultBlock text={result.markdown} />
        </div>
      ) : null}
    </div>
  )
}

const missingBook = () => 'book_id 必填：先用「书目」查一个。'
const trimmed = (raw: string | undefined) => (raw ?? '').trim() || undefined
