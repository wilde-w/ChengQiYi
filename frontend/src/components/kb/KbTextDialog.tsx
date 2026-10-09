import { useState } from 'react'

import { Modal } from '../common/Modal'
import { KbSourceTextPane } from './KbSourceTextPane'
import { NovelPane } from './NovelPane'
import { Button, Hint } from './textUi'

type Tab = 'novel' | 'own'

const TABS: { key: Tab; label: string; hint: string }[] = [
  {
    key: 'novel',
    label: '古典文学知识库',
    hint: '另一个仓（ClassicalNovelProject）的红楼梦库，经 MCP 只读查阅：书目、人物、对话、回目、整章与段落原文。',
  },
  {
    key: 'own',
    label: '观心知识库',
    hint: '导入到观心自己的内容，看的是导入时落盘的原文全文。',
  },
]

/**
 * 「获取知识库原文」——顶栏一个平级的查阅入口。
 *
 * **不进分析流水线**：这里查到的原文不会写进当前分析，也不影响任何节点。
 * 它和「＋ 导入」的关系就是「查」和「存」的关系。
 *
 * 两个 Tab 是**两个互不相干的知识库**，所以做成 Tab 而不是一个下拉：
 * 它们连检索方式都不同（一个走 MCP 去另一个仓问，一个读本仓的落盘文件），
 * 混在一个结果区里会让人以为在查同一个库。
 */
export function KbTextDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [tab, setTab] = useState<Tab>('novel')
  const current = TABS.find((t) => t.key === tab)!

  return (
    <Modal
      open={open}
      onClose={onClose}
      width="max-w-[860px]"
      title="获取知识库原文"
      subtitle="查阅原文，不写入任何东西，也不参与当前的分析。"
      footer={<Button onClick={onClose}>关闭</Button>}
    >
      <div className="flex flex-col gap-3">
        <div className="border-line flex gap-1 border-b" role="tablist" aria-label="知识库">
          {TABS.map((t) => (
            <button
              key={t.key}
              type="button"
              role="tab"
              aria-selected={tab === t.key}
              onClick={() => setTab(t.key)}
              className={`-mb-px border-b-2 px-3 py-1.5 text-[12.5px] transition-colors ${
                tab === t.key
                  ? 'border-accent text-accent'
                  : 'text-ink-faint hover:text-ink-soft border-transparent'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        <Hint>{current.hint}</Hint>

        {/* 两块面板各自记住自己的状态：切 Tab 不重置，切回来还在原处。
            用 display 而不是条件渲染——一次查询要起子进程，1.7 秒的成本
            不该因为看了一眼另一个 Tab 就被丢掉。 */}
        <div className={tab === 'novel' ? '' : 'hidden'}>
          <NovelPane active={open && tab === 'novel'} />
        </div>
        <div className={tab === 'own' ? '' : 'hidden'}>
          <KbSourceTextPane active={open && tab === 'own'} />
        </div>
      </div>
    </Modal>
  )
}
