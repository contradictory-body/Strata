/**
 * MessageList.tsx — 可滚动消息列表
 * ★ Round 5：深色模式 + 空态视觉优化
 */

import { useEffect, useRef } from 'react'
import type { ChatMessage } from '../types'
import MessageBubble, { StreamingBubble } from './MessageBubble'
import ToolProgress from './ToolProgress'

interface Props {
  messages:     ChatMessage[]
  streaming:    string
  isProcessing: boolean
  currentTool:  string | null
}

const SUGGESTIONS = [
  { icon: '🔍', text: '帮我找北京后端工程师岗位，薪资 30K 以上' },
  { icon: '📋', text: '分析这份 JD，告诉我匹配度和准备重点'     },
  { icon: '🎤', text: '帮我准备字节跳动的系统设计面试'           },
  { icon: '📄', text: '优化我的简历，突出 Python 和机器学习经验' },
]

export default function MessageList({ messages, streaming, isProcessing, currentTool }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, streaming])

  if (messages.length === 0 && !isProcessing) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center
                      px-8 py-12 text-center select-none">
        <div className="w-14 h-14 rounded-2xl mb-5 flex items-center justify-center
                        bg-accent/10 dark:bg-night-accent/15
                        border border-accent/20 dark:border-night-accent/30">
          <span className="font-serif text-2xl text-accent dark:text-night-accent">S</span>
        </div>
        <h2 className="font-serif text-2xl text-ink dark:text-night-text mb-1.5">
          你好，我是 Strata
        </h2>
        <p className="text-sm text-ink-muted dark:text-night-muted mb-8 max-w-xs">
          你的智能求职助手。告诉我你的目标，我来帮你制定最优方案。
        </p>
        <div className="grid grid-cols-1 gap-2 w-full max-w-sm">
          {SUGGESTIONS.map((s) => (
            <div key={s.text}
                 className="text-left text-sm text-ink-muted dark:text-night-muted
                            border border-canvas-border dark:border-night-border
                            rounded-xl px-4 py-2.5
                            bg-white dark:bg-night-panel
                            hover:border-accent/40 dark:hover:border-night-accent/40
                            hover:bg-accent/5 dark:hover:bg-night-accent/10
                            transition cursor-default">
              <span className="mr-2">{s.icon}</span>{s.text}
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto py-4 space-y-4">
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
      {/* 工具执行优先展示 */}
      {currentTool && <ToolProgress toolName={currentTool} />}
      {/* 工具完成后展示流式 token */}
      {isProcessing && !currentTool && <StreamingBubble content={streaming} />}
      <div ref={bottomRef} className="h-2" />
    </div>
  )
}
