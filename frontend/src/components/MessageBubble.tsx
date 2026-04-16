/**
 * MessageBubble.tsx — 单条消息气泡
 *
 * ★ Round 5：
 *   - 助手消息用 ReactMarkdown 渲染（GFM 表格/任务列表/代码块）
 *   - 代码块显示语言标签 + 一键复制按钮
 *   - StreamingBubble 含 .streaming-cursor 闪烁光标
 *   - 深色模式全面适配
 */

import { useState, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm     from 'remark-gfm'
import type { ChatMessage } from '../types'

// ─── 静态消息气泡 ──────────────────────────────────────────
export default function MessageBubble({ message }: { message: ChatMessage }) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end px-4 animate-slide-up">
        <div className="max-w-[72%] bg-ink dark:bg-night-panel
                        text-white rounded-2xl rounded-tr-sm px-4 py-2.5
                        border border-transparent dark:border-night-border">
          <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">
            {message.content}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex gap-3 px-4 animate-slide-up">
      <Avatar />
      <div className="max-w-[78%] pt-0.5 min-w-0">
        <MarkdownContent content={message.content} />
      </div>
    </div>
  )
}

// ─── 流式输出气泡（含闪烁光标）────────────────────────────
export function StreamingBubble({ content }: { content: string }) {
  return (
    <div className="flex gap-3 px-4 animate-fade-in">
      <Avatar pulse />
      <div className="max-w-[78%] pt-0.5 min-w-0">
        {content ? (
          /* 流式内容：Markdown 渲染 + 尾部光标 */
          <div className="prose-chat text-ink dark:text-night-text">
            <MarkdownContent content={content} />
            <span className="streaming-cursor" aria-hidden="true" />
          </div>
        ) : (
          /* 等待第一个 token：三点跳动 */
          <ThinkingDots />
        )}
      </div>
    </div>
  )
}

// ─── Markdown 渲染器 ───────────────────────────────────────
function MarkdownContent({ content }: { content: string }) {
  return (
    <ReactMarkdown
      className="prose-chat text-ink dark:text-night-text"
      remarkPlugins={[remarkGfm]}
      components={{
        code: CodeBlock,
        table: ({ children }) => (
          <div className="overflow-x-auto my-3">
            <table className="min-w-full">{children}</table>
          </div>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  )
}

// ─── 代码块组件 ────────────────────────────────────────────
function CodeBlock(props: {
  inline?: boolean
  className?: string
  children?: ReactNode
  [key: string]: unknown
}) {
  const { inline, className, children } = props
  const [copied, setCopied] = useState(false)

  const lang = /language-(\w+)/.exec(className ?? '')?.[1] ?? ''
  const code = String(children).replace(/\n$/, '')

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (inline) {
    return (
      <code className="px-1.5 py-0.5 rounded text-[0.85em] font-mono
                       bg-zinc-100 dark:bg-night-panel
                       text-accent dark:text-night-accent
                       border border-zinc-200 dark:border-night-border">
        {children}
      </code>
    )
  }

  return (
    <div className="my-3 rounded-xl overflow-hidden
                    border border-canvas-border dark:border-night-border">
      {/* 语言标签 + 复制按钮 */}
      <div className="code-lang-bar">
        <span>{lang || 'code'}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 transition
                     hover:text-ink dark:hover:text-night-text"
        >
          {copied ? <><CheckIcon /> 已复制</> : <><CopyIcon /> 复制</>}
        </button>
      </div>
      {/* 代码内容 */}
      <pre style={{ margin: 0, border: 'none', borderRadius: 0 }}>
        <code className={className}>{code}</code>
      </pre>
    </div>
  )
}

// ─── 思考中三点动画 ────────────────────────────────────────
function ThinkingDots() {
  return (
    <div className="flex items-center gap-1 h-5 pt-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="inline-block w-1.5 h-1.5 rounded-full
                     bg-ink-faint dark:bg-night-muted"
          style={{ animation: `pulse-dot 1.4s ${i * 0.2}s ease-in-out infinite` }}
        />
      ))}
    </div>
  )
}

// ─── 助手头像 ──────────────────────────────────────────────
function Avatar({ pulse }: { pulse?: boolean }) {
  return (
    <div className={`w-7 h-7 rounded-full flex-shrink-0 mt-0.5 flex items-center justify-center
                     bg-accent/10 dark:bg-night-accent/15
                     border border-accent/20 dark:border-night-accent/30
                     ${pulse ? 'animate-pulse' : ''}`}>
      <span className="text-accent dark:text-night-accent text-xs font-bold font-serif">S</span>
    </div>
  )
}

// ─── 图标 ──────────────────────────────────────────────────
function CopyIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor"
         strokeWidth="1.5" className="w-3.5 h-3.5">
      <rect x="5" y="5" width="9" height="9" rx="1.5" />
      <path d="M3 11H2.5A1.5 1.5 0 011 9.5v-7A1.5 1.5 0 012.5 1h7A1.5 1.5 0 0111 2.5V3"
            strokeLinecap="round" />
    </svg>
  )
}
function CheckIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor"
         strokeWidth="2" className="w-3.5 h-3.5">
      <path d="M2.5 8l4 4 7-7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}
