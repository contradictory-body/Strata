/**
 * MessageInput.tsx — 消息输入框
 * ★ Round 5：📎 文件上传按钮 + FilePreview 附件卡片 + 深色模式
 */

import {
  type FormEvent, type KeyboardEvent,
  useRef, useEffect, useCallback,
} from 'react'
import type { FileUploadItem } from '../types'
import FilePreview from './FilePreview'
import { ACCEPTED_TYPES, MAX_FILE_SIZE, isImageFile } from '../api/files'

interface Props {
  value:        string
  onChange:     (v: string) => void
  onSend:       () => void
  pendingFile:  FileUploadItem | null
  onFileSelect: (item: FileUploadItem) => void
  onFileRemove: () => void
  disabled?:    boolean
  connected?:   boolean
  uploading?:   boolean
}

export default function MessageInput({
  value, onChange, onSend,
  pendingFile, onFileSelect, onFileRemove,
  disabled = false, connected = false, uploading = false,
}: Props) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // 自适应高度
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [value])

  const canSend = !disabled && !uploading && connected && (value.trim().length > 0 || Boolean(pendingFile))

  const handleKey = useCallback((e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (canSend) onSend()
    }
  }, [canSend, onSend])

  const handleSubmit = (e: FormEvent) => { e.preventDefault(); if (canSend) onSend() }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    e.target.value = ''
    if (file.size > MAX_FILE_SIZE) { alert('文件过大，最大支持 20MB'); return }
    const kind: 'image' | 'doc' = isImageFile(file) ? 'image' : 'doc'
    const preview = kind === 'image' ? URL.createObjectURL(file) : file.name
    onFileSelect({ file, preview, kind })
  }

  return (
    <div className="border-t border-canvas-border dark:border-night-border
                    bg-white dark:bg-night">
      {/* 附件预览 */}
      {pendingFile && (
        <div className="px-4 pt-3">
          <FilePreview item={pendingFile} onRemove={onFileRemove} />
        </div>
      )}

      <form onSubmit={handleSubmit} className="px-4 py-3">
        <div className={`flex items-end gap-2 rounded-xl border transition
          ${connected
            ? 'border-canvas-border dark:border-night-border focus-within:border-accent/50 dark:focus-within:border-night-accent/50 focus-within:ring-2 focus-within:ring-accent/10 dark:focus-within:ring-night-accent/10'
            : 'border-canvas-border dark:border-night-border opacity-60'
          } bg-canvas dark:bg-night-panel px-3 py-2`}>

          {/* 📎 文件按钮 */}
          <button type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={!connected || disabled}
                  title="上传文件（PDF / Word / 图片）"
                  aria-label="上传文件"
                  className="flex-shrink-0 w-7 h-7 mb-0.5 rounded-lg
                             flex items-center justify-center
                             text-ink-faint dark:text-night-faint
                             hover:text-ink dark:hover:text-night-text
                             hover:bg-canvas-border dark:hover:bg-night-hover
                             disabled:opacity-40 disabled:cursor-not-allowed transition">
            <PaperclipIcon />
          </button>
          <input ref={fileInputRef} type="file"
                 accept={ACCEPTED_TYPES} onChange={handleFileChange} className="hidden" />

          {/* 文本域 */}
          <textarea ref={textareaRef} value={value}
                    onChange={(e) => onChange(e.target.value)}
                    onKeyDown={handleKey}
                    placeholder={
                      !connected  ? '正在连接...' :
                      uploading   ? '文件上传中...' :
                      disabled    ? '助手正在回复...' :
                      '输入消息，Enter 发送，Shift+Enter 换行'
                    }
                    disabled={!connected || uploading}
                    rows={1}
                    className="flex-1 resize-none bg-transparent text-sm
                               text-ink dark:text-night-text
                               placeholder:text-ink-faint dark:placeholder:text-night-faint
                               outline-none leading-relaxed disabled:cursor-not-allowed"
                    style={{ maxHeight: 160 }} />

          {/* 发送按钮 */}
          <button type="submit" disabled={!canSend} aria-label="发送"
                  className={`flex-shrink-0 w-8 h-8 mb-0.5 rounded-lg
                              flex items-center justify-center transition
                              ${canSend
                                ? 'bg-accent dark:bg-night-accent text-white hover:opacity-90'
                                : 'bg-canvas-border dark:bg-night-border text-ink-faint dark:text-night-faint cursor-not-allowed'
                              }`}>
            {uploading ? <SpinIcon /> : disabled ? <StopIcon /> : <SendIcon />}
          </button>
        </div>

        {/* 状态栏 */}
        <div className="flex items-center justify-between mt-1.5 px-1">
          <span className="flex items-center gap-1.5 text-xs text-ink-faint dark:text-night-faint">
            <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-emerald-400' : 'bg-amber-400 animate-pulse'}`} />
            {connected ? '已连接' : '连接中...'}
          </span>
          <span className="text-xs text-ink-faint dark:text-night-faint">
            Enter 发送 · Shift+Enter 换行
          </span>
        </div>
      </form>
    </div>
  )
}

function PaperclipIcon() {
  return (
    <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.7" className="w-4 h-4">
      <path d="M17.293 8.293l-7.586 7.586a4 4 0 01-5.656-5.656l8.485-8.485a2.5 2.5 0
               013.536 3.536L7.586 13.76a1 1 0 01-1.415-1.415l7.072-7.071"
            strokeLinecap="round" />
    </svg>
  )
}
function SendIcon() {
  return (
    <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
      <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1
               0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0
               001.17-1.408l-7-14z" />
    </svg>
  )
}
function StopIcon() {
  return <svg viewBox="0 0 20 20" fill="currentColor" className="w-3.5 h-3.5"><rect x="4" y="4" width="12" height="12" rx="2" /></svg>
}
function SpinIcon() {
  return (
    <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  )
}
