/**
 * ChatPage.tsx — 主对话页面
 * ★ Round 5：文件上传流程 + 深色模式全面适配
 *
 * 文件发送流程：
 *   用户选文件 → FilePreview 预览 → 点发送
 *   → uploadFile() POST /api/files/upload（Round 6 后端实现）
 *   → WS 发送 { type:'file', ... }（Round 6 router 处理）
 *   后端未就绪时自动降级为普通文本消息。
 */

import { useCallback, useReducer, useState, useRef, useEffect } from 'react'
import type { ChatMessage, ChatState, WSEvent, FileUploadItem } from '../types'
import * as sessionsApi from '../api/sessions'
import { uploadFile }   from '../api/files'
import { useWebSocket } from '../hooks/useWebSocket'
import Sidebar          from '../components/Sidebar'
import MessageList      from '../components/MessageList'
import MessageInput     from '../components/MessageInput'

const uid = () => crypto.randomUUID()

// ── Reducer ──────────────────────────────────────────────────
type Action =
  | { type: 'SEND_USER';    content: string }
  | { type: 'SERVER_USER';  content: string }
  | { type: 'TOKEN';        data: string }
  | { type: 'TOOL_START';   name: string }
  | { type: 'TOOL_END' }
  | { type: 'CLARIFY';      question: string }
  | { type: 'DONE' }
  | { type: 'ERROR_MSG';    message: string }
  | { type: 'LOAD_HISTORY'; messages: ChatMessage[] }
  | { type: 'CLEAR' }

const init: ChatState = {
  messages: [], streaming: '', isProcessing: false,
  currentTool: null, pendingUserMsg: false, error: null,
}

function reducer(state: ChatState, action: Action): ChatState {
  switch (action.type) {
    case 'SEND_USER':
      return { ...state,
        messages: [...state.messages, { id: uid(), role: 'user', content: action.content }],
        isProcessing: true, pendingUserMsg: true, error: null }
    case 'SERVER_USER':
      if (state.pendingUserMsg) return { ...state, pendingUserMsg: false }
      return { ...state,
        messages: [...state.messages, { id: uid(), role: 'user', content: action.content }],
        isProcessing: true }
    case 'TOKEN':
      return { ...state, streaming: state.streaming + action.data }
    case 'TOOL_START':
      return { ...state, currentTool: action.name }
    case 'TOOL_END':
      return { ...state, currentTool: null }
    case 'CLARIFY':
      return { ...state,
        messages: [...state.messages, { id: uid(), role: 'assistant', content: action.question }],
        isProcessing: false, currentTool: null }
    case 'DONE': {
      const msgs = state.streaming
        ? [...state.messages, { id: uid(), role: 'assistant' as const, content: state.streaming }]
        : state.messages
      return { ...state, messages: msgs, streaming: '', isProcessing: false, currentTool: null, pendingUserMsg: false }
    }
    case 'ERROR_MSG':
      return { ...state,
        messages: state.streaming
          ? [...state.messages, { id: uid(), role: 'assistant', content: state.streaming }]
          : state.messages,
        streaming: '', isProcessing: false, currentTool: null, error: action.message }
    case 'LOAD_HISTORY':
      return { ...init, messages: action.messages }
    case 'CLEAR':
      return { ...init }
    default: return state
  }
}

// ── ChatPage ──────────────────────────────────────────────────
export default function ChatPage() {
  const [state,    dispatch]   = useReducer(reducer, init)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [inputValue, setInputValue] = useState('')
  const [pendingFile, setPendingFile] = useState<FileUploadItem | null>(null)
  const [uploading, setUploading] = useState(false)
  const [sidebarRefresh, setSidebarRefresh] = useState(0)
  const historyLoadedRef = useRef<string | null>(null)

  // WebSocket 事件
  const handleWSEvent = useCallback((event: WSEvent) => {
    switch (event.type) {
      case 'user_message':
        dispatch({ type: 'SERVER_USER', content: (event.data as { content: string }).content }); break
      case 'token':
        dispatch({ type: 'TOKEN', data: event.data as string }); break
      case 'tool_start':
        dispatch({ type: 'TOOL_START', name: (event.data as { name: string }).name }); break
      case 'tool_end':
        dispatch({ type: 'TOOL_END' }); break
      case 'clarify':
        dispatch({ type: 'CLARIFY', question: event.data as string }); break
      case 'done':
        dispatch({ type: 'DONE' }); setSidebarRefresh((n) => n + 1); break
      case 'error':
        dispatch({ type: 'ERROR_MSG', message: event.data as string }); break
    }
  }, [])

  const { connected, send } = useWebSocket({
    sessionId, onEvent: handleWSEvent, enabled: Boolean(sessionId),
  })

  // 加载历史
  const loadHistory = useCallback(async (sid: string) => {
    if (historyLoadedRef.current === sid) return
    historyLoadedRef.current = sid
    try {
      const history = await sessionsApi.getMessages(sid)
      dispatch({ type: 'LOAD_HISTORY', messages: history.map((m) => ({
        id: String(m.id), role: m.role, content: m.content,
      })) })
    } catch { dispatch({ type: 'CLEAR' }) }
  }, [])

  const selectSession = useCallback((sid: string) => {
    if (sid === sessionId) return
    historyLoadedRef.current = null
    dispatch({ type: 'CLEAR' })
    setPendingFile(null); setInputValue('')
    setSessionId(sid); void loadHistory(sid)
  }, [sessionId, loadHistory])

  const handleNewSession = useCallback(async () => {
    try {
      const s = await sessionsApi.createSession()
      historyLoadedRef.current = null
      dispatch({ type: 'CLEAR' })
      setPendingFile(null); setInputValue('')
      setSessionId(s.id); setSidebarRefresh((n) => n + 1)
    } catch (err) { console.error('创建会话失败:', err) }
  }, [])

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { void handleNewSession() }, [])

  // 发送（含文件上传）
  const handleSend = useCallback(async () => {
    if (state.isProcessing || !connected) return
    if (!inputValue.trim() && !pendingFile) return

    if (pendingFile) {
      const label = `[附件: ${pendingFile.file.name}]`
      const content = inputValue.trim() ? `${inputValue.trim()}\n${label}` : label
      dispatch({ type: 'SEND_USER', content })
      const fileToUpload = pendingFile
      setInputValue(''); setPendingFile(null); setUploading(true)
      try {
        const result = await uploadFile(fileToUpload.file)
        send({ type: 'file', file_name: result.file_name, file_type: result.file_type,
               file_content: result.text_content, image_data: result.image_data,
               user_hint: inputValue.trim() })
      } catch {
        // 后端未就绪，降级为普通文本消息
        send({ type: 'message', content })
      } finally { setUploading(false) }
      return
    }

    const content = inputValue.trim()
    setInputValue('')
    dispatch({ type: 'SEND_USER', content })
    send({ type: 'message', content })
  }, [state.isProcessing, connected, inputValue, pendingFile, send])

  return (
    <div className="flex h-screen overflow-hidden bg-canvas dark:bg-night">
      <Sidebar currentSessionId={sessionId} onSelectSession={selectSession}
               onNewSession={() => void handleNewSession()} refreshTrigger={sidebarRefresh} />

      <main className="flex-1 flex flex-col min-w-0">
        {/* 顶部栏 */}
        <header className="flex items-center justify-between px-5 py-3
                           border-b border-canvas-border dark:border-night-border
                           bg-white dark:bg-night">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-ink dark:text-night-text">Strata</span>
            <span className="text-ink-faint dark:text-night-faint text-sm">·</span>
            <span className="text-sm text-ink-muted dark:text-night-muted">求职助手</span>
          </div>
          <div className="flex items-center gap-1.5 text-xs text-ink-faint dark:text-night-faint">
            <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-emerald-400' : 'bg-amber-400 animate-pulse'}`} />
            {connected ? '实时连接' : '连接中...'}
          </div>
        </header>

        {/* 错误提示 */}
        {state.error && (
          <div className="mx-4 mt-2 px-3 py-2 animate-fade-in
                          bg-red-50 dark:bg-red-900/20
                          border border-red-200 dark:border-red-800
                          rounded-lg text-sm text-red-600 dark:text-red-400
                          flex items-center justify-between">
            <span>{state.error}</span>
            <button onClick={() => dispatch({ type: 'CLEAR' })} className="ml-2 font-medium hover:opacity-70">×</button>
          </div>
        )}

        <div className="flex-1 flex flex-col min-h-0">
          <MessageList messages={state.messages} streaming={state.streaming}
                       isProcessing={state.isProcessing} currentTool={state.currentTool} />
        </div>

        <MessageInput value={inputValue} onChange={setInputValue}
                      onSend={() => void handleSend()}
                      pendingFile={pendingFile}
                      onFileSelect={setPendingFile}
                      onFileRemove={() => setPendingFile(null)}
                      disabled={state.isProcessing} connected={connected} uploading={uploading} />
      </main>
    </div>
  )
}
