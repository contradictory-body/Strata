/**
 * Sidebar.tsx — 会话列表侧边栏
 * ★ Round 5：🌙/☀️ 主题切换 + 深色模式全面适配
 */

import { useState, useEffect, useCallback } from 'react'
import type { Session } from '../types'
import * as sessionsApi from '../api/sessions'
import { useAuth }      from '../hooks/useAuth'
import { useAuthStore } from '../store/authStore'

interface SidebarProps {
  currentSessionId: string | null
  onSelectSession:  (id: string) => void
  onNewSession:     () => void
  refreshTrigger?:  number
}

export default function Sidebar({
  currentSessionId, onSelectSession, onNewSession, refreshTrigger = 0,
}: SidebarProps) {
  const { user, logout }        = useAuth()
  const { theme, toggleTheme }  = useAuthStore()

  const [sessions,  setSessions]  = useState<Session[]>([])
  const [loading,   setLoading]   = useState(true)
  const [hoveredId, setHoveredId] = useState<string | null>(null)

  const loadSessions = useCallback(async () => {
    try { setSessions(await sessionsApi.listSessions()) }
    catch { /* 静默失败 */ }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { void loadSessions() }, [loadSessions, refreshTrigger])

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    if (!window.confirm('确认删除该会话？')) return
    await sessionsApi.deleteSession(id)
    setSessions((prev) => prev.filter((s) => s.id !== id))
    if (currentSessionId === id) onNewSession()
  }

  return (
    <aside className="w-64 flex-shrink-0 flex flex-col h-full
                      bg-canvas-sidebar dark:bg-night-sidebar
                      border-r border-canvas-border dark:border-night-border">

      {/* 顶部 */}
      <div className="px-4 pt-5 pb-3 border-b border-canvas-border dark:border-night-border">
        <div className="flex items-center justify-between mb-3">
          <h1 className="font-serif text-xl text-ink dark:text-night-text">Strata</h1>
          {/* 主题切换 */}
          <button onClick={toggleTheme}
                  title={theme === 'light' ? '切换深色模式' : '切换浅色模式'}
                  aria-label="切换主题"
                  className="w-7 h-7 rounded-lg flex items-center justify-center
                             text-ink-muted dark:text-night-muted
                             hover:bg-canvas-border dark:hover:bg-night-hover
                             hover:text-ink dark:hover:text-night-text transition">
            {theme === 'light' ? <MoonIcon /> : <SunIcon />}
          </button>
        </div>
        <button onClick={onNewSession}
                className="flex items-center gap-2 w-full px-3 py-2 rounded-lg
                           text-sm text-ink-muted dark:text-night-muted
                           border border-canvas-border dark:border-night-border
                           bg-white dark:bg-night-panel
                           hover:border-accent/30 dark:hover:border-night-accent/30
                           hover:text-ink dark:hover:text-night-text transition">
          <PlusIcon /><span>新对话</span>
        </button>
      </div>

      {/* 会话列表 */}
      <div className="flex-1 overflow-y-auto py-2 px-2 space-y-0.5">
        {loading ? (
          <p className="text-center text-xs text-ink-faint dark:text-night-faint py-6">加载中...</p>
        ) : sessions.length === 0 ? (
          <p className="text-center text-xs text-ink-faint dark:text-night-faint py-6">还没有对话记录</p>
        ) : sessions.map((session) => (
          <SessionItem key={session.id} session={session}
            isActive={session.id === currentSessionId}
            isHovered={hoveredId === session.id}
            onSelect={() => onSelectSession(session.id)}
            onDelete={(e) => handleDelete(e, session.id)}
            onMouseEnter={() => setHoveredId(session.id)}
            onMouseLeave={() => setHoveredId(null)} />
        ))}
      </div>

      {/* 底部用户信息 */}
      <div className="border-t border-canvas-border dark:border-night-border p-3">
        <div className="flex items-center justify-between px-2 py-1.5">
          <div className="flex items-center gap-2 min-w-0">
            <div className="w-6 h-6 rounded-full flex-shrink-0 flex items-center justify-center
                            bg-accent/15 dark:bg-night-accent/15">
              <span className="text-accent dark:text-night-accent text-xs font-semibold">
                {user?.username?.[0]?.toUpperCase() ?? '?'}
              </span>
            </div>
            <span className="text-sm text-ink dark:text-night-text truncate">{user?.username}</span>
          </div>
          <button onClick={() => void logout()} title="退出登录"
                  className="text-ink-faint dark:text-night-faint
                             hover:text-accent dark:hover:text-night-accent
                             transition p-1 rounded">
            <LogoutIcon />
          </button>
        </div>
      </div>
    </aside>
  )
}

// ── 单条会话项 ──────────────────────────────────────────────
function SessionItem({ session, isActive, isHovered, onSelect, onDelete, onMouseEnter, onMouseLeave }: {
  session: Session; isActive: boolean; isHovered: boolean
  onSelect: () => void; onDelete: (e: React.MouseEvent) => void
  onMouseEnter: () => void; onMouseLeave: () => void
}) {
  return (
    <button onClick={onSelect} onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}
            className={`w-full text-left px-3 py-2 rounded-lg flex items-start
                        justify-between gap-2 transition
                        ${isActive
                          ? 'bg-white dark:bg-night-panel shadow-sm border border-canvas-border dark:border-night-border text-ink dark:text-night-text'
                          : 'text-ink-muted dark:text-night-muted hover:bg-white/60 dark:hover:bg-night-hover hover:text-ink dark:hover:text-night-text'
                        }`}>
      <div className="min-w-0 flex-1">
        <p className="text-sm truncate font-medium leading-tight">
          {session.title ?? '新对话'}
        </p>
        <p className="text-xs text-ink-faint dark:text-night-faint mt-0.5">
          {formatDate(session.updated_at)}
        </p>
      </div>
      {(isHovered || isActive) && (
        <button onClick={onDelete} aria-label="删除"
                className="flex-shrink-0 p-0.5 rounded
                           text-ink-faint dark:text-night-faint
                           hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 transition">
          <TrashIcon />
        </button>
      )}
    </button>
  )
}

function formatDate(iso: string): string {
  const d = new Date(iso), now = new Date()
  const diff = Math.floor((now.getTime() - d.getTime()) / 86_400_000)
  if (diff === 0) return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  if (diff === 1) return '昨天'
  if (diff < 7)   return `${diff} 天前`
  return d.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })
}

// ── 图标 ────────────────────────────────────────────────────
function PlusIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor"
         strokeWidth="1.8" strokeLinecap="round" className="w-4 h-4">
      <path d="M8 3v10M3 8h10" />
    </svg>
  )
}
function TrashIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor"
         strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="w-3.5 h-3.5">
      <path d="M2 4h12M5 4V3a1 1 0 011-1h4a1 1 0 011 1v1M6 7v5M10 7v5M3 4l1 9a1 1 0 001 1h6a1 1 0 001-1l1-9" />
    </svg>
  )
}
function LogoutIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor"
         strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
      <path d="M6 14H3a1 1 0 01-1-1V3a1 1 0 011-1h3M10 11l3-3-3-3M13 8H6" />
    </svg>
  )
}
function MoonIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="currentColor" className="w-4 h-4">
      <path d="M6 .278a.768.768 0 01.08.858 7.208 7.208 0 00-.878 3.46c0 4.021 3.278 7.277
               7.318 7.277.527 0 1.04-.055 1.533-.16a.787.787 0 01.81.316.733.733 0
               01-.031.893A8.349 8.349 0 018.344 16C3.734 16 0 12.286 0 7.71 0 4.266
               2.114 1.312 5.124.06A.752.752 0 016 .278z" />
    </svg>
  )
}
function SunIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="currentColor" className="w-4 h-4">
      <path d="M8 11a3 3 0 110-6 3 3 0 010 6zm0 1a4 4 0 100-8 4 4 0 000
               8zM8 0a.5.5 0 01.5.5v2a.5.5 0 01-1 0v-2A.5.5 0 018 0zm0 13a.5.5 0
               01.5.5v2a.5.5 0 01-1 0v-2A.5.5 0 018 13zm8-5a.5.5 0
               01-.5.5h-2a.5.5 0 010-1h2a.5.5 0 01.5.5zM3 8a.5.5 0
               01-.5.5h-2a.5.5 0 010-1h2A.5.5 0 013 8zm10.657-5.657a.5.5 0
               010 .707l-1.414 1.415a.5.5 0 11-.707-.708l1.414-1.414a.5.5 0
               01.707 0zm-9.193 9.193a.5.5 0 010 .707L3.05 13.657a.5.5 0
               01-.707-.707l1.414-1.414a.5.5 0 01.707 0zm9.193 2.121a.5.5 0
               01-.707 0l-1.414-1.414a.5.5 0 01.707-.707l1.414 1.414a.5.5 0
               010 .707zM4.464 4.465a.5.5 0 01-.707 0L2.343 3.05a.5.5 0
               11.707-.707l1.414 1.414a.5.5 0 010 .707z" />
    </svg>
  )
}
