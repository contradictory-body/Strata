/**
 * useWebSocket.ts — WebSocket 生命周期管理
 *
 * ★ Round 6 新增：30s ping 心跳，防止 nginx/防火墙因空闲断开连接
 *
 * 功能：
 *   - 连接到 /ws/{sessionId}?token=<jwt>
 *   - 接收 JSON 事件，调用 onEvent 回调
 *   - 指数退避自动重连（最长 30s）
 *   - 每 30s 发送 ping 心跳，维持连接活跃
 *   - send() 发送 JSON 消息
 *   - session 切换时自动重置连接
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { useAuthStore } from '../store/authStore'
import type { WSEvent } from '../types'

interface UseWebSocketOptions {
  sessionId: string | null
  onEvent:   (event: WSEvent) => void
  enabled?:  boolean
}

interface UseWebSocketReturn {
  connected: boolean
  send:      (data: object) => void
}

const MAX_RECONNECT_DELAY  = 30_000
const BASE_RECONNECT_DELAY = 1_000
const PING_INTERVAL_MS     = 30_000   // 每 30s 发送一次 ping

function buildWsUrl(sessionId: string, token: string): string {
  const base = import.meta.env.VITE_WS_BASE_URL as string | undefined
  if (base) return `${base}/ws/${sessionId}?token=${token}`
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}/ws/${sessionId}?token=${token}`
}

export function useWebSocket({
  sessionId,
  onEvent,
  enabled = true,
}: UseWebSocketOptions): UseWebSocketReturn {
  const token = useAuthStore((s) => s.token)

  const wsRef             = useRef<WebSocket | null>(null)
  const onEventRef        = useRef(onEvent)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout>>()
  const pingTimerRef      = useRef<ReturnType<typeof setInterval>>()
  const reconnectAttempts = useRef(0)
  const unmountedRef      = useRef(false)

  const [connected, setConnected] = useState(false)

  // 始终保持 onEvent 引用最新，避免 stale closure
  onEventRef.current = onEvent

  const clearPing = useCallback(() => {
    if (pingTimerRef.current) {
      clearInterval(pingTimerRef.current)
      pingTimerRef.current = undefined
    }
  }, [])

  const connect = useCallback(() => {
    if (!enabled || !sessionId || !token || unmountedRef.current) return

    // 关闭旧连接
    if (wsRef.current) {
      wsRef.current.onclose = null   // 阻止触发重连
      wsRef.current.close()
    }
    clearPing()

    const url = buildWsUrl(sessionId, token)
    const ws  = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      if (unmountedRef.current) { ws.close(); return }
      setConnected(true)
      reconnectAttempts.current = 0

      // 启动心跳：每 30s 发一次 ping，防止 nginx/防火墙因空闲断开
      pingTimerRef.current = setInterval(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: 'ping' }))
        }
      }, PING_INTERVAL_MS)
    }

    ws.onmessage = (e: MessageEvent) => {
      try {
        const event = JSON.parse(e.data as string) as WSEvent
        onEventRef.current(event)
      } catch (err) {
        console.error('[WS] 消息解析失败:', err)
      }
    }

    ws.onclose = () => {
      setConnected(false)
      clearPing()
      if (unmountedRef.current) return
      // 指数退避重连
      const delay = Math.min(
        BASE_RECONNECT_DELAY * 2 ** reconnectAttempts.current,
        MAX_RECONNECT_DELAY
      )
      reconnectAttempts.current += 1
      reconnectTimerRef.current = setTimeout(connect, delay)
    }

    ws.onerror = () => {
      ws.close()   // 触发 onclose → 重连
    }
  }, [enabled, sessionId, token, clearPing])

  useEffect(() => {
    unmountedRef.current = false
    connect()
    return () => {
      unmountedRef.current = true
      clearTimeout(reconnectTimerRef.current)
      clearPing()
      if (wsRef.current) {
        wsRef.current.onclose = null
        wsRef.current.close()
        wsRef.current = null
      }
      setConnected(false)
    }
  }, [connect, clearPing])

  const send = useCallback((data: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    } else {
      console.warn('[WS] 连接未就绪，消息丢弃')
    }
  }, [])

  return { connected, send }
}
