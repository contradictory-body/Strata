import client from './client'
import type { Session, HistoryMessage } from '../types'

/** 创建新会话 */
export async function createSession(title?: string): Promise<Session> {
  const res = await client.post<Session>('/api/sessions', { title: title ?? null })
  return res.data
}

/** 获取当前用户所有会话（按 updated_at 倒序） */
export async function listSessions(limit = 50): Promise<Session[]> {
  const res = await client.get<Session[]>('/api/sessions', { params: { limit } })
  return res.data
}

/** 获取指定会话的消息历史 */
export async function getMessages(sessionId: string, limit = 100): Promise<HistoryMessage[]> {
  const res = await client.get<HistoryMessage[]>(
    `/api/sessions/${sessionId}/messages`,
    { params: { limit } }
  )
  return res.data
}

/** 删除会话 */
export async function deleteSession(sessionId: string): Promise<void> {
  await client.delete(`/api/sessions/${sessionId}`)
}
