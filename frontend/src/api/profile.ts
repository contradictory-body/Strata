/**
 * profile.ts — 用户求职画像 API
 *
 * 对应后端 GET/PUT /api/profile（Round 6 实现）。
 *
 * 画像字段（与后端 PROFILE_SECTIONS 完全一致）：
 *   目标岗位 / 目标城市 / 技术栈 / 目标公司_行业 /
 *   薪资预期 / 面试薄弱点 / 简历修改偏好
 */

import client from './client'

// ── 类型定义 ─────────────────────────────────────────────────
export interface ProfileFields {
  目标岗位?:      string
  目标城市?:      string
  技术栈?:        string
  目标公司_行业?: string
  薪资预期?:      string
  面试薄弱点?:    string
  简历修改偏好?:  string
}

export interface ProfileResponse {
  raw:    string           // PROFILE.md 原文
  fields: ProfileFields    // 解析后的字段字典（已过滤"暂未填写"）
}

export interface ProfileSummaryResponse {
  summary: string          // 注入 system prompt 的摘要字符串
}

// ── API 函数 ──────────────────────────────────────────────────

/** 获取当前用户的完整求职画像 */
export async function getProfile(): Promise<ProfileResponse> {
  const res = await client.get<ProfileResponse>('/api/profile')
  return res.data
}

/** 批量更新画像字段（只传需要更新的字段，其余保持不变） */
export async function updateProfile(
  updates: ProfileFields
): Promise<{ updated_fields: string[] }> {
  const res = await client.put<{ updated_fields: string[] }>(
    '/api/profile',
    { updates }
  )
  return res.data
}

/** 获取画像摘要（用于注入 Agent system prompt 的简洁格式） */
export async function getProfileSummary(): Promise<ProfileSummaryResponse> {
  const res = await client.get<ProfileSummaryResponse>('/api/profile/summary')
  return res.data
}

/** 所有合法的画像字段名 */
export const PROFILE_FIELD_NAMES = [
  '目标岗位',
  '目标城市',
  '技术栈',
  '目标公司_行业',
  '薪资预期',
  '面试薄弱点',
  '简历修改偏好',
] as const

export type ProfileFieldName = typeof PROFILE_FIELD_NAMES[number]
