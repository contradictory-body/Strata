import client from './client'
import type { LoginRequest, RegisterRequest, TokenResponse, User } from '../types'

/** 注册，成功后直接返回 Token */
export async function register(data: RegisterRequest): Promise<TokenResponse> {
  const res = await client.post<TokenResponse>('/api/auth/register', data)
  return res.data
}

/** 登录（支持用户名或邮箱） */
export async function login(data: LoginRequest): Promise<TokenResponse> {
  const res = await client.post<TokenResponse>('/api/auth/login', data)
  return res.data
}

/** 获取当前用户信息 */
export async function getMe(): Promise<User> {
  const res = await client.get<User>('/api/auth/me')
  return res.data
}

/** 登出（通知服务端记录日志，前端清除 Token） */
export async function logout(): Promise<void> {
  await client.post('/api/auth/logout').catch(() => {}) // 忽略失败
}
