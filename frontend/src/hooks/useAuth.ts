/**
 * useAuth.ts — 鉴权 Hook
 *
 * 提供 isAuthenticated、user 快捷访问，
 * 以及 login / register / logout 操作（含状态更新）。
 */

import { useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'
import * as authApi from '../api/auth'
import type { LoginRequest, RegisterRequest } from '../types'

export function useAuth() {
  const { token, user, setAuth, clearAuth } = useAuthStore()
  const navigate = useNavigate()

  const isAuthenticated = Boolean(token && user)

  const handleLogin = useCallback(async (data: LoginRequest) => {
    const res = await authApi.login(data)
    setAuth(res.access_token, {
      id:         res.user_id,
      username:   res.username,
      email:      res.email,
      is_active:  true,
      created_at: new Date().toISOString(),
    })
    navigate('/', { replace: true })
  }, [setAuth, navigate])

  const handleRegister = useCallback(async (data: RegisterRequest) => {
    const res = await authApi.register(data)
    setAuth(res.access_token, {
      id:         res.user_id,
      username:   res.username,
      email:      res.email,
      is_active:  true,
      created_at: new Date().toISOString(),
    })
    navigate('/', { replace: true })
  }, [setAuth, navigate])

  const handleLogout = useCallback(async () => {
    await authApi.logout()
    clearAuth()
    navigate('/login', { replace: true })
  }, [clearAuth, navigate])

  return {
    isAuthenticated,
    user,
    token,
    login:    handleLogin,
    register: handleRegister,
    logout:   handleLogout,
  }
}
