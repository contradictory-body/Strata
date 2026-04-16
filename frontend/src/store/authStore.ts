/**
 * authStore.ts — 全局状态（Zustand + localStorage 持久化）
 * ★ Round 5：新增 theme 字段和 toggleTheme / setTheme action
 */

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { User } from '../types'

export type Theme = 'light' | 'dark'

interface AuthState {
  // 鉴权
  token:     string | null
  user:      User   | null
  setAuth:   (token: string, user: User) => void
  clearAuth: () => void
  // 主题
  theme:       Theme
  toggleTheme: () => void
  setTheme:    (t: Theme) => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      token:     null,
      user:      null,
      setAuth:   (token, user) => set({ token, user }),
      clearAuth: ()            => set({ token: null, user: null }),

      theme:       'light',
      toggleTheme: () =>
        set((s) => ({ theme: s.theme === 'light' ? 'dark' : 'light' })),
      setTheme: (t) => set({ theme: t }),
    }),
    {
      name:       'strata-auth',
      partialize: (s) => ({ token: s.token, user: s.user, theme: s.theme }),
    }
  )
)
