/**
 * LoginPage.tsx — 登录 / 注册页面
 *
 * 风格：居中卡片，暖白背景，品牌字体大标题，Tab 切换。
 * 交互：实时表单验证，提交锁定（防重复），错误提示。
 */

import { useState, type FormEvent } from 'react'
import { useAuth } from '../hooks/useAuth'

type Tab = 'login' | 'register'

export default function LoginPage() {
  const { login, register } = useAuth()

  const [tab,      setTab]      = useState<Tab>('login')
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState<string | null>(null)

  // 表单字段
  const [username, setUsername] = useState('')
  const [email,    setEmail]    = useState('')
  const [password, setPassword] = useState('')

  const switchTab = (t: Tab) => {
    setTab(t)
    setError(null)
    setUsername('')
    setEmail('')
    setPassword('')
  }

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      if (tab === 'login') {
        await login({ username: username.trim(), password })
      } else {
        if (password.length < 8) throw new Error('密码至少 8 位')
        await register({ username: username.trim(), email: email.trim(), password })
      }
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })
          .response?.data?.detail ??
        (err as Error).message ??
        '操作失败，请稍后重试'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-canvas flex items-center justify-center px-4">
      {/* 背景装饰 */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-96 h-96 rounded-full bg-accent/5 blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-96 h-96 rounded-full bg-accent/5 blur-3xl" />
      </div>

      <div className="relative w-full max-w-sm">
        {/* 品牌标识 */}
        <div className="text-center mb-8">
          <h1 className="font-serif text-4xl text-ink mb-1">Strata</h1>
          <p className="text-sm text-ink-muted">你的智能求职助手</p>
        </div>

        {/* 登录卡片 */}
        <div className="bg-white rounded-2xl border border-canvas-border shadow-sm p-8">
          {/* Tab */}
          <div className="flex rounded-lg bg-canvas-sidebar p-1 mb-6">
            {(['login', 'register'] as Tab[]).map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => switchTab(t)}
                className={`flex-1 py-1.5 rounded-md text-sm font-medium transition
                  ${tab === t
                    ? 'bg-white text-ink shadow-sm'
                    : 'text-ink-muted hover:text-ink'
                  }`}
              >
                {t === 'login' ? '登录' : '注册'}
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* 用户名 */}
            <div>
              <label className="block text-xs font-medium text-ink-muted mb-1.5">
                {tab === 'login' ? '用户名或邮箱' : '用户名'}
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="input-base"
                placeholder={tab === 'login' ? '用户名 / 邮箱' : '3-50 个字符'}
                required
                autoFocus
                autoComplete={tab === 'login' ? 'username' : 'username'}
              />
            </div>

            {/* 邮箱（仅注册） */}
            {tab === 'register' && (
              <div className="animate-fade-in">
                <label className="block text-xs font-medium text-ink-muted mb-1.5">
                  邮箱
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="input-base"
                  placeholder="your@email.com"
                  required
                  autoComplete="email"
                />
              </div>
            )}

            {/* 密码 */}
            <div>
              <label className="block text-xs font-medium text-ink-muted mb-1.5">
                密码
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="input-base"
                placeholder={tab === 'register' ? '至少 8 位' : '请输入密码'}
                required
                autoComplete={tab === 'login' ? 'current-password' : 'new-password'}
              />
            </div>

            {/* 错误提示 */}
            {error && (
              <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2 animate-fade-in">
                {error}
              </div>
            )}

            {/* 提交 */}
            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full mt-2"
            >
              {loading ? (
                <>
                  <Spinner />
                  {tab === 'login' ? '登录中...' : '注册中...'}
                </>
              ) : (
                tab === 'login' ? '登录' : '创建账号'
              )}
            </button>
          </form>
        </div>

        <p className="text-center text-xs text-ink-faint mt-6">
          Strata v2 · 智能求职助手
        </p>
      </div>
    </div>
  )
}

function Spinner() {
  return (
    <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  )
}
