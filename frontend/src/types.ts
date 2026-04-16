/* ──────────────────────────────────────────────────────────
   types.ts — 全局共享类型，与后端协议完全对齐
   ────────────────────────────────────────────────────────── */

// ── 用户与鉴权 ───────────────────────────────────────────────
export interface User {
  id: number; username: string; email: string;
  is_active: boolean; created_at: string;
}
export interface TokenResponse {
  access_token: string; token_type: string;
  user_id: number; username: string; email: string;
}
export interface RegisterRequest { username: string; email: string; password: string; }
export interface LoginRequest    { username: string; password: string; }

// ── 会话与消息（REST）───────────────────────────────────────
export interface Session {
  id: string; title: string | null;
  created_at: string; updated_at: string;
}
export interface HistoryMessage {
  id: number; role: 'user' | 'assistant';
  content: string; created_at: string;
}

// ── 前端内存消息 ─────────────────────────────────────────────
export interface ChatMessage {
  id: string; role: 'user' | 'assistant'; content: string;
}

// ── 文件上传（Round 5 前端 UI，Round 6 后端实现）────────────
/** 用户已选择、待上传的本地文件 */
export interface FileUploadItem {
  file:    File;
  preview: string;           // 图片时为 ObjectURL，文档时为文件名
  kind:    'image' | 'doc';
}

/** 后端 POST /api/files/upload 返回结构（Round 6 实现） */
export interface FileUploadResponse {
  file_name:    string;
  file_type:    string;
  text_content: string | null;
  image_data:   { base64: string; media_type: string } | null;
  char_count:   number;
}

// ── WebSocket 事件协议 ───────────────────────────────────────
export type WSEventType =
  | 'connected' | 'user_message' | 'token'
  | 'tool_start' | 'tool_end' | 'clarify'
  | 'memory_hits' | 'done' | 'error' | 'pong';

export interface WSEvent { type: WSEventType; data: unknown; }

// ── 前端对话 Reducer 状态 ────────────────────────────────────
export interface ChatState {
  messages:       ChatMessage[];
  streaming:      string;
  isProcessing:   boolean;
  currentTool:    string | null;
  pendingUserMsg: boolean;
  error:          string | null;
}
