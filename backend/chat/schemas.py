"""
chat/schemas.py — 对话模块 Pydantic 模型
==========================================
REST 接口的请求体与响应体定义。

WebSocket 的事件协议不在此处定义（直接使用 dict），详见 agent_adapter.py。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


# ── 会话 ───────────────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    """创建新会话请求体。title 可选，不传时由首条消息自动生成。"""
    title: str | None = None


class SessionResponse(BaseModel):
    """会话摘要，用于侧边栏列表展示。"""
    id:         str
    title:      str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── 消息 ───────────────────────────────────────────────────────────────────────

class MessageResponse(BaseModel):
    """单条消息响应，用于历史记录接口。"""
    id:         int
    role:       str          # "user" | "assistant"
    content:    str
    created_at: datetime

    model_config = {"from_attributes": True}
