"""
chat/models.py — 会话与消息 ORM 模型
======================================
定义两张表：

chat_sessions  — 会话元信息（id / user_id / title / 时间戳）
chat_messages  — 每轮对话消息持久化（role = user | assistant）

设计说明：
  - session_id 使用 UUID4 字符串（36 字符），服务端生成，前端携带建立 WebSocket。
  - 对话运行时状态（messages list / pending_clarification 等）
    存储在 Redis（见 session_state.py），不存入 PostgreSQL。
    PostgreSQL 只保存最终的 user / assistant 消息，用于历史记录展示。
  - 删除 session 时级联删除所有 messages（ON DELETE CASCADE）。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="会话 UUID",
    )
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="所属用户 ID",
    )
    title: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="会话标题，由第一条用户消息自动截取",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="最后一条消息的时间，用于会话列表排序",
    )

    def __repr__(self) -> str:
        return f"<ChatSession id={self.id!r} user_id={self.user_id}>"


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="所属会话 UUID",
    )
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="消息角色：user | assistant",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="消息内容（Markdown 格式）",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<ChatMessage id={self.id} session={self.session_id!r} role={self.role!r}>"
