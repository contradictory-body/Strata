"""
auth/models.py — 用户 ORM 模型
================================
定义 users 表结构。

表字段：
  id               — 自增主键
  username         — 用户名（唯一，3-50 字符，建索引）
  email            — 邮箱（唯一，建索引）
  hashed_password  — bcrypt 哈希密码（绝不存明文）
  is_active        — 账号是否启用（默认 True，禁用时登录拒绝）
  created_at       — 注册时间（数据库服务端自动填充）
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    username: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True,
        comment="用户名，唯一",
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True,
        comment="邮箱，唯一",
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255), nullable=False,
        comment="bcrypt 哈希密码",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
        comment="账号是否启用",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="注册时间",
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r}>"
