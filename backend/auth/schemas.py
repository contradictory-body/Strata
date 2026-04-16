"""
auth/schemas.py — 鉴权相关 Pydantic 模型
==========================================
定义注册、登录、Token 响应、用户信息等数据结构。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, field_validator


# ── 请求体 ─────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    """注册请求体。"""
    username: str
    email:    EmailStr
    password: str

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("用户名至少 3 个字符")
        if len(v) > 50:
            raise ValueError("用户名最多 50 个字符")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("密码至少 8 位")
        return v


class LoginRequest(BaseModel):
    """
    登录请求体。
    username 字段同时接受用户名和邮箱，service 层处理两种情况。
    """
    username: str   # 用户名 或 邮箱
    password: str


# ── 响应体 ─────────────────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    """登录/注册成功后返回的 Token 响应。"""
    access_token: str
    token_type:   str = "bearer"
    user_id:      int
    username:     str
    email:        str


class UserResponse(BaseModel):
    """用户信息响应（/api/auth/me 接口使用）。"""
    id:         int
    username:   str
    email:      str
    is_active:  bool
    created_at: datetime

    model_config = {"from_attributes": True}   # 允许从 ORM 对象直接构建


class ErrorResponse(BaseModel):
    """统一错误响应体。"""
    detail: str
