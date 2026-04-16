"""
auth/router.py — 鉴权路由
===========================
提供以下端点：
  POST /api/auth/register  — 注册新用户，返回 Token
  POST /api/auth/login     — 登录，返回 Token
  GET  /api/auth/me        — 获取当前用户信息（需鉴权）
  POST /api/auth/logout    — 登出（前端清除 Token 即可，此处记录日志）
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.dependencies import get_current_user
from backend.auth.models import User
from backend.auth.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from backend.auth import service
from backend.database import get_db

logger  = logging.getLogger("auth.router")
router  = APIRouter(prefix="/api/auth", tags=["认证"])


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="注册新用户",
    description="注册成功后直接返回 JWT Token，无需二次登录。",
)
async def register(
    req: RegisterRequest,
    db:  AsyncSession = Depends(get_db),
) -> TokenResponse:
    try:
        user = await service.register_user(db, req)
        # 注册成功后直接签发 Token，免去二次登录步骤
        token = service.create_access_token(user.id, user.username, user.email)
        return TokenResponse(
            access_token=token,
            user_id=user.id,
            username=user.username,
            email=user.email,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="用户登录",
    description="支持用户名或邮箱登录，返回 JWT Token。",
)
async def login(
    req: LoginRequest,
    db:  AsyncSession = Depends(get_db),
) -> TokenResponse:
    try:
        return await service.login_user(db, req)
    except ValueError as e:
        # 统一返回 401，不透露具体原因（防枚举攻击）
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="获取当前用户信息",
    description="需要在 Authorization 头携带有效的 Bearer Token。",
)
async def get_me(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    return UserResponse.model_validate(current_user)


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="登出",
    description=(
        "服务端登出记录操作日志。"
        "实际 Token 失效依赖前端清除本地存储，"
        "如需服务端立即失效请联系实现 Token 黑名单功能。"
    ),
)
async def logout(
    current_user: User = Depends(get_current_user),
) -> None:
    logger.info(f"用户登出: id={current_user.id} username={current_user.username!r}")
    # JWT 无状态，服务端无需做任何操作。
    # 前端调用此接口后删除本地 Token 即完成登出。
    # 如需强制失效（如安全事件），可在 Redis 维护 Token 黑名单。
