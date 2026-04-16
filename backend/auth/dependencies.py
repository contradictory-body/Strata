"""
auth/dependencies.py — FastAPI 依赖注入：当前用户鉴权
=========================================================
提供 get_current_user 依赖函数，在需要鉴权的路由中注入当前登录用户。

用法：
    from backend.auth.dependencies import get_current_user
    from backend.auth.models import User

    @router.get("/protected")
    async def protected(current_user: User = Depends(get_current_user)):
        return {"user_id": current_user.id}
"""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.models import User
from backend.auth.service import decode_token, get_user_by_id
from backend.database import get_db

logger = logging.getLogger("auth.dependencies")

# Bearer Token 提取器（从 Authorization: Bearer <token> 中提取 token）
_bearer_scheme = HTTPBearer(
    scheme_name="Bearer Token",
    description="在 Authorization 请求头中携带 JWT Token",
    auto_error=True,               # Token 缺失时自动返回 401
)

# 统一的鉴权失败异常
_CREDENTIALS_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Token 无效或已过期，请重新登录",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    从请求头提取并验证 JWT Token，返回当前登录用户的 ORM 对象。

    验证流程：
      1. 从 Authorization: Bearer <token> 提取 token 字符串
      2. 解码 JWT，提取 sub（user_id）
      3. 从数据库查询用户，检查 is_active
      4. 任一步骤失败均返回 401

    用于需要鉴权的所有路由，通过 Depends(get_current_user) 注入。
    """
    try:
        payload  = decode_token(credentials.credentials)
        user_id  = int(payload.get("sub", 0))
        if user_id <= 0:
            raise _CREDENTIALS_EXCEPTION
    except (JWTError, ValueError, TypeError):
        logger.warning("JWT 解码失败或 sub 字段无效")
        raise _CREDENTIALS_EXCEPTION

    user = await get_user_by_id(db, user_id)
    if user is None:
        logger.warning(f"Token 对应用户不存在: user_id={user_id}")
        raise _CREDENTIALS_EXCEPTION
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="账号已被禁用",
        )

    return user
