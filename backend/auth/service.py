"""
auth/service.py — 鉴权业务逻辑
================================
包含密码哈希、JWT 签发验证、注册/登录核心逻辑。
所有异常以 ValueError 抛出，由 router 层捕获转为 HTTP 响应。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.models import User
from backend.auth.schemas import LoginRequest, RegisterRequest, TokenResponse
from backend.config import settings

logger = logging.getLogger("auth.service")

# bcrypt 上下文（deprecated="auto" 表示自动迁移旧哈希算法）
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── 密码工具 ──────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """将明文密码哈希为 bcrypt 字符串。"""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """验证明文密码与哈希是否匹配。"""
    return _pwd_context.verify(plain, hashed)


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_access_token(user_id: int, username: str, email: str) -> str:
    """
    签发 JWT Access Token。

    Payload 字段：
      sub      — 用户 ID（字符串形式，符合 JWT 规范）
      username — 用户名
      email    — 邮箱
      exp      — 过期时间（UTC）
    """
    expire = datetime.now(timezone.utc) + timedelta(days=settings.JWT_EXPIRE_DAYS)
    payload = {
        "sub":      str(user_id),
        "username": username,
        "email":    email,
        "exp":      expire,
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """
    解码并验证 JWT Token，返回 payload dict。
    Token 无效或已过期时抛出 JWTError。
    """
    return jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
    )


# ── 数据库操作 ────────────────────────────────────────────────────────────────

async def get_user_by_id(db: AsyncSession, user_id: int) -> User | None:
    """按 ID 查询用户，不存在返回 None。"""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def register_user(db: AsyncSession, req: RegisterRequest) -> User:
    """
    注册新用户。

    冲突检查：用户名和邮箱均唯一，任一冲突时抛出 ValueError。
    密码在存入数据库前做 bcrypt 哈希处理。
    """
    # 检查用户名/邮箱是否已存在（一次查询）
    result = await db.execute(
        select(User).where(
            or_(User.username == req.username, User.email == req.email)
        )
    )
    existing = result.scalar_one_or_none()
    if existing is not None:
        if existing.username == req.username:
            raise ValueError("该用户名已被使用，请换一个")
        raise ValueError("该邮箱已被注册，请直接登录")

    user = User(
        username=req.username,
        email=str(req.email),
        hashed_password=hash_password(req.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    logger.info(f"新用户注册: id={user.id} username={user.username!r}")
    return user


async def login_user(db: AsyncSession, req: LoginRequest) -> TokenResponse:
    """
    用户登录，支持用户名或邮箱两种方式。

    验证失败统一返回"用户名或密码错误"，不区分哪个字段有误（防止用户枚举攻击）。
    """
    result = await db.execute(
        select(User).where(
            or_(User.username == req.username, User.email == req.username)
        )
    )
    user = result.scalar_one_or_none()

    # 故意不区分"用户不存在"和"密码错误"，防止枚举攻击
    if user is None or not verify_password(req.password, user.hashed_password):
        raise ValueError("用户名或密码错误")
    if not user.is_active:
        raise ValueError("账号已被禁用，请联系管理员")

    token = create_access_token(user.id, user.username, user.email)
    logger.info(f"用户登录: id={user.id} username={user.username!r}")
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        username=user.username,
        email=user.email,
    )
