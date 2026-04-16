"""
redis_client.py — Redis 异步连接管理
======================================
使用 redis-py v5+ 的原生 asyncio 支持（redis.asyncio）。

对外暴露：
  init_redis()   — 应用启动时建立连接池，验证连通性
  close_redis()  — 应用关闭时释放连接池
  get_redis()    — 获取全局 Redis 客户端实例
  SESSION_TTL    — session_state 的默认过期时间（秒）

会话状态存储约定（供 chat/session_state.py 使用）：
  Key:   session:{session_id}:state
  Value: JSON 序列化的 session_state dict
  TTL:   SESSION_TTL（默认 2 小时，每次活跃时刷新）
"""

from __future__ import annotations

import logging
from typing import Optional

import redis.asyncio as aioredis

from backend.config import settings

logger = logging.getLogger("redis_client")

# 会话状态过期时间：2 小时，用户活跃时每轮对话刷新
SESSION_TTL: int = 60 * 60 * 2

# 全局 Redis 客户端（在 init_redis() 中赋值）
_redis: Optional[aioredis.Redis] = None


async def init_redis() -> None:
    """
    在应用启动（lifespan）时初始化 Redis 连接池，并验证连通性。
    使用连接池模式，自动管理并发连接数。
    """
    global _redis
    _redis = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,        # 自动将 bytes 解码为 str
        max_connections=50,           # 连接池上限
    )
    # 验证连通性
    await _redis.ping()
    logger.info(f"Redis 连接成功: {settings.REDIS_URL}")


async def close_redis() -> None:
    """在应用关闭（lifespan）时释放 Redis 连接池。"""
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None
        logger.info("Redis 连接已关闭")


def get_redis() -> aioredis.Redis:
    """
    获取全局 Redis 客户端实例。

    用法（FastAPI 依赖注入）：
        from backend.redis_client import get_redis

        @router.get("/example")
        async def example(redis: aioredis.Redis = Depends(get_redis)):
            await redis.set("key", "value")

    用法（直接调用）：
        redis = get_redis()
        await redis.set("key", "value")
    """
    if _redis is None:
        raise RuntimeError("Redis 未初始化，请确认 init_redis() 已在 lifespan 中调用")
    return _redis
