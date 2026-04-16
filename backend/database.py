"""
database.py — PostgreSQL 异步数据库连接
=========================================
使用 SQLAlchemy 2.0 异步引擎 + asyncpg 驱动。

对外暴露：
  Base            — 所有 ORM 模型的基类，引入后定义 __tablename__ 即可
  engine          — 异步引擎（lifespan 关闭时调用 dispose()）
  AsyncSessionLocal — 异步 Session 工厂
  init_db()       — 首次启动时建表（CREATE TABLE IF NOT EXISTS）
  get_db()        — FastAPI 依赖注入，yield AsyncSession
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.config import settings

logger = logging.getLogger("database")

# ── 引擎 ──────────────────────────────────────────────────────────────────────
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.APP_DEBUG,          # DEBUG 模式下打印 SQL
    pool_size=10,                      # 连接池大小
    max_overflow=20,                   # 溢出连接上限
    pool_pre_ping=True,                # 每次使用前 ping，自动重连断线连接
    pool_recycle=3600,                 # 连接最长复用 1 小时
)

# ── Session 工厂 ──────────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,            # commit 后不过期，避免 lazy load 问题
)


# ── ORM 基类 ──────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """所有 ORM 模型继承此基类。"""
    pass


# ── 建表 ──────────────────────────────────────────────────────────────────────
async def init_db() -> None:
    """
    在应用启动时创建所有未存在的表（CREATE TABLE IF NOT EXISTS）。
    不会修改已存在的表结构，迁移变更请使用 Alembic。
    """
    # 必须先 import 所有 model，Base.metadata 才能感知到所有表
    import backend.auth.models  # noqa: F401
    import backend.chat.models   # noqa: F401
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("数据库表初始化完成")


# ── FastAPI 依赖注入 ──────────────────────────────────────────────────────────
async def get_db():
    """
    FastAPI 依赖函数，提供一次请求范围内的 AsyncSession。

    用法：
        @router.get("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
