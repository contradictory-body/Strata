"""
config.py — 全局配置
====================
通过 pydantic-settings 从 backend/.env 读取环境变量，
所有模块通过 `from backend.config import settings` 引用。
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── 数据库 ────────────────────────────────────────────────────────────────
    DATABASE_URL: str = (
        "postgresql+asyncpg://strata:strata_password@localhost:5432/strata_db"
    )

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── JWT ───────────────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = "CHANGE_THIS_IN_PRODUCTION"
    JWT_ALGORITHM:  str = "HS256"
    JWT_EXPIRE_DAYS: int = 7

    # ── CORS（前端地址列表，逗号分隔）────────────────────────────────────────
    # pydantic-settings 会将逗号分隔字符串自动解析为 list
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]

    # ── LLM ───────────────────────────────────────────────────────────────────
    LLM_API_KEY:  str = ""
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL:    str = "qwen-plus"
    TAVILY_API_KEY: str = ""

    # ── Agent 数据目录 ────────────────────────────────────────────────────────
    DATA_ROOT: str = "data"

    # ── 服务配置 ──────────────────────────────────────────────────────────────
    APP_HOST:  str  = "0.0.0.0"
    APP_PORT:  int  = 8000
    APP_DEBUG: bool = False

    model_config = SettingsConfigDict(
        env_file=(
            str(Path(__file__).resolve().parent.parent / ".env"),  # 项目根目录 .env
            str(Path(__file__).resolve().parent / ".env"),         # backend/.env
        ),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

settings = Settings()

print(
    "[backend.config] "
    f"LLM_MODEL={settings.LLM_MODEL} "
    f"LLM_BASE_URL={settings.LLM_BASE_URL} "
    f"HAS_LLM_API_KEY={bool(settings.LLM_API_KEY)}"
)
