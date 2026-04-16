"""
main.py — FastAPI 应用入口
===========================
启动命令（在 Strata_v6/ 根目录下运行）：
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

生产环境（单进程，WebSocket 广播依赖单进程内存）：
  uvicorn backend.main:app --host 0.0.0.0 --port 8000

API 文档（APP_DEBUG=true 时可访问）：
  http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database import init_db
from backend.redis_client import close_redis, init_redis
from backend.auth.router    import router as auth_router
from backend.chat.router    import router as chat_router
from backend.files.router   import router as files_router    # ★ Round 6
from backend.profile.router import router as profile_router  # ★ Round 6

logging.basicConfig(
    level=logging.DEBUG if settings.APP_DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("═" * 52)
    logger.info("  Strata 后端服务正在启动...")
    logger.info("═" * 52)

    logger.info("[1/3] 初始化 PostgreSQL 数据库表...")
    await init_db()

    logger.info("[2/3] 连接 Redis...")
    await init_redis()

    logger.info("[3/3] 服务就绪")
    logger.info(f"  WebSocket: ws://0.0.0.0:{settings.APP_PORT}/ws/{{session_id}}?token=<jwt>")
    if settings.APP_DEBUG:
        logger.info(f"  API 文档:  http://0.0.0.0:{settings.APP_PORT}/docs")

    yield

    logger.info("后端服务正在关闭...")
    from backend.chat.agent_adapter import shutdown_all_agents
    await shutdown_all_agents()
    await close_redis()
    logger.info("✓ 后端服务已关闭")


app = FastAPI(
    title="Strata API",
    description=(
        "Strata 求职助手后端服务。\n\n"
        "## 认证方式\n"
        "除注册/登录接口外，所有 REST 接口需在 `Authorization` 请求头携带 Bearer Token：\n"
        "```\nAuthorization: Bearer <your_token>\n```\n\n"
        "## WebSocket\n"
        "WebSocket 连接通过 query param 传递 Token：\n"
        "```\nws://host/ws/{session_id}?token=<your_token>\n```"
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs"  if settings.APP_DEBUG else None,
    redoc_url="/redoc" if settings.APP_DEBUG else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 路由注册 ──────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(files_router)    # POST /api/files/upload
app.include_router(profile_router)  # GET/PUT /api/profile

# ── 前端静态文件托管 ──────────────────────────────────────────
_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
    logger.info(f"前端静态文件托管: {_FRONTEND_DIST}")


@app.get("/api/health", tags=["系统"], summary="健康检查")
async def health_check() -> dict:
    from backend.chat.connection_manager import connection_manager
    return {
        "status":               "ok",
        "version":              "2.0.0",
        "active_ws_sessions":   len(connection_manager.get_all_sessions()),
        "total_ws_connections": connection_manager.total_connections,
    }
