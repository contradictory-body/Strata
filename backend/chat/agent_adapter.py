"""
chat/agent_adapter.py — Agent 实例池 + 消息处理适配层
======================================================
职责：
  1. 按 user_id 维护 JobAgentV2 实例池（一用户一实例，懒加载）
  2. 按 session_id 维护 asyncio.Lock（防止同一会话并发处理）
  3. 封装 process_message / process_file 流程：
       Redis 读取状态 → Agent 处理 → 广播事件 → Redis 写回状态 → PostgreSQL 持久化

多端同步机制：
  - 任意端发送消息 → handle_message 获得 session 锁
  - 从 Redis 读取最新 session_state（所有端共享同一份状态）
  - Agent 生成器逐 event yield → broadcast 广播给该 session 所有 WebSocket
  - 生成器耗尽后将就地修改后的 session_state 写回 Redis
  - 另一端连接后读取到的是包含本轮对话的最新状态

并发保护：
  - 同一 session_id：串行处理（session_lock），防止消息乱序和状态竞争
  - 同一 user_id 不同 session：并行处理（不同 session_lock），无冲突
  - Agent 实例创建：_pool_create_lock 防止重复初始化
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING
import os
from pathlib import Path
from dotenv import load_dotenv

# ── 路径设置：让 agent 目录下的模块可直接 import ──────────────────────────────
# job_agent.py 位于 Strata_v6/agent/reme_light_job_agent_v2/job_agent.py
# 将该目录加入 sys.path，使得 `from job_agent import ...` 可以直接使用
_AGENT_DIR = Path(__file__).parent.parent.parent / "agent" / "reme_light_job_agent_v2"
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

# 同时确保 Strata_v6 的父目录在 sys.path（job_agent 内部用 PACKAGE_NAME.reme_light）
_REPO_ROOT   = Path(__file__).parent.parent.parent
_REPO_PARENT = _REPO_ROOT.parent
if str(_REPO_ROOT)   not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

if TYPE_CHECKING:
    # 仅供类型检查使用，不在运行时执行（避免循环 import）
    from job_agent import JobAgentV2

logger = logging.getLogger("chat.agent_adapter")

# ── Agent 实例池 ──────────────────────────────────────────────────────────────
# user_id(int) → JobAgentV2 实例
_agent_pool: dict[int, "JobAgentV2"] = {}

# 保护实例池的创建过程（防止并发重复创建同一用户的 Agent）
_pool_create_lock = asyncio.Lock()

# ── 会话级锁 ──────────────────────────────────────────────────────────────────
# session_id(str) → asyncio.Lock
# 确保同一会话的消息串行处理，防止 session_state 竞争写
_session_locks: dict[str, asyncio.Lock] = {}
_locks_dict_lock = asyncio.Lock()          # 保护 _session_locks 字典本身


async def _get_session_lock(session_id: str) -> asyncio.Lock:
    """懒加载 session_id 对应的 asyncio.Lock。"""
    async with _locks_dict_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = asyncio.Lock()
        return _session_locks[session_id]


async def get_or_create_agent(user_id: int) -> "JobAgentV2":
    """
    按 user_id 获取或创建 JobAgentV2 实例。

    首次创建时：
      - working_dir 设为 data/{user_id}/agent（与其他用户完全隔离）
      - 调用 agent.start() 启动 ReMeLight 后台服务

    之后复用同一实例，实例本身无对话状态（状态在 Redis 中）。
    """
    if user_id in _agent_pool:
        return _agent_pool[user_id]

    async with _pool_create_lock:
        # 双重检查，防止锁等待期间被其他协程创建
        if user_id in _agent_pool:
            return _agent_pool[user_id]

        # 延迟 import，避免 backend 模块初始化时触发 agent 路径依赖
        from job_agent import create_agent
        from backend.config import settings

        from job_agent import create_agent
        from backend.config import settings
        import os
        from pathlib import Path
        from dotenv import load_dotenv

        # 强制加载项目根目录 .env
        project_root = Path(__file__).resolve().parents[2]
        root_env = project_root / ".env"
        backend_env = project_root / "backend" / ".env"

        if root_env.exists():
            load_dotenv(root_env, override=True)
        if backend_env.exists():
            load_dotenv(backend_env, override=True)

        real_model = (
            os.getenv("LLM_MODEL")
            or settings.LLM_MODEL
        )

        real_api_key = (
            os.getenv("LLM_API_KEY")
            or settings.LLM_API_KEY
        )

        real_base_url = (
            os.getenv("LLM_BASE_URL")
            or settings.LLM_BASE_URL
        )

        logger.info(
            "create_agent 最终配置 | "
            f"model={real_model} "
            f"base_url={real_base_url} "
            f"has_api_key={bool(real_api_key)} "
            f"root_env_exists={root_env.exists()} "
            f"backend_env_exists={backend_env.exists()}"
        )

        agent = create_agent(
            user_id=str(user_id),
            data_root=settings.DATA_ROOT,
            model=real_model,
            api_key=real_api_key,
            base_url=real_base_url,
            tavily_api_key=settings.TAVILY_API_KEY or None,
            enable_mcp_discovery=False,
        )
        await agent.start()
        _agent_pool[user_id] = agent
        logger.info(f"Agent 实例创建: user_id={user_id}")

    return _agent_pool[user_id]


async def shutdown_all_agents() -> None:
    """
    关闭所有 Agent 实例（应用关闭时调用）。
    等待 ReMeLight 后台摘要任务完成后再退出。
    """
    async with _pool_create_lock:
        for user_id, agent in list(_agent_pool.items()):
            try:
                await agent.close()
                logger.info(f"Agent 已关闭: user_id={user_id}")
            except Exception as e:
                logger.warning(f"Agent 关闭异常: user_id={user_id}  err={e}")
        _agent_pool.clear()


# ── 消息处理主流程 ────────────────────────────────────────────────────────────

async def handle_message(
    user_id:    int,
    session_id: str,
    text:       str,
    image_data: dict | None = None,
) -> None:
    """
    处理用户消息，流程如下：

    1. 获取 session 锁（同一会话串行化）
    2. 广播 user_message 到所有设备（多端同步第一步）
    3. 从 Redis 加载 session_state
    4. 获取/创建 Agent 实例
    5. 调用 agent.process_message() 生成器，每个 event 广播给所有设备
    6. 生成器耗尽后将修改后的 state 写回 Redis
    7. 将 user/assistant 消息持久化到 PostgreSQL
    """
    from backend.chat.connection_manager import connection_manager
    from backend.chat import session_state as ss

    lock = await _get_session_lock(session_id)

    async with lock:
        # Step 1: 广播用户消息给所有连接设备（含发送方）
        # 前端收到 user_message 事件后在对话框显示，保证多端一致性
        await connection_manager.broadcast(session_id, {
            "type": "user_message",
            "data": {"content": text, "role": "user"},
        })

        # Step 2: 加载 Redis 中的 session_state
        state = await ss.load_state(session_id)

        # Step 3: 获取 Agent
        agent = await get_or_create_agent(user_id)

        # Step 4: 运行 Agent，逐事件广播
        accumulated_assistant_text = ""
        try:
            async for event in agent.process_message(text, state, image_data):
                await connection_manager.broadcast(session_id, event)
                # 收集 token 拼接完整的 assistant 回复（用于 DB 持久化）
                if event["type"] == "token":
                    accumulated_assistant_text += event["data"]
        except Exception as e:
            logger.exception(f"Agent 处理异常: session={session_id}  err={e}")
            await connection_manager.broadcast(session_id, {
                "type": "error",
                "data": f"处理消息时发生错误: {e}",
            })

        # Step 5: 写回 Redis（包含本轮新增的 messages）
        await ss.save_state(session_id, state)

        # Step 6: 持久化到 PostgreSQL
        await _persist_messages(
            session_id=session_id,
            user_text=text,
            assistant_text=accumulated_assistant_text,
        )


async def handle_file(
    user_id:      int,
    session_id:   str,
    file_name:    str,
    file_type:    str,
    file_content: str | None,
    image_data:   dict | None = None,
    user_hint:    str         = "",
) -> None:
    """
    处理前端上传的已解析文件，流程与 handle_message 一致。
    文件解析由 FastAPI /files/upload 端点完成，此处只处理已提取的文本/图片。
    """
    from backend.chat.connection_manager import connection_manager
    from backend.chat import session_state as ss

    lock = await _get_session_lock(session_id)

    async with lock:
        # 广播文件上传通知给所有设备
        await connection_manager.broadcast(session_id, {
            "type": "user_message",
            "data": {
                "content": user_hint or f"[上传文件: {file_name}]",
                "role":    "user",
                "file":    {"name": file_name, "type": file_type},
            },
        })

        state = await ss.load_state(session_id)
        agent = await get_or_create_agent(user_id)

        accumulated_assistant_text = ""
        try:
            async for event in agent.process_file(
                file_name=file_name,
                file_type=file_type,
                file_content=file_content,
                session_state=state,
                image_data=image_data,
                user_hint=user_hint,
            ):
                await connection_manager.broadcast(session_id, event)
                if event["type"] == "token":
                    accumulated_assistant_text += event["data"]
        except Exception as e:
            logger.exception(f"文件处理异常: session={session_id}  err={e}")
            await connection_manager.broadcast(session_id, {
                "type": "error",
                "data": f"文件处理时发生错误: {e}",
            })

        await ss.save_state(session_id, state)
        await _persist_messages(
            session_id=session_id,
            user_text=user_hint or f"[上传文件: {file_name}]",
            assistant_text=accumulated_assistant_text,
        )


# ── PostgreSQL 持久化 ─────────────────────────────────────────────────────────

async def _persist_messages(
    session_id:     str,
    user_text:      str,
    assistant_text: str,
) -> None:
    """
    将本轮 user / assistant 消息写入 PostgreSQL，并更新 session.updated_at 和 title。

    使用独立的 AsyncSession（而非 WebSocket handler 的 db 依赖），
    确保每次持久化是独立事务，不受 WebSocket 长连接的事务状态影响。
    """
    from backend.database import AsyncSessionLocal
    from backend.chat.models import ChatMessage, ChatSession
    from sqlalchemy import select, update
    from datetime import datetime, timezone

    async with AsyncSessionLocal() as db:
        try:
            now = datetime.now(timezone.utc)

            # 写入 user 消息
            if user_text:
                db.add(ChatMessage(
                    session_id=session_id,
                    role="user",
                    content=user_text,
                ))

            # 写入 assistant 消息
            if assistant_text:
                db.add(ChatMessage(
                    session_id=session_id,
                    role="assistant",
                    content=assistant_text,
                ))

            # 更新 session.updated_at（用于会话列表按活跃时间排序）
            # 若 title 为 None，自动用首条 user 消息填充
            result = await db.execute(
                select(ChatSession).where(ChatSession.id == session_id)
            )
            session = result.scalar_one_or_none()
            if session:
                session.updated_at = now
                if session.title is None and user_text:
                    title_raw = user_text.strip().replace("\n", " ")
                    session.title = title_raw[:40] + ("…" if len(title_raw) > 40 else "")

            await db.commit()
            logger.debug(
                f"消息持久化完成: session={session_id}  "
                f"user={bool(user_text)}  assistant={bool(assistant_text)}"
            )

        except Exception as e:
            await db.rollback()
            logger.error(f"消息持久化失败: session={session_id}  err={e}")
