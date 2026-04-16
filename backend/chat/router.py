"""
chat/router.py — WebSocket + 会话管理 REST 路由
=================================================
WebSocket 端点：
  WS  /ws/{session_id}?token=<jwt>

  客户端发送：
    {"type": "message", "content": "...", "image": {...}?}
    {"type": "file", "file_name": "...", "file_type": "...",
                     "file_content": "...", "image_data": {...}?,
                     "user_hint": "..."}      ← Round 6 新增
    {"type": "ping"}

  服务端广播：
    {"type": "connected",    "data": {"session_id": ..., "message_count": ...}}
    {"type": "user_message", "data": {"content": ..., "role": "user"}}
    {"type": "token",        "data": "token_chunk"}
    {"type": "tool_start",   "data": {"name": ..., "args": ...}}
    {"type": "tool_end",     "data": {"name": ...}}
    {"type": "clarify",      "data": "反问文本"}
    {"type": "memory_hits",  "data": [...]}
    {"type": "done",         "data": {"has_slots": bool}}
    {"type": "error",        "data": "错误描述"}
    {"type": "pong"}

REST 端点：
  POST   /api/sessions
  GET    /api/sessions
  GET    /api/sessions/{session_id}/messages
  DELETE /api/sessions/{session_id}
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.dependencies import get_current_user
from backend.auth.models import User
from backend.auth.service import decode_token
from backend.chat import agent_adapter
from backend.chat.connection_manager import connection_manager
from backend.chat import session_state as ss
from backend.chat.models import ChatMessage, ChatSession
from backend.chat.schemas import (
    CreateSessionRequest,
    MessageResponse,
    SessionResponse,
)
from backend.database import get_db

logger = logging.getLogger("chat.router")
router = APIRouter(tags=["对话"])


# ── REST 端点 ────────────────────────────────────────────────

@router.post(
    "/api/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建新会话",
)
async def create_session(
    req:          CreateSessionRequest = CreateSessionRequest(),
    current_user: User                 = Depends(get_current_user),
    db:           AsyncSession         = Depends(get_db),
) -> SessionResponse:
    session = ChatSession(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        title=req.title,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    logger.info(f"新会话创建: session={session.id}  user_id={current_user.id}")
    return SessionResponse.model_validate(session)


@router.get(
    "/api/sessions",
    response_model=list[SessionResponse],
    summary="获取会话列表",
    description="返回当前用户所有会话，按最后活跃时间倒序排列。",
)
async def list_sessions(
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
    limit:        int          = Query(50, ge=1, le=200),
    offset:       int          = Query(0,  ge=0),
) -> list[SessionResponse]:
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return [SessionResponse.model_validate(s) for s in result.scalars().all()]


@router.get(
    "/api/sessions/{session_id}/messages",
    response_model=list[MessageResponse],
    summary="获取会话历史消息",
    description="按时间正序返回，支持 limit/offset 分页。",
)
async def get_messages(
    session_id:   str,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
    limit:        int          = Query(100, ge=1, le=500),
    offset:       int          = Query(0,   ge=0),
) -> list[MessageResponse]:
    await _get_session_or_404(db, session_id, current_user.id)
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .limit(limit)
        .offset(offset)
    )
    return [MessageResponse.model_validate(m) for m in result.scalars().all()]


@router.delete(
    "/api/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除会话",
)
async def delete_session(
    session_id:   str,
    current_user: User         = Depends(get_current_user),
    db:           AsyncSession = Depends(get_db),
) -> None:
    await _get_session_or_404(db, session_id, current_user.id)
    await db.execute(
        delete(ChatSession).where(
            ChatSession.id      == session_id,
            ChatSession.user_id == current_user.id,
        )
    )
    await db.commit()
    await ss.delete_state(session_id)
    logger.info(f"会话已删除: session={session_id}  user_id={current_user.id}")


# ── WebSocket 端点 ───────────────────────────────────────────

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket:  WebSocket,
    session_id: str,
    token:      str          = Query(..., description="JWT Token"),
    db:         AsyncSession = Depends(get_db),
) -> None:
    # Step 1: JWT 鉴权
    user_id: int
    try:
        payload = decode_token(token)
        user_id = int(payload.get("sub", 0))
        if user_id <= 0:
            raise ValueError("无效的 user_id")
    except Exception:
        await websocket.close(code=4001, reason="Token 无效或已过期")
        logger.warning(f"WS 鉴权失败: session={session_id}")
        return

    # Step 2: 验证 session 归属
    session = await _get_session_or_none(db, session_id, user_id)
    if session is None:
        await websocket.close(code=4004, reason="会话不存在或无权访问")
        logger.warning(f"WS 会话验证失败: session={session_id}  user_id={user_id}")
        return

    # Step 3: 接受连接
    await websocket.accept()
    await connection_manager.connect(session_id, websocket)
    logger.info(
        f"WS 已连接: session={session_id}  user_id={user_id}  "
        f"总连接数={connection_manager.total_connections}"
    )

    try:
        # Step 4: 发送 connected 事件
        msg_count = await _count_messages(db, session_id)
        await connection_manager.send_to(websocket, {
            "type": "connected",
            "data": {"session_id": session_id, "message_count": msg_count},
        })

        # Step 5: 消息循环
        while True:
            try:
                raw = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(f"WS 消息解析异常: {e}")
                continue

            msg_type = raw.get("type", "message")

            # ── 普通文本 / 图片消息 ───────────────────────────
            if msg_type == "message":
                content    = (raw.get("content") or "").strip()
                image_data = raw.get("image")   # dict | None
                if not content and not image_data:
                    continue
                asyncio.create_task(
                    agent_adapter.handle_message(
                        user_id=user_id,
                        session_id=session_id,
                        text=content,
                        image_data=image_data,
                    )
                )

            # ── 文件消息（Round 6 新增）────────────────────────
            # 前端上传文件 → REST /api/files/upload 解析
            # → WS 发送 type:'file' 把解析结果交给 Agent 处理
            elif msg_type == "file":
                file_name    = raw.get("file_name",    "")
                file_type    = raw.get("file_type",    "")
                file_content = raw.get("file_content") # str | None
                image_data   = raw.get("image_data")   # dict | None
                user_hint    = (raw.get("user_hint") or "").strip()

                if not file_name:
                    logger.warning(f"WS file 消息缺少 file_name: session={session_id}")
                    continue

                asyncio.create_task(
                    agent_adapter.handle_file(
                        user_id=user_id,
                        session_id=session_id,
                        file_name=file_name,
                        file_type=file_type,
                        file_content=file_content,
                        image_data=image_data,
                        user_hint=user_hint,
                    )
                )

            # ── 心跳 ─────────────────────────────────────────
            elif msg_type == "ping":
                await connection_manager.send_to(websocket, {"type": "pong"})
                await ss.refresh_ttl(session_id)

            else:
                logger.debug(f"未知消息类型: {msg_type}  session={session_id}")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception(f"WS 异常: session={session_id}  err={e}")
        try:
            await connection_manager.send_to(websocket, {
                "type": "error",
                "data": f"服务器内部错误: {e}",
            })
        except Exception:
            pass
    finally:
        connection_manager.disconnect(session_id, websocket)
        logger.info(
            f"WS 已断开: session={session_id}  "
            f"总连接数={connection_manager.total_connections}"
        )


# ── 内部辅助 ─────────────────────────────────────────────────

async def _get_session_or_404(
    db: AsyncSession, session_id: str, user_id: int
) -> ChatSession:
    session = await _get_session_or_none(db, session_id, user_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在或无权访问",
        )
    return session


async def _get_session_or_none(
    db: AsyncSession, session_id: str, user_id: int
) -> ChatSession | None:
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id      == session_id,
            ChatSession.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def _count_messages(db: AsyncSession, session_id: str) -> int:
    from sqlalchemy import func as sqlfunc
    result = await db.execute(
        select(sqlfunc.count()).where(ChatMessage.session_id == session_id)
    )
    return result.scalar_one() or 0
