"""
chat/connection_manager.py — WebSocket 多端广播管理器
======================================================
核心职责：维护 session_id → set[WebSocket] 的广播表，
将同一会话产生的所有事件广播给所有已连接的客户端。

这是实现「同一会话跨设备实时同步」的核心机制：
  - 用户在手机上发送消息 → 服务端处理 → 广播给所有连接该 session 的设备
  - 用户同时打开两个浏览器标签页，两个页面看到完全一致的对话流

单进程 uvicorn（推荐部署模式）下内存广播表完全可靠。
如需多进程部署，需将广播表替换为 Redis Pub/Sub。

广播事件类型（由 agent_adapter.py 发起）：
  connected   — WebSocket 握手完成，携带历史消息数量
  user_message — 用户发送的消息（广播给所有设备，含发送方自身）
  token        — LLM 流式输出的单个 token
  tool_start   — 工具开始执行
  tool_end     — 工具执行完成
  clarify      — 澄清反问文本
  memory_hits  — 本轮记忆检索命中
  done         — 本轮处理完成
  error        — 发生不可恢复的异常
  pong         — 心跳响应
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger("chat.connection_manager")


class ConnectionManager:
    """
    WebSocket 连接生命周期管理 + 会话级广播。

    线程安全说明：
      asyncio 是单线程事件循环，无需额外加锁。
      所有操作（connect / disconnect / broadcast）必须在同一事件循环内调用。
    """

    def __init__(self) -> None:
        # session_id → 该 session 当前活跃的所有 WebSocket 连接集合
        self._connections: dict[str, set[WebSocket]] = {}

    # ── 连接管理 ─────────────────────────────────────────────────────────────

    async def connect(self, session_id: str, ws: WebSocket) -> None:
        """
        将新 WebSocket 加入对应 session 的广播组。
        注意：调用此函数前 ws.accept() 必须已完成。
        """
        if session_id not in self._connections:
            self._connections[session_id] = set()
        self._connections[session_id].add(ws)
        count = len(self._connections[session_id])
        logger.info(f"WS 连接加入: session={session_id}  当前连接数={count}")

    def disconnect(self, session_id: str, ws: WebSocket) -> None:
        """
        从广播组中移除 WebSocket（断开或关闭时调用）。
        若该 session 无剩余连接，清理 key 释放内存。
        """
        group = self._connections.get(session_id)
        if group is None:
            return
        group.discard(ws)
        remaining = len(group)
        if remaining == 0:
            del self._connections[session_id]
        logger.info(f"WS 连接移除: session={session_id}  剩余连接数={remaining}")

    # ── 广播 ─────────────────────────────────────────────────────────────────

    async def broadcast(self, session_id: str, event: dict[str, Any]) -> None:
        """
        将事件广播给该 session 所有活跃 WebSocket。

        发送失败的连接（客户端已断开但未收到 disconnect 事件）会被自动剔除。
        这种「惰性清理」机制保证广播表不会因异常断连而膨胀。
        """
        group = self._connections.get(session_id)
        if not group:
            return

        dead: set[WebSocket] = set()
        for ws in list(group):          # list() 防止迭代中修改集合
            try:
                await ws.send_json(event)
            except Exception:
                dead.add(ws)

        # 清理死连接
        for ws in dead:
            group.discard(ws)
        if dead:
            logger.debug(f"清理死连接 {len(dead)} 个: session={session_id}")
        if not group:
            self._connections.pop(session_id, None)

    async def send_to(self, ws: WebSocket, event: dict[str, Any]) -> bool:
        """
        向单个 WebSocket 发送事件（用于 connected / pong 等单播场景）。
        返回 True 表示发送成功，False 表示连接已失效。
        """
        try:
            await ws.send_json(event)
            return True
        except Exception:
            return False

    # ── 查询 ─────────────────────────────────────────────────────────────────

    def get_connection_count(self, session_id: str) -> int:
        """返回当前 session 的活跃连接数。"""
        return len(self._connections.get(session_id, set()))

    def get_all_sessions(self) -> list[str]:
        """返回当前有活跃连接的所有 session_id，用于监控。"""
        return list(self._connections.keys())

    @property
    def total_connections(self) -> int:
        """返回全局活跃 WebSocket 连接总数。"""
        return sum(len(g) for g in self._connections.values())


# ── 全局单例 ──────────────────────────────────────────────────────────────────
# 整个进程共享同一个 ConnectionManager 实例
connection_manager = ConnectionManager()
