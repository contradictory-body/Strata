"""
chat/session_state.py — Redis 会话状态读写
===========================================
将 job_agent.py 中 process_message() 使用的 session_state dict
序列化为 JSON 存入 Redis，实现跨请求（跨 WebSocket 连接）的状态持久化。

存储格式：
  Key:   ss:{session_id}
  Value: JSON 字符串（session_state dict）
  TTL:   SESSION_TTL 秒（默认 2 小时），每次活跃时刷新

session_state 结构（与 job_agent.py 约定一致）：
  {
      "messages":              list[dict],   # OpenAI 格式对话历史
      "pending_clarification": dict | None,  # 挂起的澄清状态
      "compressed_summary":    str,          # LLM 压缩摘要
      "last_memory_hits":      list,         # 最近记忆命中（展示用）
  }

多端同步原理：
  任意设备发送消息 → agent_adapter 从 Redis 加载 state
  → process_message 就地修改 state
  → 广播完毕后将新 state 写回 Redis
  → 另一台设备下次发消息时读取的是最新 state
"""

from __future__ import annotations

import json
import logging

from backend.redis_client import SESSION_TTL, get_redis

logger = logging.getLogger("chat.session_state")


def _key(session_id: str) -> str:
    """Redis key 生成，短前缀减少内存占用。"""
    return f"ss:{session_id}"


def _empty_state() -> dict:
    """
    返回空白 session_state。
    与 job_agent.make_empty_session_state() 结构完全一致，
    此处单独定义避免在 backend 层引入 agent 包的模块级 import。
    """
    return {
        "messages":              [],
        "pending_clarification": None,
        "compressed_summary":    "",
        "last_memory_hits":      [],
    }


async def load_state(session_id: str) -> dict:
    """
    从 Redis 加载 session_state。
    若 key 不存在（新会话或 TTL 过期）返回空白状态。
    """
    redis = get_redis()
    raw = await redis.get(_key(session_id))
    if raw is None:
        logger.debug(f"session_state 未命中，返回空白状态: session={session_id}")
        return _empty_state()
    try:
        state = json.loads(raw)
        # 保证所有必要字段存在（向后兼容）
        state.setdefault("messages",              [])
        state.setdefault("pending_clarification", None)
        state.setdefault("compressed_summary",    "")
        state.setdefault("last_memory_hits",      [])
        return state
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"session_state 解析失败，重置为空白: session={session_id}  err={e}")
        return _empty_state()


async def save_state(session_id: str, state: dict) -> None:
    """
    将 session_state 序列化后写入 Redis，同时刷新 TTL。
    每次对话结束后由 agent_adapter 调用。
    """
    redis = get_redis()
    try:
        payload = json.dumps(state, ensure_ascii=False)
        await redis.setex(_key(session_id), SESSION_TTL, payload)
        logger.debug(
            f"session_state 已保存: session={session_id}  "
            f"msgs={len(state.get('messages', []))}"
        )
    except Exception as e:
        logger.error(f"session_state 写入失败: session={session_id}  err={e}")
        raise


async def delete_state(session_id: str) -> None:
    """
    删除会话状态（用户删除会话时调用）。
    PostgreSQL 中的消息记录由 ON DELETE CASCADE 自动清理。
    """
    redis = get_redis()
    deleted = await redis.delete(_key(session_id))
    if deleted:
        logger.info(f"session_state 已删除: session={session_id}")


async def refresh_ttl(session_id: str) -> None:
    """
    仅刷新 TTL（不修改内容），用于 WebSocket 心跳时延长会话有效期。
    """
    redis = get_redis()
    await redis.expire(_key(session_id), SESSION_TTL)
