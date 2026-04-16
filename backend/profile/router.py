"""
profile/router.py — 用户求职画像 API
======================================
GET  /api/profile         — 读取当前用户的完整画像
PUT  /api/profile         — 批量更新画像字段
GET  /api/profile/summary — 获取画像摘要（注入 system prompt 的格式）

画像存储在 {DATA_ROOT}/{user_id}/agent/PROFILE.md，
通过 job_agent.py 的 get_or_create_agent() 获取 Agent 实例来操作，
确保与 Agent 使用的是同一份 ProfileManager（路径完全一致）。

字段说明（对应 PROFILE_SECTIONS）：
  目标岗位 / 目标城市 / 技术栈 / 目标公司_行业 / 薪资预期 / 面试薄弱点 / 简历修改偏好
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.auth.dependencies import get_current_user
from backend.auth.models import User

logger = logging.getLogger("profile.router")
router = APIRouter(prefix="/api/profile", tags=["画像"])

# ── 画像字段（与 ProfileManager.PROFILE_SECTIONS 完全一致）──
PROFILE_FIELDS = [
    "目标岗位",
    "目标城市",
    "技术栈",
    "目标公司_行业",
    "薪资预期",
    "面试薄弱点",
    "简历修改偏好",
]


# ── Pydantic 模型 ────────────────────────────────────────────
class ProfileResponse(BaseModel):
    """画像完整内容（raw Markdown 文本 + 解析后的字段字典）"""
    raw:    str               # PROFILE.md 原文
    fields: dict[str, str]    # section 名 → 内容（过滤掉"暂未填写"）


class ProfileUpdateRequest(BaseModel):
    """批量更新请求：只传需要更新的字段"""
    updates: dict[str, str]    # { "目标岗位": "后端工程师", ... }


class ProfileUpdateResponse(BaseModel):
    updated_fields: list[str]  # 实际更新成功的字段


class ProfileSummaryResponse(BaseModel):
    summary: str               # 注入 system prompt 的摘要字符串


# ── 获取画像 ─────────────────────────────────────────────────
@router.get(
    "",
    response_model=ProfileResponse,
    summary="获取求职画像",
    description="返回当前用户的完整求职画像，包含原始 Markdown 和解析后的字段字典。",
)
async def get_profile(
    current_user: User = Depends(get_current_user),
) -> ProfileResponse:
    agent = await _get_agent(current_user.id)
    raw   = agent.profile.read()

    # 解析各字段
    fields: dict[str, str] = {}
    for field in PROFILE_FIELDS:
        val = agent.profile.read_section(field)
        if val and "暂未填写" not in val:
            fields[field] = val

    return ProfileResponse(raw=raw, fields=fields)


# ── 更新画像 ─────────────────────────────────────────────────
@router.put(
    "",
    response_model=ProfileUpdateResponse,
    summary="更新求职画像",
    description=(
        "批量更新画像字段。只传需要更新的字段，不传的字段保持不变。\n\n"
        f"合法字段名：{', '.join(PROFILE_FIELDS)}"
    ),
)
async def update_profile(
    req:          ProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
) -> ProfileUpdateResponse:
    # 过滤非法字段
    valid = {k: v for k, v in req.updates.items() if k in PROFILE_FIELDS}
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无有效字段，合法字段为: {', '.join(PROFILE_FIELDS)}",
        )

    agent   = await _get_agent(current_user.id)
    updated = agent.update_profile_dict(valid)

    logger.info(f"画像更新: user={current_user.id}  fields={updated}")
    return ProfileUpdateResponse(updated_fields=updated)


# ── 获取画像摘要 ─────────────────────────────────────────────
@router.get(
    "/summary",
    response_model=ProfileSummaryResponse,
    summary="获取画像摘要",
    description="返回用于注入 Agent system prompt 的简洁摘要字符串。",
)
async def get_profile_summary(
    current_user: User = Depends(get_current_user),
) -> ProfileSummaryResponse:
    agent   = await _get_agent(current_user.id)
    summary = agent.profile.to_summary_str()
    return ProfileSummaryResponse(summary=summary)


# ── 内部辅助 ─────────────────────────────────────────────────
async def _get_agent(user_id: int) -> Any:
    """
    获取该用户的 Agent 实例（复用 agent_adapter 的实例池）。
    确保 profile 路径与对话时完全一致。
    """
    try:
        from backend.chat.agent_adapter import get_or_create_agent
        return await get_or_create_agent(user_id)
    except Exception as e:
        logger.error(f"获取 Agent 实例失败: user={user_id}  err={e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent 服务暂时不可用，请稍后重试",
        )
