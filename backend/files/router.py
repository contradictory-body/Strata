"""
files/router.py — 文件上传与解析
==================================
POST /api/files/upload

接收前端上传的 PDF / Word / 图片文件，
调用现有的 agent/file_parser.py 解析，
返回提取的文本内容或 base64 图片数据。

前端收到解析结果后，通过 WebSocket 发送
  { type: 'file', file_name, file_type, file_content, image_data, user_hint }
Agent 端由 process_file() 处理。

文件大小限制：20 MB（FastAPI 默认无限制，需显式设置）
支持格式：PDF / DOCX / JPG / JPEG / PNG / WEBP / GIF
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel

from backend.auth.dependencies import get_current_user
from backend.auth.models import User

logger = logging.getLogger("files.router")
router = APIRouter(prefix="/api/files", tags=["文件"])

# ── 常量 ─────────────────────────────────────────────────────
MAX_FILE_BYTES = 20 * 1024 * 1024   # 20 MB

SUPPORTED_SUFFIXES = {
    ".pdf", ".docx", ".doc",
    ".jpg", ".jpeg", ".png", ".webp", ".gif",
}

# ── 确保 agent 目录在 sys.path ────────────────────────────────
_AGENT_DIR = (
    Path(__file__).parent.parent.parent
    / "agent"
    / "reme_light_job_agent_v2"
)
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))


# ── 响应模型 ─────────────────────────────────────────────────
class ImageData(BaseModel):
    base64:     str
    media_type: str


class FileUploadResponse(BaseModel):
    file_name:    str
    file_type:    str            # pdf / docx / image
    text_content: str | None     # 文本类：提取的正文
    image_data:   ImageData | None  # 图片类：base64 编码
    char_count:   int
    page_count:   int            # PDF 页数（其他类型为 0）
    truncated:    bool           # 文本是否因过长被截断


# ── 接口 ─────────────────────────────────────────────────────
@router.post(
    "/upload",
    response_model=FileUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="上传并解析文件",
    description=(
        "支持 PDF / Word / 图片（JPG / PNG / WEBP / GIF）。\n\n"
        "- 文本类文件返回 `text_content`（最多 20,000 字符）\n"
        "- 图片类文件返回 `image_data.base64` + `image_data.media_type`\n"
        "- 文件大小上限 20 MB"
    ),
)
async def upload_file(
    file:         UploadFile,
    current_user: User = Depends(get_current_user),
) -> FileUploadResponse:
    # ── 文件名与扩展名验证 ────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {suffix}，支持: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
        )

    # ── 读取文件内容（限制大小）──────────────────────────────
    raw = await file.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大，最大支持 {MAX_FILE_BYTES // 1024 // 1024} MB",
        )

    logger.info(
        f"文件上传: user={current_user.id}  name={file.filename!r}  "
        f"size={len(raw):,} bytes"
    )

    # ── 写入临时文件供 parse_file() 使用 ─────────────────────
    # parse_file 需要 Path 对象，直接传内存 bytes 不支持
    with tempfile.NamedTemporaryFile(
        suffix=suffix, delete=False, dir=tempfile.gettempdir()
    ) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    try:
        return _parse_and_respond(tmp_path, file.filename)
    finally:
        tmp_path.unlink(missing_ok=True)


# ── 解析逻辑 ─────────────────────────────────────────────────
MAX_TEXT_CHARS = 20_000   # 返回给前端的最大文本长度（避免传输过大）


def _parse_and_respond(path: Path, original_name: str) -> FileUploadResponse:
    """调用 file_parser.parse_file()，将结果转换为 API 响应。"""
    try:
        from file_parser import parse_file
    except ImportError as e:
        logger.error(f"file_parser 导入失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="文件解析服务不可用，请确认 agent 目录在 PYTHONPATH 中",
        )

    result = parse_file(path)
    result.file_name = original_name   # 恢复原始文件名（临时文件名不友好）

    if not result.success:
        raise HTTPException(
            status_code=422,
            detail=f"文件解析失败: {result.error}",
        )

    # 图片类型
    if result.is_image:
        if not result.image_base64:
            raise HTTPException(status_code=422, detail="图片编码失败")
        return FileUploadResponse(
            file_name=original_name,
            file_type=result.file_type,
            text_content=None,
            image_data=ImageData(
                base64=result.image_base64,
                media_type=result.image_media_type,
            ),
            char_count=result.char_count,
            page_count=0,
            truncated=False,
        )

    # 文本类型（PDF / DOCX）
    text      = result.text_content or ""
    truncated = len(text) > MAX_TEXT_CHARS
    if truncated:
        text = text[:MAX_TEXT_CHARS]

    return FileUploadResponse(
        file_name=original_name,
        file_type=result.file_type,
        text_content=text if text else None,
        image_data=None,
        char_count=result.char_count,
        page_count=result.page_count,
        truncated=truncated,
    )
