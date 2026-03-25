"""
profile_manager.py — 结构化求职画像管理
存储在 working_dir/PROFILE.md，固定 section 结构，支持更新合并。
"""
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# Profile 的固定 section 顺序（稳定偏好）
PROFILE_SECTIONS = [
    "目标岗位",
    "目标城市",
    "技术栈",
    "目标公司_行业",
    "薪资预期",
    "面试薄弱点",
    "简历修改偏好",
]

PROFILE_TEMPLATE = """# 求职画像 (Job Seeker Profile)
> 由 ReMeLight 求职助手 Agent v2 自动维护
> 最后更新: {updated_at}

## 目标岗位
{目标岗位}

## 目标城市
{目标城市}

## 技术栈
{技术栈}

## 目标公司_行业
{目标公司_行业}

## 薪资预期
{薪资预期}

## 面试薄弱点
{面试薄弱点}

## 简历修改偏好
{简历修改偏好}
"""


class ProfileManager:
    """
    管理用户求职画像。
    - 读取 / 写入 PROFILE.md（markdown section 格式）
    - 支持按 section 单独更新（稳定偏好覆盖更新）
    - 支持记录更新时间戳
    """

    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir)
        self.profile_path = self.working_dir / "PROFILE.md"
        self._ensure_profile()

    def _ensure_profile(self) -> None:
        """初始化 PROFILE.md（如不存在则创建默认模板）"""
        if not self.profile_path.exists():
            defaults = {s: "（暂未填写）" for s in PROFILE_SECTIONS}
            defaults["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.profile_path.write_text(
                PROFILE_TEMPLATE.format(**defaults), encoding="utf-8"
            )

    def read(self) -> str:
        """读取完整 profile 内容"""
        return self.profile_path.read_text(encoding="utf-8")

    def read_section(self, section: str) -> str:
        """读取指定 section 的内容"""
        content = self.read()
        pattern = rf"## {re.escape(section)}\n(.*?)(?=\n## |\Z)"
        m = re.search(pattern, content, re.DOTALL)
        return m.group(1).strip() if m else ""

    def update_section(self, section: str, new_content: str) -> bool:
        """
        更新指定 section（稳定偏好覆盖更新策略）。
        同时更新 header 中的最后更新时间。
        """
        if section not in PROFILE_SECTIONS:
            return False

        content = self.read()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 替换 section 内容
        pattern = rf"(## {re.escape(section)}\n)(.*?)(?=\n## |\Z)"
        replacement = rf"\g<1>{new_content.strip()}\n"
        new_content_full = re.sub(pattern, replacement, content, flags=re.DOTALL)

        # 更新最后更新时间
        new_content_full = re.sub(
            r"最后更新: .+", f"最后更新: {now_str}", new_content_full
        )

        self.profile_path.write_text(new_content_full, encoding="utf-8")
        return True

    def update_from_dict(self, updates: dict) -> list[str]:
        """
        批量更新 sections。
        updates = {"目标岗位": "后端工程师", "技术栈": "Python, FastAPI"}
        返回成功更新的 section 列表。
        """
        updated = []
        for section, value in updates.items():
            if self.update_section(section, value):
                updated.append(section)
        return updated

    def to_summary_str(self) -> str:
        """将 profile 转为简洁摘要字符串，注入 system prompt"""
        lines = ["=== 用户求职画像 ==="]
        for section in PROFILE_SECTIONS:
            val = self.read_section(section)
            if val and "暂未填写" not in val:
                lines.append(f"[{section}] {val[:100]}")
        return "\n".join(lines)

    def path_str(self) -> str:
        return str(self.profile_path)