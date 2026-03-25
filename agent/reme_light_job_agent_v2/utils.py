"""
utils.py — 消息格式转换 + 可观测性打印工具
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── ANSI 颜色 ──────────────────────────────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
MAGENTA= "\033[95m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"
RED    = "\033[91m"


def banner(text: str, color: str = CYAN) -> None:
    width = 60
    print(f"\n{color}{BOLD}{'─' * width}")
    print(f"  {text}")
    print(f"{'─' * width}{RESET}")


def info(msg: str) -> None:
    print(f"{DIM}[INFO] {msg}{RESET}")


def warn(msg: str) -> None:
    print(f"{YELLOW}[WARN] {msg}{RESET}")


def show_memory_hit(results: list[dict]) -> None:
    """展示记忆命中情况（可观测性）"""
    if not results:
        print(f"{DIM}  [memory] 本轮无命中记忆{RESET}")
        return
    print(f"{MAGENTA}  [memory] 命中 {len(results)} 条历史记忆:{RESET}")
    for r in results[:3]:
        score = r.get("score", 0)
        source = r.get("source_path", "unknown")
        text   = r.get("text", "")[:80].replace("\n", " ")
        print(f"{DIM}    score={score:.3f} | {Path(source).name}: {text}...{RESET}")


def show_tool_compaction(tool_name: str, original_len: int, compressed_len: int,
                          file_path: Optional[str] = None) -> None:
    """展示工具输出压缩情况（可观测性）"""
    ratio = compressed_len / max(original_len, 1)
    print(f"{YELLOW}  [tool_compact] {tool_name}:{RESET}")
    print(f"    原始长度: {original_len:,} chars")
    print(f"    压缩后  : {compressed_len:,} chars ({ratio:.0%})")
    if file_path:
        print(f"    外置文件: {file_path}")


def show_context_compaction(original_count: int, kept_count: int,
                             summary: str) -> None:
    """展示上下文压缩情况（可观测性）"""
    print(f"{CYAN}  [context_compact] 历史消息已压缩:{RESET}")
    print(f"    压缩消息数: {original_count}")
    print(f"    保留消息数: {kept_count}")
    if summary:
        print(f"    摘要预览: {summary[:120].replace(chr(10), ' ')}...")


# ── Msg 格式转换 ────────────────────────────────────────────────────────────
def openai_to_reme_msg(msg: dict):
    """
    将 OpenAI 格式 dict 转换为 agentscope Msg 对象。
    用于将主对话历史传入 ReMeLight 的各类方法。
    """
    from agentscope.message import Msg

    role    = msg.get("role", "user")
    content = msg.get("content", "")
    name    = msg.get("name", role)

    if role == "tool":
        # 工具结果 → agentscope tool_result block 格式
        # ToolResultCompactor 识别 type=="tool_result" 的 block
        return Msg(
            name="tool",
            role="tool",
            content=[{
                "type": "tool_result",
                "output": content,
                "name": msg.get("name", "tool"),
                "tool_call_id": msg.get("tool_call_id", ""),
            }],
        )
    else:
        return Msg(name=name, role=role, content=content or "")


def openai_msgs_to_reme(messages: list[dict]) -> list:
    """批量转换"""
    return [openai_to_reme_msg(m) for m in messages]


def extract_tool_result_file(content: str) -> Optional[str]:
    """从被截断的工具输出中提取外置文件路径"""
    import re
    m = re.search(r"\[Full content saved to: (.+?)\]", content)
    return m.group(1) if m else None


def estimate_chars(messages: list[dict]) -> int:
    """粗略估算消息总字符数（用于简单的上下文监控）"""
    total = 0
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            for block in c:
                if isinstance(block, dict):
                    total += len(str(block.get("text", "") or block.get("content", "")))
    return total