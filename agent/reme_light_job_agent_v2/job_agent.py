"""
job_agent.py  —  Strata Agent v2（Web 服务版）
================================================

本版本将原命令行 REPL 改造为 Web 后端可调用的异步生成器接口。

核心变化：
  - 去掉 run() REPL 主循环和所有 print/ANSI 输出
  - 会话状态（messages / pending_clarification / compressed_summary）
    从实例变量改为外部传入的 session_state dict（由 Redis 存储）
  - 新增 process_message() 异步生成器，yield 结构化事件供 WebSocket 层广播
  - 新增 process_file() 异步生成器，处理前端上传的已解析文件内容
  - LLM 调用全部改为 stream=True，逐 token yield
  - 所有日志改为 logging 模块，移除 ANSI 彩色输出

事件协议（yield 的 dict）：
  {"type": "token",       "data": str}                         # LLM 流式 token
  {"type": "tool_start",  "data": {"name": str, "args": dict}} # 工具开始执行
  {"type": "tool_end",    "data": {"name": str}}               # 工具执行完成
  {"type": "clarify",     "data": str}                         # 澄清反问文本
  {"type": "memory_hits", "data": list}                        # 记忆命中（供前端展示）
  {"type": "done",        "data": {"has_slots": bool}}         # 本轮处理完成
  {"type": "error",       "data": str}                         # 发生异常

session_state 结构（由调用方从 Redis 读取并传入，处理后回写）：
  {
      "messages":              list[dict],   # OpenAI 格式对话历史
      "pending_clarification": dict | None,  # 挂起的澄清状态
      "compressed_summary":    str,          # LLM 压缩摘要
      "last_memory_hits":      list,         # 最近记忆命中（展示用）
  }
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import AsyncGenerator

REPO_ROOT = Path(__file__).parent.parent.parent
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PACKAGE_NAME = REPO_ROOT.name

import yaml
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError
import httpx

from profile_manager import ProfileManager
from tools import (
    analyze_jd, search_job, generate_resume, mock_interview,
    evaluate_answer, company_research, cover_letter_gen,
    skill_gap_analysis, interview_review, career_path_planner,
    resume_keyword_optimizer,
    TOOL_DEFINITIONS,
)
from skills import (
    SKILL_TOOL_DEFINITIONS,
    FullPreparationSkill, ApplicationPackageSkill, PostInterviewSkill,
)
from file_parser import guess_content_role
from clarification_gate import ClarificationGate
from intent_query_builder import build_intent_sentence_from_query

logger = logging.getLogger("JobAgent")

# ──────────────────────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
你是一个专业的求职助手，拥有长期记忆能力和丰富的工具集。你能帮助用户：
1. 记住并管理他们的求职偏好、技术栈、面试目标
2. 分析岗位 JD，识别核心要求和差距
3. 生成和优化简历，进行 ATS 关键词优化
4. 生成定制化的求职信 / Cover Letter
5. 模拟面试出题并评估回答
6. 深度调研目标公司
7. 面试复盘和知识补齐
8. 职业路径规划
9. 使用 Tavily 搜索公司信息和面试题
10. 分析用户上传的简历、JD 文件或图片

重要规则：
- 如果用户提到了他们的技术栈、目标岗位、薪资期望等偏好信息，请明确确认并记录
- 回答时结合用户的历史偏好（从记忆中获取）给出个性化建议
- 对于复杂的一站式需求，优先使用 Skill 组合工具
- 对于单一需求，使用对应的基础工具

{profile_summary}

{memory_context}
"""

COMPRESSED_SUMMARY_MARKER = "\n\n=== 历史对话摘要（已压缩）===\n{summary}\n=== 摘要结束 ===\n"
CONTEXT_CHAR_THRESHOLD = 8000
MAX_FILE_TEXT_INJECT   = 6000
MAX_TOOL_ROUNDS        = 5


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def make_empty_session_state() -> dict:
    """返回空白 session_state，供后端首次创建会话时使用。"""
    return {
        "messages":              [],
        "pending_clarification": None,
        "compressed_summary":    "",
        "last_memory_hits":      [],
    }


def _extract_block_text(block) -> str:
    if block is None:
        return ""
    if isinstance(block, dict):
        if "text" in block:
            return block.get("text", "")
        if "content" in block:
            c = block.get("content", "")
            return c if isinstance(c, str) else str(c)
        return str(block)
    text = getattr(block, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(block, "content", None)
    if isinstance(content, str):
        return content
    return str(block)


def _reme_msgs_to_openai(reme_msgs: list) -> list[dict]:
    result = []
    for msg in reme_msgs:
        role = msg.role
        content = msg.content
        if isinstance(content, list):
            texts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "tool_result":
                    output = block.get("output", "")
                    if isinstance(output, str):
                        texts.append(output)
                    elif isinstance(output, list):
                        texts.extend(b.get("text", "") for b in output if isinstance(b, dict))
                elif btype == "text":
                    texts.append(block.get("text", ""))
                elif btype in ("image", "audio", "video"):
                    texts.append(f"[{btype}]")
            content = "\n".join(t for t in texts if t)
        result.append({"role": role, "content": str(content)})
    return result


def _estimate_chars(messages: list[dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages)


# ──────────────────────────────────────────────────────────────────────────────
# MCP 工具发现
# ──────────────────────────────────────────────────────────────────────────────

async def discover_mcp_tools(mcp_config_path: str) -> list[dict]:
    config_path = Path(mcp_config_path)
    if not config_path.exists():
        return []
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"加载 MCP 配置失败: {e}")
        return []

    mcp_servers = config.get("mcpServers", {})
    if not mcp_servers:
        return []

    discovered_tools = []
    try:
        from core.utils.mcp_client import MCPClient
    except Exception as e:
        logger.warning(f"导入 MCPClient 失败: {e}")
        return []

    try:
        client = MCPClient({"mcpServers": mcp_servers})
    except Exception as e:
        logger.warning(f"初始化 MCP Client 失败: {e}")
        return []

    for server_name, server_cfg in mcp_servers.items():
        if server_cfg is None:
            continue
        try:
            async with client.connect_to_server(server_name) as session:
                tools_response = await session.list_tools()
                for tool in tools_response.tools:
                    discovered_tools.append({
                        "type": "function",
                        "function": {
                            "name": f"mcp__{server_name}__{tool.name}",
                            "description": f"[MCP:{server_name}] {tool.description or tool.name}",
                            "parameters": tool.inputSchema if hasattr(tool, "inputSchema") else {
                                "type": "object", "properties": {},
                            },
                        },
                    })
                logger.info(f"MCP [{server_name}] 发现 {len(tools_response.tools)} 个工具")
        except Exception as e:
            logger.warning(f"MCP [{server_name}] 连接失败: {e}")

    return discovered_tools


async def call_mcp_tool(tool_name: str, tool_args: dict, mcp_config_path: str) -> str:
    parts = tool_name.split("__", 2)
    if len(parts) != 3:
        return f"无效的 MCP 工具名: {tool_name}"
    _, server_name, actual_tool_name = parts
    with open(Path(mcp_config_path), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    try:
        from core.utils.mcp_client import MCPClient
        from mcp.types import TextContent
        client = MCPClient({"mcpServers": config.get("mcpServers", {})})
        async with client.connect_to_server(server_name) as session:
            result = await session.call_tool(actual_tool_name, tool_args)
            texts = [block.text for block in result.content if isinstance(block, TextContent)]
            return "\n".join(texts) if texts else str(result)
    except Exception as e:
        return f"MCP 工具调用失败: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# 核心 Agent 类
# ──────────────────────────────────────────────────────────────────────────────

class JobAgentV2:
    """
    Strata 求职助手 Agent（Web 服务版）。

    实例按 user_id 隔离创建（通过 create_agent() 工厂函数）。
    所有会话状态从外部 session_state dict 传入，不存在实例变量中。
    """

    def __init__(
        self,
        working_dir:           str        = ".job_agent_v2",
        model:                 str        = "qwen3.5-plus",
        vision_model:          str | None = None,
        api_key:               str | None = None,
        base_url:              str | None = None,
        embedding_api_key:     str | None = None,
        embedding_base_url:    str | None = None,
        tavily_api_key:        str | None = None,
        tool_result_threshold: int        = 800,
        language:              str        = "zh",
        mcp_config_path:       str | None = None,
        enable_mcp_discovery:  bool       = True,
    ) -> None:
        self.working_dir           = Path(working_dir).absolute()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.model                 = model
        self.vision_model          = vision_model or model
        self.language              = language
        self.tool_result_threshold = tool_result_threshold
        self.tavily_api_key        = tavily_api_key
        self.enable_mcp_discovery  = enable_mcp_discovery
        self.mcp_config_path       = mcp_config_path or str(Path(__file__).parent / "mcp_config.yaml")

        real_api_key = api_key or os.environ.get("LLM_API_KEY", "")
        real_base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")

        self.llm = AsyncOpenAI(
            api_key=real_api_key,
            base_url=real_base_url,
            timeout=120.0,
            max_retries=2,
            http_client=httpx.AsyncClient(
                timeout=120.0,
                trust_env=False,
            ),
        )

        reme_mod  = importlib.import_module(f"{PACKAGE_NAME}.reme_light")
        ReMeLight = reme_mod.ReMeLight
        self.reme = ReMeLight(
            working_dir=str(self.working_dir),
            llm_api_key=api_key or os.environ.get("LLM_API_KEY", ""),
            llm_base_url=base_url or os.environ.get("LLM_BASE_URL"),
            embedding_api_key=embedding_api_key or os.environ.get("EMBEDDING_API_KEY"),
            embedding_base_url=embedding_base_url or os.environ.get("EMBEDDING_BASE_URL"),
            tool_result_threshold=tool_result_threshold,
            default_as_llm_config={"backend": "openai", "model_name": model},
        )

        self.profile               = ProfileManager(working_dir=str(self.working_dir))
        self.gate                  = ClarificationGate(llm_client=self.llm, model=self.model, reme=self.reme)
        self.all_tool_definitions: list[dict] = TOOL_DEFINITIONS + SKILL_TOOL_DEFINITIONS
        self.mcp_tool_names:       set[str]   = set()

        logger.info(
            f"JobAgentV2 初始化 | dir={self.working_dir} "
            f"model={self.model} base_url={real_base_url} "
            f"has_api_key={bool(real_api_key)}"
        )

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """启动 ReMeLight 服务，发现 MCP 工具。由后端 lifespan 调用一次。"""
        await self.reme.start()
        logger.info("ReMeLight 服务已启动")
        if self.enable_mcp_discovery:
            await self._discover_mcp_tools()

    async def close(self) -> None:
        """优雅关闭，等待后台摘要任务。由后端 lifespan 调用。"""
        await self.reme.await_summary_tasks()
        await self.reme.close()
        logger.info("JobAgentV2 已关闭")

    async def _discover_mcp_tools(self) -> None:
        try:
            mcp_tools = await discover_mcp_tools(self.mcp_config_path)
            for t in mcp_tools:
                self.all_tool_definitions.append(t)
                self.mcp_tool_names.add(t["function"]["name"])
            logger.info(f"MCP 工具发现完成，共 {len(mcp_tools)} 个")
        except Exception as e:
            logger.warning(f"MCP 工具发现失败: {e}")

    # ── 公开接口：对话处理 ────────────────────────────────────────────────────

    async def process_message(
        self,
        user_input:    str,
        session_state: dict,
        image_data:    dict | None = None,
    ) -> AsyncGenerator[dict, None]:
        """
        处理一条用户消息，异步生成事件流。

        session_state 在函数执行过程中被就地修改（messages 追加等），
        生成器耗尽后由调用方（backend/chat/agent_adapter.py）负责将修改写回 Redis。

        Yields:
            token       — LLM 输出的每个文本片段
            clarify     — 澄清反问的完整问题文本
            tool_start  — 工具开始执行（含工具名和参数）
            tool_end    — 工具执行完成（含工具名）
            memory_hits — 本轮检索命中的记忆列表
            done        — 本轮处理完成
            error       — 发生不可恢复的异常
        """
        session_state.setdefault("messages",              [])
        session_state.setdefault("pending_clarification", None)
        session_state.setdefault("compressed_summary",    "")
        session_state.setdefault("last_memory_hits",      [])

        try:
            # ── Step 1: 澄清门控 ────────────────────────────────────────────
            is_clarification_response = session_state["pending_clarification"] is not None
            gate_result = None

            if not is_clarification_response:
                if ClarificationGate.should_force_proceed(session_state["pending_clarification"]):
                    logger.info("已达最大追问次数，强制放行")
                    session_state["pending_clarification"] = None
                else:
                    profile_summary = self.profile.to_summary_str()
                    gate_result = await self.gate.clarify_gate(
                        user_input=user_input,
                        recent_messages=session_state["messages"][-4:],
                        profile_summary=profile_summary,
                    )

                    if gate_result.is_ambiguous:
                        ask_count = 1
                        if session_state["pending_clarification"] is not None:
                            ask_count = session_state["pending_clarification"].get("ask_count", 0) + 1

                        if ask_count > 2:
                            logger.info("已达最大追问次数，强制放行")
                            session_state["pending_clarification"] = None
                        else:
                            question = ClarificationGate.generate_clarify_question(gate_result, user_input)
                            session_state["pending_clarification"] = {
                                "original_input":   user_input,
                                "ambiguity_type":   gate_result.ambiguity_type,
                                "missing_slots":    gate_result.missing_slots,
                                "clarify_question": question,
                                "ask_count":        ask_count,
                            }
                            session_state["messages"].append({"role": "user",      "content": user_input})
                            session_state["messages"].append({"role": "assistant", "content": question})
                            yield {"type": "clarify", "data": question}
                            yield {"type": "done",    "data": {"has_slots": False}}
                            return

            # ── Step 2: 确定 Query 和澄清补充 ───────────────────────────────
            if is_clarification_response and session_state["pending_clarification"]:
                original_query       = session_state["pending_clarification"].get("original_input", user_input)
                clarification_answer = user_input
            else:
                original_query       = user_input
                clarification_answer = ""

            # ── Step 3: 规则层槽位提取（替代 LLM normalize_query，省 ~600ms）──
            inferred_slots = getattr(gate_result, "inferred_slots", {}) if gate_result is not None else {}
            intent_sentence, slots, has_slots = build_intent_sentence_from_query(
                query_text=original_query,
                clarification_answer=clarification_answer,
                inferred_slots=inferred_slots,
            )
            logger.info(f"意图标准句: {intent_sentence[:80]}  has_slots={has_slots}")
            session_state["pending_clarification"] = None

            # ── Step 4: 流式对话 ─────────────────────────────────────────────
            async for event in self._stream_with_tools(
                user_input=user_input,
                session_state=session_state,
                intent_sentence=intent_sentence if has_slots else None,
                image_data=image_data,
            ):
                yield event

            # ── Step 5: 异步偏好写入 ─────────────────────────────────────────
            if has_slots:
                profile_context = (
                    f"用户: {original_query}"
                    + (f" {clarification_answer}" if clarification_answer else "")
                    + f"\n助手: {self._last_assistant_text(session_state)}"
                )
                asyncio.create_task(self._async_extract_profile(profile_context))
            else:
                logger.info("槽位不完整，跳过本轮偏好写入")

            yield {"type": "done", "data": {"has_slots": has_slots}}

        except Exception as e:
            logger.exception(f"process_message 异常: {e}")
            yield {"type": "error", "data": str(e)}

    async def process_file(
        self,
        file_name:     str,
        file_type:     str,
        file_content:  str | None,
        session_state: dict,
        image_data:    dict | None = None,
        user_hint:     str         = "",
    ) -> AsyncGenerator[dict, None]:
        """
        处理前端上传的已解析文件，注入对话后流式响应。

        文件的实际解析（PDF/Word/图片）由 FastAPI /files/upload 端点完成，
        解析结果（text 或 image_data）传给本函数。

        Args:
            file_name:    原始文件名
            file_type:    文件类型（pdf / docx / jpg 等）
            file_content: 文本类文件的解析文本（图片类传 None）
            session_state:会话状态
            image_data:   图片类文件的 base64 数据 dict
            user_hint:    用户对该文件的额外说明（可为空）
        """
        session_state.setdefault("messages",              [])
        session_state.setdefault("pending_clarification", None)
        session_state.setdefault("compressed_summary",    "")
        session_state.setdefault("last_memory_hits",      [])

        try:
            if image_data:
                prompt = user_hint or (
                    "请仔细分析这张图片。如果是简历，请提取关键信息并给出优化建议。"
                    "如果是 JD，请分析岗位要求和技术栈。"
                )
                async for event in self._stream_with_tools(
                    user_input=prompt,
                    session_state=session_state,
                    intent_sentence=None,
                    image_data=image_data,
                ):
                    yield event
                asyncio.create_task(
                    self._async_extract_profile(
                        f"[图片分析: {file_name}]\n{self._last_assistant_text(session_state)}"
                    )
                )
                yield {"type": "done", "data": {"has_slots": False}}
                return

            if not file_content:
                yield {"type": "error", "data": "文件内容为空，无法分析"}
                return

            # 超长截断
            truncated      = len(file_content) > MAX_FILE_TEXT_INJECT
            text_to_inject = file_content[:MAX_FILE_TEXT_INJECT] if truncated else file_content

            content_role = guess_content_role(file_content)
            if user_hint:
                analysis_request = user_hint
            elif content_role == "jd":
                analysis_request = (
                    "请详细分析这份 JD，包括：\n"
                    "1. 核心技术要求\n2. 软性能力要求\n"
                    "3. 结合我的背景给出差距分析\n4. 面试准备重点"
                )
            elif content_role == "resume":
                analysis_request = (
                    "请分析这份简历，包括：\n"
                    "1. 优势和亮点\n2. 可以改进的地方\n3. 优化建议"
                )
            else:
                analysis_request = "请分析这份文档的内容，提取与求职相关的关键信息。"

            truncation_note = (
                f"\n\n（注：文档内容较长，以下为前 {MAX_FILE_TEXT_INJECT:,} 字符）"
                if truncated else ""
            )
            injected_message = (
                f"我上传了一份文件：{file_name}\n"
                f"文件类型：{file_type.upper()}，共 {len(file_content):,} 字符"
                f"{truncation_note}\n\n"
                f"--- 文件内容开始 ---\n{text_to_inject}\n--- 文件内容结束 ---\n\n"
                f"{analysis_request}"
            )

            async for event in self._stream_with_tools(
                user_input=injected_message,
                session_state=session_state,
                intent_sentence=None,
                image_data=None,
            ):
                yield event

            asyncio.create_task(
                self._async_extract_profile(
                    f"[文件分析: {file_name}]\n{self._last_assistant_text(session_state)}"
                )
            )
            yield {"type": "done", "data": {"has_slots": False}}

        except Exception as e:
            logger.exception(f"process_file 异常: {e}")
            yield {"type": "error", "data": str(e)}

    # ── 公开接口：画像（供 REST API 使用）────────────────────────────────────

    def get_profile_dict(self) -> dict:
        """返回当前用户画像字典。"""
        try:
            return self.profile.read() or {}
        except Exception:
            return {}

    def update_profile_dict(self, updates: dict) -> list[str]:
        """更新画像字段，返回实际更新的字段名列表。"""
        try:
            return self.profile.update_from_dict(updates) or []
        except Exception as e:
            logger.warning(f"profile 更新失败: {e}")
            return []

    # ── 私有：流式对话 + 工具调用核心 ────────────────────────────────────────

    async def _stream_with_tools(
        self,
        user_input:      str,
        session_state:   dict,
        intent_sentence: str | None,
        image_data:      dict | None,
    ) -> AsyncGenerator[dict, None]:
        """
        核心流式对话循环：
          记忆检索 → 构建 system prompt → 上下文压缩
          → 循环调用 LLM（stream=True）
          → 逐 token yield / 工具调用 yield
        """
        # 记忆检索
        search_query = intent_sentence if intent_sentence else user_input
        memory_context, hits = await self._search_memory(search_query)
        session_state["last_memory_hits"] = hits
        if hits:
            yield {"type": "memory_hits", "data": hits}

        # System prompt
        profile_summary = self.profile.to_summary_str()
        system_content  = SYSTEM_PROMPT_TEMPLATE.format(
            profile_summary=(profile_summary if "暂未填写" not in profile_summary else ""),
            memory_context=memory_context,
        )
        if session_state["compressed_summary"]:
            system_content += COMPRESSED_SUMMARY_MARKER.format(
                summary=session_state["compressed_summary"]
            )

        # 上下文压缩
        await self._maybe_compact_context(session_state)

        # 追加用户消息到历史
        history_content = (
            f"[上传图片: {image_data['file_name']}]\n{user_input}"
            if image_data else user_input
        )
        session_state["messages"].append({"role": "user", "content": history_content})

        # 构建请求消息列表
        if image_data:
            current_user_msg = {
                "role": "user",
                "content": [
                    {"type": "text",      "text": user_input},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{image_data['media_type']};base64,{image_data['base64']}"
                    }},
                ],
            }
            request_messages = (
                [{"role": "system", "content": system_content}]
                + session_state["messages"][:-1]
                + [current_user_msg]
            )
        else:
            request_messages = (
                [{"role": "system", "content": system_content}]
                + session_state["messages"]
            )

        # 流式对话 + 工具调用循环
        for _round in range(MAX_TOOL_ROUNDS):
            accumulated_content    = ""
            # index → {id, name, arguments}
            accumulated_tool_calls: dict[int, dict] = {}
            finish_reason          = None

            try:
                stream = await self.llm.chat.completions.create(
                    model=self.model,
                    messages=request_messages,
                    tools=self.all_tool_definitions,
                    tool_choice="auto",
                    temperature=0.7,
                    stream=True,
                )
            except APIConnectionError as e:
                logger.exception(f"LLM 连接失败: {e}")
                yield {
                    "type": "error",
                    "data": "无法连接到大模型服务。请检查当前网络是否能访问 LLM_BASE_URL；如果你本机开了代理软件但 Python 没走代理，或系统里残留了错误代理，也会出现这个错误。"
                }
                return
            except APITimeoutError as e:
                logger.exception(f"LLM 请求超时: {e}")
                yield {
                    "type": "error",
                    "data": "大模型请求超时，请稍后重试。"
                }
                return

            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    continue
                finish_reason = choice.finish_reason
                delta         = choice.delta

                # 文本 token
                if delta.content:
                    accumulated_content += delta.content
                    yield {"type": "token", "data": delta.content}

                # 工具调用 chunk 累积
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc_delta.id:
                            accumulated_tool_calls[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                accumulated_tool_calls[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                accumulated_tool_calls[idx]["arguments"] += tc_delta.function.arguments

            # 处理本轮结果
            if not accumulated_tool_calls or finish_reason == "stop":
                # 纯文本回复，本轮结束
                session_state["messages"].append(
                    {"role": "assistant", "content": accumulated_content}
                )
                return

            # 有工具调用：记录 assistant 消息，执行工具，追加结果，继续下一轮
            tool_calls_for_msg = [
                {
                    "id": tc["id"], "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in accumulated_tool_calls.values()
            ]
            assistant_msg = {
                "role": "assistant",
                "content": accumulated_content,
                "tool_calls": tool_calls_for_msg,
            }
            session_state["messages"].append(assistant_msg)
            request_messages.append(assistant_msg)

            for tc in accumulated_tool_calls.values():
                tool_name = tc["name"]
                tool_id   = tc["id"]
                try:
                    tool_args = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    tool_args = {}

                yield {"type": "tool_start", "data": {"name": tool_name, "args": tool_args}}
                tool_output = await self._execute_tool(tool_name, tool_args, tool_id)
                yield {"type": "tool_end",   "data": {"name": tool_name}}

                tool_msg = {
                    "role": "tool", "tool_call_id": tool_id,
                    "name": tool_name, "content": tool_output,
                }
                session_state["messages"].append(tool_msg)
                request_messages.append(tool_msg)

        # 超过最大轮次
        overflow_msg = "（工具调用轮次已达上限，请重新提问）"
        session_state["messages"].append({"role": "assistant", "content": overflow_msg})
        yield {"type": "token", "data": overflow_msg}

    # ── 私有：记忆检索 ────────────────────────────────────────────────────────

    async def _search_memory(self, query: str) -> tuple[str, list]:
        try:
            tool_response = await self.reme.memory_search(
                query=query, max_results=4, min_score=0.1
            )
            raw = (
                _extract_block_text(tool_response.content[0])
                if getattr(tool_response, "content", None)
                else "[]"
            )
            results = json.loads(raw)
            if not results:
                return "", []

            lines = ["=== 相关历史记忆 ==="]
            for r in results[:3]:
                score = r.get("score", 0)
                text  = r.get("text", "").strip()
                src   = Path(r.get("source_path", "")).name
                lines.append(f"[{src} | score={score:.2f}] {text[:200]}")
            lines.append("=== 记忆结束 ===")
            return "\n".join(lines), results

        except Exception as e:
            logger.warning(f"memory_search 失败: {type(e).__name__}: {e!r}")
            return "", []

    # ── 私有：上下文压缩 ──────────────────────────────────────────────────────

    async def _maybe_compact_context(self, session_state: dict) -> None:
        messages = session_state["messages"]
        if not messages or _estimate_chars(messages) < CONTEXT_CHAR_THRESHOLD:
            return
        logger.info(f"触发上下文压缩，当前 {_estimate_chars(messages):,} 字符")
        try:
            from utils import openai_msgs_to_reme
            reme_msgs = openai_msgs_to_reme(messages)
        except Exception:
            return
        try:
            kept_msgs, new_summary = await self.reme.pre_reasoning_hook(
                messages=reme_msgs,
                system_prompt=SYSTEM_PROMPT_TEMPLATE,
                compressed_summary=session_state["compressed_summary"],
                language=self.language,
                max_input_length=128 * 1024,
                compact_ratio=0.7,
                memory_compact_reserve=10000,
                enable_tool_result_compact=True,
                tool_result_compact_keep_n=3,
            )
            if len(kept_msgs) < len(reme_msgs):
                session_state["messages"]           = _reme_msgs_to_openai(kept_msgs)
                session_state["compressed_summary"] = new_summary
                logger.info(f"上下文压缩: {len(reme_msgs)} → {len(kept_msgs)} 条")
        except Exception as e:
            logger.warning(f"pre_reasoning_hook 失败: {e}")

    # ── 私有：工具执行（保留完整工具分发逻辑）───────────────────────────────

    async def _execute_tool(self, tool_name: str, tool_args: dict, tool_call_id: str) -> str:
        logger.info(f"执行工具: {tool_name}  args={json.dumps(tool_args, ensure_ascii=False)[:80]}")
        profile_summary = self.profile.to_summary_str()
        raw_output      = ""

        try:
            if tool_name in self.mcp_tool_names:
                raw_output = await call_mcp_tool(tool_name, tool_args, self.mcp_config_path)

            elif tool_name == "full_preparation_skill":
                skill = FullPreparationSkill(
                    llm_client=self.llm, model=self.model,
                    on_step=lambda n, s: logger.info(f"Skill: {n} → {s}"),
                )
                raw_output = json.dumps(await skill.execute(
                    jd_text=tool_args.get("jd_text", ""),
                    profile_summary=profile_summary,
                    company_name=tool_args.get("company_name", ""),
                    tavily_api_key=self.tavily_api_key or "",
                ), ensure_ascii=False, indent=2)

            elif tool_name == "application_package_skill":
                skill = ApplicationPackageSkill(
                    llm_client=self.llm, model=self.model,
                    on_step=lambda n, s: logger.info(f"Skill: {n} → {s}"),
                )
                raw_output = json.dumps(await skill.execute(
                    jd_text=tool_args.get("jd_text", ""),
                    profile_summary=profile_summary,
                    existing_resume=tool_args.get("existing_resume", ""),
                    company_name=tool_args.get("company_name", ""),
                    language=tool_args.get("language", self.language),
                ), ensure_ascii=False, indent=2)

            elif tool_name == "post_interview_skill":
                skill = PostInterviewSkill(
                    llm_client=self.llm, model=self.model,
                    on_step=lambda n, s: logger.info(f"Skill: {n} → {s}"),
                )
                raw_output = json.dumps(await skill.execute(
                    interview_description=tool_args.get("interview_description", ""),
                    company_and_position=tool_args.get("company_and_position", ""),
                    tavily_api_key=self.tavily_api_key or "",
                ), ensure_ascii=False, indent=2)

            elif tool_name == "analyze_jd":
                raw_output = json.dumps(await analyze_jd(
                    jd_text=tool_args.get("jd_text", ""),
                    llm_client=self.llm, model=self.model,
                ), ensure_ascii=False, indent=2)

            elif tool_name == "search_job":
                raw_output = json.dumps(await search_job(
                    query=tool_args.get("query", ""),
                    tavily_api_key=self.tavily_api_key or "",
                    max_results=tool_args.get("max_results", 5),
                    search_depth=tool_args.get("search_depth", "basic"),
                ), ensure_ascii=False, indent=2)

            elif tool_name == "generate_resume":
                raw_output = json.dumps(await generate_resume(
                    profile_summary=profile_summary,
                    llm_client=self.llm, model=self.model,
                    jd_text=tool_args.get("jd_text", ""),
                    existing_resume=tool_args.get("existing_resume", ""),
                ), ensure_ascii=False, indent=2)

            elif tool_name == "mock_interview":
                raw_output = json.dumps(await mock_interview(
                    position=tool_args.get("position", ""),
                    llm_client=self.llm, model=self.model,
                    interview_type=tool_args.get("interview_type", "技术面"),
                    difficulty=tool_args.get("difficulty", "中等"),
                    tech_stack=tool_args.get("tech_stack", ""),
                    question_count=tool_args.get("question_count", 8),
                ), ensure_ascii=False, indent=2)

            elif tool_name == "evaluate_answer":
                raw_output = json.dumps(await evaluate_answer(
                    question=tool_args.get("question", ""),
                    answer=tool_args.get("answer", ""),
                    llm_client=self.llm, model=self.model,
                    position=tool_args.get("position", ""),
                ), ensure_ascii=False, indent=2)

            elif tool_name == "company_research":
                raw_output = json.dumps(await company_research(
                    company_name=tool_args.get("company_name", ""),
                    llm_client=self.llm, model=self.model,
                    tavily_api_key=self.tavily_api_key or "",
                ), ensure_ascii=False, indent=2)

            elif tool_name == "cover_letter_gen":
                raw_output = json.dumps(await cover_letter_gen(
                    jd_text=tool_args.get("jd_text", ""),
                    llm_client=self.llm, model=self.model,
                    profile_summary=profile_summary,
                    company_name=tool_args.get("company_name", ""),
                    language=tool_args.get("language", self.language),
                ), ensure_ascii=False, indent=2)

            elif tool_name == "skill_gap_analysis":
                raw_output = json.dumps(await skill_gap_analysis(
                    jd_text=tool_args.get("jd_text", ""),
                    llm_client=self.llm, model=self.model,
                    profile_summary=profile_summary,
                    resume_info=tool_args.get("resume_info", ""),
                ), ensure_ascii=False, indent=2)

            elif tool_name == "interview_review":
                raw_output = json.dumps(await interview_review(
                    interview_description=tool_args.get("interview_description", ""),
                    llm_client=self.llm, model=self.model,
                    company_and_position=tool_args.get("company_and_position", ""),
                ), ensure_ascii=False, indent=2)

            elif tool_name == "career_path_planner":
                raw_output = json.dumps(await career_path_planner(
                    llm_client=self.llm, model=self.model,
                    profile_summary=profile_summary,
                    current_stage=tool_args.get("current_stage", ""),
                    target_direction=tool_args.get("target_direction", ""),
                ), ensure_ascii=False, indent=2)

            elif tool_name == "resume_keyword_optimizer":
                raw_output = json.dumps(await resume_keyword_optimizer(
                    jd_text=tool_args.get("jd_text", ""),
                    resume_text=tool_args.get("resume_text", ""),
                    llm_client=self.llm, model=self.model,
                ), ensure_ascii=False, indent=2)

            else:
                raw_output = f"未知工具: {tool_name}"

        except Exception as e:
            logger.exception(f"工具 {tool_name} 执行异常: {e}")
            raw_output = f"工具执行失败: {e}"

        # 工具输出压缩
        original_len = len(raw_output)
        try:
            from agentscope.message import Msg
            tool_msg = Msg(
                name=tool_name, role="assistant",
                content=[{
                    "type": "tool_result", "output": raw_output,
                    "name": tool_name, "tool_call_id": tool_call_id,
                }],
            )
            compacted_msgs = await self.reme.compact_tool_result(messages=[tool_msg])
            compacted_msg  = compacted_msgs[0] if compacted_msgs else tool_msg
            if isinstance(compacted_msg.content, list) and compacted_msg.content:
                block      = compacted_msg.content[0]
                output_str = block.get("output", raw_output) if isinstance(block, dict) else str(block)
            else:
                output_str = str(compacted_msg.content)
        except Exception as e:
            logger.warning(f"compact_tool_result 失败: {e}")
            output_str = raw_output

        logger.info(f"工具 {tool_name} 完成: {original_len} → {len(output_str)} chars")
        return output_str

    # ── 私有：异步画像提取 ────────────────────────────────────────────────────

    async def _async_extract_profile(self, conversation: str) -> None:
        extract_prompt = f"""\
从以下对话片段中提取用户的求职偏好信息。
只返回 JSON，不要其他文字。如某项没有提及，值为 null。

字段：
- 目标岗位: 用户想找什么岗位
- 目标城市: 目标工作城市
- 技术栈: 用户掌握或提到的技术
- 目标公司_行业: 目标公司或行业
- 薪资预期: 期望薪资
- 面试薄弱点: 用户认为自己薄弱的方向
- 简历修改偏好: 简历风格或修改偏好

对话：
{conversation}

JSON："""
        try:
            resp = await self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": extract_prompt}],
                temperature=0,
                max_tokens=512,
            )
            raw  = resp.choices[0].message.content.strip()
            raw  = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(raw)
            updates = {k: v for k, v in data.items() if v and k in self.profile.read()}
            if updates:
                updated = self.profile.update_from_dict(updates)
                if updated:
                    logger.info(f"画像已更新: {updated}")
        except Exception:
            pass

    # ── 私有：辅助 ───────────────────────────────────────────────────────────

    @staticmethod
    def _last_assistant_text(session_state: dict) -> str:
        for msg in reversed(session_state.get("messages", [])):
            if msg.get("role") == "assistant":
                return str(msg.get("content", ""))
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# 工厂函数（供 backend/chat/agent_adapter.py 调用）
# ──────────────────────────────────────────────────────────────────────────────

def create_agent(user_id: str, data_root: str = "data", **kwargs) -> JobAgentV2:
    """
    按 user_id 创建隔离的 Agent 实例。

    working_dir = data/{user_id}/agent，保证每个用户的记忆库和画像完全隔离。
    kwargs 透传给 JobAgentV2.__init__（model / api_key / tavily_api_key 等）。
    """
    working_dir = str(Path(data_root) / user_id / "agent")
    return JobAgentV2(working_dir=working_dir, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# CLI 入口（开发调试用，演示如何驱动新接口）
# ──────────────────────────────────────────────────────────────────────────────

async def _cli_main() -> None:
    """轻量 CLI 测试入口。生产环境由 FastAPI WebSocket 层调用，不使用此函数。"""
    import dotenv
    dotenv.load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    agent = JobAgentV2(
        working_dir=os.environ.get("AGENT_WORKING_DIR", ".job_agent_v2"),
        model=os.environ.get("LLM_MODEL", "qwen-plus"),
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
        tavily_api_key=os.environ.get("TAVILY_API_KEY"),
    )
    await agent.start()

    session_state = make_empty_session_state()
    print("Strata Agent CLI（输入 quit 退出）\n")

    while True:
        try:
            user_input = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("quit", "exit"):
            break

        print("助手 > ", end="", flush=True)
        async for event in agent.process_message(user_input, session_state):
            t = event["type"]
            if t == "token":
                print(event["data"], end="", flush=True)
            elif t == "clarify":
                print(f"\n[澄清] {event['data']}")
                break
            elif t == "tool_start":
                print(f"\n  ▶ 工具: {event['data']['name']}", flush=True)
            elif t == "tool_end":
                print(f"  ✓ 完成: {event['data']['name']}")
            elif t == "error":
                print(f"\n[错误] {event['data']}")
            elif t == "done":
                print()

    await agent.close()


if __name__ == "__main__":
    asyncio.run(_cli_main())
