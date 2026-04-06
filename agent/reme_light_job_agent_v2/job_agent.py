import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from openai import AsyncOpenAI

from profile_manager import ProfileManager
from tools import (
    analyze_jd, search_job, generate_resume, mock_interview,
    evaluate_answer, company_research, cover_letter_gen,
    skill_gap_analysis, interview_review, career_path_planner,
    resume_keyword_optimizer,
    TOOL_DEFINITIONS,
)
from skills import (
    SKILL_REGISTRY, SKILL_TOOL_DEFINITIONS,
    FullPreparationSkill, ApplicationPackageSkill, PostInterviewSkill,
)
from file_parser import parse_file, guess_content_role, get_supported_formats, FileParseResult
from clarification_gate import ClarificationGate
from utils import (
    banner, info, warn,
    show_memory_hit, show_tool_compaction, show_context_compaction,
    openai_msgs_to_reme, extract_tool_result_file, estimate_chars,
    CYAN, GREEN, YELLOW, MAGENTA, BOLD, RESET, DIM, RED,
)


# ── MCP Client 工具发现 ──────────────────────────────────────────────────────

async def discover_mcp_tools(mcp_config_path: str) -> list[dict]:
    """
    连接 mcp_config.yaml 中声明的 MCP Server，
    自动发现并返回 OpenAI function calling 格式的工具定义列表。
    """
    config_path = Path(mcp_config_path)
    if not config_path.exists():
        return []

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        warn(f"加载 MCP 配置失败: {e}")
        return []

    mcp_servers = config.get("mcpServers", {})
    if not mcp_servers:
        return []

    discovered_tools = []
    from core.utils.mcp_client import MCPClient

    try:
        client = MCPClient({"mcpServers": mcp_servers})
    except Exception as e:
        warn(f"初始化 MCP Client 失败: {e}")
        return []

    for server_name, server_cfg in mcp_servers.items():
        # 跳过注释掉的 server（值为 None）
        if server_cfg is None:
            continue
        try:
            async with client.connect_to_server(server_name) as session:
                tools_response = await session.list_tools()
                for tool in tools_response.tools:
                    # 转换 MCP tool schema 到 OpenAI function calling 格式
                    tool_def = {
                        "type": "function",
                        "function": {
                            "name": f"mcp__{server_name}__{tool.name}",
                            "description": f"[MCP:{server_name}] {tool.description or tool.name}",
                            "parameters": tool.inputSchema if hasattr(tool, "inputSchema") else {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    }
                    discovered_tools.append(tool_def)
                info(f"MCP [{server_name}] 发现 {len(tools_response.tools)} 个工具")
        except Exception as e:
            warn(f"MCP [{server_name}] 连接失败: {e}")

    return discovered_tools


async def call_mcp_tool(
    tool_name: str, tool_args: dict, mcp_config_path: str
) -> str:
    """
    调用 MCP 工具。tool_name 格式: mcp__{server_name}__{actual_tool_name}
    """
    parts = tool_name.split("__", 2)
    if len(parts) != 3:
        return f"无效的 MCP 工具名: {tool_name}"

    _, server_name, actual_tool_name = parts

    config_path = Path(mcp_config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    from core.utils.mcp_client import MCPClient

    try:
        client = MCPClient({"mcpServers": config.get("mcpServers", {})})
        async with client.connect_to_server(server_name) as session:
            result = await session.call_tool(actual_tool_name, tool_args)
            # 提取文本结果
            from mcp.types import TextContent
            texts = []
            for block in result.content:
                if isinstance(block, TextContent):
                    texts.append(block.text)
            return "\n".join(texts) if texts else str(result)
    except Exception as e:
        return f"MCP 工具调用失败: {e}"


# ── 系统提示词 ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """你是一个专业的求职助手，拥有长期记忆能力和丰富的工具集。你能帮助用户：
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
- 对于复杂的一站式需求，优先使用 Skill 组合工具（如 full_preparation_skill、application_package_skill、post_interview_skill）
- 对于单一需求，使用对应的基础工具

{profile_summary}

{memory_context}
"""

COMPRESSED_SUMMARY_MARKER = "\n\n=== 历史对话摘要（已压缩）===\n{summary}\n=== 摘要结束 ===\n"

CONTEXT_CHAR_THRESHOLD = 8000
MAX_FILE_TEXT_INJECT = 6000


class JobAgentV2:
    """
    ReMeLight 求职助手 Agent v2（升级版）

    升级亮点：
    - 11 个求职专用工具（基础工具）
    - 3 个 Agent Skills（组合工具链）
    - MCP 协议支持（可连接外部 MCP Server，也可自身作为 MCP Server 暴露）
    - 请求澄清门控（ClarificationGate）
    - 标准化 Query（normalize_query）
    - 槽位完整度校验
    """

    def __init__(
        self,
        working_dir: str = ".job_agent_v2",
        model: str = "qwen-plus",
        vision_model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        tavily_api_key: str | None = None,
        tool_result_threshold: int = 800,
        language: str = "zh",
        mcp_config_path: str | None = None,
        enable_mcp_discovery: bool = True,
    ):
        self.working_dir = Path(working_dir).absolute()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.vision_model = vision_model or model
        self.language = language
        self.tool_result_threshold = tool_result_threshold
        self.tavily_api_key = tavily_api_key
        self.enable_mcp_discovery = enable_mcp_discovery

        # MCP 配置路径
        self.mcp_config_path = mcp_config_path or str(
            Path(__file__).parent / "mcp_config.yaml"
        )

        # OpenAI 兼容客户端
        self.llm = AsyncOpenAI(
            api_key=api_key or os.environ.get("LLM_API_KEY", ""),
            base_url=base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        )

        # ReMeLight
        from reme_light import ReMeLight
        self.reme = ReMeLight(
            working_dir=str(self.working_dir),
            llm_api_key=api_key or os.environ.get("LLM_API_KEY", ""),
            llm_base_url=base_url or os.environ.get("LLM_BASE_URL"),
            embedding_api_key=embedding_api_key or os.environ.get("EMBEDDING_API_KEY"),
            embedding_base_url=embedding_base_url or os.environ.get("EMBEDDING_BASE_URL"),
            tool_result_threshold=tool_result_threshold,
            default_as_llm_config={
                "backend": "openai",
                "model_name": model,
            },
        )

        # Profile 管理器
        self.profile = ProfileManager(working_dir=str(self.working_dir))

        # 澄清门控
        self.gate = ClarificationGate(
            llm_client=self.llm,
            model=self.model,
            reme=self.reme,
        )

        # 对话历史
        self.messages: list[dict] = []
        self.compressed_summary: str = ""
        self.pending_clarification: dict | None = None
        self.last_memory_hits: list[dict] = []

        # 工具定义列表（基础工具 + Skills + MCP 动态发现）
        self.all_tool_definitions: list[dict] = TOOL_DEFINITIONS + SKILL_TOOL_DEFINITIONS
        self.mcp_tool_names: set[str] = set()  # 记录 MCP 工具名

        # 初始化信息
        print(f"\n{CYAN}{BOLD}🧠 ReMeLight 求职助手 Agent v2（升级版）已初始化{RESET}")
        print(f"{DIM}工作目录: {self.working_dir}{RESET}")
        print(f"{DIM}文本模型: {self.model}{RESET}")
        print(f"{DIM}视觉模型: {self.vision_model}{RESET}")
        print(f"{GREEN}✅ 基础工具: 11 个{RESET}")
        print(f"{GREEN}✅ Agent Skills: 3 个（full_preparation / application_package / post_interview）{RESET}")
        print(f"{GREEN}✅ 澄清门控: 已启用{RESET}")

        if self.tavily_api_key:
            print(f"{GREEN}✅ Tavily 联网搜索: 已启用{RESET}")
        else:
            print(f"{YELLOW}⚠️  Tavily 联网搜索: 未配置{RESET}")

        if self.enable_mcp_discovery:
            print(f"{GREEN}✅ MCP 工具发现: 已启用（{self.mcp_config_path}）{RESET}")
        else:
            print(f"{DIM}   MCP 工具发现: 已禁用{RESET}")

        print(f"{GREEN}✅ 文件输入: PDF / Word / 图片 已支持{RESET}\n")

    # ── 启动与关闭 ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await self.reme.start()
        info("ReMeLight 服务已启动")

        # MCP 工具发现
        if self.enable_mcp_discovery:
            await self._discover_mcp_tools()

    async def _discover_mcp_tools(self) -> None:
        """动态发现 MCP Server 暴露的工具并注册。"""
        try:
            mcp_tools = await discover_mcp_tools(self.mcp_config_path)
            if mcp_tools:
                self.all_tool_definitions.extend(mcp_tools)
                for t in mcp_tools:
                    self.mcp_tool_names.add(t["function"]["name"])
                info(f"MCP 工具发现完成，共注册 {len(mcp_tools)} 个外部工具")
            else:
                info("未发现可用的 MCP 外部工具")
        except Exception as e:
            warn(f"MCP 工具发现失败: {e}")

    async def close(self) -> None:
        info("等待后台摘要任务完成...")
        result = await self.reme.await_summary_tasks()
        if result:
            info(f"摘要任务结果: {result[:200]}")
        await self.reme.close()

    # ── 记忆检索 ──────────────────────────────────────────────────────────────

    async def _search_memory(self, query: str) -> str:
        try:
            tool_response = await self.reme.memory_search(
                query=query, max_results=4, min_score=0.1
            )
            raw = tool_response.content[0].text if tool_response.content else "[]"
            results = json.loads(raw)
            self.last_memory_hits = results
            show_memory_hit(results)

            if not results:
                return ""

            lines = ["=== 相关历史记忆 ==="]
            for r in results[:3]:
                score = r.get("score", 0)
                text  = r.get("text", "").strip()
                src   = Path(r.get("source_path", "")).name
                lines.append(f"[{src} | score={score:.2f}] {text[:200]}")
            lines.append("=== 记忆结束 ===")
            return "\n".join(lines)

        except Exception as e:
            warn(f"memory_search 失败: {e}")
            return ""

    # ── 上下文压缩 ──────────────────────────────────────────────────────────

    async def _maybe_compact_context(self) -> None:
        if not self.messages:
            return

        char_count = estimate_chars(self.messages)
        if char_count < CONTEXT_CHAR_THRESHOLD:
            return

        info(f"上下文字符数 {char_count:,}，触发 pre_reasoning_hook...")
        reme_msgs = openai_msgs_to_reme(self.messages)

        try:
            kept_msgs, new_summary = await self.reme.pre_reasoning_hook(
                messages=reme_msgs,
                system_prompt=SYSTEM_PROMPT_TEMPLATE,
                compressed_summary=self.compressed_summary,
                language=self.language,
                max_input_length=128 * 1024,
                compact_ratio=0.7,
                memory_compact_reserve=10000,
                enable_tool_result_compact=True,
                tool_result_compact_keep_n=3,
            )

            compacted_count = len(reme_msgs) - len(kept_msgs)
            if compacted_count > 0:
                show_context_compaction(
                    original_count=len(reme_msgs),
                    kept_count=len(kept_msgs),
                    summary=new_summary,
                )
                self.messages = _reme_msgs_to_openai(kept_msgs)
                self.compressed_summary = new_summary

        except Exception as e:
            warn(f"pre_reasoning_hook 失败: {e}")

    # ── 工具执行 ─────────────────────────────────────────────────────────────

    async def _execute_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str
    ) -> str:
        print(
            f"\n{YELLOW}  🔧 执行工具: {tool_name}"
            f"({json.dumps(tool_args, ensure_ascii=False)[:80]}){RESET}"
        )

        profile_summary = self.profile.to_summary_str()
        raw_output = ""

        # ── MCP 工具 ──
        if tool_name in self.mcp_tool_names:
            raw_output = await call_mcp_tool(
                tool_name, tool_args, self.mcp_config_path
            )

        # ── Agent Skills ──
        elif tool_name == "full_preparation_skill":
            skill = FullPreparationSkill(
                llm_client=self.llm, model=self.model,
                on_step=lambda name, status: print(
                    f"{DIM}    📋 Skill Step: {name} → {status}{RESET}"
                ),
            )
            result = await skill.execute(
                jd_text=tool_args.get("jd_text", ""),
                profile_summary=profile_summary,
                company_name=tool_args.get("company_name", ""),
                tavily_api_key=self.tavily_api_key or "",
            )
            raw_output = json.dumps(result, ensure_ascii=False, indent=2)

        elif tool_name == "application_package_skill":
            skill = ApplicationPackageSkill(
                llm_client=self.llm, model=self.model,
                on_step=lambda name, status: print(
                    f"{DIM}    📋 Skill Step: {name} → {status}{RESET}"
                ),
            )
            result = await skill.execute(
                jd_text=tool_args.get("jd_text", ""),
                profile_summary=profile_summary,
                existing_resume=tool_args.get("existing_resume", ""),
                company_name=tool_args.get("company_name", ""),
                language=tool_args.get("language", self.language),
            )
            raw_output = json.dumps(result, ensure_ascii=False, indent=2)

        elif tool_name == "post_interview_skill":
            skill = PostInterviewSkill(
                llm_client=self.llm, model=self.model,
                on_step=lambda name, status: print(
                    f"{DIM}    📋 Skill Step: {name} → {status}{RESET}"
                ),
            )
            result = await skill.execute(
                interview_description=tool_args.get("interview_description", ""),
                company_and_position=tool_args.get("company_and_position", ""),
                tavily_api_key=self.tavily_api_key or "",
            )
            raw_output = json.dumps(result, ensure_ascii=False, indent=2)

        # ── 基础工具 ──
        elif tool_name == "analyze_jd":
            result = await analyze_jd(
                jd_text=tool_args.get("jd_text", ""),
                llm_client=self.llm, model=self.model,
            )
            raw_output = json.dumps(result, ensure_ascii=False, indent=2)

        elif tool_name == "search_job":
            result = await search_job(
                query=tool_args.get("query", ""),
                tavily_api_key=self.tavily_api_key or "",
                max_results=tool_args.get("max_results", 5),
                search_depth=tool_args.get("search_depth", "basic"),
            )
            raw_output = json.dumps(result, ensure_ascii=False, indent=2)

        elif tool_name == "generate_resume":
            result = await generate_resume(
                profile_summary=profile_summary,
                llm_client=self.llm, model=self.model,
                jd_text=tool_args.get("jd_text", ""),
                existing_resume=tool_args.get("existing_resume", ""),
            )
            raw_output = json.dumps(result, ensure_ascii=False, indent=2)

        elif tool_name == "mock_interview":
            result = await mock_interview(
                position=tool_args.get("position", ""),
                llm_client=self.llm, model=self.model,
                interview_type=tool_args.get("interview_type", "技术面"),
                difficulty=tool_args.get("difficulty", "中等"),
                tech_stack=tool_args.get("tech_stack", ""),
                question_count=tool_args.get("question_count", 8),
            )
            raw_output = json.dumps(result, ensure_ascii=False, indent=2)

        elif tool_name == "evaluate_answer":
            result = await evaluate_answer(
                question=tool_args.get("question", ""),
                answer=tool_args.get("answer", ""),
                llm_client=self.llm, model=self.model,
                position=tool_args.get("position", ""),
            )
            raw_output = json.dumps(result, ensure_ascii=False, indent=2)

        elif tool_name == "company_research":
            result = await company_research(
                company_name=tool_args.get("company_name", ""),
                llm_client=self.llm, model=self.model,
                tavily_api_key=self.tavily_api_key or "",
            )
            raw_output = json.dumps(result, ensure_ascii=False, indent=2)

        elif tool_name == "cover_letter_gen":
            result = await cover_letter_gen(
                jd_text=tool_args.get("jd_text", ""),
                llm_client=self.llm, model=self.model,
                profile_summary=profile_summary,
                company_name=tool_args.get("company_name", ""),
                language=tool_args.get("language", self.language),
            )
            raw_output = json.dumps(result, ensure_ascii=False, indent=2)

        elif tool_name == "skill_gap_analysis":
            result = await skill_gap_analysis(
                jd_text=tool_args.get("jd_text", ""),
                llm_client=self.llm, model=self.model,
                profile_summary=profile_summary,
                resume_info=tool_args.get("resume_info", ""),
            )
            raw_output = json.dumps(result, ensure_ascii=False, indent=2)

        elif tool_name == "interview_review":
            result = await interview_review(
                interview_description=tool_args.get("interview_description", ""),
                llm_client=self.llm, model=self.model,
                company_and_position=tool_args.get("company_and_position", ""),
            )
            raw_output = json.dumps(result, ensure_ascii=False, indent=2)

        elif tool_name == "career_path_planner":
            result = await career_path_planner(
                llm_client=self.llm, model=self.model,
                profile_summary=profile_summary,
                current_stage=tool_args.get("current_stage", ""),
                target_direction=tool_args.get("target_direction", ""),
            )
            raw_output = json.dumps(result, ensure_ascii=False, indent=2)

        elif tool_name == "resume_keyword_optimizer":
            result = await resume_keyword_optimizer(
                jd_text=tool_args.get("jd_text", ""),
                resume_text=tool_args.get("resume_text", ""),
                llm_client=self.llm, model=self.model,
            )
            raw_output = json.dumps(result, ensure_ascii=False, indent=2)

        else:
            raw_output = f"未知工具: {tool_name}"

        original_len = len(raw_output)

        # 压缩长工具输出
        from agentscope.message import Msg
        tool_msg = Msg(
            name="tool",
            role="tool",
            content=[{
                "type": "tool_result",
                "output": raw_output,
                "name": tool_name,
                "tool_call_id": tool_call_id,
            }],
        )

        try:
            compacted_msgs = await self.reme.compact_tool_result(messages=[tool_msg])
            compacted_msg  = compacted_msgs[0] if compacted_msgs else tool_msg
            if isinstance(compacted_msg.content, list):
                block = compacted_msg.content[0]
                output_str = block.get("output", raw_output)
                if isinstance(output_str, list):
                    output_str = " ".join(
                        b.get("text", "") for b in output_str if isinstance(b, dict)
                    )
            else:
                output_str = str(compacted_msg.content)
        except Exception as e:
            warn(f"compact_tool_result 失败: {e}")
            output_str = raw_output

        compressed_len = len(output_str)
        file_path = extract_tool_result_file(output_str)
        show_tool_compaction(tool_name, original_len, compressed_len, file_path)

        return output_str

    # ── 主 LLM 调用 ──────────────────────────────────────────────────────────

    async def _chat_with_tools(
        self,
        user_input: str,
        image_data: dict | None = None,
    ) -> str:
        # 1. 记忆检索
        memory_context  = await self._search_memory(user_input)
        profile_summary = self.profile.to_summary_str()

        system_content = SYSTEM_PROMPT_TEMPLATE.format(
            profile_summary=(
                profile_summary if "暂未填写" not in profile_summary else ""
            ),
            memory_context=memory_context,
        )
        if self.compressed_summary:
            system_content += COMPRESSED_SUMMARY_MARKER.format(
                summary=self.compressed_summary
            )

        # 2. 上下文预检
        await self._maybe_compact_context()

        # 3. 追加到对话历史
        history_content = (
            f"[上传图片: {image_data['file_name']}]\n{user_input}"
            if image_data else user_input
        )
        self.messages.append({"role": "user", "content": history_content})

        # 4. 构建请求消息
        if image_data:
            current_user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{image_data['media_type']};base64,{image_data['base64']}"
                    }},
                ],
            }
            request_messages = (
                [{"role": "system", "content": system_content}]
                + self.messages[:-1]
                + [current_user_msg]
            )
        else:
            request_messages = (
                [{"role": "system", "content": system_content}]
                + self.messages
            )

        # 5. 工具调用循环
        max_tool_rounds = 5
        for _ in range(max_tool_rounds):
            response = await self.llm.chat.completions.create(
                model=self.model,
                messages=request_messages,
                tools=self.all_tool_definitions,
                tool_choice="auto",
                temperature=0.7,
            )
            choice = response.choices[0]
            msg    = choice.message

            if not msg.tool_calls:
                assistant_text = msg.content or ""
                self.messages.append({"role": "assistant", "content": assistant_text})
                request_messages.append({"role": "assistant", "content": assistant_text})
                return assistant_text

            tool_calls_data = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
            self.messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": tool_calls_data,
            })
            request_messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": tool_calls_data,
            })

            for tc in msg.tool_calls:
                tool_args   = json.loads(tc.function.arguments)
                tool_output = await self._execute_tool(tc.function.name, tool_args, tc.id)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": tool_output,
                })
                request_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": tool_output,
                })

        return "（工具调用轮次已达上限）"

    # ── 视觉模型调用 ──────────────────────────────────────────────────────────

    async def _chat_with_image(
        self, result: "FileParseResult", user_prompt: str
    ) -> str:
        if not result.image_base64:
            return "图片编码失败，无法分析。"
        return await self._chat_with_tools(
            user_input=user_prompt,
            image_data={
                "base64":     result.image_base64,
                "media_type": result.image_media_type,
                "file_name":  result.file_name,
            },
        )

    # ── 文件输入处理 ──────────────────────────────────────────────────────────

    async def _handle_file_input(
        self, file_path: str, user_hint: str = ""
    ) -> None:
        path = Path(file_path.strip().strip('"').strip("'"))

        print(f"\n{DIM}📂 正在解析: {path.name}...{RESET}")
        result = parse_file(path)

        if not result.success:
            print(f"{RED}❌ 文件解析失败: {result.error}{RESET}")
            return

        print(f"{GREEN}✅ 解析成功:{RESET}")
        print(f"   文件类型 : {result.file_type.upper()}")
        print(f"   文件名称 : {result.file_name}")
        if result.is_text_based:
            print(f"   提取字数 : {result.char_count:,} 字符")
            if result.page_count:
                print(f"   页数     : {result.page_count} 页")
        elif result.is_image:
            print(f"   文件大小 : {result.char_count:,} bytes")

        if result.is_image:
            if not user_hint:
                user_hint = (
                    "请仔细分析这张图片。"
                    "如果是简历，请提取关键信息并给出优化建议。"
                    "如果是 JD，请分析岗位要求和技术栈。"
                )
            print(f"\n{DIM}  图片类型，使用多模态模型分析...{RESET}")
            response = await self._chat_with_tools(
                user_input=user_hint,
                image_data={
                    "base64":     result.image_base64,
                    "media_type": result.image_media_type,
                    "file_name":  result.file_name,
                },
            )
            print(f"\n{CYAN}{BOLD}助手 > {RESET}{response}\n")
            asyncio.create_task(
                self._async_extract_profile(f"[图片分析]\n{response}")
            )
            return

        if not result.text_content:
            print(f"{YELLOW}⚠️  文件解析成功但未提取到文字内容{RESET}")
            return

        content_role = guess_content_role(result.text_content)
        role_label = {"jd": "JD（岗位描述）", "resume": "简历", "unknown": "文档"}
        print(f"   内容识别 : {role_label.get(content_role, '文档')}")

        text_to_inject = result.text_content
        was_truncated  = False
        if len(text_to_inject) > MAX_FILE_TEXT_INJECT:
            text_to_inject = text_to_inject[:MAX_FILE_TEXT_INJECT]
            was_truncated  = True
            print(
                f"{YELLOW}  ⚠️  文档过长，已截取前 {MAX_FILE_TEXT_INJECT:,} 字符{RESET}"
            )

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
            if was_truncated else ""
        )
        injected_message = (
            f"我上传了一份文件：{result.file_name}\n"
            f"文件类型：{result.file_type.upper()}，共 {result.char_count:,} 字符"
            f"{truncation_note}\n\n"
            f"--- 文件内容开始 ---\n"
            f"{text_to_inject}\n"
            f"--- 文件内容结束 ---\n\n"
            f"{analysis_request}"
        )

        print(f"\n{DIM}正在分析文件内容...{RESET}")
        response = await self._chat_with_tools(injected_message)
        print(f"\n{CYAN}{BOLD}助手 > {RESET}{response}\n")

        asyncio.create_task(
            self._async_extract_profile(
                f"[文件分析: {result.file_name}]\n{response}"
            )
        )

    # ── Profile 异步提取 ──────────────────────────────────────────────────────

    async def _async_extract_profile(self, conversation: str) -> None:
        extract_prompt = f"""从以下对话片段中提取用户的求职偏好信息。
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
            raw = resp.choices[0].message.content.strip()
            raw = (
                raw.removeprefix("```json")
                   .removeprefix("```")
                   .removesuffix("```")
                   .strip()
            )
            data    = json.loads(raw)
            updates = {k: v for k, v in data.items() if v and k in self.profile.read()}
            if updates:
                updated = self.profile.update_from_dict(updates)
                if updated:
                    info(f"画像已更新: {updated}")
        except Exception:
            pass

    # ── CLI 命令 ──────────────────────────────────────────────────────────────

    def _parse_file_command(self, raw: str) -> tuple[str, str]:
        rest = raw[5:].strip()
        if not rest:
            return "", ""

        if rest.startswith('"') or rest.startswith("'"):
            quote_char = rest[0]
            end_quote  = rest.find(quote_char, 1)
            if end_quote != -1:
                file_path = rest[1:end_quote]
                user_hint = rest[end_quote + 1:].strip()
                return file_path, user_hint

        parts = rest.split()
        for i in range(len(parts), 0, -1):
            candidate = " ".join(parts[:i])
            if Path(candidate).exists():
                user_hint = " ".join(parts[i:])
                return candidate, user_hint

        parts = rest.split(None, 1)
        file_path = parts[0]
        user_hint = parts[1] if len(parts) > 1 else ""
        return file_path, user_hint

    def _handle_command(self, cmd_raw: str) -> bool:
        cmd = cmd_raw.strip().lower().split()[0] if cmd_raw.strip() else ""

        if cmd == "/help":
            print(f"""
{CYAN}{BOLD}📖 可用命令:{RESET}
  /file <路径>      — 解析并分析 PDF / Word / 图片文件
                      支持格式：{get_supported_formats()}
  /tools            — 查看所有已注册的工具和技能
  /profile          — 查看当前结构化求职画像
  /memory           — 查看最近命中的历史记忆
  /status           — 查看工作目录文件状态
  /clear            — 清除当前对话历史（不影响长期记忆）
  /exit             — 退出
  /help             — 显示此帮助
""")
            return True

        if cmd == "/tools":
            banner("🔧 已注册的工具和技能", CYAN)
            print(f"\n{BOLD}基础工具（11 个）：{RESET}")
            for t in TOOL_DEFINITIONS:
                func = t["function"]
                print(f"  • {func['name']:30s} — {func['description'][:60]}")
            print(f"\n{BOLD}Agent Skills（3 个）：{RESET}")
            for t in SKILL_TOOL_DEFINITIONS:
                func = t["function"]
                print(f"  ★ {func['name']:30s} — {func['description'][:60]}")
            if self.mcp_tool_names:
                print(f"\n{BOLD}MCP 外部工具（{len(self.mcp_tool_names)} 个）：{RESET}")
                for name in sorted(self.mcp_tool_names):
                    print(f"  ⚡ {name}")
            print()
            return True

        if cmd == "/profile":
            banner("📋 当前求职画像", MAGENTA)
            print(self.profile.read())
            print(f"{DIM}文件路径: {self.profile.path_str()}{RESET}")
            return True

        if cmd == "/memory":
            banner("🧠 最近命中的历史记忆", MAGENTA)
            if self.last_memory_hits:
                for r in self.last_memory_hits:
                    print(
                        f"  score={r.get('score', 0):.3f} | "
                        f"{r.get('text', '')[:150]}"
                    )
            else:
                print("  暂无命中记忆（请先进行几轮对话）")
            return True

        if cmd == "/status":
            banner("📂 工作目录状态", CYAN)
            memory_dir      = self.working_dir / "memory"
            tool_result_dir = self.working_dir / "tool_result"
            md_files = list(memory_dir.glob("*.md"))      if memory_dir.exists()      else []
            tr_files = list(tool_result_dir.glob("*.txt")) if tool_result_dir.exists() else []

            print(f"  工作目录   : {self.working_dir}")
            print(f"  文本模型   : {self.model}")
            print(f"  视觉模型   : {self.vision_model}")
            print(f"  Tavily     : {'✅ 已配置' if self.tavily_api_key else '⚠️  未配置'}")
            print(f"  澄清门控   : ✅ 已启用")
            print(f"  基础工具   : 11 个")
            print(f"  Agent Skills: 3 个")
            print(f"  MCP 工具   : {len(self.mcp_tool_names)} 个")
            print(f"  支持格式   : {get_supported_formats()}")
            print(f"  记忆文件   : {len(md_files)} 个 .md 文件")
            for f in md_files:
                print(f"    → {f.name} ({f.stat().st_size:,} bytes)")
            print(f"  工具输出   : {len(tr_files)} 个 .txt 外置压缩文件")
            for f in tr_files:
                print(f"    → {f.name} ({f.stat().st_size:,} bytes)")
            if self.compressed_summary:
                print(f"  in-context 摘要: {len(self.compressed_summary)} chars")
            if self.pending_clarification:
                print(f"  待澄清状态 : ✅ 挂起中 (ask_count={self.pending_clarification.get('ask_count', 0)})")
            print(f"  对话轮次   : {len(self.messages)} 条消息")
            return True

        if cmd == "/clear":
            self.messages.clear()
            self.compressed_summary = ""
            self.pending_clarification = None
            print(f"{GREEN}✓ 对话历史已清除（长期记忆和画像保留）{RESET}")
            return True

        if cmd == "/exit":
            return False

        return False

    # ── 主对话循环 ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        banner("🚀 ReMeLight 求职助手 Agent v2（升级版）", CYAN)
        print("  输入 /help 查看命令   输入 /tools 查看工具列表")
        print("  输入 /exit 退出\n")

        while True:
            try:
                user_input = input(f"{GREEN}{BOLD}你 > {RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not user_input:
                continue

            if user_input.lower() in ("/exit", "exit", "quit"):
                print("正在保存记忆并退出...")
                break

            if user_input.lower().startswith("/file"):
                file_path, user_hint = self._parse_file_command(user_input)
                if not file_path:
                    file_path = input(
                        f"{DIM}  请输入文件路径（支持 PDF/DOCX/图片）: {RESET}"
                    ).strip().strip('"').strip("'")
                if file_path:
                    try:
                        await self._handle_file_input(file_path, user_hint)
                    except Exception as e:
                        print(f"{RED}文件处理出错: {e}{RESET}")
                continue

            if user_input.startswith("/"):
                handled = self._handle_command(user_input)
                if not handled and user_input.lower() != "/exit":
                    print(f"{DIM}未知命令，输入 /help 查看帮助{RESET}")
                    continue
                if user_input.lower() == "/exit":
                    break
                continue

            # ── 普通文字对话 ──────────────────────────────────────────────────

            print(f"\n{DIM}思考中...{RESET}")

            try:
                is_clarification_response = self.pending_clarification is not None

                if not is_clarification_response:
                    if ClarificationGate.should_force_proceed(self.pending_clarification):
                        info("已达最大追问次数，强制放行")
                        self.pending_clarification = None
                        is_clarification_response = False
                    else:
                        profile_summary = self.profile.to_summary_str()
                        gate_result = await self.gate.clarify_gate(
                            user_input=user_input,
                            recent_messages=self.messages[-4:],
                            profile_summary=profile_summary,
                        )

                        if gate_result.is_ambiguous:
                            ask_count = 1
                            if self.pending_clarification is not None:
                                ask_count = self.pending_clarification.get("ask_count", 0) + 1

                            if ask_count > 2:
                                info("已达最大追问次数，强制放行")
                                self.pending_clarification = None
                            else:
                                question = ClarificationGate.generate_clarify_question(
                                    gate_result, user_input
                                )
                                self.pending_clarification = {
                                    "original_input":   user_input,
                                    "ambiguity_type":   gate_result.ambiguity_type,
                                    "missing_slots":    gate_result.missing_slots,
                                    "clarify_question": question,
                                    "ask_count":        ask_count,
                                }
                                self.messages.append({"role": "user",      "content": user_input})
                                self.messages.append({"role": "assistant",  "content": question})
                                print(f"\n{CYAN}{BOLD}助手 > {RESET}{question}\n")
                                continue

                profile_summary = self.profile.to_summary_str()
                normalized_query = await self.gate.normalize_query(
                    current_input=user_input,
                    pending=self.pending_clarification,
                    profile_summary=profile_summary,
                    recent_messages=self.messages[-4:],
                )

                self.pending_clarification = None
                info(f"标准化 Query: {normalized_query[:80]}")

                response = await self._chat_with_tools(normalized_query)
                print(f"\n{CYAN}{BOLD}助手 > {RESET}{response}\n")

                if ClarificationGate.check_slot_completeness(normalized_query):
                    asyncio.create_task(
                        self._async_extract_profile(
                            f"用户: {normalized_query}\n助手: {response}"
                        )
                    )
                else:
                    info("槽位不完整，跳过本轮偏好写入")

            except Exception as e:
                print(f"{RED}错误: {e}{RESET}")

        await self.close()


# ── Msg → OpenAI dict 转换 ────────────────────────────────────────────────────

def _reme_msgs_to_openai(reme_msgs: list) -> list[dict]:
    result = []
    for msg in reme_msgs:
        role    = msg.role
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
                        texts.extend(
                            b.get("text", "") for b in output if isinstance(b, dict)
                        )
                elif btype == "text":
                    texts.append(block.get("text", ""))
                elif btype in ("image", "audio", "video"):
                    texts.append(f"[{btype}]")
            content = "\n".join(t for t in texts if t)
        result.append({"role": role, "content": str(content)})
    return result


# ── 入口 ──────────────────────────────────────────────────────────────────────

async def main():
    import dotenv
    dotenv.load_dotenv(dotenv_path=".env", override=True)

    agent = JobAgentV2(
        working_dir=os.environ.get("AGENT_WORKING_DIR", ".job_agent_v2"),
        model=os.environ.get("LLM_MODEL", "qwen3.5-plus"),
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
        embedding_api_key=os.environ.get("EMBEDDING_API_KEY"),
        embedding_base_url=os.environ.get("EMBEDDING_BASE_URL"),
        tavily_api_key=os.environ.get("TAVILY_API_KEY"),
        tool_result_threshold=int(os.environ.get("TOOL_RESULT_THRESHOLD", "800")),
        language=os.environ.get("AGENT_LANGUAGE", "zh"),
        enable_mcp_discovery=os.environ.get("ENABLE_MCP_DISCOVERY", "true").lower() == "true",
    )

    await agent.start()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
