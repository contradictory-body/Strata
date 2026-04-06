"""
mcp_job_server.py — 求职助手 MCP Server

将核心求职工具通过 MCP（Model Context Protocol）协议暴露为标准化服务，
使得任何兼容 MCP 的 AI Agent / IDE / 客户端均可调用。

支持的传输方式：
  - stdio  : 本地进程通信（适合 IDE 插件、Claude Desktop）
  - sse    : Server-Sent Events（适合 Web 客户端）
  - http   : Streamable HTTP（适合服务化部署）

启动方式：
  # stdio 模式（供 Claude Desktop / Cursor 等调用）
  python mcp_job_server.py --transport stdio

  # SSE 模式
  python mcp_job_server.py --transport sse --port 8010

  # HTTP 模式
  python mcp_job_server.py --transport http --port 8010
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastmcp import FastMCP

# ═══════════════════════════════════════════════════════════════════════════════
# 初始化 MCP Server
# ═══════════════════════════════════════════════════════════════════════════════

mcp = FastMCP(
    name="job-assistant-mcp",
    instructions=(
        "求职助手 MCP 服务：提供 JD 分析、简历优化、模拟面试、公司调研等求职相关工具。"
        "所有工具均接受文本输入并返回 JSON 格式的结构化结果。"
    ),
)

# 延迟初始化 LLM 客户端（首次调用时创建）
_llm_client = None
_model = None
_tavily_api_key = None


def _get_llm_client():
    global _llm_client, _model, _tavily_api_key
    if _llm_client is None:
        from openai import AsyncOpenAI
        import dotenv
        dotenv.load_dotenv(dotenv_path=".env", override=True)

        _llm_client = AsyncOpenAI(
            api_key=os.environ.get("LLM_API_KEY", ""),
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        )
        _model = os.environ.get("LLM_MODEL", "qwen3.5-plus")
        _tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
    return _llm_client, _model, _tavily_api_key


# ═══════════════════════════════════════════════════════════════════════════════
# 注册 MCP Tools
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="analyze_jd",
    description="深度分析岗位 JD，返回技术要求、简历建议、面试准备清单",
)
async def mcp_analyze_jd(jd_text: str) -> str:
    """分析岗位 JD。"""
    from tools import analyze_jd
    client, model, _ = _get_llm_client()
    result = await analyze_jd(jd_text=jd_text, llm_client=client, model=model)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(
    name="search_job",
    description="联网搜索公司信息、岗位行情、面试经验",
)
async def mcp_search_job(query: str, max_results: int = 5) -> str:
    """联网搜索职位信息。"""
    from tools import search_job
    _, _, tavily_key = _get_llm_client()
    result = await search_job(
        query=query, tavily_api_key=tavily_key, max_results=max_results,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(
    name="generate_resume",
    description="根据用户画像和目标 JD 生成/优化简历",
)
async def mcp_generate_resume(
    jd_text: str = "", existing_resume: str = "", profile_summary: str = "",
) -> str:
    """生成/优化简历。"""
    from tools import generate_resume
    client, model, _ = _get_llm_client()
    result = await generate_resume(
        profile_summary=profile_summary,
        llm_client=client,
        model=model,
        jd_text=jd_text,
        existing_resume=existing_resume,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(
    name="mock_interview",
    description="生成模拟面试题目，支持技术面/行为面/系统设计等类型",
)
async def mcp_mock_interview(
    position: str,
    interview_type: str = "技术面",
    difficulty: str = "中等",
    tech_stack: str = "",
    question_count: int = 8,
) -> str:
    """模拟面试出题。"""
    from tools import mock_interview
    client, model, _ = _get_llm_client()
    result = await mock_interview(
        position=position,
        llm_client=client,
        model=model,
        interview_type=interview_type,
        difficulty=difficulty,
        tech_stack=tech_stack,
        question_count=question_count,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(
    name="evaluate_answer",
    description="评估面试回答质量，给出评分和改进建议",
)
async def mcp_evaluate_answer(
    question: str, answer: str, position: str = "",
) -> str:
    """评估面试回答。"""
    from tools import evaluate_answer
    client, model, _ = _get_llm_client()
    result = await evaluate_answer(
        question=question,
        answer=answer,
        llm_client=client,
        model=model,
        position=position,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(
    name="company_research",
    description="深度调研目标公司的技术栈、面试流程、薪资福利、企业文化",
)
async def mcp_company_research(company_name: str) -> str:
    """公司深度调研。"""
    from tools import company_research
    client, model, tavily_key = _get_llm_client()
    result = await company_research(
        company_name=company_name,
        llm_client=client,
        model=model,
        tavily_api_key=tavily_key,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(
    name="cover_letter_gen",
    description="生成针对特定公司和岗位的求职信 / Cover Letter",
)
async def mcp_cover_letter_gen(
    jd_text: str, company_name: str = "", language: str = "zh",
    profile_summary: str = "",
) -> str:
    """生成求职信。"""
    from tools import cover_letter_gen
    client, model, _ = _get_llm_client()
    result = await cover_letter_gen(
        jd_text=jd_text,
        llm_client=client,
        model=model,
        profile_summary=profile_summary,
        company_name=company_name,
        language=language,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(
    name="skill_gap_analysis",
    description="对比用户技能与目标 JD 要求，输出技能差距和学习路线",
)
async def mcp_skill_gap_analysis(
    jd_text: str, profile_summary: str = "", resume_info: str = "",
) -> str:
    """技能差距分析。"""
    from tools import skill_gap_analysis
    client, model, _ = _get_llm_client()
    result = await skill_gap_analysis(
        jd_text=jd_text,
        llm_client=client,
        model=model,
        profile_summary=profile_summary,
        resume_info=resume_info,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(
    name="interview_review",
    description="对面试经历进行复盘分析，总结教训和改进方向",
)
async def mcp_interview_review(
    interview_description: str, company_and_position: str = "",
) -> str:
    """面试复盘。"""
    from tools import interview_review
    client, model, _ = _get_llm_client()
    result = await interview_review(
        interview_description=interview_description,
        llm_client=client,
        model=model,
        company_and_position=company_and_position,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(
    name="career_path_planner",
    description="制定个性化职业发展路径和成长规划",
)
async def mcp_career_path_planner(
    current_stage: str = "", target_direction: str = "",
    profile_summary: str = "",
) -> str:
    """职业路径规划。"""
    from tools import career_path_planner
    client, model, _ = _get_llm_client()
    result = await career_path_planner(
        llm_client=client,
        model=model,
        profile_summary=profile_summary,
        current_stage=current_stage,
        target_direction=target_direction,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(
    name="resume_keyword_optimizer",
    description="分析 JD 关键词，优化简历以通过 ATS 机器筛选",
)
async def mcp_resume_keyword_optimizer(jd_text: str, resume_text: str) -> str:
    """ATS 关键词优化。"""
    from tools import resume_keyword_optimizer
    client, model, _ = _get_llm_client()
    result = await resume_keyword_optimizer(
        jd_text=jd_text,
        resume_text=resume_text,
        llm_client=client,
        model=model,
    )
    return json.dumps(result, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# MCP Resources（MCP 资源 - 提供上下文信息）
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.resource("profile://current")
async def get_current_profile() -> str:
    """获取当前用户的求职画像。"""
    profile_path = Path(
        os.environ.get("AGENT_WORKING_DIR", ".job_agent_v2")
    ) / "PROFILE.md"
    if profile_path.exists():
        return profile_path.read_text(encoding="utf-8")
    return "暂无用户画像"


@mcp.resource("tools://list")
async def get_tools_list() -> str:
    """获取所有可用工具列表。"""
    from tools import TOOL_DEFINITIONS
    tools_info = []
    for t in TOOL_DEFINITIONS:
        func = t["function"]
        tools_info.append(f"- {func['name']}: {func['description'][:80]}")
    return "\n".join(tools_info)


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="求职助手 MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse", "http"],
        default="stdio", help="传输方式",
    )
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8010, help="监听端口")
    args = parser.parse_args()

    run_args = {"transport": args.transport, "show_banner": False}
    if args.transport != "stdio":
        run_args.update({"host": args.host, "port": args.port})

    print(f"🚀 求职助手 MCP Server 启动中... (transport={args.transport})")
    mcp.run(**run_args)


if __name__ == "__main__":
    main()
