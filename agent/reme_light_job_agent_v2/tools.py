"""
tools.py — 求职助手工具集

工具 1: analyze_jd  — JD 详细分析（长输出，触发 tool_result_compactor）
工具 2: search_job  — 使用 Tavily 联网搜索公司/岗位/面试信息
"""
import json
from datetime import datetime
from typing import Optional


# ── 工具 1: JD 分析 ──────────────────────────────────────────────────────────
JD_ANALYSIS_PROMPT = """你是一名资深 HR 顾问和技术面试官。
请对以下 JD 进行深度分析，输出结构化报告，要求详细（不少于 800 字）：

## 岗位基本信息
- 岗位名称、级别、所在公司/部门

## 核心技术要求
- 必须掌握的技术栈（逐一列出）
- 加分项技术
- 技术深度要求分析

## 软性能力要求
- 沟通能力、团队协作、自驱力等

## 业务背景分析
- 该岗位所在业务方向
- 岗位在团队中的角色定位

## 简历匹配建议
- 应该重点突出哪些经历
- 哪些关键词应该出现在简历中
- 简历常见雷区

## 面试准备建议
- 技术面可能考察的核心知识点（至少 10 条）
- 行为面（STAR 法则）可能问到的问题
- 反问面试官的好问题

## 薪资与发展分析
- 根据岗位描述推断薪资范围
- 岗位发展路径

---
JD 内容：
{jd_text}
"""


async def analyze_jd(jd_text: str, llm_client, model: str) -> dict:
    """
    分析岗位 JD，返回详细结构化报告。
    输出故意保持较长（>1000 字），以触发 ReMeLight 的 tool_result_compactor。

    Args:
        jd_text: JD 原文
        llm_client: openai.AsyncOpenAI 实例
        model: 模型名称

    Returns:
        dict with keys: analysis (str), tool_name (str), timestamp (str)
    """
    prompt = JD_ANALYSIS_PROMPT.format(jd_text=jd_text)

    response = await llm_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是专业的 JD 分析顾问，输出详细、结构化的分析报告。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
    )

    analysis = response.choices[0].message.content

    return {
        "tool_name": "analyze_jd",
        "timestamp": datetime.now().isoformat(),
        "jd_preview": jd_text[:200] + ("..." if len(jd_text) > 200 else ""),
        "analysis": analysis,
    }


# ── 工具 2: Tavily 联网搜索 ───────────────────────────────────────────────────
async def search_job(
    query: str,
    tavily_api_key: str,
    max_results: int = 5,
    search_depth: str = "basic",
) -> dict:
    """
    使用 Tavily API 搜索职位/公司/面试信息。
    Tavily 专为 AI Agent 设计，返回高质量的正文摘要。

    Args:
        query: 搜索关键词，例如 "字节跳动 后端工程师 面试题 2024"
        tavily_api_key: Tavily API Key（tvly-xxx 格式）
        max_results: 最多返回结果数，默认 5
        search_depth: "basic"（快速）或 "advanced"（深度，消耗更多配额）

    Returns:
        dict with keys: query, results (list), answer, timestamp
    """
    try:
        from tavily import AsyncTavilyClient
    except ImportError:
        return {
            "tool_name": "search_job",
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "error": "tavily-python 未安装，请执行：pip install tavily-python",
            "results": [],
        }

    if not tavily_api_key:
        return {
            "tool_name": "search_job",
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "error": "TAVILY_API_KEY 未配置，请在 .env 中添加",
            "results": [],
        }

    try:
        client = AsyncTavilyClient(api_key=tavily_api_key)

        response = await client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            # 针对求职场景，聚焦以下域名质量更高
            include_domains=[
                "zhihu.com",
                "nowcoder.com",
                "leetcode.cn",
                "boss直聘.com",
                "lagou.com",
                "linkedin.com",
                "glassdoor.com",
                "36kr.com",
            ],
        )

        # 提取结构化结果
        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", "")[:500],  # 截取前 500 字
                "score": round(item.get("score", 0), 3),
            })

        # Tavily 可以提供一个 AI 综合摘要（answer 字段）
        answer = response.get("answer", "")

        return {
            "tool_name": "search_job",
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "answer": answer,          # Tavily 的 AI 综合答案（可能为空）
            "results": results,
            "result_count": len(results),
        }

    except Exception as e:
        return {
            "tool_name": "search_job",
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "results": [],
        }


# ── 工具注册表（OpenAI function calling 格式）────────────────────────────────
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_jd",
            "description": (
                "深度分析岗位 JD，返回详细的技术要求解读、简历匹配建议、面试准备清单。"
                "当用户粘贴 JD 或说'帮我分析这个岗位'时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "jd_text": {
                        "type": "string",
                        "description": "JD 全文或用户粘贴的岗位描述文字",
                    }
                },
                "required": ["jd_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_job",
            "description": (
                "使用 Tavily 联网搜索公司信息、岗位行情、面试经验、技术资料。"
                "当用户询问某公司背景、某技术面试题、某岗位薪资行情时调用。"
                "搜索关键词尽量具体，例如 '字节跳动后端面试题2024' 而非 '面试题'。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，建议包含公司名/岗位/年份，例如 '美团 算法工程师 面试经验 2024'",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最多返回几条结果，默认 5，最大 10",
                    },
                    "search_depth": {
                        "type": "string",
                        "description": "搜索深度：basic（快速，默认）或 advanced（深度，质量更高但消耗更多配额）",
                        "enum": ["basic", "advanced"],
                    },
                },
                "required": ["query"],
            },
        },
    },
]