"""
tools.py — 求职助手工具集（扩展版）

工具 1:  analyze_jd             — JD 详细分析
工具 2:  search_job             — Tavily 联网搜索
工具 3:  generate_resume        — 根据 JD + 用户画像 生成/优化简历
工具 4:  mock_interview         — 模拟面试出题
工具 5:  evaluate_answer        — 评估面试回答并打分
工具 6:  company_research       — 公司深度调研
工具 7:  cover_letter_gen       — 生成求职信 / Cover Letter
工具 8:  skill_gap_analysis     — 技能差距分析
工具 9:  interview_review       — 面试复盘分析
工具 10: career_path_planner    — 职业路径规划
工具 11: resume_keyword_optimizer — ATS 简历关键词优化
"""

import json
from datetime import datetime
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 工具 1: JD 分析
# ═══════════════════════════════════════════════════════════════════════════════

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
    """分析岗位 JD，返回详细结构化报告。"""
    prompt = JD_ANALYSIS_PROMPT.format(jd_text=jd_text)
    response = await llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是专业的 JD 分析顾问，输出详细、结构化的分析报告。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    return {
        "tool_name": "analyze_jd",
        "timestamp": datetime.now().isoformat(),
        "jd_preview": jd_text[:200] + ("..." if len(jd_text) > 200 else ""),
        "analysis": response.choices[0].message.content,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 工具 2: Tavily 联网搜索
# ═══════════════════════════════════════════════════════════════════════════════

async def search_job(
    query: str,
    tavily_api_key: str,
    max_results: int = 5,
    search_depth: str = "basic",
) -> dict:
    """使用 Tavily API 搜索职位/公司/面试信息。"""
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
            "error": "TAVILY_API_KEY 未配置",
            "results": [],
        }

    try:
        client = AsyncTavilyClient(api_key=tavily_api_key)
        response = await client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_domains=[
                "zhihu.com", "nowcoder.com", "leetcode.cn",
                "lagou.com", "linkedin.com", "glassdoor.com", "36kr.com",
            ],
        )
        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", "")[:500],
                "score": round(item.get("score", 0), 3),
            })
        return {
            "tool_name": "search_job",
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "answer": response.get("answer", ""),
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


# ═══════════════════════════════════════════════════════════════════════════════
# 工具 3: 简历生成/优化
# ═══════════════════════════════════════════════════════════════════════════════

RESUME_GEN_PROMPT = """你是一名资深简历顾问。请根据以下信息生成/优化一份专业简历。

## 用户画像
{profile_summary}

## 目标岗位 JD（如果提供）
{jd_text}

## 用户现有简历/补充信息
{existing_resume}

## 要求
1. 使用 Markdown 格式输出完整简历
2. 针对目标岗位调整关键词和经历描述
3. 使用 STAR 法则描述项目经历
4. 量化成果（数字、百分比、规模等）
5. 突出与 JD 匹配的技术栈和经验
6. 简历长度控制在 1-2 页
"""


async def generate_resume(
    profile_summary: str,
    llm_client,
    model: str,
    jd_text: str = "",
    existing_resume: str = "",
) -> dict:
    """根据用户画像和目标 JD 生成/优化简历。"""
    prompt = RESUME_GEN_PROMPT.format(
        profile_summary=profile_summary or "暂无",
        jd_text=jd_text or "未提供",
        existing_resume=existing_resume or "暂无",
    )
    response = await llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是专业的简历优化顾问，擅长针对特定岗位定制简历。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=4096,
    )
    return {
        "tool_name": "generate_resume",
        "timestamp": datetime.now().isoformat(),
        "resume_content": response.choices[0].message.content,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 工具 4: 模拟面试出题
# ═══════════════════════════════════════════════════════════════════════════════

MOCK_INTERVIEW_PROMPT = """你是一名资深技术面试官。请根据以下信息生成一套模拟面试题。

## 目标岗位
{position}

## 面试类型
{interview_type}

## 难度等级
{difficulty}

## 用户技术栈
{tech_stack}

## 要求
1. 生成 {question_count} 道面试题
2. 每道题包含：题目、考察点、参考答案要点、评分标准
3. 技术题需要覆盖：基础知识、项目经验、系统设计、编码能力
4. 行为题使用 STAR 法则设计
5. 题目由浅入深排列
"""


async def mock_interview(
    position: str,
    llm_client,
    model: str,
    interview_type: str = "技术面",
    difficulty: str = "中等",
    tech_stack: str = "",
    question_count: int = 8,
) -> dict:
    """生成模拟面试题目。"""
    prompt = MOCK_INTERVIEW_PROMPT.format(
        position=position,
        interview_type=interview_type,
        difficulty=difficulty,
        tech_stack=tech_stack or "未指定",
        question_count=question_count,
    )
    response = await llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是严谨的技术面试官，出题专业且有梯度。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=4096,
    )
    return {
        "tool_name": "mock_interview",
        "timestamp": datetime.now().isoformat(),
        "position": position,
        "interview_type": interview_type,
        "difficulty": difficulty,
        "questions": response.choices[0].message.content,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 工具 5: 面试回答评估
# ═══════════════════════════════════════════════════════════════════════════════

EVALUATE_ANSWER_PROMPT = """你是一名资深面试官，请评估以下面试回答。

## 面试问题
{question}

## 候选人回答
{answer}

## 目标岗位
{position}

## 评估要求
请从以下维度评分（每项 1-10 分）并给出详细反馈：
1. **完整性**：是否覆盖了问题的所有要点
2. **深度**：技术理解的深度和准确性
3. **逻辑性**：回答的条理和逻辑是否清晰
4. **实战性**：是否结合了实际项目经验
5. **表达力**：语言表达是否简洁专业

最后给出：
- 综合评分（1-10）
- 优点总结
- 改进建议
- 优化后的参考回答
"""


async def evaluate_answer(
    question: str,
    answer: str,
    llm_client,
    model: str,
    position: str = "",
) -> dict:
    """评估面试回答质量，给出评分和改进建议。"""
    prompt = EVALUATE_ANSWER_PROMPT.format(
        question=question,
        answer=answer,
        position=position or "未指定",
    )
    response = await llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是公正严格的面试评估专家。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=3000,
    )
    return {
        "tool_name": "evaluate_answer",
        "timestamp": datetime.now().isoformat(),
        "question": question[:100],
        "evaluation": response.choices[0].message.content,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 工具 6: 公司深度调研
# ═══════════════════════════════════════════════════════════════════════════════

COMPANY_RESEARCH_PROMPT = """你是一名企业研究分析师。请根据以下搜索结果，对目标公司进行深度分析。

## 公司名称
{company_name}

## 搜索结果
{search_results}

## 分析维度
1. **公司概况**：成立时间、规模、融资阶段、主营业务
2. **技术栈**：公司主要使用的技术栈和架构
3. **团队文化**：工作氛围、加班情况、管理风格
4. **薪资福利**：薪资水平、股票期权、福利待遇
5. **发展前景**：行业地位、增长趋势、潜在风险
6. **面试流程**：面试轮次、常见题型、面试风格
7. **员工评价**：优缺点总结、离职率相关信息
8. **求职建议**：针对该公司的准备策略
"""


async def company_research(
    company_name: str,
    llm_client,
    model: str,
    tavily_api_key: str = "",
) -> dict:
    """对目标公司进行深度调研。先搜索再分析。"""
    # 第一步：多维度搜索
    search_queries = [
        f"{company_name} 公司介绍 技术栈",
        f"{company_name} 面试经验 面试流程",
        f"{company_name} 薪资待遇 员工评价",
    ]
    all_results = []
    for query in search_queries:
        result = await search_job(
            query=query,
            tavily_api_key=tavily_api_key,
            max_results=3,
            search_depth="basic",
        )
        if result.get("results"):
            all_results.extend(result["results"])

    search_text = "\n\n".join(
        f"[{r.get('title', '')}] {r.get('snippet', '')}" for r in all_results
    ) if all_results else "未找到相关搜索结果，请根据你的知识库信息进行分析。"

    # 第二步：LLM 分析
    prompt = COMPANY_RESEARCH_PROMPT.format(
        company_name=company_name,
        search_results=search_text[:4000],
    )
    response = await llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是企业研究分析师，输出客观、全面的公司分析报告。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    return {
        "tool_name": "company_research",
        "timestamp": datetime.now().isoformat(),
        "company_name": company_name,
        "sources_count": len(all_results),
        "report": response.choices[0].message.content,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 工具 7: 生成求职信
# ═══════════════════════════════════════════════════════════════════════════════

COVER_LETTER_PROMPT = """你是一名求职文书专家。请撰写一封专业的求职信 / Cover Letter。

## 用户画像
{profile_summary}

## 目标岗位 JD
{jd_text}

## 目标公司
{company_name}

## 语言偏好
{language}

## 要求
1. 开头有吸引力，避免"尊敬的HR您好"等套路
2. 突出与 JD 最匹配的 2-3 个核心优势
3. 结合具体项目案例展示能力
4. 表达对公司/团队的了解和兴趣
5. 结尾有明确的行动号召
6. 控制在 300-500 字（中文）或 250-400 words（英文）
"""


async def cover_letter_gen(
    jd_text: str,
    llm_client,
    model: str,
    profile_summary: str = "",
    company_name: str = "",
    language: str = "zh",
) -> dict:
    """生成定制化的求职信。"""
    prompt = COVER_LETTER_PROMPT.format(
        profile_summary=profile_summary or "暂无",
        jd_text=jd_text,
        company_name=company_name or "未指定",
        language="中文" if language == "zh" else "English",
    )
    response = await llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是资深求职文书专家，文笔优美、逻辑清晰。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=2000,
    )
    return {
        "tool_name": "cover_letter_gen",
        "timestamp": datetime.now().isoformat(),
        "company_name": company_name,
        "cover_letter": response.choices[0].message.content,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 工具 8: 技能差距分析
# ═══════════════════════════════════════════════════════════════════════════════

SKILL_GAP_PROMPT = """你是一名职业发展顾问。请对比用户现有技能与目标岗位要求，输出差距分析报告。

## 用户技能/画像
{profile_summary}

## 用户简历信息
{resume_info}

## 目标岗位 JD
{jd_text}

## 分析要求
1. **技能匹配矩阵**：逐项列出 JD 要求 vs 用户现有技能，标注 ✅ 匹配 / ⚠️ 部分匹配 / ❌ 缺失
2. **核心差距**：最关键的 3-5 个差距，按优先级排序
3. **学习路线**：针对每个差距，给出具体学习资源和时间估算
4. **速赢策略**：1-2 周内可以快速补齐的短板
5. **长期规划**：需要 1-3 个月持续学习的技能
6. **匹配度评估**：给出整体匹配度百分比
"""


async def skill_gap_analysis(
    jd_text: str,
    llm_client,
    model: str,
    profile_summary: str = "",
    resume_info: str = "",
) -> dict:
    """分析用户技能与目标 JD 之间的差距。"""
    prompt = SKILL_GAP_PROMPT.format(
        profile_summary=profile_summary or "暂无",
        resume_info=resume_info or "暂无",
        jd_text=jd_text,
    )
    response = await llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是职业发展顾问，分析精准且提供可执行的建议。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    return {
        "tool_name": "skill_gap_analysis",
        "timestamp": datetime.now().isoformat(),
        "analysis": response.choices[0].message.content,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 工具 9: 面试复盘
# ═══════════════════════════════════════════════════════════════════════════════

INTERVIEW_REVIEW_PROMPT = """你是一名面试复盘专家。请根据用户提供的面试经历进行深度复盘分析。

## 面试经历描述
{interview_description}

## 面试公司/岗位
{company_and_position}

## 复盘要求
1. **面试流程还原**：梳理面试的关键环节
2. **表现评估**：
   - 回答得好的部分（值得保持）
   - 回答欠佳的部分（需要改进）
   - 遗漏或未被问到但应该准备的知识点
3. **失误分析**：具体失误的原因和改进方案
4. **知识盲区**：暴露出的知识盲区及补齐方案
5. **心态与表达**：沟通表达和心态管理的建议
6. **针对性提升计划**：按优先级给出下一步行动清单
7. **同类公司准备**：如果面试同类公司，需要额外准备什么
"""


async def interview_review(
    interview_description: str,
    llm_client,
    model: str,
    company_and_position: str = "",
) -> dict:
    """面试复盘分析，提取教训和改进建议。"""
    prompt = INTERVIEW_REVIEW_PROMPT.format(
        interview_description=interview_description,
        company_and_position=company_and_position or "未指定",
    )
    response = await llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是面试复盘专家，善于从面试经历中提取关键教训。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    return {
        "tool_name": "interview_review",
        "timestamp": datetime.now().isoformat(),
        "review": response.choices[0].message.content,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 工具 10: 职业路径规划
# ═══════════════════════════════════════════════════════════════════════════════

CAREER_PATH_PROMPT = """你是一名资深职业规划师。请为用户制定个性化的职业发展路径。

## 用户画像
{profile_summary}

## 当前阶段
{current_stage}

## 目标方向
{target_direction}

## 规划要求
1. **现状诊断**：当前职业阶段的优劣势
2. **短期目标（3-6 个月）**：近期应聚焦的方向和具体行动
3. **中期目标（1-2 年）**：职级/薪资/技能的发展里程碑
4. **长期目标（3-5 年）**：职业终极目标和实现路径
5. **关键转折点**：需要做出选择的关键节点
6. **技能投资组合**：应该在哪些技能上投入时间（核心技能 vs 增值技能）
7. **风险与备选方案**：可能的风险和 Plan B
8. **行业趋势**：相关行业/技术的发展趋势如何影响规划
"""


async def career_path_planner(
    llm_client,
    model: str,
    profile_summary: str = "",
    current_stage: str = "",
    target_direction: str = "",
) -> dict:
    """制定个性化职业发展路径。"""
    prompt = CAREER_PATH_PROMPT.format(
        profile_summary=profile_summary or "暂无",
        current_stage=current_stage or "未指定",
        target_direction=target_direction or "未指定",
    )
    response = await llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是资深职业规划师，规划务实且有前瞻性。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=4096,
    )
    return {
        "tool_name": "career_path_planner",
        "timestamp": datetime.now().isoformat(),
        "career_plan": response.choices[0].message.content,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 工具 11: ATS 简历关键词优化
# ═══════════════════════════════════════════════════════════════════════════════

ATS_OPTIMIZER_PROMPT = """你是一名 ATS（Applicant Tracking System）优化专家。

## 目标 JD
{jd_text}

## 用户简历内容
{resume_text}

## 优化要求
1. **关键词提取**：从 JD 中提取 ATS 可能扫描的所有关键词
2. **匹配度检查**：逐一检查简历中是否包含这些关键词
3. **缺失关键词**：列出简历中缺失但 JD 要求的关键词
4. **嵌入建议**：对每个缺失关键词，建议如何自然地嵌入简历
5. **格式建议**：ATS 友好的简历格式注意事项
6. **ATS 评分预估**：预估当前简历通过 ATS 初筛的概率
7. **优化后简历片段**：给出关键部分的优化示例
"""


async def resume_keyword_optimizer(
    jd_text: str,
    resume_text: str,
    llm_client,
    model: str,
) -> dict:
    """ATS 关键词优化，提升简历通过机器筛选的概率。"""
    prompt = ATS_OPTIMIZER_PROMPT.format(
        jd_text=jd_text,
        resume_text=resume_text,
    )
    response = await llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是 ATS 优化专家，精通各类招聘系统的关键词匹配机制。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=4096,
    )
    return {
        "tool_name": "resume_keyword_optimizer",
        "timestamp": datetime.now().isoformat(),
        "optimization": response.choices[0].message.content,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 工具注册表 — OpenAI function calling 格式
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS = [
    # ── 1. JD 分析 ────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "analyze_jd",
            "description": (
                "深度分析岗位 JD，返回技术要求解读、简历匹配建议、面试准备清单。"
                "当用户粘贴 JD 或说'帮我分析这个岗位'时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "jd_text": {
                        "type": "string",
                        "description": "JD 全文或岗位描述文字",
                    }
                },
                "required": ["jd_text"],
            },
        },
    },
    # ── 2. 联网搜索 ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "search_job",
            "description": (
                "联网搜索公司信息、岗位行情、面试经验、技术资料。"
                "当用户询问某公司背景、面试题、薪资行情时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，例如'字节跳动 后端工程师 面试经验 2025'",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最多返回结果数，默认 5",
                    },
                    "search_depth": {
                        "type": "string",
                        "description": "搜索深度：basic 或 advanced",
                        "enum": ["basic", "advanced"],
                    },
                },
                "required": ["query"],
            },
        },
    },
    # ── 3. 简历生成/优化 ──────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "generate_resume",
            "description": (
                "根据用户画像和目标 JD 生成或优化简历。"
                "当用户说'帮我写简历'、'优化简历'、'针对这个岗位改简历'时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "jd_text": {
                        "type": "string",
                        "description": "目标岗位 JD 文本（可选，有则针对性更强）",
                    },
                    "existing_resume": {
                        "type": "string",
                        "description": "用户现有简历文本（可选）",
                    },
                },
                "required": [],
            },
        },
    },
    # ── 4. 模拟面试 ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "mock_interview",
            "description": (
                "生成模拟面试题目。"
                "当用户说'模拟面试'、'出几道面试题'、'帮我练习面试'时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "string",
                        "description": "目标岗位名称，例如'后端开发工程师'",
                    },
                    "interview_type": {
                        "type": "string",
                        "description": "面试类型：技术面、行为面、系统设计、HR面",
                        "enum": ["技术面", "行为面", "系统设计", "HR面", "综合"],
                    },
                    "difficulty": {
                        "type": "string",
                        "description": "难度等级",
                        "enum": ["初级", "中等", "高级", "专家"],
                    },
                    "tech_stack": {
                        "type": "string",
                        "description": "技术栈，例如'Java, Spring Boot, MySQL, Redis'",
                    },
                    "question_count": {
                        "type": "integer",
                        "description": "出题数量，默认 8",
                    },
                },
                "required": ["position"],
            },
        },
    },
    # ── 5. 回答评估 ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "evaluate_answer",
            "description": (
                "评估用户的面试回答，给出评分和改进建议。"
                "当用户回答了面试题并希望获得评价时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "面试问题",
                    },
                    "answer": {
                        "type": "string",
                        "description": "用户的面试回答",
                    },
                    "position": {
                        "type": "string",
                        "description": "目标岗位（可选）",
                    },
                },
                "required": ["question", "answer"],
            },
        },
    },
    # ── 6. 公司调研 ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "company_research",
            "description": (
                "深度调研目标公司，包括技术栈、面试流程、薪资福利、企业文化。"
                "当用户说'调研一下XX公司'、'XX公司怎么样'时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "公司名称，例如'字节跳动'",
                    },
                },
                "required": ["company_name"],
            },
        },
    },
    # ── 7. 求职信生成 ─────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "cover_letter_gen",
            "description": (
                "生成针对特定公司和岗位的求职信 / Cover Letter。"
                "当用户说'写求职信'、'写 Cover Letter'时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "jd_text": {
                        "type": "string",
                        "description": "目标岗位 JD 文本",
                    },
                    "company_name": {
                        "type": "string",
                        "description": "目标公司名称",
                    },
                    "language": {
                        "type": "string",
                        "description": "语言偏好: zh（中文）或 en（英文）",
                        "enum": ["zh", "en"],
                    },
                },
                "required": ["jd_text"],
            },
        },
    },
    # ── 8. 技能差距分析 ──────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "skill_gap_analysis",
            "description": (
                "对比用户技能与目标 JD 要求，输出技能差距和学习路线。"
                "当用户说'我和这个岗位差多少'、'差距分析'、'我能胜任吗'时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "jd_text": {
                        "type": "string",
                        "description": "目标岗位 JD 文本",
                    },
                    "resume_info": {
                        "type": "string",
                        "description": "用户简历或技能描述（可选，系统会自动补充画像信息）",
                    },
                },
                "required": ["jd_text"],
            },
        },
    },
    # ── 9. 面试复盘 ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "interview_review",
            "description": (
                "对用户的面试经历进行复盘分析，总结教训和改进方向。"
                "当用户说'面试完了帮我复盘'、'今天面试挂了'、'面试总结'时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "interview_description": {
                        "type": "string",
                        "description": "面试经历的详细描述",
                    },
                    "company_and_position": {
                        "type": "string",
                        "description": "面试公司和岗位，例如'字节跳动 后端开发'",
                    },
                },
                "required": ["interview_description"],
            },
        },
    },
    # ── 10. 职业路径规划 ──────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "career_path_planner",
            "description": (
                "为用户制定个性化职业发展路径和成长规划。"
                "当用户说'帮我规划职业'、'未来怎么发展'、'职业方向'时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "current_stage": {
                        "type": "string",
                        "description": "当前职业阶段描述，例如'3年Java开发，目前在中厂'",
                    },
                    "target_direction": {
                        "type": "string",
                        "description": "目标方向，例如'大厂P7'、'技术管理'、'架构师'",
                    },
                },
                "required": [],
            },
        },
    },
    # ── 11. ATS 关键词优化 ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "resume_keyword_optimizer",
            "description": (
                "分析 JD 关键词，优化简历以通过 ATS 机器筛选。"
                "当用户说'简历过不了筛选'、'ATS优化'、'关键词优化'时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "jd_text": {
                        "type": "string",
                        "description": "目标岗位 JD 文本",
                    },
                    "resume_text": {
                        "type": "string",
                        "description": "用户简历全文",
                    },
                },
                "required": ["jd_text", "resume_text"],
            },
        },
    },
]
