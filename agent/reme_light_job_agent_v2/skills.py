"""
skills.py — Agent Skills（智能技能编排层）

Agent Skills 是在基础工具之上的高阶编排能力，
将多个工具按照特定场景组合成端到端的工作流。

技能 1: FullPreparationSkill    — 完整备战技能（JD分析 + 差距分析 + 模拟面试）
技能 2: ApplicationPackageSkill — 申请材料包技能（简历优化 + 求职信 + ATS优化）
技能 3: PostInterviewSkill      — 面后复盘技能（面试复盘 + 知识补齐搜索 + 行动计划）

设计理念：
- 每个 Skill 封装一条完整链路，内部自动编排多个工具调用
- 工具调用之间有数据流传递（上一个工具的输出作为下一个的输入）
- 支持中间结果缓存和增量输出
- 通过 SkillRegistry 统一注册，Agent 可以自动选择合适的 Skill
"""

import json
from datetime import datetime
from typing import Optional, Callable, Any

from tools import (
    analyze_jd,
    search_job,
    generate_resume,
    mock_interview,
    skill_gap_analysis,
    cover_letter_gen,
    resume_keyword_optimizer,
    interview_review,
    company_research,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Skill 基类
# ═══════════════════════════════════════════════════════════════════════════════

class BaseSkill:
    """Agent Skill 基类。"""

    name: str = ""
    description: str = ""
    # 该 Skill 内部会调用的工具列表（用于可观测性）
    tool_chain: list[str] = []

    def __init__(self, llm_client, model: str, **kwargs):
        self.llm_client = llm_client
        self.model = model
        self.kwargs = kwargs
        self.step_results: dict[str, Any] = {}
        self.on_step: Optional[Callable[[str, str], None]] = kwargs.get("on_step")

    def _report_step(self, step_name: str, status: str = "running"):
        """回调通知当前步骤进展。"""
        if self.on_step:
            self.on_step(step_name, status)

    async def execute(self, **params) -> dict:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════════
# 技能 1: 完整备战技能
# ═══════════════════════════════════════════════════════════════════════════════

class FullPreparationSkill(BaseSkill):
    """
    完整备战技能：一键完成从 JD 分析到面试准备的全链路。

    工具链: analyze_jd → skill_gap_analysis → company_research → mock_interview

    输入: JD 文本 + 用户画像
    输出: 综合备战报告
    """

    name = "full_preparation"
    description = (
        "一键完整备战：自动执行 JD 分析 → 技能差距分析 → 公司调研 → 模拟面试出题，"
        "生成一份完整的面试准备方案。"
        "当用户说'帮我全面准备这个岗位'、'一键备战'时触发。"
    )
    tool_chain = ["analyze_jd", "skill_gap_analysis", "company_research", "mock_interview"]

    async def execute(
        self,
        jd_text: str,
        profile_summary: str = "",
        company_name: str = "",
        tavily_api_key: str = "",
        **kwargs,
    ) -> dict:
        report_sections = {}

        # ── Step 1: JD 分析 ──
        self._report_step("analyze_jd", "running")
        jd_result = await analyze_jd(
            jd_text=jd_text,
            llm_client=self.llm_client,
            model=self.model,
        )
        report_sections["jd_analysis"] = jd_result.get("analysis", "")
        self.step_results["analyze_jd"] = jd_result
        self._report_step("analyze_jd", "done")

        # ── Step 2: 技能差距分析 ──
        self._report_step("skill_gap_analysis", "running")
        gap_result = await skill_gap_analysis(
            jd_text=jd_text,
            llm_client=self.llm_client,
            model=self.model,
            profile_summary=profile_summary,
        )
        report_sections["skill_gap"] = gap_result.get("analysis", "")
        self.step_results["skill_gap_analysis"] = gap_result
        self._report_step("skill_gap_analysis", "done")

        # ── Step 3: 公司调研（如果提供了公司名） ──
        if company_name:
            self._report_step("company_research", "running")
            company_result = await company_research(
                company_name=company_name,
                llm_client=self.llm_client,
                model=self.model,
                tavily_api_key=tavily_api_key,
            )
            report_sections["company_report"] = company_result.get("report", "")
            self.step_results["company_research"] = company_result
            self._report_step("company_research", "done")

        # ── Step 4: 模拟面试出题 ──
        # 从 JD 分析中提取岗位名称
        self._report_step("mock_interview", "running")
        position = company_name + " 目标岗位" if company_name else "目标岗位"
        interview_result = await mock_interview(
            position=position,
            llm_client=self.llm_client,
            model=self.model,
            interview_type="综合",
            difficulty="中等",
            question_count=6,
        )
        report_sections["mock_questions"] = interview_result.get("questions", "")
        self.step_results["mock_interview"] = interview_result
        self._report_step("mock_interview", "done")

        # ── 合成最终报告 ──
        final_report = self._compose_report(report_sections, company_name)

        return {
            "tool_name": "full_preparation_skill",
            "timestamp": datetime.now().isoformat(),
            "steps_completed": list(report_sections.keys()),
            "report": final_report,
        }

    def _compose_report(self, sections: dict, company_name: str) -> str:
        parts = [f"# 🎯 完整备战报告 — {company_name or '目标岗位'}\n"]

        if "jd_analysis" in sections:
            parts.append("## 一、JD 深度分析\n")
            parts.append(sections["jd_analysis"])
            parts.append("\n---\n")

        if "skill_gap" in sections:
            parts.append("## 二、技能差距分析\n")
            parts.append(sections["skill_gap"])
            parts.append("\n---\n")

        if "company_report" in sections:
            parts.append("## 三、公司调研报告\n")
            parts.append(sections["company_report"])
            parts.append("\n---\n")

        if "mock_questions" in sections:
            parts.append("## 四、模拟面试题\n")
            parts.append(sections["mock_questions"])

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# 技能 2: 申请材料包技能
# ═══════════════════════════════════════════════════════════════════════════════

class ApplicationPackageSkill(BaseSkill):
    """
    申请材料包技能：自动生成完整的求职申请材料。

    工具链: generate_resume → resume_keyword_optimizer → cover_letter_gen

    输入: JD 文本 + 用户画像 + 现有简历
    输出: 优化后的简历 + ATS 建议 + 求职信
    """

    name = "application_package"
    description = (
        "一键生成完整申请材料包：自动优化简历 → ATS 关键词优化 → 生成求职信，"
        "输出可直接投递的全套材料。"
        "当用户说'帮我准备申请材料'、'一键投递'时触发。"
    )
    tool_chain = ["generate_resume", "resume_keyword_optimizer", "cover_letter_gen"]

    async def execute(
        self,
        jd_text: str,
        profile_summary: str = "",
        existing_resume: str = "",
        company_name: str = "",
        language: str = "zh",
        **kwargs,
    ) -> dict:
        package = {}

        # ── Step 1: 简历生成/优化 ──
        self._report_step("generate_resume", "running")
        resume_result = await generate_resume(
            profile_summary=profile_summary,
            llm_client=self.llm_client,
            model=self.model,
            jd_text=jd_text,
            existing_resume=existing_resume,
        )
        resume_content = resume_result.get("resume_content", "")
        package["resume"] = resume_content
        self._report_step("generate_resume", "done")

        # ── Step 2: ATS 关键词优化（基于 Step 1 的简历输出） ──
        self._report_step("resume_keyword_optimizer", "running")
        ats_result = await resume_keyword_optimizer(
            jd_text=jd_text,
            resume_text=resume_content,
            llm_client=self.llm_client,
            model=self.model,
        )
        package["ats_optimization"] = ats_result.get("optimization", "")
        self._report_step("resume_keyword_optimizer", "done")

        # ── Step 3: 求职信生成 ──
        self._report_step("cover_letter_gen", "running")
        letter_result = await cover_letter_gen(
            jd_text=jd_text,
            llm_client=self.llm_client,
            model=self.model,
            profile_summary=profile_summary,
            company_name=company_name,
            language=language,
        )
        package["cover_letter"] = letter_result.get("cover_letter", "")
        self._report_step("cover_letter_gen", "done")

        final_output = self._compose_package(package, company_name)

        return {
            "tool_name": "application_package_skill",
            "timestamp": datetime.now().isoformat(),
            "steps_completed": ["generate_resume", "resume_keyword_optimizer", "cover_letter_gen"],
            "package": final_output,
        }

    def _compose_package(self, pkg: dict, company_name: str) -> str:
        parts = [f"# 📦 求职申请材料包 — {company_name or '目标岗位'}\n"]

        parts.append("## 一、优化后的简历\n")
        parts.append(pkg.get("resume", ""))
        parts.append("\n---\n")

        parts.append("## 二、ATS 关键词优化建议\n")
        parts.append(pkg.get("ats_optimization", ""))
        parts.append("\n---\n")

        parts.append("## 三、求职信\n")
        parts.append(pkg.get("cover_letter", ""))

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# 技能 3: 面后复盘技能
# ═══════════════════════════════════════════════════════════════════════════════

class PostInterviewSkill(BaseSkill):
    """
    面后复盘技能：面试结束后的完整复盘流程。

    工具链: interview_review → search_job（搜索薄弱知识点） → mock_interview（针对性出题）

    输入: 面试经历描述
    输出: 复盘报告 + 知识补齐资源 + 针对性练习题
    """

    name = "post_interview"
    description = (
        "面后完整复盘：自动分析面试表现 → 搜索薄弱知识点的学习资源 → 生成针对性练习题。"
        "当用户说'帮我复盘今天的面试'、'面试完了总结一下'时触发。"
    )
    tool_chain = ["interview_review", "search_job", "mock_interview"]

    async def execute(
        self,
        interview_description: str,
        company_and_position: str = "",
        tavily_api_key: str = "",
        **kwargs,
    ) -> dict:
        sections = {}

        # ── Step 1: 面试复盘 ──
        self._report_step("interview_review", "running")
        review_result = await interview_review(
            interview_description=interview_description,
            llm_client=self.llm_client,
            model=self.model,
            company_and_position=company_and_position,
        )
        review_text = review_result.get("review", "")
        sections["review"] = review_text
        self._report_step("interview_review", "done")

        # ── Step 2: 搜索薄弱知识点资源 ──
        # 从复盘结果中提取关键词用于搜索
        self._report_step("search_resources", "running")
        search_query = f"{company_and_position} 面试 常见知识点 学习资源"
        search_result = await search_job(
            query=search_query,
            tavily_api_key=tavily_api_key,
            max_results=5,
            search_depth="basic",
        )
        resources = search_result.get("results", [])
        sections["resources"] = "\n".join(
            f"- [{r.get('title', '')}]({r.get('url', '')})" for r in resources
        ) if resources else "未找到相关资源"
        self._report_step("search_resources", "done")

        # ── Step 3: 针对性练习题 ──
        self._report_step("targeted_practice", "running")
        position = company_and_position or "目标岗位"
        practice_result = await mock_interview(
            position=position,
            llm_client=self.llm_client,
            model=self.model,
            interview_type="技术面",
            difficulty="高级",
            question_count=5,
        )
        sections["practice"] = practice_result.get("questions", "")
        self._report_step("targeted_practice", "done")

        final_report = self._compose_review(sections, company_and_position)

        return {
            "tool_name": "post_interview_skill",
            "timestamp": datetime.now().isoformat(),
            "steps_completed": list(sections.keys()),
            "report": final_report,
        }

    def _compose_review(self, sections: dict, company_and_position: str) -> str:
        parts = [f"# 📝 面试复盘报告 — {company_and_position or '未知岗位'}\n"]

        parts.append("## 一、面试复盘分析\n")
        parts.append(sections.get("review", ""))
        parts.append("\n---\n")

        parts.append("## 二、学习资源推荐\n")
        parts.append(sections.get("resources", ""))
        parts.append("\n---\n")

        parts.append("## 三、针对性练习题\n")
        parts.append(sections.get("practice", ""))

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Skill 注册表
# ═══════════════════════════════════════════════════════════════════════════════

SKILL_REGISTRY: dict[str, type[BaseSkill]] = {
    "full_preparation": FullPreparationSkill,
    "application_package": ApplicationPackageSkill,
    "post_interview": PostInterviewSkill,
}

# 将 Skills 以 function calling 格式注册，供 LLM 选择
SKILL_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "full_preparation_skill",
            "description": FullPreparationSkill.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "jd_text": {
                        "type": "string",
                        "description": "目标岗位 JD 全文",
                    },
                    "company_name": {
                        "type": "string",
                        "description": "目标公司名称（可选，提供后会进行公司调研）",
                    },
                },
                "required": ["jd_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "application_package_skill",
            "description": ApplicationPackageSkill.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "jd_text": {
                        "type": "string",
                        "description": "目标岗位 JD 全文",
                    },
                    "existing_resume": {
                        "type": "string",
                        "description": "用户现有简历文本（可选）",
                    },
                    "company_name": {
                        "type": "string",
                        "description": "目标公司名称",
                    },
                    "language": {
                        "type": "string",
                        "description": "语言：zh 或 en",
                        "enum": ["zh", "en"],
                    },
                },
                "required": ["jd_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "post_interview_skill",
            "description": PostInterviewSkill.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "interview_description": {
                        "type": "string",
                        "description": "面试经历的详细描述",
                    },
                    "company_and_position": {
                        "type": "string",
                        "description": "面试公司和岗位",
                    },
                },
                "required": ["interview_description"],
            },
        },
    },
]
