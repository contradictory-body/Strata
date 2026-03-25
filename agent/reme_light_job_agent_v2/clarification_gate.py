"""
clarification_gate.py
=====================
请求澄清门控机制 (Clarification Gate)

在触发长期记忆检索之前，对用户输入执行双层模糊性检测：
  Layer 1 — 规则层：正则匹配，零 LLM 开销，毫秒级返回
  Layer 2 — LLM 兜底：仅在规则层置信度 < 0.7 时触发

拦截四类模糊输入：
  reference_unclear   : 指代词无法解引用（"这个"、"那个"）
  intent_unclear      : 过泛动词意图不明（"看看"、"了解一下"）
  constraint_missing  : 求职核心槽位严重缺失
  intent_mixed        : 单条输入混杂多个不相关意图

失败保守策略：LLM 兜底失败时默认放行，不阻塞主流程。
最大追问上限：ask_count >= 2 时强制放行，防止死循环。
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("ClarifyGate")

# ──────────────────────────────────────────────────────────────────────────────
# 常量与正则
# ──────────────────────────────────────────────────────────────────────────────

# 模糊类型枚举
AMBIGUITY_TYPES = {
    "reference_unclear":   "存在无法解引用的指代词",
    "intent_unclear":      "动词过泛，意图不明",
    "constraint_missing":  "核心槽位严重缺失",
    "intent_mixed":        "一句话包含多个不相关意图",
}

# 最大追问次数
MAX_ASK_COUNT = 2

# ── 规则层正则 ────────────────────────────────────────────────────────────────

VAGUE_REFERENCES = re.compile(
    r"(这个|那个|上次|之前|刚才|它|这份|那份|这家|那家|上面|这里|那里)"
)

VAGUE_VERBS = re.compile(
    r"^(看看|了解|知道|说说|讲讲|介绍|帮我|看一下|了解一下|说一说|介绍一下|讲一讲)\s*[吗吧？?。]?$"
)

# 可解引用的对象关键词库（判断上下文中是否有具体指代对象）
REFERRABLE_KEYWORDS = re.compile(
    r"(岗位|职位|JD|工作|简历|公司|面试|[Oo]ffer|"

    # 职能方向
    r"工程师|开发|产品|算法|运营|设计|测试|架构师|"
    r"前端|后端|全栈|客户端|移动端|数据|[Aa][Ii]|机器学习|"
    r"分析师|顾问|销售|市场|品牌|法务|财务|[Hh][Rr]|"

    # 知名公司
    r"字节|抖音|[Tt]ik[Tt]ok|阿里|淘宝|天猫|蚂蚁|"
    r"腾讯|微信|京东|美团|百度|网易|滴滴|拼多多|"
    r"华为|小米|[Oo][Pp][Pp][Oo]|[Vv]ivo|联想|快手|[Bb]ilibili|"
    r"微软|谷歌|苹果|[Mm]eta|亚马逊|[Aa]dobe|[Ss]alesforce|"
    r"蔚来|理想|小鹏|比亚迪|宁德时代|"
    r"[Pp]ing[Cc][Aa][Pp]|[Dd]atabricks|[Ss]nowflake|[Cc]onfluent|"
    r"[Ss]hopee|[Gg]rab|[Mm]icrosoft|[Gg]oogle|[Aa]pple|[Aa]mazon|"

    # 求职流程节点
    r"投递|笔试|一面|二面|三面|[Hh][Rr]面|终面|复试|"
    r"实习|校招|社招|猎头|内推|背调|薪资|薪酬|包裹|"

    # 具体技术栈
    r"[Pp]ython|[Jj]ava|[Gg]o(?:lang)?|[Rr]ust|[Cc]\+\+|"
    r"[Jj]ava[Ss]cript|[Tt]ype[Ss]cript|[Kk]otlin|[Ss]wift|"
    r"[Rr]eact|[Vv]ue|[Ss]pring|[Ff]ast[Aa][Pp][Ii]|[Dd]jango|[Ff]lask|"
    r"[Mm]y[Ss][Qq][Ll]|[Pp]ostgre[Ss][Qq][Ll]|[Rr]edis|[Mm]ongo[Dd][Bb]|"
    r"[Kk]afka|[Ee]lasticsearch|[Dd]ocker|[Kk]ubernetes|"
    r"[Aa][Ww][Ss]|[Aa]zure|[Gg][Cc][Pp]|[Ll]inux|"

    # 文件与内容类型
    r"简历|作品集|项目|经历|背景|证书|[Pp]ortfolio)"
)

# 求职意图关键词（用于触发槽位缺失检测）
JOB_INTENT_PATTERN = re.compile(
    r"(找工作|投简历|面试|岗位|职位|[Jj][Dd]|薪资|跳槽|求职|换工作|投递|应聘)"
)

# 核心槽位关键词
SLOT_KEYWORDS: dict[str, list[str]] = {
    "target_position": [
        "工程师", "开发", "产品", "算法", "运营", "设计", "测试",
        "架构师", "前端", "后端", "全栈", "岗位", "职位",
        "分析师", "数据", "AI", "机器学习", "研发",
    ],
    "city": [
        "北京", "上海", "广州", "深圳", "杭州", "成都", "武汉",
        "西安", "南京", "苏州", "重庆", "天津", "厦门", "远程", "异地", "城市",
    ],
    "intent": [
        "投递", "分析", "准备", "搜索", "修改", "推荐", "评估",
        "对比", "了解", "优化", "改进", "查找", "看",
    ],
}

# 意图正则（用于混杂检测）
INTENT_PATTERNS = [
    re.compile(r"(改|修改|优化|完善).{0,8}(简历|履历)"),
    re.compile(r"(找|搜|推荐|查找).{0,8}(岗位|工作|职位|[Jj][Dd])"),
    re.compile(r"(准备|复习|学习).{0,8}(面试|题目|知识点)"),
    re.compile(r"(分析|解读|看看).{0,8}([Jj][Dd]|岗位描述|职位要求)"),
]

# 反问模板
CLARIFY_TEMPLATES: dict[str, str] = {
    "reference_unclear": "你提到的「{ref}」具体指的是什么？请描述一下。",
    "intent_unclear": (
        "你希望我帮你做什么？\n"
        "A. 分析岗位 JD\n"
        "B. 修改简历\n"
        "C. 搜索面试题\n"
        "D. 其他（请说明）"
    ),
    "constraint_missing": (
        "为了给你更精准的建议，请补充以下信息：\n{missing_prompts}"
    ),
    "intent_mixed": (
        "你的问题包含多个方向，本次你最想先处理哪一个？\n"
        "A. 修改简历\n"
        "B. 搜索岗位\n"
        "C. 准备面试"
    ),
}

SLOT_PROMPTS: dict[str, str] = {
    "target_position": "· 目标岗位是什么？（如：后端工程师、产品经理）",
    "city":            "· 目标城市？（如：北京、上海，或不限城市）",
    "intent":          "· 你想做什么？（如：投递、面试准备、薪资了解）",
}

LLM_GATE_PROMPT = """\
你是一个意图分析器。请判断以下用户输入是否需要澄清。

用户画像：{profile}
最近对话：{recent_context}
历史记忆：{memory_hint}
当前输入：{user_input}

只需输出 JSON，不要其他文字：
{{
  "is_ambiguous": true或false,
  "ambiguity_type": "reference_unclear或intent_unclear或constraint_missing或intent_mixed或null",
  "missing_slots": ["slot1", "slot2"],
  "reason": "一句话说明原因"
}}
"""

NORMALIZE_PROMPT = """\
你是一个查询标准化器。请将下方的用户信息合并，
输出一个结构化的标准检索 Query，格式固定为：
[任务目标] | 岗位={岗位} | 城市={城市} | 约束={其他约束}

用户画像：{profile}
原始请求：{original_input}
用户补充：{clarification}
历史记忆提示：{memory_hint}

只输出一行标准化 Query，不要其他文字。
示例：岗位推荐 | 岗位=后端工程师 | 城市=北京 | 约束=薪资35K以上,互联网行业
"""


# ──────────────────────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    """门控判定结果。

    Attributes:
        is_ambiguous:   是否需要澄清
        ambiguity_type: 模糊类型（reference_unclear / intent_unclear /
                        constraint_missing / intent_mixed / None）
        missing_slots:  缺失的槽位名称列表
        confidence:     规则层置信度（1.0=确定，<0.7=升级 LLM）
        reason:         判定原因（调试用）
    """
    is_ambiguous:   bool
    ambiguity_type: Optional[str]
    missing_slots:  list[str] = field(default_factory=list)
    confidence:     float = 1.0
    reason:         str = ""


# ──────────────────────────────────────────────────────────────────────────────
# 核心类
# ──────────────────────────────────────────────────────────────────────────────

class ClarificationGate:
    """请求澄清门控，集成进 JobAgentV2。

    Usage:
        gate = ClarificationGate(llm_client, model, reme)
        result = await gate.clarify_gate(user_input, ctx)
    """

    def __init__(self, llm_client, model: str, reme=None):
        """
        Args:
            llm_client: openai.AsyncOpenAI 实例
            model:      LLM 模型名称
            reme:       ReMeLight 实例，用于轻量 memory_search
        """
        self._llm    = llm_client
        self._model  = model
        self._reme   = reme

    # ── 公开入口 ──────────────────────────────────────────────────────────────

    async def clarify_gate(
        self,
        user_input:      str,
        recent_messages: list[dict],
        profile_summary: str,
    ) -> GateResult:
        """双层门控入口。

        Args:
            user_input:      用户当前输入文本
            recent_messages: self.messages 的最近 4 条（OpenAI dict 格式）
            profile_summary: ProfileManager.to_summary_str() 返回的画像摘要

        Returns:
            GateResult，调用方根据 is_ambiguous 决定是否中断本轮。
        """
        ctx = await self._assemble_gate_context(
            user_input, recent_messages, profile_summary
        )
        rule_result = self._rule_based_gate(ctx)

        if rule_result.confidence >= 0.7:
            return rule_result

        logger.info(f"规则层置信度 {rule_result.confidence:.2f}，升级 LLM 门控")
        return await self._llm_gate(ctx)

    async def normalize_query(
        self,
        current_input:   str,
        pending:         Optional[dict],
        profile_summary: str,
        recent_messages: list[dict],
    ) -> str:
        """将输入标准化为结构化检索 Query。

        Args:
            current_input:   当前用户输入（澄清补充或清晰请求）
            pending:         pending_clarification 状态（None 表示无挂起）
            profile_summary: 用户画像摘要
            recent_messages: 最近对话记录

        Returns:
            "[任务目标] | 岗位=X | 城市=Y | 约束=Z" 格式字符串
        """
        memory_hint = await self._get_memory_hint(current_input)
        original    = pending["original_input"] if pending else current_input
        clarif      = current_input if pending else ""

        prompt = NORMALIZE_PROMPT.format(
            profile=profile_summary,
            original_input=original,
            clarification=clarif,
            memory_hint=memory_hint,
        )
        try:
            resp = await self._llm.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=150,
            )
            normalized = resp.choices[0].message.content.strip()
            logger.info(f"标准化 Query: {normalized}")
            return normalized
        except Exception as e:
            logger.warning(f"Query 标准化失败，使用原始输入: {e}")
            return current_input

    @staticmethod
    def check_slot_completeness(normalized_query: str) -> bool:
        """检查标准化 Query 的槽位完整度。

        要求岗位、城市、任务目标三个核心槽位中至少填充两个，
        才允许触发偏好写入。

        Args:
            normalized_query: normalize_query() 返回的标准化字符串

        Returns:
            True 表示槽位充足，可触发偏好写入。
        """
        filled = 0
        if re.search(r"岗位=[^|]{1,}", normalized_query):
            filled += 1
        if re.search(r"城市=[^|]{1,}", normalized_query):
            filled += 1
        if re.search(r"\[.+?\]", normalized_query):
            filled += 1
        return filled >= 2

    @staticmethod
    def generate_clarify_question(
        gate_result: GateResult, user_input: str
    ) -> str:
        """根据模糊类型从模板库生成最小必要反问。

        优先使用模板，不调用 LLM，最小化延迟。

        Args:
            gate_result: 门控判定结果
            user_input:  原始用户输入（用于提取指代词）

        Returns:
            反问字符串，直接展示给用户。
        """
        atype = gate_result.ambiguity_type

        if atype == "reference_unclear":
            refs = VAGUE_REFERENCES.findall(user_input)
            ref  = refs[0] if refs else "提到的内容"
            return CLARIFY_TEMPLATES["reference_unclear"].format(ref=ref)

        if atype == "intent_unclear":
            return CLARIFY_TEMPLATES["intent_unclear"]

        if atype == "constraint_missing":
            prompts = "\n".join(
                SLOT_PROMPTS[s]
                for s in gate_result.missing_slots
                if s in SLOT_PROMPTS
            )
            return CLARIFY_TEMPLATES["constraint_missing"].format(
                missing_prompts=prompts or "请补充目标岗位、城市或求职意图。"
            )

        if atype == "intent_mixed":
            return CLARIFY_TEMPLATES["intent_mixed"]

        return "请问您的问题能否说得更具体一些？"

    @staticmethod
    def should_force_proceed(pending: Optional[dict]) -> bool:
        """判断是否已达最大追问次数，需要强制放行。

        Args:
            pending: pending_clarification 状态对象

        Returns:
            True 表示应强制放行，不再追问。
        """
        if pending is None:
            return False
        return pending.get("ask_count", 0) >= MAX_ASK_COUNT

    # ── 内部实现 ──────────────────────────────────────────────────────────────

    async def _assemble_gate_context(
        self,
        user_input:      str,
        recent_messages: list[dict],
        profile_summary: str,
    ) -> dict:
        """组装门控检测上下文。"""
        recent_text = "\n".join(
            f"{m['role']}: {str(m.get('content', ''))[:100]}"
            for m in recent_messages[-4:]
        )
        memory_hint = await self._get_memory_hint(user_input)
        return {
            "user_input":     user_input,
            "recent_context": recent_text,
            "profile":        profile_summary,
            "memory_hint":    memory_hint,
        }

    async def _get_memory_hint(self, query: str) -> str:
        """发起轻量 memory_search，仅取 Top-2 用于辅助解引用。"""
        if self._reme is None:
            return ""
        try:
            import json as _json
            tool_response = await self._reme.memory_search(
                query=query, max_results=2, min_score=0.5
            )
            raw     = tool_response.content[0].text if tool_response.content else "[]"
            results = _json.loads(raw)
            return " | ".join(r.get("text", "")[:80] for r in results[:2] if r.get("text"))
        except Exception:
            return ""

    def _rule_based_gate(self, ctx: dict) -> GateResult:
        """规则层门控，短路执行，首条匹配即返回。"""
        text = ctx["user_input"]

        # ── 规则 1：指代词检测 ────────────────────────────────────────────────
        if VAGUE_REFERENCES.search(text):
            combined = ctx["recent_context"] + " " + ctx["memory_hint"]
            if not combined.strip():
                return GateResult(
                    is_ambiguous=True,
                    ambiguity_type="reference_unclear",
                    confidence=1.0,
                    reason=f"存在指代词且无任何上下文: {VAGUE_REFERENCES.findall(text)}",
                )
            if REFERRABLE_KEYWORDS.search(combined):
                # 上下文里有可解引用的对象，直接放行
                return GateResult(
                    is_ambiguous=False,
                    ambiguity_type=None,
                    confidence=0.9,
                    reason="上下文中存在可解引用的具体对象，放行",
                )
            # 有上下文但无可解引用对象，进入灰色地带
            return GateResult(
                is_ambiguous=False,
                ambiguity_type=None,
                confidence=0.6,
                reason="有上下文但无可解引用对象关键词，交 LLM 判断",
            )

        # ── 规则 2：过泛动词检测 ──────────────────────────────────────────────
        if VAGUE_VERBS.match(text.strip()):
            return GateResult(
                is_ambiguous=True,
                ambiguity_type="intent_unclear",
                confidence=1.0,
                reason=f"整句为过泛动词，意图不明: {text}",
            )

        # ── 规则 3：核心槽位缺失检测 ──────────────────────────────────────────
        if JOB_INTENT_PATTERN.search(text):
            profile = ctx.get("profile", "")
            missing = []
            for slot, keywords in SLOT_KEYWORDS.items():
                # 画像里已有该槽位，不算缺失
                slot_label = {"target_position": "目标岗位", "city": "目标城市", "intent": "意图"}.get(slot, slot)
                if slot_label in profile and "暂未填写" not in profile:
                    continue
                if not any(kw in text for kw in keywords):
                    missing.append(slot)

            if len(missing) >= 2:
                return GateResult(
                    is_ambiguous=True,
                    ambiguity_type="constraint_missing",
                    missing_slots=missing,
                    confidence=1.0,
                    reason=f"求职意图明确但核心槽位缺失: {missing}",
                )

        # ── 规则 4：意图混杂检测 ──────────────────────────────────────────────
        matched = sum(1 for p in INTENT_PATTERNS if p.search(text))
        if matched >= 2:
            return GateResult(
                is_ambiguous=True,
                ambiguity_type="intent_mixed",
                confidence=0.9,
                reason=f"检测到 {matched} 个不相关意图",
            )

        # ── 默认：清晰，放行 ──────────────────────────────────────────────────
        return GateResult(
            is_ambiguous=False,
            ambiguity_type=None,
            confidence=1.0,
            reason="规则层判定清晰，放行",
        )

    async def _llm_gate(self, ctx: dict) -> GateResult:
        """LLM 兜底分类，只在规则层置信度 < 0.7 时调用。"""
        prompt = LLM_GATE_PROMPT.format(**ctx)
        try:
            resp = await self._llm.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content.strip()
            raw = (
                raw.removeprefix("```json")
                   .removeprefix("```")
                   .removesuffix("```")
                   .strip()
            )
            data = json.loads(raw)
            return GateResult(
                is_ambiguous=data.get("is_ambiguous", False),
                ambiguity_type=data.get("ambiguity_type"),
                missing_slots=data.get("missing_slots", []),
                confidence=1.0,
                reason=data.get("reason", "LLM 兜底判定"),
            )
        except Exception as e:
            logger.warning(f"LLM 门控失败，保守放行: {e}")
            return GateResult(
                is_ambiguous=False,
                ambiguity_type=None,
                confidence=0.5,
                reason=f"LLM 门控异常，保守放行: {e}",
            )
