"""
intent_query_builder.py
=======================
目标目录：Strata_v5/agent/reme_light_job_agent_v2/intent_query_builder.py

职责：
  写路径 — 从 Chunk 原文提取槽位，生成意图标准句，供 Embedding 后写入向量库。
  读路径 — 从用户 Query（含澄清补充、LLM 推测槽位）生成意图标准句，供双路检索。

核心函数：
  extract_slots(text)                                     规则层槽位提取，无 LLM，≈ 2ms
  slots_to_sentence(slots)                                槽位字典 → 标准句字符串
  build_intent_sentence_from_chunk(text, llm_client, model) 写路径异步入口
  build_intent_sentence_from_query(q, clarif, inferred)   读路径同步入口

接入点：
  写路径 → advanced_memory_manager.py :: update_memory_from_file()
  读路径 → advanced_memory_manager.py :: retrieve()
  LLM槽位 → clarification_gate.py :: GateResult.inferred_slots
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("IntentQueryBuilder")

# ──────────────────────────────────────────────────────────────────────────────
# 规则层正则
# ──────────────────────────────────────────────────────────────────────────────

_POSITION_RE = re.compile(
    r"(工程师|开发|产品经理|算法|运营|设计师?|测试|架构师|"
    r"前端|后端|全栈|客户端|移动端|数据|AI|机器学习|研发|"
    r"分析师|顾问|销售|市场|品牌|法务|财务|HR)"
)

_CITY_RE = re.compile(
    r"(北京|上海|广州|深圳|杭州|成都|武汉|西安|南京|苏州|"
    r"重庆|天津|厦门|远程|异地)"
)

_SALARY_RE = re.compile(
    r"(\d{1,3})[Kk万](?:\s*[-~～到]\s*(\d{1,3})[Kk万])?"
)

_INTENT_MAP = {
    "投递": "求职投递", "应聘": "求职投递", "找工作": "求职投递",
    "推荐": "岗位推荐", "搜索": "岗位推荐", "查找": "岗位推荐",
    "修改": "简历优化", "优化": "简历优化", "完善": "简历优化",
    "准备": "面试准备", "复习": "面试准备",
    "分析": "JD分析",  "解读": "JD分析",
    "薪资": "薪资了解", "薪酬": "薪资了解",
}

_TECH_RE = re.compile(
    r"(Python|Java|Go(?:lang)?|Rust|C\+\+|JavaScript|TypeScript|"
    r"Kotlin|Swift|React|Vue|Spring|FastAPI|Django|Flask|"
    r"MySQL|PostgreSQL|Redis|MongoDB|Kafka|Elasticsearch|"
    r"Docker|Kubernetes|AWS|Azure|GCP)",
    re.IGNORECASE,
)

_POSITION_SUFFIX = {"前端", "后端", "全栈", "算法", "数据", "移动端", "客户端"}

_SLOT_DEFAULTS = {"未知岗位", "不限城市", "面议", "求职", ""}

_SENTENCE_TEMPLATE = (
    "{intent} | 岗位={position} | 城市={city} | 薪资={salary} | 技术={tech}"
)

_LLM_SLOT_PROMPT = """\
你是一个求职信息提取器。从以下文本中提取求职相关槽位，严格输出 JSON，不要其他文字。

文本：{text}

输出格式（无法确定的字段填 null）：
{{
  "position": "目标岗位，如：后端工程师",
  "city":     "目标城市，如：北京",
  "salary":   "薪资期望，如：35K-50K",
  "intent":   "求职意图，从以下选择：求职投递/岗位推荐/简历优化/面试准备/JD分析/薪资了解",
  "tech":     "技术栈，空格分隔，如：Python Kafka Redis"
}}
"""


# ──────────────────────────────────────────────────────────────────────────────
# 规则层槽位提取（写路径和读路径共用）
# ──────────────────────────────────────────────────────────────────────────────

def extract_slots(text: str) -> dict[str, str]:
    """
    用正则从任意文本提取求职槽位，无 LLM，耗时 ≈ 2ms。

    Returns:
        {
            "position": "后端工程师",
            "city":     "北京",
            "salary":   "35K-50K",
            "intent":   "求职投递",
            "tech":     "Python Kafka",
        }
        未识别字段返回默认值：未知岗位 / 不限城市 / 面议 / 求职 / ""
    """
    pos_m = _POSITION_RE.search(text)
    if pos_m:
        position = pos_m.group(1)
        if position in _POSITION_SUFFIX:
            position = position + "工程师"
    else:
        position = "未知岗位"

    city_m = _CITY_RE.search(text)
    city = city_m.group(1) if city_m else "不限城市"

    sal_m = _SALARY_RE.search(text)
    if sal_m:
        low, high = sal_m.group(1), sal_m.group(2)
        salary = f"{low}K-{high}K" if high else f"{low}K"
    else:
        salary = "面议"

    intent = "求职"
    for kw, mapped in _INTENT_MAP.items():
        if kw in text:
            intent = mapped
            break

    tech_list = _TECH_RE.findall(text)
    tech = " ".join(dict.fromkeys(tech_list))

    return {
        "position": position,
        "city":     city,
        "salary":   salary,
        "intent":   intent,
        "tech":     tech,
    }


def slots_to_sentence(slots: dict[str, str]) -> str:
    """
    将槽位字典拼成标准句用于 Embedding。

    示例输出：
        "求职投递 | 岗位=后端工程师 | 城市=北京 | 薪资=35K-50K | 技术=Python Kafka"
    """
    return _SENTENCE_TEMPLATE.format(
        intent=slots.get("intent", "求职"),
        position=slots.get("position", "未知岗位"),
        city=slots.get("city", "不限城市"),
        salary=slots.get("salary", "面议"),
        tech=slots.get("tech") or "不限",
    )


def _merge_slots(
    base: dict[str, str],
    override: dict[str, str],
) -> dict[str, str]:
    """
    合并两组槽位，override 中非默认值的字段覆盖 base。
    用于澄清补充槽位覆盖原始 Query 槽位，LLM 推测槽位覆盖规则层槽位。
    """
    merged = base.copy()
    for key, val in override.items():
        if val and val not in _SLOT_DEFAULTS:
            merged[key] = val
    return merged


def _has_valid_slots(slots: dict[str, str]) -> bool:
    """判断槽位是否包含至少一个有效（非默认）值，用于决定是否启动意图向量路。"""
    return any(v and v not in _SLOT_DEFAULTS for v in slots.values())


# ──────────────────────────────────────────────────────────────────────────────
# 写路径：LLM 槽位抽取（精度高于规则层，仅在写入时异步调用）
# ──────────────────────────────────────────────────────────────────────────────

async def _llm_extract_slots(
    text: str,
    llm_client,
    model: str,
) -> Optional[dict[str, str]]:
    """
    调用 LLM 从 Chunk 原文抽取槽位。
    失败时返回 None，调用方自动退化规则层。
    写路径专用，不在读路径调用。
    """
    import json as _json

    prompt = _LLM_SLOT_PROMPT.format(text=text[:800])
    try:
        resp = await llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,
        )
        raw = (
            resp.choices[0].message.content.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        data = _json.loads(raw)
        _null_strs = {"null", "none", ""}
        slots = {
            "position": str(data.get("position") or "未知岗位"),
            "city":     str(data.get("city")     or "不限城市"),
            "salary":   str(data.get("salary")   or "面议"),
            "intent":   str(data.get("intent")   or "求职"),
            "tech":     str(data.get("tech")     or ""),
        }
        _defaults_map = {
            "position": "未知岗位",
            "city":     "不限城市",
            "salary":   "面议",
            "intent":   "求职",
            "tech":     "",
        }
        for k, default in _defaults_map.items():
            if slots[k].lower() in _null_strs:
                slots[k] = default
        return slots
    except Exception as e:
        logger.warning(f"[Write] LLM 槽位抽取失败，退化规则层: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 写路径入口
# ──────────────────────────────────────────────────────────────────────────────

async def build_intent_sentence_from_chunk(
    chunk_text: str,
    llm_client=None,
    model: str = "",
) -> tuple[str, dict[str, str]]:
    """
    写路径入口，在 update_memory_from_file() 中对每个 Chunk 调用。

    优先使用 LLM 抽取槽位（写入是低频操作，精度值得投入），
    失败或未提供 llm_client 时退化规则层。

    Args:
        chunk_text:  Chunk 原始文本
        llm_client:  openai.AsyncOpenAI 实例（可为 None，退化规则层）
        model:       LLM 模型名称

    Returns:
        (intent_sentence, slots_dict)
        intent_sentence → Embedding 后以 id=chunk_id+"_intent" 写入 ChromaDB
        slots_dict      → 写入 metadata，key 含 vec_type/position/city/salary/intent/tech
    """
    slots: Optional[dict[str, str]] = None
    if llm_client and model:
        slots = await _llm_extract_slots(chunk_text, llm_client, model)
    if slots is None:
        slots = extract_slots(chunk_text)

    sentence = slots_to_sentence(slots)
    logger.debug(f"[Write] 意图标准句: {sentence}")
    return sentence, slots


# ──────────────────────────────────────────────────────────────────────────────
# 读路径入口（同步，无 LLM）
# ──────────────────────────────────────────────────────────────────────────────

def build_intent_sentence_from_query(
    query_text: str,
    clarification_answer: str = "",
    inferred_slots: Optional[dict[str, str]] = None,
) -> tuple[str, dict[str, str], bool]:
    """
    读路径入口，在 retrieve() 中替代原 normalize_query() LLM 调用。
    全程无 LLM 调用，耗时 ≈ 2ms。

    槽位合并优先级（从低到高）：
      1. 原始 Query 规则提取
      2. 澄清补充规则提取（覆盖）
      3. GateResult.inferred_slots，LLM 兜底推测槽位（覆盖）

    Args:
        query_text:           用户原始输入
        clarification_answer: 澄清反问后的用户补充（可为空字符串）
        inferred_slots:       ClarificationGate LLM 兜底时推测的槽位 dict

    Returns:
        (intent_sentence, slots_dict, has_slots)
        intent_sentence → Embedding 后查意图向量
        slots_dict      → 含有效槽位时可用于 ChromaDB where 精准过滤
        has_slots       → False 时意图向量路不启动，降级两路检索（原文+BM25）
    """
    slots = extract_slots(query_text)

    if clarification_answer.strip():
        clarif_slots = extract_slots(clarification_answer)
        slots = _merge_slots(slots, clarif_slots)

    if inferred_slots:
        slots = _merge_slots(slots, inferred_slots)

    has_slots = _has_valid_slots(slots)
    sentence  = slots_to_sentence(slots)
    logger.debug(f"[Retrieve] 意图标准句: {sentence}  has_slots={has_slots}")
    return sentence, slots, has_slots


# ──────────────────────────────────────────────────────────────────────────────
# 快速验证（直接运行）
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("读路径槽位提取验证")
    print("=" * 60)

    cases = [
        ("目标岗位是后端工程师，目标城市北京，期望薪资35K，熟悉Python和Kafka。", "", None),
        ("帮我找后端岗位",                    "",              None),
        ("帮我看看这个",                      "北京后端35K以上", {"position": "后端工程师", "city": "北京", "intent": "岗位推荐", "salary": "面议", "tech": ""}),
        ("我现在适合去哪家公司",              "",              None),
    ]
    for q, clarif, inferred in cases:
        sent, slots, has = build_intent_sentence_from_query(q, clarif, inferred)
        tag = "三路检索" if has else "降级两路"
        print(f"\nQ: {q[:35]:<35} → {tag}")
        print(f"   {sent}")
