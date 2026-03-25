"""
generate_dataset.py
===================
生成 Strata 项目评估数据集

结构：
  - A 类（20组）：干净对话，无模糊输入，无工具调用
  - B 类（15组）：含模糊输入，用于澄清门控评估
  - C 类（15组）：长对话含工具调用，用于工具压缩评估

每组对话包含完整标注：
  ambiguous_turns / retrieval_queries / relevant_chunks /
  tool_calls / key_information / preference_changes
"""

import json
import random
import uuid
from pathlib import Path
from datetime import date, timedelta
from copy import deepcopy

random.seed(42)
OUTPUT_DIR = Path("strata_eval_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 素材库
# ──────────────────────────────────────────────────────────────────────────────

POSITIONS = [
    "后端工程师", "前端工程师", "全栈工程师", "算法工程师",
    "数据工程师", "产品经理", "测试工程师", "架构师",
    "机器学习工程师", "推荐系统工程师", "平台工程师", "DevOps工程师",
]

CITIES = ["北京", "上海", "杭州", "深圳", "成都", "广州", "武汉", "西安"]

COMPANIES = [
    "字节跳动", "阿里巴巴", "腾讯", "美团", "百度",
    "蚂蚁集团", "京东", "滴滴", "快手", "网易",
    "华为", "小米", "PingCAP", "Shopee", "Grab",
]

TECH_STACKS = [
    ["Python", "FastAPI", "Redis", "MySQL"],
    ["Java", "Spring Boot", "Kafka", "Elasticsearch"],
    ["Go", "gRPC", "PostgreSQL", "Docker"],
    ["Python", "PyTorch", "TensorFlow", "CUDA"],
    ["JavaScript", "React", "Node.js", "MongoDB"],
    ["Rust", "C++", "Linux", "内核开发"],
    ["Python", "Spark", "Flink", "Hive", "数据仓库"],
    ["Java", "Kubernetes", "Prometheus", "微服务"],
]

SALARIES = [
    "20-30K", "25-35K", "30-45K", "35-50K",
    "40-60K", "50-70K", "60-80K", "80-120K",
]

WEAKNESSES = [
    "系统设计", "分布式事务", "算法与数据结构",
    "网络编程", "数据库调优", "英文面试", "行为面试",
]

INTERVIEW_STAGES = ["简历筛选", "笔试", "一面", "二面", "三面", "HR面", "终面"]

AMBIGUITY_TYPES = [
    "reference_unclear",
    "intent_unclear",
    "constraint_missing",
    "intent_mixed",
]

# 各模糊类型对应的模糊输入模板
AMBIGUOUS_INPUTS = {
    "reference_unclear": [
        "帮我分析一下这个",
        "这个岗位怎么样？",
        "你觉得那家公司如何？",
        "上次说的那个，再详细讲讲",
        "帮我看看这份材料",
        "刚才提到的那个方向我很感兴趣",
        "那个薪资合理吗？",
        "这家公司背景怎么样？",
    ],
    "intent_unclear": [
        "了解一下",
        "说说？",
        "看看",
        "帮我了解一下吧",
        "讲讲",
        "介绍",
        "看一下",
    ],
    "constraint_missing": [
        "帮我找找工作吧",
        "我想换工作",
        "帮我推荐几个岗位",
        "有什么好的机会吗？",
        "我想找个新工作",
        "帮我搜索一下岗位",
    ],
    "intent_mixed": [
        "帮我改一下简历，顺便推荐几个合适的岗位",
        "我想准备面试，同时也想找找新岗位",
        "帮我优化简历，再搜索一下字节跳动的面试题",
        "给我推荐岗位，另外也帮我准备一下面试",
        "我需要修改简历，还想了解一下目标公司背景",
    ],
}

# 清晰输入模板
CLEAR_INPUTS = [
    "帮我分析一下{company}的{position}岗位JD，薪资{salary}",
    "我想了解{city}的{position}就业市场情况",
    "针对{company}的{position}面试，我该如何准备？",
    "帮我优化简历中的{position}项目经历描述",
    "搜索一下{company}最近的面试经验分享",
    "我的技术栈是{tech}，适合投{position}岗位吗？",
    "对比一下{company}和{company2}的{position}岗位",
    "我在{stage}阶段，下一步应该怎么准备？",
    "{city}的{position}平均薪资是多少？",
    "帮我制定一份{position}的面试备考计划",
    "我的薄弱点是{weakness}，怎么在3个月内提升？",
    "分析一下我当前技术栈{tech}的市场竞争力",
]

# 助手回复模板（简洁版，真实数据集中由LLM生成）
ASSISTANT_REPLIES = [
    "好的，根据您的情况，我来为您详细分析。{position}方向在{city}的需求非常旺盛，尤其是有{tech}经验的候选人。",
    "了解了，您目前的技术背景与{position}岗位匹配度较高。建议重点准备以下几个方向：系统设计、{tech}深度应用、分布式架构。",
    "根据我的记忆，您之前提到目标是{city}的{position}岗位，这与{company}的招聘需求非常吻合。",
    "针对您的情况，{company}的{position}岗位有几点需要特别注意：技术深度要求较高，建议重点补充{weakness}方向的准备。",
    "好的，我已经记录了您的求职偏好。接下来我们针对{company}的面试流程逐一分析。",
    "从市场角度看，拥有{tech}技能的{position}候选人在{city}的薪资区间通常在{salary}，您的期望是合理的。",
    "明白了，您目前处于{stage}阶段。接下来最重要的是针对技术面的深度准备，特别是{weakness}这个薄弱点。",
    "根据您提供的信息，{company}的{position}岗位与您的背景有较高契合度。核心差距在于{weakness}方向需要加强。",
]

# 工具调用相关
JD_TEXTS = [
    """岗位：{position} | 公司：{company} | 地点：{city}
职责：1.负责{tech}相关系统的设计与开发 2.性能优化与架构改进 3.与团队协作推动技术演进
要求：1.3年以上相关经验 2.熟练掌握{tech} 3.有大规模分布式系统经验优先
薪资：{salary}""",
    """高级{position} | {company} | {city}
核心职责：负责推荐系统/搜索系统的工程化实现，保障亿级QPS下的低延迟响应。
技术要求：精通{tech}，熟悉分布式系统，有{weakness}相关经验者优先。
薪资范围：{salary}*16，base{city}""",
]

SEARCH_RESULTS = [
    """搜索结果：{company} {position} 面试经验
1. 技术面重点：系统设计（分布式锁、消息队列）、{tech}原理深入
2. 算法面：LeetCode中等难度，重点考察{tech}相关
3. 行为面：STAR法则，重点考察团队协作和技术影响力
4. 薪资反馈：{salary}，谈判空间约10-15%
来源：牛客网、知乎、Glassdoor""",
]

# 偏好变更模板
PREF_CHANGE_TEMPLATES = [
    {"field": "目标城市",   "values": CITIES},
    {"field": "薪资预期",   "values": SALARIES},
    {"field": "目标岗位",   "values": POSITIONS},
    {"field": "目标公司_行业", "values": COMPANIES},
    {"field": "面试薄弱点", "values": WEAKNESSES},
]


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def random_date(start_days_ago: int = 180) -> str:
    base = date.today()
    delta = random.randint(0, start_days_ago)
    return (base - timedelta(days=delta)).strftime("%Y-%m-%d")


def fill_template(template: str, ctx: dict) -> str:
    try:
        return template.format(**ctx)
    except KeyError:
        return template


def make_ctx() -> dict:
    tech = random.choice(TECH_STACKS)
    company_list = random.sample(COMPANIES, 2)
    return {
        "position":  random.choice(POSITIONS),
        "city":      random.choice(CITIES),
        "company":   company_list[0],
        "company2":  company_list[1],
        "tech":      "、".join(tech[:2]),
        "tech_full": "、".join(tech),
        "salary":    random.choice(SALARIES),
        "weakness":  random.choice(WEAKNESSES),
        "stage":     random.choice(INTERVIEW_STAGES),
    }


def make_chunk_id() -> str:
    return str(uuid.uuid4())


def make_memory_chunks(ctx: dict, n: int = 6) -> list[dict]:
    """生成一组模拟已索引的 memory chunk，带 metadata。"""
    templates = [
        f"用户目标岗位是{ctx['position']}，技术栈包含{ctx['tech_full']}",
        f"用户目标城市为{ctx['city']}，期望薪资{ctx['salary']}",
        f"用户正在准备{ctx['company']}的面试，面试薄弱点是{ctx['weakness']}",
        f"用户对{ctx['company2']}的{ctx['position']}岗位感兴趣",
        f"用户已完成{ctx['stage']}阶段，正在准备下一轮",
        f"用户倾向于{ctx['city']}，但也考虑远程机会",
        f"用户技术栈{ctx['tech']}在市场上竞争力较强",
        f"用户希望在3个月内解决{ctx['weakness']}薄弱点",
    ]
    chunks = []
    for i, text in enumerate(random.sample(templates, min(n, len(templates)))):
        cid = make_chunk_id()
        d   = random_date()
        chunks.append({
            "chunk_id": cid,
            "text":     text,
            "metadata": {
                "source":   f"{d}.md",
                "date":     d,
                "header_1": random.choice(["求职目标", "技术背景", "面试进展", "偏好记录"]),
                "chunk_id": cid,
            },
        })
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# A 类对话生成（20组，干净对话）
# ──────────────────────────────────────────────────────────────────────────────

def gen_type_a(conv_id: str, turn_count: int = 15) -> dict:
    ctx = make_ctx()
    chunks = make_memory_chunks(ctx)
    turns = []
    retrieval_queries = []
    relevant_chunks_per_query = []
    preference_changes = []

    # 初始偏好设定（第1轮）
    pref_fields = random.sample(PREF_CHANGE_TEMPLATES, 3)
    for pf in pref_fields:
        preference_changes.append({
            "turn":  1,
            "field": pf["field"],
            "value": random.choice(pf["values"]),
        })

    for i in range(turn_count):
        turn_idx = i + 1
        tmpl = random.choice(CLEAR_INPUTS)
        user_text = fill_template(tmpl, ctx)

        reply_tmpl = random.choice(ASSISTANT_REPLIES)
        assistant_text = fill_template(reply_tmpl, ctx)

        turns.append({
            "turn":      turn_idx,
            "role":      "user",
            "content":   user_text,
        })
        turns.append({
            "turn":      turn_idx,
            "role":      "assistant",
            "content":   assistant_text,
        })

        # 每轮都有检索
        relevant = random.sample(chunks, min(3, len(chunks)))
        retrieval_queries.append({
            "turn":  turn_idx,
            "query": user_text,
        })
        relevant_chunks_per_query.append({
            "turn":            turn_idx,
            "relevant_chunks": [c["chunk_id"] for c in relevant],
        })

        # 随机在某几轮触发偏好变更
        if i in [4, 9, 13] and i < turn_count:
            pf = random.choice(PREF_CHANGE_TEMPLATES)
            preference_changes.append({
                "turn":  turn_idx,
                "field": pf["field"],
                "value": random.choice(pf["values"]),
            })

    return {
        "conversation_id":     conv_id,
        "type":                "A",
        "description":         "干净对话，无模糊输入，无工具调用",
        "context":             ctx,
        "memory_chunks":       chunks,
        "turns":               turns,
        "ambiguous_turns":     [],
        "ambiguity_types":     [],
        "retrieval_queries":   retrieval_queries,
        "relevant_chunks":     relevant_chunks_per_query,
        "tool_calls":          [],
        "key_information":     [],
        "preference_changes":  preference_changes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# B 类对话生成（15组，含模糊输入）
# ──────────────────────────────────────────────────────────────────────────────

def gen_type_b(conv_id: str, turn_count: int = 15) -> dict:
    ctx = make_ctx()
    chunks = make_memory_chunks(ctx)
    turns = []
    retrieval_queries = []
    relevant_chunks_per_query = []
    preference_changes = []
    ambiguous_turns = []
    ambiguity_types_list = []

    # 随机选 3-5 轮植入模糊输入（避开第1轮）
    ambiguous_turn_indices = sorted(random.sample(range(2, turn_count), min(4, turn_count - 2)))

    # 为每个模糊轮次分配一个模糊类型
    chosen_types = random.choices(AMBIGUITY_TYPES, k=len(ambiguous_turn_indices))

    ambiguous_map = dict(zip(ambiguous_turn_indices, chosen_types))

    # 初始偏好
    pref_fields = random.sample(PREF_CHANGE_TEMPLATES, 2)
    for pf in pref_fields:
        preference_changes.append({
            "turn":  1,
            "field": pf["field"],
            "value": random.choice(pf["values"]),
        })

    for i in range(turn_count):
        turn_idx = i + 1

        if (i + 1) in ambiguous_map:
            atype = ambiguous_map[i + 1]
            user_text = random.choice(AMBIGUOUS_INPUTS[atype])
            ambiguous_turns.append(turn_idx)
            ambiguity_types_list.append(atype)

            # 模糊输入后跟一个澄清回复，再跟用户补充说明
            assistant_clarify = {
                "reference_unclear":  f"你提到的内容具体指的是什么？请描述一下。",
                "intent_unclear":     f"你希望我帮你做什么？A.分析JD B.修改简历 C.搜索面试题",
                "constraint_missing": f"为了精准推荐，请告诉我：目标岗位、目标城市和期望薪资。",
                "intent_mixed":       f"你提到了多个需求，本次你最想先处理哪一个？",
            }[atype]

            turns.append({"turn": turn_idx, "role": "user",      "content": user_text})
            turns.append({"turn": turn_idx, "role": "assistant",  "content": assistant_clarify})

            # 下一轮为用户补充说明（算作同一个"模糊事件"的收尾）
            clarify_reply = fill_template(
                f"就是{ctx['company']}的{ctx['position']}岗位，{ctx['city']}，薪资{ctx['salary']}", ctx
            )
            turns.append({"turn": turn_idx, "role": "user",      "content": clarify_reply})
            assistant_text = fill_template(random.choice(ASSISTANT_REPLIES), ctx)
            turns.append({"turn": turn_idx, "role": "assistant",  "content": assistant_text})

        else:
            tmpl = random.choice(CLEAR_INPUTS)
            user_text = fill_template(tmpl, ctx)
            assistant_text = fill_template(random.choice(ASSISTANT_REPLIES), ctx)
            turns.append({"turn": turn_idx, "role": "user",      "content": user_text})
            turns.append({"turn": turn_idx, "role": "assistant",  "content": assistant_text})

        relevant = random.sample(chunks, min(3, len(chunks)))
        retrieval_queries.append({"turn": turn_idx, "query": user_text})
        relevant_chunks_per_query.append({
            "turn":            turn_idx,
            "relevant_chunks": [c["chunk_id"] for c in relevant],
        })

        if i in [3, 8, 12]:
            pf = random.choice(PREF_CHANGE_TEMPLATES)
            preference_changes.append({
                "turn":  turn_idx,
                "field": pf["field"],
                "value": random.choice(pf["values"]),
            })

    return {
        "conversation_id":    conv_id,
        "type":               "B",
        "description":        "含模糊输入对话，用于澄清门控机制评估",
        "context":            ctx,
        "memory_chunks":      chunks,
        "turns":              turns,
        "ambiguous_turns":    ambiguous_turns,
        "ambiguity_types":    [
            {"turn": t, "type": at}
            for t, at in zip(ambiguous_turns, ambiguity_types_list)
        ],
        "retrieval_queries":  retrieval_queries,
        "relevant_chunks":    relevant_chunks_per_query,
        "tool_calls":         [],
        "key_information":    [],
        "preference_changes": preference_changes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# C 类对话生成（15组，长对话含工具调用）
# ──────────────────────────────────────────────────────────────────────────────

def gen_type_c(conv_id: str, turn_count: int = 20) -> dict:
    ctx = make_ctx()
    chunks = make_memory_chunks(ctx, n=8)
    turns = []
    retrieval_queries = []
    relevant_chunks_per_query = []
    preference_changes = []
    tool_calls_log = []
    key_information_log = []

    # 在第 5、10、16 轮植入工具调用
    tool_turn_indices = [5, 10, 16]
    tool_types = random.choices(["analyze_jd", "search_job"], k=3)
    tool_map = dict(zip(tool_turn_indices, tool_types))

    # 初始偏好
    pref_fields = random.sample(PREF_CHANGE_TEMPLATES, 3)
    for pf in pref_fields:
        preference_changes.append({
            "turn":  1,
            "field": pf["field"],
            "value": random.choice(pf["values"]),
        })

    for i in range(turn_count):
        turn_idx = i + 1

        if (i + 1) in tool_map:
            tool_type = tool_map[i + 1]
            tool_call_id = f"call_{uuid.uuid4().hex[:8]}"

            if tool_type == "analyze_jd":
                jd_tmpl = random.choice(JD_TEXTS)
                jd_text = fill_template(jd_tmpl, ctx)
                user_text = f"帮我分析这个JD：{jd_text[:80]}..."

                # 模拟超长工具输出（约1500字符）
                raw_output = fill_template(jd_tmpl, ctx) * 3 + \
                    f"\n\n## 核心技术要求\n必须掌握：{ctx['tech_full']}\n" \
                    f"## 面试准备\n重点：系统设计、{ctx['weakness']}、分布式架构\n" \
                    f"## 薪资分析\n{ctx['salary']}，在{ctx['city']}属于中高端水平\n" \
                    f"## 简历建议\n突出{ctx['tech']}经验，量化项目成果\n" * 2

                key_points = [
                    f"岗位名称：{ctx['position']}",
                    f"目标公司：{ctx['company']}",
                    f"核心技术：{ctx['tech_full']}",
                    f"薪资范围：{ctx['salary']}",
                    f"面试重点：{ctx['weakness']}",
                    f"工作地点：{ctx['city']}",
                ]

            else:  # search_job
                user_text = f"搜索一下{ctx['company']}{ctx['position']}的面试经验"
                search_tmpl = random.choice(SEARCH_RESULTS)
                extra_lines = []
                for j in range(5):
                    extra_lines.append(f"[{j+1}] {ctx['company']} {ctx['position']} 面试经历 - 牛客网")
                    extra_lines.append(f"    薪资反馈：{ctx['salary']}，流程：{' -> '.join(random.sample(INTERVIEW_STAGES, 4))}")
                raw_output = fill_template(search_tmpl, ctx) * 4 + "\n更多结果：\n" + "\n".join(extra_lines)

                key_points = [
                    f"目标公司：{ctx['company']}",
                    f"岗位方向：{ctx['position']}",
                    f"技术考察：{ctx['tech']}",
                    f"薪资水平：{ctx['salary']}",
                    f"面试流程：{' -> '.join(random.sample(INTERVIEW_STAGES, 3))}",
                ]

            # 截断模拟（前800字符 + 文件引用）
            file_id = uuid.uuid4().hex
            compacted_output = raw_output[:800] + \
                f"\n\n[Full content saved to: .job_agent_v2/tool_result/{file_id}.txt]"

            tool_calls_log.append({
                "turn":             turn_idx,
                "tool_call_id":     tool_call_id,
                "tool_name":        tool_type,
                "input_summary":    user_text[:100],
                "raw_output":       raw_output,
                "raw_output_len":   len(raw_output),
                "compacted_output": compacted_output,
                "compacted_len":    len(compacted_output),
                "file_id":          file_id,
            })
            key_information_log.append({
                "turn":       turn_idx,
                "tool_name":  tool_type,
                "key_points": key_points,
            })

            turns.append({"turn": turn_idx, "role": "user",      "content": user_text})
            turns.append({"turn": turn_idx, "role": "assistant",
                          "content": f"[调用{tool_type}工具]\n{compacted_output[:200]}..."})

        else:
            tmpl = random.choice(CLEAR_INPUTS)
            user_text = fill_template(tmpl, ctx)
            assistant_text = fill_template(random.choice(ASSISTANT_REPLIES), ctx)
            turns.append({"turn": turn_idx, "role": "user",      "content": user_text})
            turns.append({"turn": turn_idx, "role": "assistant",  "content": assistant_text})

        relevant = random.sample(chunks, min(3, len(chunks)))
        retrieval_queries.append({"turn": turn_idx, "query": user_text})
        relevant_chunks_per_query.append({
            "turn":            turn_idx,
            "relevant_chunks": [c["chunk_id"] for c in relevant],
        })

        if i in [3, 7, 12, 17]:
            pf = random.choice(PREF_CHANGE_TEMPLATES)
            preference_changes.append({
                "turn":  turn_idx,
                "field": pf["field"],
                "value": random.choice(pf["values"]),
            })

    return {
        "conversation_id":    conv_id,
        "type":               "C",
        "description":        "长对话含工具调用，用于工具压缩评估",
        "context":            ctx,
        "memory_chunks":      chunks,
        "turns":              turns,
        "ambiguous_turns":    [],
        "ambiguity_types":    [],
        "retrieval_queries":  retrieval_queries,
        "relevant_chunks":    relevant_chunks_per_query,
        "tool_calls":         tool_calls_log,
        "key_information":    key_information_log,
        "preference_changes": preference_changes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 生成全部数据集
# ──────────────────────────────────────────────────────────────────────────────

def generate_all():
    all_conversations = []

    print("生成 A 类对话（20组）...")
    for i in range(20):
        conv_id = f"conv_A_{i+1:03d}"
        turns   = random.randint(13, 17)
        conv    = gen_type_a(conv_id, turns)
        all_conversations.append(conv)
        print(f"  {conv_id}: {turns} 轮, {len(conv['preference_changes'])} 个偏好变更")

    print("生成 B 类对话（15组）...")
    for i in range(15):
        conv_id = f"conv_B_{i+1:03d}"
        turns   = random.randint(13, 17)
        conv    = gen_type_b(conv_id, turns)
        all_conversations.append(conv)
        print(f"  {conv_id}: {turns} 轮, {len(conv['ambiguous_turns'])} 个模糊输入")

    print("生成 C 类对话（15组）...")
    for i in range(15):
        conv_id = f"conv_C_{i+1:03d}"
        turns   = random.randint(18, 23)
        conv    = gen_type_c(conv_id, turns)
        all_conversations.append(conv)
        tc = len(conv['tool_calls'])
        total_raw  = sum(t['raw_output_len']  for t in conv['tool_calls'])
        total_comp = sum(t['compacted_len']   for t in conv['tool_calls'])
        print(f"  {conv_id}: {turns} 轮, {tc} 次工具调用, "
              f"原始输出合计 {total_raw:,} 字符 → 压缩后 {total_comp:,} 字符")

    # ── 写出完整数据集（单文件）────────────────────────────────────────────
    full_path = OUTPUT_DIR / "dataset_full.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)
    print(f"\n完整数据集已写出: {full_path}  ({full_path.stat().st_size / 1024:.1f} KB)")

    # ── 按类型分文件写出（便于各模块单独加载）──────────────────────────────
    for type_label in ["A", "B", "C"]:
        subset = [c for c in all_conversations if c["type"] == type_label]
        path   = OUTPUT_DIR / f"dataset_type_{type_label}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        print(f"  Type {type_label} 数据集: {path}  ({path.stat().st_size / 1024:.1f} KB)")

    # ── 生成统计摘要 ─────────────────────────────────────────────────────────
    stats = _compute_stats(all_conversations)
    stats_path = OUTPUT_DIR / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n统计摘要:\n{json.dumps(stats, ensure_ascii=False, indent=2)}")

    return all_conversations


def _compute_stats(convs: list[dict]) -> dict:
    type_a = [c for c in convs if c["type"] == "A"]
    type_b = [c for c in convs if c["type"] == "B"]
    type_c = [c for c in convs if c["type"] == "C"]

    def avg_turns(cs):
        if not cs: return 0
        return round(sum(
            max(t["turn"] for t in c["turns"]) for c in cs
        ) / len(cs), 1)

    # 工具压缩统计
    all_tool_calls = [tc for c in type_c for tc in c["tool_calls"]]
    raw_lens  = [tc["raw_output_len"]  for tc in all_tool_calls]
    comp_lens = [tc["compacted_len"]   for tc in all_tool_calls]
    token_save = round(
        (1 - sum(comp_lens) / max(sum(raw_lens), 1)) * 100, 1
    ) if raw_lens else 0

    # 偏好变更统计
    all_pref = [p for c in convs for p in c["preference_changes"]]
    field_counts = {}
    for p in all_pref:
        field_counts[p["field"]] = field_counts.get(p["field"], 0) + 1

    # 模糊类型分布
    all_amb = [a for c in type_b for a in c["ambiguity_types"]]
    amb_dist = {}
    for a in all_amb:
        key = a["type"]
        amb_dist[key] = amb_dist.get(key, 0) + 1

    return {
        "总对话组数": len(convs),
        "A类": {
            "数量": len(type_a),
            "平均轮次": avg_turns(type_a),
            "总偏好变更事件": sum(len(c["preference_changes"]) for c in type_a),
            "总检索样本数": sum(len(c["retrieval_queries"]) for c in type_a),
        },
        "B类": {
            "数量": len(type_b),
            "平均轮次": avg_turns(type_b),
            "总模糊输入数": sum(len(c["ambiguous_turns"]) for c in type_b),
            "模糊类型分布": amb_dist,
            "总检索样本数": sum(len(c["retrieval_queries"]) for c in type_b),
        },
        "C类": {
            "数量": len(type_c),
            "平均轮次": avg_turns(type_c),
            "总工具调用次数": len(all_tool_calls),
            "工具输出原始字符合计": sum(raw_lens),
            "工具输出压缩后字符合计": sum(comp_lens),
            "估算Token节省率": f"{token_save}%",
            "总检索样本数": sum(len(c["retrieval_queries"]) for c in type_c),
        },
        "全量": {
            "总检索样本数": sum(len(c["retrieval_queries"]) for c in convs),
            "总偏好变更事件": len(all_pref),
            "偏好字段分布": field_counts,
        },
    }


if __name__ == "__main__":
    generate_all()
