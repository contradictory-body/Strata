"""
ablation_study.py — Strata 消融实验主脚本
==========================================

评估四个核心指标：
  Gate-F1  : 澄清门控分类准确率（以 ambiguous_turns 为 Ground Truth）
  CRA@4    : 上下文召回准确率 Hit Rate@4（以 relevant_chunks 为 Ground Truth）
  MRR      : 平均倒数排名
  TOR      : 工具输出保留率（以 key_information 为 Ground Truth）
  TER      : Token 节省率
  PCS      : 偏好提取召回率（以 preference_changes 为 Ground Truth）

运行方式：
  cd experiments/
  python ablation_study.py

环境要求：
  - .env 中配置好 API Key
  - dataset/ 目录下有完整数据集 json 文件
  - pip install sentence-transformers rank-bm25 langchain-text-splitters cachetools chromadb transformers
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# 路径设置
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT   = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dotenv
dotenv.load_dotenv(dotenv_path=REPO_ROOT / ".env", override=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("AblationStudy")

DATASET_DIR  = SCRIPT_DIR / "dataset"
RESULTS_DIR  = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(type_filter: str | None = None) -> list[dict]:
    if type_filter:
        path = DATASET_DIR / f"dataset_gt_v3_type_{type_filter}.json"
    else:
        path = DATASET_DIR / "dataset_gt_v3_full.json"

    if not path.exists():
        raise FileNotFoundError(
            f"数据集文件不存在: {path}\n"
            "请先运行 dataset/generate_dataset.py 生成数据集。"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# 指标一：澄清门控 Gate-F1
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GateEvalResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    type_breakdown: dict = field(default_factory=dict)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


async def eval_gate_f1(
    dataset: list[dict],
    llm_client,
    model: str,
    use_llm_fallback: bool = True,
) -> GateEvalResult:
    """评估澄清门控 Gate-F1。

    Args:
        dataset:         B 类数据集
        llm_client:      openai.AsyncOpenAI 实例
        model:           模型名称
        use_llm_fallback: 是否启用 LLM 兜底（消融用）

    Returns:
        GateEvalResult
    """
    from agent.reme_light_job_agent_v2.clarification_gate import ClarificationGate

    gate   = ClarificationGate(llm_client=llm_client, model=model, reme=None)
    result = GateEvalResult()

    total_turns_checked = 0

    for conv in dataset:
        if conv["type"] != "B":
            continue

        # 模拟对话上下文（逐轮积累）
        simulated_messages: list[dict] = []
        ambiguous_turn_set = set(conv.get("ambiguous_turns", []))
        profile_summary = ""  # 简化：不加载真实画像

        for turn_data in conv.get("turns", []):
            turn_idx = turn_data["turn"]
            role     = turn_data["role"]
            content  = turn_data["content"]

            if role != "user":
                simulated_messages.append({"role": role, "content": content})
                continue

            # 执行门控
            try:
                gate_result = await gate.clarify_gate(
                    user_input=content,
                    recent_messages=simulated_messages[-4:],
                    profile_summary=profile_summary,
                )

                is_labeled = turn_idx in ambiguous_turn_set
                is_predicted = gate_result.is_ambiguous

                if is_predicted and is_labeled:
                    result.tp += 1
                    # 记录类型分布
                    for at in conv.get("ambiguity_types", []):
                        if at["turn"] == turn_idx:
                            t = at["type"]
                            result.type_breakdown[t] = result.type_breakdown.get(t, {"tp": 0, "fn": 0})
                            result.type_breakdown[t]["tp"] += 1
                elif is_predicted and not is_labeled:
                    result.fp += 1
                elif not is_predicted and is_labeled:
                    result.fn += 1
                    for at in conv.get("ambiguity_types", []):
                        if at["turn"] == turn_idx:
                            t = at["type"]
                            result.type_breakdown[t] = result.type_breakdown.get(t, {"tp": 0, "fn": 0})
                            result.type_breakdown[t]["fn"] += 1

                total_turns_checked += 1

            except Exception as e:
                logger.warning(f"门控评估异常 conv={conv['conversation_id']} turn={turn_idx}: {e}")

            simulated_messages.append({"role": role, "content": content})

    print(f"  门控评估完成: 总检查轮次={total_turns_checked}, "
          f"TP={result.tp}, FP={result.fp}, FN={result.fn}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 指标二：CRA@4 / MRR
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalEvalResult:
    total:     int = 0
    hits:      int = 0
    mrr_sum:   float = 0.0
    p_at_4_sum: float = 0.0

    @property
    def hit_rate_at_4(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0

    @property
    def mrr(self) -> float:
        return self.mrr_sum / self.total if self.total > 0 else 0.0

    @property
    def precision_at_4(self) -> float:
        return self.p_at_4_sum / self.total if self.total > 0 else 0.0


async def eval_retrieval(
    dataset: list[dict],
    retrieval_fn,
    setup_fn=None,
) -> RetrievalEvalResult:
    """评估检索链路 CRA@4 / MRR。

    Args:
        dataset:      数据集（任意类型）
        retrieval_fn: async 函数，签名 (query: str) -> list[str]（UUID 列表，最多 4 个）
        setup_fn:     每组对话开始前调用，用于加载 memory_chunks

    Returns:
        RetrievalEvalResult
    """
    result = RetrievalEvalResult()

    for conv in dataset:
        # 每组对话前加载该组的 memory_chunks
        if setup_fn:
            await setup_fn(conv.get("memory_chunks", []))

        queries       = conv.get("retrieval_queries", [])
        ground_truths = conv.get("relevant_chunks", [])

        for q_item, gt_item in zip(queries, ground_truths):
            query    = q_item.get("query", "")
            gt_uuids = set(gt_item.get("relevant_chunks", []))

            if not query or not gt_uuids:
                continue

            try:
                retrieved = await retrieval_fn(query)   # list[str] of UUIDs, top-4
            except Exception as e:
                logger.warning(f"检索失败: {e}")
                retrieved = []

            result.total += 1

            # Hit Rate @4
            hit = any(uid in gt_uuids for uid in retrieved)
            if hit:
                result.hits += 1

            # MRR
            rr = 0.0
            for rank, uid in enumerate(retrieved, start=1):
                if uid in gt_uuids:
                    rr = 1.0 / rank
                    break
            result.mrr_sum += rr

            # Precision @4
            hits_in_top4 = sum(1 for uid in retrieved if uid in gt_uuids)
            result.p_at_4_sum += hits_in_top4 / 4.0

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 指标三：TOR / TER
# ──────────────────────────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    """使用 transformers tokenizer 计算 token 数，失败则退化为字符数/2。"""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-small-zh-v1.5", use_fast=True
        )
        return len(tokenizer.encode(text))
    except Exception:
        return len(text) // 2


def _keyword_match(text: str, key_point: str) -> bool:
    """检查关键点是否在文本中出现（提取冒号后的核心词）。"""
    # "岗位名称：全栈工程师" → 检查 "全栈工程师"
    parts = re.split(r"[：:]", key_point, maxsplit=1)
    core = parts[-1].strip() if len(parts) > 1 else key_point.strip()
    # 进一步取最具体的部分（去掉括号说明）
    core = re.sub(r"[（(].*?[)）]", "", core).strip()
    if not core:
        return False
    return core in text


@dataclass
class CompressionEvalResult:
    total_calls:      int = 0
    total_key_points: int = 0
    retained_points:  int = 0
    total_raw_tokens: int = 0
    total_cmp_tokens: int = 0

    @property
    def tor(self) -> float:
        return self.retained_points / self.total_key_points if self.total_key_points > 0 else 0.0

    @property
    def ter(self) -> float:
        if self.total_raw_tokens == 0:
            return 0.0
        return (self.total_raw_tokens - self.total_cmp_tokens) / self.total_raw_tokens


def eval_compression(dataset: list[dict]) -> CompressionEvalResult:
    """评估工具输出压缩保留率和 Token 节省率。

    Args:
        dataset: C 类数据集

    Returns:
        CompressionEvalResult
    """
    result = CompressionEvalResult()

    for conv in dataset:
        if conv["type"] != "C":
            continue

        tool_calls    = conv.get("tool_calls", [])
        key_info_list = conv.get("key_information", [])

        # 建立 turn → key_info 的映射
        ki_map = {ki["turn"]: ki for ki in key_info_list}

        for tc in tool_calls:
            turn = tc["turn"]
            ki   = ki_map.get(turn)
            if not ki:
                continue

            compacted   = tc.get("compacted_output", "")
            key_points  = ki.get("key_points", [])
            raw_output  = tc.get("raw_output", "")

            result.total_calls += 1

            # 保留率
            for kp in key_points:
                result.total_key_points += 1
                if _keyword_match(compacted, kp):
                    result.retained_points += 1

            # Token 统计
            result.total_raw_tokens += _count_tokens(raw_output)
            result.total_cmp_tokens += _count_tokens(compacted)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 指标四：PCS（偏好提取召回率）
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PreferenceEvalResult:
    total_events:   int = 0
    captured_events: int = 0
    field_breakdown: dict = field(default_factory=dict)

    @property
    def pcs(self) -> float:
        return self.captured_events / self.total_events if self.total_events > 0 else 0.0


async def eval_preference(
    dataset: list[dict],
    llm_client,
    model: str,
) -> PreferenceEvalResult:
    """评估偏好提取召回率 PCS。

    模拟运行每组对话，检查对话结束后画像是否捕获了标注的偏好变更。

    Args:
        dataset:    全量数据集
        llm_client: openai.AsyncOpenAI 实例
        model:      模型名称

    Returns:
        PreferenceEvalResult
    """
    result = PreferenceEvalResult()

    EXTRACT_PROMPT = """\
从以下对话片段中提取用户的求职偏好信息。
只返回 JSON，不要其他文字。如某项没有提及，值为 null。

字段：目标岗位, 目标城市, 技术栈, 目标公司_行业, 薪资预期, 面试薄弱点, 简历修改偏好

对话：
{conversation}

JSON："""

    for conv in dataset:
        # 只对标注了偏好变更的对话进行评估
        pref_changes = conv.get("preference_changes", [])
        if not pref_changes:
            continue

        # 获取所有对话内容
        turns = conv.get("turns", [])
        conversation_text = "\n".join(
            f"{t['role']}: {t['content'][:200]}" for t in turns
        )

        # 调用 LLM 提取偏好
        try:
            resp = await llm_client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": EXTRACT_PROMPT.format(conversation=conversation_text[:3000]),
                }],
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
            extracted = json.loads(raw)
        except Exception as e:
            logger.warning(f"偏好提取失败 conv={conv['conversation_id']}: {e}")
            extracted = {}

        # 对比标注的偏好变更事件
        # 以最后一次变更为准（覆盖逻辑）
        final_prefs: dict[str, str] = {}
        for pref in pref_changes:
            field_name = pref["field"]
            value      = pref["value"]
            final_prefs[field_name] = value

        for field_name, expected_value in final_prefs.items():
            result.total_events += 1
            if field_name not in result.field_breakdown:
                result.field_breakdown[field_name] = {"total": 0, "captured": 0}
            result.field_breakdown[field_name]["total"] += 1

            # 字段名映射
            field_map = {
                "目标岗位":    "目标岗位",
                "目标城市":    "目标城市",
                "技术栈":      "技术栈",
                "目标公司_行业": "目标公司_行业",
                "薪资预期":    "薪资预期",
                "面试薄弱点":  "面试薄弱点",
                "简历修改偏好": "简历修改偏好",
            }
            ext_key = field_map.get(field_name, field_name)
            ext_val = extracted.get(ext_key, "")

            if ext_val and expected_value in str(ext_val):
                result.captured_events += 1
                result.field_breakdown[field_name]["captured"] += 1

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 检索后端：基于数据集 memory_chunks 的简单向量检索
# ──────────────────────────────────────────────────────────────────────────────

class SimpleVectorRetriever:
    """基于数据集 memory_chunks 的向量检索后端，用于消融实验。"""

    def __init__(self, top_k: int = 4):
        self._chunks: list[dict] = []
        self._embeddings: list[list[float]] = []
        self._model = None
        self._top_k = top_k

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        return self._model

    async def setup(self, chunks: list[dict]) -> None:
        """加载一组对话的 memory_chunks。"""
        self._chunks = chunks
        if not chunks:
            self._embeddings = []
            return
        model = self._get_model()
        texts = [c["text"] for c in chunks]
        vecs  = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self._embeddings = vecs.tolist()

    async def retrieve(self, query: str) -> list[str]:
        """纯向量检索，返回 Top-K UUID。"""
        if not self._chunks:
            return []
        model   = self._get_model()
        q_vec   = model.encode(query, normalize_embeddings=True, show_progress_bar=False)
        scores  = [
            sum(a * b for a, b in zip(q_vec, e))
            for e in self._embeddings
        ]
        ranked  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._chunks[i]["chunk_id"] for i in ranked[:self._top_k]]


class HybridRetriever:
    """向量 + BM25 + RRF + Reranker 完整检索链路，用于消融实验。"""

    def __init__(
        self,
        top_k_vector:  int = 15,
        top_k_bm25:    int = 15,
        top_k_rrf:     int = 8,
        top_k_rerank:  int = 4,
        rrf_k:         int = 60,
        use_reranker:  bool = True,
    ):
        self._chunks       = []
        self._embeddings   = []
        self._bm25_index   = None
        self._bm25_ids     = []
        self._embed_model  = None
        self._rerank_model = None
        self.top_k_vector  = top_k_vector
        self.top_k_bm25    = top_k_bm25
        self.top_k_rrf     = top_k_rrf
        self.top_k_rerank  = top_k_rerank
        self.rrf_k         = rrf_k
        self.use_reranker  = use_reranker

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        return self._embed_model

    def _get_rerank_model(self):
        if self._rerank_model is None:
            from sentence_transformers import CrossEncoder
            self._rerank_model = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
        return self._rerank_model

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        try:
            import jieba
            return [t for t in jieba.cut(text) if t.strip()]
        except ImportError:
            return re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]|[a-zA-Z0-9_]+", text.lower())

    async def setup(self, chunks: list[dict]) -> None:
        from rank_bm25 import BM25Okapi
        self._chunks = chunks
        if not chunks:
            self._embeddings = []
            self._bm25_index = None
            return
        texts = [c["text"] for c in chunks]

        # Embeddings
        m    = self._get_embed_model()
        vecs = m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self._embeddings = vecs.tolist()

        # BM25
        self._bm25_ids   = [c["chunk_id"] for c in chunks]
        corpus           = [self._tokenize(t) for t in texts]
        self._bm25_index = BM25Okapi(corpus)

    async def retrieve(self, query: str) -> list[str]:
        if not self._chunks:
            return []

        # 向量路
        m      = self._get_embed_model()
        q_vec  = m.encode(query, normalize_embeddings=True, show_progress_bar=False)
        v_scores = [
            sum(a * b for a, b in zip(q_vec, e))
            for e in self._embeddings
        ]
        v_ranked = sorted(range(len(v_scores)), key=lambda i: v_scores[i], reverse=True)
        v_results = [(self._chunks[i]["chunk_id"], v_scores[i]) for i in v_ranked[:self.top_k_vector]]

        # BM25 路
        b_results: list[tuple[str, float]] = []
        if self._bm25_index:
            tokens   = self._tokenize(query)
            b_scores = self._bm25_index.get_scores(tokens).tolist()
            max_s    = max(b_scores) if b_scores else 0.0
            if max_s > 0:
                b_ranked = sorted(range(len(b_scores)), key=lambda i: b_scores[i], reverse=True)
                b_results = [
                    (self._bm25_ids[i], b_scores[i] / max_s)
                    for i in b_ranked[:self.top_k_bm25]
                    if b_scores[i] > 0
                ]

        # RRF 融合
        rrf: dict[str, float] = {}
        for rank, (uid, _) in enumerate(v_results, 1):
            rrf[uid] = rrf.get(uid, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, (uid, _) in enumerate(b_results, 1):
            rrf[uid] = rrf.get(uid, 0.0) + 1.0 / (self.rrf_k + rank)
        fused = [uid for uid, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)][:self.top_k_rrf]

        if not self.use_reranker:
            return fused[:self.top_k_rerank]

        # Reranker 精排
        uid_to_chunk = {c["chunk_id"]: c["text"] for c in self._chunks}
        pairs  = [[query, uid_to_chunk.get(uid, "")] for uid in fused]
        rm     = self._get_rerank_model()
        scores = rm.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        reranked = sorted(zip(fused, scores), key=lambda x: x[1], reverse=True)

        # 时序消解（按 date 降序）
        uid_to_date = {c["chunk_id"]: c.get("metadata", {}).get("date", "") for c in self._chunks}
        top4 = [uid for uid, _ in reranked[:self.top_k_rerank]]
        top4.sort(key=lambda uid: uid_to_date.get(uid, ""), reverse=True)

        return top4


# ──────────────────────────────────────────────────────────────────────────────
# 主实验运行
# ──────────────────────────────────────────────────────────────────────────────

async def run_all_experiments():
    from openai import AsyncOpenAI

    print("\n" + "=" * 65)
    print("  Strata 消融实验")
    print("=" * 65)

    api_key  = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model    = os.environ.get("LLM_MODEL", "qwen-plus")

    llm = AsyncOpenAI(api_key=api_key, base_url=base_url)

    results = {}

    # ──────────────────────────────────────────────────────────────────────────
    # 指标一：Gate-F1
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[1/4] 评估澄清门控 Gate-F1（B 类数据集）...")
    dataset_b = load_dataset("B")

    print("  → 仅规则层（无 LLM 兜底）")
    gate_rule_only = await eval_gate_f1(dataset_b, llm, model, use_llm_fallback=False)

    print("  → 规则层 + LLM 兜底（完整门控）")
    gate_full = await eval_gate_f1(dataset_b, llm, model, use_llm_fallback=True)

    results["gate"] = {
        "rule_only": {
            "precision": round(gate_rule_only.precision, 4),
            "recall":    round(gate_rule_only.recall, 4),
            "f1":        round(gate_rule_only.f1, 4),
        },
        "full": {
            "precision": round(gate_full.precision, 4),
            "recall":    round(gate_full.recall, 4),
            "f1":        round(gate_full.f1, 4),
            "type_breakdown": gate_full.type_breakdown,
        },
    }
    print(f"  规则层: P={gate_rule_only.precision:.3f} R={gate_rule_only.recall:.3f} F1={gate_rule_only.f1:.3f}")
    print(f"  完整:   P={gate_full.precision:.3f} R={gate_full.recall:.3f} F1={gate_full.f1:.3f}")

    # ──────────────────────────────────────────────────────────────────────────
    # 指标二：CRA@4 / MRR（消融各检索模块）
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[2/4] 评估检索链路 CRA@4 / MRR（全量数据集）...")
    dataset_all = load_dataset(None)

    retrieval_results = {}

    configs = [
        ("仅向量检索",              SimpleVectorRetriever(top_k=4)),
        ("完整 AD-LTM（向量+BM25+RRF+Reranker+时序）",
         HybridRetriever(use_reranker=True)),
    ]

    for name, retriever in configs:
        print(f"  → {name}")

        async def setup_fn(chunks, _r=retriever):
            await _r.setup(chunks)

        async def retrieve_fn(query, _r=retriever):
            return await _r.retrieve(query)

        r = await eval_retrieval(dataset_all, retrieve_fn, setup_fn)
        retrieval_results[name] = {
            "hit_rate_at_4": round(r.hit_rate_at_4, 4),
            "mrr":           round(r.mrr, 4),
            "precision_at_4": round(r.precision_at_4, 4),
            "total_queries": r.total,
        }
        print(f"    Hit@4={r.hit_rate_at_4:.3f}  MRR={r.mrr:.3f}  P@4={r.precision_at_4:.3f}  (n={r.total})")

    results["retrieval"] = retrieval_results

    # ──────────────────────────────────────────────────────────────────────────
    # 指标三：TOR / TER
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[3/4] 评估工具输出压缩（C 类数据集）...")
    dataset_c = load_dataset("C")
    comp = eval_compression(dataset_c)

    results["compression"] = {
        "tor":               round(comp.tor, 4),
        "ter":               round(comp.ter, 4),
        "total_calls":       comp.total_calls,
        "total_key_points":  comp.total_key_points,
        "retained_points":   comp.retained_points,
        "total_raw_tokens":  comp.total_raw_tokens,
        "total_cmp_tokens":  comp.total_cmp_tokens,
    }
    print(f"  TOR={comp.tor:.3f}  TER={comp.ter:.3f}")
    print(f"  工具调用数={comp.total_calls}  关键点={comp.total_key_points}  保留={comp.retained_points}")

    # ──────────────────────────────────────────────────────────────────────────
    # 指标四：PCS
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[4/4] 评估偏好提取召回率 PCS（全量数据集）...")
    pref = await eval_preference(dataset_all, llm, model)

    results["preference"] = {
        "pcs":            round(pref.pcs, 4),
        "total_events":   pref.total_events,
        "captured_events": pref.captured_events,
        "field_breakdown": pref.field_breakdown,
    }
    print(f"  PCS={pref.pcs:.3f}  总事件={pref.total_events}  捕获={pref.captured_events}")

    # ──────────────────────────────────────────────────────────────────────────
    # 汇总输出
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  实验结果汇总")
    print("=" * 65)
    print(f"  Gate-F1  (完整双层)  : {results['gate']['full']['f1']:.3f}")
    print(f"  CRA@4    (完整AD-LTM): {list(retrieval_results.values())[-1]['hit_rate_at_4']:.3f}")
    print(f"  MRR      (完整AD-LTM): {list(retrieval_results.values())[-1]['mrr']:.3f}")
    print(f"  TOR                  : {results['compression']['tor']:.3f}")
    print(f"  TER                  : {results['compression']['ter']:.3f}")
    print(f"  PCS                  : {results['preference']['pcs']:.3f}")

    # 写出结果
    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已写出: {out_path}")
    print("=" * 65)
    return results


if __name__ == "__main__":
    asyncio.run(run_all_experiments())
