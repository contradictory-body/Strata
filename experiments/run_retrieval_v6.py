"""
run_retrieval_v6.py — v6 全链路三路检索 CRA@4 全量评估
=========================================================
运行位置：Strata_v6/experiments/ 目录下

测指标的过程：
  对全量 50 组对话（822 个检索查询），测量 v6 完整三路检索链路的召回准确率。

  【写路径（每组对话开始前）】
  将该组对话的 memory_chunks 加载到内存检索器。对每个 Chunk 的原文做
  BGE Embedding（原文向量），同时调用 intent_query_builder.extract_slots()
  提取槽位并生成意图标准句，对意图标准句也做 BGE Embedding（意图向量）。
  此外，对所有 Chunk 的分词结果构建 BM25 倒排索引。

  【读路径（每个 Query）】
  用 build_intent_sentence_from_query() 从 Query 中提取槽位生成意图标准句。
  然后同时启动三路检索：
    路1 — 原文向量路：Query 原文与所有 Chunk 原文向量的余弦相似度，取 Top-15
    路2 — 意图向量路（has_slots=True 时启动）：意图标准句与所有 Chunk
          意图向量的余弦相似度，取 Top-15，命中后映射回原文 Chunk ID
    路3 — BM25 路：Query 分词后在 BM25 索引中检索，取 Top-15
  三路结果经 RRF（k=60）融合为 Top-8，再送 BGE-Reranker-Base 交叉编码精排，
  取 Top-4，最后按 metadata 中的 date 字段降序做时序消解。

  【指标计算】
  将返回的 4 个 Chunk ID 与 Ground Truth relevant_chunks 对比：
    Hit@4 (CRA@4)：4 个中至少 1 个命中则计 Hit，统计全量命中率
    MRR：         第一个命中 Chunk 的位置倒数，统计平均
    P@4：         4 个中命中数量除以 4，统计平均

  同时与基线（v5 两路 HybridRetriever）的结果对比，展示 v6 的提升幅度。

运行方式：
  cd experiments/
  python run_retrieval_v6.py

约需时间：30~60 分钟（Reranker 在 CPU 上约每 Query 0.1~0.3s，全量 822 个）
输出文件：results/retrieval_v6_result.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT   = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "agent" / "reme_light_job_agent_v2"))

import dotenv
dotenv.load_dotenv(dotenv_path=REPO_ROOT / "agent" / "reme_light_job_agent_v2" / ".env", override=True)

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("RetrievalV6Eval")

DATASET_DIR = SCRIPT_DIR / "dataset"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BANNER = "=" * 65


# ──────────────────────────────────────────────────────────────────────────────
# 评估结果数据类
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    total:      int   = 0
    hits:       int   = 0
    mrr_sum:    float = 0.0
    p_at_4_sum: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0

    @property
    def mrr(self) -> float:
        return self.mrr_sum / self.total if self.total > 0 else 0.0

    @property
    def precision_at_4(self) -> float:
        return self.p_at_4_sum / self.total if self.total > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# v6 三路检索后端（内存版，与 ablation_study 的 HybridRetriever 结构一致）
# ──────────────────────────────────────────────────────────────────────────────

class HybridRetrieverV6:
    """
    v6 完整三路检索链路：原文向量 + 意图向量 + BM25 + RRF + Reranker + 时序消解。

    与 ablation_study.HybridRetriever 的区别：
      - setup() 额外为每个 Chunk 生成意图标准句并编码（意图向量预计算）
      - retrieve() 新增意图向量路，has_slots=True 时启动三路，否则退化两路
    """

    TOP_K_VECTOR = 15
    TOP_K_BM25   = 15
    TOP_K_RRF    = 8
    TOP_K_RERANK = 4
    RRF_K        = 60

    def __init__(self):
        self._chunks:           list[dict]         = []
        self._orig_embeddings:  list[list[float]]  = []   # 原文向量
        self._intent_embeddings: list[list[float]] = []   # 意图向量
        self._intent_sentences:  list[str]         = []   # 意图标准句
        self._bm25_index = None
        self._bm25_ids:    list[str] = []
        self._embed_model  = None
        self._rerank_model = None

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5",device="cuda",)
        return self._embed_model

    def _get_rerank_model(self):
        if self._rerank_model is None:
            from sentence_transformers import CrossEncoder
            self._rerank_model = CrossEncoder("BAAI/bge-reranker-base", max_length=512,device="cuda",)
        return self._rerank_model

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        try:
            import jieba
            return [t for t in jieba.cut(text) if t.strip()]
        except ImportError:
            return re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]|[a-zA-Z0-9_]+", text.lower())

    async def setup(self, chunks: list[dict]) -> None:
        """
        加载一组对话的 memory_chunks，预计算原文向量 + 意图向量 + BM25 索引。

        写路径预计算：意图标准句在这里生成并编码，检索时直接使用，
        与 advanced_memory_manager.update_memory_from_file() 的写路径逻辑对齐。
        """
        from rank_bm25 import BM25Okapi
        from intent_query_builder import extract_slots, slots_to_sentence

        self._chunks = chunks
        if not chunks:
            self._orig_embeddings = []
            self._intent_embeddings = []
            self._intent_sentences = []
            self._bm25_index = None
            return

        model = self._get_embed_model()
        texts = [c["text"] for c in chunks]

        # 原文向量
        orig_vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self._orig_embeddings = orig_vecs.tolist()

        # 意图向量：对每个 Chunk 提取槽位 → 标准句 → 编码
        self._intent_sentences = [
            slots_to_sentence(extract_slots(text))
            for text in texts
        ]
        intent_vecs = model.encode(
            self._intent_sentences, normalize_embeddings=True, show_progress_bar=False
        )
        self._intent_embeddings = intent_vecs.tolist()

        # BM25 倒排索引（只用原文）
        self._bm25_ids   = [c["chunk_id"] for c in chunks]
        corpus           = [self._tokenize(t) for t in texts]
        self._bm25_index = BM25Okapi(corpus)

    async def retrieve(self, query: str) -> list[str]:
        """
        三路检索 + RRF + Reranker + 时序消解，返回 Top-4 Chunk ID。
        """
        if not self._chunks:
            return []

        from intent_query_builder import build_intent_sentence_from_query
        import numpy as np

        model = self._get_embed_model()

        # 原文向量检索
        q_orig_vec = model.encode(query, normalize_embeddings=True, show_progress_bar=False)
        orig_mat   = np.array(self._orig_embeddings, dtype=np.float32)
        orig_sims  = (orig_mat @ q_orig_vec).tolist()
        orig_ranked = sorted(range(len(orig_sims)), key=lambda i: orig_sims[i], reverse=True)
        v_results   = [
            (self._chunks[i]["chunk_id"], orig_sims[i])
            for i in orig_ranked[:self.TOP_K_VECTOR]
        ]

        # 意图向量检索（has_slots=True 时启动）
        intent_sent, _, has_slots = build_intent_sentence_from_query(query)
        intent_results: list[tuple[str, float]] = []
        if has_slots and intent_sent:
            q_intent_vec  = model.encode(intent_sent, normalize_embeddings=True, show_progress_bar=False)
            intent_mat    = np.array(self._intent_embeddings, dtype=np.float32)
            intent_sims   = (intent_mat @ q_intent_vec).tolist()
            intent_ranked = sorted(range(len(intent_sims)), key=lambda i: intent_sims[i], reverse=True)
            intent_results = [
                (self._chunks[i]["chunk_id"], intent_sims[i])
                for i in intent_ranked[:self.TOP_K_VECTOR]
            ]

        # BM25 检索
        bm25_results: list[tuple[str, float]] = []
        if self._bm25_index:
            tokens   = self._tokenize(query)
            b_scores = self._bm25_index.get_scores(tokens).tolist()
            max_s    = max(b_scores) if b_scores else 0.0
            if max_s > 0:
                b_ranked = sorted(range(len(b_scores)), key=lambda i: b_scores[i], reverse=True)
                bm25_results = [
                    (self._bm25_ids[i], b_scores[i] / max_s)
                    for i in b_ranked[:self.TOP_K_BM25]
                    if b_scores[i] > 0
                ]

        # RRF 三路融合
        rrf: dict[str, float] = {}
        for rank, (uid, _) in enumerate(v_results, 1):
            rrf[uid] = rrf.get(uid, 0.0) + 1.0 / (self.RRF_K + rank)
        for rank, (uid, _) in enumerate(intent_results, 1):
            rrf[uid] = rrf.get(uid, 0.0) + 1.0 / (self.RRF_K + rank)
        for rank, (uid, _) in enumerate(bm25_results, 1):
            rrf[uid] = rrf.get(uid, 0.0) + 1.0 / (self.RRF_K + rank)
        fused = [uid for uid, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)][:self.TOP_K_RRF]

        if not fused:
            return []

        # Reranker 精排
        uid_to_text = {c["chunk_id"]: c["text"] for c in self._chunks}
        pairs = [[query, uid_to_text.get(uid, "")] for uid in fused]
        rm    = self._get_rerank_model()
        scores = rm.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        reranked = sorted(zip(fused, scores), key=lambda x: x[1], reverse=True)

        # 时序消解（按 date 字段降序，新记忆优先）
        uid_to_date = {
            c["chunk_id"]: c.get("metadata", {}).get("date", "")
            for c in self._chunks
        }
        top4 = [uid for uid, _ in reranked[:self.TOP_K_RERANK]]
        top4.sort(key=lambda uid: uid_to_date.get(uid, ""), reverse=True)

        return top4


# ──────────────────────────────────────────────────────────────────────────────
# 通用检索评估函数
# ──────────────────────────────────────────────────────────────────────────────

async def eval_retrieval_with_retriever(
    dataset: list[dict],
    retriever,
    label: str,
) -> RetrievalResult:
    """对全量数据集运行检索评估，返回 RetrievalResult。"""
    result = RetrievalResult()

    for conv_idx, conv in enumerate(dataset, 1):
        chunks = conv.get("memory_chunks", [])
        if not chunks:
            continue

        await retriever.setup(chunks)

        queries       = conv.get("retrieval_queries", [])
        ground_truths = conv.get("relevant_chunks", [])

        for q_item, gt_item in zip(queries, ground_truths):
            query    = q_item.get("query", "")
            gt_uuids = set(gt_item.get("relevant_chunks", []))
            if not query or not gt_uuids:
                continue

            try:
                retrieved = await retriever.retrieve(query)
            except Exception as e:
                logger.warning(f"[{label}] 检索失败: {e}")
                retrieved = []

            result.total += 1

            # Hit@4
            hit = any(uid in gt_uuids for uid in retrieved)
            if hit:
                result.hits += 1

            # MRR
            for rank, uid in enumerate(retrieved, 1):
                if uid in gt_uuids:
                    result.mrr_sum += 1.0 / rank
                    break

            # P@4
            hits_in_4 = sum(1 for uid in retrieved if uid in gt_uuids)
            result.p_at_4_sum += hits_in_4 / 4.0

        if conv_idx % 5 == 0 or conv_idx == len(dataset):
            print(f"    [{label}] {conv_idx}/{len(dataset)} 组完成  "
                  f"当前 Hit@4={result.hit_rate:.3f}")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

async def run_retrieval_v6():
    print(f"\n{BANNER}")
    print("  v6 全链路三路检索 CRA@4 全量评估")
    print(f"{BANNER}")
    print("  数据集: 全量 50 组对话，822 个检索查询")
    print("  链路:   原文向量 + 意图向量 + BM25 → RRF → Reranker → 时序消解")
    print("  对照:   v5 两路基线（向量 + BM25 + Reranker）")

    # 加载数据集
    path = DATASET_DIR / "dataset_gt_v3_full.json"
    if not path.exists():
        print(f"❌ 全量数据集不存在: {path}")
        return
    with open(path, encoding="utf-8") as f:
        dataset = json.load(f)

    n_queries = sum(len(c.get("retrieval_queries", [])) for c in dataset)
    print(f"  加载完成: {len(dataset)} 组对话，{n_queries} 个查询")
    print("\n  ⚠️  CPU 模式下约需 30~60 分钟，请耐心等待\n")

    # ── v5 两路基线 ───────────────────────────────────────────────────────
    print(f"  [1/2] v5 基线（两路：向量 + BM25 + Reranker）")
    from ablation_study import HybridRetriever
    retriever_v5 = HybridRetriever(use_reranker=True)
    result_v5 = await eval_retrieval_with_retriever(dataset, retriever_v5, "v5-两路")

    # ── v6 三路检索 ───────────────────────────────────────────────────────
    print(f"\n  [2/2] v6 三路检索（原文向量 + 意图向量 + BM25 + Reranker）")
    retriever_v6 = HybridRetrieverV6()
    result_v6 = await eval_retrieval_with_retriever(dataset, retriever_v6, "v6-三路")

    # ── 汇总 ──────────────────────────────────────────────────────────────
    delta_hit = result_v6.hit_rate - result_v5.hit_rate
    delta_mrr = result_v6.mrr     - result_v5.mrr

    print(f"\n{BANNER}")
    print("  检索评估结果汇总")
    print(BANNER)
    print(f"  {'指标':<12}  {'v5 两路基线':>14}  {'v6 三路检索':>14}  {'提升 Δ':>10}")
    print(f"  {'─'*12}  {'─'*14}  {'─'*14}  {'─'*10}")
    print(f"  {'Hit@4 (CRA@4)':<12}  {result_v5.hit_rate:>14.3f}  {result_v6.hit_rate:>14.3f}  {delta_hit:>+10.3f}")
    print(f"  {'MRR':<12}  {result_v5.mrr:>14.3f}  {result_v6.mrr:>14.3f}  {delta_mrr:>+10.3f}")
    print(f"  {'P@4':<12}  {result_v5.precision_at_4:>14.3f}  {result_v6.precision_at_4:>14.3f}  {result_v6.precision_at_4-result_v5.precision_at_4:>+10.3f}")
    print(f"  {'查询总数':<12}  {result_v5.total:>14}  {result_v6.total:>14}")

    status = "✅ 通过（CRA@4 ≥ 0.65）" if result_v6.hit_rate >= 0.65 else f"⚠️  CRA@4={result_v6.hit_rate:.3f} 未达 0.65"
    print(f"\n  最终判定: {status}")
    print(f"  相比 v5 基线: {'✅ 有提升' if delta_hit > 0 else '⚠️  未见提升'} (Δ={delta_hit:+.3f})")

    # ── 写出 JSON ─────────────────────────────────────────────────────────
    out = {
        "v5_baseline": {
            "hit_rate_at_4":  round(result_v5.hit_rate, 4),
            "mrr":            round(result_v5.mrr, 4),
            "precision_at_4": round(result_v5.precision_at_4, 4),
            "total_queries":  result_v5.total,
            "description":    "两路检索：原文向量 + BM25 + RRF + Reranker + 时序消解",
        },
        "v6_three_path": {
            "hit_rate_at_4":  round(result_v6.hit_rate, 4),
            "mrr":            round(result_v6.mrr, 4),
            "precision_at_4": round(result_v6.precision_at_4, 4),
            "total_queries":  result_v6.total,
            "description":    "三路检索：原文向量 + 意图向量 + BM25 + RRF + Reranker + 时序消解",
        },
        "delta": {
            "hit_rate_at_4":  round(delta_hit, 4),
            "mrr":            round(delta_mrr, 4),
        },
        "dataset": {"type": "full", "n_conv": len(dataset), "n_queries": n_queries},
        "target_cra4": 0.65,
        "pass": result_v6.hit_rate >= 0.65,
    }
    out_path = RESULTS_DIR / "retrieval_v6_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  详细结果已写出: {out_path}")
    print(BANNER)


if __name__ == "__main__":
    asyncio.run(run_retrieval_v6())
