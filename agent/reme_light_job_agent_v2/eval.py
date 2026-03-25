import asyncio
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET_DIR = REPO_ROOT / "experiments" / "dataset"

import dotenv
dotenv.load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from openai import AsyncOpenAI


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(type_filter: str | None = None, max_convs: int | None = None) -> list[dict]:
    if type_filter:
        path = DATASET_DIR / f"dataset_gt_v3_type_{type_filter}.json"
    else:
        path = DATASET_DIR / "dataset_gt_v3_full.json"
    if not path.exists():
        print(f"❌ 数据集文件不存在: {path}")
        print("   请先运行: cd experiments/ && python dataset/generate_dataset.py")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if max_convs:
        data = data[:max_convs]
    return data


def _tokenize(text: str) -> list[str]:
    try:
        import jieba
        return [t for t in jieba.cut(text) if t.strip()]
    except ImportError:
        return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())


def _count_tokens_simple(text: str) -> int:
    """简单 token 估算（字符数 / 2）。"""
    return max(1, len(text) // 2)


def _keyword_match(text: str, key_point: str) -> bool:
    parts = re.split(r"[：:]", key_point, maxsplit=1)
    core = parts[-1].strip() if len(parts) > 1 else key_point.strip()
    core = re.sub(r"[（(].*?[)）]", "", core).strip()
    return bool(core) and core in text


BANNER = "=" * 60


# ──────────────────────────────────────────────────────────────────────────────
# 指标一：Gate-F1（小样本验证）
# ──────────────────────────────────────────────────────────────────────────────

async def eval_gate_f1_quick(llm: AsyncOpenAI, model: str) -> dict:
    print(f"\n{'─'*60}")
    print("📌 指标一：澄清门控 Gate-F1（B 类前 5 组）")
    print('─'*60)

    from clarification_gate import ClarificationGate
    gate = ClarificationGate(llm_client=llm, model=model, reme=None)

    dataset = load_dataset("B", max_convs=5)
    tp = fp = fn = 0

    for conv in dataset:
        simulated: list[dict] = []
        amb_set = set(conv.get("ambiguous_turns", []))

        for t in conv.get("turns", []):
            if t["role"] != "user":
                simulated.append({"role": t["role"], "content": t["content"]})
                continue

            try:
                r = await gate.clarify_gate(
                    user_input=t["content"],
                    recent_messages=simulated[-4:],
                    profile_summary="",
                )
                labeled    = t["turn"] in amb_set
                predicted  = r.is_ambiguous
                if predicted and labeled:     tp += 1
                elif predicted and not labeled: fp += 1
                elif not predicted and labeled: fn += 1
            except Exception as e:
                print(f"  ⚠️  门控异常: {e}")
            simulated.append({"role": "user", "content": t["content"]})

    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")
    status = "✅ 通过（F1 >= 0.80）" if f1 >= 0.80 else "⚠️  F1 偏低"
    print(f"  → {status}")
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "status": f1 >= 0.80}


# ──────────────────────────────────────────────────────────────────────────────
# 指标二：CRA@4 / MRR（小样本验证）
# ──────────────────────────────────────────────────────────────────────────────

async def eval_retrieval_quick() -> dict:
    print(f"\n{'─'*60}")
    print("📌 指标二：检索链路 CRA@4 / MRR（A 类前 5 组）")
    print('─'*60)

    try:
        from sentence_transformers import SentenceTransformer
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("  ⚠️  缺少依赖，跳过检索评估。请安装: sentence-transformers rank-bm25")
        return {"hit_rate_at_4": None, "mrr": None, "status": False}

    dataset    = load_dataset("A", max_convs=5)
    embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    total = hits = 0
    mrr_sum = 0.0

    for conv in dataset:
        chunks = conv.get("memory_chunks", [])
        if not chunks:
            continue

        texts = [c["text"] for c in chunks]
        vecs  = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        corpus   = [_tokenize(t) for t in texts]
        bm25_idx = BM25Okapi(corpus)
        uid_list = [c["chunk_id"] for c in chunks]

        for q_item, gt_item in zip(
            conv.get("retrieval_queries", []),
            conv.get("relevant_chunks", []),
        ):
            query    = q_item.get("query", "")
            gt_uuids = set(gt_item.get("relevant_chunks", []))
            if not query or not gt_uuids:
                continue

            # 向量路
            q_vec = embed_model.encode(query, normalize_embeddings=True, show_progress_bar=False)
            v_scores = [sum(a * b for a, b in zip(q_vec, e)) for e in vecs]
            v_ranked = sorted(range(len(v_scores)), key=lambda i: v_scores[i], reverse=True)[:15]
            v_results = [(uid_list[i], v_scores[i]) for i in v_ranked]

            # BM25 路
            b_scores = bm25_idx.get_scores(_tokenize(query)).tolist()
            max_s = max(b_scores) if b_scores else 0.0
            b_ranked = sorted(range(len(b_scores)), key=lambda i: b_scores[i], reverse=True)[:15]
            b_results = [(uid_list[i], b_scores[i] / max_s) for i in b_ranked if b_scores[i] > 0] if max_s > 0 else []

            # RRF
            rrf: dict[str, float] = {}
            for rank, (uid, _) in enumerate(v_results, 1):
                rrf[uid] = rrf.get(uid, 0.0) + 1.0 / (60 + rank)
            for rank, (uid, _) in enumerate(b_results, 1):
                rrf[uid] = rrf.get(uid, 0.0) + 1.0 / (60 + rank)
            retrieved = [uid for uid, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)][:4]

            total += 1
            hit = any(uid in gt_uuids for uid in retrieved)
            if hit:
                hits += 1
            for rank, uid in enumerate(retrieved, 1):
                if uid in gt_uuids:
                    mrr_sum += 1.0 / rank
                    break

    hit_rate = hits / total if total > 0 else 0.0
    mrr      = mrr_sum / total if total > 0 else 0.0

    print(f"  总查询数={total}  命中={hits}")
    print(f"  Hit Rate@4={hit_rate:.3f}  MRR={mrr:.3f}")
    status = "✅ 通过（Hit@4 >= 0.65）" if hit_rate >= 0.65 else "⚠️  召回偏低"
    print(f"  → {status}")
    return {"hit_rate_at_4": round(hit_rate, 4), "mrr": round(mrr, 4), "status": hit_rate >= 0.65}


# ──────────────────────────────────────────────────────────────────────────────
# 指标三：TOR / TER（小样本验证）
# ──────────────────────────────────────────────────────────────────────────────

def eval_compression_quick() -> dict:
    print(f"\n{'─'*60}")
    print("📌 指标三：工具输出压缩 TOR / TER（C 类前 5 组）")
    print('─'*60)

    dataset = load_dataset("C", max_convs=5)
    total_kp = retained_kp = 0
    total_raw = total_cmp = 0

    for conv in dataset:
        ki_map = {ki["turn"]: ki for ki in conv.get("key_information", [])}
        for tc in conv.get("tool_calls", []):
            ki = ki_map.get(tc["turn"])
            if not ki:
                continue
            compacted  = tc.get("compacted_output", "")
            raw_output = tc.get("raw_output", "")
            for kp in ki.get("key_points", []):
                total_kp += 1
                if _keyword_match(compacted, kp):
                    retained_kp += 1
            total_raw += _count_tokens_simple(raw_output)
            total_cmp += _count_tokens_simple(compacted)

    tor = retained_kp / total_kp if total_kp > 0 else 0.0
    ter = (total_raw - total_cmp) / total_raw if total_raw > 0 else 0.0

    print(f"  关键点总数={total_kp}  保留={retained_kp}")
    print(f"  TOR={tor:.3f}  TER={ter:.3f}")
    status_tor = "✅ 通过（TOR >= 0.80）" if tor >= 0.80 else "⚠️  保留率偏低"
    status_ter = "✅ 通过（TER >= 0.40）" if ter >= 0.40 else "⚠️  节省率偏低"
    print(f"  → TOR: {status_tor}")
    print(f"  → TER: {status_ter}")
    return {
        "tor": round(tor, 4),
        "ter": round(ter, 4),
        "status_tor": tor >= 0.80,
        "status_ter": ter >= 0.40,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 指标四：PCS（小样本验证）
# ──────────────────────────────────────────────────────────────────────────────

async def eval_preference_quick(llm: AsyncOpenAI, model: str) -> dict:
    print(f"\n{'─'*60}")
    print("📌 指标四：偏好提取召回率 PCS（前 5 组）")
    print('─'*60)

    PROMPT = """\
从以下对话片段中提取用户的求职偏好信息。
只返回 JSON，不要其他文字。如某项没有提及，值为 null。

字段：目标岗位, 目标城市, 技术栈, 目标公司_行业, 薪资预期, 面试薄弱点, 简历修改偏好

对话：
{conversation}

JSON："""

    dataset = load_dataset(None, max_convs=5)
    total = captured = 0

    for conv in dataset:
        pref_changes = conv.get("preference_changes", [])
        if not pref_changes:
            continue

        turns = conv.get("turns", [])
        conv_text = "\n".join(
            f"{t['role']}: {t['content'][:150]}" for t in turns
        )[:3000]

        try:
            resp = await llm.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT.format(conversation=conv_text)}],
                temperature=0,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            extracted = json.loads(raw)
        except Exception as e:
            print(f"  ⚠️  提取失败: {e}")
            extracted = {}

        final_prefs: dict[str, str] = {}
        for p in pref_changes:
            final_prefs[p["field"]] = p["value"]

        for field_name, expected in final_prefs.items():
            total += 1
            ext_val = extracted.get(field_name, "")
            if ext_val and expected in str(ext_val):
                captured += 1

    pcs = captured / total if total > 0 else 0.0
    print(f"  偏好变更事件总数={total}  捕获={captured}")
    print(f"  PCS={pcs:.3f}")
    status = "✅ 通过（PCS >= 0.80）" if pcs >= 0.80 else "⚠️  召回率偏低"
    print(f"  → {status}")
    return {"pcs": round(pcs, 4), "total": total, "captured": captured, "status": pcs >= 0.80}


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────

async def run_quick_eval():
    print(f"\n{BANNER}")
    print("  Strata Agent v2 — 四维快速评估")
    print(BANNER)

    api_key  = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model    = os.environ.get("LLM_MODEL", "qwen-plus")

    if not api_key:
        print("❌ 未配置 LLM_API_KEY，请检查 .env 文件")
        return

    llm = AsyncOpenAI(api_key=api_key, base_url=base_url)
    results = {}

    results["gate"]        = await eval_gate_f1_quick(llm, model)
    results["retrieval"]   = await eval_retrieval_quick()
    results["compression"] = eval_compression_quick()
    results["preference"]  = await eval_preference_quick(llm, model)

    print(f"\n{BANNER}")
    print("  评估总结")
    print(BANNER)

    all_pass = True
    for name, r in results.items():
        statuses = [v for k, v in r.items() if k.startswith("status")]
        passed   = all(statuses) if statuses else False
        if not passed:
            all_pass = False
        icon = "✅" if passed else "⚠️ "
        print(f"  {icon} {name}: {r}")

    print(f"\n最终结果: {'🎉 全部通过' if all_pass else '⚠️  部分指标需关注'}")
    print(BANNER)
    print("\n如需完整消融实验，请运行：")
    print("  cd experiments/ && python ablation_study.py\n")
    return results


if __name__ == "__main__":
    asyncio.run(run_quick_eval())
