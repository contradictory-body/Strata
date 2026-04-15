"""
run_gate_eval.py — 澄清门控 Gate-F1 全量评估
===============================================
运行位置：Strata_v6/experiments/ 目录下

测指标的过程：
  对 B 类全量 15 组对话（共约 219 次用户发言），逐对话逐轮模拟真实对话场景。
  每一轮用户输入送入澄清门控（规则层优先，置信度低时升级 LLM 兜底），
  得到 is_ambiguous 二分类预测结果，与人工标注的 ambiguous_turns 列表对比。
  统计 TP（该追问且预测追问）、FP（不该追问但预测追问）、FN（该追问但放行了），
  最终计算 Precision = TP/(TP+FP)、Recall = TP/(TP+FN)、F1 = 调和平均。
  同时按模糊类型（reference_unclear / intent_unclear / constraint_missing /
  intent_mixed）分别统计各类型的命中率，定位规则层的覆盖薄弱点。

运行方式：
  cd experiments/
  python run_gate_eval.py

约需时间：5~15 分钟（取决于 LLM API 响应速度，共约 219 次 API 调用）
输出文件：results/gate_f1_result.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT   = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "agent" / "reme_light_job_agent_v2"))

import dotenv
dotenv.load_dotenv(dotenv_path=REPO_ROOT / "agent" / "reme_light_job_agent_v2" / ".env", override=True)

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

DATASET_DIR = SCRIPT_DIR / "dataset"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BANNER = "=" * 65


def load_dataset_b() -> list[dict]:
    path = DATASET_DIR / "dataset_gt_v3_type_B.json"
    if not path.exists():
        raise FileNotFoundError(f"B 类数据集不存在: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


async def run_gate_eval():
    from openai import AsyncOpenAI
    # 复用 ablation_study 中已有的评估函数和结果类
    from ablation_study import eval_gate_f1, GateEvalResult

    api_key  = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model    = os.environ.get("LLM_MODEL", "qwen-plus")

    if not api_key:
        print("❌ 未配置 LLM_API_KEY，请检查 .env 文件")
        return

    llm = AsyncOpenAI(api_key=api_key, base_url=base_url)

    print(f"\n{BANNER}")
    print("  澄清门控 Gate-F1 全量评估")
    print(f"{BANNER}")
    print(f"  模型: {model}")
    print(f"  数据集: B 类全量（15 组对话，约 219 次用户发言）")
    print(f"  说明: 每次用户发言经过门控二分类，与人工标注对比计算 F1")

    dataset_b = load_dataset_b()
    n_conv    = len(dataset_b)
    n_turns   = sum(
        sum(1 for t in c.get("turns", []) if t["role"] == "user")
        for c in dataset_b
    )
    print(f"  加载完成: {n_conv} 组对话，共 {n_turns} 轮用户发言")
    print(f"\n  正在评估（每轮可能调用 1 次 LLM，请耐心等待）...\n")

    # ── 两种门控配置的消融对比 ────────────────────────────────────────────
    print("  [1/2] 仅规则层（无 LLM 兜底）...")
    result_rule = await eval_gate_f1(dataset_b, llm, model, use_llm_fallback=False)
    print(f"        P={result_rule.precision:.3f}  R={result_rule.recall:.3f}  F1={result_rule.f1:.3f}")

    print("\n  [2/2] 完整双层门控（规则层 + LLM 兜底）...")
    result_full = await eval_gate_f1(dataset_b, llm, model, use_llm_fallback=True)
    print(f"        P={result_full.precision:.3f}  R={result_full.recall:.3f}  F1={result_full.f1:.3f}")

    # ── 按模糊类型分析 ─────────────────────────────────────────────────────
    print(f"\n{BANNER}")
    print("  各模糊类型命中率（完整门控）")
    print(BANNER)
    for amb_type, counts in sorted(result_full.type_breakdown.items()):
        tp_t = counts.get("tp", 0)
        fn_t = counts.get("fn", 0)
        total_t = tp_t + fn_t
        recall_t = tp_t / total_t if total_t > 0 else 0.0
        print(f"  {amb_type:<25}  TP={tp_t}  FN={fn_t}  Recall={recall_t:.3f}")

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print(f"\n{BANNER}")
    print("  Gate-F1 评估结果汇总")
    print(BANNER)
    print(f"  仅规则层:    P={result_rule.precision:.3f}  R={result_rule.recall:.3f}  F1={result_rule.f1:.3f}")
    print(f"  完整双层:    P={result_full.precision:.3f}  R={result_full.recall:.3f}  F1={result_full.f1:.3f}")
    print(f"  LLM 提升 ΔF1: {result_full.f1 - result_rule.f1:+.3f}")

    f1_full = result_full.f1
    status = "✅ 通过（F1 ≥ 0.80）" if f1_full >= 0.80 else f"⚠️  F1={f1_full:.3f} 未达 0.80"
    print(f"\n  最终判定: {status}")

    # ── 写出 JSON ─────────────────────────────────────────────────────────
    out = {
        "rule_only": {
            "precision": round(result_rule.precision, 4),
            "recall":    round(result_rule.recall, 4),
            "f1":        round(result_rule.f1, 4),
            "tp": result_rule.tp, "fp": result_rule.fp, "fn": result_rule.fn,
        },
        "full_gate": {
            "precision": round(result_full.precision, 4),
            "recall":    round(result_full.recall, 4),
            "f1":        round(result_full.f1, 4),
            "tp": result_full.tp, "fp": result_full.fp, "fn": result_full.fn,
            "type_breakdown": result_full.type_breakdown,
        },
        "dataset":   {"type": "B", "n_conv": n_conv, "n_turns": n_turns},
        "model":     model,
    }
    out_path = RESULTS_DIR / "gate_f1_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  详细结果已写出: {out_path}")
    print(BANNER)


if __name__ == "__main__":
    asyncio.run(run_gate_eval())
