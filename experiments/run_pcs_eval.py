"""
run_pcs_eval.py — 偏好提取召回率 PCS 全量评估（v2 修复版）
============================================================
修复：
  - 每组对话实时打印进度，避免看起来卡住
  - 每次 LLM 调用加 120 秒超时，超时自动跳过并计入失败
  - 打印每组耗时，方便估计总剩余时间

运行方式（在 experiments/ 目录下）：
  cd experiments/
  python run_pcs_eval.py

约需时间：40~80 分钟（50 次 LLM 调用，每次 30~90 秒）
输出文件：results/pcs_result.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT   = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "agent" / "reme_light_job_agent_v2"))

import dotenv
dotenv.load_dotenv(
    dotenv_path=REPO_ROOT / "agent" / "reme_light_job_agent_v2" / ".env",
    override=True,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

DATASET_DIR = SCRIPT_DIR / "dataset"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BANNER = "=" * 65

LLM_TIMEOUT_SECONDS = 120   # 单次 LLM 调用超时

EXTRACT_PROMPT = """\
从以下对话片段中提取用户的求职偏好信息。
只返回 JSON，不要其他文字。如某项没有提及，值为 null。

字段：目标岗位, 目标城市, 技术栈, 目标公司_行业, 薪资预期, 面试薄弱点, 简历修改偏好

对话：
{conversation}

JSON："""

FIELD_MAP = {
    "目标岗位":    "目标岗位",
    "目标城市":    "目标城市",
    "技术栈":      "技术栈",
    "目标公司_行业": "目标公司_行业",
    "薪资预期":    "薪资预期",
    "面试薄弱点":  "面试薄弱点",
    "简历修改偏好": "简历修改偏好",
}


async def extract_preference(llm, model: str, conversation_text: str) -> dict:
    """单次 LLM 调用，带超时保护，失败返回空字典。"""
    try:
        resp = await asyncio.wait_for(
            llm.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": EXTRACT_PROMPT.format(conversation=conversation_text[:3000]),
                }],
                temperature=0,
                max_tokens=512,
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
        raw = (
            resp.choices[0].message.content.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        return json.loads(raw)
    except asyncio.TimeoutError:
        print(f"      ⚠️  LLM 超时（>{LLM_TIMEOUT_SECONDS}s），跳过本组")
        return {}
    except Exception as e:
        print(f"      ⚠️  LLM 调用失败: {type(e).__name__}: {e!r}")
        return {}


async def run_pcs_eval():
    from openai import AsyncOpenAI

    api_key  = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model    = os.environ.get("LLM_MODEL", "qwen-plus")

    if not api_key:
        print("❌ 未配置 LLM_API_KEY，请检查 .env 文件")
        return

    llm = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # 加载数据集
    path = DATASET_DIR / "dataset_gt_v3_full.json"
    if not path.exists():
        print(f"❌ 全量数据集不存在: {path}")
        return
    with open(path, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\n{BANNER}")
    print("  偏好提取召回率 PCS 全量评估（v2 修复版）")
    print(BANNER)
    print(f"  模型:     {model}")
    print(f"  数据集:   全量 {len(dataset)} 组对话")
    print(f"  超时设置: 每次 LLM 调用最多 {LLM_TIMEOUT_SECONDS} 秒")

    # 只统计有偏好变更的对话
    convs_with_pref = [c for c in dataset if c.get("preference_changes")]
    n_events_total  = sum(len(c["preference_changes"]) for c in convs_with_pref)
    print(f"  有偏好变更: {len(convs_with_pref)} 组，共 {n_events_total} 个变更事件")
    print(f"  预计耗时:   {len(convs_with_pref) * 60 // 60}~{len(convs_with_pref) * 90 // 60} 分钟")
    print(f"\n  开始评估（实时打印每组进度）...\n")

    # 统计变量
    total_events   = 0
    captured_events = 0
    field_breakdown: dict[str, dict] = {}
    skipped = 0

    eval_start = time.time()

    for conv_idx, conv in enumerate(convs_with_pref, 1):
        conv_id   = conv.get("conversation_id", f"conv_{conv_idx}")
        pref_changes = conv.get("preference_changes", [])
        turns     = conv.get("turns", [])

        conv_text = "\n".join(
            f"{t['role']}: {t['content'][:200]}" for t in turns
        )

        t0 = time.time()
        print(f"  [{conv_idx:2d}/{len(convs_with_pref)}] {conv_id}  ({len(pref_changes)} 个变更事件)  ", end="", flush=True)

        extracted = await extract_preference(llm, model, conv_text)
        elapsed   = time.time() - t0

        if not extracted:
            skipped += 1
            total_events += len(pref_changes)
            print(f"跳过  ({elapsed:.1f}s)")
            continue

        # 以最后一次变更为准
        final_prefs: dict[str, str] = {}
        for pref in pref_changes:
            final_prefs[pref["field"]] = pref["value"]

        conv_captured = 0
        for field_name, expected_value in final_prefs.items():
            total_events += 1
            if field_name not in field_breakdown:
                field_breakdown[field_name] = {"total": 0, "captured": 0}
            field_breakdown[field_name]["total"] += 1

            ext_key = FIELD_MAP.get(field_name, field_name)
            ext_val = extracted.get(ext_key, "")
            if ext_val and expected_value in str(ext_val):
                captured_events += 1
                conv_captured += 1
                field_breakdown[field_name]["captured"] += 1

        print(f"命中 {conv_captured}/{len(final_prefs)}  ({elapsed:.1f}s)")

        # 实时滚动 PCS
        cur_pcs = captured_events / total_events if total_events > 0 else 0.0
        elapsed_total = time.time() - eval_start
        avg_per_conv  = elapsed_total / conv_idx
        remaining     = avg_per_conv * (len(convs_with_pref) - conv_idx)
        print(f"       当前 PCS={cur_pcs:.3f}  已用 {elapsed_total/60:.1f}min  预计剩余 {remaining/60:.1f}min")

    # ── 汇总 ──────────────────────────────────────────────────────────────
    pcs = captured_events / total_events if total_events > 0 else 0.0

    print(f"\n{BANNER}")
    print("  各偏好字段命中率")
    print(BANNER)
    for field_name, counts in sorted(field_breakdown.items()):
        total_f    = counts.get("total", 0)
        captured_f = counts.get("captured", 0)
        rate       = captured_f / total_f if total_f > 0 else 0.0
        bar        = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"  {field_name:<12}  [{bar}]  {captured_f}/{total_f} = {rate:.1%}")

    print(f"\n{BANNER}")
    print("  PCS 评估结果汇总")
    print(BANNER)
    print(f"  有偏好变更组数: {len(convs_with_pref)}")
    print(f"  跳过（超时/失败）: {skipped}")
    print(f"  总变更事件数:   {total_events}")
    print(f"  成功捕获事件:   {captured_events}")
    print(f"  PCS:            {pcs:.3f}")
    print(f"  总耗时:         {(time.time() - eval_start)/60:.1f} 分钟")

    status = "✅ 通过（PCS ≥ 0.80）" if pcs >= 0.80 else f"⚠️  PCS={pcs:.3f} 未达 0.80"
    print(f"\n  最终判定: {status}")

    # ── 写出 JSON ─────────────────────────────────────────────────────────
    out = {
        "pcs":              round(pcs, 4),
        "total_events":     total_events,
        "captured_events":  captured_events,
        "skipped_convs":    skipped,
        "field_breakdown":  field_breakdown,
        "dataset":          {"type": "full", "n_conv": len(dataset),
                             "n_conv_with_pref": len(convs_with_pref)},
        "model":            model,
    }
    out_path = RESULTS_DIR / "pcs_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  详细结果已写出: {out_path}")
    print(BANNER)


if __name__ == "__main__":
    asyncio.run(run_pcs_eval())
