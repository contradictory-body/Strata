"""
semantic_cache.py
=================
目标目录：Strata_v5/agent/reme_light_job_agent_v2/semantic_cache.py

语义缓存模块（Semantic Cache）

与 AdvancedMemoryManager 内的精确匹配 LRU 缓存互补：
  · LRU 缓存   → MD5 精确匹配，命中率低，速度极快（dict 查找）
  · 语义缓存   → 向量余弦相似度，命中率高，速度快（矩阵点积）

两级缓存在 retrieve() / retrieve_with_intent() 中串联：
  ① LRU 命中（完全相同 Query）→ 直接返回，≈ 0ms
  ② 语义缓存命中（近似 Query, sim ≥ 0.95）→ 直接返回，≈ 5ms
  ③ 全量检索管线 → Reranker → 写入两级缓存

失效策略：
  · 每条缓存条目绑定 GLOBAL_MEMORY_VERSION
  · 记忆文件写入后版本号递增，旧条目在 get() 时自动跳过
  · set() 时懒清理过期条目，防止内存无限增长
  · invalidate_version() 支持主动清理，在版本递增后立即调用

缓存键：
  · has_slots=True  时以 intent_sentence 向量为 key（语义稳定）
  · has_slots=False 时以 query_text 向量为 key
  两种情况均由 AdvancedMemoryManager 在调用方决定，本模块不感知。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("SemanticCache")

DEFAULT_CAPACITY:  int   = 200
DEFAULT_THRESHOLD: float = 0.95


@dataclass
class _CacheEntry:
    """单条语义缓存条目（内部使用）。"""
    query_vec: list[float]    # 归一化向量（单位向量，点积即余弦相似度）
    answer:    str            # 检索结果字符串（可直接注入 System Prompt）
    version:   int            # 写入时的 GLOBAL_MEMORY_VERSION
    timestamp: float = field(default_factory=time.monotonic)  # 写入时刻，供 FIFO 驱逐使用


class SemanticCache:
    """
    基于向量余弦相似度的语义缓存。

    Usage（在 AdvancedMemoryManager 中）：

        # 初始化（一次）
        self._semantic_cache = SemanticCache(capacity=200, threshold=0.95)

        # 检索路径（retrieve_with_intent 内部）
        sem_result = self._semantic_cache.get(cache_vec, self.GLOBAL_MEMORY_VERSION)
        if sem_result is not None:
            return sem_result          # 命中，跳过全量检索

        context = ... 全量检索 ...

        # 清晰路径写入，模糊路径不写
        self._semantic_cache.set(cache_vec, context, self.GLOBAL_MEMORY_VERSION)

        # 记忆写入时主动失效
        self.GLOBAL_MEMORY_VERSION += 1
        self._semantic_cache.invalidate_version(self.GLOBAL_MEMORY_VERSION)

    Args:
        capacity:  缓存条目上限，超出时 FIFO 驱逐最旧的同版本条目，默认 200。
        threshold: 命中阈值，余弦相似度 ≥ threshold 判定为命中，默认 0.95。
                   建议范围 [0.92, 0.97]：
                     0.95 → 只有近义改写才命中（保守，精度高）
                     0.92 → 语义相近即命中（激进，命中率高但可能误命中）
    """

    def __init__(
        self,
        capacity:  int   = DEFAULT_CAPACITY,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self._entries:   list[_CacheEntry] = []
        self._capacity   = capacity
        self._threshold  = threshold
        self._hits:   int = 0
        self._misses: int = 0
        logger.info(
            f"[SemanticCache] 初始化: capacity={capacity}, threshold={threshold}"
        )

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def get(
        self,
        query_vec:       list[float],
        current_version: int,
    ) -> Optional[str]:
        """
        语义查询。

        遍历所有当前版本条目，批量计算余弦相似度（向量已归一化，点积即余弦）。
        最高相似度 ≥ threshold 则命中，返回对应缓存答案。

        Args:
            query_vec:       归一化查询向量（由 BGEEmbeddingFunction.encode_single 生成）
            current_version: 当前 GLOBAL_MEMORY_VERSION，仅检查同版本条目

        Returns:
            命中时返回缓存结果字符串，未命中返回 None。

        时间复杂度：O(n) 向量点积，n = 同版本条目数，通常 < 200，< 1ms。
        """
        if not self._entries:
            self._misses += 1
            logger.debug("[SemanticCache] MISS 缓存为空")
            return None

        valid = [e for e in self._entries if e.version == current_version]
        if not valid:
            self._misses += 1
            logger.debug(f"[SemanticCache] MISS 无 version={current_version} 的条目")
            return None

        qv = np.array(query_vec, dtype=np.float32)
        # 批量点积（归一化向量的点积 = 余弦相似度）
        mat = np.array([e.query_vec for e in valid], dtype=np.float32)  # (n, dim)
        sims = mat @ qv                                                  # (n,)

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self._threshold:
            self._hits += 1
            logger.info(
                f"[SemanticCache] HIT  sim={best_sim:.4f} "
                f"version={current_version} valid_entries={len(valid)}"
            )
            return valid[best_idx].answer

        self._misses += 1
        logger.debug(
            f"[SemanticCache] MISS max_sim={best_sim:.4f} < {self._threshold}  "
            f"version={current_version} valid_entries={len(valid)}"
        )
        return None

    def set(
        self,
        query_vec:       list[float],
        answer:          str,
        current_version: int,
    ) -> None:
        """
        写入缓存条目（仅在清晰路径调用，模糊路径不调用）。

        写入前执行懒清理：
          1. 移除所有旧版本（version ≠ current_version）条目
          2. 若同版本条目仍超容量，按时间戳 FIFO 驱逐最旧条目

        Args:
            query_vec:       归一化查询向量
            answer:          检索结果字符串
            current_version: 当前 GLOBAL_MEMORY_VERSION
        """
        # ── 懒清理：移除旧版本条目 ─────────────────────────────────────────
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.version == current_version]
        stale = before - len(self._entries)
        if stale > 0:
            logger.debug(f"[SemanticCache] 懒清理 {stale} 条旧版本条目")

        # ── FIFO 驱逐：同版本超容量时移除最旧条目 ────────────────────────
        while len(self._entries) >= self._capacity:
            evicted = self._entries.pop(0)
            logger.debug(
                f"[SemanticCache] FIFO 驱逐 ts={evicted.timestamp:.2f} "
                f"version={evicted.version}"
            )

        self._entries.append(
            _CacheEntry(
                query_vec=list(query_vec),
                answer=answer,
                version=current_version,
            )
        )
        logger.debug(
            f"[SemanticCache] SET  version={current_version} "
            f"total={len(self._entries)}"
        )

    def invalidate_version(self, keep_version: int) -> int:
        """
        主动清除所有非 keep_version 的条目。

        在 GLOBAL_MEMORY_VERSION 递增后立即调用，确保旧缓存及时失效。
        比 get() 的懒过滤更彻底，可释放内存。

        Args:
            keep_version: 要保留的版本号（即新的 GLOBAL_MEMORY_VERSION）

        Returns:
            清除的条目数量。
        """
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.version == keep_version]
        removed = before - len(self._entries)
        if removed > 0:
            logger.info(
                f"[SemanticCache] 主动失效: 清除 {removed} 条旧版本条目，"
                f"保留 version={keep_version} 的 {len(self._entries)} 条"
            )
        return removed

    def clear(self) -> None:
        """清空全部缓存条目并重置统计。"""
        count = len(self._entries)
        self._entries.clear()
        self._hits   = 0
        self._misses = 0
        logger.info(f"[SemanticCache] 已清空 {count} 条条目，统计已重置")

    @property
    def stats(self) -> dict:
        """返回缓存运行统计，用于监控和调试。"""
        total    = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        versions = sorted({e.version for e in self._entries})
        return {
            "entries":   len(self._entries),
            "capacity":  self._capacity,
            "threshold": self._threshold,
            "hits":      self._hits,
            "misses":    self._misses,
            "hit_rate":  round(hit_rate, 4),
            "versions":  versions,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 快速验证（直接运行此文件）
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    def _rand_unit_vec(dim: int = 512) -> list[float]:
        v = np.random.randn(dim).astype(np.float32)
        v /= np.linalg.norm(v)
        return v.tolist()

    def _perturb(vec: list[float], noise: float = 0.02) -> list[float]:
        """微小扰动，模拟同义改写的向量"""
        v = np.array(vec) + np.random.randn(len(vec)) * noise
        v /= np.linalg.norm(v)
        return v.tolist()

    print("=" * 55)
    print("  SemanticCache 快速验证")
    print("=" * 55)

    cache = SemanticCache(capacity=10, threshold=0.95)

    # ── 写入 3 条版本 1 的缓存 ────────────────────────────────────────────
    v1 = _rand_unit_vec()
    v2 = _rand_unit_vec()
    v3 = _rand_unit_vec()

    cache.set(v1, "结果A：后端工程师 北京 35K", current_version=1)
    cache.set(v2, "结果B：产品经理 上海 30K",   current_version=1)
    cache.set(v3, "结果C：前端工程师 深圳 28K", current_version=1)
    print(f"\n写入 3 条 version=1 条目: {cache.stats}")

    # ── 精确命中 ─────────────────────────────────────────────────────────
    r = cache.get(v1, current_version=1)
    print(f"\n[精确命中] {r}")
    assert r == "结果A：后端工程师 北京 35K"

    # ── 近似命中（微小扰动，模拟同义改写）───────────────────────────────
    v1_approx = _perturb(v1, noise=0.01)
    r = cache.get(v1_approx, current_version=1)
    print(f"[近似命中] {r}")
    assert r is not None, "近似向量应该命中"

    # ── 版本不匹配 → 未命中 ───────────────────────────────────────────
    r = cache.get(v1, current_version=2)
    print(f"[版本不匹配] {r}")
    assert r is None, "版本不匹配应该未命中"

    # ── 主动失效 ─────────────────────────────────────────────────────────
    removed = cache.invalidate_version(keep_version=2)
    print(f"\n主动失效 (keep_version=2): 清除 {removed} 条，剩余 {cache.stats['entries']} 条")
    assert cache.stats["entries"] == 0

    # ── 写入新版本并验证 FIFO 驱逐 ───────────────────────────────────────
    for i in range(12):   # 超过 capacity=10
        cache.set(_rand_unit_vec(), f"结果{i}", current_version=2)
    print(f"\nFIFO 驱逐后: entries={cache.stats['entries']}（应 ≤ 10）")
    assert cache.stats["entries"] <= 10

    print(f"\n最终统计: {cache.stats}")
    print("\n  SemanticCache 验证通过 ✓")
    print("=" * 55)
