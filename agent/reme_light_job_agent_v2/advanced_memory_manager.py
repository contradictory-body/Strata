"""
advanced_memory_manager.py
==========================
高级动态长期记忆模块 (AD-LTM) — v2（写路径预计算升级版）
Advanced Dynamic Long-Term Memory Module for Agent Systems

架构概览：
  ┌─────────────────────────────────────────────────────────────┐
  │  子系统 1 — 状态与缓存管理                                    │
  │    GLOBAL_MEMORY_VERSION  +  LRU 精确缓存 + 语义缓存          │
  ├─────────────────────────────────────────────────────────────┤
  │  子系统 2 — 记忆更新链路 (Write Pipeline) ← 本版升级          │
  │    .md 文件 → MarkdownHeaderSplitter → RecursiveCharSplitter │
  │    → 槽位提取（规则层 or LLM）→ 意图标准句生成               │
  │    → [原文向量 + 意图向量] 双写 ChromaDB                     │
  │    → BM25 全量重建（仅索引原文向量记录）                      │
  ├─────────────────────────────────────────────────────────────┤
  │  子系统 3 — 混合检索精排链路 (Read Pipeline)                  │
  │    原文向量 Top-15 + BM25 Top-15 → RRF 融合 → Top-8         │
  │    → BGE-Reranker 精排 → Top-4                              │
  │    （意图向量路由留待读路径改造轮次接入）                      │
  ├─────────────────────────────────────────────────────────────┤
  │  子系统 4 — 时序冲突消解 (Temporal Resolution)                │
  │    强制按 metadata['date'] 降序，新知优先                     │
  └─────────────────────────────────────────────────────────────┘

ChromaDB 记录结构（每个 Chunk 写入两条）：
  原文记录  id=chunk_id              vec_type=original  document=原文
  意图记录  id=chunk_id+"_intent"    vec_type=intent    document=标准句

依赖安装：
    pip install chromadb rank-bm25 langchain-text-splitters \\
                sentence-transformers cachetools

可选（中文分词，提升 BM25 召回质量）：
    pip install jieba

模型（首次使用自动下载到 HuggingFace 缓存）：
    Embedding : BAAI/bge-small-zh-v1.5  (~90MB, 512 维, 中英双语)
    Reranker  : BAAI/bge-reranker-base  (~280MB, 中英双语 CrossEncoder)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ── 第三方依赖 ──────────────────────────────────────────────────────────────
import chromadb
from chromadb.config import Settings
from cachetools import LRUCache
from rank_bm25 import BM25Okapi
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── 本项目依赖 ──────────────────────────────────────────────────────────────
from intent_query_builder import (
    extract_slots,
    slots_to_sentence,
    build_intent_sentence_from_chunk,
)
from semantic_cache import SemanticCache

# ──────────────────────────────────────────────────────────────────────────────
# Logging 配置
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AD-LTM")

# ──────────────────────────────────────────────────────────────────────────────
# 模块级常量
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"
DEFAULT_RERANKER_MODEL:  str = "BAAI/bge-reranker-base"
DEFAULT_COLLECTION_NAME: str = "ad_ltm_memory"

# LRU 缓存容量
CACHE_CAPACITY: int = 100

# 检索超参数
VECTOR_TOPK:  int = 15
BM25_TOPK:    int = 15
RRF_K:        int = 60
RRF_TOPK:     int = 8
RERANK_TOPK:  int = 4

# 文本切分参数
CHUNK_SIZE:    int = 500
CHUNK_OVERLAP: int = 50

# ChromaDB vec_type 标记
VEC_TYPE_ORIGINAL: str = "original"
VEC_TYPE_INTENT:   str = "intent"
INTENT_ID_SUFFIX:  str = "_intent"

# 日期正则（用于从文件名提取 YYYY-MM-DD）
_DATE_RE: re.Pattern[str] = re.compile(r"\d{4}-\d{2}-\d{2}")


# ──────────────────────────────────────────────────────────────────────────────
# 分词器（jieba 可选，退化为正则分词）
# ──────────────────────────────────────────────────────────────────────────────
def _build_tokenizer():
    try:
        import jieba
        jieba.setLogLevel(logging.WARNING)

        def _tokenize(text: str) -> list[str]:
            return [t for t in jieba.cut(text) if t.strip()]

        logger.info("分词器：jieba（中文词粒度模式）")
        return _tokenize

    except ImportError:
        def _tokenize(text: str) -> list[str]:
            return re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]|[a-zA-Z0-9_]+", text.lower())

        logger.warning(
            "jieba 未安装，退化为正则分词（pip install jieba 可提升中文 BM25 质量）"
        )
        return _tokenize


tokenize = _build_tokenizer()


# ──────────────────────────────────────────────────────────────────────────────
# BGE Embedding 模型封装
# ──────────────────────────────────────────────────────────────────────────────
class BGEEmbeddingFunction:
    """
    将 SentenceTransformer（BGE 系列）包装为统一 Embedding 接口。

    始终启用 normalize_embeddings=True，输出单位向量，与 ChromaDB
    cosine 距离空间保持一致（cosine_sim = dot_product）。
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        logger.info(f"正在加载 Embedding 模型: {model_name} ...")
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension: int = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding 模型加载完成: {model_name}, 维度={self.dimension}")

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码，返回 List[List[float]]，供 ChromaDB add() 使用。"""
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return vecs.tolist()

    def encode_single(self, text: str) -> list[float]:
        """单条编码，返回 List[float]，供 ChromaDB query() 使用。"""
        vec = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# 核心类：AdvancedMemoryManager
# ──────────────────────────────────────────────────────────────────────────────
class AdvancedMemoryManager:
    """
    高级动态长期记忆管理器 (AD-LTM) v2
    ======================================

    v2 写路径新增：
      - 每个 Chunk 写入两条 ChromaDB 记录：原文向量（vec_type=original）
        和意图向量（vec_type=intent）。
      - 意图向量由槽位拼成的标准句生成，写入时通过规则层提取槽位；
        若传入 llm_client，则异步调用 LLM 获得更高精度槽位。
      - BM25 索引仅覆盖原文向量记录，不索引意图向量记录。
      - 槽位字段（position/city/salary/intent/tech）写入 metadata，
        支持读路径的 ChromaDB where 精准过滤。

    Usage（同步写入，规则层槽位）：
        manager = AdvancedMemoryManager(persist_dir=".ad_ltm_db")
        manager.update_memory_from_file("memory/2024-03-15.md")

    Usage（异步写入，LLM 槽位，精度更高）：
        manager = AdvancedMemoryManager(
            persist_dir=".ad_ltm_db",
            llm_client=AsyncOpenAI(...),
            llm_model="gpt-4o-mini",
        )
        await manager.async_update_memory_from_file("memory/2024-03-15.md")

    Attributes:
        GLOBAL_MEMORY_VERSION (int): 全局版本号，每次写入自动递增。
    """

    def __init__(
        self,
        persist_dir:              str | Path = ".ad_ltm_db",
        collection_name:          str        = DEFAULT_COLLECTION_NAME,
        embedding_model:          str        = DEFAULT_EMBEDDING_MODEL,
        reranker_model:           str        = DEFAULT_RERANKER_MODEL,
        cache_capacity:           int        = CACHE_CAPACITY,
        llm_client=None,
        llm_model:                str        = "",
        semantic_cache_capacity:  int        = 200,
        semantic_cache_threshold: float      = 0.95,
    ) -> None:
        """
        初始化 AD-LTM 管理器。

        Args:
            persist_dir:     ChromaDB 持久化目录（自动创建）。
            collection_name: ChromaDB 集合名。
            embedding_model: HuggingFace Embedding 模型名称或本地路径。
            reranker_model:  BGE Reranker 模型名称或本地路径。
            cache_capacity:  LRU 缓存最大条数，默认 100。
            llm_client:      openai.AsyncOpenAI 实例（可为 None）。
                             提供后，写入时使用 LLM 抽取槽位，精度更高。
                             不提供时退化为规则层槽位提取。
            llm_model:       LLM 模型名称，配合 llm_client 使用。
        """
        self._persist_dir     = Path(persist_dir)
        self._collection_name = collection_name
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        # ── 子系统 1: 全局版本 + LRU 精确缓存 + 语义缓存 ───────────────
        self.GLOBAL_MEMORY_VERSION: int       = 1
        self._lru_cache: LRUCache[str, str]   = LRUCache(maxsize=cache_capacity)
        self._semantic_cache = SemanticCache(
            capacity=semantic_cache_capacity,
            threshold=semantic_cache_threshold,
        )

        # ── LLM 客户端（可选，用于写入时高精度槽位抽取）────────────────────
        self._llm_client = llm_client
        self._llm_model  = llm_model

        # ── 模型加载 ──────────────────────────────────────────────────────
        self._embedding_fn = BGEEmbeddingFunction(embedding_model)

        logger.info(f"正在加载 Reranker 模型: {reranker_model} ...")
        self._reranker = CrossEncoder(reranker_model, max_length=512)
        logger.info(f"Reranker 模型加载完成: {reranker_model}")

        # ── ChromaDB 初始化 ──────────────────────────────────────────────
        self._chroma_client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB 初始化完成: collection='{collection_name}', "
            f"当前文档数={self._collection.count()}"
        )

        # ── BM25 内存索引 ────────────────────────────────────────────────
        self._bm25_index:    Optional[BM25Okapi] = None
        self._bm25_doc_ids:  list[str]           = []
        self._bm25_corpus:   list[list[str]]     = []

        self._rebuild_bm25()

        # ── LangChain 文本切分器 ─────────────────────────────────────────
        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#",   "header_1"),
                ("##",  "header_2"),
                ("###", "header_3"),
            ],
            strip_headers=False,
        )
        self._char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        )

        logger.info(
            "AD-LTM v2 初始化完成 ✓  "
            f"[Embedding={embedding_model}, Reranker={reranker_model}, "
            f"LRU={cache_capacity}, SemanticCache={semantic_cache_capacity}@{semantic_cache_threshold}, "
            f"LLM={'已配置' if llm_client else '未配置（规则层槽位）'}]"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 子系统 1 — 状态与缓存管理
    # ══════════════════════════════════════════════════════════════════════════

    def _make_cache_key(self, query_text: str) -> str:
        query_hash = hashlib.md5(query_text.encode("utf-8")).hexdigest()
        return f"v{self.GLOBAL_MEMORY_VERSION}_{query_hash}"

    def _cache_get(self, key: str) -> Optional[str]:
        result = self._lru_cache.get(key)
        if result is not None:
            logger.debug(f"[Cache HIT]  key={key[:40]}")
        else:
            logger.debug(f"[Cache MISS] key={key[:40]}")
        return result

    def _cache_set(self, key: str, value: str) -> None:
        self._lru_cache[key] = value
        logger.debug(f"[Cache SET]  key={key[:40]}, len={len(value)}")

    @property
    def cache_info(self) -> dict[str, Any]:
        return {
            "global_version": self.GLOBAL_MEMORY_VERSION,
            "cache_size":     len(self._lru_cache),
            "cache_maxsize":  self._lru_cache.maxsize,
            "bm25_docs":      len(self._bm25_doc_ids),
            "chroma_count":   self._collection.count(),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # 子系统 2 — 记忆更新链路 (Write Pipeline) ← v2 核心改动
    # ══════════════════════════════════════════════════════════════════════════

    def update_memory_from_file(self, file_path: str | Path) -> dict[str, Any]:
        """
        同步写入入口（规则层槽位，不需要 llm_client）。

        改造点（相比 v1）：
          - 每个 Chunk 额外生成意图标准句并 Embed，以 chunk_id+"_intent"
            写入 ChromaDB，metadata 含 vec_type=intent 和槽位字段。
          - 原文记录 metadata 新增 vec_type=original。
          - BM25 重建时仅索引 vec_type=original 记录。

        Args:
            file_path: Markdown 文件路径，文件名须含 YYYY-MM-DD。

        Returns:
            {
                "source":         "2024-03-15.md",
                "date":           "2024-03-15",
                "chunk_count":    7,      # 原文 Chunk 数量
                "intent_count":   7,      # 意图向量数量（与原文一一对应）
                "version":        3,
            }
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"记忆文件不存在: {file_path}")

        source   = file_path.name
        date_str = self._extract_date_from_filename(source)
        logger.info(f"[Write] 开始更新: source={source}, date={date_str}")

        # ── Step 1: 版本递增，使 LRU 和语义缓存旧条目全部失效 ──────────
        self.GLOBAL_MEMORY_VERSION += 1
        logger.info(f"[Write] GLOBAL_MEMORY_VERSION → {self.GLOBAL_MEMORY_VERSION}")
        self._semantic_cache.invalidate_version(self.GLOBAL_MEMORY_VERSION)

        # ── Step 2: Markdown 切分 ─────────────────────────────────────────
        content = file_path.read_text(encoding="utf-8")
        chunks  = self._split_markdown(content, source, date_str)

        if not chunks:
            logger.warning(f"[Write] {source} 切分结果为空，跳过写入。")
            return {
                "source": source, "date": date_str,
                "chunk_count": 0, "intent_count": 0,
                "version": self.GLOBAL_MEMORY_VERSION,
            }

        # ── Step 3: 为每个 Chunk 生成意图标准句（规则层，同步）────────────
        intent_sentences: list[str] = []
        slots_list:       list[dict[str, str]] = []
        for chunk in chunks:
            slots    = extract_slots(chunk["text"])
            sentence = slots_to_sentence(slots)
            intent_sentences.append(sentence)
            slots_list.append(slots)
            logger.debug(f"[Write] chunk={chunk['id'][:8]}  intent='{sentence[:60]}'")

        # ── Step 4: 清除旧记录（原文 + 意图，共用 source 字段）─────────────
        try:
            self._collection.delete(where={"source": source})
            logger.debug(f"[Write] 已清除旧记录: source={source}")
        except Exception as exc:
            logger.debug(f"[Write] 清除旧记录异常（可能首次写入）: {exc}")

        # ── Step 5a: 批量写入原文向量 ─────────────────────────────────────
        orig_ids       = [c["id"] for c in chunks]
        orig_documents = [c["text"] for c in chunks]
        orig_metadatas = [
            {
                **c["metadata"],
                "vec_type": VEC_TYPE_ORIGINAL,
                # 槽位字段写入 metadata，供读路径 where 过滤
                "slot_position": slots_list[i].get("position", ""),
                "slot_city":     slots_list[i].get("city",     ""),
                "slot_salary":   slots_list[i].get("salary",   ""),
                "slot_intent":   slots_list[i].get("intent",   ""),
                "slot_tech":     slots_list[i].get("tech",     ""),
            }
            for i, c in enumerate(chunks)
        ]
        orig_embeddings = self._embedding_fn.encode_batch(orig_documents)

        self._collection.add(
            ids=orig_ids,
            documents=orig_documents,
            embeddings=orig_embeddings,
            metadatas=orig_metadatas,
        )
        logger.info(f"[Write] 原文向量写入: {len(chunks)} 条，source={source}")

        # ── Step 5b: 批量写入意图向量 ─────────────────────────────────────
        intent_ids       = [c["id"] + INTENT_ID_SUFFIX for c in chunks]
        intent_metadatas = [
            {
                **c["metadata"],          # 保留 source/date/header_* 等字段
                "vec_type":      VEC_TYPE_INTENT,
                "original_id":   c["id"],   # 记录对应的原文 id，供读路径回溯原文
                "slot_position": slots_list[i].get("position", ""),
                "slot_city":     slots_list[i].get("city",     ""),
                "slot_salary":   slots_list[i].get("salary",   ""),
                "slot_intent":   slots_list[i].get("intent",   ""),
                "slot_tech":     slots_list[i].get("tech",     ""),
            }
            for i, c in enumerate(chunks)
        ]
        intent_embeddings = self._embedding_fn.encode_batch(intent_sentences)

        self._collection.add(
            ids=intent_ids,
            documents=intent_sentences,   # 存标准句，不存原文
            embeddings=intent_embeddings,
            metadatas=intent_metadatas,
        )
        logger.info(f"[Write] 意图向量写入: {len(chunks)} 条，source={source}")

        # ── Step 6: BM25 全量重建（仅索引原文向量记录）───────────────────
        self._rebuild_bm25()

        stats = {
            "source":       source,
            "date":         date_str,
            "chunk_count":  len(chunks),
            "intent_count": len(chunks),
            "version":      self.GLOBAL_MEMORY_VERSION,
        }
        logger.info(f"[Write] 更新完成: {stats}")
        return stats

    async def async_update_memory_from_file(
        self, file_path: str | Path
    ) -> dict[str, Any]:
        """
        异步写入入口（LLM 槽位，精度高于规则层）。

        需要在初始化时提供 llm_client 和 llm_model，否则退化为规则层。
        写入是低频操作，LLM 额外开销可接受；读路径不受影响。

        与 update_memory_from_file() 的唯一区别：
          槽位提取调用 build_intent_sentence_from_chunk()，
          优先使用 LLM 抽取（可识别"字节三面"→"面试准备"等隐含意图），
          LLM 失败时自动退化规则层，不影响写入流程。
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"记忆文件不存在: {file_path}")

        source   = file_path.name
        date_str = self._extract_date_from_filename(source)
        logger.info(f"[Write-Async] 开始更新: source={source}, date={date_str}")

        self.GLOBAL_MEMORY_VERSION += 1
        logger.info(f"[Write-Async] GLOBAL_MEMORY_VERSION → {self.GLOBAL_MEMORY_VERSION}")
        self._semantic_cache.invalidate_version(self.GLOBAL_MEMORY_VERSION)

        content = file_path.read_text(encoding="utf-8")
        chunks  = self._split_markdown(content, source, date_str)

        if not chunks:
            logger.warning(f"[Write-Async] {source} 切分结果为空，跳过写入。")
            return {
                "source": source, "date": date_str,
                "chunk_count": 0, "intent_count": 0,
                "version": self.GLOBAL_MEMORY_VERSION,
            }

        # ── 并发调用 build_intent_sentence_from_chunk，LLM 槽位提取 ──────
        tasks = [
            build_intent_sentence_from_chunk(
                chunk["text"],
                llm_client=self._llm_client,
                model=self._llm_model,
            )
            for chunk in chunks
        ]
        results = await asyncio.gather(*tasks)
        intent_sentences = [r[0] for r in results]
        slots_list       = [r[1] for r in results]

        # ── 清除旧记录 ────────────────────────────────────────────────────
        try:
            self._collection.delete(where={"source": source})
        except Exception as exc:
            logger.debug(f"[Write-Async] 清除旧记录异常: {exc}")

        # ── 写入原文向量 ──────────────────────────────────────────────────
        orig_ids       = [c["id"] for c in chunks]
        orig_documents = [c["text"] for c in chunks]
        orig_metadatas = [
            {
                **c["metadata"],
                "vec_type":      VEC_TYPE_ORIGINAL,
                "slot_position": slots_list[i].get("position", ""),
                "slot_city":     slots_list[i].get("city",     ""),
                "slot_salary":   slots_list[i].get("salary",   ""),
                "slot_intent":   slots_list[i].get("intent",   ""),
                "slot_tech":     slots_list[i].get("tech",     ""),
            }
            for i, c in enumerate(chunks)
        ]
        orig_embeddings = self._embedding_fn.encode_batch(orig_documents)
        self._collection.add(
            ids=orig_ids, documents=orig_documents,
            embeddings=orig_embeddings, metadatas=orig_metadatas,
        )
        logger.info(f"[Write-Async] 原文向量写入: {len(chunks)} 条")

        # ── 写入意图向量 ──────────────────────────────────────────────────
        intent_ids       = [c["id"] + INTENT_ID_SUFFIX for c in chunks]
        intent_metadatas = [
            {
                **c["metadata"],
                "vec_type":      VEC_TYPE_INTENT,
                "original_id":   c["id"],
                "slot_position": slots_list[i].get("position", ""),
                "slot_city":     slots_list[i].get("city",     ""),
                "slot_salary":   slots_list[i].get("salary",   ""),
                "slot_intent":   slots_list[i].get("intent",   ""),
                "slot_tech":     slots_list[i].get("tech",     ""),
            }
            for i, c in enumerate(chunks)
        ]
        intent_embeddings = self._embedding_fn.encode_batch(intent_sentences)
        self._collection.add(
            ids=intent_ids, documents=intent_sentences,
            embeddings=intent_embeddings, metadatas=intent_metadatas,
        )
        logger.info(f"[Write-Async] 意图向量写入: {len(chunks)} 条（LLM 槽位）")

        self._rebuild_bm25()

        stats = {
            "source":       source,
            "date":         date_str,
            "chunk_count":  len(chunks),
            "intent_count": len(chunks),
            "version":      self.GLOBAL_MEMORY_VERSION,
        }
        logger.info(f"[Write-Async] 更新完成: {stats}")
        return stats

    @staticmethod
    def _extract_date_from_filename(filename: str) -> str:
        match = _DATE_RE.search(filename)
        if not match:
            raise ValueError(
                f"文件名 '{filename}' 中未找到 YYYY-MM-DD 格式的日期。"
                "请使用如 '2024-03-15.md' 的命名规范。"
            )
        date_str = match.group()
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"文件名中的日期 '{date_str}' 不合法: {exc}") from exc
        return date_str

    def _split_markdown(
        self, content: str, source: str, date_str: str
    ) -> list[dict[str, Any]]:
        """
        两阶段 Markdown 切分，生成带完整 metadata 的 Chunk 列表。
        （v2 不改变切分逻辑，vec_type 由上层写入时注入）
        """
        md_docs = self._md_splitter.split_text(content)
        chunks: list[dict[str, Any]] = []

        for doc in md_docs:
            page_content: str  = doc.page_content.strip()
            header_meta:  dict = dict(doc.metadata)

            if not page_content:
                continue

            sub_texts: list[str] = (
                self._char_splitter.split_text(page_content)
                if len(page_content) > CHUNK_SIZE
                else [page_content]
            )

            for sub_text in sub_texts:
                sub_text = sub_text.strip()
                if not sub_text:
                    continue

                chunk_id = str(uuid.uuid4())
                chunks.append({
                    "id":   chunk_id,
                    "text": sub_text,
                    "metadata": {
                        "source":   source,
                        "date":     date_str,
                        **header_meta,
                        "chunk_id": chunk_id,
                    },
                })

        logger.debug(f"[Split] {source} → {len(chunks)} 个 Chunk")
        return chunks

    def _rebuild_bm25(self) -> None:
        """
        全量拉取 ChromaDB 原文向量记录，重建内存 BM25 倒排索引。

        v2 改动：增加 where={"vec_type": "original"} 过滤，
        确保 BM25 不索引意图向量记录（意图标准句不是自然语言，
        直接 BM25 分词后质量差且会干扰召回）。
        """
        total = self._collection.count()
        if total == 0:
            self._bm25_index   = None
            self._bm25_doc_ids = []
            self._bm25_corpus  = []
            logger.info("[BM25] ChromaDB 为空，BM25 索引已重置。")
            return

        # 只取原文向量记录
        result: dict = self._collection.get(
            where={"vec_type": VEC_TYPE_ORIGINAL},
            include=["documents", "metadatas"],
        )
        all_ids:  list[str] = result.get("ids", [])
        all_docs: list[str] = result.get("documents", [])

        if not all_ids:
            self._bm25_index   = None
            self._bm25_doc_ids = []
            self._bm25_corpus  = []
            logger.info("[BM25] 无原文向量记录，BM25 索引已重置。")
            return

        self._bm25_doc_ids = all_ids
        self._bm25_corpus  = [tokenize(doc) for doc in all_docs]
        self._bm25_index   = BM25Okapi(self._bm25_corpus)

        logger.info(f"[BM25] 索引重建完成: {len(all_ids)} 个原文文档")

    # ══════════════════════════════════════════════════════════════════════════
    # 子系统 3 — 混合检索精排链路 (Read Pipeline)
    # 本版保持与 v1 兼容，意图向量路由待读路径改造轮次接入
    # ══════════════════════════════════════════════════════════════════════════


    def retrieve_with_intent(
        self,
        query_text:      str,
        intent_sentence: str  = "",
        has_slots:       bool = False,
        vector_topk:     int  = VECTOR_TOPK,
        bm25_topk:       int  = BM25_TOPK,
        rrf_topk:        int  = RRF_TOPK,
        rerank_topk:     int  = RERANK_TOPK,
    ) -> str:
        """
        三路检索入口（v2 读路径核心方法）。

        路径组成：
          路1 — 原文向量检索（vec_type=original，原始 Query Embedding）
          路2 — 意图向量检索（vec_type=intent，标准句 Embedding，has_slots=True 时启动）
          路3 — BM25 关键词检索（始终启动）

        三路结果经 RRF 融合 → Reranker 精排 → 时序消解后返回。
        has_slots=False 时意图向量路不启动，自动降级为两路检索（原文+BM25）。

        缓存策略：
          has_slots=True  时以 intent_sentence 向量为 cache_key（更稳定，同义 Query 可复用）
          has_slots=False 时以 query_text 为 cache_key

        Args:
            query_text:      用户原始 Query（用于原文向量路和 BM25 路）
            intent_sentence: 槽位拼成的标准句（用于意图向量路）
            has_slots:       是否成功提取到有效槽位，决定是否启动意图向量路
            vector_topk:     原文向量路召回上限
            bm25_topk:       BM25 路召回上限
            rrf_topk:        RRF 融合后保留数
            rerank_topk:     Reranker 精排后最终输出数

        Returns:
            格式化好的 Context 字符串，可直接注入 Agent System Prompt。
        """
        if self._collection.count() == 0:
            logger.warning("[Retrieve-3路] 记忆库为空，返回空 Context。")
            return "[系统注入的历史记忆]\n（暂无相关历史记忆）"

        # ── Step 1: LRU 精确缓存查询（MD5 完全匹配，≈ 0ms）────────────────
        # 有槽位时用 intent_sentence 做 key（语义更稳定），无槽位用原文
        cache_query = intent_sentence if (has_slots and intent_sentence) else query_text
        cache_key   = self._make_cache_key(cache_query)
        cached      = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # ── Step 1.5: 预计算缓存向量（供语义缓存查询 + 向量检索复用）────
        # cache_vec：语义缓存的 key 向量（intent_sentence 或 query_text）
        # search_vec：向量检索始终用 query_text（保留原文语义细节）
        cache_vec  = self._embedding_fn.encode_single(cache_query)
        search_vec = (
            cache_vec
            if not (has_slots and intent_sentence)   # has_slots=False 时两者相同
            else self._embedding_fn.encode_single(query_text)
        )

        # ── Step 2: 语义缓存查询（余弦相似度 ≥ threshold，≈ 5ms）─────────
        sem_cached = self._semantic_cache.get(cache_vec, self.GLOBAL_MEMORY_VERSION)
        if sem_cached is not None:
            self._cache_set(cache_key, sem_cached)   # 同步预热 LRU
            return sem_cached

        logger.info(
            f"[Retrieve-3路] Query={query_text[:50]}  "
            f"has_slots={has_slots}  intent={intent_sentence[:50] if intent_sentence else ''}"
        )

        # ── Step 3: 三路并行召回 ─────────────────────────────────────────────
        vector_results = self._vector_search(query_text, top_k=vector_topk, query_vec=search_vec)

        intent_results: list[tuple[str, float]] = []
        if has_slots and intent_sentence:
            intent_results = self._intent_vector_search(intent_sentence, top_k=vector_topk)

        bm25_results = self._bm25_search(query_text, top_k=bm25_topk)

        if not vector_results and not intent_results and not bm25_results:
            logger.warning("[Retrieve-3路] 三路召回均为空，返回空 Context。")
            return "[系统注入的历史记忆]\n（暂无相关历史记忆）"

        # ── Step 3: RRF 三路融合 ─────────────────────────────────────────────
        fused_uuids = self._rrf_fusion_three(
            vector_results=vector_results,
            intent_results=intent_results,
            bm25_results=bm25_results,
            top_k=rrf_topk,
        )
        if not fused_uuids:
            return "[系统注入的历史记忆]\n（暂无相关历史记忆）"

        # ── Step 4: Reranker 精排 ────────────────────────────────────────────
        top_chunks = self._rerank(query_text=query_text, uuids=fused_uuids, top_k=rerank_topk)
        if not top_chunks:
            return "[系统注入的历史记忆]\n（暂无相关历史记忆）"

        # ── Step 5: 时序消解 + 格式化 ───────────────────────────────────────
        context = self._temporal_resolution(top_chunks)

        # ── Step 6: 写入 LRU 缓存 + 语义缓存（清晰路径，双写）────────────
        self._cache_set(cache_key, context)
        self._semantic_cache.set(cache_vec, context, self.GLOBAL_MEMORY_VERSION)
        return context

    def _intent_vector_search(
        self, intent_sentence: str, top_k: int
    ) -> list[tuple[str, float]]:
        """
        意图向量检索（查 vec_type=intent 记录）。

        命中后通过 metadata["original_id"] 映射回对应的原文 Chunk id，
        使意图路和原文路的 id 空间统一，RRF 融合时可正确合并。

        Returns:
            [(original_chunk_id, similarity), ...] 按相似度降序。
        """
        try:
            intent_ids = self._collection.get(
                where={"vec_type": VEC_TYPE_INTENT}, include=[]
            ).get("ids", [])
            n_results = min(top_k, len(intent_ids))
            if n_results == 0:
                logger.debug("[IntentVector] 意图向量记录为空，跳过。")
                return []

            intent_vec = self._embedding_fn.encode_single(intent_sentence)
            result = self._collection.query(
                query_embeddings=[intent_vec],
                n_results=n_results,
                where={"vec_type": VEC_TYPE_INTENT},
                include=["distances", "metadatas"],
            )
            ids:       list[str]   = result["ids"][0]
            distances: list[float] = result["distances"][0]
            metadatas: list[dict]  = result["metadatas"][0]

            # 映射回原文 id，使意图路和原文路在 RRF 中对齐
            scored: list[tuple[str, float]] = []
            for uid, dist, meta in zip(ids, distances, metadatas):
                original_id = meta.get("original_id") or uid.replace(INTENT_ID_SUFFIX, "")
                sim = 1.0 - dist
                scored.append((original_id, sim))

            if scored:
                logger.debug(
                    f"[IntentVector] 返回 {len(scored)} 条，Top1 sim={scored[0][1]:.4f}"
                )
            return scored

        except Exception as exc:
            logger.error(f"[IntentVector] 检索失败: {exc}", exc_info=True)
            return []

    @staticmethod
    def _rrf_fusion_three(
        vector_results: list[tuple[str, float]],
        intent_results: list[tuple[str, float]],
        bm25_results:   list[tuple[str, float]],
        top_k:          int = RRF_TOPK,
        k:              int = RRF_K,
    ) -> list[str]:
        """
        三路倒数排名融合（RRF）。

        与两路 _rrf_fusion() 逻辑相同，扩展为三路输入。
        意图路若为空（has_slots=False 时），自动退化为两路融合，无需特殊处理。

        Returns:
            融合后按 RRF 分数降序的 UUID 列表（长度 ≤ top_k）。
        """
        rrf_scores: dict[str, float] = {}

        for rank, (uid, _) in enumerate(vector_results, start=1):
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + 1.0 / (k + rank)

        for rank, (uid, _) in enumerate(intent_results, start=1):
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + 1.0 / (k + rank)

        for rank, (uid, _) in enumerate(bm25_results, start=1):
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + 1.0 / (k + rank)

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        result = [uid for uid, _ in sorted_items[:top_k]]

        logger.debug(
            f"[RRF-3路] 原文路={len(vector_results)}, 意图路={len(intent_results)}, "
            f"BM25路={len(bm25_results)}, 合并去重={len(rrf_scores)}, 输出 Top-{len(result)}"
        )
        return result

    def retrieve(
        self,
        query_text:  str,
        vector_topk: int = VECTOR_TOPK,
        bm25_topk:   int = BM25_TOPK,
        rrf_topk:    int = RRF_TOPK,
        rerank_topk: int = RERANK_TOPK,
    ) -> str:
        """
        完整检索链路入口（v2 读路径暂与 v1 兼容）。

        LRU 缓存查询
          → 原文向量检索 Top-{vector_topk} + BM25 检索 Top-{bm25_topk}
          → RRF 融合 Top-{rrf_topk}
          → BGE-Reranker 精排 Top-{rerank_topk}
          → 时序冲突消解（强制按 date 降序）
          → 写入 LRU 缓存并返回 Context 字符串

        意图向量路（三路检索）将在读路径改造轮次中通过
        retrieve_with_intent() 方法提供。
        """
        if self._collection.count() == 0:
            logger.warning("[Retrieve] 记忆库为空，返回空 Context。")
            return "[系统注入的历史记忆]\n（暂无相关历史记忆）"

        # ── Step 1: LRU 精确缓存（≈ 0ms）───────────────────────────────────
        cache_key = self._make_cache_key(query_text)
        cached    = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # ── Step 1.5: 预计算向量（供语义缓存 + 向量检索共用，只编码一次）──
        query_vec = self._embedding_fn.encode_single(query_text)

        # ── Step 2: 语义缓存（≈ 5ms）────────────────────────────────────────
        sem_cached = self._semantic_cache.get(query_vec, self.GLOBAL_MEMORY_VERSION)
        if sem_cached is not None:
            self._cache_set(cache_key, sem_cached)  # 预热 LRU
            return sem_cached

        logger.info(f"[Retrieve] Query: {query_text[:60]}...")
        vector_results = self._vector_search(query_text, top_k=vector_topk, query_vec=query_vec)
        bm25_results   = self._bm25_search(query_text,   top_k=bm25_topk)

        if not vector_results and not bm25_results:
            logger.warning("[Retrieve] 双路召回均为空，返回空 Context。")
            return "[系统注入的历史记忆]\n（暂无相关历史记忆）"

        fused_uuids = self._rrf_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            top_k=rrf_topk,
        )
        if not fused_uuids:
            return "[系统注入的历史记忆]\n（暂无相关历史记忆）"

        top_chunks = self._rerank(
            query_text=query_text,
            uuids=fused_uuids,
            top_k=rerank_topk,
        )
        if not top_chunks:
            return "[系统注入的历史记忆]\n（暂无相关历史记忆）"

        context = self._temporal_resolution(top_chunks)
        # 双写缓存：LRU 精确缓存 + 语义缓存
        self._cache_set(cache_key, context)
        self._semantic_cache.set(query_vec, context, self.GLOBAL_MEMORY_VERSION)
        return context

    def _vector_search(
        self,
        query_text: str,
        top_k:      int,
        query_vec:  list[float] | None = None,
    ) -> list[tuple[str, float]]:
        """
        原文向量语义检索。

        v2 改动：增加 where={"vec_type": "original"} 过滤，避免召回意图向量记录。
        v3 改动：新增可选 query_vec 参数，避免调用方已预计算时重复编码。

        Args:
            query_text: 查询文本（用于日志）
            top_k:      召回条数上限
            query_vec:  预计算的归一化查询向量（None 时内部编码）
        """
        try:
            if query_vec is None:
                query_vec = self._embedding_fn.encode_single(query_text)

            # 只计算原文向量记录数，避免意图记录干扰 n_results 上限
            orig_count = len(
                self._collection.get(
                    where={"vec_type": VEC_TYPE_ORIGINAL},
                    include=[],
                ).get("ids", [])
            )
            n_results = min(top_k, orig_count)
            if n_results == 0:
                return []

            result = self._collection.query(
                query_embeddings=[query_vec],
                n_results=n_results,
                where={"vec_type": VEC_TYPE_ORIGINAL},
                include=["distances"],
            )
            ids:       list[str]   = result["ids"][0]
            distances: list[float] = result["distances"][0]

            scored = [(uid, 1.0 - dist) for uid, dist in zip(ids, distances)]
            if scored:
                logger.debug(f"[Vector] 返回 {len(scored)} 条，Top1 sim={scored[0][1]:.4f}")
            return scored

        except Exception as exc:
            logger.error(f"[Vector] 检索失败: {exc}", exc_info=True)
            return []

    def _bm25_search(
        self, query_text: str, top_k: int
    ) -> list[tuple[str, float]]:
        """BM25Okapi 关键词检索（索引已在 _rebuild_bm25 中过滤为原文记录）。"""
        if self._bm25_index is None or not self._bm25_doc_ids:
            logger.debug("[BM25] 索引为空，跳过关键词检索。")
            return []

        try:
            query_tokens = tokenize(query_text)
            if not query_tokens:
                logger.debug("[BM25] Query 分词结果为空，跳过。")
                return []

            raw_scores: list[float] = self._bm25_index.get_scores(query_tokens).tolist()
            max_score = max(raw_scores) if raw_scores else 0.0
            if max_score <= 0.0:
                logger.debug("[BM25] 所有文档 BM25 分数为 0。")
                return []

            scored = [
                (uid, score / max_score)
                for uid, score in zip(self._bm25_doc_ids, raw_scores)
                if score > 0.0
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            result = scored[:top_k]
            if result:
                logger.debug(f"[BM25] 返回 {len(result)} 条，Top1 score={result[0][1]:.4f}")
            return result

        except Exception as exc:
            logger.error(f"[BM25] 检索失败: {exc}", exc_info=True)
            return []

    @staticmethod
    def _rrf_fusion(
        vector_results: list[tuple[str, float]],
        bm25_results:   list[tuple[str, float]],
        top_k:          int = RRF_TOPK,
        k:              int = RRF_K,
    ) -> list[str]:
        """倒数排名融合（RRF），合并两路召回结果。"""
        rrf_scores: dict[str, float] = {}

        for rank, (uid, _) in enumerate(vector_results, start=1):
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + 1.0 / (k + rank)

        for rank, (uid, _) in enumerate(bm25_results, start=1):
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + 1.0 / (k + rank)

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        result = [uid for uid, _ in sorted_items[:top_k]]

        logger.debug(
            f"[RRF] 向量路={len(vector_results)}, BM25路={len(bm25_results)}, "
            f"合并去重={len(rrf_scores)}, 输出 Top-{len(result)}"
        )
        return result

    def _rerank(
        self,
        query_text: str,
        uuids:      list[str],
        top_k:      int = RERANK_TOPK,
    ) -> list[dict[str, Any]]:
        """
        BGE-Reranker-Base 交叉编码精排。

        v2 兼容说明：uuids 现在只含原文向量 id（无 _intent 后缀），
        ChromaDB.get(ids=uuids) 正常工作，无需改动。
        """
        if not uuids:
            return []

        try:
            fetch_result = self._collection.get(
                ids=uuids,
                include=["documents", "metadatas"],
            )
            ids:       list[str]  = fetch_result["ids"]
            documents: list[str]  = fetch_result["documents"]
            metadatas: list[dict] = fetch_result["metadatas"]

            if not ids:
                logger.warning("[Rerank] 根据 UUID 未能从 ChromaDB 取回任何文档。")
                return []

            pairs = [[query_text, doc] for doc in documents]
            raw_scores = self._reranker.predict(pairs)
            scores: list[float] = (
                raw_scores.tolist()
                if hasattr(raw_scores, "tolist")
                else [float(s) for s in raw_scores]
            )

            chunk_list = [
                {
                    "id":           uid,
                    "text":         doc,
                    "metadata":     meta,
                    "rerank_score": score,
                }
                for uid, doc, meta, score in zip(ids, documents, metadatas, scores)
            ]
            chunk_list.sort(key=lambda x: x["rerank_score"], reverse=True)
            result = chunk_list[:top_k]

            logger.debug(
                f"[Rerank] 输入 {len(chunk_list)} → 输出 Top-{len(result)}, "
                f"分数范围=[{result[-1]['rerank_score']:.4f}, {result[0]['rerank_score']:.4f}]"
            )
            return result

        except Exception as exc:
            logger.error(f"[Rerank] 精排失败: {exc}", exc_info=True)
            return []

    # ══════════════════════════════════════════════════════════════════════════
    # 子系统 4 — 时序冲突消解机制 (Temporal Resolution)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _temporal_resolution(chunks: list[dict[str, Any]]) -> str:
        """
        强制按 metadata['date'] 降序（新→旧）重排，确保最新记忆优先。
        """
        def _parse_date(chunk: dict[str, Any]) -> datetime:
            date_str: str = chunk.get("metadata", {}).get("date", "")
            try:
                return datetime.strptime(date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                logger.warning(
                    f"[Temporal] Chunk '{chunk.get('id', '?')[:8]}' "
                    f"的日期 '{date_str}' 无法解析，排至最后。"
                )
                return datetime.min

        sorted_chunks = sorted(chunks, key=_parse_date, reverse=True)
        header = "[系统注入的历史记忆（按时间从新到旧排列）]"
        lines  = [header]

        for chunk in sorted_chunks:
            date_label:   str   = chunk.get("metadata", {}).get("date", "未知日期")
            text:         str   = chunk.get("text", "").strip()
            rerank_score: float = chunk.get("rerank_score", float("nan"))
            score_str = (
                f" | score={rerank_score:.4f}"
                if not (isinstance(rerank_score, float) and rerank_score != rerank_score)
                else ""
            )
            lines.append(f">>> [记录于 {date_label}{score_str}] {text}")

        context = "\n".join(lines)
        logger.debug(
            f"[Temporal] 消解完成: {len(sorted_chunks)} 条记忆, "
            f"context_len={len(context)}"
        )
        return context

    # ══════════════════════════════════════════════════════════════════════════
    # 工具方法 (Utilities)
    # ══════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> dict[str, Any]:
        """返回当前记忆库完整统计（含语义缓存命中率）。"""
        total = self._collection.count()
        try:
            orig_count = len(
                self._collection.get(
                    where={"vec_type": VEC_TYPE_ORIGINAL}, include=[]
                ).get("ids", [])
            )
            intent_count = total - orig_count
        except Exception:
            orig_count = intent_count = -1

        return {
            "global_version":       self.GLOBAL_MEMORY_VERSION,
            "chroma_total":         total,
            "chroma_original":      orig_count,
            "chroma_intent":        intent_count,
            "bm25_docs":            len(self._bm25_doc_ids),
            "lru_cache_size":       len(self._lru_cache),
            "lru_cache_maxsize":    self._lru_cache.maxsize,
            "semantic_cache":       self._semantic_cache.stats,
            "embedding_model":      self._embedding_fn.model_name,
            "embedding_dim":        self._embedding_fn.dimension,
            "persist_dir":          str(self._persist_dir),
            "collection_name":      self._collection_name,
            "llm_slot":             "LLM" if self._llm_client else "规则层",
        }

    def list_indexed_sources(self) -> list[str]:
        """列出当前已索引的所有 source 文件名（去重，按日期字母排序）。"""
        if self._collection.count() == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        sources = {m.get("source", "") for m in result["metadatas"] if m}
        return sorted(sources)

    def delete_memory_file(self, source_filename: str) -> dict[str, Any]:
        """
        按 source 文件名删除对应的所有记录（原文向量 + 意图向量），
        重建 BM25，并递增版本号。

        v2 说明：原文和意图记录共用 source 字段，
        where={"source": source_filename} 一次删除两类记录，无需额外处理。
        """
        count_before = self._collection.count()
        self._collection.delete(where={"source": source_filename})
        self.GLOBAL_MEMORY_VERSION += 1
        self._semantic_cache.invalidate_version(self.GLOBAL_MEMORY_VERSION)
        self._rebuild_bm25()
        count_after = self._collection.count()
        result = {
            "deleted_source": source_filename,
            "chunks_before":  count_before,
            "chunks_after":   count_after,
            "chunks_removed": count_before - count_after,
            "version":        self.GLOBAL_MEMORY_VERSION,
        }
        logger.info(f"[Delete] {result}")
        return result

    def clear_cache(self) -> None:
        """手动清空 LRU 精确缓存和语义缓存（不影响向量库和 BM25 索引）。"""
        self._lru_cache.clear()
        self._semantic_cache.clear()
        logger.info("[Cache] LRU 缓存和语义缓存已手动清空。")

    def reset_all(self) -> None:
        """危险操作：清空所有数据，版本重置为 1。仅用于测试环境。"""
        self._chroma_client.delete_collection(self._collection_name)
        self._collection = self._chroma_client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25_index   = None
        self._bm25_doc_ids = []
        self._bm25_corpus  = []
        self._lru_cache.clear()
        self._semantic_cache.clear()
        self.GLOBAL_MEMORY_VERSION = 1
        logger.warning("[Reset] 所有记忆数据（含语义缓存）已清空，版本重置为 1。")


# ──────────────────────────────────────────────────────────────────────────────
# 端到端验证（直接运行此文件）
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, textwrap

    BANNER = "=" * 62

    OLD_MEMORY = textwrap.dedent("""\
    # 求职目标

    用户的目标岗位是 **后端工程师**，目标城市是 **上海**，
    期望薪资范围为 25K-35K，对互联网大厂有强烈意向。

    ## 技术栈

    熟悉 Python、FastAPI、Redis、MySQL。
    对 Kafka 消息队列有初步了解，面试薄弱点在系统设计方向。

    ## 面试状态

    目前正在准备美团和字节跳动的面试，已完成两轮电话面试。
    """)

    NEW_MEMORY = textwrap.dedent("""\
    # 求职目标（最新更新）

    用户已将目标城市从上海更改为 **北京**，期望薪资上调至 **35K-50K**。
    新增目标公司：蚂蚁集团、PingCAP。

    ## 新增技术方向

    开始系统学习 **Golang** 和 **Kubernetes**，计划 3 个月达到中级水平。
    面试薄弱点已转变为 **分布式事务与一致性协议**（Raft、2PC）。

    ## 面试进展

    字节跳动三面已通过，等待 HR 面。美团决定放弃，转投蚂蚁集团。
    """)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir   = Path(tmpdir)
        old_file = tmpdir / "2023-05-01.md"
        new_file = tmpdir / "2024-03-15.md"
        old_file.write_text(OLD_MEMORY, encoding="utf-8")
        new_file.write_text(NEW_MEMORY, encoding="utf-8")

        print(f"\n{BANNER}")
        print("  AD-LTM v2 写路径验证")
        print(BANNER)

        db_dir  = tmpdir / "ad_ltm_db"
        manager = AdvancedMemoryManager(persist_dir=str(db_dir))

        print("\n[1/5] 写入旧记忆 (2023-05-01.md) ...")
        r = manager.update_memory_from_file(old_file)
        print(f"      {r}")
        print(f"      ChromaDB 总记录: {manager._collection.count()} "
              f"（原文×{r['chunk_count']} + 意图×{r['intent_count']}）")

        print(f"\n{BANNER}")
        print("[2/5] 写入新记忆 (2024-03-15.md) ...")
        r = manager.update_memory_from_file(new_file)
        print(f"      {r}")
        print(f"      已索引文件: {manager.list_indexed_sources()}")
        print(f"      统计: {manager.get_stats()}")

        print(f"\n{BANNER}")
        print("[3/5] 验证意图向量已写入 ChromaDB ...")
        intent_sample = manager._collection.get(
            where={"vec_type": "intent"}, limit=2, include=["documents", "metadatas"]
        )
        for doc, meta in zip(intent_sample["documents"], intent_sample["metadatas"]):
            print(f"      标准句: {doc}")
            print(f"      slots : position={meta.get('slot_position')} "
                  f"city={meta.get('slot_city')} salary={meta.get('slot_salary')}")

        print(f"\n{BANNER}")
        print("[4/5] 检索验证（读路径兼容性）...")
        query   = "用户现在的目标城市和期望薪资是多少？"
        context = manager.retrieve(query)
        print(context)

        print(f"\n{BANNER}")
        print("[5/5] 删除文件后验证两类记录同步清除 ...")
        before = manager._collection.count()
        manager.delete_memory_file("2023-05-01.md")
        after  = manager._collection.count()
        print(f"      删除前: {before} 条，删除后: {after} 条，"
              f"移除: {before - after} 条（原文+意图）")

        print(f"\n{BANNER}")
        print("  AD-LTM v2 写路径验证完成 ✓")
        print(BANNER + "\n")
