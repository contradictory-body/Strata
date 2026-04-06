"""
advanced_memory_manager.py
==========================
高级动态长期记忆模块 (AD-LTM)
Advanced Dynamic Long-Term Memory Module for Agent Systems

架构概览：
  ┌─────────────────────────────────────────────────────────────┐
  │  子系统 1 — 状态与缓存管理                                    │
  │    GLOBAL_MEMORY_VERSION  +  LRU 语义缓存 (100 条)            │
  ├─────────────────────────────────────────────────────────────┤
  │  子系统 2 — 记忆更新链路 (Write Pipeline)                     │
  │    .md 文件 → MarkdownHeaderSplitter → RecursiveCharSplitter │
  │    → Embed → ChromaDB 局部替换 → BM25 全量重建               │
  ├─────────────────────────────────────────────────────────────┤
  │  子系统 3 — 混合检索精排链路 (Read Pipeline)                   │
  │    向量 Top-15 + BM25 Top-15 → RRF 融合 → Top-8              │
  │    → BGE-Reranker 精排 → Top-4                               │
  ├─────────────────────────────────────────────────────────────┤
  │  子系统 4 — 时序冲突消解 (Temporal Resolution)                 │
  │    强制按 metadata['date'] 降序，新知优先                      │
  └─────────────────────────────────────────────────────────────┘

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
VECTOR_TOPK:  int = 15      # 向量路过采样数
BM25_TOPK:    int = 15      # BM25 路过采样数
RRF_K:        int = 60      # RRF 平滑常数（经典取值 60）
RRF_TOPK:     int = 8       # RRF 融合后截取数
RERANK_TOPK:  int = 4       # Reranker 精排后最终输出数

# 文本切分参数
CHUNK_SIZE:    int = 500
CHUNK_OVERLAP: int = 50

# 日期正则（用于从文件名提取 YYYY-MM-DD）
_DATE_RE: re.Pattern[str] = re.compile(r"\d{4}-\d{2}-\d{2}")


# ──────────────────────────────────────────────────────────────────────────────
# 分词器（jieba 可选，退化为正则分词）
# ──────────────────────────────────────────────────────────────────────────────
def _build_tokenizer():
    """
    构建分词函数，优先使用 jieba（中文词粒度），
    若未安装则退化为正则字符粒度（CJK 字符 + ASCII 词汇）。
    """
    try:
        import jieba
        jieba.setLogLevel(logging.WARNING)

        def _tokenize(text: str) -> list[str]:
            return [t for t in jieba.cut(text) if t.strip()]

        logger.info("分词器：jieba（中文词粒度模式）")
        return _tokenize

    except ImportError:
        def _tokenize(text: str) -> list[str]:
            # CJK 字符逐字切分 + 英文/数字按词切分
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

    · 始终启用 normalize_embeddings=True，输出单位向量，与 ChromaDB
      cosine 距离空间保持一致（cosine_sim = dot_product）。
    · 对外暴露 encode_batch / encode_single 两个接口，供 Write / Read
      链路分别调用。
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        logger.info(f"正在加载 Embedding 模型: {model_name} ...")
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension: int = self._model.get_sentence_embedding_dimension()
        logger.info(
            f"Embedding 模型加载完成: {model_name}, 维度={self.dimension}"
        )

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
    高级动态长期记忆管理器 (AD-LTM)
    ===================================

    Usage:
        manager = AdvancedMemoryManager(persist_dir=".ad_ltm_db")

        # 写入 / 更新记忆
        manager.update_memory_from_file("memory/2024-03-15.md")

        # 检索（自动走双轨召回 → RRF → Rerank → 时序消解）
        context = manager.retrieve("用户的目标城市和薪资期望？")
        print(context)

    Attributes:
        GLOBAL_MEMORY_VERSION (int): 全局版本号，每次写入自动递增。
    """

    def __init__(
        self,
        persist_dir:      str | Path = ".ad_ltm_db",
        collection_name:  str        = DEFAULT_COLLECTION_NAME,
        embedding_model:  str        = DEFAULT_EMBEDDING_MODEL,
        reranker_model:   str        = DEFAULT_RERANKER_MODEL,
        cache_capacity:   int        = CACHE_CAPACITY,
    ) -> None:
        """
        初始化 AD-LTM 管理器，加载所有子系统。

        Args:
            persist_dir:      ChromaDB 持久化目录（自动创建）。
            collection_name:  ChromaDB 集合名。
            embedding_model:  HuggingFace Embedding 模型名称或本地路径。
            reranker_model:   BGE Reranker 模型名称或本地路径。
            cache_capacity:   LRU 缓存最大条数，默认 100。
        """
        self._persist_dir     = Path(persist_dir)
        self._collection_name = collection_name
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        # ── 子系统 1: 全局版本 + LRU 语义缓存 ─────────────────────────────
        self.GLOBAL_MEMORY_VERSION: int                = 1
        self._lru_cache: LRUCache[str, str]            = LRUCache(maxsize=cache_capacity)

        # ── 模型加载 ────────────────────────────────────────────────────────
        self._embedding_fn = BGEEmbeddingFunction(embedding_model)

        logger.info(f"正在加载 Reranker 模型: {reranker_model} ...")
        self._reranker = CrossEncoder(reranker_model, max_length=512)
        logger.info(f"Reranker 模型加载完成: {reranker_model}")

        # ── ChromaDB 初始化（不注册 embedding_function，始终手动传入向量）──
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

        # ── BM25 内存索引（与 ChromaDB 保持同步）─────────────────────────
        self._bm25_index:    Optional[BM25Okapi]  = None
        self._bm25_doc_ids:  list[str]            = []   # UUID 列表（与语料对齐）
        self._bm25_corpus:   list[list[str]]      = []   # tokenized 语料

        # 启动时从 ChromaDB 恢复 BM25，支持进程重启后无缝继续
        self._rebuild_bm25()

        # ── LangChain 文本切分器 ───────────────────────────────────────────
        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#",  "header_1"),
                ("##", "header_2"),
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
            "AD-LTM 初始化完成 ✓  "
            f"[Embedding={embedding_model}, Reranker={reranker_model}, "
            f"Cache={cache_capacity}]"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 子系统 1 — 状态与缓存管理
    # ══════════════════════════════════════════════════════════════════════════

    def _make_cache_key(self, query_text: str) -> str:
        """
        生成版本化 LRU Key：``f"v{VERSION}_{md5(query)}"``.

        只要 GLOBAL_MEMORY_VERSION 递增（即任意文件写入），
        所有旧 Key 因前缀不同而自然失效，无需手动 invalidate。
        """
        query_hash = hashlib.md5(query_text.encode("utf-8")).hexdigest()
        return f"v{self.GLOBAL_MEMORY_VERSION}_{query_hash}"

    def _cache_get(self, key: str) -> Optional[str]:
        """命中 LRU 缓存返回字符串，未命中返回 None。"""
        result = self._lru_cache.get(key)
        if result is not None:
            logger.debug(f"[Cache HIT]  key={key[:40]}")
        else:
            logger.debug(f"[Cache MISS] key={key[:40]}")
        return result

    def _cache_set(self, key: str, value: str) -> None:
        """将检索结果写入 LRU 缓存。"""
        self._lru_cache[key] = value
        logger.debug(f"[Cache SET]  key={key[:40]}, len={len(value)}")

    @property
    def cache_info(self) -> dict[str, Any]:
        """返回缓存与版本状态，方便监控。"""
        return {
            "global_version": self.GLOBAL_MEMORY_VERSION,
            "cache_size":     len(self._lru_cache),
            "cache_maxsize":  self._lru_cache.maxsize,
            "bm25_docs":      len(self._bm25_doc_ids),
            "chroma_count":   self._collection.count(),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # 子系统 2 — 记忆更新链路 (Write Pipeline)
    # ══════════════════════════════════════════════════════════════════════════

    def update_memory_from_file(self, file_path: str | Path) -> dict[str, Any]:
        """
        记忆更新入口：读取 Markdown 文件 → 切分 → 嵌入 → ChromaDB 局部替换
        → BM25 全量重建 → 全局版本递增（使旧缓存全部失效）。

        文件命名规范：文件名须包含 ``YYYY-MM-DD``，例如 ``2024-03-15.md``。
        日期会被提取进每个 Chunk 的 metadata，供时序消解使用。

        Args:
            file_path: Markdown 文件的路径（str 或 Path）。

        Returns:
            包含更新统计信息的字典::

                {
                    "source":      "2024-03-15.md",
                    "date":        "2024-03-15",
                    "chunk_count": 7,
                    "version":     3,
                }

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError:        文件名不含合法 YYYY-MM-DD 日期。
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"记忆文件不存在: {file_path}")

        source   = file_path.name
        date_str = self._extract_date_from_filename(source)
        logger.info(f"[Write] 开始更新: source={source}, date={date_str}")

        # ── Step 1: 版本递增（使所有旧版本 LRU Key 失效）─────────────────
        self.GLOBAL_MEMORY_VERSION += 1
        logger.info(f"[Write] GLOBAL_MEMORY_VERSION → {self.GLOBAL_MEMORY_VERSION}")

        # ── Step 2: 文本切分 ────────────────────────────────────────────────
        content = file_path.read_text(encoding="utf-8")
        chunks  = self._split_markdown(content, source, date_str)

        if not chunks:
            logger.warning(f"[Write] 文件 {source} 切分结果为空，跳过 ChromaDB 写入。")
            return {
                "source":      source,
                "date":        date_str,
                "chunk_count": 0,
                "version":     self.GLOBAL_MEMORY_VERSION,
            }

        # ── Step 3: ChromaDB 局部替换（先删旧，再写新）─────────────────────
        #    定向删除该 source 的全部旧 Chunk，避免与新版本混用
        try:
            self._collection.delete(where={"source": source})
            logger.debug(f"[Write] 已清除旧向量: source={source}")
        except Exception as exc:
            # 首次写入时集合为空，部分 ChromaDB 版本会抛异常，忽略即可
            logger.debug(f"[Write] 删除旧向量时异常（可能首次写入）: {exc}")

        ids       = [c["id"]       for c in chunks]
        documents = [c["text"]     for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # 批量 Embed（利用 batch_size=32 减少推理次数）
        embeddings = self._embedding_fn.encode_batch(documents)

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(f"[Write] ChromaDB 写入 {len(chunks)} 个 Chunk，source={source}")

        # ── Step 4: BM25 全量重建（保持关键词索引与向量库一致）──────────────
        self._rebuild_bm25()

        stats = {
            "source":      source,
            "date":        date_str,
            "chunk_count": len(chunks),
            "version":     self.GLOBAL_MEMORY_VERSION,
        }
        logger.info(f"[Write] 更新完成: {stats}")
        return stats

    @staticmethod
    def _extract_date_from_filename(filename: str) -> str:
        """
        从文件名提取 YYYY-MM-DD 日期字符串。

        Examples:
            "2024-03-15.md"          → "2024-03-15"
            "memory_2023-10-27.md"   → "2023-10-27"

        Raises:
            ValueError: 文件名中不含合法日期。
        """
        match = _DATE_RE.search(filename)
        if not match:
            raise ValueError(
                f"文件名 '{filename}' 中未找到 YYYY-MM-DD 格式的日期。"
                "请使用如 '2024-03-15.md' 的命名规范。"
            )
        date_str = match.group()
        # 验证日期本身合法（防止 2024-13-40 这类）
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

        阶段 1: MarkdownHeaderTextSplitter
            按 #/##/### 分块，提取标题作为 ``header_1/2/3``。
        阶段 2: RecursiveCharacterTextSplitter
            对超过 CHUNK_SIZE 的块二次切分，保留 CHUNK_OVERLAP 重叠。

        每个 Chunk 的 metadata 结构::

            {
                "source":   "2024-03-15.md",
                "date":     "2024-03-15",
                "header_1": "求职目标",        # 若存在
                "header_2": "技术栈",           # 若存在
                "chunk_id": "<uuid>",
            }

        Returns:
            Chunk 字典列表，每条含 ``id``、``text``、``metadata``。
        """
        md_docs = self._md_splitter.split_text(content)
        chunks: list[dict[str, Any]] = []

        for doc in md_docs:
            page_content: str  = doc.page_content.strip()
            header_meta:  dict = dict(doc.metadata)     # header_1, header_2, ...

            if not page_content:
                continue

            # 阶段 2：超长块递归切分
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
                        **header_meta,        # header_1, header_2, header_3
                        "chunk_id": chunk_id,
                    },
                })

        logger.debug(f"[Split] {source} → {len(chunks)} 个 Chunk")
        return chunks

    def _rebuild_bm25(self) -> None:
        """
        全量拉取 ChromaDB 文档，重建内存 BM25 倒排索引。

        调用时机：每次 ``update_memory_from_file`` 完成后触发，
        确保 BM25 关键词索引与向量库始终保持一致。

        内部维护三个对齐数组:
            _bm25_doc_ids : UUID 列表
            _bm25_corpus  : 对应的 tokenized 文档列表
            _bm25_index   : BM25Okapi 实例
        """
        total = self._collection.count()
        if total == 0:
            self._bm25_index   = None
            self._bm25_doc_ids = []
            self._bm25_corpus  = []
            logger.info("[BM25] ChromaDB 为空，BM25 索引已重置。")
            return

        result: dict = self._collection.get(include=["documents", "metadatas"])
        all_ids:  list[str] = result["ids"]
        all_docs: list[str] = result["documents"]

        self._bm25_doc_ids = all_ids
        self._bm25_corpus  = [tokenize(doc) for doc in all_docs]
        self._bm25_index   = BM25Okapi(self._bm25_corpus)

        logger.info(f"[BM25] 索引重建完成: {len(all_ids)} 个文档")

    # ══════════════════════════════════════════════════════════════════════════
    # 子系统 3 — 混合检索与精排链路 (Read Pipeline)
    # ══════════════════════════════════════════════════════════════════════════

    def retrieve(
        self,
        query_text:  str,
        vector_topk: int = VECTOR_TOPK,
        bm25_topk:   int = BM25_TOPK,
        rrf_topk:    int = RRF_TOPK,
        rerank_topk: int = RERANK_TOPK,
    ) -> str:
        """
        完整检索链路入口：

        LRU 缓存查询
          → 向量检索 Top-{vector_topk} + BM25 检索 Top-{bm25_topk}
          → RRF 融合 Top-{rrf_topk}
          → BGE-Reranker 精排 Top-{rerank_topk}
          → 时序冲突消解（强制按 date 降序）
          → 写入 LRU 缓存并返回 Context 字符串

        Args:
            query_text:  用户查询文本。
            vector_topk: 向量检索召回上限，默认 15。
            bm25_topk:   BM25 检索召回上限，默认 15。
            rrf_topk:    RRF 融合后保留数，默认 8。
            rerank_topk: 最终精排输出数，默认 4。

        Returns:
            格式化好的 Context 字符串，可直接注入 Agent System Prompt。
        """
        if self._collection.count() == 0:
            logger.warning("[Retrieve] 记忆库为空，返回空 Context。")
            return "[系统注入的历史记忆]\n（暂无相关历史记忆）"

        # ── Step 1: 查 LRU 缓存 ─────────────────────────────────────────────
        cache_key = self._make_cache_key(query_text)
        cached    = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # ── Step 2: 双路过采样召回 ───────────────────────────────────────────
        logger.info(f"[Retrieve] Query: {query_text[:60]}...")
        vector_results = self._vector_search(query_text, top_k=vector_topk)
        bm25_results   = self._bm25_search(query_text,   top_k=bm25_topk)

        if not vector_results and not bm25_results:
            logger.warning("[Retrieve] 双路召回均为空，返回空 Context。")
            return "[系统注入的历史记忆]\n（暂无相关历史记忆）"

        # ── Step 3: RRF 融合 → Top-rrf_topk ─────────────────────────────────
        fused_uuids = self._rrf_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            top_k=rrf_topk,
        )
        if not fused_uuids:
            return "[系统注入的历史记忆]\n（暂无相关历史记忆）"

        # ── Step 4: BGE-Reranker 精排 → Top-rerank_topk ──────────────────────
        top_chunks = self._rerank(
            query_text=query_text,
            uuids=fused_uuids,
            top_k=rerank_topk,
        )
        if not top_chunks:
            return "[系统注入的历史记忆]\n（暂无相关历史记忆）"

        # ── Step 5: 时序冲突消解 + 格式化输出 ─────────────────────────────────
        context = self._temporal_resolution(top_chunks)

        # ── Step 6: 写入 LRU 缓存 ───────────────────────────────────────────
        self._cache_set(cache_key, context)
        return context

    def _vector_search(
        self, query_text: str, top_k: int
    ) -> list[tuple[str, float]]:
        """
        向量语义检索（ChromaDB cosine 空间）。

        ChromaDB 返回的是 cosine distance（值域 [0, 2]），
        转换为 cosine similarity：``sim = 1.0 - distance``。

        Returns:
            [(uuid, similarity), ...] 按相似度降序。
        """
        try:
            query_vec = self._embedding_fn.encode_single(query_text)
            n_results = min(top_k, self._collection.count())
            if n_results == 0:
                return []

            result = self._collection.query(
                query_embeddings=[query_vec],
                n_results=n_results,
                include=["distances"],
            )
            ids:       list[str]   = result["ids"][0]
            distances: list[float] = result["distances"][0]

            # distance → similarity（cosine distance = 1 - cosine_similarity）
            scored = [(uid, 1.0 - dist) for uid, dist in zip(ids, distances)]
            logger.debug(f"[Vector] 返回 {len(scored)} 条，Top1 sim={scored[0][1]:.4f}")
            return scored

        except Exception as exc:
            logger.error(f"[Vector] 检索失败: {exc}", exc_info=True)
            return []

    def _bm25_search(
        self, query_text: str, top_k: int
    ) -> list[tuple[str, float]]:
        """
        BM25Okapi 关键词检索。

        分数归一化到 [0, 1]（除以最大分数），使得 BM25 分数可与
        向量相似度在 RRF 中处于相同量级（RRF 本身不依赖分数，但
        归一化分数可用于调试排名合理性）。

        Returns:
            [(uuid, normalized_bm25_score), ...] 按分数降序，去除 score=0 的文档。
        """
        if self._bm25_index is None or not self._bm25_doc_ids:
            logger.debug("[BM25] 索引为空，跳过关键词检索。")
            return []

        try:
            query_tokens = tokenize(query_text)
            if not query_tokens:
                logger.debug("[BM25] Query 分词结果为空，跳过。")
                return []

            raw_scores: list[float] = self._bm25_index.get_scores(query_tokens).tolist()

            # 归一化
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
        """
        倒数排名融合（Reciprocal Rank Fusion, RRF）。

        融合公式（两路通过 UUID 对齐，仅出现在一路的文档也参与计算）::

            RRF_Score(d) = Σ  1 / (k + rank_i(d))
                           i

        经典取 k=60（Cormack et al., 2009），对排名靠前的文档给予
        较大权重，同时平滑尾部文档的影响。

        Returns:
            融合后按 RRF 分数降序排列的 UUID 列表（长度 ≤ top_k）。
        """
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

        构建 ``[Query, Document]`` Pair，批量送入 CrossEncoder 打分。
        CrossEncoder 对 Query 和 Document 联合建模，精度显著高于
        向量点积近似。

        Args:
            query_text: 原始查询文本。
            uuids:      RRF 融合后的候选 UUID 列表。
            top_k:      精排后保留条数，默认 4。

        Returns:
            包含 text、metadata、rerank_score 的 Chunk 字典列表，按 rerank_score 降序。
        """
        if not uuids:
            return []

        try:
            # 批量拉取 UUID 对应的原文和 metadata
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

            # 构建 [Query, Document] Pair 列表
            pairs = [[query_text, doc] for doc in documents]

            # CrossEncoder 批量打分（返回 numpy array 或 list）
            raw_scores = self._reranker.predict(pairs)
            if hasattr(raw_scores, "tolist"):
                scores: list[float] = raw_scores.tolist()
            else:
                scores = [float(s) for s in raw_scores]

            # 合并并降序排列
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
        时序冲突消解：**绝对不信任 Reranker 的排序**，
        强制按 ``metadata['date']`` 降序（从新到旧）重新排列。

        这是 AD-LTM 中最关键的设计：

        ·  Reranker 按语义相关度排序，可能把 2022 年的旧记录
           排在 2024 年的更新前面，导致 Agent 使用过期信息。
        ·  强制时序排序确保「最新记忆优先出现在 Context 最前端」，
           LLM 在 in-context 位置偏好下会优先引用新知识。

        输出格式（每行一条，包含日期标签）::

            [系统注入的历史记忆（按时间从新到旧排列）]
            >>> [记录于 2024-03-15 | score=3.2156] 用户目标城市更新为北京...
            >>> [记录于 2023-05-01 | score=2.8871] 用户最初目标城市为上海...

        Args:
            chunks: Reranker 精排后的 Chunk 列表（含 text, metadata, rerank_score）。

        Returns:
            格式化好的多行 Context 字符串。
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

        # ── 核心：强制按日期降序（新→旧），无论 Reranker 给出何种排名 ──────
        sorted_chunks = sorted(chunks, key=_parse_date, reverse=True)

        # ── 格式化输出 ────────────────────────────────────────────────────────
        header = "[系统注入的历史记忆（按时间从新到旧排列）]"
        lines  = [header]

        for chunk in sorted_chunks:
            date_label:   str   = chunk.get("metadata", {}).get("date", "未知日期")
            text:         str   = chunk.get("text", "").strip()
            rerank_score: float = chunk.get("rerank_score", float("nan"))

            score_str = (
                f" | score={rerank_score:.4f}"
                if not isinstance(rerank_score, float) or not rerank_score != rerank_score
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
        """返回当前记忆库完整统计，用于监控与调试。"""
        return {
            "global_version":   self.GLOBAL_MEMORY_VERSION,
            "chroma_count":     self._collection.count(),
            "bm25_docs":        len(self._bm25_doc_ids),
            "cache_size":       len(self._lru_cache),
            "cache_maxsize":    self._lru_cache.maxsize,
            "embedding_model":  self._embedding_fn.model_name,
            "embedding_dim":    self._embedding_fn.dimension,
            "persist_dir":      str(self._persist_dir),
            "collection_name":  self._collection_name,
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
        按 source 文件名删除对应的所有 Chunk，重建 BM25，并递增版本号。

        Args:
            source_filename: 要删除的文件名，例如 "2023-05-01.md"。

        Returns:
            包含操作结果的字典。
        """
        count_before = self._collection.count()
        self._collection.delete(where={"source": source_filename})
        self.GLOBAL_MEMORY_VERSION += 1
        self._rebuild_bm25()
        count_after = self._collection.count()
        result = {
            "deleted_source":  source_filename,
            "chunks_before":   count_before,
            "chunks_after":    count_after,
            "chunks_removed":  count_before - count_after,
            "version":         self.GLOBAL_MEMORY_VERSION,
        }
        logger.info(f"[Delete] {result}")
        return result

    def clear_cache(self) -> None:
        """手动清空 LRU 缓存（不影响向量库和 BM25 索引）。"""
        self._lru_cache.clear()
        logger.info("[Cache] LRU 缓存已手动清空。")

    def reset_all(self) -> None:
        """
        **危险操作**：清空 ChromaDB 集合、BM25 索引和 LRU 缓存，版本重置为 1。
        仅用于测试环境，生产环境请勿调用。
        """
        self._chroma_client.delete_collection(self._collection_name)
        self._collection = self._chroma_client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25_index   = None
        self._bm25_doc_ids = []
        self._bm25_corpus  = []
        self._lru_cache.clear()
        self.GLOBAL_MEMORY_VERSION = 1
        logger.warning("[Reset] 所有记忆数据已清空，版本重置为 1。")


# ──────────────────────────────────────────────────────────────────────────────
# 端到端演示（直接运行此文件可快速验证功能）
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, textwrap

    BANNER = "=" * 62

    # ── 模拟两份不同日期的求职记忆文件 ─────────────────────────────────────
    OLD_MEMORY = textwrap.dedent("""\
    # 求职目标

    用户的目标岗位是 **后端工程师**，目标城市是 **上海**，
    期望薪资范围为 25K-35K，对互联网大厂有强烈意向。

    ## 技术栈

    熟悉 Python、FastAPI、Redis、MySQL。
    对 Kafka 消息队列有初步了解，BM25 面试薄弱点在系统设计方向。

    ## 面试状态

    目前正在准备美团和字节跳动的面试，已完成两轮电话面试。
    """)

    NEW_MEMORY = textwrap.dedent("""\
    # 求职目标（最新更新）

    用户已将目标城市从上海更改为 **北京**，期望薪资上调至 **35K-50K**。
    新增目标公司：蚂蚁集团、PingCAP。

    ## 新增技术方向

    开始系统学习 **Golang** 和 **Kubernetes**，计划 3 个月达到中级水平。
    面试薄弱点已从系统设计转变为 **分布式事务与一致性协议**（Raft、2PC）。

    ## 面试进展

    字节跳动三面已通过，等待 HR 面。美团决定放弃，转投蚂蚁集团。
    """)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        old_file = tmpdir / "2023-05-01.md"
        new_file = tmpdir / "2024-03-15.md"
        old_file.write_text(OLD_MEMORY, encoding="utf-8")
        new_file.write_text(NEW_MEMORY, encoding="utf-8")

        print(f"\n{BANNER}")
        print("  AD-LTM 功能验证演示")
        print(BANNER)

        # ── 1. 初始化管理器 ───────────────────────────────────────────────
        print("\n[1/6] 初始化 AdvancedMemoryManager ...")
        db_dir  = tmpdir / "ad_ltm_db"
        manager = AdvancedMemoryManager(persist_dir=str(db_dir))
        print(f"      初始统计: {manager.get_stats()}")

        # ── 2. 写入旧记忆文件 ─────────────────────────────────────────────
        print(f"\n{BANNER}")
        print("[2/6] 写入旧记忆 (2023-05-01.md) ...")
        result = manager.update_memory_from_file(old_file)
        print(f"      写入结果: {result}")

        # ── 3. 写入新记忆文件 ─────────────────────────────────────────────
        print(f"\n{BANNER}")
        print("[3/6] 写入新记忆 (2024-03-15.md) ...")
        result = manager.update_memory_from_file(new_file)
        print(f"      写入结果: {result}")
        print(f"      已索引文件: {manager.list_indexed_sources()}")

        # ── 4. 首次检索（验证时序消解：新记忆应排在旧记忆前面）────────────
        query = "用户现在的目标城市和期望薪资是多少？"
        print(f"\n{BANNER}")
        print(f"[4/6] 首次检索 (验证时序消解)")
        print(f"      Query: {query}")
        print(BANNER)
        context = manager.retrieve(query)
        print(context)

        # ── 5. 验证 LRU 缓存命中 ─────────────────────────────────────────
        print(f"\n{BANNER}")
        print("[5/6] 再次检索（验证 LRU 缓存命中，不重新召回）...")
        context2 = manager.retrieve(query)
        cache_hit = context == context2
        print(f"      LRU 缓存命中: {cache_hit} ✓" if cache_hit else "      ❌ 缓存未命中")
        print(f"      缓存状态: {manager.cache_info}")

        # ── 6. 验证版本化缓存失效 ─────────────────────────────────────────
        print(f"\n{BANNER}")
        print("[6/6] 更新记忆文件后版本递增，旧缓存自动失效 ...")
        new_file.write_text(
            NEW_MEMORY + "\n## 最新动态\n用户已拿到蚂蚁集团 Offer，薪资 48K。\n",
            encoding="utf-8",
        )
        manager.update_memory_from_file(new_file)
        context3 = manager.retrieve(query)
        invalidated = context3 != context2
        print(f"      版本: {manager.GLOBAL_MEMORY_VERSION}")
        print(f"      缓存失效并重新检索: {invalidated} ✓" if invalidated else "      ❌ 缓存未失效")

        print(f"\n{BANNER}")
        print("  最终统计:")
        for k, v in manager.get_stats().items():
            print(f"    {k:22s}: {v}")
        print(BANNER)
        print("  AD-LTM 端到端验证完成 ✓")
        print(BANNER + "\n")
