# Strata 架构设计详解

## 概述

Strata 是基于 AgentScope 框架构建的多轮求职对话 Agent，核心创新在于三层记忆架构与两个升级模块（AD-LTM + Clarification Gate）。本文档详细描述各子系统的设计决策与数据流转。

---

## 一、三层记忆架构

### Layer 1 — 工作记忆（Working Memory）

**存储介质**：`self.messages: list[dict]`（内存，OpenAI 格式）

**生命周期**：进程内，`/clear` 命令或进程重启后清空

**触发压缩**：`estimate_chars(self.messages) > 8000` 时，调用 `pre_reasoning_hook`

**消息格式**：遵循 OpenAI Chat Completions API 规范，包含四种角色：
- `user`：用户输入（图片以 `[上传图片: xxx.png]` 占位，base64 不进历史）
- `assistant`：助手回复（含 `tool_calls` 字段的为工具调用轮）
- `tool`：工具返回结果（content 为截断后的字符串）
- `system`：每轮临时构建，不存入 `self.messages`

### Layer 2 — 压缩摘要记忆（Compressed Summary）

**存储介质**：`self.compressed_summary: str`（内存，纯文本）

**生命周期**：进程内，随工作记忆压缩而更新，进程重启后清空

**生成机制**：
1. `ContextChecker` 检测 token 占用，将旧消息切分为"待压缩"和"保留"两组
2. `Compactor`（同步）将待压缩消息用 LLM 生成纯文本摘要，立即注入本轮 System Prompt
3. `Summarizer`（异步后台）将同一批消息写入结构化 `.md` 文件落盘

**注入位置**：每轮 System Prompt 末尾，以 `=== 历史对话摘要 ===` 标记包裹

### Layer 3 — 长期语义记忆（Long-term Memory）

**存储介质**：`memory/*.md` 文件 + ChromaDB 向量索引 + BM25 内存索引

**生命周期**：持久化在磁盘，跨会话保留

**写入链路**（Summarizer 异步执行）：
```
对话消息 → MarkdownHeaderTextSplitter → RecursiveCharacterTextSplitter
→ BGE-small-zh-v1.5 向量化 → ChromaDB 局部替换
→ BM25 全量重建 → GLOBAL_MEMORY_VERSION += 1（LRU 失效）
```

**读取链路**（每轮 memory_search）：
```
标准化 Query
→ 向量路 Top-15（ChromaDB cosine）
→ BM25 路 Top-15（jieba 分词 + BM25Okapi）
→ RRF 融合（k=60）→ Top-8
→ BGE-Reranker-Base 精排 → Top-4
→ 时序消解（date 降序）
→ LRU 缓存（版本化 Key）
→ 格式化 Context 字符串注入 System Prompt
```

---

## 二、请求澄清门控（Clarification Gate）

### 设计动机

"垃圾输入导致垃圾召回"是 RAG 系统最常见的失效模式。用户在求职对话中频繁使用指代词、过泛表达、或在槽位严重缺失的情况下发出请求，若直接触发检索，召回质量会大幅下降，且错误信息可能写入长期记忆污染画像。

### 双层结构

**Layer 1 — 规则层**（零 LLM 开销，毫秒级）：
- 指代词检测：有指代词 → 检查上下文中是否有可解引用的对象关键词
  - 无上下文 → 直接判模糊（confidence=1.0）
  - 有上下文含可解引用关键词 → 直接放行（confidence=0.9）
  - 有上下文无可解引用关键词 → 灰色地带（confidence=0.6）
- 过泛动词检测：整句匹配 → 模糊（confidence=1.0）
- 槽位缺失检测：求职意图 + 核心槽位缺失≥2 → 模糊（confidence=1.0）
- 意图混杂检测：命中≥2个独立意图正则 → 模糊（confidence=0.9）

**Layer 2 — LLM 兜底**（仅在 confidence < 0.7 时触发）：
- 将用户输入、近期对话、历史记忆、画像摘要打包发给 LLM
- 要求 JSON 输出（temperature=0）：`is_ambiguous`、`ambiguity_type`、`reason`
- 失败时保守放行，不阻塞主流程

### 状态机

```python
pending_clarification = {
    "original_input":   str,   # 原始模糊输入
    "ambiguity_type":   str,   # 模糊类型
    "missing_slots":    list,  # 缺失槽位
    "clarify_question": str,   # 生成的反问
    "ask_count":        int,   # 已追问次数，上限 2
}
```

超过 `MAX_ASK_COUNT=2` 时强制放行，防止 Agent 陷入追问死循环。

### Query 标准化

澄清完成（或直接清晰）后，将请求标准化为结构化检索 Query：

```
[任务目标] | 岗位=X | 城市=Y | 约束=Z
```

标准化 Query 同时用于：
1. 长期记忆检索（比自然语言 Query 精度更高）
2. 槽位完整度校验（决定是否触发偏好写入）

---

## 三、AD-LTM 高级长期记忆模块

### 核心设计决策

**为什么同时用向量和 BM25**：向量检索擅长语义相似性（"在北京工作"≈"北漂"），BM25 擅长精确关键词（公司名、技术名词）。两者互补，RRF 融合后稳定提升召回率约 10 个百分点。

**为什么用 RRF 而非分数加权**：两路分数量级不同（余弦相似度 vs BM25 分数），直接加权需要精心调参，RRF 基于排名融合天然规避此问题。

**Reranker 精排的代价**：CrossEncoder 精度高但速度慢，因此只对 RRF 筛出的 Top-8 做精排，而非全量文档。

**时序消解的必要性**：Reranker 按语义相关度排序，完全不感知时间。将"目标城市上海（2023-05-01）"排在"目标城市更新为北京（2024-03-15）"前面，会导致 Agent 使用过期信息。强制时序排序确保最新记忆出现在 context 最前端，与 LLM 的 in-context 位置偏好配合，使新知被优先引用。

**版本化 LRU 缓存**：Key = `f"v{GLOBAL_MEMORY_VERSION}_{md5(query)}"` 。每次写入记忆（版本号递增），所有旧 Key 因前缀变化自动失效，无需手动 invalidate，O(1) 复杂度完成全量缓存失效。

### ChromaDB 距离转换

ChromaDB 配置 `hnsw:space: cosine` 后，`query()` 返回的 `distances` 是 cosine 距离（值域 [0, 2]），需转换：

```python
similarity = 1.0 - distance  # cosine distance = 1 - cosine_similarity
```

---

## 四、工具结果外置（Tool Result Compaction）

### 问题背景

`analyze_jd` 工具输出约 2000-4000 字符的结构化分析报告，`search_job` 返回约 1500-3000 字符的搜索结果。多轮工具调用后，这些内容积累在 `self.messages` 中，快速消耗上下文窗口。

### 解决方案

超过阈值（默认 800 字符）的工具输出：
1. 截断保留前 800 字符（关键信息集中在前段）
2. 完整内容写入 `tool_result/{uuid}.txt`
3. 截断处追加 `[Full content saved to: ...txt]` 引用
4. 7 天后自动清理过期文件

截断后 Token 节省约 56.6%，关键信息点保留率约 87%。

---

## 五、消息格式转换

系统内部存在两套消息格式，通过转换函数对接：

| 格式 | 用途 | 工具结果存储方式 |
|------|------|-----------------|
| OpenAI dict | 主对话历史 `self.messages` | `content` 字符串 |
| AgentScope Msg | ReMeLight 内部方法调用 | `content[0]["output"]` |

转换发生在：
- `openai_msgs_to_reme()`: `self.messages` → `list[Msg]`，传入 `pre_reasoning_hook`
- `_reme_msgs_to_openai()`: `list[Msg]` → `self.messages`，压缩后替换

图片在历史中只存文字占位符（`[上传图片: xxx.png]`），base64 原始数据不进历史，从根本上防止上下文因图片数据膨胀。

---

## 六、结构化画像持久化

### PROFILE.md 结构

```markdown
# 求职画像 (Job Seeker Profile)
> 最后更新: 2024-03-15 10:23

## 目标岗位
后端工程师（推荐系统方向）

## 目标城市
北京

## 技术栈
Python, FastAPI, Redis, Kafka

## 目标公司_行业
字节跳动 / 互联网大厂

## 薪资预期
40-60K

## 面试薄弱点
分布式事务与一致性协议

## 简历修改偏好
（暂未填写）
```

### 写入时机

每轮对话结束后，若满足以下两个条件，才触发异步 `_async_extract_profile`：
1. 标准化 Query 的槽位完整度 ≥ 2/3（防止碎片化信息写入）
2. LLM 从对话中成功提取到非空偏好字段

更新策略：覆盖写，后写的值覆盖先写的值，确保最新偏好优先。
