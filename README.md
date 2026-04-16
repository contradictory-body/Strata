<div align="center">

# Strata

**分层记忆支持的多模态求职助手 Agent / Web 应用**

*A Hierarchical Memory-Augmented Multimodal Job-Seeking Agent*

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-336791)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/Cache-Redis-DC382D)](https://redis.io/)
[![AgentScope](https://img.shields.io/badge/Framework-AgentScope-FF6B6B)](https://github.com/modelscope/agentscope)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-4A90D9)](https://www.trychroma.com/)
[![BM25](https://img.shields.io/badge/Retrieval-BM25-orange)](https://github.com/dorianbrown/rank_bm25)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-v6-blue)](https://github.com/contradictory-body/Strata)

[中文](#概述) · [English](#overview) · [快速开始](#-快速开始) · [前后端说明](#-前后端说明) · [架构设计](#-架构设计) · [实验结果](#-实验结果)

</div>

---

## 概述

**Strata** 是一个面向求职场景的智能助手系统，支持**对话求职辅导、长期记忆检索、结构化用户画像、JD 分析、简历与面试准备建议**等能力。

项目由两部分组成：

- **Agent 核心**：基于大语言模型与分层记忆架构，负责推理、记忆、工具调用与求职任务执行
- **Web 应用**：包含前端聊天界面与后端 API 服务，用于提供完整的可交互产品形态

当前版本的核心创新在于受认知科学启发的**三层记忆架构**（Hierarchical Memory Architecture），以及持续迭代的长期记忆检索系统。当前 README 原本主要介绍了 Agent 与实验部分，这一版补充了前端和后端的整体说明、启动方式与目录结构。

---

## Overview

**Strata** is an intelligent job-seeking assistant system for conversational career support, long-term memory retrieval, structured user profiling, JD analysis, and interview preparation.

The project consists of two major parts:

- **Agent Core**: handles reasoning, memory, tool calling, and job-seeking workflows
- **Web Application**: provides the frontend chat UI and backend API service for an end-to-end product experience

Its main innovation is a **Hierarchical Memory Architecture** for long-context, preference-aware job assistance.

---

## ✨ 核心特性

- 🧠 **三层记忆架构** — 工作记忆 / LLM 压缩摘要 / 长期语义记忆，自动分级管理
- 🔍 **三路混合检索（v6）** — 原文向量 + 意图向量 + BM25，经 RRF 融合后再精排
- 💡 **意图预计算写路径** — 写入时同步生成意图标准句并向量化
- 🚀 **规则层槽位提取替代 LLM** — 降低检索链路延迟
- 💾 **两级缓存** — 精确缓存 + 语义缓存，提升相似 Query 命中率
- ⏱ **时序冲突消解** — 新近记忆优先注入上下文
- 🚦 **请求澄清门控** — 对模糊输入进行澄清反问
- 👤 **结构化画像持久化** — 跨会话保存用户求职偏好
- 📄 **多模态文件处理** — PDF / Word / 图片统一解析
- 🌐 **Web 前后端集成** — 提供完整聊天界面、会话管理与实时交互体验
- ⚡ **ReAct + 异步管线** — 工具调用、记忆更新与画像更新异步执行

---

## 🖥 前后端说明

### 前端（Web UI）

前端提供一个面向用户的聊天式求职助手界面，支持：

- 新建对话与历史会话切换
- 实时消息流式展示
- 快捷提示词入口
- 在线连接状态展示
- 适合桌面端使用的双栏布局

从当前界面来看，产品风格偏简洁、轻量，核心目标是围绕“求职对话 + 任务引导”构建使用体验。

**前端主要职责：**

- 用户登录后进入求职助手界面
- 展示历史会话列表
- 向后端发送消息并接收流式回复
- 展示 Agent 生成的文本结果
- 承载求职任务入口，例如：
  - 找岗位
  - 分析 JD
  - 准备面试
  - 优化简历

> 建议在仓库中新增一张截图文件，例如：
>
> `docs/images/frontend-home.png`
>
> 然后在 README 中插入：
>
> ```md
> ![Strata 前端界面](docs/images/frontend-home.png)
> ```

### 后端（API Service）

后端负责整个 Web 应用的数据与业务逻辑支撑，主要包括：

- 用户、会话、消息等基础 API
- 调用 Agent 核心进行推理与回复生成
- 数据库存储（用户、会话、消息）
- Redis 缓存
- 将前端请求路由到 Agent 与工具链
- 管理 Agent 的工作目录、记忆文件和画像文件

从当前项目启动日志和环境变量配置可以确认，后端使用了：

- **FastAPI** 作为 Web 服务框架
- **PostgreSQL** 作为主数据库
- **Redis** 作为缓存
- **OpenAI 兼容接口** 作为模型调用方式
- **Agent Adapter** 负责将 Web 请求桥接到 `JobAgentV2`

### Agent 核心

Agent 是整个系统的“智能中枢”，主要负责：

- 意图识别
- 记忆检索
- 用户画像抽取
- 工具调用
- 求职建议生成
- 文件分析与长期记忆写入

也就是说：

- **前端** 负责交互
- **后端** 负责服务编排与数据管理
- **Agent** 负责智能能力本身

---

## 🏗 系统整体架构

```text
┌─────────────────────────────────────────────────────────────┐
│                         Frontend Web UI                     │
│   会话列表 / 聊天窗口 / 快捷操作 / 状态展示 / 流式消息显示      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Backend API (FastAPI)                │
│  Auth / Chat API / Session API / Message API / AgentAdapter│
│  PostgreSQL 持久化 / Redis 缓存 / 流式响应 / 配置管理         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Strata Agent Core                       │
│  clarify_gate → intent_extraction → memory_search → tools  │
│  profile_update → async memory write → response generation │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Storage & Retrieval                    │
│ PostgreSQL / Redis / ChromaDB / BM25 / Profile Markdown    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 架构设计

### 记忆分层模型

```text
┌─────────────────────────────────────────────────────────────┐
│                        用户对话输入                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐  │
│  │  Layer 1    │   │  Layer 2    │   │    Layer 3      │  │
│  │  工作记忆   │   │  压缩摘要   │   │  长期语义记忆   │  │
│  │             │   │             │   │                 │  │
│  │ 最近N轮原始 │→  │ LLM结构化   │→  │  AD-LTM v2      │  │
│  │ 对话历史    │   │ 增量压缩     │   │  原文向量       │  │
│  │             │   │ (~85% 压缩) │   │  + 意图向量     │  │
│  └─────────────┘   └─────────────┘   │  + BM25 索引   │  │
│                                       └─────────────────┘  │
│  ┌──────────────┐   ┌────────────────────────────────────┐ │
│  │ 工具结果外置  │   │        Profile 结构化画像            │ │
│  │ 截断+文件存储 │   │  7 维求职偏好 · 异步提取 · 跨会话   │ │
│  └──────────────┘   └────────────────────────────────────┘ │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│              ReAct Agent (Reasoning + Acting Loop)          │
│  clarify_gate → intent_extraction → memory_search          │
│  → pre_reasoning_hook → LLM → tools → async profile update │
└─────────────────────────────────────────────────────────────┘
```

### v6 写路径（Write Pipeline）

```text
update_memory_from_file(file_path)
  ├── GLOBAL_MEMORY_VERSION += 1
  │     └── semantic_cache.invalidate_version()
  ├── MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter
  ├── 对每个 Chunk:
  │     ├── extract_slots(text)
  │     ├── slots_to_sentence(slots)
  │     ├── Embedding(原文)    → vec_type=original
  │     └── Embedding(标准句) → vec_type=intent, original_id=chunk_id
  ├── ChromaDB 双写
  └── BM25 全量重建
```

### v6 读路径（Read Pipeline）

```text
retrieve_with_intent(query_text)
  ├── [Level 1] LRU 精确缓存
  ├── [Level 2] 语义缓存
  ├── 未命中 → 全量检索管线
  │     ├── build_intent_sentence_from_query()
  │     ├── 原文向量检索
  │     ├── 意图向量检索
  │     ├── BM25 检索
  │     ├── RRF 融合
  │     ├── Reranker 精排
  │     └── 时序消解
  └── 返回最终 Context
```

### 请求澄清门控

```text
用户输入
  │
  ├─ 规则层（零 LLM 开销）
  ├─ confidence < 0.7 → LLM 兜底
  ├─ is_ambiguous=True → 反问 + 挂起 pending_clarification
  └─ is_ambiguous=False → 进入检索与推理
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
git clone https://github.com/contradictory-body/Strata.git
cd Strata

conda create -n strata python=3.11 -y
conda activate strata

pip install -r requirements.txt
pip install jieba
```

---

## ⚙️ 环境变量配置

项目至少涉及两类配置文件：

### 根目录 `.env`
用于 Agent / 模型 / Embedding / Tavily 等配置

### `backend/.env`
用于 Web 后端服务配置，包括：

- PostgreSQL
- Redis
- JWT
- CORS
- LLM API
- 服务端口等

---

## 🌐 启动 Web 版（推荐）

### 1. 启动 PostgreSQL 与 Redis

确保本地 PostgreSQL 与 Redis 已启动。

### 2. 配置 `backend/.env`

示例：

```env
DATABASE_URL=postgresql+asyncpg://strata:strata_password@localhost:5432/strata_db
REDIS_URL=redis://localhost:6379/0

JWT_SECRET_KEY=CHANGE_THIS_TO_A_RANDOM_SECRET_KEY_IN_PRODUCTION
JWT_ALGORITHM=HS256
JWT_EXPIRE_DAYS=7

CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]

LLM_API_KEY=your_llm_api_key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3.5-plus

TAVILY_API_KEY=your_tavily_api_key

DATA_ROOT=data

APP_HOST=0.0.0.0
APP_PORT=8000
APP_DEBUG=false
```

### 3. 启动后端

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

默认后端地址：

```text
http://127.0.0.1:8000
```

### 4. 启动前端

如果前端为独立项目，进入前端目录后启动开发服务器。通常开发地址为：

```text
http://localhost:5173
```

> 如果你的前端确实基于 Vite，可使用：
>
> ```bash
> npm install
> npm run dev
> ```

### 5. 打开浏览器访问

```text
http://localhost:5173
```

---

## 💬 Web 版使用流程

1. 启动前后端服务
2. 打开前端页面
3. 新建对话
4. 输入求职问题，例如：
   - 我是一名 Python 后端工程师，有 3 年经验，帮我找工作
   - 帮我分析这份 JD
   - 帮我准备字节跳动的系统设计面试
5. Agent 根据历史记忆、画像与工具结果生成回复
6. 会话与用户画像会持续累积，支持跨轮记忆

---

## 🧪 CLI 版快速体验

除了 Web 版，也可以直接启动 Agent CLI。

### 1. 配置 API Key

```bash
cp .env.example .env
```

填入模型相关配置后，运行：

```bash
cd agent/reme_light_job_agent_v2
python job_agent.py
```

### 2. 常用命令

```text
/file <路径>   — 解析并分析 PDF / Word / 图片文件
/profile       — 查看当前结构化求职画像
/memory        — 查看最近命中的历史记忆
/status        — 查看工作目录状态
/clear         — 清除当前对话历史（不影响长期记忆）
/help          — 显示帮助
/exit          — 退出
```

---

## 📁 项目结构

```text
Strata/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
├── pyproject.toml
│
├── docs/
│   ├── architecture.md
│   └── images/
│       └── frontend-home.png         # 前端截图（建议新增）
│
├── backend/
│   ├── main.py                       # FastAPI 应用入口
│   ├── config.py                     # 后端配置读取
│   ├── database.py                   # PostgreSQL 初始化与连接
│   ├── chat/
│   │   └── agent_adapter.py          # Web API 到 Agent 的桥接层
│   ├── api/                          # 各类路由（按项目实际补充）
│   ├── models/                       # ORM / 数据模型（按项目实际补充）
│   ├── schemas/                      # Pydantic Schema（按项目实际补充）
│   └── .env.example
│
├── frontend/                         # Web 前端（若独立目录存在）
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.* / 其他前端配置
│
├── experiments/
│   ├── ablation_study.py
│   ├── run_gate_eval.py
│   ├── run_pcs_eval.py
│   ├── run_retrieval_v6.py
│   ├── dataset/
│   └── results/
│
└── agent/
    └── reme_light_job_agent_v2/
        ├── job_agent.py
        ├── advanced_memory_manager.py
        ├── intent_query_builder.py
        ├── semantic_cache.py
        ├── clarification_gate.py
        ├── tools.py
        ├── profile_manager.py
        ├── file_parser.py
        ├── eval.py
        ├── skills.py
        └── utils.py
```

> 如果你的实际仓库中前端目录名称不是 `frontend/`，请按真实目录名调整。

---

## 🛠 技术栈

| 分类 | 技术 |
|------|------|
| 前端 | Web UI（聊天式交互界面，支持会话列表与流式消息） |
| 后端 | FastAPI / Pydantic / asyncio |
| 数据库 | PostgreSQL |
| 缓存 | Redis |
| 核心框架 | Python 3.11+ / AgentScope |
| 大模型 | OpenAI 兼容 API（Qwen / DashScope 等） |
| 向量存储 | ChromaDB |
| 长期记忆检索 | BGE + BM25 + RRF |
| 精排模型 | BGE-reranker-base |
| 意图提取 | 规则层槽位提取 |
| 文件解析 | PyMuPDF / pdfplumber / python-docx / Pillow |

---

## 🧪 实验结果

所有指标均在真实运行中测量得出，不使用 RAGAS 等通用框架。当前 README 中的实验部分主要围绕 Agent 与记忆系统展开。以下保留原有实验结论。

### 数据集

构建 50 组多轮求职对话，按场景分为三类：

| 类型 | 数量 | 平均轮次 | 用途 |
|------|:---:|:---:|------|
| **A 类**（干净对话） | 20 组 | 15.2 轮 | 检索评估、偏好提取评估 |
| **B 类**（含模糊输入） | 15 组 | 14.6 轮 | 澄清门控评估 |
| **C 类**（长对话含工具） | 15 组 | 19.9 轮 | 工具压缩评估 |

### 关键指标

| 指标 | 数据集 | 实测值 | 阈值 | 状态 |
|------|------|:---:|:---:|:---:|
| Gate-F1 | B 类全量 | 0.294 | ≥ 0.80 | ⚠️ |
| CRA@4（v6） | 全量 | 0.477 | ≥ 0.65 | ↑ |
| MRR（v6） | 全量 | 0.306 | — | ↑ |
| TOR | C 类全量 | 0.861 | ≥ 0.80 | ✅ |
| PCS | 全量 | 0.876 | ≥ 0.80 | ✅ |

---

## 📌 使用建议

- 若想快速体验产品形态，优先使用 **Web 版**
- 若想调试 Agent 逻辑与记忆机制，优先使用 **CLI 版**
- 若想复现实验结果，使用 `experiments/` 下脚本
- 若想扩展求职能力，可从 `tools.py`、`profile_manager.py` 与 `agent_adapter.py` 入手

---

## 📝 引用

```bibtex
@software{strata2026,
  title   = {Strata: A Hierarchical Memory-Augmented Multimodal Job-Seeking Agent},
  version = {6.0},
  year    = {2026},
  url     = {https://github.com/contradictory-body/Strata}
}
```

---

## 📄 License

本项目采用 [MIT License](LICENSE) 开源协议。

---

<div align="center">

**如果觉得有帮助，请给个 ⭐ Star 支持一下！**

</div>