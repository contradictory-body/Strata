# Strata 实验说明

## 目录结构

```
experiments/
├── ablation_study.py          # 完整消融实验主脚本
├── dataset/
│   ├── generate_dataset.py    # 数据集生成脚本
│   ├── dataset_full.json      # 完整数据集（50 组）
│   ├── dataset_type_A.json    # A 类：干净对话（20 组）
│   ├── dataset_type_B.json    # B 类：含模糊输入（15 组）
│   ├── dataset_type_C.json    # C 类：含工具调用（15 组）
│   └── dataset_stats.json     # 数据集统计摘要
└── results/
    ├── results.json           # 实验结果（运行后生成）
    └── ablation_chart.png     # 对比图表（运行后生成）
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r ../requirements.txt
pip install jieba  # 可选，提升 BM25 中文召回

# 2. 配置 API Key
cp ../.env.example ../.env
# 编辑 ../.env，填入 LLM_API_KEY 等

# 3. 生成数据集（如 dataset/ 下没有 json 文件）
python dataset/generate_dataset.py

# 4. 运行完整消融实验
python ablation_study.py
```

## 数据集设计

50 组多轮求职对话，按场景分为三类：

| 类型 | 数量 | 平均轮次 | 核心用途 |
|------|:---:|:---:|------|
| **A 类** | 20 组 | 15.2 轮 | 基线检索评估、偏好提取评估 |
| **B 类** | 15 组 | 14.6 轮 | 澄清门控评估（60 个标注模糊输入） |
| **C 类** | 15 组 | 19.9 轮 | 工具压缩评估（45 次工具调用） |

### 每组对话的标注字段

| 字段 | 内容 | 用于指标 |
|------|------|---------|
| `ambiguous_turns` | 模糊输入所在轮次 + 类型 | Gate-F1 |
| `relevant_chunks` | 每轮的相关 chunk UUID（Ground Truth） | CRA@4 / MRR |
| `key_information` | 工具调用中必须保留的关键信息点 | TOR |
| `preference_changes` | 偏好变更事件（字段 + 值 + 轮次） | PCS |
| `memory_chunks` | 该组对话已索引的记忆 chunk 列表 | 所有检索指标 |

## 评估指标

| 指标 | 全称 | 计算方式 | 目标值 |
|------|------|---------|:---:|
| **Gate-F1** | 门控分类 F1 | TP/(TP+FP/2+FN/2) | ≥ 0.88 |
| **CRA@4** | 上下文召回准确率 | Hit Rate@4 | ≥ 0.75 |
| **MRR** | 平均倒数排名 | mean(1/rank_i) | ≥ 0.65 |
| **TOR** | 工具输出保留率 | 关键点覆盖率均值 | ≥ 0.85 |
| **TER** | Token 节省率 | (raw-cmp)/raw | ≥ 0.50 |
| **PCS** | 偏好提取召回率 | 捕获事件/总事件 | ≥ 0.85 |

## 为什么不使用 RAGAS

RAGAS 是优秀的通用 RAG 评估框架，但不适合本项目：

1. **评估者同源问题**：RAGAS 依赖 LLM 打分，而被评估系统也是 LLM。评估者与被评估系统同源，评估结果可能存在系统性偏差。

2. **不支持 Agent 特有能力**：RAGAS 的 `context_recall`、`faithfulness` 等指标面向单轮问答设计，无法评估多轮对话、工具调用、画像持久化、门控机制等 Agent 特有行为。

3. **定制化指标更可解释**：面向任务设计的指标（如 Gate-F1、TOR）能精确定位每个模块的性能，便于迭代优化和向面试官解释。

## 注意事项

- 首次运行会下载 HuggingFace 模型（BGE-small-zh-v1.5 ~90MB，BGE-Reranker-Base ~280MB）
- 完整实验耗时约 20-40 分钟（取决于 API 速度和模型下载速度）
- 如需快速验证，使用 Agent 目录下的 `eval.py`（小样本冒烟验证，约 5 分钟）
