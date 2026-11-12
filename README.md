# 境内外法规智能检索与分析系统

面向境内外法规文本的智能检索与问答平台。系统以检索增强生成（RAG）为核心，融合混合检索、跨编码器精排、知识图谱与智能体多步推理，支持多法域、多语言的法规检索、问答、对比分析与知识可视化。

## 核心能力

| 模块 | 说明 |
|------|------|
| 条款级混合检索 | BM25 + 稠密向量双路召回，RRF 融合 + 两阶段跨编码器精排 |
| RAG 多轮问答 | 基于检索条文生成带来源标注的回答，支持多轮追问与多模态（图片）输入 |
| 智能体多步推理 | 复杂问题由 LLM 自主调用「检索 / 引用追溯 / 图谱查询」工具多步收集证据后再作答 |
| 跨法域专题分析 | 10 个法律专题的结构化对比报告（LLM 生成 + JSON 约束输出） |
| 法规对比分析 | 双法域同主题条款对照、差异统计与 AI 解读 |
| 相似法规推荐 | 法规级 / 条款级相似推荐，主题感知排序 + 跨法域多样性 |
| 法律知识图谱 | 实体-关系抽取、四层实体消歧、多视图可视化与路径查询 |
| 法规摘要 | 一键生成结构化摘要（目的 / 范围 / 要点 / 章节 / 关键词） |
| 数据看板与法规管理 | 统计可视化；法规新增 / 修改 / 删除并写回数据目录 |

## 系统架构

```
浏览器 :3000
   │
   ▼
nginx (legal-frontend) ──反向代理──► FastAPI (legal-backend) :8001
                                        │
                                        ▼
                                Elasticsearch (legal-es) :9200
                                (dense_vector + BM25/IK 分词)
```

后端分层：`API 路由层 → 服务层（检索/问答/分析/图谱/Agent）→ 数据访问层（Elasticsearch）`，模型层包含嵌入模型、精排模型与 LLM。

## 技术栈

| 层次 | 选型 |
|------|------|
| 后端框架 | FastAPI + uvicorn（Python 3.10） |
| 检索/存储 | Elasticsearch 8.11.0（dense_vector 向量检索 + BM25，IK 中文分词） |
| 嵌入模型 | BGE-M3（多语言稠密向量，CUDA + FP16 推理） |
| 精排模型 | bge-reranker-v2-m3（Cross-encoder，CUDA + FP16 推理） |
| 生成模型 | DeepSeek API（OpenAI 兼容协议） |
| 智能体编排 | LangChain + LangGraph（原生 Function Calling / ReAct） |
| 前端 | 原生 JavaScript（无框架）+ nginx |
| 流式协议 | SSE（Server-Sent Events） |
| 部署 | Docker Compose 多容器（es / redis / backend / frontend） |

## 快速开始

### 前置条件

- Docker / Docker Compose
- 模型权重：`BGE-M3` 与 `bge-reranker-v2-m3`（放置于 `models/` 目录，以只读方式挂载）
- DeepSeek API Key

### 配置

在项目根目录创建 `.env`（**含密钥，禁止入库**）：

```dotenv
LEGAL_LLM_API_KEY=your_deepseek_api_key
LEGAL_DEEPSEEK_API_KEY=your_deepseek_api_key
LEGAL_ES_PASSWORD=your_es_password
LEGAL_ADMIN_TOKEN=your_admin_token
LEGAL_MODELS_DIR=./models

# 可选：检索精排窗口、Agent 开关、复杂度路由等
LEGAL_RERANKER_WINDOW=20
LEGAL_AGENT_ENABLED=true
LEGAL_AGENT_COMPLEXITY_ROUTING_ENABLED=true
```

### 启动

```bash
# 构建并启动全部服务（首次需构建镜像）
docker compose up -d --build

# 查看状态（backend 健康检查依赖 /health/ready，模型预热完成后才就绪）
docker compose ps

# 查看日志
docker compose logs -f backend
```

启动后访问：

- 前端界面：http://localhost:3000
- 后端 API 文档：http://localhost:8001/docs
- 就绪探针：http://localhost:8001/health/ready

### 数据入库

将法规原文（PDF/DOCX 等可提取文本的格式）放入 `legal_rag/data/raw/<法域>/`，随后触发入库或通过管理端接口维护；入库扫描目录由 `LEGAL_INGEST_DATA_DIR` 指定。

阶段 1 清洗产物 `legal_rag/data/raw_v2/` 已随仓库提供，克隆后可直接入库；若修改了 `raw/`，需先重新运行清洗脚本再入库。

```bash
# 容器内执行（扫描阶段 1 清洗产物 raw_v2）
docker exec -e PYTHONUNBUFFERED=1 legal-backend python -m scripts.ingest
```

## 目录结构

```
.
├── docker-compose.yml        # 多容器编排（es / redis / backend / frontend）
├── Dockerfile                # 后端镜像（Python 依赖 + 源码）
├── Dockerfile.es             # Elasticsearch 镜像（烘焙 IK 分词插件）
├── docs/                     # 技术报告、评测报告、面试知识体系（仅本地留存，不入库）
├── legal_rag/                # 后端
│   ├── main.py               # 应用入口、模型预热、健康检查
│   ├── schemas.py            # API 响应模型
│   ├── scripts/              # 可执行入口（统一 python -m scripts.<name> 调用）
│   │   ├── ingest.py         # 数据入库
│   │   ├── run_abstracts.py  # 摘要补生成
│   │   ├── run_kg_extraction.py  # 知识图谱预抽取
│   │   └── dev_frontend.py   # 本地前端静态服务（开发用）
│   ├── api/v1/               # 路由：search / qa / agent / compare / topic / kg / stats / admin
│   ├── core/                 # 配置、模型加载、法域知识（法域别名/触发词唯一来源）
│   ├── services/             # 业务服务（检索 / 问答 / 分析 / 图谱 / 智能体）
│   │   └── agent/            # 智能体：工具声明与 ReAct 编排
│   ├── repositories/         # Elasticsearch 数据访问
│   ├── utils/                # 分块、查询增强、缓存、法规目录等
│   ├── data_tools/           # 数据治理脚本（清洗 / 校验 / 索引构建）
│   ├── evals/                # 评测：scripts / datasets / results
│   └── data/                 # 语料与法规目录（运行时挂载）
│       ├── raw/              # 原始语料（管理端写入，看板雷达图扫描）
│       └── raw_v2/           # 阶段 1 清洗产物（ingest 扫描目录，随仓库提供）
├── frontend/                 # 前端静态资源（原生 JS + nginx 配置）
└── models/                   # 本地模型权重（不入库）
```

## API 概览

| 路径 | 方法 | 说明 |
|------|------|------|
| `/api/v1/search` | GET | 法规条款混合检索 |
| `/api/v1/qa/stream` | GET / POST | 智能问答（SSE 流式，支持多模态；POST 走请求体，推荐前端使用以避免长历史触发 414） |
| `/api/v1/agent/stream` | GET / POST | 智能体多步推理问答（SSE 流式；POST 走请求体，推荐前端使用以避免长历史触发 414） |
| `/api/v1/compare` | GET | 双法域条款对比 |
| `/api/v1/topic-analysis` | POST | 跨法域专题结构化分析 |
| `/api/v1/kg/*` | GET | 知识图谱（法规图 / 实体搜索 / 统计） |
| `/api/v1/stats/*` | GET | 数据统计与看板 |
| `/api/v1/admin/*` | GET / POST | 法规管理（需 Token 鉴权） |
| `/health` `/health/ready` | GET | 存活 / 就绪探针 |

## 评测结果

自建条款级评测集（80 条基础查询 × 4 种语言变体）上的检索策略对比：

| 策略 | MRR | 前20命中率 | 严格Top-1 | R@10 | NDCG@10 |
|------|-----|-----------|-----------|------|---------|
| 纯 BM25 | 0.4213 | 72.19% | 33.12% | 0.1704 | 0.1410 |
| 纯向量 | 0.5842 | 76.88% | 49.06% | 0.1888 | 0.1981 |
| 混合 RRF | 0.5544 | 86.88% | 43.75% | 0.2324 | 0.2096 |
| **混合 + 精排** | **0.6591** | **86.88%** | **55.63%** | **0.2757** | **0.2766** |

> 口径：`前20命中率` 为前 20 条结果中至少命中一条 gold（即落盘字段 `Top1_Accuracy` 的真实含义）；`严格Top-1` 为排名第 1 条即命中 gold。数据来自 `legal_rag/evals/results/eval_viz_data.json` 与 `eval_per_query.csv`（2026-10-13，320 条 = 80 × 4 语言变体）。

问答评测（LLM-as-judge）与智能体对照（同批 80 条）：智能体多步推理在答案命中率、忠实度、相关性三项均优于单跳问答，平均延迟仅增加约 0.8s。

> 说明：LLM 评分不跨模型版本 / 提示词批次比较，上述数字仅在同批对照内有效。

## 开发说明

- **模型预热**：服务启动后在后台并行加载嵌入模型与精排模型并做推理预热；`/health/ready` 在预热完成前返回 503，前端容器据此等待，避免请求排队。
- **性能约束**：GPU（CUDA + FP16）部署下，冷启动就绪实测 31–43 秒（中位约 37 秒）；精排 20 条条款耗时由 CPU 的 11.4 秒降至 0.45 秒。仍采用两阶段精排（仅精排前 N 条）、并发信号量、事件循环卸阻塞、LRU 结果缓存等手段优化。
- **配置方式**：所有可调参数与密钥均通过环境变量注入（`LEGAL_` 前缀），无硬编码密钥。
- **智能体开关**：`LEGAL_AGENT_ENABLED` 控制 `/agent/stream` 是否走多步推理（关闭时自动降级为既有问答链路）；`LEGAL_AGENT_COMPLEXITY_ROUTING_ENABLED` 控制复杂度路由（简单/定向查询自动走快速链路）。

## 说明

本仓库包含代码、配置与阶段 1 清洗语料（`legal_rag/data/raw_v2/`）。技术报告、面试知识体系、内部方案等文档仅供本地留存，不纳入版本管理。
