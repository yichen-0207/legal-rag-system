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

自建双层条款级评测集（149 条基础查询 × 4 语言变体 = 596 个变体；条款 gold 有效 452 条）上的检索策略对比（2026-11-15，四策略同源）：

**法规级（n=596）**

| 策略 | MRR | Hit@20 | 严格 rank1 | R@20 | NDCG@20 |
|------|-----|--------|-----------|------|---------|
| 纯 BM25 | 0.5869 | 81.88% | 47.99% | 0.6683 | 0.5645 |
| 纯向量 | 0.6856 | 84.90% | 58.39% | 0.6532 | 0.6130 |
| 混合 RRF | 0.7116 | 93.96% | 59.73% | 0.7686 | 0.6689 |
| **混合 + 精排** | **0.7264** | 93.62% | **60.07%** | **0.7688** | **0.6830** |

**条款级（n=452）**

| 策略 | MRR | Hit@20 | 严格 rank1 | R@20 | P@5 | NDCG@20 |
|------|-----|--------|-----------|------|-----|---------|
| 纯 BM25 | 0.1454 | 38.27% | 8.41% | 0.1800 | 0.0664 | 0.1169 |
| 纯向量 | 0.1833 | 38.50% | 9.96% | 0.1345 | 0.0836 | 0.1159 |
| 混合 RRF | 0.1888 | 55.75% | 9.73% | 0.2521 | 0.0845 | 0.1596 |
| **混合 + 精排** | **0.2241** | **55.97%** | **12.83%** | **0.2529** | **0.0912** | **0.1683** |

> 口径：`Hit@20` 为前 20 条中至少命中一条 gold（旧字段 `Top1_Accuracy` 的实际含义，代码已改名并新增 `Rank1_Accuracy`）；`严格 rank1` 为第 1 条即命中 gold，与除以 gold 数的 `R@1` 不同。数据来自 `legal_rag/evals/results/eval_metric_correction_v2.json`（由 2026-11-15 的 `eval_per_query_v2.csv` 零成本复算，未重跑检索）；条款级 NDCG@20 不在该 CSV 内，取自同一次运行的 `eval_viz_data_v2.json`。

> 独立精排脚本（v2rr）敏感性口径：法规级 MRR 0.7426 / 严格 rank1 63.59% / R@20 76.84% / NDCG@20 0.6892；条款级 MRR 0.2364 / 严格 rank1 14.38% / R@20 25.35% / P@5 9.47% / NDCG@20 0.1709。与四策略 0.2241 的差异来自精排查询增强、文档长度 500 vs 2000、引用扩展，不是同一配置的重复测量。

> 2026-11-15 起生产检索加入查询归一：单法域查询在召回前去掉法域名与问句套话（不删「法律/法规/规定」等词），精排仍用原查询。同口径（v2rr）复测：法规级 MRR 0.7573 / 严格 rank1 64.60% / R@20 78.64% / NDCG@20 0.7097；条款级 MRR 0.2456 / 严格 rank1 14.38% / R@20 28.53% / P@5 10.31% / NDCG@20 0.1883。上表四策略中的混合两行是归一前基线，未随本次改动重跑。

> 子集拆分：template（276 变体）条款 MRR 0.2069→0.2437，legacy 诊断 gold（176 条有条款 gold）0.1603→0.1934；法规级在 template 上精排没有提升（0.7659→0.7617），在 legacy 上提升（0.6647→0.6959）——法规级的精排增益全部来自 legacy 诊断集。

问答评测（LLM-as-judge，5 分制）：

- 单链路 80 条基础查询（2026-11-14）：HasAns@5 81.25%，忠实度 4.53，相关性 3.08。
- Agent vs RAG 同批 60 条回归子集（2026-11-14）：来源命中率 80.00%→83.33%，忠实度 3.25→3.73，相关性 3.67→3.57，平均延迟 16.6s→21.4s，平均工具调用 0→3.97。
- 该对比中 22/60 条 Agent 未调用工具（复杂度路由走快速链路），且 Agent 平均返回 9.6 条来源、RAG 5.6 条。固定 5 条证据预算复算后：来源命中率 80.00% vs 81.67%（5 升 / 4 降，McNemar p=1.0）；忠实度在 30 对工具样本上从 +0.80 缩到 +0.20（11 升 / 6 降，p=0.33）。即两项表面优势主要来自"返回了更多来源"，不是推理质量。
- Agent 的相关性差值不显著（12 降 / 8 升 / 40 平，符号检验 p=0.503），且 12 条掉分全部来自调用工具的子集；7 条与作答政策（拒答、引错法、只答子问题）相关，不是多跳检索稀释。
- 2026-11-15 作答政策调整（结论先行、单法域不做分法域对照、边界内先答）后，20 条定向复测：相关性 3.25→3.80（11 升 / 1 降，p=0.006），忠实度 3.70→3.75（持平）；其中原 12 条掉分 3.00→3.83（9 升 / 0 降），8 条对照 +0.12（2 升 / 1 降）。样本经过富集，不等于全量 60 条的效果。
- 80 条与 60 条的子集、判定输入（chunk 1500 字 vs 法条全文）不同，两组数字不可横向比较；`HasAns` 的 gold 按 base 并集，口径偏宽，不能与条款级 `R@20` 并列。

> RAGAS（60 条，2026-10-17，仅 2/5 指标可用）：context precision 0.6906（无参考）/ 0.5008（有参考）；faithfulness、noise、context recall 当时停用，原记的「DeepSeek NLI 不兼容」已更正——真实原因是推理模型未关闭思考，思维链耗尽输出预算（见 2026-11-15 记录）。该批次早于 11 月改动，不作为当前结论。补齐 `thinking: disabled` 后已复测通过（2 条，2/2 有效，97 秒），但官方 NLI 指标约 3.2 万 token/条，全量 40 条需约 120 万 token，暂不全量跑。
>
> RAGAS 定义对齐的自实现指标（20 对，2026-11-15，非官方 ragas 输出，不可与榜单比较）：断言级忠实度 RAG 0.981 vs Agent 1.000（3 对 RAG 更低 / 16 平，符号检验 p=0.25），指标已饱和；非 gold 来源贡献率 0.462 vs 0.460（p=0.58）。统一压到 5 条证据后 Agent 0.653 / RAG 0.919（11 升 / 1 降，p=0.0063），即 Agent 约 35% 的断言依赖第 6 条以后的证据。该指标与手写 5 分制 judge 的相关系数仅 0.17，说明旧口径分辨力不足以观察断言级差异。详见 `2026-11-15/clause-funnel/work/RAGAS_ALIGNED_MINIMAL.md`。
>
> 多跳检索（35 条，2026-10-29）：MRR 0.4453→0.5560，R@10 0.7857→0.9143，NDCG@10 0.5188→0.6267；样本小，entity navigation 的 MRR 反而下降（0.7292→0.5918）。

> 说明：LLM 评分不跨模型版本 / 提示词批次比较，上述数字仅在同批对照内有效。

## 开发说明

- **模型预热**：服务启动后在后台并行加载嵌入模型与精排模型并做推理预热；`/health/ready` 在预热完成前返回 503，前端容器据此等待，避免请求排队。
- **性能约束**：GPU（CUDA + FP16）部署下，冷启动就绪实测 31–43 秒（中位约 37 秒）；精排 20 条条款耗时由 CPU 的 11.4 秒降至 0.45 秒。仍采用两阶段精排（仅精排前 N 条）、并发信号量、事件循环卸阻塞、LRU 结果缓存等手段优化。
- **配置方式**：所有可调参数与密钥均通过环境变量注入（`LEGAL_` 前缀），无硬编码密钥。
- **智能体开关**：`LEGAL_AGENT_ENABLED` 控制 `/agent/stream` 是否走多步推理（关闭时自动降级为既有问答链路）；`LEGAL_AGENT_COMPLEXITY_ROUTING_ENABLED` 控制复杂度路由（简单/定向查询自动走快速链路）。

## 说明

本仓库包含代码、配置与阶段 1 清洗语料（`legal_rag/data/raw_v2/`）。技术报告、面试知识体系、内部方案等文档仅供本地留存，不纳入版本管理。
