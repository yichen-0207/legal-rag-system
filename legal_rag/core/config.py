import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, List, Optional


class Settings(BaseSettings):
    # 管理端写入的原始语料目录
    data_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    # 入库扫描目录：派生语料 raw_v2，由 data_tools/phase1_clean_data.py 全量重建
    ingest_data_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw_v2")

    # Elasticsearch 配置
    es_hosts: List[str] = ["http://localhost:9200"]
    es_user: str = "elastic"
    es_password: str = "123456"
    es_index_name: str = "legal_documents"
    es_verify_certs: bool = False
    es_request_timeout: int = 30

    # 管理端鉴权 Token（保护 /api/v1/admin/*），生产环境务必通过 .env 覆盖
    admin_token: str = "dev_admin_token_change_me"

    # 模型路径（Docker 环境: /models/bge-m3）
    embedding_model: str = "/models/bge-m3"
    # embedding 推理设备：auto（按空闲显存自动选 GPU/CPU）/ cuda / cpu。运行期一键回退用 cpu
    embedding_device: str = "auto"

    # DeepSeek API 配置
    llm_api_type: str = "deepseek"  # deepseek
    # 密钥不设默认值，必须通过环境变量注入
    llm_api_key: str = ""
    llm_api_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-flash"  # 默认模型

    # 场景特定模型配置
    llm_summary_model: str = "deepseek-flash"  # 摘要生成
    # abstract_service 独立客户端使用的 DeepSeek 配置
    llm_deepseek_api_key: str = ""  # 环境变量 LEGAL_DEEPSEEK_API_KEY
    llm_deepseek_api_base_url: str = "https://api.deepseek.com/v1"  # DeepSeek API 地址
    llm_deepseek_model: str = "deepseek-flash"  # DeepSeek 模型名
    llm_dashboard_model: str = "deepseek-flash"  # 仪表盘数据，速度优先
    llm_extraction_model: str = "deepseek-flash"  # 知识图谱/结构化提取
    llm_comparison_model: str = "deepseek-flash"  # 结构化对比表，推理优先
    llm_qa_model: str = "deepseek-flash"  # 智能问答
    # 默认关闭思考模式（思考会挤占 max_tokens 并增加首字延迟），可设 LEGAL_LLM_ENABLE_THINKING=true
    llm_enable_thinking: bool = False

    # 多模态配置
    enable_image_support: bool = True
    max_images_per_request: int = 4
    supported_image_formats: list = ["jpeg", "png", "webp", "gif"]
    max_tokens: int = 4096  # 最大输出 token 数，避免长回答截断
    temperature: float = 0.3

    # 分块参数
    chunk_size: int = 800
    chunk_overlap: int = 150

    # 向量化批大小
    batch_size: int = 4

    # CPU 推理线程数，按容器核数调整
    torch_num_threads: int = 8

    # 检索参数
    default_top_k: int = 5

    # 精排配置：默认启用，首次搜索时懒加载
    reranker_enabled: bool = True
    reranker_model: str = "/models/bge-reranker-v2-m3"
    reranker_top_k: int = 20
    # 精排批大小
    reranker_batch_size: int = 8
    # 两阶段精排窗口：仅对 RRF 前 N 条精排，窗口外候选按序拼接在精排结果之后
    reranker_window: int = 20
    # 精排并发上限，避免 CPU 抢占与内存超限
    reranker_max_concurrency: int = 2

    # 引用链多跳：把同法规内引用到的条款补入候选（纯正则 + ES，不调 LLM）
    reference_expand_enabled: bool = True
    # 只对排名前 N 条结果解析引用
    reference_expand_sources: int = 3
    # 单次检索最多补入的引用条款数
    reference_expand_max: int = 5
    # 引用条款进入 QA prompt 的正文长度上限（定义类条款可能极长）
    reference_expand_max_chars: int = 1000

    # 跨法规引用扩展：仅覆盖可确定性解析的澳门/香港引用模式
    reference_expand_cross_law_enabled: bool = True

    # 多跳 B 查询分解：规则切分 + LLM 兜底，子查询并行检索后 RRF 合并
    multihop_decompose_enabled: bool = True
    # 单次最多使用的子查询数（含锚定的原查询）
    multihop_decompose_max_subqueries: int = 3
    # LLM 兜底开关与超时（秒）
    multihop_decompose_llm_enabled: bool = True
    multihop_decompose_llm_timeout: float = 3.0

    # 跨法域多跳：同一查询在多部法规上检索时是否并行（编码一次 + ES 并发，结果按下标回填）
    # 关闭后回到逐个串行，行为与改造前一致
    multihop_parallel_search_enabled: bool = True
    # 并行度上限：每个 worker 内部还会并发跑 knn+bm25，整体连接数约为 2×该值（ES 连接池默认 10）
    multihop_parallel_max_workers: int = 4

    # 纵向深挖：从图谱路径末端沿高置信关系再扩展若干轮（纯 ES，默认关闭）
    multihop_deepen_enabled: bool = False
    multihop_deepen_max_rounds: int = 2
    multihop_deepen_beam: int = 4
    multihop_deepen_fanout: int = 3
    # 每轮最多接入的新边数
    multihop_deepen_max_extra_edges: int = 8

    # KG-RAG 融合：把图谱子图作为补充证据并入问答 prompt，异常时静默降级
    qa_kg_evidence_enabled: bool = True
    # 子图扩展跳数
    qa_kg_evidence_hops: int = 2
    # 种子实体数上限
    qa_kg_evidence_seed_entities: int = 5
    # 进入 prompt 的三元组条数上限
    qa_kg_evidence_max_edges: int = 12
    # 进入 prompt 的条款摘录条数上限
    qa_kg_evidence_max_articles: int = 6
    # 单条条款摘录字符上限
    qa_kg_evidence_article_chars: int = 400

    # 问答检索在纯向量之外并联 BM25，应用层 RRF 融合
    qa_hybrid_enabled: bool = True
    # RRF 融合常数（与检索侧一致）
    qa_hybrid_rrf_k: int = 60
    # 仅被 BM25 召回条款的 similarity 占位值，不参与排序
    qa_hybrid_bm25_only_similarity: float = 0.5

    # Agent 多步推理：由 LLM 自主调用检索/引用追溯/图谱工具后合成答案
    agent_enabled: bool = True
    # 探索循环最大步数
    agent_max_steps: int = 5
    # 单次提问允许的工具调用总次数上限
    agent_max_tool_calls: int = 6
    # 连续无新增证据即提前收口
    agent_max_stagnant_steps: int = 2
    # 端到端超时（秒）
    agent_timeout_total: float = 180.0
    # 单次工具检索返回条数
    agent_tool_top_k: int = 5
    # 单个观测片段字符上限
    agent_obs_max_chars: int = 300
    # search_laws 每条结果是否附带 total_chars/truncated/omitted_chars。
    # A1 盲评：开臂去偏后 9 平 1 胜（唯一胜例来自关臂）、token 多 13.1%，且有一例稳定回归，
    # 无「开得更准」的证据，故默认关闭；开关保留，需要时可用环境变量打开做对照。
    agent_search_obs_meta_enabled: bool = False
    # 收口时送精排的候选数上限（同时是最终证据条数上限，控制 prompt 规模）
    agent_finalize_max_candidates: int = 12
    # 最终送入 LLM 的证据条数（应大于 top_k）
    agent_finalize_pass_k: int = 12
    # 收口精排开关：本机 cross-encoder 精排 12 条约需 22s，且只决定「12 选 12」的次序；
    # 关闭后按 RRF 序全量入 prompt，probe 增加 <1s，可显式开启回滚
    agent_finalize_rerank_enabled: bool = False
    # search_laws 单次批量检索的子查询数上限（并行执行，合并去重）
    agent_search_batch_max: int = 4
    # 批量检索合并后返回给模型的条款条数上限（控制观测值规模）
    agent_search_batch_result_max: int = 12
    # get_article_text 返回的正文截断字符数
    agent_article_text_chars: int = 2000
    # 探索循环模型，留空复用 llm_qa_model
    agent_model: str = ""
    # 复杂度路由：简单/定向查询直接走快速 RAG
    agent_complexity_routing_enabled: bool = True
    # 短于该长度的查询一律走快速 RAG
    agent_min_query_len: int = 12
    # 并列连词触发多步探索的最小查询长度
    agent_conjunction_min_len: int = 15
    # 灰区探测（实测区分度不足，默认关闭）
    agent_probe_enabled: bool = False
    # 探测判定「一次检索足够」的 top1 相似度下限
    agent_probe_min_similarity: float = 0.7
    # 探测要求的最少证据条数
    agent_probe_min_results: int = 2
    # 显式规划：多步探索前先产出检索计划
    agent_plan_enabled: bool = True
    # 规划调用输出上限
    agent_plan_max_tokens: int = 300
    # 多轮追问指代消解（仅在有历史且命中指代时触发）
    agent_query_rewrite_enabled: bool = True
    # 单工具执行超时（秒）
    agent_tool_timeout: float = 20.0

    # 检索结果缓存：按 law_catalog.json 的 mtime 失效，仅精确命中（含繁简归一）
    search_cache_enabled: bool = True
    # 缓存条数上限
    search_cache_max_entries: int = 500

    # 问答答案缓存：同一归一化问题与筛选条件直接复用完整答案，仅用于首轮提问
    qa_answer_cache_enabled: bool = True
    # 缓存条数上限
    qa_answer_cache_max_entries: int = 300

    # 答案层 Redis（跨 backend 重启留存）；默认关闭，关闭时行为与进程内缓存完全一致
    redis_cache_enabled: bool = False
    redis_url: str = "redis://legal-redis:6379/0"
    redis_key_prefix: str = "legal:qa:"
    # 连接/读写超时（秒）：必须短于一次检索，故障时不能拖慢请求
    redis_connect_timeout: float = 0.05
    redis_socket_timeout: float = 0.05
    # 答案条目 TTL（秒），24 小时
    redis_answer_cache_ttl: int = 86400
    # Redis 内答案条数上限（按写入时间淘汰）
    redis_answer_cache_max_entries: int = 300
    # bump_data_version 后是否后台异步清理旧版本前缀的答案缓存键（SCAN，非阻塞）
    redis_stale_cleanup_enabled: bool = True

    # 雷达图原始语料索引落盘目录。留空时退回系统临时目录（容器内 recreate 即丢）；
    # compose 通过命名卷挂到 /var/cache/legal-radar 并注入 LEGAL_RADAR_CACHE_DIR。
    radar_cache_dir: str = ""

    # 实体消歧 L3 向量对齐：仅作变体兜底，误合并由编辑距离/长度/数字三重守卫拦截
    entity_align_enabled: bool = True
    # 向量相似度下限（廉价预筛，判别靠守卫）
    entity_align_threshold: float = 0.95
    # 归一化名称编辑距离上限
    entity_align_max_edit: int = 1
    # 归一化名称长度上限，排除描述型长串
    entity_align_max_name_len: int = 40
    # 同批次内两两比对，避免同批新词重复落库
    entity_align_intra_batch: bool = True
    # 单次最多编码的实体名条数
    entity_align_embed_max: int = 1000
    # 向量编码批大小
    entity_align_batch_size: int = 32

    def llm_extra_body(self) -> Optional[Dict]:
        """构造 LLM 请求的 extra_body；默认显式关闭思考模式。"""
        if self.llm_enable_thinking:
            return None
        return {"thinking": {"type": "disabled"}}

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LEGAL_",
        case_sensitive=False
    )


settings = Settings()
