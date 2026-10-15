import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, List, Optional


class Settings(BaseSettings):
    # 管理端写入的权威语料目录（原始语料 data/raw；可用 LEGAL_DATA_DIR 覆盖）
    data_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    # 入库扫描目录（阶段 1 清洗产物 raw_v2；可用 LEGAL_INGEST_DATA_DIR 覆盖）。
    # 与 data_dir 分离：管理端只写 data/raw 原始语料，派生语料由
    # data_tools/phase1_clean_data.py 全量重建；raw_v2 禁止被管理端写入
    #（LawManagementService 有保护性校验），.uploads 等隐藏目录不入扫描。
    ingest_data_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw_v2")
    
    # Elasticsearch 配置
    es_hosts: List[str] = ["http://localhost:9200"]
    es_user: str = "elastic"
    es_password: str = "123456"
    es_index_name: str = "legal_documents"
    es_verify_certs: bool = False
    es_request_timeout: int = 30

    # 管理端鉴权 Token（环境变量 LEGAL_ADMIN_TOKEN）
    # 仅用于保护 /api/v1/admin/* 接口；默认值仅供本地开发使用，生产环境务必通过 .env 覆盖
    admin_token: str = "dev_admin_token_change_me"
    
    # 模型配置
    # 本地路径示例: r"D:\Embeding\BAAI\bge-m3"
    # Docker 环境: /models/bge-m3
    embedding_model: str = "/models/bge-m3"
    
    # DeepSeek API 配置
    llm_api_type: str = "deepseek"  # deepseek
    # 密钥不设默认值：必须通过 .env 的 LEGAL_LLM_API_KEY 或环境变量注入，避免明文入库
    llm_api_key: str = ""
    llm_api_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-flash"  # 默认模型
    
    # 场景特定模型配置
    llm_summary_model: str = "deepseek-flash"  # 摘要生成（批量任务）
    # DeepSeek 连接配置（abstract_service 独立客户端使用）
    llm_deepseek_api_key: str = ""  # 通过 .env 或环境变量 LEGAL_DEEPSEEK_API_KEY 设置
    llm_deepseek_api_base_url: str = "https://api.deepseek.com/v1"  # DeepSeek API 地址
    llm_deepseek_model: str = "deepseek-flash"  # DeepSeek 模型名
    llm_dashboard_model: str = "deepseek-flash"  # 仪表盘数据（快速响应）- 速度优先
    llm_extraction_model: str = "deepseek-flash"  # 知识图谱/结构化提取 - 平衡与可靠性
    llm_comparison_model: str = "deepseek-flash"  # 结构化对比表（复杂推理）- 推理能力优先
    llm_qa_model: str = "deepseek-flash"  # 智能问答
    # 思考模式开关：平台默认开启思考，思考内容走 delta.reasoning_content，
    # 会挤占 max_tokens（思考吃满时 finish_reason=length、content 为空）并显著增加首字延迟。
    # 默认关闭；如确需深度推理可设 LEGAL_LLM_ENABLE_THINKING=true。
    llm_enable_thinking: bool = False

    # 多模态配置
    enable_image_support: bool = True  # 启用图片支持
    max_images_per_request: int = 4  # 每次请求最大图片数量
    supported_image_formats: list = ["jpeg", "png", "webp", "gif"]  # 支持的图片格式
    max_tokens: int = 4096  # 最大输出token数，保证长回答不截断
    temperature: float = 0.3  # 降低温度，输出更确定性，生成更快
    
    # 分块参数
    chunk_size: int = 800
    chunk_overlap: int = 150
    
    # 向量化参数
    batch_size: int = 4

    # CPU 推理线程数
    # 原实现硬编码为 2，在 16 核容器上导致 CPU 严重闲置（检索耗时被放大约 4 倍）。
    # 默认 8：兼顾单次请求延迟与并发请求间的公平性；可通过 LEGAL_TORCH_NUM_THREADS 覆盖。
    torch_num_threads: int = 8
    
    # 检索参数
    default_top_k: int = 5
    
    # Re-ranker 配置
    # Docker 部署下已启用：reranker 加载使用 low_cpu_mem_usage=True，
    # 峰值 ~2.4GB，叠加 bge-m3 (~2.4GB) 总占用 ~7.6GB，10g mem_limit 下安全。
    # 懒加载：仅在首次搜索触发 rerank 时加载，不影响启动。
    reranker_enabled: bool = True
    # 本地路径示例: r"D:\Embeding\BAAI\bge-reranker-v2-m3"
    # Docker 环境: /models/bge-reranker-v2-m3
    reranker_model: str = "/models/bge-reranker-v2-m3"
    reranker_top_k: int = 20
    # 精排批大小。原实现硬编码为 2，50 条候选需 25 次前向；默认 8 可显著减少批间开销
    reranker_batch_size: int = 8
    # 两阶段精排窗口：只对 RRF 融合结果的前 N 条跑 cross-encoder。
    # reranker 是 CPU 推理且占检索总延迟约 93%，窗口从 50 降到 20 可把该阶段耗时压到约 40%；
    # 窗口外的候选不丢弃，按 RRF 顺序拼在精排结果之后（详见 RetrieverService._rerank_two_stage）。
    reranker_window: int = 20
    # 精排并发上限（信号量）。
    # 路由改由线程池执行后，多个检索请求会同时进入 cross-encoder 推理，必须限制路数：
    #   CPU：单次推理已占用 torch_num_threads 个线程，并发数 × 8 超过 16 核会互相抢占；
    #  内存：单次 batch=8×512 的中间激活是瞬时的，多路叠加会在 10g mem_limit 下逼近 OOM。
    # 取 2（2×8=16 线程刚好吃满 16 核）：超出的请求在各自线程池线程内等待信号量，
    # 事件循环与轻量接口不受影响。
    reranker_max_concurrency: int = 2

    # 引用链多跳（组件D）：把已召回条款在同法规内引用到的条款一并补入候选/上下文。
    # 解析走正文正则（同法规内「第X條」的编号是局部的，无需解析法规名），
    # 不发起 LLM 调用，单次扩展只增加 1 次 ES msearch。
    reference_expand_enabled: bool = True
    # 只对排名前 N 条结果解析引用，控制 ES 往返与扩展规模
    reference_expand_sources: int = 3
    # 单次检索最多补入的引用条款数
    reference_expand_max: int = 5
    # 引用条款进入 QA prompt 时的正文长度上限。
    # 定义类条款（如香港《私隐条例》第 2 条）正文可达上万字符，
    # 多条全量拼接会显著挤占上下文预算，故截断保留开头（规则主体部分）。
    reference_expand_max_chars: int = 1000

    # 跨法规引用扩展（多跳 A）：把条款正文里指向【其他已入库法规】的引用也补入候选。
    # 仅覆盖可从正文确定性推导 law_id 的模式（澳门「第N/YYYY號法律/行政法規」、
    # 香港「第N章」），复用同一次的批量回取，不额外发起 LLM 调用。
    # 实测（2000 条澳门语料抽样）：约 0.9% 的条款能解析出跨法规引用，其中约一半能
    # 回取到已入库的目标条款；折算到查询级约 2.7%（top-3 种子命中概率）。
    # 属补充能力而非主力召回通道。
    reference_expand_cross_law_enabled: bool = True

    # 多跳 B：查询分解。把「工伤赔偿和年假规定」这类复合查询拆成单一意图子查询，
    # 各子查询并行检索（跳过精排）后 RRF 合并，再对合并结果精排一次。
    # 规则前置（字面切分）→ LLM 兜底（明显复合但规则切不干净）→ 失败降级单跳。
    multihop_decompose_enabled: bool = True
    # 单次最多使用的子查询数（含锚定的原查询）。锚定原查询保证「只加候选、不丢召回」。
    multihop_decompose_max_subqueries: int = 3
    # LLM 兜底开关与超时（秒）。关闭后仅用规则，不产生任何额外延迟。
    multihop_decompose_llm_enabled: bool = True
    multihop_decompose_llm_timeout: float = 3.0

    # KG-RAG 融合：把知识图谱子图（实体关系三元组 + 来源条款摘录）作为一路补充证据
    # 并入问答 prompt，补全「只召回被查条文」时缺失的实体间关联。
    # 纯复用已有知识层产物（实体链接 + 定跳子图），不新增任何 LLM 调用；成本仅为一次
    # 查询向量编码 + 若干次 ES 检索，相对检索+生成的秒级耗时占比很低。
    # 链接不到实体 / 子图为空 / 任何异常一律静默降级为不注入，绝不阻断问答链路。
    qa_kg_evidence_enabled: bool = True
    # 子图扩展跳数（与图谱问答口径一致，默认 2 跳）
    qa_kg_evidence_hops: int = 2
    # 实体链接保留的种子实体数上限
    qa_kg_evidence_seed_entities: int = 5
    # 进入 prompt 的三元组条数上限（控制上下文膨胀，避免挤占法条原文预算）
    qa_kg_evidence_max_edges: int = 12
    # 进入 prompt 的条款摘录条数上限
    qa_kg_evidence_max_articles: int = 6
    # 单条条款摘录的字符上限
    qa_kg_evidence_article_chars: int = 400

    # 问答检索召回通道：开启后在纯向量 KNN 之外并联 BM25 全文检索，应用层 RRF 融合。
    # 背景：问答路径原仅用纯向量召回，遇到「靠机构名/法规名字面命中」的查询
    #（如「IMDA 对电信服务的监管要求」「ESIGN 法案对电子签名效力的规定」）top20 内
    # 0 命中 gold，而 Agent 侧走混合检索可命中 5 条 —— 这是 P2 分流丢召回的真实根因。
    # 开启后问答主链路召回质量提升；代价是多一次 BM25 的 ES 查询（纯本地检索，成本低）。
    # 注意：该开关会改变 /qa 问答链路行为，需重跑问答评测基线确认不劣化。
    qa_hybrid_enabled: bool = True
    # RRF 融合常数（与检索侧口径一致）
    qa_hybrid_rrf_k: int = 60
    # 仅被 BM25 召回（无余弦分）的条款，展示/质量判定用的 similarity 占位值。
    # 取 0.5 = 余弦映射公式 2/(3-cos) 在 cos=-1 时的下限，语义为「无向量分」。
    # 该值不参与排序（混合模式按 RRF 序），仅用于 prompt 展示与 _is_retrieval_poor 判定。
    qa_hybrid_bm25_only_similarity: float = 0.5

    # Agent（迭代式工具调用问答）：由 LLM 自主决定"下一步查什么"，
    # 在「检索 / 引用追溯 / 图谱查询」三个工具间多轮调用后统一合成答案。
    # 已启用：配合 agent_complexity_routing_enabled（默认开）做复杂度路由，复杂查询走
    # 多步探索、简单/定向查询自动走快速 RAG，故对简单查询几乎不增加延迟。
    # 全量 80 条实测（同批 RAG 对照）：HasAns 81.25% vs 77.50%，Faith 2.90 vs 2.66，
    # 平均耗时 18.6s vs 17.8s（+0.8s）。关闭时 /agent/stream 自动降级为既有 RAG 链路。
    agent_enabled: bool = True
    # 探索循环的最大步数（每步 1 次 LLM 决策调用）。本机 CPU 精排是瓶颈，
    # 步数直接决定成本上界，默认取小值。
    agent_max_steps: int = 5
    # 单次提问允许的工具调用总次数上限（一步内可能发起多个工具调用）。
    # 实测旧值 6 时几乎每次都跑满预算（avg≈5.9），说明模型是被预算截断、而非
    # 「证据足够自然收敛」。降到 4 配合系统提示词里的预算提示，促使模型主动收敛；
    # 仍不收敛时由 agent_max_stagnant_steps 的停滞短路兜底。
    agent_max_tool_calls: int = 4
    # 连续多少次工具调用未给证据池带来「新增」即提前结束探索（停滞短路）。
    # 模型可能用不同措辞重复检索同一内容，这类调用不产生新证据、只增加延迟，
    # 达到该阈值立即收口。取 2 以留出「一次空手 + 一次补救」的余量，避免误伤。
    agent_max_stagnant_steps: int = 2
    # 端到端时长上限（秒）。超时强制收口，用已收集证据作答，避免无限拖长
    agent_timeout_total: float = 180.0
    # 单次工具检索返回的条数（探索步够用即可，控制上下文膨胀）
    agent_tool_top_k: int = 5
    # 单个观测片段字符上限
    agent_obs_max_chars: int = 300
    # 证据收口时送精排的候选数上限（先收敛再精排）。
    # 工具多轮调用可累积数十条候选，而本机 cross-encoder 精排占检索延迟约 93%，
    # 候选数直接决定收口耗时（实测 ~30 条需 40s+）。故精排前先压到该上限，
    # 最终仍只取 top_k 条送 LLM，精度损失可忽略而成本大幅下降。
    agent_finalize_max_candidates: int = 12
    # 最终送入 LLM 的证据条数。
    # 需大于 top_k：多步探索额外召回的证据若在收口处被截到 top_k，探索就白做了
    #（实测 Agent 证据条数 5.00 < RAG 5.25，多召回未转化为更好答案）。取 8 使
    # 精排后的额外高相关条款也能进入 prompt，且不增加精排成本（精排上限仍是 12）。
    agent_finalize_pass_k: int = 8
    # 探索循环使用的模型；留空则复用 llm_qa_model
    agent_model: str = ""
    # 复杂度路由总开关：开启后（默认）只有复杂查询才走多步探索，简单/定向查询直接
    # 走既有快速 RAG，避免用简单查询的延迟换复杂查询的收益。关闭则恢复「一律走 Agent」。
    agent_complexity_routing_enabled: bool = True
    # 短于该长度的查询一律走快速 RAG（太短的句子几乎不会是真正的复合问题）
    agent_min_query_len: int = 12
    # 并列连词触发多步探索的最小查询长度。单独出现「和/与」太宽泛，需配合长度门槛，
    # 避免把「A 和 B 的关系」这类简单查询误判为复合查询。
    agent_conjunction_min_len: int = 15
    # 自适应分流（灰区探测）：规则无法确定走 Agent 还是快速 RAG 的「灰区」查询，
    # 先做一次廉价检索探测再决定。
    # **默认关闭**：实测该探测在本地语料上区分度不足——灰区查询的 top1 相似度
    # （similarity = 1/(1+distance)）集中在 0.82~0.88，无法区分「一次检索够不够」，
    # 15 条灰区查询全部被判为「足够」，未达到升级目的。开启还会多付一次向量检索成本。
    # 根因见 retriever 侧：快速路径用纯向量、Agent 用 BM25+向量混合检索，差距不在
    # 「检索质量特征」上，无法用相似度阈值预测。代码保留以便后续结合该结论重启。
    agent_probe_enabled: bool = False
    # 探测判定「一次检索是否足够」的 top1 相似度下限。
    # 注意：此处 similarity = 1/(1+distance) ∈ [0.5,1]，与 _is_retrieval_poor 的默认
    # 0.6（≈cos 0.33）相比，这里取更高值作为「高置信命中」门槛——只有 top1 足够强
    # 才认为一次检索足够；否则说明只是「都没有强匹配」，需要多轮探索。
    agent_probe_min_similarity: float = 0.7
    # 探测要求的最少证据条数
    agent_probe_min_results: int = 2

    # 检索结果缓存：精排是单次检索的耗时主体（约 93%），而前端刷新页面/切换标签后会
    # 按设计重新发起同一查询，缓存命中后整条链路（含精排）直接跳过。
    # 失效以 law_catalog.json 的 mtime 作为数据版本，版本变化时整体清空。
    # 只做精确命中（含繁简归一）：语义命中经实测不可行，依据见 utils/search_cache.py 模块注释。
    search_cache_enabled: bool = True
    # 条数上限。单条结果约 50 项、每项正文千字量级，500 条在百 MB 以内
    search_cache_max_entries: int = 500

    # 问答答案精确缓存：同一归一化问题 + 同一筛选条件 → 直接复用完整答案，跳过
    # 「检索 + LLM 生成」整条链路。LLM 生成是问答链路的耗时主体，缓存命中在计量上
    # 直接等于省下一次模型调用（对应「降低模型使用量」）。
    # 只做精确命中（含繁简归一）：语义命中经实测不可行，依据见 utils/search_cache.py 模块注释。
    # 生效范围由调用方收窄为「首轮提问」——带对话历史的追问、带图片的问答，其答案
    # 无法由问题文本唯一确定，不查本缓存。评测脚本可置 False 关闭，避免结果被缓存串场。
    qa_answer_cache_enabled: bool = True
    # 条数上限。单条为「答案正文 + 来源列表」，量级与检索缓存同档
    qa_answer_cache_max_entries: int = 300

    # 实体消歧 L3（向量对齐）：兜住 L1 繁简字表未覆盖的异体与措辞差异。
    # 实测结论（378 个真实实体、17323 个同类型候选对）：
    #   bge-m3 无法区分「语义相近但法律含义不同」的名称 —— 「最高一年徒刑」(0.933)
    #   与「最高二年徒刑」、「總體規劃」(0.929) 与「詳細規劃」余弦都极高，纯阈值方案
    #   在 0.90 时会产生至少 6 对错误合并。故 L3 定位为「变体兜底」而非「语义聚类」，
    #   必须叠加下方三重守卫，守卫才是判别主力。
    entity_align_enabled: bool = True
    # 向量相似度下限（配合守卫使用）。定位是「廉价预筛 + 控制候选审计量」，不是判别主力：
    # 真正的误合并拦截由下方三重守卫完成。取 0.95 而非 0.90，是为了让最接近的危险对
    #（「一年徒刑」vs「二年徒刑」0.933）根本不进入守卫判定，留出安全边际；
    # 实测把阈值降到 0.85，该对仍被数字守卫拦下、「總體/詳細規劃」仍被编辑距离守卫拦下
    entity_align_threshold: float = 0.95
    # 守卫一：归一化后名称的编辑距离上限。
    # 取 1 而非 2：实测编辑距离为 2 时仍有「總體規劃」vs「詳細規劃」(0.929)、
    # 「明確同意」vs「明確許可」、「一年徒刑」vs「二年徒刑」这类法律含义不同的对通过，
    # 且在阈值被调低时会真的误合并。收到 1 后只剩「纯词缀差异」可过
    #（前缀「處」、助词「的」、繁简表未覆盖的单个异体字），代价是放宽了极少数边缘同义对
    entity_align_max_edit: int = 1
    # 守卫二：归一化后名称长度上限。超长的是「描述型串」（整段目的条款、完整刑罚短语），
    # 它们表面相似度虚高且法律含义各异，一律不参与向量对齐
    entity_align_max_name_len: int = 40
    # 守卫三「数字一致」无开关：法律实体名中的数字承载「量」（刑期、金额、期限），
    # 数字不同必然不是同一实体，该守卫不做成可配项以避免误开
    # 批次内部两两比对：一对變体若在同一批次首次共同出现、词典里都还没有条目，
    # 只与词典比对会双双落库形成新的重复实体（历史遗留重复即由此产生）。
    # 开启后复用同一套守卫做同类型内部比对，可根治重复的复发源。
    entity_align_intra_batch: bool = True
    # 单次最多编码的实体名条数（覆盖词典回填 + 新实体补写），用于成本上界
    entity_align_embed_max: int = 1000
    # 向量编码批大小
    entity_align_batch_size: int = 32

    def llm_extra_body(self) -> Optional[Dict]:
        """构造 LLM 请求的额外参数（extra_body）。

        当前平台默认开启思考模式：思考内容通过 delta.reasoning_content 返回，
        会占用 max_tokens 预算（思考吃满时 finish_reason=length 且 content 为空，
        表现为"生成无内容"），并显著拉高响应耗时。默认显式关闭；
        需要深度推理时可通过 LEGAL_LLM_ENABLE_THINKING=true 打开。
        """
        if self.llm_enable_thinking:
            return None
        return {"thinking": {"type": "disabled"}}

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LEGAL_",
        case_sensitive=False
    )


settings = Settings()