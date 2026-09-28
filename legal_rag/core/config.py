import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, List, Optional


class Settings(BaseSettings):
    # 数据目录
    data_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    
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