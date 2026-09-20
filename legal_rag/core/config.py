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