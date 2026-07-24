import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional


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
    
    # 模型配置
    embedding_model: str = r"D:\Embeding\BAAI\bge-m3"
    
    # DashScope API 配置（阿里云通义千问）
    llm_api_type: str = "dashscope"  # dashscope
    llm_api_key: str = "sk-34b64f1b62654c9e9143ef3dc52844a6"
    llm_api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "qwen3.6-plus"  # 默认模型（多模态支持）
    
    # 场景特定模型配置
    llm_summary_model: str = "qwen3.5-flash"  # 摘要生成（批量任务）- 极致性价比
    llm_summary_fallback_model: str = "qwen3.6-max-preview"  # 摘要生成备用模型 - 当主模型token耗尽或限流时替换
    # DeepSeek 兜底配置：当 DashScope 主/备模型都不可用（token耗尽/限流）时使用
    llm_deepseek_api_key: str = "sk-53b09eddf8a84f8b8649ab9c01f1d96c"  # DeepSeek API Key
    llm_deepseek_api_base_url: str = "https://api.deepseek.com/v1"  # DeepSeek API 地址
    llm_deepseek_model: str = "deepseek-v4-flash"  # DeepSeek 模型名
    llm_dashboard_model: str = "qwen3.6-flash"  # 仪表盘数据（快速响应）- 速度优先
    llm_extraction_model: str = "qwen3.6-plus"  # 知识图谱/结构化提取 - 平衡与可靠性
    llm_comparison_model: str = "qwen3.7-plus"  # 结构化对比表（复杂推理）- 推理能力优先
    llm_qa_model: str = "qwen3.7-plus"  # 智能问答（多模态）
    llm_qa_fallback_model: str = "qwen3.5-omni-plus"  # 智能问答备用模型 - token耗尽/限流时降级
    llm_fallback_model: str = "qwen3.5-plus"  # 兜底模型 - 当主模型token耗尽或限流时自动替换
    llm_extraction_fallback_model: str = "qwen3.7-max"  # 结构化提取专用备用模型

    # 多模态配置
    enable_image_support: bool = True  # 启用图片支持
    max_images_per_request: int = 4  # 每次请求最大图片数量
    supported_image_formats: list = ["jpeg", "png", "webp", "gif"]  # 支持的图片格式
    max_tokens: int = 4096  # 最大输出token数，保证长回答不截断
    temperature: float = 0.3  # 降低温度，输出更确定性，生成更快
    llm_enable_thinking: bool = False  # 关闭思考模式，显著降低首token延迟
    
    # 分块参数
    chunk_size: int = 800
    chunk_overlap: int = 150
    
    # 向量化参数
    batch_size: int = 4
    
    # 检索参数
    default_top_k: int = 5
    
    # Re-ranker 配置（默认开启，模型已下载）
    reranker_enabled: bool = True
    reranker_model: str = r"D:\Embeding\BAAI\bge-reranker-v2-m3"
    reranker_top_k: int = 20
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LEGAL_",
        case_sensitive=False
    )


settings = Settings()