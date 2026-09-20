from typing import Optional
from sentence_transformers import SentenceTransformer
from core.config import settings


class ModelLoader:
    """单例模式的模型加载器"""
    
    _instance: Optional['ModelLoader'] = None
    _model: Optional[SentenceTransformer] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """获取共享的模型实例"""
        if cls._model is None:
            import torch
            import os
            num_threads = str(settings.torch_num_threads)
            os.environ.setdefault("OMP_NUM_THREADS", num_threads)
            os.environ.setdefault("MKL_NUM_THREADS", num_threads)
            # embedding 与 reranker 均走 CPU，共享同一线程预算
            torch.set_num_threads(settings.torch_num_threads)
            # low_cpu_mem_usage=True (需要 accelerate) 避免加载时的双倍内存分配
            # bge-m3 FP32 ~2.4GB，普通加载峰值 ~4.8GB 超过 6g mem_limit (exit 137)
            # low_cpu_mem_usage 直接加载到目标位置，峰值控制在 ~2.4GB
            cls._model = SentenceTransformer(
                settings.embedding_model,
                device='cpu',
                model_kwargs={"low_cpu_mem_usage": True},
            )
        return cls._model


def get_embedding_model() -> SentenceTransformer:
    """获取嵌入模型的便捷函数"""
    return ModelLoader.get_model()
