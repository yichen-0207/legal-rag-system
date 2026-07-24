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
            os.environ.setdefault("OMP_NUM_THREADS", "2")
            os.environ.setdefault("MKL_NUM_THREADS", "2")
            # embedding 模型走 CPU，避免与 reranker 争 GPU 显存
            torch.set_num_threads(2)
            cls._model = SentenceTransformer(settings.embedding_model, device='cpu')
        return cls._model


def get_embedding_model() -> SentenceTransformer:
    """获取嵌入模型的便捷函数"""
    return ModelLoader.get_model()
