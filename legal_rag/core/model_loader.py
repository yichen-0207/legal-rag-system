import threading
from typing import Optional
from sentence_transformers import SentenceTransformer
from core.config import settings


class ModelLoader:
    """单例模式的模型加载器"""

    _instance: Optional['ModelLoader'] = None
    _model: Optional[SentenceTransformer] = None
    # 加载锁：路由改由线程池执行后，并发首次请求会同时进入加载流程；
    # 若不加锁会各自加载一份 bge-m3（各 ~2.4GB），在 10g mem_limit 下直接 OOM。
    _load_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """获取共享的模型实例（并发安全）"""
        if cls._model is not None:
            return cls._model

        with cls._load_lock:
            # 双重检查：等锁期间可能已被其他线程加载完成
            if cls._model is not None:
                return cls._model
            cls._load_model()
        return cls._model

    @classmethod
    def _load_model(cls) -> None:
        """真正执行模型加载（调用方必须持有 _load_lock）。

        注意：嵌入模型是检索链路的必需组件，加载失败时不做降级、直接抛出，
        由上层返回错误。失败后 _model 仍为 None，下次请求会重新尝试加载。
        """
        import os
        import torch

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


def get_embedding_model() -> SentenceTransformer:
    """获取嵌入模型的便捷函数"""
    return ModelLoader.get_model()
