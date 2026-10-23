import threading
from typing import Optional
from sentence_transformers import SentenceTransformer
from core.config import settings


class ModelLoader:
    """单例模式的模型加载器"""

    _instance: Optional['ModelLoader'] = None
    _model: Optional[SentenceTransformer] = None
    # 最近一次加载失败原因，供 /health/ready 报告状态
    _load_error: Optional[str] = None
    # 加载锁：并发首次请求只加载一份 bge-m3，避免内存翻倍 OOM
    _load_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def load_state(cls) -> str:
        """模型加载状态：loaded / failed / pending（只读，不触发加载）。"""
        if cls._model is not None:
            return "loaded"
        if cls._load_error is not None:
            return "failed"
        return "pending"

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """获取共享的模型实例（并发安全）"""
        if cls._model is not None:
            return cls._model

        with cls._load_lock:
            # 双重检查：等锁期间可能已加载完成
            if cls._model is not None:
                return cls._model
            cls._load_model()
        return cls._model

    @classmethod
    def _load_model(cls) -> None:
        """执行模型加载（调用方必须持有 _load_lock）；失败直接抛出，下次请求重试。"""
        import torch

        try:
            # OMP/MKL 线程数需在进程启动前由环境变量注入，此处只设 intra-op
            torch.set_num_threads(settings.torch_num_threads)
            # low_cpu_mem_usage 避免加载时双倍内存分配（bge-m3 FP32 峰值 ~4.8GB → ~2.4GB）
            cls._model = SentenceTransformer(
                settings.embedding_model,
                device='cpu',
                model_kwargs={"low_cpu_mem_usage": True},
            )
        except Exception as e:
            cls._load_error = f"{type(e).__name__}: {e}"
            raise


def get_embedding_model() -> SentenceTransformer:
    """获取嵌入模型的便捷函数"""
    return ModelLoader.get_model()
