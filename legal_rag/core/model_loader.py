import threading
from typing import Optional
from sentence_transformers import SentenceTransformer
from core.config import settings


class ModelLoader:
    """单例模式的模型加载器"""

    _instance: Optional['ModelLoader'] = None
    _model: Optional[SentenceTransformer] = None
    # 最近一次加载失败的原因；仅用于 /health/ready 报告状态，不影响"失败后仍会重试"的现有行为
    _load_error: Optional[str] = None
    # 加载锁：路由改由线程池执行后，并发首次请求会同时进入加载流程；
    # 若不加锁会各自加载一份 bge-m3（各 ~2.4GB），在 10g mem_limit 下直接 OOM。
    _load_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def load_state(cls) -> str:
        """模型加载状态，供 /health/ready 判断就绪（只读，不触发加载）。

        - loaded：加载完成，检索链路可用
        - failed：最近一次加载失败（下次请求仍会重试）
        - pending：尚未开始加载或正在加载中
        """
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
            # 双重检查：等锁期间可能已被其他线程加载完成
            if cls._model is not None:
                return cls._model
            cls._load_model()
        return cls._model

    @classmethod
    def _load_model(cls) -> None:
        """真正执行模型加载（调用方必须持有 _load_lock）。

        注意：嵌入模型是检索链路的必需组件，加载失败时不做降级、直接抛出，
        由上层返回错误。失败后 _model 仍为 None，下次请求会重新尝试加载；
        失败原因记录到 _load_error，供 /health/ready 暴露未就绪状态。
        """
        import torch

        try:
            # 这里只设 PyTorch intra-op 线程数。OMP/MKL 的线程数与自旋策略必须由进程启动前的
            # 环境变量注入（compose 已注入）：OpenMP 运行时在首次并行区初始化时读取它们，
            # 而本函数执行时 torch 早已导入，此时再 os.environ.setdefault 不会生效。
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
        except Exception as e:
            cls._load_error = f"{type(e).__name__}: {e}"
            raise


def get_embedding_model() -> SentenceTransformer:
    """获取嵌入模型的便捷函数"""
    return ModelLoader.get_model()
