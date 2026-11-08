import logging
import threading
import time
from typing import Optional

from sentence_transformers import SentenceTransformer
from core.config import settings

logger = logging.getLogger(__name__)

# auto 模式下上 GPU 的最低空闲显存：embedding FP16 实测增量约 1090 MiB，
# 这里留一倍余量；低于阈值直接用 CPU，避免与精排/桌面进程抢显存导致加载 OOM。
GPU_MIN_FREE_MB = 2048
# 加载失败后的冷却窗口：期间请求快速失败，不再反复付出数十秒的加载代价。
LOAD_FAILURE_COOLDOWN_SEC = 60


class ModelLoader:
    """单例模式的模型加载器"""

    _instance: Optional['ModelLoader'] = None
    _model: Optional[SentenceTransformer] = None
    # 最近一次加载失败原因与发生时间，供 /health/ready 与冷却判定使用
    _load_error: Optional[str] = None
    _load_error_at: float = 0.0
    # 加载锁：并发首次请求只加载一份 bge-m3，避免内存翻倍 OOM
    _load_lock = threading.Lock()
    # GPU 加载/预热失败后置真：本进程内不再尝试 GPU，避免每次请求重复触发 OOM
    _force_cpu: bool = False
    # 实际生效的设备（cpu / cuda），供健康探针与日志核对
    _active_device: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def load_state(cls) -> str:
        """模型加载状态：loaded / cooldown / failed / pending（只读，不触发加载）。

        cooldown 表示最近一次加载失败且仍在冷却窗口内，请求会快速失败而非重试加载；
        冷却结束后转为 failed，下一次请求会重新尝试加载。
        """
        if cls._model is not None:
            return "loaded"
        if cls._load_error is not None:
            elapsed = time.time() - cls._load_error_at
            return "cooldown" if elapsed < LOAD_FAILURE_COOLDOWN_SEC else "failed"
        return "pending"

    @classmethod
    def load_error(cls) -> Optional[str]:
        """最近一次加载失败原因（无失败时为 None），供健康探针输出。"""
        return cls._load_error

    @classmethod
    def active_device(cls) -> Optional[str]:
        """实际生效的推理设备（未加载完成时为 None）。"""
        return cls._active_device

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """获取共享的模型实例（并发安全）。冷却期内直接失败，不再重试加载。"""
        if cls._model is not None:
            return cls._model

        with cls._load_lock:
            # 双重检查：等锁期间可能已加载完成
            if cls._model is not None:
                return cls._model
            if cls._load_error is not None:
                elapsed = time.time() - cls._load_error_at
                if elapsed < LOAD_FAILURE_COOLDOWN_SEC:
                    raise RuntimeError(
                        f"嵌入模型加载失败，冷却中（剩余 {LOAD_FAILURE_COOLDOWN_SEC - elapsed:.0f}s 内不重试）："
                        f"{cls._load_error}"
                    )
            cls._load_model()
        return cls._model

    @classmethod
    def _resolve_device(cls) -> str:
        """按 settings.embedding_device 决定推理设备；auto 时按空闲显存判断。"""
        import torch

        mode = (settings.embedding_device or "auto").strip().lower()
        if mode == "cpu":
            return "cpu"
        if cls._force_cpu:
            # 本进程已因 GPU 失败回落，无论配置如何都不再尝试 GPU
            return "cpu"
        if not torch.cuda.is_available():
            if mode == "cuda":
                logger.warning("配置为 embedding_device=cuda，但 CUDA 不可用，改用 CPU 加载")
            return "cpu"
        if mode == "cuda":
            return "cuda"

        # auto：先看空闲显存，不够就用 CPU，避免与精排/桌面进程抢显存
        free_mb = torch.cuda.mem_get_info()[0] / 1024 ** 2
        if free_mb < GPU_MIN_FREE_MB:
            logger.warning(
                f"GPU 空闲显存 {free_mb:.0f} MiB 低于阈值 {GPU_MIN_FREE_MB} MiB，embedding 改用 CPU"
            )
            return "cpu"
        return "cuda"

    @classmethod
    def _build_model(cls, device: str) -> SentenceTransformer:
        """按设备构建模型：GPU 用 FP16（显存减半），CPU 保持 FP32。"""
        if device == "cuda":
            import torch

            return SentenceTransformer(
                settings.embedding_model,
                device="cuda",
                model_kwargs={"torch_dtype": torch.float16, "low_cpu_mem_usage": True},
            )
        return SentenceTransformer(
            settings.embedding_model,
            device="cpu",
            model_kwargs={"low_cpu_mem_usage": True},
        )

    @classmethod
    def _dtype_of(cls, model: SentenceTransformer) -> str:
        """读出主干 transformer 的权重精度（cuda 应为 float16，cpu 为 float32）。"""
        try:
            return str(next(model[0].auto_model.parameters()).dtype)
        except Exception:
            return "unknown"

    @classmethod
    def _warm_gpu(cls, model: SentenceTransformer) -> None:
        """GPU 路径做一次真实编码：首次 cuDNN/CUBLAS 初始化与显存分配都在此处暴露。"""
        import torch

        with torch.no_grad():
            model.encode(
                ["warmup 劳动关系"],
                batch_size=1,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

    @classmethod
    def _load_model(cls) -> None:
        """执行模型加载（调用方必须持有 _load_lock）；失败直接抛出，下次请求按冷却规则重试。

        GPU 加载或预热抛异常（含 torch.cuda.OutOfMemoryError）时回落到 CPU 重载一次，
        并置 _force_cpu，本进程内不再尝试 GPU。
        """
        import torch

        try:
            # OMP/MKL 线程数需在进程启动前由环境变量注入，此处只设 intra-op
            torch.set_num_threads(settings.torch_num_threads)
            device = cls._resolve_device()
            try:
                model = cls._build_model(device)
                if device == "cuda":
                    cls._warm_gpu(model)
            except Exception as e:
                if device != "cuda":
                    raise
                # GPU 失败（显存不足、驱动/kernel 不兼容等）→ 回落 CPU 重载一次
                logger.warning(
                    f"embedding GPU 加载/预热失败（{type(e).__name__}: {e}），"
                    f"回落 CPU 重载；本进程后续不再尝试 GPU"
                )
                cls._force_cpu = True
                device = "cpu"
                model = cls._build_model(device)
            cls._model = model
            cls._active_device = device
            cls._load_error = None
            cls._load_error_at = 0.0
            logger.info(f"embedding 模型加载完成，device={device}, dtype={cls._dtype_of(model)}")
        except Exception as e:
            cls._model = None
            cls._active_device = None
            cls._load_error = f"{type(e).__name__}: {e}"
            cls._load_error_at = time.time()
            # 记录失败原因与冷却窗口：后续请求会在冷却期内快速失败，不再重试加载
            logger.error(
                f"embedding 模型加载失败（{cls._load_error}），"
                f"{LOAD_FAILURE_COOLDOWN_SEC}s 内请求快速失败"
            )
            raise


def get_embedding_model() -> SentenceTransformer:
    """获取嵌入模型的便捷函数"""
    return ModelLoader.get_model()
