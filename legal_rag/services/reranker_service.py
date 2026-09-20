"""
Re-ranker Service
================
使用 Cross-encoder 模型对混合检索结果进行精排重排序。

模型: BAAI/bge-reranker-v2-m3
理论: Cross-encoder 通过查询-文档对的 token 级交互计算相关度分数，
      比 Bi-encoder 的独立编码+余弦相似度精度更高。
"""

from typing import List, Dict, Optional
from core.config import settings


class ReRankerService:
    """Cross-encoder 重排序服务（单例，避免重复加载模型）"""

    _instance: Optional['ReRankerService'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._tokenizer = None
            cls._instance._device = None
            cls._instance._loaded = False
            cls._instance._load_error = None  # 加载失败原因；失败后不再重试，直接降级
        return cls._instance

    def __init__(self):
        if hasattr(self, '_loaded'):
            return  # 单例已初始化，跳过

    def _lazy_load(self) -> bool:
        """延迟加载模型，仅在首次使用时加载。

        返回 True 表示模型可用；返回 False 表示加载失败（不抛异常，调用方降级为不精排），
        避免单个模型加载失败导致整个服务起不来或检索接口报错。
        """
        if self._loaded:
            return True
        if self._load_error is not None:
            return False  # 已失败过，进程内不再重试，避免每次检索都卡在加载上

        import sys
        import logging
        _log = logging.getLogger(__name__)
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            model_name = settings.reranker_model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _log.info(f"开始加载 Re-ranker 模型: {model_name}, device={device}")

            if device == "cpu":
                import os as _os
                _num_threads = str(settings.torch_num_threads)
                _os.environ.setdefault("OMP_NUM_THREADS", _num_threads)
                _os.environ.setdefault("MKL_NUM_THREADS", _num_threads)
                torch.set_num_threads(settings.torch_num_threads)
                _log.info(f"已设置 OMP/MKL_NUM_THREADS={_num_threads}, torch.set_num_threads({_num_threads})")
            else:
                import os as _os
                _os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

            _log.info("加载 tokenizer...")
            sys.stdout.flush()
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            _log.info("tokenizer 加载完成")

            _log.info("加载模型权重（low_cpu_mem_usage 避免双倍内存峰值）...")
            sys.stdout.flush()
            # low_cpu_mem_usage=True (需要 accelerate) 直接加载到目标位置，
            # 避免 transformers 默认的「空权重 + checkpoint 缓冲」双倍分配。
            # bge-reranker-v2-m3 FP32 ~2.4GB，普通加载峰值 ~4.8GB，
            # 叠加已加载的 bge-m3 (~2.4GB) 会超过 10g mem_limit 触发 OOM (exit 137)。
            # 启用后峰值控制在 ~2.4GB，总占用 ~7.6GB 安全。
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            if device == "cuda":
                self._model = self._model.to("cuda")
            _log.info("模型权重加载完成")

            self._model.eval()
            self._device = device
            self._loaded = True
            _log.info(f"Re-ranker 模型加载完成: {model_name}")
            sys.stdout.flush()
            return True
        except Exception as e:
            # 降级处理：不抛异常、不阻塞启动，服务继续以 RRF 融合顺序提供检索结果
            self._load_error = str(e)
            _log.error(
                "Re-ranker 模型加载失败，本进程内跳过精排（检索与问答仍可用，结果按 RRF 融合顺序返回）。"
                "常见原因：镜像缺少 accelerate（low_cpu_mem_usage 依赖）或模型未下载。"
                f"模型={settings.reranker_model}，错误={e}。"
                "修复：docker compose build backend && docker compose up -d backend；"
                "或临时用 LEGAL_RERANKER_ENABLED=false 关闭精排。"
            )
            sys.stdout.flush()
            return False

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        对候选项进行重排序。

        Args:
            query: 原始查询文本
            candidates: 候选项列表（每项含 content 等字段）
            top_k: 返回前 k 条，默认使用 settings.reranker_top_k

        Returns:
            按重排序分数降序排列的候选项列表（每项增加 rerank_score 字段）
        """
        if not candidates:
            return []

        top_k = top_k or settings.reranker_top_k
        if not self._lazy_load():
            # 模型不可用：保持原有（RRF 融合）顺序返回前 top_k，不阻塞检索
            return candidates[:top_k]

        import torch

        # 构造 query-doc pairs（截断 content 到 500 字符，cross-encoder 评分只需摘要语义）
        pairs = []
        valid_candidates = []
        for doc in candidates:
            content = doc.get("content", "")
            if not content:
                continue
            pairs.append([query, content[:500]])
            valid_candidates.append(doc)

        if not pairs:
            return candidates[:top_k]

        # batch 推理（CPU 下批量适中 + 序列截断，控制中间张量内存）
        all_scores = []
        batch_size = settings.reranker_batch_size
        import gc
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            inputs = self._tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._model.device if hasattr(self._model, 'device') else self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().tolist()
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                all_scores.extend(batch_scores)

            # 显式释放中间张量并立即回收：批量推理的中间张量会驻留在 PyTorch
            # 缓存分配器中，若不及时回收，多轮检索后容器内存会持续增长
            # （实测从 5.7GiB 涨到 8.7GiB，逼近 mem_limit 反而拖慢推理）。
            # batch_size 已由 2 提升到 8，回收次数从 25 次降到 7 次，开销可接受。
            del inputs, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        # 将分数附加到文档上
        for doc, score in zip(valid_candidates, all_scores):
            doc["rerank_score"] = round(score, 4)

        # 按 rerank_score 降序排列
        valid_candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        return valid_candidates[:top_k]
