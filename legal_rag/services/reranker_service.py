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
        return cls._instance

    def __init__(self):
        if hasattr(self, '_loaded'):
            return  # 单例已初始化，跳过

    def _lazy_load(self):
        """延迟加载模型，仅在首次使用时加载"""
        if self._loaded:
            return
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
                _os.environ.setdefault("OMP_NUM_THREADS", "2")
                _os.environ.setdefault("MKL_NUM_THREADS", "2")
                torch.set_num_threads(2)
                _log.info("已设置 OMP/MKL_NUM_THREADS=2, torch.set_num_threads(2)")
            else:
                import os as _os
                _os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

            _log.info("加载 tokenizer...")
            sys.stdout.flush()
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            _log.info("tokenizer 加载完成")

            _log.info("加载模型权重（先 CPU 加载，避免 accelerate device_map 崩溃）...")
            sys.stdout.flush()
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                device_map=None,
                trust_remote_code=True,
            )
            if device == "cuda":
                self._model = self._model.to("cuda")
            _log.info("模型权重加载完成")

            self._model.eval()
            self._device = device
            self._loaded = True
            _log.info(f"Re-ranker 模型加载完成: {model_name}")
            sys.stdout.flush()
        except Exception as e:
            raise RuntimeError(
                f"Re-ranker 模型加载失败: {e}\n"
                f"请先下载模型: git lfs clone {settings.reranker_model}"
            )

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
        self._lazy_load()

        import torch

        # 构造 query-doc pairs（截断 content 到 2000 字符，cross-encoder 评分不需要全文）
        pairs = []
        valid_candidates = []
        for doc in candidates:
            content = doc.get("content", "")
            if not content:
                continue
            pairs.append([query, content[:2000]])
            valid_candidates.append(doc)

        if not pairs:
            return candidates[:top_k]

        # batch 推理（每批 8 对，防止 OOM）
        all_scores = []
        batch_size = 8
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            inputs = self._tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            ).to(self._model.device if hasattr(self._model, 'device') else self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().tolist()
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                all_scores.extend(batch_scores)

        # 将分数附加到文档上
        for doc, score in zip(valid_candidates, all_scores):
            doc["rerank_score"] = round(score, 4)

        # 按 rerank_score 降序排列
        valid_candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        return valid_candidates[:top_k]
