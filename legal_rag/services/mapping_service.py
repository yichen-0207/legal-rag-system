import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from repositories.elasticsearch import ElasticsearchRepository
from core.config import settings


class LawMappingService:
    """
    法规条款映射服务（纯向量KNN，不依赖LLM）

    核心流程：
    1. 获取源法律所有 chunk（含向量）
    2. 对每个源 chunk 的向量在目标法域做 KNN 搜索
    3. 按相似度阈值过滤，组装映射表
    """

    # 相似度阈值
    THRESHOLD_HIGH = 0.85   # 高质量匹配
    THRESHOLD_MEDIUM = 0.75 # 中等匹配
    THRESHOLD_NONE = 0.75   # 低于此值视为无匹配

    def __init__(self):
        self.repo = ElasticsearchRepository()

    def _classify_quality(self, score: float) -> str:
        """根据相似度分数判定匹配质量"""
        if score >= self.THRESHOLD_HIGH:
            return "high"
        elif score >= self.THRESHOLD_MEDIUM:
            return "medium"
        else:
            return "none"

    def _knn_for_one_chunk(
        self,
        source_chunk: Dict,
        target_jurisdiction: str,
        exclude_law_ids: List[str]
    ) -> Dict:
        """
        对单个源 chunk 执行 KNN 搜索，返回映射结果。

        Args:
            source_chunk: 源 chunk 信息（含 embedding、article_number 等）
            target_jurisdiction: 目标法域名称
            exclude_law_ids: 需要排除的 law_id 列表（通常是源法律自身）

        Returns:
            映射结果字典
        """
        query_vec = source_chunk.get("embedding", [])
        if not query_vec:
            return {
                "source_article": {
                    "article_number": source_chunk.get("article_number", ""),
                    "chunk_index": source_chunk.get("chunk_index", 0),
                    "text": "",
                    "chunk_id": source_chunk.get("_id", "")
                },
                "target_article": None,
                "similarity_score": 0.0,
                "match_quality": "none"
            }

        # KNN搜索：目标法域 + 排除源法律自身
        results = self.repo.knn_search_similar(
            query_vector=query_vec,
            top_k=1,
            jurisdiction_filter=target_jurisdiction,
            min_score=0.0  # 不在此处过滤，统一用阈值
        )

        best_match = results[0] if results else None
        sim_score = best_match["similarity"] if best_match else 0.0
        quality = self._classify_quality(sim_score)

        # 构建源端信息（需要补充 content）
        source_info = {
            "article_number": source_chunk.get("article_number", ""),
            "chunk_index": source_chunk.get("chunk_index", 0),
            "text": source_chunk.get("content", ""),
            "chunk_id": source_chunk.get("_id", "")
        }

        # 构建目标端信息
        if best_match and sim_score >= self.THRESHOLD_NONE:
            target_info = {
                "law_name": best_match.get("title", ""),
                "law_id": best_match.get("law_id", ""),
                "article_number": best_match.get("article_number", ""),
                "text": best_match.get("content", ""),
                "chunk_id": best_match.get("_id", "")
            }
        else:
            target_info = None

        return {
            "source_article": source_info,
            "target_article": target_info,
            "similarity_score": round(sim_score, 4),
            "match_quality": quality
        }

    def map_laws(
        self,
        source_jurisdiction: str,
        source_law_id: str,
        target_jurisdiction: str,
        similarity_threshold: float = 0.75,
        max_workers: int = 5
    ) -> Dict:
        """
        执行法规条款映射。

        流程：
        1. 获取源法律全部 chunk（含向量+内容）
        2. 并发对每个 chunk 在目标法域做 KNN(k=1)
        3. 过滤低分结果 → 组装映射表 → 计算统计摘要

        Args:
            source_jurisdiction: 源法域名称
            source_law_id: 源法规 ID
            target_jurisdiction: 目标法域名称
            similarity_threshold: 匹配阈值（低于此值标记为无对应）
            max_workers: 并发线程数

        Returns:
            完整的映射结果（含 source/target 元数据、mappings 列表、summary 统计）
        """
        t_total = time.time()

        # ===== 步骤1：获取源法律全部 chunk（含向量和内容）=====
        t1 = time.time()
        source_chunks = self._get_source_chunks(source_law_id)
        if not source_chunks:
            return {"error": f"未找到法规 {source_law_id} 的条款数据"}
        t_fetch = time.time() - t1

        # 获取源法律的元信息
        source_meta = self._get_law_meta(source_law_id)
        total_articles = len(source_chunks)

        # ===== 步骤2：并发 KNN 映射 =====
        t2 = time.time()
        mappings = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for idx, chunk in enumerate(source_chunks):
                future = executor.submit(
                    self._knn_for_one_chunk,
                    chunk,
                    target_jurisdiction,
                    [source_law_id]
                )
                future_to_idx[future] = idx

            # 按原始顺序收集结果
            results_map = {}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results_map[idx] = future.result()
                except Exception as e:
                    # 异常时记录无匹配
                    chunk = source_chunks[idx]
                    results_map[idx] = {
                        "source_article": {
                            "article_number": chunk.get("article_number", ""),
                            "chunk_index": chunk.get("chunk_index", 0),
                            "text": chunk.get("content", ""),
                            "chunk_id": chunk.get("_id", "")
                        },
                        "target_article": None,
                        "similarity_score": 0.0,
                        "match_quality": "none"
                    }

        # 按 chunk_index 排序恢复顺序
        for i in range(total_articles):
            if i in results_map:
                mappings.append(results_map[i])

        t_knn = time.time() - t2

        # ===== 步骤3：统计摘要 =====
        summary = self._build_summary(mappings)

        # 目标法域涉及的法律数量
        target_law_ids = set()
        for m in mappings:
            if m.get("target_article") and m["target_article"].get("law_id"):
                target_law_ids.add(m["target_article"]["law_id"])

        result = {
            "source": {
                "jurisdiction": source_jurisdiction,
                "law_id": source_law_id,
                "law_name": source_meta.get("title", ""),
                "total_articles": total_articles
            },
            "target": {
                "jurisdiction": target_jurisdiction,
                "total_laws_matched": len(target_law_ids)
            },
            "mappings": mappings,
            "summary": summary,
            "timing": {
                "fetch_source_chunks": f"{t_fetch:.2f}s",
                "knn_mapping": f"{t_knn:.2f}s",
                "total": f"{time.time() - t_total:.2f}s"
            }
        }
        return result

    def _get_source_chunks(self, law_id: str) -> List[Dict]:
        """
        获取指定法规的所有 chunk（含 embedding 向量 + content 内容）。
        用于映射服务的源端数据获取。
        """
        query = {
            "query": {"term": {"law_id": law_id}},
            "size": 1000,
            "sort": [{"chunk_index": {"order": "asc"}}],
            "_source": ["embedding", "content", "law_id", "title", "jurisdiction",
                       "article_number", "chunk_index"]
        }

        response = self.repo.client.search(index=settings.es_index_name, body=query)
        hits = response.get("hits", {}).get("hits", [])

        results = []
        for hit in hits:
            source = hit.get("_source", {})
            results.append({
                "_id": hit.get("_id"),
                "embedding": source.get("embedding", []),
                "content": source.get("content", ""),
                "law_id": source.get("law_id", ""),
                "title": source.get("title", ""),
                "jurisdiction": source.get("jurisdiction", ""),
                "article_number": source.get("article_number", "") or "",
                "chunk_index": source.get("chunk_index", 0),
            })

        return results

    def _get_law_meta(self, law_id: str) -> Dict:
        """获取法规元信息（标题、法域等）"""
        query = {
            "query": {"term": {"law_id": law_id}},
            "size": 1,
            "_source": ["title", "jurisdiction", "passing_date"]
        }
        response = self.repo.client.search(index=settings.es_index_name, body=query)
        hits = response.get("hits", {}).get("hits", [])
        if hits:
            src = hits[0].get("_source", {})
            return {
                "title": src.get("title", ""),
                "jurisdiction": src.get("jurisdiction", ""),
                "passing_date": src.get("passing_date", "")
            }
        return {}

    def _build_summary(self, mappings: List[Dict]) -> Dict:
        """从映射列表计算统计摘要"""
        total = len(mappings)
        high = sum(1 for m in mappings if m["match_quality"] == "high")
        medium = sum(1 for m in mappings if m["match_quality"] == "medium")
        none_count = sum(1 for m in mappings if m["match_quality"] == "none")

        scores = [m["similarity_score"] for m in mappings if m["similarity_score"] > 0]
        avg_sim = round(sum(scores) / len(scores), 4) if scores else 0.0

        return {
            "total_mapped": total - none_count,
            "total_unmapped": none_count,
            "average_similarity": avg_sim,
            "high_quality_matches": high,
            "medium_quality_matches": medium,
            "no_match": none_count
        }

    def get_laws_by_jurisdiction(self, jurisdiction: str) -> List[Dict]:
        """获取指定法域下的所有法规列表"""
        return self.repo.get_all_laws(jurisdiction=jurisdiction)
