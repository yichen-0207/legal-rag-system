from typing import List, Dict, Optional
import numpy as np
from repositories.elasticsearch import ElasticsearchRepository


class SimilarLawService:
    """相似法规推荐服务（主题感知版）。

    法律文本的向量相似度主要反映结构相似性而非主题相关性，故采用「主题优先 + 向量为辅」：
    主题硬过滤、主题重叠度主导排序（60%），向量相似度仅区分同主题内的相关程度（40%）。
    """

    def __init__(self, repo: Optional[ElasticsearchRepository] = None):
        self.repo = repo or ElasticsearchRepository()

    # ===== 主题分析 =====

    def _analyze_law_topics(self, chunks: List[Dict]) -> Dict[str, int]:
        """统计法规各主题的出现频次，返回 {"数据保护": 12, ...}。"""
        topic_counts: Dict[str, int] = {}
        for chunk in chunks:
            for topic in chunk.get("topic_labels", []):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        return topic_counts

    def _compute_topic_overlap(self, query_topics: Dict[str, int], candidate_topics: Dict[str, int]) -> float:
        """计算两个法规的主题重叠度（频次加权 Jaccard，0~1）：高频主题的重叠更有意义。"""
        if not query_topics or not candidate_topics:
            return 0.0

        common_topics = set(query_topics.keys()) & set(candidate_topics.keys())
        if not common_topics:
            return 0.0

        # 加权交集：取两个法规中该主题频次的较小值
        weighted_intersection = sum(
            min(query_topics[t], candidate_topics[t]) for t in common_topics
        )

        # 加权并集：取两个法规中该主题频次的较大值
        all_topics = set(query_topics.keys()) | set(candidate_topics.keys())
        weighted_union = sum(
            max(query_topics.get(t, 0), candidate_topics.get(t, 0)) for t in all_topics
        )

        if weighted_union == 0:
            return 0.0

        return weighted_intersection / weighted_union

    # ===== 向量计算 =====

    def _compute_median_vector(self, chunks: List[Dict]) -> Optional[List[float]]:
        """计算法规所有 chunk 向量的中位数向量（比均值更鲁棒，不受极端值影响）。"""
        if not chunks:
            return None

        vectors = []
        for chunk in chunks:
            emb = chunk.get("embedding", [])
            if emb and len(emb) > 0:
                vectors.append(emb)

        if not vectors:
            return None

        arr = np.array(vectors, dtype=np.float32)
        median_vec = np.median(arr, axis=0)

        # L2 归一化
        norm = np.linalg.norm(median_vec)
        if norm > 0:
            median_vec = median_vec / norm

        return median_vec.tolist()

    def _compute_representative_vector(self, chunks: List[Dict]) -> Optional[List[float]]:
        """法规代表性向量：按 chunk_index 均匀采样（最多 20 个，避免长法规主导）后取中位数。"""
        if not chunks:
            return None

        sample_size = min(20, len(chunks))
        if sample_size >= len(chunks):
            sampled = chunks
        else:
            indices = np.linspace(0, len(chunks) - 1, sample_size, dtype=int)
            sampled = [chunks[i] for i in indices]

        return self._compute_median_vector(sampled)

    # ===== 批量获取主题 =====

    def _batch_get_law_topics(self, law_ids: List[str]) -> Dict[str, Dict[str, int]]:
        """批量获取多个法规的主题分布，单次 ES 聚合查询（不拉 embedding/正文），避免 N+1。"""
        if not law_ids:
            return {}
        try:
            return self.repo.batch_get_law_topic_distributions(law_ids)
        except Exception:
            return {law_id: {} for law_id in law_ids}

    # ===== 质心预计算 =====

    def compute_law_centroid(self, law_id: str) -> Optional[Dict]:
        """复现 median-of-sampled-20 逻辑，计算法规质心及元数据（供落库）。

        先仅投影 chunk_index 定位采样位、再回取 ≤20 条 chunk 的 embedding，避免传输整部法规向量。
        返回 None 表示该法规无有效向量数据。
        """
        indices = self.repo.get_law_chunk_indices(law_id)
        if not indices:
            return None

        n = len(indices)
        sample_size = min(20, n)
        if sample_size < n:
            positions = np.linspace(0, n - 1, sample_size, dtype=int)
            needed = sorted({int(indices[p]) for p in positions})
        else:
            needed = sorted(int(i) for i in indices)

        chunks = self.repo.get_law_chunks_by_indices(law_id, needed)
        if not chunks:
            return None

        vectors = [c["embedding"] for c in chunks if c.get("embedding")]
        if not vectors:
            return None

        arr = np.array(vectors, dtype=np.float32)
        median_vec = np.median(arr, axis=0)
        norm = np.linalg.norm(median_vec)
        if norm > 0:
            median_vec = median_vec / norm

        topic_dist = self.repo.batch_get_law_topic_distributions([law_id]).get(law_id, {})

        return {
            "centroid": median_vec.tolist(),
            "jurisdiction": chunks[0].get("jurisdiction", ""),
            "title": chunks[0].get("title", ""),
            "chunk_count": n,
            "topic_distribution": topic_dist,
        }

    def build_and_save_centroid(self, law_id: str) -> bool:
        """计算并落库指定法规质心（失败静默返回 False，不影响主流程）"""
        try:
            data = self.compute_law_centroid(law_id)
            if not data:
                return False
            return self.repo.save_law_centroid(law_id, **data)
        except Exception:
            return False

    # ===== 核心方法 =====

    def get_similar_laws(
        self,
        query_law_id: str,
        top_k: int = 5,
        jurisdiction_filter: Optional[str] = None,
        preferred_jurisdictions: Optional[List[str]] = None,
    ) -> Dict:
        """核心方法：根据 law_id 查找最相似的 top_k 部法规（主题感知版）。

        优先用预计算质心，缺失时降级为运行时计算代表向量；KNN 多取候选后按主题硬过滤，
        以 主题重叠度 60% + 归一化向量相似度 40% 综合评分，最后做跨法域多样性选择。

        preferred_jurisdictions：需要优先占位的法域（调用方已从问题中点名）。
        多样性选择是"每个法域只取排名第一"，因此当候选法域数 ≥ top_k 时，未点名的
        法域会按评分把点名法域挤出结果集（新增法域入库后尤其明显），故这里显式前置。
        """
        # 1. 优先读取预计算质心；缺失时降级为运行时拉取整部法规向量
        centroid_doc = self.repo.get_law_centroid(query_law_id)
        rep_vector = None

        if centroid_doc and centroid_doc.get("centroid"):
            query_law_title = centroid_doc.get("title", "")
            query_jurisdiction = centroid_doc.get("jurisdiction", "")
            chunk_count = centroid_doc.get("chunk_count", 0)
            query_topic_dist = centroid_doc.get("topic_distribution", {}) or {}
            rep_vector = centroid_doc.get("centroid")
        else:
            chunks = self.repo.get_law_chunk_vectors(query_law_id)
            if not chunks:
                return {
                    "query_law_id": query_law_id,
                    "similar_laws": [],
                    "total_found": 0,
                    "message": "未找到该法规的向量数据"
                }
            query_law_title = chunks[0].get("title", "")
            query_jurisdiction = chunks[0].get("jurisdiction", "")
            chunk_count = len(chunks)
            query_topic_dist = self._analyze_law_topics(chunks)
            rep_vector = self._compute_representative_vector(chunks)

        if rep_vector is None:
            return {
                "query_law_id": query_law_id,
                "similar_laws": [],
                "total_found": 0,
                "message": "无法计算法规的代表性向量"
            }

        # 2. 主题集合；无主题信息时降级为纯向量搜索
        query_topics_set = set(query_topic_dist.keys())
        if not query_topics_set:
            return self._fallback_vector_search(
                rep_vector, query_law_id, query_law_title, query_jurisdiction,
                chunk_count, top_k, jurisdiction_filter, preferred_jurisdictions
            )

        # 3. KNN 搜索候选法规（多取一些用于后续过滤）
        raw_results = self.repo.knn_search_similar(
            query_vector=rep_vector,
            top_k=top_k * 8,  # 多取，因为后续要过滤
            exclude_law_id=query_law_id,
            jurisdiction_filter=jurisdiction_filter,
            min_score=0.2  # 降低阈值，让主题过滤来做主要筛选
        )

        if not raw_results:
            return {
                "query_law_id": query_law_id,
                "similar_laws": [],
                "total_found": 0,
                "message": "未找到相似的法规"
            }

        # 4. 按 law_id 聚合，获取每个候选法规的最高相似度
        law_max_sim: Dict[str, float] = {}
        law_chunks: Dict[str, List[Dict]] = {}

        for item in raw_results:
            law_id = item["law_id"]
            sim = item["similarity"]

            if law_id not in law_max_sim:
                law_max_sim[law_id] = sim
                law_chunks[law_id] = []

            if sim > law_max_sim[law_id]:
                law_max_sim[law_id] = sim

            law_chunks[law_id].append({
                "article_number": item["article_number"],
                "content_preview": item["content_preview"],
                "similarity": sim
            })

        # 5. 批量获取候选法规的主题分布
        candidate_law_ids = list(law_max_sim.keys())
        candidate_topics = self._batch_get_law_topics(candidate_law_ids)

        # 6. 主题硬过滤 + 综合评分
        scored_laws = []
        for law_id, max_sim in law_max_sim.items():
            cand_topics = candidate_topics.get(law_id, {})
            cand_topics_set = set(cand_topics.keys())

            # 主题硬过滤：必须有至少一个主题重叠
            if not (query_topics_set & cand_topics_set):
                continue

            # 计算主题重叠度
            topic_overlap = self._compute_topic_overlap(query_topic_dist, cand_topics)

            # 综合评分：主题重叠度 60% + 向量相似度 40%
            # 向量相似度归一化到 0~1（假设 max_score 在 0.2~1.0 之间）
            normalized_sim = (max_sim - 0.2) / 0.8 if max_sim > 0.2 else 0
            normalized_sim = max(0, min(1, normalized_sim))

            composite_score = 0.6 * topic_overlap + 0.4 * normalized_sim

            # 获取候选法规的标题和法域（从 raw_results 中取）
            title = ""
            jurisdiction = ""
            for item in raw_results:
                if item["law_id"] == law_id:
                    title = item["title"]
                    jurisdiction = item["jurisdiction"]
                    break

            # 获取 top 匹配条款
            top_chunks = sorted(
                law_chunks[law_id], key=lambda x: x["similarity"], reverse=True
            )[:3]

            scored_laws.append({
                "law_id": law_id,
                "title": title,
                "jurisdiction": jurisdiction,
                "similarity": round(max_sim, 4),
                "topic_overlap": round(topic_overlap, 4),
                "composite_score": round(composite_score, 4),
                "match_chunk_count": len(law_chunks[law_id]),
                "top_matched_chunks": top_chunks,
                "shared_topics": list(query_topics_set & cand_topics_set)
            })

        # 7. 按综合评分排序
        scored_laws.sort(key=lambda x: x["composite_score"], reverse=True)

        # 8. 跨法域多样性选择
        selected = self._select_diverse_results(
            scored_laws, top_k, query_jurisdiction, preferred_jurisdictions
        )

        # 9. 构建返回结果
        similar_laws = []
        for entry in selected:
            similar_laws.append({
                "law_id": entry["law_id"],
                "title": entry["title"],
                "jurisdiction": entry["jurisdiction"],
                "similarity": entry["similarity"],
                "composite_score": entry["composite_score"],
                "topic_overlap": entry["topic_overlap"],
                "match_chunk_count": entry["match_chunk_count"],
                "shared_topics": entry["shared_topics"],
                "top_matched_chunks": entry["top_matched_chunks"]
            })

        return {
            "query_law_id": query_law_id,
            "query_law_title": query_law_title,
            "query_jurisdiction": query_jurisdiction,
            "query_chunk_count": chunk_count,
            "query_topics": list(query_topics_set),
            "similar_laws": similar_laws,
            "total_found": len(scored_laws),
            "returned_count": len(similar_laws)
        }

    def _fallback_vector_search(
        self,
        rep_vector: Optional[List[float]],
        query_law_id: str,
        query_law_title: str,
        query_jurisdiction: str,
        chunk_count: int,
        top_k: int,
        jurisdiction_filter: Optional[str],
        preferred_jurisdictions: Optional[List[str]] = None,
    ) -> Dict:
        """降级方案：查询法规无主题信息时使用纯向量搜索。"""
        if rep_vector is None:
            return {
                "query_law_id": query_law_id,
                "similar_laws": [],
                "total_found": 0,
                "message": "无法计算法规的代表性向量"
            }

        raw_results = self.repo.knn_search_similar(
            query_vector=rep_vector,
            top_k=top_k * 3,
            exclude_law_id=query_law_id,
            jurisdiction_filter=jurisdiction_filter,
            min_score=0.3
        )

        # 简单聚合
        law_map: Dict[str, Dict] = {}
        for item in raw_results:
            law_id = item["law_id"]
            if law_id not in law_map:
                law_map[law_id] = {
                    "law_id": law_id,
                    "title": item["title"],
                    "jurisdiction": item["jurisdiction"],
                    "max_similarity": item["similarity"],
                    "match_count": 1,
                    "matched_chunks": [{
                        "article_number": item["article_number"],
                        "content_preview": item["content_preview"],
                        "similarity": item["similarity"]
                    }]
                }
            else:
                entry = law_map[law_id]
                entry["match_count"] += 1
                if item["similarity"] > entry["max_similarity"]:
                    entry["max_similarity"] = item["similarity"]
                entry["matched_chunks"].append({
                    "article_number": item["article_number"],
                    "content_preview": item["content_preview"],
                    "similarity": item["similarity"]
                })

        aggregated = sorted(law_map.values(), key=lambda x: x["max_similarity"], reverse=True)
        # 与主路径口径一致：点名法域先各占一席，其余按相似度补满
        preferred = set(preferred_jurisdictions or [])
        if preferred:
            head, seen, rest = [], set(), []
            for entry in aggregated:
                if entry["jurisdiction"] in preferred and entry["jurisdiction"] not in seen:
                    seen.add(entry["jurisdiction"])
                    head.append(entry)
                else:
                    rest.append(entry)
            aggregated = head + rest
        aggregated = aggregated[:top_k]

        similar_laws = []
        for entry in aggregated:
            top_chunks = sorted(entry["matched_chunks"], key=lambda x: x["similarity"], reverse=True)[:3]
            similar_laws.append({
                "law_id": entry["law_id"],
                "title": entry["title"],
                "jurisdiction": entry["jurisdiction"],
                "similarity": round(entry["max_similarity"], 4),
                "composite_score": round(entry["max_similarity"], 4),
                "topic_overlap": 0.0,
                "match_chunk_count": entry["match_count"],
                "shared_topics": [],
                "top_matched_chunks": top_chunks
            })

        return {
            "query_law_id": query_law_id,
            "query_law_title": query_law_title,
            "query_jurisdiction": query_jurisdiction,
            "query_chunk_count": chunk_count,
            "query_topics": [],
            "similar_laws": similar_laws,
            "total_found": len(aggregated),
            "returned_count": len(similar_laws),
            "message": "（降级模式：查询法规无主题信息，使用纯向量搜索）"
        }

    # ===== 跨法域多样性 =====

    def _select_diverse_results(self, scored_laws: List[Dict], top_k: int,
                                query_jurisdiction: str,
                                preferred_jurisdictions: Optional[List[str]] = None) -> List[Dict]:
        """从评分结果中选 top_k：先每法域取第一（优先非查询法域），再按评分补满。

        preferred_jurisdictions 非空时，这些法域排在最前占位：多样性选择每个法域
        只取一条，若候选法域数 ≥ top_k，未点名的法域会按评分先占满名额，
        点名法域（如问题里的"澳门"）就再也进不来。
        """
        if not scored_laws:
            return []

        # 按法域分组
        by_jurisdiction: Dict[str, List[Dict]] = {}
        for entry in scored_laws:
            jur = entry["jurisdiction"]
            if jur not in by_jurisdiction:
                by_jurisdiction[jur] = []
            by_jurisdiction[jur].append(entry)

        # 第一阶段：每个法域取排名第一的（保证多样性）
        selected = []
        used_ids = set()

        # 法域顺序：点名的 → 其他非查询法域 → 查询法域自身
        preferred = set(preferred_jurisdictions or [])
        other_jurs = [j for j in by_jurisdiction if j != query_jurisdiction]
        ordered_jurs = (
            [j for j in other_jurs if j in preferred]
            + [j for j in other_jurs if j not in preferred]
            + [j for j in by_jurisdiction if j == query_jurisdiction]
        )

        for jur in ordered_jurs:
            candidates = by_jurisdiction[jur]
            if candidates and candidates[0]["law_id"] not in used_ids:
                selected.append(candidates[0])
                used_ids.add(candidates[0]["law_id"])
            if len(selected) >= top_k:
                break

        # 第二阶段：填充剩余位置
        if len(selected) < top_k:
            remaining = [e for e in scored_laws if e["law_id"] not in used_ids]
            for entry in remaining:
                if len(selected) >= top_k:
                    break
                selected.append(entry)
                used_ids.add(entry["law_id"])

        return selected[:top_k]

    # ===== 单条 chunk 相似推荐 =====

    def get_similar_chunks(
        self,
        query_chunk_id: str,
        top_k: int = 10,
        jurisdiction_filter: Optional[str] = None
    ) -> Dict:
        """基于单条 chunk 的相似条款推荐（向量 KNN，排除同法规）。"""
        doc = self.repo.get(query_chunk_id)
        if not doc:
            return {
                "query_chunk_id": query_chunk_id,
                "similar_chunks": [],
                "message": "未找到该条款"
            }

        query_vector = doc.get("embedding")
        if not query_vector:
            return {
                "query_chunk_id": query_chunk_id,
                "similar_chunks": [],
                "message": "该条款没有向量数据"
            }

        query_law_id = doc.get("law_id", "")

        raw_results = self.repo.knn_search_similar(
            query_vector=query_vector,
            top_k=top_k,
            exclude_law_id=query_law_id,
            jurisdiction_filter=jurisdiction_filter,
            min_score=0.2
        )

        similar_chunks = []
        for item in raw_results[:top_k]:
            similar_chunks.append({
                "chunk_id": item["id"],
                "law_id": item["law_id"],
                "title": item["title"],
                "jurisdiction": item["jurisdiction"],
                "article_number": item["article_number"],
                "content_preview": item["content_preview"],
                "similarity": round(item["similarity"], 4)
            })

        return {
            "query_chunk_id": query_chunk_id,
            "query_content": doc.get("content", "")[:200],
            "query_article_number": doc.get("article_number", ""),
            "query_law_id": query_law_id,
            "similar_chunks": similar_chunks,
            "total_found": len(raw_results),
            "returned_count": len(similar_chunks)
        }
