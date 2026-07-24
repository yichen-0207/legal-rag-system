from typing import List, Dict, Optional
import time
import re
import hashlib
import json
from collections import defaultdict, Counter

from repositories.elasticsearch import ElasticsearchRepository


class CompareService:
    """
    法规对比服务

    核心能力：分别检索两个法域的条款 → 聚合为法规级 → 结构化表格展示 + 自动总结 + 条款对照 + 关键词提取。
    与专题分析的区别：专题分析调用LLM生成深度报告；法规对比以检索+展示为主，可选LLM增强。
    """

    def __init__(self):
        self.repo = ElasticsearchRepository()

    # ===== 中文主题 → 英文检索关键词映射（提升新加坡/英文法域检索准确率）=====
    _TOPIC_EN_QUERIES: Dict[str, List[str]] = {
        "数据保护": ["data protection", "personal data", "privacy", "sensitive data"],
        "网络安全": ["cybersecurity", "critical infrastructure", "network security", "incident"],
        "人工智能": ["artificial intelligence", "AI governance", "automated decision", "algorithm"],
        "金融监管": ["financial regulation", "banking supervision", "prudential", "capital adequacy"],
        "跨境贸易": ["cross-border trade", "customs", "import export", "trade facilitation"],
        "知识产权": ["intellectual property", "patent", "copyright", "trademark", "IP rights"],
    }

    def _get_search_queries(self, topic: str, jurisdiction: str) -> List[str]:
        """根据法域语言特性选择检索词：新加坡等英文法域使用英文查询"""
        en_jurisdictions = {"新加坡", "香港", "Singapore", "Hong Kong"}
        if jurisdiction in en_jurisdictions:
            for cn_key, en_queries in self._TOPIC_EN_QUERIES.items():
                if cn_key in topic:
                    return en_queries
            if re.search(r'[a-zA-Z]', topic):
                return [topic]
            return [topic] + self._translate_topic_to_en(topic)
        return [topic]

    @staticmethod
    def _translate_topic_to_en(topic: str) -> List[str]:
        """简单的中文主题→英文关键词启发式映射（覆盖常见法律领域）"""
        mapping = {
            "数据": ["data"], "保护": ["protection"], "隐私": ["privacy"], "安全": ["security"],
            "网络": ["cyber", "network"], "跨境": ["cross-border"], "传输": ["transfer"],
            "金融": ["financial"], "银行": ["banking"], "洗钱": ["money laundering"],
            "消费者": ["consumer"], "电信": ["telecom"], "通讯":["communication"],
            "电子": ["electronic"], "签名": ["signature"], "犯罪": ["crime", "offense"],
            "知识产权": ["intellectual property"], "监管": ["regulation", "supervision"],
            "贸易": ["trade"], "人工智能": ["AI", "artificial intelligence"],
        }
        en_parts = []
        for cn, en_list in mapping.items():
            if cn in topic:
                en_parts.extend(en_list)
        return list(set(en_parts))[:5]

    # ================================================================
    #  缓存指纹机制（与专题分析一致）
    # ================================================================

    def _compute_fingerprint(self, topic: str, jurisdiction_a: str, jurisdiction_b: str) -> str:
        """
        生成对比分析的综合指纹（MD5）。

        指纹仅基于稳定输入参数（topic + jurisdictions + 检索关键词），
        不依赖 ES 检索结果（向量检索每次返回的列表可能不同）。
        """
        queries_a = self._get_search_queries(topic, jurisdiction_a)
        queries_b = self._get_search_queries(topic, jurisdiction_b)
        raw = json.dumps({
            "topic": topic,
            "jurisdiction_a": jurisdiction_a,
            "jurisdiction_b": jurisdiction_b,
            "queries_a": queries_a,
            "queries_b": queries_b,
        }, sort_keys=True, ensure_ascii=False)
        fp = hashlib.md5(raw.encode("utf-8")).hexdigest()
        logger.info(f"[Compare-FP] computed={fp[:16]}..., topic={topic}, {jurisdiction_a} vs {jurisdiction_b}")
        return fp

    def _try_get_cached(self, topic: str, jur_a: str, jur_b: str, current_fp: str) -> Optional[Dict]:
        """尝试从缓存获取有效结果，返回 report_data 或 None"""
        cached = self.repo.get_analysis_cache(topic, jur_a, jur_b, cache_type="compare")
        if not cached:
            return None
        if not isinstance(cached, dict):
            logger.warning(f"[Compare-Cache] 缓存数据格式异常（非dict），跳过")
            return None
        cached_fp = cached.get("fingerprint", "")
        report_data = cached.get("report_data")
        if not isinstance(report_data, dict):
            logger.warning(f"[Compare-Cache] report_data 格式异常，跳过")
            return None
        if cached_fp != current_fp:
            logger.info(f"[Compare-Cache] 指纹不匹配，失效。cached={cached_fp[:16]}... current={current_fp[:16]}...")
            return None
        logger.info(f"[Compare-Cache] 命中！topic={topic}, {jur_a} vs {jur_b}")

        # 反序列化 topic 字段（写入时可能转为JSON字符串）
        import json
        if isinstance(report_data, dict):
            topic_raw = report_data.get("topic")
            if isinstance(topic_raw, str):
                try:
                    report_data["topic"] = json.loads(topic_raw)
                except (json.JSONDecodeError, TypeError):
                    pass

        return report_data

    def _search_and_aggregate(
        self,
        topic: str,
        jurisdiction: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        检索指定法域的法规并聚合为法规级别结果。

        Args:
            topic: 搜索主题
            jurisdiction: 法域名称
            top_k: 每个法域返回的条款数上限
        Returns:
            聚合后的法规列表，每个元素包含：
              - law_id, title, article_numbers (list), chunk_count
              - passing_date, keywords, chunks_preview (前5条摘要)
              - max_similarity, all_chunks (完整列表用于前端渲染)
        """
        start_time = time.time()

        # 根据法域选择合适的检索词（新加坡用英文，澳门用中文）
        queries = self._get_search_queries(topic, jurisdiction)
        raw = self.repo.query(queries, n_results=top_k, where={"jurisdiction": jurisdiction})

        # 将 ChromaDB 格式（{ids: [[...]], documents: [[...]], metadatas: [[...]]}）
        # 展平为 [{law_id, content, metadata, similarity, _id}, ...]
        results = []
        all_ids = raw.get("ids", [])
        all_docs = raw.get("documents", [])
        all_metas = raw.get("metadatas", [])
        all_dists = raw.get("distances", [])
        for q_idx in range(len(all_ids)):
            for i in range(len(all_ids[q_idx])):
                meta = all_metas[q_idx][i] if q_idx < len(all_metas) and i < len(all_metas[q_idx]) else {}
                dist = all_dists[q_idx][i] if q_idx < len(all_dists) and i < len(all_dists[q_idx]) else 1.0
                results.append({
                    "law_id": meta.get("law_id", ""),
                    "content": all_docs[q_idx][i] if q_idx < len(all_docs) and i < len(all_docs[q_idx]) else "",
                    "metadata": meta,
                    "similarity": round(1 - dist, 4),
                    "_id": all_ids[q_idx][i],
                })

        if not results:
            logger.info(f"[Compare] {jurisdiction} 未检索到任何结果（queries={queries}）")
            return []

        # 按法规聚合
        laws_map: Dict[str, dict] = defaultdict(lambda: {
            "title": "", "article_numbers": set(), "chunks": [],
            "passing_date": None, "all_content": ""
        })

        for r in results:
            lid = r.get("law_id", "")
            # 防御：metadata 可能是字符串（ES 某些场景下）
            meta = r.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            art_num = meta.get("article_number") or ""

            entry = laws_map[lid]
            if not entry["title"]:
                entry["title"] = meta.get("title", lid)
            if art_num:
                entry["article_numbers"].add(art_num)

            content = r.get("content", "")
            similarity = float(r.get("similarity", 0))
            entry["chunks"].append({
                "content": content,
                "similarity": similarity,
                "article_number": art_num,
                "law_id": lid,
                "law_title": entry["title"],
                "_id": r.get("_id", ""),
                "chunk_index": len(entry["chunks"]),
            })
            entry["all_content"] += " " + content
            pd = meta.get("passing_date")
            if pd and not entry["passing_date"]:
                entry["passing_date"] = pd

        aggregated = []
        for lid, info in laws_map.items():
            sorted_chunks = sorted(info["chunks"], key=lambda c: c["chunk_index"])
            article_list = sorted(
                [a for a in info["article_numbers"] if a],
                key=lambda x: x.lstrip("第").rstrip("条")[:5] if x else ""
            )

            keywords = self._extract_keywords(info["all_content"])

            aggregated.append({
                "law_id": lid,
                "title": info["title"],
                "article_numbers": article_list,
                "chunk_count": len(sorted_chunks),
                "article_count": len(article_list),
                "passing_date": info["passing_date"],
                "keywords": keywords,
                "chunks_preview": [
                    {"content": c["content"][:300], "similarity": c["similarity"],
                     "article_number": c.get("article_number", "")}
                    for c in sorted_chunks[:5]
                ],
                "max_similarity": max(c["similarity"] for c in sorted_chunks) if sorted_chunks else 0,
                "all_chunks": [
                    {"content": c["content"], "similarity": c["similarity"],
                     "article_number": c.get("article_number", ""), "_id": c["_id"]}
                    for c in sorted_chunks
                ]
            })

        elapsed = time.time() - start_time
        logger.info(f"[Compare] {jurisdiction}: 命中 {len(aggregated)} 部法规, "
                     f"共 {sum(a['chunk_count'] for a in aggregated)} 条条款 ({elapsed:.2f}s)")
        return aggregated

    @staticmethod
    def _extract_keywords(text: str, top_n: int = 5) -> List[str]:
        """从文本中提取高频实词作为关键词"""
        # 简单分词 + 词频统计（实际项目可替换为 jieba）
        words = re.findall(r'[\u4e00-\u9fff]{2,}|[\w]{3,}', text.lower())
        stop_words = {
            'the', 'and', 'for', 'that', 'this', 'with', 'are', 'was', 'has',
            'been', 'have', 'from', 'they', 'will', 'would', 'could', 'should',
            'may', 'can', 'shall', 'must', 'which', 'their', 'each', 'other',
            'into', 'than', 'them', 'being', 'some', 'such', 'only', 'also',
            'data', 'personal', 'information', 'protection', 'person'
        }
        filtered = [w for w in words if w not in stop_words and len(w) >= 2]
        counter = Counter(filtered)
        return [word for word, count in counter.most_common(top_n)]

    def _build_stats(
        self,
        results_a: List[Dict],
        results_b: List[Dict],
        jurisdiction_a: str,
        jurisdiction_b: str,
    ) -> Dict:
        """
        构建对比指标统计。

        返回结构与前端的 renderCompareStats / renderKeywordsComparison 对齐：
        {
            "total_chunks": {jurA: int, jurB: int},
            "law_counts": {jurA: int, jurB: int},
            "best_similarity": {jurA: float, jurB: float},
            "avg_similarity": {jurA: float, jurB: float},
            "top_keywords": {jurA: list, jurB: list},
        }
        """
        all_chunks_a = [c for r in results_a for c in r.get("all_chunks", [])]
        all_chunks_b = [c for r in results_b for c in r.get("all_chunks", [])]

        article_count_a = sum(r.get("article_count", 0) for r in results_a)
        article_count_b = sum(r.get("article_count", 0) for r in results_b)

        sims_a = [float(c.get("similarity", 0)) for c in all_chunks_a]
        sims_b = [float(c.get("similarity", 0)) for c in all_chunks_b]

        def _avg(sims: List[float]) -> float:
            return round(sum(sims) / len(sims), 4) if sims else 0.0

        def _merge_keywords(results: List[Dict]) -> List[str]:
            merged = []
            for r in results:
                merged.extend(r.get("keywords", []))
            counter = Counter(merged)
            return [word for word, _ in counter.most_common(10)]

        return {
            "total_chunks": {
                jurisdiction_a: len(all_chunks_a),
                jurisdiction_b: len(all_chunks_b),
            },
            "total_articles": {
                jurisdiction_a: article_count_a,
                jurisdiction_b: article_count_b,
            },
            "law_counts": {
                jurisdiction_a: len(results_a),
                jurisdiction_b: len(results_b),
            },
            "best_similarity": {
                jurisdiction_a: round(max(sims_a), 4) if sims_a else 0.0,
                jurisdiction_b: round(max(sims_b), 4) if sims_b else 0.0,
            },
            "avg_similarity": {
                jurisdiction_a: _avg(sims_a),
                jurisdiction_b: _avg(sims_b),
            },
            "top_keywords": {
                jurisdiction_a: _merge_keywords(results_a),
                jurisdiction_b: _merge_keywords(results_b),
            },
        }

    def _build_auto_summary(
        self,
        results_a: List[Dict],
        results_b: List[Dict],
        jurisdiction_a: str,
        jurisdiction_b: str,
        topic: str,
    ) -> str:
        """基于检索结果生成自动对比总结"""
        if not results_a and not results_b:
            return f"未检索到{jurisdiction_a}与{jurisdiction_b}关于「{topic}」的相关法规条款。"

        law_titles_a = [f"《{r['title']}》" for r in results_a[:3] if r.get("title")]
        law_titles_b = [f"《{r['title']}》" for r in results_b[:3] if r.get("title")]

        parts = [f"本次围绕「{topic}」对比了{jurisdiction_a}与{jurisdiction_b}的相关法规。"]

        if law_titles_a:
            parts.append(
                f"{jurisdiction_a}主要涉及{'、'.join(law_titles_a)}等"
                f"{len(results_a)}部法规，命中{sum(r['chunk_count'] for r in results_a)}条相关条款。"
            )
        else:
            parts.append(f"{jurisdiction_a}未检索到相关法规。")

        if law_titles_b:
            parts.append(
                f"{jurisdiction_b}主要涉及{'、'.join(law_titles_b)}等"
                f"{len(results_b)}部法规，命中{sum(r['chunk_count'] for r in results_b)}条相关条款。"
            )
        else:
            parts.append(f"{jurisdiction_b}未检索到相关法规。")

        if results_a and results_b:
            parts.append(
                "两法域在立法侧重点、监管机构和处罚机制等方面存在差异，"
                "详见下方逐维度对比表与条款对照。"
            )
        elif results_a or results_b:
            parts.append("仅一方检索到相关法规，建议扩大检索范围或调整主题词后重试。")

        return "\n".join(parts)

    def compare(
        self,
        topic: str,
        jurisdiction_a: str,
        jurisdiction_b: str,
        top_k_per_jurisdiction: int = 10,
    ) -> Dict:
        """
        执行双法域对比分析（带缓存）。

        流程：检索 → 计算指纹 → 查缓存（命中直接返回）→ 未命中则构建结果并写入缓存

        Returns:
            {
                "topic": str,
                "jurisdictions": [str, str],
                "results_a": [...],
                "results_b": [...],
                "elapsed_seconds": float,
                "_cache_hit": bool   # 缓存状态标记
            }
        """
        overall_start = time.time()

        # 步骤1：检索两个法域的法规（ES 检索，~1s）
        results_a = self._search_and_aggregate(topic, jurisdiction_a, top_k=top_k_per_jurisdiction)
        results_b = self._search_and_aggregate(topic, jurisdiction_b, top_k=top_k_per_jurisdiction)

        if not results_a and not results_b:
            elapsed = time.time() - overall_start
            logger.warning(f"[Compare] 双方均无结果，跳过缓存。耗时 {elapsed:.2f}s")
            return {
                "topic": topic,
                "jurisdictions": [jurisdiction_a, jurisdiction_b],
                "results_a": [], "results_b": [],
                "elapsed_seconds": round(elapsed, 2),
                "_cache_hit": False,
            }

        # 步骤1.5：计算指纹 + 检查缓存
        current_fingerprint = self._compute_fingerprint(topic, jurisdiction_a, jurisdiction_b)
        cached_result = self._try_get_cached(topic, jurisdiction_a, jurisdiction_b, current_fingerprint)
        if cached_result is not None:
            elapsed = time.time() - overall_start
            logger.info(f"[Compare] 缓存命中，返回缓存结果。耗时 {elapsed:.2f}s")
            # 兼容旧缓存：若缺少 stats/auto_summary 则补充计算
            if "stats" not in cached_result or "auto_summary" not in cached_result:
                cached_result["stats"] = self._build_stats(
                    cached_result.get("results_a", []),
                    cached_result.get("results_b", []),
                    jurisdiction_a,
                    jurisdiction_b,
                )
                cached_result["auto_summary"] = self._build_auto_summary(
                    cached_result.get("results_a", []),
                    cached_result.get("results_b", []),
                    jurisdiction_a,
                    jurisdiction_b,
                    topic,
                )
            return {
                **cached_result,
                "_cache_hit": True,
                "elapsed_seconds": round(elapsed, 2),
            }

        # 步骤2：缓存未命中 — 构建完整结果
        elapsed = time.time() - overall_start

        stats = self._build_stats(results_a, results_b, jurisdiction_a, jurisdiction_b)
        auto_summary = self._build_auto_summary(
            results_a, results_b, jurisdiction_a, jurisdiction_b, topic
        )

        resp = {
            "topic": topic,
            "jurisdictions": [jurisdiction_a, jurisdiction_b],
            "results_a": results_a,
            "results_b": results_b,
            "stats": stats,
            "auto_summary": auto_summary,
            "elapsed_seconds": round(elapsed, 2),
            "_cache_hit": False,
        }

        total_laws = len(results_a) + len(results_b)
        total_articles = sum(r['chunk_count'] for r in results_a) + sum(r['chunk_count'] for r in results_b)
        logger.info(f"[Compare] 对比完成(未命中): {jurisdiction_a}({len(results_a)}部/{sum(r['chunk_count'] for r in results_a)}条) "
                     f"vs {jurisdiction_b}({len(results_b)}部/{sum(r['chunk_count'] for r in results_b)}条), "
                     f"耗时 {elapsed:.2f}s")

        # 步骤3：写入缓存（异步失败不影响结果）
        try:
            self.repo.save_analysis_cache(
                topic, jurisdiction_a, jurisdiction_b,
                current_fingerprint, resp, cache_type="compare"
            )
        except Exception as e:
            print(f"[Compare-Cache] 写入缓存失败（不影响结果）: {e}")

        return resp


# 模块级 logger（避免循环导入）
import logging
logger = logging.getLogger(__name__)
