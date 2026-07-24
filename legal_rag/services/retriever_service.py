import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from repositories.elasticsearch import ElasticsearchRepository, _jurisdiction_to_zh
from core.config import settings


class RetrieverService:
    def __init__(self):
        self.repo = ElasticsearchRepository()
        self._reranker = None

    def _get_reranker(self):
        """延迟获取 reranker 实例"""
        if not settings.reranker_enabled:
            return None
        if self._reranker is None:
            from services.reranker_service import ReRankerService
            self._reranker = ReRankerService()
        return self._reranker

    def _deduplicate_and_enrich_articles(self, raw_results: Dict, top_k: int) -> Dict:
        """
        对原始检索结果按 article 去重，保留每个 article 相似度最高的 sub_chunk，
        并回取完整 article 文本替换 sub_chunk 内容。
        """
        if not raw_results.get('ids') or not raw_results['ids'][0]:
            return raw_results

        seen = set()
        new_ids, new_docs, new_metas, new_distances = [], [], [], []
        hybrid_scores = raw_results.get('_scores')
        new_scores = []

        for i in range(len(raw_results['ids'][0])):
            metadata = raw_results['metadatas'][0][i]
            law_id = metadata.get('law_id', '')
            article_number = metadata.get('article_number', '')
            key = (law_id, article_number)
            if key in seen:
                continue
            seen.add(key)

            # 回取完整 article 文本；失败则保留原始 sub_chunk
            full_text = self.repo.get_article_full_text(law_id, article_number)
            new_docs.append(full_text if full_text else raw_results['documents'][0][i])
            new_ids.append(raw_results['ids'][0][i])
            new_metas.append(metadata)
            new_distances.append(raw_results['distances'][0][i])
            if hybrid_scores and hybrid_scores[0] and i < len(hybrid_scores[0]):
                new_scores.append(hybrid_scores[0][i])

            if len(new_ids) >= top_k:
                break

        result = {
            'ids': [new_ids],
            'documents': [new_docs],
            'metadatas': [new_metas],
            'distances': [new_distances],
        }
        if hybrid_scores:
            result['_scores'] = [new_scores]
        return result

    def _detect_jurisdiction(self, query: str) -> Optional[str]:
        jurisdiction_triggers = {
            '澳门': ['澳门', 'macau', 'Macau', 'MACAU', '澳門', 'MO'],
            '香港': ['香港', 'hong kong', 'hongkong', 'Hong Kong', 'Hongkong', 'HONG KONG', 'HK'],
            '新加坡': ['新加坡', 'singapore', 'Singapore', 'SINGAPORE', 'sg', 'SG']
        }

        for jurisdiction, triggers in jurisdiction_triggers.items():
            for trigger in triggers:
                if trigger in query:
                    return jurisdiction
        return None

    def _extract_law_name(self, query: str) -> Optional[str]:
        """从查询中提取法规名称
        
        仅提取原始名称，不进行跨语言映射（跨语言匹配由前端通过搜索结果频次分析处理）。
        
        支持:
        1. 含"第X条": "澳门打击电脑犯罪法的第2条" → "打击电脑犯罪法"
        2. 中文: "澳门网络安全法的定义" → "网络安全法"
        3. 英文: "新加坡 Personal Data Protection Act 2012" → "Personal Data Protection Act 2012"
        4. 其他语言: "日本個人情報保護法" → "個人情報保護法"
        """
        juris_triggers = ['澳门', '澳門', 'macau', 'Macau', 'MACAU',
                          '香港', 'Hong Kong', 'hongkong', 'Hongkong', 'HONG KONG',
                          '新加坡', 'Singapore', 'SINGAPORE']
        
        # 模式1: 含"第X条"
        article_match = re.search(r'第(\d+)[条條]', query)
        if article_match:
            for trigger in juris_triggers:
                juris_pos = query.find(trigger)
                if juris_pos != -1:
                    name = query[juris_pos + len(trigger):article_match.start()].strip('的 \t')
                    return name if name else None
        
        # 模式2: 提取法域后的法规名称（不限语言）
        for trigger in juris_triggers:
            juris_pos = query.find(trigger)
            if juris_pos != -1:
                after_juris = query[juris_pos + len(trigger):]
                
                # 中文法规名称
                cn_match = re.match(r'^[的\s]*([\u4e00-\u9fff]{2,20}(?:法|條例|条例|法规|規則|规则|法令|令|守则|守則|制度|指引))(?=[的关于和与,，。．\s]|$)', after_juris)
                if cn_match:
                    return cn_match.group(1)
                
                # 英文法规名称
                en_match = re.match(r"^\s*((?:[A-Za-z][-'A-Za-z]+[\s-]){0,5}(?:Act|Ordinance|Rule|Regulation|Program|Directive|Policy|Standard|Order|Code)(?:\s\d{4})?)(?=[\s\u4e00-\u9fff,，。．;(]|$)", after_juris)
                if en_match:
                    return en_match.group(1).strip()
                
                # 日韩等其他语言：捕获连续的法规名称
                other_match = re.match(r'^[的\s]*([^\s,，。．;;(（]+)', after_juris)
                if other_match:
                    return other_match.group(1).strip()
        
        return None

    # ============================================================
    # 区域→法域路由映射表
    # ============================================================
    REGION_MAP = {
        "亚太地区": ["香港", "日本", "新加坡", "马来西亚", "泰国", "越南", "澳门"],
        "亚太": ["香港", "日本", "新加坡", "马来西亚", "泰国", "越南", "澳门"],
        "东南亚各国": ["新加坡", "马来西亚", "泰国", "越南"],
        "东南亚": ["新加坡", "马来西亚", "泰国", "越南"],
        "东亚": ["日本", "香港", "澳门"],
        "亚洲各国": ["日本", "新加坡", "马来西亚", "泰国", "越南", "香港", "澳门"],
        "亚洲主要国家": ["日本", "新加坡", "马来西亚", "泰国", "越南", "香港", "澳门"],
        "亚洲各司法管辖区": ["日本", "新加坡", "马来西亚", "泰国", "越南", "香港", "澳门"],
        "各国": ["香港", "日本", "新加坡", "马来西亚", "泰国", "越南", "澳门", "美国"],
        "不同法域": ["香港", "日本", "新加坡", "马来西亚", "泰国", "越南", "澳门", "美国"],
        "各司法管辖区": ["香港", "日本", "新加坡", "马来西亚", "泰国", "越南", "澳门", "美国"],
    }

    COMPARE_KEYWORDS = ["有何异同", "有何差异", "比较", "区别", "差异", "异同", "不同模式", "versus", "vs"]

    # ============================================================
    # 中→英关键词扩展表（覆盖竞赛检索场景）
    # ============================================================
    ZH_EN_MAP = [
        # 数据保护与隐私
        (('数据保护', '个人数据', '隐私', '个人信息', '个人资料'), 'personal data protection privacy GDPR'),
        # 网络安全
        (('网络安全', '网络', '信息安全', '网络犯罪', '黑客'), 'cybersecurity network security information security cybercrime'),
        # 电子交易与签名
        (('电子', '数字化', '数字签名', '电子签名', '电子交易', '电子政务'), 'electronic digital signature e-commerce e-government'),
        # 电信与通讯
        (('电信', '通讯', '通信', '电话营销', '电话'), 'telecommunications communication telemarketing telephone'),
        # 金融监管
        (('金融', '银行', '保险', '证券', '金融科技'), 'financial banking insurance securities fintech'),
        # 反洗钱
        (('反洗钱', '反恐融资', '反恐'), 'anti-money laundering AML counter-terrorism financing CTF'),
        # 消费者保护
        (('消费者', '消费者保护', '消费', '消费者权益'), 'consumer protection consumer rights'),
        # 人工智能
        (('人工智能', 'AI', '机器学习'), 'artificial intelligence machine learning AI'),
        # 跨境数据流动
        (('跨境', '数据流动', '数据传输', '数据跨境'), 'cross-border data flow data transfer cross-border data'),
        # 牌照与监管
        (('牌照', '许可', '监管', '许可证', '牌照发放'), 'license licensing regulation permit'),
        # 数据泄露与通知
        (('通知', '通报', '披露', '数据泄露', '泄露'), 'notification disclosure breach notification data breach'),
        # 版权与知识产权
        (('版权', '著作权', '知识产权', '专利', '商标'), 'copyright intellectual property patent trademark'),
        # 公司与企业
        (('公司', '企业', '经营', '商业'), 'company corporate business enterprise commercial'),
        # 劳动与雇佣
        (('劳动', '雇佣', '劳工', '员工', '就业'), 'labor employment employee worker'),
        # 环境与环保
        (('环境', '环保', '排放', '污染'), 'environmental protection emission pollution'),
        # 医疗与药品
        (('药品', '医疗', '健康', '药物'), 'medical pharmaceutical healthcare health drug'),
        # 行政与政府
        (('行政', '政府', '官员', '公共'), 'administrative government official public'),
        # 刑事与处罚
        (('刑事', '犯罪', '处罚', '刑罚', '罚款'), 'criminal penalty offense punishment fine'),
        # 法律与法规
        (('法律', '法规', '条例', '法令', '法例'), 'law regulation act ordinance legislation'),
        # 电子认证
        (('电子签名', '电子认证', '认证'), 'electronic signature e-signature authentication'),
        # 金融科技（补充）
        (('金融科技', 'fintech', 'FinTech', 'FINTECH'), 'financial technology fintech'),
        # 版权（补充英文）
        (('版权', 'copyright'), 'copyright'),
        # 垃圾信息
        (('垃圾', '骚扰', '垃圾信息'), 'spam junk unsolicited'),
        # 广播与电视
        (('广播', '电视', '无线电', '频段'), 'broadcast television radio spectrum frequency'),
        # 数据保留
        (('数据保留', '数据保存', '留存'), 'data retention data preservation'),
    ]

    def _enhance_query(self, query: str) -> str:
        """查询增强：中→英关键词扩展 + 条款号提取"""
        extra_terms = []

        # 中→英关键词扩展
        for cn_keywords, en_keywords in self.ZH_EN_MAP:
            for cn_kw in cn_keywords:
                if cn_kw in query:
                    extra_terms.append(en_keywords)
                    break

        # 提取 "第X条/條" 中的数字和法规名称
        article_match = re.search(r'第(\d+)[条條]', query)
        if article_match:
            extra_terms.append(article_match.group(1))  # 条款号数字
            # 提取法域关键词和"第X条"之间的内容作为法规名称
            juris_triggers = ['澳门', '澳門', 'macau', 'Macau', 'MACAU',
                              '香港', 'Hong Kong', 'hongkong', 'Hongkong', 'HONG KONG',
                              '新加坡', 'Singapore', 'SINGAPORE',
                              '日本', 'Japan', 'JAPAN',
                              '美国', 'US', 'United States', 'UNITED STATES']
            for trigger in juris_triggers:
                juris_pos = query.find(trigger)
                if juris_pos != -1:
                    law_name = query[juris_pos + len(trigger):article_match.start()].strip('的 \t')
                    if law_name:
                        extra_terms.append(law_name)
                    break

        if extra_terms:
            return query + ' ' + ' '.join(extra_terms)
        return query

    # ============================================================
    # 跨法域路由模块
    # ============================================================
    def _route_to_jurisdictions(self, query: str) -> Optional[List[str]]:
        """
        检测查询涉及的法域列表。
        返回 None 表示不限法域（跨所有法域检索）。
        返回 List[str] 表示应仅在这些法域检索。
        """
        # 检查是否已包含具体法域名（如"美国电话营销"）
        named_jurs = []
        SPECIFIC = {
            "美国": "美国", "日本": "日本", "新加坡": "新加坡",
            "泰国": "泰国", "越南": "越南", "香港": "香港",
            "澳门": "澳门", "马来西亚": "马来西亚",
            "macau": "澳门", "macao": "澳门",
            "hong kong": "香港", "singapore": "新加坡",
            "japan": "日本", "thailand": "泰国",
            "vietnam": "越南", "malaysia": "马来西亚",
            "united states": "美国", "usa": "美国",
        }

        # 检测具体法域
        for keyword, jur in SPECIFIC.items():
            # 使用 word boundary 或子串匹配
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            if pattern.search(query):
                if jur not in named_jurs:
                    named_jurs.append(jur)

        # 如果明确指名了具体法域，先返回这些法域（不触发路由）
        # 注意：如果只有1个法域，走正常单法域检索
        # 如果多个法域被指名，走路由
        if len(named_jurs) >= 2:
            return named_jurs

        # 检查区域性关键词
        for region, jur_list in self.REGION_MAP.items():
            if region in query:
                return jur_list

        # 检查比较型关键词
        is_comparative = any(kw in query for kw in self.COMPARE_KEYWORDS)
        if is_comparative:
            # 比较型但无法确定区域 → 查所有法域（排除跨法域自身）
            return ["美国", "日本", "新加坡", "香港", "澳门", "马来西亚", "泰国", "越南"]

        return None

    def _rrf_merge_results(self, results_list: List[List[Dict]], rrf_k: int = 60) -> List[Dict]:
        """RRF 合并多组结果"""
        scores: Dict[str, float] = {}
        sources: Dict[str, Dict] = {}

        for results in results_list:
            for rank, doc in enumerate(results, start=1):
                doc_id = f"{doc['law_id']}_{doc.get('article_number', '')}"
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
                if doc_id not in sources:
                    sources[doc_id] = doc

        merged = []
        for doc_id, rrf_score in scores.items():
            doc = sources[doc_id]
            doc["rrf_score"] = rrf_score
            merged.append(doc)

        merged.sort(key=lambda x: x["rrf_score"], reverse=True)
        return merged

    def _hybrid_search_routed(self, query: str, jurisdictions: List[str],
                               top_k: int = 20) -> List[Dict]:
        """
        多法域路由检索：每个法域独立检索 Top-10，RRF 融合后精排
        """
        enhanced_query = self._enhance_query(query) or query
        all_per_jur_results = []

        for jur in jurisdictions:
            where_filter = {"jurisdiction": jur}
            raw = self.repo.hybrid_search(
                query_text=enhanced_query,
                n_results=10,  # 每个法域取 Top-10
                where=where_filter,
            )
            # 去重
            raw = self._deduplicate_and_enrich_articles(raw, 10)

            # 格式化
            formatted = []
            if raw.get('ids') and raw['ids'][0]:
                for i in range(len(raw['ids'][0])):
                    metadata = raw['metadatas'][0][i]
                    distance = raw['distances'][0][i]
                    similarity = 1 / (1 + distance)
                    formatted.append({
                        'content': raw['documents'][0][i],
                        'title': metadata.get('title', '') or '',
                        'article_number': metadata.get('article_number', '') or '',
                        'jurisdiction': _jurisdiction_to_zh(metadata.get('jurisdiction', '') or ''),
                        'score': similarity,
                        'law_id': metadata.get('law_id', '') or '',
                        'source_file': metadata.get('source_file', '') or '',
                        'chunk_index': metadata.get('chunk_index', 0) or 0,
                    })
            all_per_jur_results.append(formatted)

        # RRF 合并
        merged = self._rrf_merge_results(all_per_jur_results, rrf_k=60)

        # Re-ranker 精排（只排前 50，其余直接追加在尾部）
        reranker = self._get_reranker()
        if reranker:
            rerank_window = min(len(merged), 50)
            top_candidates = merged[:rerank_window]
            rest = merged[rerank_window:]
            top_candidates = reranker.rerank(query=query, candidates=top_candidates, top_k=top_k)
            merged = top_candidates + rest

        return merged[:top_k]

    def search(self, query: str, top_k: int = 5, jurisdiction: Optional[str] = None,
               topic: Optional[str] = None, topics: Optional[List[str]] = None) -> List[Dict]:
        """
        搜索法规内容

        参数:
            query: 搜索关键词
            top_k: 返回数量
            jurisdiction: 法域筛选
            topic: 单个主题（向后兼容）
            topics: 多个主题列表（新功能，优先使用）
        """
        detected_jurisdiction = jurisdiction or self._detect_jurisdiction(query)

        where_filter = {}
        if detected_jurisdiction:
            where_filter["jurisdiction"] = detected_jurisdiction

        # 主题筛选：支持多选（OR 关系：命中任一主题即返回）
        # 前端传中文标签（如"金融监管"），直接用 topic_labels 字段筛选
        active_topics = topics or ([topic] if topic else None)
        if active_topics and len(active_topics) > 0:
            if len(active_topics) == 1:
                where_filter["topic_labels"] = active_topics[0]
            else:
                where_filter["topic_labels"] = active_topics

        if query.strip():
            enhanced_query = self._enhance_query(query)
            # 扩大召回，确保按 article 去重后仍有足够结果
            results = self.repo.query([enhanced_query], n_results=max(top_k * 3, 20), where=where_filter)
        else:
            results = self.repo.query_with_filters(n_results=max(top_k * 3, 20), where=where_filter)

        # 候选窗口扩展：有 reranker 时保留更多候选供精排
        reranker_available = settings.reranker_enabled
        dedup_k = max(top_k * 3, 50) if reranker_available else top_k
        results = self._deduplicate_and_enrich_articles(results, dedup_k)

        formatted_results = []
        if results.get('ids') and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                similarity = 1 / (1 + distance)

                metadata = results['metadatas'][0][i]
                result = {
                    'content': results['documents'][0][i],
                    'title': metadata.get('title', '') or '',
                    'article_number': metadata.get('article_number', '') or '',
                    'jurisdiction': _jurisdiction_to_zh(metadata.get('jurisdiction', '') or ''),
                    'score': similarity,
                    'law_id': metadata.get('law_id', '') or '',
                    'source_file': metadata.get('source_file', '') or '',
                    'chunk_index': metadata.get('chunk_index', 0) or 0,
                    # 返回主题信息供前端展示
                    'topics': metadata.get('topics', []),
                    'topic_labels': metadata.get('topic_labels', [])
                }
                formatted_results.append(result)

        # 如果查询包含 "第X条"，尝试找到目标条款（无论是启用 re-ranker）
        article_match = re.search(r'第(\d+)[条條]', self._enhance_query(query))
        target_doc = None
        query_law_name = None
        target = None
        if article_match:
            target = article_match.group(1)
            query_law_name = self._extract_law_name(query)
            best_idx = None
            for idx, doc in enumerate(formatted_results):
                if doc.get("article_number", "").strip() != target:
                    continue
                if query_law_name:
                    title = doc.get("title", "")
                    if query_law_name in title or title in query_law_name:
                        best_idx = idx
                        break
                else:
                    best_idx = idx
                    break
            if best_idx is not None:
                target_doc = formatted_results.pop(best_idx)

            if not target_doc and query_law_name:
                found = self.repo.find_article(query_law_name, target)
                if found:
                    found["score"] = 1.0
                    target_doc = found

        # Re-ranker 精排（配置启用时自动执行，至多排 50 条）
        reranker = self._get_reranker()
        if reranker:
            rerank_window = min(max(top_k * 3, 50), 50)
            top_candidates = formatted_results[:rerank_window]
            rest = formatted_results[rerank_window:]
            top_candidates = reranker.rerank(
                query=query,
                candidates=top_candidates,
                top_k=top_k,
            )
            formatted_results = top_candidates + rest

        # 把目标条款插到最前面
        if target_doc:
            target_doc["score"] = 1.0
            formatted_results.insert(0, target_doc)

        return formatted_results

    def bm25_search(self, query: str, top_k: int = 20,
                     jurisdiction: Optional[str] = None) -> List[Dict]:
        """仅 BM25 全文检索"""
        from core.config import settings
        settings.reranker_enabled = False

        where_filter = {}
        if jurisdiction:
            where_filter["jurisdiction"] = jurisdiction

        enhanced_query = self._enhance_query(query)
        bm25_results = self.repo._bm25_search(enhanced_query, top_k, where_filter)

        formatted_results = []
        for hit in bm25_results:
            src = hit.get("_source", {})
            formatted_results.append({
                'content': src.get('content', ''),
                'title': src.get('title', '') or '',
                'article_number': src.get('article_number', '') or '',
                'jurisdiction': src.get('jurisdiction', '') or '',
                'score': hit.get('_score', 0),
                'law_id': src.get('law_id', '') or '',
                'source_file': src.get('source_file', '') or '',
                'chunk_index': src.get('chunk_index', 0) or 0,
            })
        return formatted_results

    def hybrid_search(self, query: str, top_k: int = 5,
                      jurisdiction: Optional[str] = None,
                      topic: Optional[str] = None,
                      topics: Optional[List[str]] = None) -> List[Dict]:
        """
        混合检索（向量 + BM25 + RRF 融合），返回格式与 search() 完全一致。

        参数与 search() 相同：
            query: 搜索关键词
            top_k: 返回数量
            jurisdiction: 法域筛选
            topic: 单个主题（向后兼容）
            topics: 多主题列表（优先使用）
        """
        enhanced_query = self._enhance_query(query)
        search_query = enhanced_query or query

        # 跨法域路由：仅当 jurisdiction 未显式指定时触发
        if jurisdiction is None:
            jurisdiction = self._detect_jurisdiction(query)
            if jurisdiction is None:
                # 尝试路由检测
                routed_jurs = self._route_to_jurisdictions(query)
                if routed_jurs and len(routed_jurs) > 1:
                    return self._hybrid_search_routed(
                        query=query, jurisdictions=routed_jurs, top_k=top_k,
                    )

        where_filter = {}
        if jurisdiction:
            where_filter["jurisdiction"] = jurisdiction

        active_topics = topics or ([topic] if topic else None)
        if active_topics and len(active_topics) > 0:
            if len(active_topics) == 1:
                where_filter["topic_labels"] = active_topics[0]
            else:
                where_filter["topic_labels"] = active_topics

        if search_query.strip():
            results = self.repo.hybrid_search(
                query_text=search_query,
                n_results=max(top_k * 3, 20),
                where=where_filter or None,
            )
        else:
            results = self.repo.query_with_filters(n_results=max(top_k * 3, 20), where=where_filter)

        # 候选窗口扩展：有 reranker 时保留更多候选供精排
        reranker_available = settings.reranker_enabled
        dedup_k = max(top_k * 3, 50) if reranker_available else top_k
        results = self._deduplicate_and_enrich_articles(results, dedup_k)

        formatted_results = []
        hybrid_scores = results.get("_scores", None)

        if results.get('ids') and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                # 排序用 RRF 分数（保证混合检索最佳排序），显示用余弦相似度（有意义的百分比）
                if hybrid_scores and hybrid_scores[0] and i < len(hybrid_scores[0]):
                    # 内部排序用的 RRF 分数（不回传给前端）
                    pass
                distance = results['distances'][0][i]
                similarity = 1 / (1 + distance)  # 余弦相似度，0~1 有意义的百分比

                metadata = results['metadatas'][0][i]
                result = {
                    'content': results['documents'][0][i],
                    'title': metadata.get('title', '') or '',
                    'article_number': metadata.get('article_number', '') or '',
                    'jurisdiction': _jurisdiction_to_zh(metadata.get('jurisdiction', '') or ''),
                    'score': similarity,
                    'law_id': metadata.get('law_id', '') or '',
                    'source_file': metadata.get('source_file', '') or '',
                    'chunk_index': metadata.get('chunk_index', 0) or 0,
                    'topics': metadata.get('topics', []),
                    'topic_labels': metadata.get('topic_labels', [])
                }
                formatted_results.append(result)

        # 如果查询包含 "第X条"，尝试找到目标条款（无论是启用 re-ranker）
        article_match = re.search(r'第(\d+)[条條]', search_query)
        target_doc = None
        query_law_name = None
        target = None
        if article_match:
            target = article_match.group(1)
            query_law_name = self._extract_law_name(search_query)
            best_idx = None
            for idx, doc in enumerate(formatted_results):
                if doc.get("article_number", "").strip() != target:
                    continue
                if query_law_name:
                    title = doc.get("title", "")
                    if query_law_name in title or title in query_law_name:
                        best_idx = idx
                        break
                else:
                    best_idx = idx
                    break
            if best_idx is not None:
                target_doc = formatted_results.pop(best_idx)

            if not target_doc and query_law_name:
                found = self.repo.find_article(query_law_name, target)
                if found:
                    found["score"] = 1.0
                    target_doc = found

        # Re-ranker 精排（配置启用时自动执行，至多排 50 条）
        reranker = self._get_reranker()
        if reranker:
            rerank_window = min(max(top_k * 3, 50), 50)
            top_candidates = formatted_results[:rerank_window]
            rest = formatted_results[rerank_window:]
            top_candidates = reranker.rerank(
                query=search_query,
                candidates=top_candidates,
                top_k=top_k,
            )
            formatted_results = top_candidates + rest

        # 把目标条款插到最前面
        if target_doc:
            target_doc["score"] = 1.0  # 确保前端按 score 排序时排第一
            formatted_results.insert(0, target_doc)

        return formatted_results
