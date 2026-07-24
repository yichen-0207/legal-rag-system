from typing import List, Dict, Optional, Any
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError, TransportError, ApiError
from core.config import settings
from core.model_loader import get_embedding_model
from utils.topic_classifier import TopicClassifier


_ZH_T2S_MAP = {
    '網': '网', '絡': '络', '個': '个', '資': '资', '訊': '讯', '護': '护',
    '產': '产', '電': '电', '腦': '脑', '擊': '击', '竊': '窃', '盜': '盗',
    '僞': '伪', '關': '关', '係': '系', '機': '机', '構': '构', '團': '团',
    '體': '体', '監': '监', '辦': '办', '廳': '厅', '處': '处', '員': '员',
    '職': '职', '區': '区', '域': '域', '內': '内', '務': '务', '險': '险',
    '報': '报', '舉': '举', '復': '复', '製': '制', '傳': '传', '輸': '输',
    '發': '发', '佈': '布', '獲': '获', '數': '数', '據': '据', '隱': '隐',
    '私': '私', '祕': '秘', '密': '密', '條': '条', '約': '约', '規': '规',
    '則': '则', '範': '范', '圍': '围', '標': '标', '準': '准', '審': '审',
    '計': '计', '訴': '诉', '訟': '讼', '罰': '罚', '執': '执', '實': '实',
    '施': '施', '細': '细', '總': '总', '附': '附', '錄': '录', '匯': '汇',
    '兌': '兑', '幣': '币', '貨': '货', '銀': '银', '證': '证', '券': '券',
    '債': '债', '權': '权', '務': '务', '業': '业', '廠': '厂', '號': '号',
    '碼': '码', '圖': '图', '視': '视', '頻': '频', '聲': '声', '軟': '软',
    '頁': '页', '連': '连', '鏈': '链', '結': '结', '載': '载', '郵': '邮',
    '即': '即', '時': '时', '討': '讨', '論': '论', '壇': '坛',
    '購': '购', '簽': '签', '署': '署', '訂': '订', '契': '契',
    '知': '知', '識': '识', '版': '版', '專': '专', '利': '利',
    '著': '著', '營': '营', '經': '经', '照': '照', '許': '许',
    '可': '可', '書': '书', '登': '登', '記': '记', '註': '注', '冊': '册',
    '備': '备', '案': '案', '開': '开', '創': '创', '新': '新', '研': '研',
    '究': '究', '試': '试', '驗': '验', '測': '测', '檢': '检', '查': '查',
    '鑑': '鉴', '定': '定', '評': '评', '估': '估', '核': '核', '准': '准',
    '認': '认', '責': '责', '任': '任', '義': '义', '利': '利', '益': '益',
    '損': '损', '害': '害', '賠': '赔', '償': '偿', '補': '补', '助': '助',
    '獎': '奖', '勵': '励', '懲': '惩', '戒': '戒', '紀': '纪', '律': '律',
    '衛': '卫', '生': '生', '醫': '医', '療': '疗', '藥': '药', '健': '健',
    '康': '康', '環': '环', '境': '境', '汙': '污', '染': '染', '廢': '废',
    '棄': '弃', '勞': '劳', '僱': '雇', '傭': '佣', '就': '就', '薪': '薪',
    '福': '福', '休': '休', '假': '假', '解': '解', '辭': '辞', '退': '退',
    '消': '消', '費': '费', '者': '者', '兒': '儿', '童': '童', '少': '少',
    '青': '青', '老': '老', '殘': '残', '障': '障', '疾': '疾', '婦': '妇',
    '幼': '幼', '家': '家', '庭': '庭', '教': '教', '育': '育', '學': '学',
    '習': '习', '校': '校', '院': '院', '科': '科', '技': '技', '術': '术',
    '工': '工', '農': '农', '漁': '渔', '礦': '矿', '交': '交', '通': '通',
    '運': '运', '物': '物', '流': '流', '能': '能', '源': '源', '力': '力',
    '水': '水', '土': '土', '建': '建', '設': '设', '築': '筑', '屋': '屋',
    '宇': '宇', '公': '公', '共': '共', '秩': '秩', '序': '序', '國': '国',
    '家': '家', '民': '民', '族': '族', '社': '社', '會': '会', '政': '政',
    '治': '治', '軍': '军', '事': '事', '外': '外', '司': '司', '法': '法',
    '立': '立', '行': '行', '政': '政', '察': '察', '警': '警', '獄': '狱',
    '稅': '税', '海': '海', '關': '关', '財': '财', '預': '预', '算': '算',
    '統': '统', '際': '际', '地': '地', '城': '城', '市': '市', '鄉': '乡',
    '鎮': '镇', '村': '村', '莊': '庄', '歐': '欧', '盟': '盟', '聯': '联',
    '合': '合', '世': '世', '界': '界', '貿': '贸', '易': '易', '投': '投',
    '引': '引', '並': '并', '重': '重', '組': '组', '整': '整', '合': '合',
    '打': '打', '犯': '犯', '罪': '罪', '刑': '刑', '民': '民', '仲': '仲',
    '裁': '裁', '調': '调', '節': '节', '約': '约',
}


def _zh_t2s(text: str) -> str:
    """繁体转简体（轻量级，覆盖法规名称常用字）"""
    if not text:
        return text
    return ''.join(_ZH_T2S_MAP.get(ch, ch) for ch in text)


def _zh_s2t(text: str) -> str:
    """简体转繁体（反向映射，轻量级）"""
    if not text:
        return text
    s2t = {v: k for k, v in _ZH_T2S_MAP.items()}
    return ''.join(s2t.get(ch, ch) for ch in text)


# 法域英文 -> 中文映射
_JURISDICTION_ZH_MAP: Dict[str, str] = {
    "us": "美国",
    "usa": "美国",
    "united states": "美国",
    "hk": "香港",
    "hong kong": "香港",
    "mo": "澳门",
    "macau": "澳门",
    "sg": "新加坡",
    "singapore": "新加坡",
}


def _jurisdiction_to_zh(jur: str) -> str:
    """将法域名称转换为简体中文"""
    if not jur:
        return jur
    # 如果已经是中文，直接返回
    if any('\u4e00' <= c <= '\u9fff' for c in jur):
        return jur
    # 尝试英文映射
    lower_jur = jur.lower().strip()
    return _JURISDICTION_ZH_MAP.get(lower_jur, jur)
_TOPIC_LABEL_ZH_CACHE: Optional[Dict[str, str]] = None
_TOPIC_FALLBACK_ZH: Dict[str, str] = {
    # 美国词表中未在 taxonomy_merged.json 中定义的主题
    "public_administration": "公共治理",
    "government_contracting": "政府采购",
    "intellectual_property_licensing": "知识产权",
    "information_classification": "信息分类",
    "administrative_procedures": "行政程序",
    "records_management": "档案管理",
    "electronic_transactions": "电子交易与电子政务",
}


def _load_topic_label_zh_mapping() -> Dict[str, str]:
    """加载 taxonomy_merged.json，返回 {id: label_zh} 映射"""
    global _TOPIC_LABEL_ZH_CACHE
    if _TOPIC_LABEL_ZH_CACHE is not None:
        return _TOPIC_LABEL_ZH_CACHE

    taxonomy_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "taxonomy_merged.json"
    )
    mapping = {}
    try:
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for t in data.get("topics", []):
            tid = t.get("id", "")
            label = t.get("label_zh", "")
            if tid and label:
                mapping[tid] = label
    except Exception:
        pass

    _TOPIC_LABEL_ZH_CACHE = mapping
    return mapping


def _to_label_zh(topic_id: str) -> str:
    """将英文 topic_id 转为中文标签，找不到则返回原值"""
    mapping = _load_topic_label_zh_mapping()
    if topic_id in mapping:
        return mapping[topic_id]
    return _TOPIC_FALLBACK_ZH.get(topic_id, topic_id)


def _label_zh_to_topic_id(label_zh: str) -> str:
    """将中文标签转为英文 topic_id，找不到则返回原值"""
    mapping = _load_topic_label_zh_mapping()
    # 反向查找：label_zh -> id
    for tid, label in mapping.items():
        if label == label_zh:
            return tid
    # fallback 反查
    for tid, label in _TOPIC_FALLBACK_ZH.items():
        if label == label_zh:
            return tid
    return label_zh


class ElasticsearchRepository:
    def __init__(self):
        self._init_client()
        self._model = None  # 懒加载，避免启动时加载大模型
        self._init_classifier()
        self._create_index_if_not_exists()
    
    def _init_classifier(self):
        self.classifier = TopicClassifier()
    
    def _init_client(self):
        client_kwargs: Dict[str, Any] = {
            "hosts": settings.es_hosts,
            "verify_certs": settings.es_verify_certs,
            "request_timeout": settings.es_request_timeout,
            "headers": {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "compatible-with": "8"
            }
        }
        
        if settings.es_user and settings.es_password:
            client_kwargs["basic_auth"] = (settings.es_user, settings.es_password)
        
        self.client = Elasticsearch(**client_kwargs)
        
        if not self.client.ping():
            raise ConnectionError("无法连接到 Elasticsearch")
    
    def _get_model(self):
        """懒加载 embedding 模型（首次使用时加载）"""
        if self._model is None:
            self._model = get_embedding_model()
        return self._model
    
    def _create_index_if_not_exists(self):
        if not self.client.indices.exists(index=settings.es_index_name):
            index_settings = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "text_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "stop"]
                            },
                            "chinese_analyzer": {
                                "type": "custom",
                                "tokenizer": "ik_max_word",
                                "filter": ["lowercase"]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "chinese_analyzer",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            }
                        },
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 1024,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "metadata": {"type": "object"},
                        "law_id": {"type": "keyword"},
                        "jurisdiction": {"type": "keyword"},
                        "title": {
                            "type": "text",
                            "analyzer": "chinese_analyzer",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            }
                        },
                        "article_number": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "passing_date": {"type": "keyword"},
                        "effective_date": {"type": "keyword"},
                        "publication_date": {"type": "keyword"},
                        "topic": {"type": "keyword"},
                        "topics": {"type": "keyword"},
                        "topic_labels": {"type": "keyword"},
                        "topics_details": {
                            "type": "nested",
                            "properties": {
                                "id": {"type": "keyword"},
                                "label_zh": {"type": "keyword"},
                                "score": {"type": "float"},
                                "matched_keywords": {"type": "keyword"}
                            }
                        },
                        "taxonomy_version": {"type": "keyword"}
                    }
                }
            }
            self.client.indices.create(index=settings.es_index_name, body=index_settings)
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        import torch
        import os
        with torch.no_grad():
            # 禁用 tqdm 进度条，避免与 uvicorn 日志系统冲突导致进程退出
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            embeddings = self._get_model().encode(
                texts,
                batch_size=settings.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return embeddings.tolist()
    
    def _extract_topic_ids(self, raw_topics: Any) -> List[str]:
        """将 metadata 中的 topics（对象数组或字符串数组）统一转换为字符串数组。"""
        if not isinstance(raw_topics, list):
            return []
        result = []
        for t in raw_topics:
            if isinstance(t, dict):
                tid = t.get("id") or t.get("topic_id") or t.get("label_en")
                if tid:
                    result.append(str(tid))
            elif isinstance(t, str) and t:
                result.append(t)
        return result

    def _normalize_topic_fields(self, content: str, metadata: Dict) -> Dict:
        """
        统一生成主题字段。
        优先使用 metadata 中已有的 topics（如果有效），否则使用分类器自动标注。
        """
        # 清理 metadata 中的主题相关字段，避免与顶层字段重复/冲突
        clean_metadata = {k: v for k, v in metadata.items()
                          if k not in ('topic', 'topics', 'topic_labels', 'topics_details', 'taxonomy_version',
                                       'passing_date', 'effective_date', 'publication_date')}

        # 尝试从 metadata 提取已有的 topic id
        existing_ids = self._extract_topic_ids(metadata.get("topics"))
        existing_details = metadata.get("topics_details")

        if existing_ids:
            # 使用已有 topics，转换为统一格式
            topic_ids = existing_ids
            topic_labels = []
            details = []
            for tid in topic_ids:
                label = _to_label_zh(tid)
                topic_labels.append(label)
                detail = {"id": tid, "label_zh": label, "score": 0.0, "matched_keywords": []}
                # 如果已有 details，尝试补充 score 和 matched_keywords
                if isinstance(existing_details, list):
                    for d in existing_details:
                        if isinstance(d, dict) and (d.get("id") == tid or d.get("topic_id") == tid):
                            detail["score"] = d.get("score", 0.0)
                            detail["matched_keywords"] = d.get("matched_keywords", [])
                            break
                details.append(detail)
            return {
                "metadata": clean_metadata,
                "topics": topic_ids,
                "topic_labels": topic_labels,
                "topics_details": details,
                "taxonomy_version": metadata.get("taxonomy_version", self.classifier.taxonomy_version),
            }

        # 否则自动标注
        auto = self.classifier.classify_document(content)
        return {
            "metadata": clean_metadata,
            "topics": auto["topics"],
            "topic_labels": auto["topic_labels"],
            "topics_details": auto["topics_details"],
            "taxonomy_version": auto["taxonomy_version"],
        }

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        embeddings = self._embed_texts(documents)
        operations = []
        failed_docs = []

        for doc_id, content, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            topic_fields = self._normalize_topic_fields(content, metadata)
            operations.append({"index": {"_index": settings.es_index_name, "_id": doc_id}})
            doc = {
                "content": content,
                "embedding": embedding,
                "metadata": topic_fields["metadata"],
                "law_id": metadata.get("law_id", ""),
                "jurisdiction": metadata.get("jurisdiction", ""),
                "title": metadata.get("title", ""),
                "article_number": metadata.get("article_number", "") or "",
                "chunk_index": metadata.get("chunk_index", 0),
                "passing_date": metadata.get("passing_date", "") or None,
                "effective_date": metadata.get("effective_date", "") or None,
                "publication_date": metadata.get("publication_date", "") or None,
                # 主题字段提升到顶层，供筛选/聚合/统计使用
                "topic": metadata.get("topic", ""),
                "topics": topic_fields["topics"],
                "topic_labels": topic_fields["topic_labels"],
                "topics_details": topic_fields["topics_details"],
                "taxonomy_version": topic_fields["taxonomy_version"],
            }
            operations.append(doc)

        response = self.client.bulk(body=operations, refresh=True)
        
        # 检查每个文档的索引状态
        if response.get("errors"):
            error_items = []
            for item in response.get("items", []):
                if "error" in item.get("index", {}):
                    error_info = item["index"]
                    error_items.append({
                        "doc_id": error_info.get("_id"),
                        "error_type": error_info.get("error", {}).get("type", ""),
                        "error_reason": error_info.get("error", {}).get("reason", "")
                    })
                    failed_docs.append(error_info.get("_id"))
            
            if error_items:
                # 重试失败的文档
                retry_operations = []
                for i, doc_id in enumerate(ids):
                    if doc_id in failed_docs:
                        # 找到对应的文档信息
                        idx = ids.index(doc_id)
                        retry_operations.append({"index": {"_index": settings.es_index_name, "_id": doc_id}})
                        retry_operations.append(operations[idx * 2 + 1])  # 文档内容在偶数索引+1位置
                
                if retry_operations:
                    retry_response = self.client.bulk(body=retry_operations, refresh=True)
                    if retry_response.get("errors"):
                        for item in retry_response.get("items", []):
                            if "error" in item.get("index", {}):
                                error_info = item["index"]
                                error_items.append({
                                    "doc_id": error_info.get("_id"),
                                    "error_type": error_info.get("error", {}).get("type", ""),
                                    "error_reason": error_info.get("error", {}).get("reason", "")
                                })
                
                raise ValueError(f"部分文档添加失败: {error_items}")
    
    def query(self, query_texts: List[str], n_results: int = 5, where: Optional[Dict] = None) -> Dict:
        results = {"ids": [], "documents": [], "metadatas": [], "distances": []}
        
        for text in query_texts:
            query_embedding = self._embed_texts([text])[0]
            
            knn_base = {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": n_results,
                "num_candidates": n_results * 10
            }
            
            if where:
                # 构建过滤条件
                filter_clauses = []
                for k, v in where.items():
                    # 支持列表值：多选主题的 OR 逻辑
                    if isinstance(v, list) and len(v) > 1:
                        filter_clauses.append({
                            "bool": {
                                "should": [{"term": {k: item}} for item in v],
                                "minimum_should_match": 1
                            }
                        })
                    else:
                        # 单值或单元素列表
                        actual_value = v[0] if isinstance(v, list) else v
                        filter_clauses.append({"term": {k: actual_value}})

                # 将 filter 嵌入 KNN 内部，确保 AND 关系（同时满足向量相似 + 过滤条件）
                search_body = {
                    "knn": {
                        **knn_base,
                        "filter": [{"bool": {"filter": filter_clauses}}]
                    }
                }
            else:
                search_body = {"knn": knn_base}
            
            # 不指定 _source，获取所有字段
            response = self.client.search(index=settings.es_index_name, body=search_body)
            
            hits = response.get("hits", {}).get("hits", [])
            query_ids, query_docs, query_metas, query_distances = [], [], [], []
            
            for hit in hits:
                source = hit.get("_source", {})
                query_ids.append(hit.get("_id"))
                query_docs.append(source.get("content", ""))
                # 合并顶层字段到 metadata 中
                metadata = source.get("metadata", {}).copy()
                metadata["law_id"] = source.get("law_id", "")
                metadata["title"] = source.get("title", "")
                metadata["jurisdiction"] = source.get("jurisdiction", "")
                metadata["article_number"] = source.get("article_number", "")
                metadata["chunk_index"] = source.get("chunk_index", 0)
                # 附加主题信息
                metadata["topic"] = source.get("topic", "")
                metadata["topics"] = source.get("topics", [])
                metadata["topic_labels"] = source.get("topic_labels", [])
                query_metas.append(metadata)
                score = hit.get("_score", 0)
                query_distances.append(1 - score if score > 0 else 1.0)
            
            results["ids"].append(query_ids)
            results["documents"].append(query_docs)
            results["metadatas"].append(query_metas)
            results["distances"].append(query_distances)
        
        return results

    def hybrid_search(self, query_text: str, n_results: int = 5,
                      where: Optional[Dict] = None,
                      knn_weight: float = 0.5,
                      bm25_weight: float = 0.5,
                      rrf_k: int = 60) -> Dict:
        """
        混合检索：向量检索 + BM25 全文检索，使用 ES 内置 RRF 融合。

        Args:
            query_text: 查询文本
            n_results: 返回结果数量
            where: 过滤条件（支持单值或多选）
            knn_weight: 向量检索权重（RRF 模式未使用，保留兼容）
            bm25_weight: 全文检索权重（RRF 模式未使用，保留兼容）
            rrf_k: RRF 常数，默认 60

        Returns:
            与 query() 相同格式的字典
        """
        if not query_text.strip():
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]], "_scores": [[]]}

        query_embedding = self._embed_texts([query_text])[0]
        return self._hybrid_search_with_vector(query_text, query_embedding, n_results, where, rrf_k)

    def hybrid_search_batch(self, query_texts: List[str], n_results: int = 5,
                           where: Optional[Dict] = None, rrf_k: int = 60) -> List[Dict]:
        """
        批量混合检索：一次批量编码所有查询词，并发执行 ES 搜索。

        Args:
            query_texts: 查询文本列表
            n_results: 每个查询返回结果数量
            where: 过滤条件（共享）
            rrf_k: RRF 常数

        Returns:
            List[Dict]，每个元素与 hybrid_search() 返回格式相同
        """
        if not query_texts:
            return []
        # 一次性批量编码所有查询词（关键优化：避免 N 次独立 CPU 推理）
        embeddings = self._embed_texts(query_texts)
        # 并发执行 ES 搜索（IO 密集型）
        with ThreadPoolExecutor(max_workers=min(len(query_texts), 8)) as executor:
            futures = [
                executor.submit(self._hybrid_search_with_vector, query_text, embedding, n_results, where, rrf_k)
                for query_text, embedding in zip(query_texts, embeddings)
            ]
            results = [f.result() for f in futures]
        return results

    def _hybrid_search_with_vector(self, query_text: str, query_embedding: List[float],
                                   n_results: int = 5, where: Optional[Dict] = None,
                                   rrf_k: int = 60) -> Dict:
        """使用预计算的向量执行混合搜索（内部方法，跳过 embedding 计算）

        说明：当前 ES 许可证可能不支持内置 RRF，因此分别执行 KNN 和 BM25 搜索，
        在应用层手动计算 RRF 分数后合并排序，避免依赖 ES 付费特性。
        """
        # 分别执行 KNN 和 BM25 搜索，各取足够候选供 RRF 融合
        knn_results = self._knn_search(query_embedding, n_results * 3, where)
        bm25_results = self._bm25_search(query_text, n_results * 3, where)

        # 使用 RRF 公式合并两个结果列表
        merged = self._merge_rrf_results(knn_results, bm25_results, rrf_k=rrf_k)

        # 截取前 n_results 条
        merged = merged[:n_results]

        query_ids, query_docs, query_metas, query_distances, query_scores = [], [], [], [], []
        for item in merged:
            source = item["source"]
            query_ids.append(item["_id"])
            query_docs.append(source.get("content", ""))

            metadata = source.get("metadata", {}).copy()
            metadata["law_id"] = source.get("law_id", "")
            metadata["title"] = source.get("title", "")
            metadata["jurisdiction"] = source.get("jurisdiction", "")
            metadata["article_number"] = source.get("article_number", "")
            metadata["chunk_index"] = source.get("chunk_index", 0)
            metadata["topic"] = source.get("topic", "")
            metadata["topics"] = source.get("topics", [])
            metadata["topic_labels"] = source.get("topic_labels", [])
            query_metas.append(metadata)

            rrf_score = item["rrf_score"]
            query_scores.append(rrf_score)
            query_distances.append(1.0 / (1.0 + abs(rrf_score)) if rrf_score != 0 else 1.0)

        return {
            "ids": [query_ids],
            "documents": [query_docs],
            "metadatas": [query_metas],
            "distances": [query_distances],
            "_scores": [query_scores],
        }

    def _knn_search(self, query_embedding: List[float], n_results: int,
                    where: Optional[Dict] = None) -> List[Dict]:
        """纯向量 KNN 搜索，返回原始 hits 列表。"""
        knn_base = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": n_results,
            "num_candidates": min(n_results * 10, 10000)
        }

        if where:
            filter_clauses = []
            for k, v in where.items():
                if isinstance(v, list) and len(v) > 1:
                    filter_clauses.append({
                        "bool": {
                            "should": [{"term": {k: item}} for item in v],
                            "minimum_should_match": 1
                        }
                    })
                else:
                    actual_value = v[0] if isinstance(v, list) else v
                    filter_clauses.append({"term": {k: actual_value}})
            search_body = {
                "knn": {
                    **knn_base,
                    "filter": [{"bool": {"filter": filter_clauses}}]
                }
            }
        else:
            search_body = {"knn": knn_base}

        response = self.client.search(index=settings.es_index_name, body=search_body)
        return response.get("hits", {}).get("hits", [])

    def _bm25_search(self, query_text: str, n_results: int,
                     where: Optional[Dict] = None) -> List[Dict]:
        """BM25 全文搜索，返回原始 hits 列表。"""
        bm25_query = {
            "multi_match": {
                "query": query_text,
                "fields": ["content^3", "title^5", "article_number^2"],
                "type": "best_fields",
                "analyzer": "chinese_analyzer"
            }
        }

        if where:
            filter_clauses = []
            for k, v in where.items():
                if isinstance(v, list) and len(v) > 1:
                    filter_clauses.append({
                        "bool": {
                            "should": [{"term": {k: item}} for item in v],
                            "minimum_should_match": 1
                        }
                    })
                else:
                    actual_value = v[0] if isinstance(v, list) else v
                    filter_clauses.append({"term": {k: actual_value}})
            query_body = {
                "bool": {
                    "must": bm25_query,
                    "filter": filter_clauses
                }
            }
        else:
            query_body = bm25_query

        search_body = {
            "query": query_body,
            "size": n_results,
            "_source": True
        }

        response = self.client.search(index=settings.es_index_name, body=search_body)
        return response.get("hits", {}).get("hits", [])

    def _merge_rrf_results(self, knn_hits: List[Dict], bm25_hits: List[Dict],
                           rrf_k: int = 60) -> List[Dict]:
        """手动 RRF 融合：对 KNN 和 BM25 结果按 RRF 分数合并去重排序。"""
        scores: Dict[str, float] = {}
        sources: Dict[str, Dict] = {}

        for rank, hit in enumerate(knn_hits, start=1):
            doc_id = hit.get("_id")
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            sources.setdefault(doc_id, hit.get("_source", {}))

        for rank, hit in enumerate(bm25_hits, start=1):
            doc_id = hit.get("_id")
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            sources.setdefault(doc_id, hit.get("_source", {}))

        merged = []
        for doc_id, rrf_score in scores.items():
            merged.append({
                "_id": doc_id,
                "rrf_score": rrf_score,
                "source": sources.get(doc_id, {})
            })

        merged.sort(key=lambda x: x["rrf_score"], reverse=True)
        return merged

    def find_article(self, title: str, article_number: str) -> Optional[Dict]:
        """通过标题（精确匹配+繁简兼容）和条款号精确查找一条法规条款。"""
        try:
            title_variants = list({title, _zh_t2s(title), _zh_s2t(title)})
            query_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"terms": {"title.keyword": title_variants}},
                            {"term": {"article_number": article_number}}
                        ]
                    }
                },
                "size": 1,
                "_source": True
            }
            response = self.client.search(index=settings.es_index_name, body=query_body)
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                for t in title_variants:
                    query_body = {
                        "query": {
                            "bool": {
                                "must": [
                                    {"match": {"title": t}},
                                    {"term": {"article_number": article_number}}
                                ]
                            }
                        },
                        "size": 1,
                        "_source": True
                    }
                    response = self.client.search(index=settings.es_index_name, body=query_body)
                    hits = response.get("hits", {}).get("hits", [])
                    if hits:
                        break
            if not hits:
                return None
            hit = hits[0]
            source = hit.get("_source", {})
            metadata = source.get("metadata", {}).copy()
            metadata["law_id"] = source.get("law_id", "")
            metadata["title"] = source.get("title", "")
            metadata["jurisdiction"] = source.get("jurisdiction", "")
            metadata["article_number"] = source.get("article_number", "")
            metadata["chunk_index"] = source.get("chunk_index", 0)
            metadata["topics"] = source.get("topics", [])
            metadata["topic_labels"] = source.get("topic_labels", [])
            return {
                "content": source.get("content", ""),
                "title": metadata["title"],
                "article_number": metadata["article_number"],
                "jurisdiction": _jurisdiction_to_zh(metadata.get("jurisdiction", "")),
                "score": 1.0,
                "law_id": metadata["law_id"],
                "source_file": source.get("source_file", ""),
                "chunk_index": metadata["chunk_index"],
                "topics": metadata["topics"],
                "topic_labels": metadata["topic_labels"],
            }
        except Exception as e:
            return None

    def get(self, doc_id: str) -> Optional[Dict]:
        try:
            response = self.client.get(index=settings.es_index_name, id=doc_id)
            return response.get("_source", {})
        except NotFoundError:
            return None

    def get_article_full_text(self, law_id: str, article_number: str) -> Optional[str]:
        """
        通过 law_id + article_number 聚合同一 article 下的所有 chunks，按顺序拼接为完整文本。
        用于 sub_chunk 定位后回取完整 article 上下文。
        """
        if not law_id or not article_number:
            return None
        try:
            query_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"law_id": law_id}},
                            {"term": {"article_number": article_number}}
                        ]
                    }
                },
                "sort": [{"chunk_index": "asc"}],
                "size": 1000,
                "_source": ["content", "chunk_index"]
            }
            response = self.client.search(index=settings.es_index_name, body=query_body)
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                return None
            chunks = [(hit["_source"].get("chunk_index", 0), hit["_source"].get("content", "")) for hit in hits]
            chunks.sort(key=lambda x: x[0])
            return "\n\n".join(content for _, content in chunks if content)
        except Exception:
            return None

    def update(self, doc_id: str, content: Optional[str] = None, metadata: Optional[Dict] = None):
        update_body = {}
        if content:
            update_body["doc"] = {"content": content}
            if content:
                embedding = self._embed_texts([content])[0]
                update_body["doc"]["embedding"] = embedding
        if metadata:
            if "doc" not in update_body:
                update_body["doc"] = {}
            update_body["doc"]["metadata"] = metadata
        
        if update_body:
            self.client.update(index=settings.es_index_name, id=doc_id, body=update_body)
    
    def delete(self, doc_id: str):
        self.client.delete(index=settings.es_index_name, id=doc_id)
    
    def count(self) -> int:
        response = self.client.count(index=settings.es_index_name)
        return response.get("count", 0)
    
    def clear(self):
        index_name = settings.es_index_name
        # 判断该名称是否为 alias，并找出对应的真实索引
        is_alias = False
        concrete_indices = []
        try:
            alias_info = self.client.indices.get_alias(name=index_name)
            is_alias = True
            concrete_indices = list(alias_info.keys())
        except NotFoundError:
            is_alias = False

        if is_alias and concrete_indices:
            # 删除 alias 指向的真实索引
            self.client.indices.delete(index=",".join(concrete_indices))
            # 删除 alias 本身
            self.client.indices.delete_alias(index=concrete_indices, name=index_name)
        elif self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)

        # 重新创建真实索引（名称为 es_index_name）
        self._create_index_if_not_exists()
    
    def get_all_laws(self, jurisdiction: Optional[str] = None, topics: Optional[str] = None) -> List[Dict]:
        """获取所有法规列表（按law_id聚合）"""
        query = {
            "size": 0,
            "aggs": {
                "laws": {
                    "terms": {
                        "field": "law_id",
                        "size": 10000
                    },
                    "aggs": {
                        "title": {"terms": {"field": "title.keyword", "size": 1}},
                        "jurisdiction": {"terms": {"field": "jurisdiction", "size": 1}},
                        "chunk_count": {"value_count": {"field": "chunk_index"}},
                        # 新增：统计实际条款数（按article_number去重）
                        "article_count": {
                            "cardinality": {
                                "field": "article_number"
                            }
                        }
                    }
                }
            }
        }
        
        # 构建查询条件（AND 逻辑）
        must_clauses = []
        if jurisdiction:
            must_clauses.append({"term": {"jurisdiction": jurisdiction}})
        if topics:
            for topic_label in [t.strip() for t in topics.split(",") if t.strip()]:
                must_clauses.append({"term": {"topic_labels": topic_label}})
        
        if must_clauses:
            query["query"] = {"bool": {"must": must_clauses}}
        
        response = self.client.search(index=settings.es_index_name, body=query)
        
        laws = []
        for bucket in response.get("aggregations", {}).get("laws", {}).get("buckets", []):
            law_id = bucket.get("key")
            title_buckets = bucket.get("title", {}).get("buckets", [])
            title = title_buckets[0].get("key") if title_buckets else ""
            jur_buckets = bucket.get("jurisdiction", {}).get("buckets", [])
            jur = jur_buckets[0].get("key") if jur_buckets else ""
            chunk_count = bucket.get("chunk_count", {}).get("value", 0)
            # 使用去重后的条款数
            article_count = bucket.get("article_count", {}).get("value", 0)
            
            laws.append({
                "law_id": law_id,
                "title": title,
                "jurisdiction": _jurisdiction_to_zh(jur),
                "chunk_count": chunk_count,
                "article_count": article_count if article_count > 0 else chunk_count
            })
        
        return laws
    
    def get_law_by_id(self, law_id: str) -> Optional[Dict]:
        """根据law_id获取法规的所有条款"""
        query = {
            "query": {"term": {"law_id": law_id}},
            "size": 1000,
            "sort": [{"chunk_index": {"order": "asc"}}]
        }
        
        response = self.client.search(index=settings.es_index_name, body=query)
        hits = response.get("hits", {}).get("hits", [])
        
        if not hits:
            return None
        
        chunks = []
        for hit in hits:
            source = hit.get("_source", {})
            chunks.append({
                "content": source.get("content", ""),
                "article_number": source.get("article_number", ""),
                "article_title": source.get("article_title", ""),
                "chunk_index": source.get("chunk_index", 0)
            })
        
        first_source = hits[0].get("_source", {})
        return {
            "law_id": law_id,
            "title": first_source.get("title", ""),
            "jurisdiction": _jurisdiction_to_zh(first_source.get("jurisdiction", "")),
            "passing_date": first_source.get("passing_date", ""),
            "chunks": chunks
        }
    
    def delete_by_law_id(self, law_id: str) -> int:
        """根据law_id批量删除所有相关文档"""
        query = {
            "query": {"term": {"law_id": law_id}}
        }

        response = self.client.delete_by_query(index=settings.es_index_name, body=query, refresh=True)
        return response.get("deleted", 0)

    def exists_law_id(self, law_id: str) -> bool:
        """检查数据库中是否已存在指定law_id的文档"""
        query = {
            "size": 1,
            "query": {"term": {"law_id": law_id}}
        }

        try:
            response = self.client.search(index=settings.es_index_name, body=query)
            return response.get("hits", {}).get("total", {}).get("value", 0) > 0
        except Exception:
            return False

    def get_law_chunk_count(self, law_id: str) -> int:
        """获取指定law_id的文档块数量"""
        query = {
            "size": 0,
            "query": {"term": {"law_id": law_id}}
        }

        response = self.client.search(index=settings.es_index_name, body=query)
        return response.get("hits", {}).get("total", {}).get("value", 0)
    
    def query_with_filters(self, n_results: int = 10, where: Optional[Dict] = None) -> Dict:
        """仅按筛选条件检索，不进行语义搜索"""
        results = {"ids": [], "documents": [], "metadatas": [], "distances": []}
        
        search_body = {}

        if where:
            # 构建过滤条件（支持多值 OR 逻辑）
            filter_clauses = []
            for k, v in where.items():
                if isinstance(v, list) and len(v) > 1:
                    # 多选主题的 OR 逻辑
                    filter_clauses.append({
                        "bool": {
                            "should": [{"term": {k: item}} for item in v],
                            "minimum_should_match": 1
                        }
                    })
                else:
                    actual_value = v[0] if isinstance(v, list) else v
                    filter_clauses.append({"term": {k: actual_value}})
            search_body["query"] = {"bool": {"filter": filter_clauses}}
        else:
            search_body["query"] = {"match_all": {}}
        
        search_body["size"] = n_results
        # 不指定 _source，获取所有字段
        search_body["sort"] = [{"law_id": {"order": "asc"}}, {"chunk_index": {"order": "asc"}}]
        
        response = self.client.search(index=settings.es_index_name, body=search_body)
        
        hits = response.get("hits", {}).get("hits", [])
        query_ids, query_docs, query_metas, query_distances = [], [], [], []
        
        for hit in hits:
            source = hit.get("_source", {})
            query_ids.append(hit.get("_id"))
            query_docs.append(source.get("content", ""))
            # 合并顶层字段到 metadata 中
            metadata = source.get("metadata", {}).copy()
            metadata["law_id"] = source.get("law_id", "")
            metadata["title"] = source.get("title", "")
            metadata["jurisdiction"] = source.get("jurisdiction", "")
            metadata["article_number"] = source.get("article_number", "")
            metadata["chunk_index"] = source.get("chunk_index", 0)
            # 附加主题信息
            metadata["topic"] = source.get("topic", "")
            metadata["topics"] = source.get("topics", [])
            metadata["topic_labels"] = source.get("topic_labels", [])
            query_metas.append(metadata)
            query_distances.append(0.0)
        
        results["ids"].append(query_ids)
        results["documents"].append(query_docs)
        results["metadatas"].append(query_metas)
        results["distances"].append(query_distances)
        
        return results
    
    def get_all_jurisdictions(self) -> List[str]:
        """获取所有法域列表（返回简体中文）"""
        query = {
            "size": 0,
            "aggs": {
                "jurisdictions": {
                    "terms": {
                        "field": "jurisdiction",
                        "size": 100
                    }
                }
            }
        }
        
        response = self.client.search(index=settings.es_index_name, body=query)
        buckets = response.get("aggregations", {}).get("jurisdictions", {}).get("buckets", [])
        
        # 转换为中文并去重
        zh_jurisdictions = set()
        result = []
        for bucket in buckets:
            jur = bucket.get("key")
            zh_jur = _jurisdiction_to_zh(jur)
            if zh_jur not in zh_jurisdictions:
                zh_jurisdictions.add(zh_jur)
                result.append(zh_jur)
        
        return result

    def sample_documents(self, jurisdiction: str, size: int = 200,
                         fields: Optional[List[str]] = None) -> List[Dict]:
        """按法域随机采样文档，用于构建法域语义向量。

        Args:
            jurisdiction: 法域名称（简体中文）
            size: 采样数量
            fields: 返回字段，默认 ['title', 'content']

        Returns:
            [{'title': ..., 'content': ...}, ...]
        """
        if fields is None:
            fields = ['title', 'content']

        # 使用 random_score 函数实现随机采样
        query = {
            "size": size,
            "query": {
                "function_score": {
                    "query": {"term": {"jurisdiction": jurisdiction}},
                    "random_score": {"seed": int(time.time()), "field": "_seq_no"},
                    "boost_mode": "replace"
                }
            },
            "_source": fields
        }

        try:
            response = self.client.search(index=settings.es_index_name, body=query)
            hits = response.get("hits", {}).get("hits", [])
            return [hit["_source"] for hit in hits]
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"采样法域 {jurisdiction} 文档失败: {e}")
            return []

    def get_all_topics(self) -> List[Dict]:
        """获取所有主题列表（含统计信息）

        返回格式:
        [
            {
                "id": "cybersecurity",
                "label_zh": "网络安全",
                "count": 15,
                "by_jurisdiction": {"澳门": 10, "新加坡": 5}
            },
            ...
        ]
        """
        query = {
            "size": 0,
            "aggs": {
                "topics": {
                    "terms": {
                        "field": "topic_labels",
                        "size": 100
                    },
                    "aggs": {
                        "jurisdiction_breakdown": {
                            "terms": {"field": "jurisdiction", "size": 10}
                        }
                    }
                }
            }
        }

        response = self.client.search(index=settings.es_index_name, body=query)
        buckets = response.get("aggregations", {}).get("topics", {}).get("buckets", [])

        result = []
        for bucket in buckets:
            raw_label = bucket.get("key", "")
            count = bucket.get("doc_count", 0)

            # 确保返回中文标签：如果是英文ID则转换
            label_zh = _to_label_zh(raw_label) if raw_label and raw_label[0].islower() and '_' in raw_label else raw_label

            # 统计各法域分布
            by_jur = {}
            for jur_bucket in bucket.get("jurisdiction_breakdown", {}).get("buckets", []):
                by_jur[jur_bucket["key"]] = jur_bucket["doc_count"]

            result.append({
                "id": _label_zh_to_topic_id(label_zh),
                "label_zh": label_zh,
                "count": count,
                "by_jurisdiction": by_jur
            })

        # 按 count 降序排列
        result.sort(key=lambda x: x["count"], reverse=True)
        return result

    def get_topic_filter_values(self) -> List[str]:
        """获取所有可用的主题标签值（用于下拉筛选的简单列表，返回中文）"""
        query = {
            "size": 0,
            "aggs": {
                "topics": {
                    "terms": {
                        "field": "topic_labels",
                        "size": 100
                    }
                }
            }
        }

        response = self.client.search(index=settings.es_index_name, body=query)
        buckets = response.get("aggregations", {}).get("topics", {}).get("buckets", [])

        result = []
        for bucket in buckets:
            raw_label = bucket.get("key", "")
            if not raw_label:
                continue
            # 确保返回中文标签：如果是英文ID则转换
            label_zh = _to_label_zh(raw_label) if raw_label[0].islower() and '_' in raw_label else raw_label
            result.append(label_zh)
        return result

    def get_topic_frequencies(self) -> Dict[str, int]:
        """
        获取所有主题标签的出现频次（用于 IDF 加权）
        返回: {"数据保护": 120, "网络安全": 85, ...}
        """
        query = {
            "size": 0,
            "aggs": {
                "topics": {
                    "terms": {
                        "field": "topic_labels",
                        "size": 200
                    }
                }
            }
        }

        response = self.client.search(index=settings.es_index_name, body=query)
        buckets = response.get("aggregations", {}).get("topics", {}).get("buckets", [])

        return {bucket.get("key"): bucket.get("doc_count", 0) for bucket in buckets if bucket.get("key")}

    def get_law_chunk_vectors(self, law_id: str) -> List[Dict]:
        """
        获取指定法规所有 chunk 的向量及元数据（仅取必要字段，不返回 content）
        返回格式：[{"embedding": [...], "law_id": ..., "title": ..., "jurisdiction": ..., "topic_labels": [...]}, ...]
        """
        query = {
            "query": {"term": {"law_id": law_id}},
            "size": 1000,
            "sort": [{"chunk_index": {"order": "asc"}}],
            "_source": ["embedding", "law_id", "title", "jurisdiction", "chunk_index", "topic_labels"]
        }

        response = self.client.search(index=settings.es_index_name, body=query)
        hits = response.get("hits", {}).get("hits", [])

        if not hits:
            return []

        results = []
        for hit in hits:
            source = hit.get("_source", {})
            results.append({
                "embedding": source.get("embedding", []),
                "law_id": source.get("law_id", ""),
                "title": source.get("title", ""),
                "jurisdiction": source.get("jurisdiction", ""),
                "chunk_index": source.get("chunk_index", 0),
                "topic_labels": source.get("topic_labels", []),
            })

        return results

    def knn_search_similar(
        self,
        query_vector: List[float],
        top_k: int = 20,
        exclude_law_id: Optional[str] = None,
        jurisdiction_filter: Optional[str] = None,
        min_score: float = 0.3
    ) -> List[Dict]:
        """
        使用给定向量执行 KNN 相似搜索，返回去重前的原始 chunk 结果。
        排除指定 law_id，可选法域过滤和最低分数阈值。

        注意：使用 knn.filter 做预过滤（法域），用 post_filter 做后置排除（自身），
        确保排除逻辑严格生效而非仅降权。
        """
        # 构建 KNN 预过滤器（在向量搜索前缩小候选集）
        knn_filters = []
        if jurisdiction_filter:
            knn_filters.append({"term": {"jurisdiction": jurisdiction_filter}})

        knn_body = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k * 5,
            "num_candidates": max(top_k * 20, 100)
        }
        if knn_filters:
            knn_body["filter"] = knn_filters

        search_body = {"knn": knn_body, "size": top_k * 5}

        # post_filter：在搜索结果上做硬性排除（确保自身不被返回）
        post_filter_clauses = []
        if exclude_law_id:
            post_filter_clauses.append({"term": {"law_id": exclude_law_id}})
        if post_filter_clauses:
            search_body["post_filter"] = {"bool": {"must_not": post_filter_clauses}}

        response = self.client.search(index=settings.es_index_name, body=search_body)
        hits = response.get("hits", {}).get("hits", [])

        results = []
        for hit in hits:
            source = hit.get("_source", {})
            score = hit.get("_score", 0.0)
            # cosine similarity: ES dense_vector with cosine similarity returns cosine value as _score
            similarity = float(score)

            if similarity < min_score:
                continue

            results.append({
                "id": hit.get("_id"),
                "law_id": source.get("law_id", ""),
                "title": source.get("title", ""),
                "jurisdiction": source.get("jurisdiction", ""),
                "article_number": source.get("article_number", ""),
                "content_preview": source.get("content", "")[:150] if source.get("content") else "",
                "similarity": similarity,
            })

        return results

    def get_topic_statistics(self, limit: int = 50) -> List[Dict]:
        """
        获取主题统计信息，包括各法域的分布
        
        用于可视化看板的主题词云、热力图等
        """
        query = {
            "size": 0,
            "aggs": {
                "topics": {
                    "terms": {
                        "field": "topics",
                        "size": limit
                    },
                    "aggs": {
                        "by_jurisdiction": {
                            "terms": {
                                "field": "jurisdiction",
                                "size": 10
                            }
                        }
                    }
                }
            }
        }

        try:
            response = self.client.search(index=settings.es_index_name, body=query)
            topic_buckets = response.get("aggregations", {}).get("topics", {}).get("buckets", [])

            result = []
            for bucket in topic_buckets:
                topic_name = bucket.get("key", "")
                total_count = bucket.get("doc_count", 0)

                # 获取各法域的分布
                jur_buckets = bucket.get("by_jurisdiction", {}).get("buckets", [])
                by_jurisdiction = {b.get("key"): b.get("doc_count") for b in jur_buckets}

                # 获取示例法律（取前3个）
                sample_query = {
                    "size": 3,
                    "query": {"term": {"topic_labels": topic_name}},
                    "_source": ["law_id", "title", "jurisdiction"]
                }
                sample_response = self.client.search(index=settings.es_index_name, body=sample_query)
                samples = []
                for hit in sample_response.get("hits", {}).get("hits", []):
                    source = hit.get("_source", {})
                    samples.append({
                        "law_id": source.get("law_id"),
                        "law_name_zh": source.get("title"),
                        "jurisdiction": source.get("jurisdiction")
                    })

                result.append({
                    "topic": topic_name,
                    "label_zh": _to_label_zh(topic_name),
                    "count": total_count,
                    "by_jurisdiction": by_jurisdiction,
                    "laws": samples[:5]
                })

            return result

        except Exception as e:
            print(f"⚠️ 获取主题统计失败: {e}")
            return []

    # ================================================================
    #  跨法域专题分析 — 缓存层（analysis_reports_cache）
    # ================================================================

    CACHE_INDEX = "analysis_reports_cache"

    def _ensure_cache_index(self):
        """确保缓存索引存在，不存在则自动创建（静默失败）"""
        try:
            if self.client.indices.exists(index=self.CACHE_INDEX):
                return
            self.client.indices.create(
                index=self.CACHE_INDEX,
                body={
                    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                    "mappings": {
                        "properties": {
                            "cache_type": {"type": "keyword"},  # "analysis" | "compare"
                            "topic_id": {"type": "keyword"},
                            "jurisdiction_a": {"type": "keyword"},
                            "jurisdiction_b": {"type": "keyword"},
                            "fingerprint": {"type": "keyword"},
                            "report_data": {"type": "object", "enabled": False},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"},
                            "cache_version": {"type": "integer"},
                        }
                    },
                }
            )
        except Exception as e:
            print(f"[Cache] 索引创建/检查失败（缓存功能将不可用）: {e}")

    def _cache_doc_id(self, cache_type: str, topic_id: str, jurisdiction_a: str, jurisdiction_b: str) -> str:
        """生成缓存文档 ID"""
        return f"{cache_type}::{topic_id}::{jurisdiction_a}::{jurisdiction_b}"

    def save_analysis_cache(
        self,
        topic_id: str,
        jurisdiction_a: str,
        jurisdiction_b: str,
        fingerprint: str,
        report_data: Dict,
        cache_type: str = "analysis",
    ) -> bool:
        """保存/更新缓存（upsert），支持 analysis 和 compare 两种类型
        
        增强功能：写入前自动清除旧缓存，确保数据一致性
        """
        import datetime, copy
        self._ensure_cache_index()
        doc_id = self._cache_doc_id(cache_type, topic_id, jurisdiction_a, jurisdiction_b)

        # ===== 自动清除旧缓存（确保最新结果）=====
        try:
            old_doc = self.client.get(index=self.CACHE_INDEX, id=doc_id, ignore=[404])
            if old_doc.get("found"):
                self.client.delete(index=self.CACHE_INDEX, id=doc_id)
                self.client.indices.refresh(index=self.CACHE_INDEX)
                logger.info(f"[Cache] 已清除旧缓存: {doc_id} (准备写入新数据)")
        except Exception as e:
            logger.warning(f"[Cache] 清除旧缓存失败（不影响后续写入）: {e}")

        # 深拷贝 + 剥离运行时内部字段（不存入缓存）
        cache_data = copy.deepcopy(report_data)
        for key in ("_raw_response_a", "_raw_response_b", "_cache_hit", "_cache_fingerprint", "timing"):
            cache_data.pop(key, None)
        # extraction 中的 _raw_response 也剥离
        ext = cache_data.get("extraction", {})
        if isinstance(ext, dict):
            for side_key in list(ext.keys()):
                if side_key.startswith("_"):
                    ext.pop(side_key, None)
                elif isinstance(ext[side_key], dict):
                    ext[side_key].pop("_raw_response", None)

        # 修复：将 topic 对象转换为 JSON 字符串（ES字段定义为text类型，不支持对象）
        # compare类型存储的是纯字符串（如"数据保护"），analysis类型传入的是字典对象
        import json
        topic_data = cache_data.get("topic")
        if isinstance(topic_data, dict):
            # 完整序列化为JSON字符串，保留所有字段（name、id、dimensions等）
            cache_data["topic"] = json.dumps(topic_data, ensure_ascii=False)
            logger.info(f"[Cache] topic对象已转换为JSON字符串: {cache_data['topic'][:80]}...")

        try:
            self.client.index(
                index=self.CACHE_INDEX,
                id=doc_id,
                body={
                    "cache_type": cache_type,
                    "topic_id": topic_id,
                    "jurisdiction_a": jurisdiction_a,
                    "jurisdiction_b": jurisdiction_b,
                    "fingerprint": fingerprint,
                    "report_data": cache_data,
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "cache_version": 1,
                },
            )
            self.client.indices.refresh(index=self.CACHE_INDEX)

            # 验证写入：立即读回确认
            verify = self.client.get(index=self.CACHE_INDEX, id=doc_id)
            if verify["found"]:
                logger.info(f"[Cache] 已保存并验证[{cache_type}]: {doc_id}, fp={fingerprint[:16]}...")
                return True
            else:
                logger.error(f"[Cache] 写入后验证失败！doc_id={doc_id}")
                return False
        except Exception as e:
            logger.error(f"[Cache] 保存失败: {type(e).__name__}: {e}")
            return False

    def get_analysis_cache(
        self,
        topic_id: str,
        jurisdiction_a: str,
        jurisdiction_b: str,
        cache_type: str = "analysis",
    ) -> Optional[Dict]:
        """
        查询缓存。返回 {fingerprint, report_data, updated_at} 或 None。
        特殊：topic_id="__all__" 时返回该类型的全部记录列表。
        """
        self._ensure_cache_index()

        # 特殊模式：返回全部记录
        if topic_id == "__all__":
            try:
                resp = self.client.search(
                    index=self.CACHE_INDEX,
                    body={"query": {"term": {"cache_type": cache_type}}, "size": 50},
                )
                return [hit["_source"] for hit in resp["hits"]["hits"]]
            except Exception as e:
                logger.error(f"[Cache] 批量查询失败: {e}")
                return []

        doc_id = self._cache_doc_id(cache_type, topic_id, jurisdiction_a, jurisdiction_b)
        try:
            resp = self.client.get(index=self.CACHE_INDEX, id=doc_id)
            src = resp.get("_source", {})
            return {
                "fingerprint": src.get("fingerprint"),
                "report_data": src.get("report_data"),
                "updated_at": src.get("updated_at"),
            }
        except NotFoundError:
            return None
        except Exception as e:
            print(f"[Cache] 查询失败: {e}")
            return None

    def delete_analysis_cache(
        self,
        topic_id: str,
        jurisdiction_a: str,
        jurisdiction_b: str,
        cache_type: str = "analysis",
    ) -> bool:
        """删除指定缓存"""
        doc_id = self._cache_doc_id(cache_type, topic_id, jurisdiction_a, jurisdiction_b)
        try:
            self.client.delete(index=self.CACHE_INDEX, id=doc_id)
            return True
        except NotFoundError:
            return True
        except Exception as e:
            print(f"[Cache] 删除失败: {e}")
            return False

    # ================================================================
    #  法规摘要缓存层（summary）
    # ================================================================

    def _summary_cache_doc_id(self, law_id: str) -> str:
        """生成摘要缓存文档 ID"""
        return f"summary::{law_id}::_::_"

    def _compute_law_fingerprint(self, law_id: str) -> str:
        """计算法规的指纹（用于检测数据变化）"""
        import hashlib
        try:
            law = self.get_law_by_id(law_id)
            if law:
                chunk_count = len(law.get("chunks", []))
                title = law.get("title", "")
                passing_date = law.get("passing_date", "")
                raw = f"{title}_{chunk_count}_{passing_date}"
                return hashlib.md5(raw.encode("utf-8")).hexdigest()
        except Exception:
            pass
        return ""

    def save_summary_cache(self, law_id: str, summary_data: Dict) -> bool:
        """
        保存法规摘要缓存（无过期时间，仅在数据变化时失效）
        
        Args:
            law_id: 法规ID
            summary_data: 摘要数据
        
        Returns:
            是否保存成功
        """
        import datetime, copy
        self._ensure_cache_index()
        doc_id = self._summary_cache_doc_id(law_id)
        fingerprint = self._compute_law_fingerprint(law_id)

        try:
            old_doc = self.client.get(index=self.CACHE_INDEX, id=doc_id, ignore=[404])
            if old_doc.get("found"):
                self.client.delete(index=self.CACHE_INDEX, id=doc_id)

            cache_data = copy.deepcopy(summary_data)

            self.client.index(
                index=self.CACHE_INDEX,
                id=doc_id,
                body={
                    "cache_type": "summary",
                    "law_id": law_id,
                    "topic_id": law_id,
                    "jurisdiction_a": "_",
                    "jurisdiction_b": "_",
                    "fingerprint": fingerprint,
                    "report_data": cache_data,
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "cache_version": 1,
                },
            )
            self.client.indices.refresh(index=self.CACHE_INDEX)

            verify = self.client.get(index=self.CACHE_INDEX, id=doc_id)
            if verify["found"]:
                logger.info(f"[Cache] 已保存摘要缓存: {doc_id}, fp={fingerprint[:16]}...")
                return True
            else:
                logger.error(f"[Cache] 摘要缓存写入后验证失败！doc_id={doc_id}")
                return False
        except Exception as e:
            logger.error(f"[Cache] 保存摘要缓存失败: {type(e).__name__}: {e}")
            return False

    def get_summary_cache(self, law_id: str) -> Optional[Dict]:
        """
        获取法规摘要缓存
        
        返回：
        - 如果缓存存在且数据未变化，返回摘要数据
        - 如果缓存不存在或数据已变化，返回 None
        
        Args:
            law_id: 法规ID
        
        Returns:
            摘要数据或 None
        """
        self._ensure_cache_index()
        doc_id = self._summary_cache_doc_id(law_id)
        
        try:
            resp = self.client.get(index=self.CACHE_INDEX, id=doc_id)
            src = resp.get("_source", {})
            cached_fingerprint = src.get("fingerprint", "")
            report_data = src.get("report_data")
            
            if not cached_fingerprint or not report_data:
                return None
            
            current_fingerprint = self._compute_law_fingerprint(law_id)
            if current_fingerprint and cached_fingerprint != current_fingerprint:
                logger.info(f"[Cache] 摘要缓存指纹不匹配，失效。law_id={law_id}")
                self.client.delete(index=self.CACHE_INDEX, id=doc_id)
                return None
            
            logger.info(f"[Cache] 命中摘要缓存: {law_id}")
            return report_data
        except NotFoundError:
            return None
        except Exception as e:
            logger.error(f"[Cache] 查询摘要缓存失败: {e}")
            return None

    def delete_summary_cache(self, law_id: str) -> bool:
        """删除指定法规的摘要缓存"""
        doc_id = self._summary_cache_doc_id(law_id)
        try:
            self.client.delete(index=self.CACHE_INDEX, id=doc_id)
            return True
        except NotFoundError:
            return True
        except Exception as e:
            logger.error(f"[Cache] 删除摘要缓存失败: {e}")
            return False

    def delete_all_summary_cache(self) -> bool:
        """删除所有摘要缓存"""
        try:
            query = {"query": {"term": {"cache_type": "summary"}}}
            response = self.client.delete_by_query(index=self.CACHE_INDEX, body=query, refresh=True)
            logger.info(f"[Cache] 已删除 {response.get('deleted', 0)} 条摘要缓存")
            return True
        except Exception as e:
            logger.error(f"[Cache] 删除所有摘要缓存失败: {e}")
            return False


# 日志别名
import logging
logger = logging.getLogger(__name__)
