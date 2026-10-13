import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from repositories.elasticsearch import ElasticsearchRepository, _jurisdiction_to_zh, zh_convert
from core.config import settings
from services.query_decomposer import decompose_query
from utils.search_cache import get_search_cache


logger = logging.getLogger(__name__)


def _normalize_query_for_cache(query: str) -> str:
    """检索缓存的查询归一化：压缩空白 + 统一小写 + 繁→简。

    繁→简保证「澳門 勞動關係法」与「澳门劳动关系法」落到同一个缓存键，
    避免同一问题因输入法差异重复支付一次完整检索。
    """
    if not query:
        return ""
    collapsed = re.sub(r"\s+", " ", query).strip().lower()
    return zh_convert(collapsed, "t2s")


def _flatten_topic_filters(topics: Optional[List[str]], topic: Optional[str]) -> List[str]:
    """把 topic / topics 归一为排序去重后的字符串列表，供缓存键使用。

    调用方形态不一：单主题传字符串（topic）、多主题传列表（topics），
    而 /search 接口还存在把主题列表按位置传入 topic 的写法，使 topic 本身是列表。
    这里统一拉平，避免嵌套列表在 ','.join() 处抛 TypeError。
    """
    raw: List = []
    if topics:
        raw = list(topics) if isinstance(topics, (list, tuple, set)) else [topics]
    elif topic:
        raw = list(topic) if isinstance(topic, (list, tuple, set)) else [topic]

    flat: List[str] = []
    for item in raw:
        if isinstance(item, (list, tuple, set)):
            flat.extend(str(x) for x in item if x)
        elif item:
            flat.append(str(item))
    return sorted(set(flat))


def _title_matches_law_name(law_name: str, title: str) -> bool:
    """判断条款标题与法规名是否指向同一部法规（双向包含）。

    统一到简体再比较：语料里澳门/香港等法域的法规标题是繁体，而提问里的法规名常是简体，
    直接字符串比较永远不相等，会让「第X条」的定向置顶对简体提问失效。
    繁→简 基本一一对应、方向可靠，反向（简→繁）存在多候选，故只做这一侧归一。
    """
    if not law_name or not title:
        return False
    a = zh_convert(law_name, "t2s")
    b = zh_convert(title, "t2s")
    return a in b or b in a


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

    def _rerank_two_stage(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        两阶段重排：只精排前 reranker_window 条，其余候选按 RRF 顺序拼在尾部。

        与「全量精排后截断」的区别在于尾部候选不再被丢弃：窗口从 50 降到 20 后，
        排在 20 名之后的 30 条仍以原始 RRF 次序参与最终排序，因此 top_k 较大时
        （如问答取 50 条候选）不会因为不进精排窗口而整体消失。
        收益是 reranker 对数减半，CPU 推理耗时同步下降（reranker 占检索延迟约 93%）。

        精排模型不可用时原样返回 candidates（保持 RRF 顺序），不阻断检索。
        """
        reranker = self._get_reranker()
        if not reranker or not candidates:
            return candidates

        window = min(len(candidates), settings.reranker_window)
        if window <= 0:
            return candidates

        head = candidates[:window]
        tail = candidates[window:]
        # top_k 传窗口大小而非调用方的 top_k，确保窗口内 20 条全部拿到精排分数，
        # 否则 rerank 内部按 top_k 截断会让尾部候选的排序基准与头部不一致。
        reranked = reranker.rerank(query=query, candidates=head, top_k=window)

        # rerank 会跳过 content 为空的候选（不产生 rerank_score），
        # 这些条目按原 RRF 顺序归入尾部，避免被静默丢弃。
        reranked_ids = {id(doc) for doc in reranked}
        skipped = [doc for doc in head if id(doc) not in reranked_ids]
        return reranked + skipped + tail

    def _expand_by_references(self, results: List[Dict]) -> List[Dict]:
        """
        引用链单跳扩展（组件D）：把排名靠前的结果引用到的条款补到候选尾部。

        分两类引用（详见 repositories.elasticsearch）：
          - 同法规内引用（parse_intra_law_refs）：条款编号是法规内局部的，正文里的
            「第X條」必然指向本法规的条款，解析精确、无需法规名映射；
          - 跨法规引用（parse_cross_law_refs，多跳 A）：仅覆盖能从正文确定性推导出
            law_id 的形式（澳门「第N/YYYY號法律/行政法規」、香港「第N章」），
            由 settings.reference_expand_cross_law_enabled 控制。

        扩展项排在原有结果之后且 score 置 0，因此不挤占原有排序位置：调用方按 top_k
        截断时主结果行为不变，多跳结果只在 top_k 有余量或被显式使用时生效。
        扩展失败静默降级为不扩展，不影响主检索链路。
        """
        if not settings.reference_expand_enabled or not results:
            return results

        seeds = [
            (r.get("law_id", ""), r.get("article_number", ""), r.get("content", ""))
            for r in results[: settings.reference_expand_sources]
        ]
        try:
            docs = self.repo.get_referenced_articles(seeds, settings.reference_expand_max)
        except Exception:
            return results
        if not docs:
            return results

        # law_id -> 法规名：跨法规项的「来源条号」属于另一部法规，需一并给出法规名，
        # 前端据此显示「引用自《来源法规》第 X 条」而非误标成本法规条号。
        law_titles = {r.get("law_id"): r.get("title") for r in results if r.get("law_id")}

        existing = {(r.get("law_id", ""), r.get("article_number", "")) for r in results}
        expanded = []
        for doc in docs:
            if (doc.get("law_id", ""), doc.get("article_number", "")) in existing:
                continue
            # 引用条款不是语义命中，给固定分 0；retrieval_hop/referenced_by 供前端标注来源
            doc["score"] = 0.0
            doc["match_type"] = "reference"
            if doc.get("cross_law"):
                doc["referenced_law_name"] = law_titles.get(doc.get("referenced_law_id"), "")
            expanded.append(doc)

        return results + expanded

    @staticmethod
    def _attach_match_scores(results: List[Dict]) -> List[Dict]:
        """为每条结果计算展示用「匹配度」match_score ∈ [0,1]，缺失置 None。

        取值优先级：
          1. 真实向量余弦相似度（`_cosine`）：语义明确、有界、可跨查询比较；
          2. 都没有 → None，前端不显示百分比。

        说明：曾考虑用精排分 sigmoid 作为优先项，但实测 bge-reranker-v2-m3 的
        logit 在本语料整体偏负（相关知识条款中位数 ≈ -4.5），sigmoid 后会得到
        ~1% 的荒谬值，故不采用。精排分只用于排序，不参与展示。
        另外，旧实现用 RRF 分反推相似度（1/(1+1/(1+rrf))），因 RRF 分动态范围极小，
        导致所有条款都显示约 50%，这里一并修正。
        """
        for r in results:
            cos = r.get("_cosine")
            if cos is not None:
                r["match_score"] = round(max(0.0, min(1.0, float(cos))), 4)
            else:
                r["match_score"] = None
        return results

    def _deduplicate_and_enrich_articles(self, raw_results: Dict, top_k: int) -> Dict:
        """
        对原始检索结果按 article 去重，保留每个 article 相似度最高的 sub_chunk，
        并回取完整 article 文本替换 sub_chunk 内容。

        回取采用单次 msearch 批量完成；原实现为逐条 ES 查询（50 条候选约 3.5s 网络往返）。
        """
        if not raw_results.get('ids') or not raw_results['ids'][0]:
            return raw_results

        # 阶段一：按 (law_id, article_number) 去重，收集待回取的条目
        seen = set()
        picked = []  # [(原始下标, law_id, article_number)]
        for i in range(len(raw_results['ids'][0])):
            metadata = raw_results['metadatas'][0][i]
            law_id = metadata.get('law_id', '')
            article_number = metadata.get('article_number', '')
            key = (law_id, article_number)
            if key in seen:
                continue
            seen.add(key)
            picked.append((i, law_id, article_number))
            if len(picked) >= top_k:
                break

        # 阶段二：一次性批量回取完整 article 文本
        full_texts = self.repo.get_articles_full_text_batch(
            [(law_id, article_number) for _, law_id, article_number in picked]
        )

        # 阶段三：按原顺序组装，回取失败则保留原始 sub_chunk
        hybrid_scores = raw_results.get('_scores')
        has_scores = bool(hybrid_scores and hybrid_scores[0])
        sims = raw_results.get('_similarities')
        has_sims = bool(sims and sims[0])
        new_ids, new_docs, new_metas, new_distances, new_scores = [], [], [], [], []
        new_sims = []
        for i, law_id, article_number in picked:
            full_text = full_texts.get((law_id, article_number))
            new_docs.append(full_text if full_text else raw_results['documents'][0][i])
            new_ids.append(raw_results['ids'][0][i])
            new_metas.append(raw_results['metadatas'][0][i])
            new_distances.append(raw_results['distances'][0][i])
            # 分数必须与文档一一对齐：越界时补 0 占位，而不是跳过追加。
            # 原实现跳过会导致 _scores 比 documents 短，后续按下标取分数会错配到别的条款上。
            if has_scores:
                new_scores.append(hybrid_scores[0][i] if i < len(hybrid_scores[0]) else 0.0)
            # 余弦相似度同样按下标对齐；缺失记为 None（前端不显示百分比）
            if has_sims:
                new_sims.append(sims[0][i] if i < len(sims[0]) else None)

        result = {
            'ids': [new_ids],
            'documents': [new_docs],
            'metadatas': [new_metas],
            'distances': [new_distances],
        }
        if has_scores:
            result['_scores'] = [new_scores]
        if has_sims:
            result['_similarities'] = [new_sims]
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
            sims = raw.get('_similarities')
            if raw.get('ids') and raw['ids'][0]:
                for i in range(len(raw['ids'][0])):
                    metadata = raw['metadatas'][0][i]
                    distance = raw['distances'][0][i]
                    similarity = 1 / (1 + distance)
                    cosine = sims[0][i] if (sims and sims[0] and i < len(sims[0])) else None
                    formatted.append({
                        'content': raw['documents'][0][i],
                        'title': metadata.get('title', '') or '',
                        'article_number': metadata.get('article_number', '') or '',
                        'jurisdiction': _jurisdiction_to_zh(metadata.get('jurisdiction', '') or ''),
                        'score': similarity,
                        '_cosine': cosine,
                        'law_id': metadata.get('law_id', '') or '',
                        'source_file': metadata.get('source_file', '') or '',
                        'chunk_index': metadata.get('chunk_index', 0) or 0,
                    })
            all_per_jur_results.append(formatted)

        # RRF 合并
        merged = self._rrf_merge_results(all_per_jur_results, rrf_k=60)

        # 两阶段精排：前 reranker_window 条精排，其余按 RRF 顺序拼接
        merged = self._rerank_two_stage(query=query, candidates=merged)

        merged = merged[:top_k]
        # 展示用匹配度（真实余弦）
        return self._attach_match_scores(merged)

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
                    if _title_matches_law_name(query_law_name, doc.get("title", "")):
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

        # 两阶段精排：前 reranker_window 条精排，其余按 RRF 顺序拼接
        formatted_results = self._rerank_two_stage(query=query, candidates=formatted_results)

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
        # 条款级去重：新切分下一个条款可能命中多个子块，若直接返回会让重复子块
        # 挤占 top-k 名额（BM25 结果已按 _score 降序，故每个条款保留首次即最高分子块）。
        # 先多取候选再去重，保证去重后仍有 top_k 条；与 search/hybrid_search 口径一致。
        bm25_results = self.repo._bm25_search(enhanced_query, max(top_k * 3, 20), where_filter)

        formatted_results = []
        seen_keys = set()
        for hit in bm25_results:
            src = hit.get("_source", {})
            key = (src.get('law_id', '') or '', src.get('article_number', '') or '')
            if key in seen_keys:
                continue
            seen_keys.add(key)
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
            if len(formatted_results) >= top_k:
                break
        return formatted_results

    def _decompose_query(self, query: str) -> List[str]:
        """多跳 B：把复合查询拆成子查询列表（含锚定的原查询）。无需分解返回 []。

        锚定原查询是「不退化」的关键：分解只往候选池里加条目，原有召回的文档仍在
        池中，最终排序由统一的精排决定，不会因为某条子查询拆得不好而丢失主召回。

        让路跨法域路由：区域词（各国/东南亚/亚太/不同法域…）或比较词（比较/异同…）
        会让 _route_to_jurisdictions 触发多法域扇出检索，这是比较类查询的主召回通道。
        而分解会把锚点也以 _decompose=False 重跑，从而绕过该路由、丢掉扇出召回
        （A/B 实测：6 条变差全部落在此类查询，其中 4 条 gold 直接归零），故直接跳过。
        """
        try:
            routed = self._route_to_jurisdictions(query)
        except Exception:
            routed = None
        if routed and len(routed) > 1:
            return []

        parts = decompose_query(query)
        if len(parts) < 2:
            return []
        uniq: List[str] = []
        for p in [query.strip()] + parts:
            if p and p not in uniq:
                uniq.append(p)
        return uniq if len(uniq) > 1 else []

    def _decomposed_search(self, sub_queries: List[str], top_k: int,
                           jurisdiction: Optional[str], topic: Optional[str],
                           topics: Optional[List[str]]) -> Optional[List[Dict]]:
        """并行执行各子查询检索（跳过精排/引用扩展）→ RRF 合并 → 合并后精排一次。

        子查询跳过精排是为了不把 cross-encoder 成本乘上子查询数：子查询只负责
        「多召回」，排序质量交给合并后的这一次精排。任一步失败返回 None，调用方降级单跳。
        """
        lists: List[List[Dict]] = []
        max_workers = max(1, min(len(sub_queries), settings.multihop_decompose_max_subqueries))
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(self.hybrid_search, sq, top_k, jurisdiction, topic, topics,
                              False, False)
                    for sq in sub_queries
                ]
                for f in futures:
                    try:
                        res = f.result()
                        if res:
                            lists.append(res)
                    except Exception as e:
                        logger.warning("子查询检索失败，忽略该子查询: %s", e)
        except Exception as e:
            logger.warning("查询分解并行检索失败，降级单跳: %s", e)
            return None

        if not lists:
            return None
        merged = self._rrf_merge_results(lists, rrf_k=60)
        # 统一用「原查询」做精排，保证与单跳路径同一打分基准（sub_queries[0] 即锚点）
        merged = self._rerank_two_stage(query=sub_queries[0], candidates=merged)
        # 子查询合并后条数是各子查询之和，同样要收敛回 top_k，再追加引用扩展
        merged = merged[:top_k]
        merged = self._expand_by_references(merged)
        # 合并后统一补齐展示分：合并/扩展会引入无 _cosine 的条目，需保证 match_score 键齐全
        return self._attach_match_scores(merged)

    def hybrid_search(self, query: str, top_k: int = 5,
                      jurisdiction: Optional[str] = None,
                      topic: Optional[str] = None,
                      topics: Optional[List[str]] = None,
                      _decompose: bool = True,
                      _rerank: bool = True) -> List[Dict]:
        """
        混合检索（向量 + BM25 + RRF 融合），返回格式与 search() 完全一致。

        参数与 search() 相同：
            query: 搜索关键词
            top_k: 返回数量
            jurisdiction: 法域筛选
            topic: 单个主题（向后兼容）
            topics: 多主题列表（优先使用）

        _decompose / _rerank 为内部参数：子查询检索时置 False，用于关闭「再分解」
        与「精排/引用扩展」，避免递归与成本放大；外部调用方无需传。
        """
        # 子查询检索跳过缓存：其缓存键用的是全局精排开关，与子查询实际关闭精排的
        # 口径不一致，读写都可能污染主查询结果，故直接绕过。
        use_cache = settings.search_cache_enabled and _rerank
        cache = get_search_cache() if use_cache else None
        cache_key = None
        if cache is not None:
            cache_key, cached = self._cache_lookup(
                query, top_k, jurisdiction, _flatten_topic_filters(topics, topic)
            )
            if cached is not None:
                return cached

        # 多跳 B：复合查询分解（仅顶层调用、且未关闭精排时触发；子查询不再分解）
        if _decompose and _rerank and settings.multihop_decompose_enabled:
            try:
                sub_queries = self._decompose_query(query)
            except Exception as e:
                logger.warning("查询分解失败，降级单跳: %s", e)
                sub_queries = []
            if len(sub_queries) > 1:
                merged = self._decomposed_search(sub_queries, top_k, jurisdiction, topic, topics)
                if merged:
                    if cache is not None and cache_key is not None:
                        cache.set(cache_key, merged)
                    return merged

        enhanced_query = self._enhance_query(query)
        search_query = enhanced_query or query

        # 跨法域路由：仅顶层调用（_decompose=True）且未显式指定 jurisdiction 时触发。
        # 子查询（_decompose=False）不参与路由，避免路由内部再次精排造成成本放大。
        if _decompose and jurisdiction is None:
            jurisdiction = self._detect_jurisdiction(query)
            if jurisdiction is None:
                # 尝试路由检测
                routed_jurs = self._route_to_jurisdictions(query)
                if routed_jurs and len(routed_jurs) > 1:
                    routed = self._hybrid_search_routed(
                        query=query, jurisdictions=routed_jurs, top_k=top_k,
                    )
                    if cache is not None:
                        cache.set(cache_key, routed)
                    return routed

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

        # 候选窗口扩展：有 reranker 时保留更多候选供精排（子查询关闭精排，收窄窗口）
        reranker_available = settings.reranker_enabled and _rerank
        dedup_k = max(top_k * 3, 50) if reranker_available else top_k
        results = self._deduplicate_and_enrich_articles(results, dedup_k)

        formatted_results = []
        hybrid_scores = results.get("_scores", None)
        sims = results.get("_similarities", None)

        if results.get('ids') and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                # 排序用 RRF 分（保证混合检索最佳排序）；展示分另算（见 _attach_match_scores）。
                # score 字段保留旧口径以维持前端既有排序稳定，不再作为展示百分比来源。
                distance = results['distances'][0][i]
                similarity = 1 / (1 + distance)
                cosine = sims[0][i] if (sims and sims[0] and i < len(sims[0])) else None

                metadata = results['metadatas'][0][i]
                result = {
                    'content': results['documents'][0][i],
                    'title': metadata.get('title', '') or '',
                    'article_number': metadata.get('article_number', '') or '',
                    'jurisdiction': _jurisdiction_to_zh(metadata.get('jurisdiction', '') or ''),
                    'score': similarity,
                    '_cosine': cosine,
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
                    if _title_matches_law_name(query_law_name, doc.get("title", "")):
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
                    found["_cosine"] = 1.0  # 定向命中：展示为 100%
                    target_doc = found

        # 两阶段精排：前 reranker_window 条精排，其余按 RRF 顺序拼接（子查询关闭时跳过）
        if _rerank:
            formatted_results = self._rerank_two_stage(query=search_query, candidates=formatted_results)

        # 把目标条款插到最前面
        if target_doc:
            target_doc["score"] = 1.0  # 确保前端按 score 排序时排第一
            if target_doc.get("_cosine") is None:
                target_doc["_cosine"] = 1.0
            formatted_results.insert(0, target_doc)

        # 主结果按 top_k 截断：上面的 dedup_k（= max(top_k*3, 50)）只是为精排保留的
        # 候选余量，不应作为最终条数外泄（此前前端要 50 条却拿到 150+ 条）。
        formatted_results = formatted_results[:top_k]

        # 引用链扩展在其后追加：扩展项是「附加引用链接」而非排名结果，不计入 top_k 名额。
        # 若先追加再截断，因主结果始终填满 top_k，扩展项会被永久截掉（多跳A 在检索页失效）。
        if _rerank:
            formatted_results = self._expand_by_references(formatted_results)

        # 展示用匹配度（真实余弦）在这里统一计算：引用扩展补入的条款无余弦，置 None
        formatted_results = self._attach_match_scores(formatted_results)

        if cache is not None:
            cache.set(cache_key, formatted_results)

        return formatted_results

    def _cache_lookup(self, query: str, top_k: int, jurisdiction: Optional[str],
                      topics: Optional[List[str]]):
        """缓存查询，返回 (缓存键, 命中结果或 None)。

        键由「筛选条件 + 结果相关开关 + 归一化查询」构成：只按查询串比对会让
        「同一问题 + 不同法域筛选」错误复用结果；而评测脚本会在同一进程内切换
        reranker_enabled，若不把开关计入键，先跑的「无精排」结果会被「有精排」
        那轮直接命中，得到未精排的错误顺序。
        """
        if not query or not query.strip():
            return None, None

        filter_key = "|".join([
            f"top_k={top_k}",
            f"jur={jurisdiction or ''}",
            f"topics={','.join(topics) if topics else ''}",
            f"rerank={int(settings.reranker_enabled)}:{settings.reranker_window}",
            f"refexp={int(settings.reference_expand_enabled)}:{int(settings.reference_expand_cross_law_enabled)}",
            f"mhdec={int(settings.multihop_decompose_enabled)}",
        ])
        key = (filter_key, _normalize_query_for_cache(query))
        return key, get_search_cache().get(key)
