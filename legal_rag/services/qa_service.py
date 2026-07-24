import logging
import threading
from typing import List, Dict, Optional
import time
import math
import numpy as np
from openai import OpenAI
from repositories.elasticsearch import ElasticsearchRepository
from core.config import settings

logger = logging.getLogger(__name__)


class QAService:
    def __init__(self):
        self.repo = ElasticsearchRepository()
        self.embedding_model = None
        self.jurisdiction_vectors = {}
        # 启动后台线程预热法域向量，避免首次请求长时间等待
        self._jurisdiction_vectors_lock = threading.Lock()
        self._jurisdiction_vectors_loading = False
        threading.Thread(target=self._warmup_jurisdiction_vectors, daemon=True).start()

    def _warmup_jurisdiction_vectors(self):
        """后台预热法域语义向量，不影响服务启动速度。"""
        try:
            with self._jurisdiction_vectors_lock:
                if self._jurisdiction_vectors_loading or self.jurisdiction_vectors:
                    return
                self._jurisdiction_vectors_loading = True
            logger.info("[Warmup] 后台开始预热法域语义向量...")
            self._ensure_jurisdiction_vectors()
            logger.info("[Warmup] 法域语义向量后台预热完成")
        except Exception as e:
            logger.warning(f"[Warmup] 法域语义向量后台预热失败: {e}")
        finally:
            with self._jurisdiction_vectors_lock:
                self._jurisdiction_vectors_loading = False

    def _get_embedding_model(self):
        """延迟加载共享的 embedding 模型。"""
        if self.embedding_model is None:
            from core.model_loader import get_embedding_model
            self.embedding_model = get_embedding_model()
        return self.embedding_model

    def _build_jurisdiction_vectors_from_documents(self) -> Dict[str, np.ndarray]:
        """从 ES 中各法域的已有文档语料，聚合生成法域语义向量。

        步骤：
          1. 从 ES 获取所有实际存在文档的法域列表
          2. 对每个法域，采样代表性文档（取 title + content 前 200 字）
          3. 计算各文档的 embedding，平均后 L2 归一化
          4. 跳过文档数为 0 的法域（降级为名称向量）

        当 ES 中新入库法域数据后，重启服务或调用 _refresh_jurisdiction_vectors_if_needed() 自动生效。
        """
        model = self._get_embedding_model()
        vectors = {}

        try:
            jurisdictions = self.repo.get_all_jurisdictions()
            for jur in jurisdictions:
                docs = self.repo.sample_documents(
                    jurisdiction=jur,
                    size=30,
                    fields=['title', 'content']
                )
                if not docs:
                    # 该法域暂无文档，降级：用法域名称本身生成向量
                    fallback_text = f"{jur} 法律 法规"
                    vec = model.encode(
                        [fallback_text],
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )[0]
                    vectors[jur] = vec
                    continue

                # 构造语料文本：标题 + 内容前 200 字
                texts = []
                for doc in docs:
                    title = doc.get('title', '')
                    content = doc.get('content', '')[:200]
                    texts.append(f"{title} {content}")

                # 计算所有文档向量的平均向量
                doc_vectors = model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=32
                )
                avg_vector = np.mean(doc_vectors, axis=0)
                # L2 归一化
                avg_vector = avg_vector / np.linalg.norm(avg_vector)
                vectors[jur] = avg_vector

        except Exception as e:
            logger.warning(f"从文档构建法域语义向量失败: {e}")
            # 完全失败时用空字典，不影响关键词方案
            return {}

        return vectors

    def _ensure_jurisdiction_vectors(self):
        """懒加载法域语义向量（首次使用时预计算）"""
        if self.jurisdiction_vectors:
            return
        with self._jurisdiction_vectors_lock:
            if self.jurisdiction_vectors:
                return
            try:
                self.jurisdiction_vectors = self._build_jurisdiction_vectors_from_documents()
                logger.info(f"法域语义向量懒加载完成，共 {len(self.jurisdiction_vectors)} 个法域")
            except Exception as e:
                logger.warning(f"法域语义向量懒加载失败: {e}")
                self.jurisdiction_vectors = {}

    def _refresh_jurisdiction_vectors_if_needed(self):
        """检测 ES 中的法域列表是否发生变化，变化时重新生成语义向量。"""
        try:
            current_jurs = set(self.repo.get_all_jurisdictions())
            cached_jurs = set(self.jurisdiction_vectors.keys())
            if current_jurs != cached_jurs:
                logger.info(f"法域列表变化: {cached_jurs} → {current_jurs}，重新生成语义向量")
                self.jurisdiction_vectors = self._build_jurisdiction_vectors_from_documents()
        except Exception:
            pass  # 降级使用现有向量

    def _cosine_similarity(self, a, b) -> float:
        """计算两个归一化向量的余弦相似度。"""
        return float(sum(x * y for x, y in zip(a, b)))

    def _get_jurisdiction_triggers(self) -> Dict[str, List[str]]:
        """构建法域触发词映射表（基础触发词 + 动态从ES获取的法域）"""
        # 基础触发词：覆盖常见的中英文及缩写形式
        base_triggers = {
            '澳门': ['澳门', 'macau', 'Macau', 'MACAU', '澳門', 'MO'],
            '香港': ['香港', 'hong kong', 'hongkong', 'Hong Kong', 'Hongkong', 'HONG KONG', 'HK'],
            '新加坡': ['新加坡', 'singapore', 'Singapore', 'SINGAPORE', 'sg', 'SG'],
            '美国': ['美国', '美國', 'usa', 'USA', 'us', 'US', 'america', 'America', 'united states', 'United States'],
            '英国': ['英国', '英國', 'uk', 'UK', 'britain', 'Britain', 'united kingdom', 'United Kingdom'],
            '日本': ['日本', 'japan', 'Japan', 'JAPAN', 'jp', 'JP'],
            '韩国': ['韩国', '韓国', 'korea', 'Korea', 'kr', 'KR', 'south korea'],
            '法国': ['法国', '法國', 'france', 'France', 'FRANCE', 'fr', 'FR'],
            '德国': ['德国', '德國', 'germany', 'Germany', 'GERMANY', 'de', 'DE'],
            '澳大利亚': ['澳大利亚', '澳洲', 'australia', 'Australia', 'AUSTRALIA', 'au', 'AU'],
        }

        # 从ES动态获取实际存在的法域，补充到触发词表中
        try:
            existing_jurs = self.repo.get_all_jurisdictions()
            for jur in existing_jurs:
                if jur not in base_triggers:
                    # 未配置触发词的法域，至少用其名称本身作为触发词
                    base_triggers[jur] = [jur, jur.lower()]
        except Exception:
            pass  # ES不可用时降级为基础触发词

        return base_triggers

    def _score_jurisdictions(self, question: str) -> Dict[str, float]:
        """
        多层级法域候选召回（方案 B）：基于触发词权重打分，返回各法域置信度。
        """
        jurisdiction_triggers = self._get_jurisdiction_triggers()
        raw_scores = {}

        for jurisdiction, triggers in jurisdiction_triggers.items():
            score = 0.0
            matched = False
            for trigger in triggers:
                if trigger in question:
                    # 基础权重：触发词越长、越精确，权重越高
                    weight = max(1.0, len(trigger) * 0.3)
                    # 英文字母触发词（如 HK、SG）降低权重，避免缩写误命中
                    if trigger.isalpha() and len(trigger) <= 2:
                        weight *= 0.3
                    score += weight
                    matched = True
            if matched:
                raw_scores[jurisdiction] = score

        if not raw_scores:
            return {}

        # softmax 归一化，转换为概率分布
        import math
        max_score = max(raw_scores.values())
        exp_scores = {j: math.exp(s - max_score) for j, s in raw_scores.items()}
        total = sum(exp_scores.values())
        return {j: s / total for j, s in exp_scores.items()}

    def _refine_jurisdictions_by_rules(self, question: str, candidates: List[str]) -> Optional[List[str]]:
        """
        规则精修（方案 C）：识别排除、对比、多法域等句式，调整候选结果。
        """
        import re
        if not candidates:
            return None

        # 排除结构："除了澳门之外"、"除澳门外"
        exclusion_match = re.search(r'除了?\s*([^，,。；;]+?)\s*(?:之外|以外|外|)', question)
        if exclusion_match and '除了' in question:
            excluded_text = exclusion_match.group(1)
            excluded = [j for j in candidates if any(t in excluded_text for t in self._get_jurisdiction_triggers()[j])]
            refined = [j for j in candidates if j not in excluded]
            return refined if refined else candidates

        # 对比结构："澳门和香港有什么不同"、"A 与 B 的区别"
        comparison_patterns = [
            r'(.+?)\s*和\s*(.+?)\s*(?:有?什么|的)?(?:区别|差异|不同|对比|比较)',
            r'(.+?)\s*与\s*(.+?)\s*(?:的)?(?:区别|差异|不同|对比|比较)',
            r'(.+?)\s*跟\s*(.+?)\s*(?:的)?(?:区别|差异|不同|对比|比较)',
            r'(?:不同|区别|差异|对比)\s*在于?\s*(.+?)\s*和\s*(.+)',
        ]
        for pattern in comparison_patterns:
            if re.search(pattern, question):
                # 对比问句保留所有候选，不强制单一法域
                return candidates

        # 否定结构："不是澳门，是香港"、"不同于澳门"
        negation_match = re.search(r'(?:不是|不同于|并非|非)\s*([^，,。；;]+?)\s*(?:，|而是|是|，)',
                                   question)
        if negation_match:
            # 否定句通常涉及多个法域，保留候选让打分决定
            return candidates

        return candidates

    def _detect_jurisdictions_semantic(self, question: str) -> Dict[str, float]:
        """
        方案 A：语义化法域识别。
        将用户问题编码为向量，与预计算的法域描述向量计算余弦相似度。
        返回高于阈值的法域及其相似度。
        """
        self._ensure_jurisdiction_vectors()  # 首次使用时懒加载
        if not self.jurisdiction_vectors:
            return {}

        try:
            model = self._get_embedding_model()
            query_vector = model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]

            scores = {}
            for jur, vec in self.jurisdiction_vectors.items():
                sim = self._cosine_similarity(query_vector, vec)
                if sim >= 0.40:  # 语义识别阈值（文档语料覆盖更广，适当降低门槛）
                    scores[jur] = sim
            return scores
        except Exception as e:
            logger.warning(f"法域语义识别失败: {e}")
            return {}

    def _inherit_jurisdictions_from_history(self, history: Optional[List[Dict]]) -> Optional[List[str]]:
        """
        方案 D：从历史会话中继承法域。
        倒序查找最近一条带有 jurisdictions 的消息，返回其法域列表。
        """
        if not history:
            return None
        for msg in reversed(history):
            jurs = msg.get('jurisdictions')
            if isinstance(jurs, list) and len(jurs) > 0:
                return jurs
        return None

    def _detect_jurisdictions(self, question: str, history: Optional[List[Dict]] = None) -> List[str]:
        """
        法域检测：A + B + C + D 融合方案。
        1. 触发词打分得到候选法域及置信度（B）
        2. 语义识别补充别名、口语化表达（A）
        3. 规则精修处理排除/对比/否定句式（C）
        4. 应用阈值策略决定最终法域列表
        5. 当前未检测到时，继承历史会话法域（D）
        """
        keyword_scores = self._score_jurisdictions(question)
        semantic_scores = self._detect_jurisdictions_semantic(question)

        # 融合 B 和 A 的分数
        all_jurisdictions = set(keyword_scores.keys()) | set(semantic_scores.keys())
        scores = {}
        for jur in all_jurisdictions:
            kw = keyword_scores.get(jur, 0.0)
            sem = semantic_scores.get(jur, 0.0)
            if kw > 0 and sem > 0:
                # 两者都命中：文档语料向量更可靠，适当提升语义权重
                scores[jur] = 0.6 * kw + 0.4 * sem
            elif kw > 0:
                scores[jur] = kw
            else:
                # 仅语义命中，文档语料向量质量较高，降权系数提升
                scores[jur] = sem * 0.9

        detected = []

        if scores:
            # 按置信度排序
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            max_score = sorted_scores[0][1]
            candidates = [j for j, s in sorted_scores]

            # 规则精修
            refined = self._refine_jurisdictions_by_rules(question, candidates)
            if refined:
                # 阈值策略：语义-only 命中需要更高阈值
                min_confidence = 0.45 if refined[0] in keyword_scores else 0.55
                if max_score < min_confidence:
                    refined = []
                elif max_score >= 0.7 and len(refined) == 1:
                    detected = [refined[0]]
                elif len(refined) >= 2:
                    second_score = scores[refined[1]]
                    if max_score - second_score < 0.3:
                        detected = refined[:3]
                    else:
                        detected = [refined[0]]
                else:
                    detected = [refined[0]]

        # 方案 D：当前未检测到明确法域时，尝试从历史会话继承
        if not detected:
            inherited = self._inherit_jurisdictions_from_history(history)
            if inherited:
                return inherited

        return detected

    def _enhance_query(self, query: str) -> str:
        enhancements = [
            (('数据保护', '个人数据', '隐私', '个人信息'), 'personal data protection privacy'),
            (('网络安全', '网络', '信息安全'), 'cybersecurity network security information security'),
            (('电子', '数字化', '数字签名', '电子签名'), 'electronic digital signature'),
            (('计算机', '电脑', '网络犯罪'), 'computer cybercrime'),
            (('行政', '政府', '官员'), 'administrative government official'),
            (('刑事', '犯罪', '处罚'), 'criminal penalty offense'),
            (('法律', '法规', '条例'), 'law regulation act ordinance'),
        ]

        for cn_keywords, en_keywords in enhancements:
            for cn_kw in cn_keywords:
                if cn_kw in query:
                    return f"{query} {en_keywords}"
        return query

    def _is_retrieval_poor(self, contexts: List[Dict], min_results: int = 2, min_similarity: float = 0.6) -> bool:
        """
        方案 E：检索质量门。
        当检索结果数量不足或最高相似度过低时，认为质量不达标。
        """
        if len(contexts) < min_results:
            return True
        return contexts[0]['similarity'] < min_similarity

    def _merge_deduplicate_contexts(self, contexts_list: List[List[Dict]]) -> List[Dict]:
        """合并多组检索结果并按 article 去重，按相似度降序排列。"""
        seen = set()
        merged = []
        for contexts in contexts_list:
            for ctx in contexts:
                key = (ctx['metadata'].get('law_id', ''), ctx['metadata'].get('article_number', ''))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(ctx)
        merged.sort(key=lambda x: x['similarity'], reverse=True)
        return merged

    def _retrieve_contexts(self, query: str, top_k: int, jurisdictions: Optional[List[str]] = None) -> List[Dict]:
        enhanced_query = self._enhance_query(query)
        # 扩大召回数量：sub_chunk 检索后需要按 article 去重，避免去重后结果不足
        candidate_k = max(top_k * 3, 20)
        contexts = []

        if jurisdictions and len(jurisdictions) > 1:
            for jurisdiction in jurisdictions:
                results = self.repo.query([enhanced_query], n_results=candidate_k, where={"jurisdiction": jurisdiction})
                if results.get('ids') and results['ids'][0]:
                    for i in range(len(results['ids'][0])):
                        distance = results['distances'][0][i]
                        contexts.append({
                            'content': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'similarity': 1 / (1 + distance)
                        })
        else:
            where_filter = {"jurisdiction": jurisdictions[0]} if (jurisdictions and len(jurisdictions) == 1) else None
            results = self.repo.query([enhanced_query], n_results=candidate_k, where=where_filter)
            if results.get('ids') and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    distance = results['distances'][0][i]
                    contexts.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity': 1 / (1 + distance)
                    })

        contexts.sort(key=lambda x: x['similarity'], reverse=True)

        # 按 article 去重，保留每个 article 相似度最高的 sub_chunk
        seen_articles = {}
        for ctx in contexts:
            law_id = ctx['metadata'].get('law_id', '')
            article_number = ctx['metadata'].get('article_number', '')
            key = (law_id, article_number)
            if key not in seen_articles:
                seen_articles[key] = ctx

        # 回取完整 article 文本作为 LLM 上下文，兼顾检索精度与阅读完整性
        deduped_contexts = []
        for key, ctx in seen_articles.items():
            law_id, article_number = key
            full_text = self.repo.get_article_full_text(law_id, article_number)
            if full_text:
                ctx['content'] = full_text
            deduped_contexts.append(ctx)

        deduped_contexts.sort(key=lambda x: x['similarity'], reverse=True)
        return deduped_contexts[:top_k]

    def _retrieve_contexts_with_fallback(
        self,
        query: str,
        top_k: int,
        jurisdictions: Optional[List[str]] = None,
        allow_fallback: bool = True
    ) -> List[Dict]:
        """
        方案 E：检索-反馈自适应。
        先按指定法域检索，若质量不达标且允许回退，则追加全域检索并合并去重。
        """
        contexts = self._retrieve_contexts(query, top_k, jurisdictions)

        if not allow_fallback or not jurisdictions:
            return contexts

        if not self._is_retrieval_poor(contexts):
            return contexts

        # 限定法域检索结果过差，回退到全域检索
        fallback_contexts = self._retrieve_contexts(query, top_k, None)
        return self._merge_deduplicate_contexts([contexts, fallback_contexts])

    def _build_prompt(self, question: str, contexts: List[Dict], history: Optional[List[Dict]] = None) -> str:
        top_contexts = contexts[:5]
        references = []

        for i, ctx in enumerate(top_contexts, 1):
            article_num = ctx['metadata'].get('article_number', '')
            title = ctx['metadata'].get('title', '')
            jurisdiction = ctx['metadata'].get('jurisdiction', '')
            law_id = ctx['metadata'].get('law_id', '')
            similarity = ctx['similarity']
            content = ctx['content']

            references.append(
                f"【来源{i}】{jurisdiction}《{title}》{article_num}\n法ID: {law_id}\n相似度: {similarity:.4f}\n{content}")

        references_text = "\n\n".join(references)

        # 检测是否为多法域（用于决定是否生成跨法域对比）
        jurisdictions_in_contexts = set()
        for ctx in top_contexts:
            jur = ctx['metadata'].get('jurisdiction', '')
            if jur:
                jurisdictions_in_contexts.add(jur)
        is_multi_jurisdiction = len(jurisdictions_in_contexts) > 1

        # 构建对话历史部分
        history_text = ""
        is_follow_up = False
        if history and len(history) >= 1:
            # 只要存在历史对话，就视为多轮追问
            is_follow_up = True
            history_parts = []
            for h in history[-6:]:  # 保留最近3轮对话（6条消息）
                role = "用户" if h.get("role") == "user" else "助手"
                content = h.get("content", "")
                if content:
                    history_parts.append(f"{role}：{content}")
            if history_parts:
                history_text = "\n\n【对话上下文】\n用户正在进行追问，以下是之前的对话记录，请结合上下文理解当前问题的具体指向：\n" + "\n".join(history_parts) + "\n"

        # 根据是否为追问调整prompt
        if is_follow_up:
            prompt_template = """你是一位精通境外法律法规的资深法律顾问。请严格基于以下参考资料回答用户追问，不得编造。

{history}
## 参考资料库（请严格基于以下法规条文回答）：

{references}

---

## 用户追问：{question}

## 一、简要定位（1-2句话）
指出针对哪个具体点补充说明，结合上下文明确追问意图。

## 二、详细展开（分点阐述，每点200-280字，共3-5个要点）
以深度解读和实务分析为主，法条原文仅作支撑。每个要点按以下结构展开：
1. 法规出处：法域 + 《法规全称》第 X 条/第 X 款；来源标注【来源X】。
2. 深度解读（重点）：阐释法律依据、适用范围、核心含义、立法意图和相互关联，避免停留在表面复述。
3. 重点提示：合规风险点、特殊情况、时限或程序性规定。
4. 实务指引（重点）：给出企业或个人可操作的具体步骤、最佳实践、需准备的材料、落地建议。

## 三、总结回顾（150-200字）
归纳核心答案，结合前文给出结论，并列出 2-3 条延伸提醒。

## 四、核心结论重申（2-3句话）
用最简洁的语言重申结论和关键行动项。

**写作规范**
- 使用 Markdown 层级标题，关键术语加粗，重要法条用行内引用或短引用块（仅摘录关键短语，不要大段复制原文）。
- 回答以分析、解读、实务建议为主，不要堆砌法条原文。
- 所有论断必须有法条支撑，引用必须标注【来源X】。
- 语言专业但不晦涩，像资深律师给客户解释。
- 总字数要求：1000-1500字。
- 严禁编造资料中没有的信息；资料不足时明确说明。

请开始回答："""
        else:
            # 根据是否为多法域，动态调整回答结构要求
            if is_multi_jurisdiction:
                structure_section = """
## 三、总结与建议（200-250字）
先根据参考资料中多个法域的规定制作简明对比表，指出主要差异点、共同点和不同法域的严格程度差异；然后精炼总结答案要点，给出行动指引，并列出 2-3 条延伸提醒。"""
                total_words_hint = "总字数：1500-2200字（确保内容详实）"
            else:
                structure_section = """
## 三、总结与建议（150-200字）
精炼总结答案要点，给出行动指引，并列出 2-3 条延伸提醒。"""
                total_words_hint = "总字数：1200-1800字（确保内容详实）"

            prompt_template = f"""
你是一位精通境外法律法规的资深法律顾问，拥有20年实务经验，擅长用通俗易懂的语言解释复杂的法律问题。

{{history}}
## 参考资料库（请严格基于以下法规条文回答，不得编造或推测）：

{{references}}

---

## 用户问题：{{question}}

## 一、核心概括（80-100字）
用 1-2 句话直接回答问题的核心，给出明确结论，并简要预告将涉及的主要法规和关键点。

## 二、法律依据详解（主体部分，分4-6个要点，每个要点250-320字）
以深度解读、实务分析和案例化说明为主，法条原文仅作支撑。每个要点按以下结构展开：
1. 法规出处：法域 + 《法规全称》第 X 条/第 X 款；来源标注【来源X】。
2. 深度解读（重点）：阐释法律依据、适用范围、核心含义、立法意图、相互关联和实务影响，避免停留在表面复述。
3. 重点提示：合规风险点、特殊情况处理、时限或程序性规定。
4. 实务指引（重点）：给出企业或个人可操作的具体步骤、最佳实践、需准备的材料、落地建议和常见误区。

{structure_section}

**写作规范**
- 使用清晰的 Markdown 层级标题，关键术语加粗，重要法条用行内引用或短引用块（仅摘录关键短语，不要大段复制原文）。
- 回答以分析、解读、实务建议为主，不要堆砌法条原文。
- 不要过多使用 emoji，保持专业、简洁的排版。
- 语言风格：像资深律师给客户做法律解读，专业但不晦涩。
- 字数控制：{total_words_hint}
  - 开头概括：80-100字
  - 每个要点：250-320字
  - 结尾总结：150-200字

**严格约束**
- 必须基于参考资料，所有论断有法条支撑，引用标注【来源X】。
- 严禁编造资料中没有的信息。
- 如遇资料不足，明确说明："关于XX方面，现有法规资料暂未提供明确规定..."

请开始您的专业解答：
"""

        prompt = prompt_template.format(
            history=history_text,
            references=references_text,
            question=question
        )
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """调用远程API（使用智能问答模型）"""
        return self._call_llm_with_model(prompt, model=settings.llm_qa_model, fallback_model=settings.llm_qa_fallback_model)

    def _call_llm_with_model(self, prompt: str, model: Optional[str] = None, 
                           response_format: Optional[Dict] = None, 
                           enable_thinking: Optional[bool] = None,
                           fallback_model: Optional[str] = None) -> str:
        """调用远程API（支持指定模型，带fallback降级机制）
        
        Args:
            prompt: 提示词
            model: 模型名称，不传则使用 settings.llm_model 默认值
            response_format: 响应格式配置，如 {'type': 'json_object'}，用于结构化输出
            enable_thinking: 是否启用思考模式，None表示使用全局配置，False强制关闭
            fallback_model: 备用模型名称，不传则使用 settings.llm_fallback_model
        
        Returns:
            LLM响应内容
        
        Fallback机制:
            当主模型调用失败（token耗尽、限流等），自动使用 fallback 模型重试一次
        """
        import logging
        from openai import RateLimitError, AuthenticationError, APIError

        logger = logging.getLogger(__name__)

        if not prompt or not prompt.strip():
            logger.error(f"[LLM] prompt 为空！调用栈已截断，请检查上游 prompt 构建")
            raise ValueError("prompt 不能为空")

        base_url = settings.llm_api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key = settings.llm_api_key or ""
        target_model = model or settings.llm_model or "qwen-plus"
        fallback_model = fallback_model or settings.llm_fallback_model or "qwen3.5-plus"

        thinking_mode = enable_thinking if enable_thinking is not None else settings.llm_enable_thinking

        def _call_model(m: str):
            logger.info(f"[LLM] 调用 LLM，模型={m}，prompt 长度={len(prompt)}，前100字: {prompt[:100]!r}")
            client = OpenAI(api_key=api_key, base_url=base_url)
            
            kwargs = {
                "model": m,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens,
                "stream": False,
                "extra_body": {"enable_thinking": thinking_mode}
            }
            
            if response_format:
                kwargs["response_format"] = response_format
            
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()

        try:
            return _call_model(target_model)
        except (RateLimitError, AuthenticationError) as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "token" in error_msg or "rate limit" in error_msg or "429" in error_msg or "402" in error_msg:
                logger.warning(f"[LLM] 主模型 {target_model} 调用失败（token耗尽/限流），错误: {e}，尝试降级到 fallback 模型 {fallback_model}")
                try:
                    result = _call_model(fallback_model)
                    logger.warning(f"[LLM] fallback 模型 {fallback_model} 调用成功")
                    return result
                except Exception as fallback_e:
                    logger.error(f"[LLM] fallback 模型 {fallback_model} 调用也失败，错误: {fallback_e}")
                    raise
            raise
        except APIError as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "token" in error_msg or "rate limit" in error_msg or "429" in error_msg or "402" in error_msg:
                logger.warning(f"[LLM] 主模型 {target_model} API错误（token耗尽/限流），错误: {e}，尝试降级到 fallback 模型 {fallback_model}")
                try:
                    result = _call_model(fallback_model)
                    logger.warning(f"[LLM] fallback 模型 {fallback_model} 调用成功")
                    return result
                except Exception as fallback_e:
                    logger.error(f"[LLM] fallback 模型 {fallback_model} 调用也失败，错误: {fallback_e}")
                    raise
            raise

    def _call_llm_stream(self, prompt: str, model: Optional[str] = None, fallback_model: Optional[str] = None):
        """流式调用远程API（支持指定模型，带fallback降级机制）

        Args:
            prompt: 提示词
            model: 模型名称，不传则使用 settings.llm_qa_model 默认值
            fallback_model: 备用模型名称，不传则使用 settings.llm_fallback_model
        """
        import logging
        from openai import RateLimitError, AuthenticationError, APIError

        logger = logging.getLogger(__name__)

        base_url = settings.llm_api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key = settings.llm_api_key or ""
        target_model = model or settings.llm_qa_model
        fallback_model = fallback_model or settings.llm_fallback_model or "qwen3.5-plus"
        logger.info(f"[LLM] _call_llm_stream: 传入 model={model!r}, qa_model={settings.llm_qa_model!r}, 解析结果 target={target_model!r}, fallback={fallback_model!r}")

        def _stream_model(m: str):
            logger.info(f"[LLM] 流式调用 LLM，模型={m}")
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                stream=True,
                extra_body={"enable_thinking": settings.llm_enable_thinking}
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        try:
            yield from _stream_model(target_model)
        except (RateLimitError, AuthenticationError) as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "token" in error_msg or "rate limit" in error_msg or "429" in error_msg or "402" in error_msg:
                logger.warning(f"[LLM] 流式主模型 {target_model} 调用失败（token耗尽/限流），错误: {e}，尝试降级到 fallback 模型 {fallback_model}")
                try:
                    yield from _stream_model(fallback_model)
                    logger.warning(f"[LLM] fallback 模型 {fallback_model} 流式调用成功")
                    return
                except Exception as fallback_e:
                    logger.error(f"[LLM] fallback 模型 {fallback_model} 流式调用也失败，错误: {fallback_e}")
                    raise
            raise
        except APIError as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "token" in error_msg or "rate limit" in error_msg or "429" in error_msg or "402" in error_msg:
                logger.warning(f"[LLM] 流式主模型 {target_model} API错误（token耗尽/限流），错误: {e}，尝试降级到 fallback 模型 {fallback_model}")
                try:
                    yield from _stream_model(fallback_model)
                    logger.warning(f"[LLM] fallback 模型 {fallback_model} 流式调用成功")
                    return
                except Exception as fallback_e:
                    logger.error(f"[LLM] fallback 模型 {fallback_model} 流式调用也失败，错误: {fallback_e}")
                    raise
            raise

    def _call_llm_stream_with_messages(self, messages: List[Dict], model: Optional[str] = None, 
                                        fallback_model: Optional[str] = None):
        """流式调用远程API（使用标准 messages 数组，带 fallback 降级机制）

        Args:
            messages: 标准 OpenAI 格式的 messages 数组 [{"role": "system/user/assistant", "content": "..."}]
            model: 模型名称，不传则使用 settings.llm_qa_model 默认值
            fallback_model: 备用模型名称，不传则使用 settings.llm_fallback_model
        """
        import logging
        from openai import RateLimitError, AuthenticationError, APIError

        logger = logging.getLogger(__name__)

        base_url = settings.llm_api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key = settings.llm_api_key or ""
        target_model = model or settings.llm_qa_model
        fallback_model = fallback_model or settings.llm_fallback_model or "qwen3.5-plus"
        logger.info(f"[LLM] _call_llm_stream_with_messages: 模型={target_model}, messages数量={len(messages)}")

        def _stream_model(m: str, msgs: List[Dict]):
            logger.info(f"[LLM] 流式调用 LLM，模型={m}")
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=m,
                messages=msgs,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                stream=True,
                extra_body={"enable_thinking": settings.llm_enable_thinking}
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        try:
            yield from _stream_model(target_model, messages)
        except (RateLimitError, AuthenticationError) as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "token" in error_msg or "rate limit" in error_msg or "429" in error_msg or "402" in error_msg:
                logger.warning(f"[LLM] 流式主模型 {target_model} 调用失败（token耗尽/限流），错误: {e}，尝试降级到 fallback 模型 {fallback_model}")
                try:
                    yield from _stream_model(fallback_model, messages)
                    logger.warning(f"[LLM] fallback 模型 {fallback_model} 流式调用成功")
                    return
                except Exception as fallback_e:
                    logger.error(f"[LLM] fallback 模型 {fallback_model} 流式调用也失败，错误: {fallback_e}")
                    raise
            raise
        except APIError as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "token" in error_msg or "rate limit" in error_msg or "429" in error_msg or "402" in error_msg:
                logger.warning(f"[LLM] 流式主模型 {target_model} API错误（token耗尽/限流），错误: {e}，尝试降级到 fallback 模型 {fallback_model}")
                try:
                    yield from _stream_model(fallback_model, messages)
                    logger.warning(f"[LLM] fallback 模型 {fallback_model} 流式调用成功")
                    return
                except Exception as fallback_e:
                    logger.error(f"[LLM] fallback 模型 {fallback_model} 流式调用也失败，错误: {fallback_e}")
                    raise
            raise

    def _build_messages(self, question: str, contexts: List[Dict], history: Optional[List[Dict]] = None) -> List[Dict]:
        """构建标准 messages 数组（system + history + 当前用户消息）

        Args:
            question: 当前用户问题
            contexts: 检索到的法规上下文
            history: 对话历史列表，每条包含 role 和 content

        Returns:
            标准 OpenAI 格式的 messages 数组
        """
        top_contexts = contexts[:5]
        references = []

        for i, ctx in enumerate(top_contexts, 1):
            article_num = ctx['metadata'].get('article_number', '')
            title = ctx['metadata'].get('title', '')
            jurisdiction = ctx['metadata'].get('jurisdiction', '')
            law_id = ctx['metadata'].get('law_id', '')
            similarity = ctx['similarity']
            content = ctx['content']

            references.append(
                f"【来源{i}】{jurisdiction}《{title}》{article_num}\n法ID: {law_id}\n相似度: {similarity:.4f}\n{content}")

        references_text = "\n\n".join(references)

        jurisdictions_in_contexts = set()
        for ctx in top_contexts:
            jur = ctx['metadata'].get('jurisdiction', '')
            if jur:
                jurisdictions_in_contexts.add(jur)
        is_multi_jurisdiction = len(jurisdictions_in_contexts) > 1

        is_follow_up = bool(history and len(history) >= 1)

        if is_multi_jurisdiction:
            structure_section = """
## 三、总结与建议（200-250字）
先根据参考资料中多个法域的规定制作简明对比表，指出主要差异点、共同点和不同法域的严格程度差异；然后精炼总结答案要点，给出行动指引，并列出 2-3 条延伸提醒。"""
            total_words_hint = "总字数：1500-2200字（确保内容详实）"
        else:
            structure_section = """
## 三、总结与建议（150-200字）
精炼总结答案要点，给出行动指引，并列出 2-3 条延伸提醒。"""
            total_words_hint = "总字数：1200-1800字（确保内容详实）"

        if is_follow_up:
            system_prompt = """你是一位精通境外法律法规的资深法律顾问。请严格基于参考资料回答用户追问，不得编造。

【重要约束 - 资料引用规则】
- 每次追问都会重新检索法规资料，请仅引用当前用户消息中【参考资料库】标注的来源。
- 历史对话中提到的【来源X】是上一轮的资料，已失效，不得引用。

【回答结构要求】
## 一、简要定位（1-2句话）
指出针对哪个具体点补充说明，结合上下文明确追问意图。

## 二、详细展开（分点阐述，每点200-280字，共3-5个要点）
以深度解读和实务分析为主，法条原文仅作支撑。每个要点按以下结构展开：
1. 法规出处：法域 + 《法规全称》第 X 条/第 X 款；来源标注【来源X】。
2. 深度解读（重点）：阐释法律依据、适用范围、核心含义、立法意图和相互关联，避免停留在表面复述。
3. 重点提示：合规风险点、特殊情况、时限或程序性规定。
4. 实务指引（重点）：给出企业或个人可操作的具体步骤、最佳实践、需准备的材料、落地建议。

## 三、总结回顾（150-200字）
归纳核心答案，结合前文给出结论，并列出 2-3 条延伸提醒。

## 四、核心结论重申（2-3句话）
用最简洁的语言重申结论和关键行动项。

【写作规范】
- 使用 Markdown 层级标题，关键术语加粗，重要法条用行内引用或短引用块（仅摘录关键短语，不要大段复制原文）。
- 回答以分析、解读、实务建议为主，不要堆砌法条原文。
- 所有论断必须有法条支撑，引用必须标注【来源X】。
- 语言专业但不晦涩，像资深律师给客户解释。
- 总字数要求：1000-1500字。
- 严禁编造资料中没有的信息；资料不足时明确说明。"""
        else:
            system_prompt = f"""你是一位精通境外法律法规的资深法律顾问，拥有20年实务经验，擅长用通俗易懂的语言解释复杂的法律问题。

【回答结构要求】
## 一、核心概括（80-100字）
用 1-2 句话直接回答问题的核心，给出明确结论，并简要预告将涉及的主要法规和关键点。

## 二、法律依据详解（主体部分，分4-6个要点，每个要点250-320字）
以深度解读、实务分析和案例化说明为主，法条原文仅作支撑。每个要点按以下结构展开：
1. 法规出处：法域 + 《法规全称》第 X 条/第 X 款；来源标注【来源X】。
2. 深度解读（重点）：阐释法律依据、适用范围、核心含义、立法意图、相互关联和实务影响，避免停留在表面复述。
3. 重点提示：合规风险点、特殊情况处理、时限或程序性规定。
4. 实务指引（重点）：给出企业或个人可操作的具体步骤、最佳实践、需准备的材料、落地建议和常见误区。
{structure_section}

【写作规范】
- 使用清晰的 Markdown 层级标题，关键术语加粗，重要法条用行内引用或短引用块（仅摘录关键短语，不要大段复制原文）。
- 回答以分析、解读、实务建议为主，不要堆砌法条原文。
- 不要过多使用 emoji，保持专业、简洁的排版。
- 语言风格：像资深律师给客户做法律解读，专业但不晦涩。
- 字数控制：{total_words_hint}
  - 开头概括：80-100字
  - 每个要点：250-320字
  - 结尾总结：150-200字

【严格约束】
- 必须基于参考资料，所有论断有法条支撑，引用标注【来源X】。
- 严禁编造资料中没有的信息。
- 如遇资料不足，明确说明："关于XX方面，现有法规资料暂未提供明确规定..." """

        messages = [{"role": "system", "content": system_prompt}]

        if history and len(history) >= 1:
            trimmed_history = history[-6:]
            for h in trimmed_history:
                role = h.get("role", "")
                content = h.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        if is_follow_up:
            user_content = f"""【追问 - 以下是为当前问题重新检索的最新法规资料，历史对话中引用的来源已失效，请仅基于以下资料回答】

{references_text}

---

## 用户追问：{question}

请继续解答："""
        else:
            user_content = f"""## 参考资料库（请严格基于以下法规条文回答，不得编造或推测）：

{references_text}

---

## 用户问题：{question}

请开始您的专业解答："""

        messages.append({"role": "user", "content": user_content})
        return messages

    def ask_stream(self, question: str, top_k: int = 5, jurisdictions: Optional[List[str]] = None,
                    history: Optional[List[Dict]] = None):
        """流式问答"""
        total_start = time.time()

        try:
            t1 = time.time()
            if jurisdictions:
                detected_jurisdictions = jurisdictions
                allow_fallback = False
            else:
                detected_jurisdictions = self._detect_jurisdictions(question, history)
                allow_fallback = True
            t_detect = time.time() - t1

            t2 = time.time()
            contexts = self._retrieve_contexts_with_fallback(question, top_k, detected_jurisdictions, allow_fallback)
            t_retrieve = time.time() - t2
        except Exception as e:
            yield {"type": "error", "data": f"检索法律法规信息时发生错误：{str(e)}"}
            return

        if not contexts:
            yield {"type": "error", "data": "未找到相关的法律法规信息。"}
            return

        sources = []
        source_jurisdictions = set()
        for ctx in contexts:
            jur = ctx['metadata'].get('jurisdiction', '')
            if jur:
                source_jurisdictions.add(jur)
            sources.append({
                "law_id": ctx['metadata'].get('law_id', ''),
                "title": ctx['metadata'].get('title', ''),
                "article_number": ctx['metadata'].get('article_number', ''),
                "jurisdiction": jur,
                "similarity": ctx['similarity'],
                "content": ctx['content'][:100] + "..." if len(ctx['content']) > 100 else ctx['content']
            })

        # 将实际检索到的法域附加到 sources 返回，便于前端保存到历史记录
        detected_jurisdictions = list(source_jurisdictions) if not jurisdictions else detected_jurisdictions

        t3 = time.time()
        messages = self._build_messages(question, contexts, history)
        t_prompt = time.time() - t3

        yield {"type": "sources", "sources": sources, "retrieved_count": len(contexts), "jurisdictions": detected_jurisdictions}

        try:
            t4 = time.time()
            llm_start_time = t4

            for chunk in self._call_llm_stream_with_messages(messages, fallback_model=settings.llm_qa_fallback_model):
                yield {"type": "content", "data": chunk}

            t_llm = time.time() - t4
        except Exception as e:
            yield {"type": "error", "data": f"生成回答时发生错误：{str(e)}"}
            return

        total_time = time.time() - total_start
        print(
            f"【QA流式计时】总耗时: {total_time:.4f}s | 法域检测: {t_detect:.4f}s | 检索: {t_retrieve:.4f}s | 构建Prompt: {t_prompt:.4f}s | LLM调用: {t_llm:.4f}s")

        yield {"type": "end", "timing": {
            "detect_jurisdictions": f"{t_detect:.4f}s",
            "retrieve_contexts": f"{t_retrieve:.4f}s",
            "build_prompt": f"{t_prompt:.4f}s",
            "call_llm": f"{t_llm:.4f}s",
            "total": f"{total_time:.4f}s"
        }}

    def ask_stream_with_image(self, question: str, top_k: int = 5, jurisdictions: Optional[List[str]] = None,
                              history: Optional[List[Dict]] = None, images: Optional[List[Dict]] = None):
        """
        支持图片的流式问答（使用Qwen3.6-Plus原生多模态）

        Args:
            question: 用户问题
            top_k: 检索结果数量
            jurisdictions: 法域列表
            history: 对话历史
            images: 图片列表，每项包含 {data: base64字符串, mime_type: 图片类型}
        """
        import logging
        logger = logging.getLogger(__name__)
        total_start = time.time()

        try:
            # 1. 检索法规上下文（支持自适应回退）
            t1 = time.time()
            if jurisdictions:
                detected_jurisdictions = jurisdictions
                allow_fallback = False
            else:
                detected_jurisdictions = self._detect_jurisdictions(question, history)
                allow_fallback = True
            t_detect = time.time() - t1

            t2 = time.time()
            contexts = self._retrieve_contexts_with_fallback(question, top_k, detected_jurisdictions, allow_fallback)
            t_retrieve = time.time() - t2
        except Exception as e:
            logger.error(f"[多模态QA] 检索失败: {str(e)}")
            yield {"type": "error", "data": f"检索法律法规信息时发生错误：{str(e)}"}
            return

        if not contexts:
            yield {"type": "error", "data": "未找到相关的法律法规信息。"}
            return

        # 2. 构建多模态消息（核心改动）
        try:
            messages = self._build_multimodal_messages(question, contexts, history, images)
        except Exception as e:
            logger.error(f"[多模态QA] 构建消息失败: {str(e)}")
            yield {"type": "error", "data": f"构建请求消息时发生错误：{str(e)}"}
            return

        # 3. 先返回sources
        sources = []
        source_jurisdictions = set()
        for ctx in contexts[:5]:
            jur = ctx['metadata'].get('jurisdiction', '')
            if jur:
                source_jurisdictions.add(jur)
            sources.append({
                "law_id": ctx['metadata'].get('law_id', ''),
                "title": ctx['metadata'].get('title', ''),
                "article_number": ctx['metadata'].get('article_number', ''),
                "jurisdiction": jur,
                "similarity": ctx['similarity'],
                "content": ctx['content'][:100] + "..." if len(ctx['content']) > 100 else ctx['content']
            })

        detected_jurisdictions = list(source_jurisdictions) if not jurisdictions else detected_jurisdictions

        yield {"type": "sources", "sources": sources, "retrieved_count": len(contexts), "jurisdictions": detected_jurisdictions}

        # 4. 流式调用Qwen3.6-Plus
        try:
            t3 = time.time()
            for chunk in self._call_qwen36_plus_stream(messages):
                yield {"type": "content", "data": chunk}

            t_llm = time.time() - t3
        except Exception as e:
            logger.error(f"[多模态QA] LLM调用失败: {str(e)}")
            yield {"type": "error", "data": f"生成回答时发生错误：{str(e)}"}
            return

        total_time = time.time() - total_start
        print(
            f"【多模态QA流式计时】总耗时: {total_time:.4f}s | 法域检测: {t_detect:.4f}s | 检索: {t_retrieve:.4f}s | LLM调用: {t_llm:.4f}s")

        yield {"type": "end", "timing": {
            "detect_jurisdictions": f"{t_detect:.4f}s",
            "retrieve_contexts": f"{t_retrieve:.4f}s",
            "call_llm": f"{t_llm:.4f}s",
            "total": f"{total_time:.4f}s"
        }}

    def _build_multimodal_messages(self, question: str, contexts: List[Dict], history: Optional[List[Dict]] = None,
                                   images: Optional[List[Dict]] = None) -> List[Dict]:
        """
        构建Qwen3.6-Plus多模态消息格式

        Args:
            question: 用户问题
            contexts: 检索到的法规上下文
            history: 对话历史
            images: 图片列表

        Returns:
            符合OpenAI多模态格式的messages列表
        """
        import logging
        logger = logging.getLogger(__name__)

        # System prompt
        is_follow_up = bool(history and len(history) >= 1)
        if is_follow_up:
            system_prompt = """你是一位精通境外法律法规的资深法律顾问，拥有20年实务经验。
请严格基于【参考资料】回答用户追问；如果用户上传了图片，请结合图片内容进行分析和解读。
使用通俗易懂的语言解释复杂的法律问题。

【重要约束 - 资料引用规则】
- 每次追问都会重新检索法规资料，请仅引用当前用户消息中【参考资料】标注的来源。
- 历史对话中提到的【来源X】是上一轮的资料，已失效，不得引用。

【输出格式要求】请严格按照以下结构输出，使用清晰的 Markdown 标题，不要添加多余 emoji：

## 一、核心概括（80-100字）
用 1-2 句话直接给出明确结论，并简要预告涉及的主要法规和关键点。

## 二、法律依据详解（主体部分，分 4-6 个要点，每点 250-320 字）
以深度解读、实务分析和案例化说明为主，法条原文仅作支撑。每个要点按以下结构展开：
1. 法规出处：法域 + 《法规全称》第 X 条/第 X 款；来源标注【来源X】。
2. 深度解读（重点）：阐释法律依据、适用范围、核心含义、立法意图、相互关联和实务影响，避免停留在表面复述。
3. 重点提示：合规风险点、特殊情况处理、时限或程序性规定。
4. 实务指引（重点）：给出企业或个人可操作的具体步骤、最佳实践、需准备的材料、落地建议和常见误区。

## 三、总结与建议（150-200字）
精炼总结答案要点，给出明确的行动指引，并列出 2-3 条延伸提醒。

【写作规范】
- 使用 Markdown 层级标题，关键术语加粗，重要法条用行内引用或短引用块（仅摘录关键短语，不要大段复制原文）。
- 回答以分析、解读、实务建议为主，不要堆砌法条原文。
- 所有论断必须有法条支撑，引用必须标注【来源X】。
- 语言风格专业但不晦涩，像资深律师给客户做法律解读。
- 总字数：单法域问题 1200-1800 字，多法域对比问题 1500-2200 字。
- 严禁编造资料中没有的信息；资料不足时明确说明。"""
        else:
            system_prompt = """你是一位精通境外法律法规的资深法律顾问，拥有20年实务经验。
请严格基于【参考资料】回答用户问题；如果用户上传了图片，请结合图片内容进行分析和解读。
使用通俗易懂的语言解释复杂的法律问题。

【输出格式要求】请严格按照以下结构输出，使用清晰的 Markdown 标题，不要添加多余 emoji：

## 一、核心概括（80-100字）
用 1-2 句话直接给出明确结论，并简要预告涉及的主要法规和关键点。

## 二、法律依据详解（主体部分，分 4-6 个要点，每点 250-320 字）
以深度解读、实务分析和案例化说明为主，法条原文仅作支撑。每个要点按以下结构展开：
1. 法规出处：法域 + 《法规全称》第 X 条/第 X 款；来源标注【来源X】。
2. 深度解读（重点）：阐释法律依据、适用范围、核心含义、立法意图、相互关联和实务影响，避免停留在表面复述。
3. 重点提示：合规风险点、特殊情况处理、时限或程序性规定。
4. 实务指引（重点）：给出企业或个人可操作的具体步骤、最佳实践、需准备的材料、落地建议和常见误区。

## 三、总结与建议（150-200字）
精炼总结答案要点，给出明确的行动指引，并列出 2-3 条延伸提醒。

【写作规范】
- 使用 Markdown 层级标题，关键术语加粗，重要法条用行内引用或短引用块（仅摘录关键短语，不要大段复制原文）。
- 回答以分析、解读、实务建议为主，不要堆砌法条原文。
- 所有论断必须有法条支撑，引用必须标注【来源X】。
- 语言风格专业但不晦涩，像资深律师给客户做法律解读。
- 总字数：单法域问题 1200-1800 字，多法域对比问题 1500-2200 字。
- 严禁编造资料中没有的信息；资料不足时明确说明。"""

        # 构建参考资料文本
        references_text = self._build_references_text_for_multimodal(contexts[:5])

        # 构建 messages 数组
        messages = [{"role": "system", "content": system_prompt}]

        # 添加历史对话（作为独立的 user/assistant 消息）
        if history and len(history) >= 1:
            trimmed_history = history[-6:]
            for h in trimmed_history:
                role = h.get("role", "")
                content = h.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        # 当前用户消息（文本 + 图片 + 参考资料）
        user_content = []

        if is_follow_up:
            text_part = f"""【追问 - 以下是为当前问题重新检索的最新法规资料，历史对话中引用的来源已失效，请仅基于以下资料回答】

{references_text}

---
用户追问：{question}"""
        else:
            text_part = f"""{references_text}

---
用户问题：{question}"""

        user_content.append({"type": "text", "text": text_part})

        # 图片部分（如果有）
        if images and settings.enable_image_support:
            if len(images) > settings.max_images_per_request:
                logger.warning(f"[多模态QA] 图片数量超限：{len(images)}张，最多支持{settings.max_images_per_request}张")
                images = images[:settings.max_images_per_request]

            for idx, img in enumerate(images, 1):
                # 验证图片格式
                mime_type = img.get('mime_type', '')
                img_data = img.get('data', '')

                if not img_data:
                    logger.warning(f"[多模态QA] 第{idx}张图片数据为空，跳过")
                    continue

                # 提取格式（如 image/jpeg -> jpeg）
                format_ext = mime_type.split('/')[-1] if '/' in mime_type else mime_type
                if format_ext not in settings.supported_image_formats:
                    logger.warning(f"[多模态QA] 不支持的图片格式: {mime_type}，跳过")
                    continue

                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{img_data}",
                        "detail": "high"  # 高清模式，确保OCR准确
                    }
                })

            logger.info(f"[多模态QA] 已添加 {len([c for c in user_content if c.get('type') == 'image_url'])} 张图片")

        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_references_text_for_multimodal(self, contexts: List[Dict]) -> str:
        """为多模态场景构建参考资料文本"""
        references = []

        for i, ctx in enumerate(contexts, 1):
            article_num = ctx['metadata'].get('article_number', '')
            title = ctx['metadata'].get('title', '')
            jurisdiction = ctx['metadata'].get('jurisdiction', '')
            law_id = ctx['metadata'].get('law_id', '')
            similarity = ctx['similarity']
            content = ctx['content']

            references.append(
                f"【来源{i}】{jurisdiction}《{title}》{article_num}\n法ID: {law_id}\n相似度: {similarity:.4f}\n{content}")

        return "\n\n".join(references)

    def _call_qwen36_plus_stream(self, messages: List[Dict]):
        """
        调用多模态流式API（使用智能问答模型，带fallback降级机制）

        Args:
            messages: 多模态消息列表

        Yields:
            流式文本片段
        """
        import logging
        from openai import RateLimitError, AuthenticationError, APIError

        logger = logging.getLogger(__name__)

        base_url = settings.llm_api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key = settings.llm_api_key or ""
        target_model = settings.llm_qa_model
        fallback_model = settings.llm_qa_fallback_model or settings.llm_fallback_model or "qwen3.5-plus"

        def _stream_model(m: str, msgs: List[Dict]):
            logger.info(f"[多模态LLM] 调用模型: {m}, 消息数量: {len(msgs)}")
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=m,
                messages=msgs,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                stream=True,
                extra_body={"enable_thinking": settings.llm_enable_thinking}
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        try:
            yield from _stream_model(target_model, messages)
        except (RateLimitError, AuthenticationError) as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "token" in error_msg or "rate limit" in error_msg or "429" in error_msg or "402" in error_msg:
                logger.warning(f"[多模态LLM] 主模型 {target_model} 调用失败（token耗尽/限流），错误: {e}，尝试降级到 fallback 模型 {fallback_model}")
                try:
                    yield from _stream_model(fallback_model, messages)
                    logger.warning(f"[多模态LLM] fallback 模型 {fallback_model} 调用成功")
                    return
                except Exception as fallback_e:
                    logger.error(f"[多模态LLM] fallback 模型 {fallback_model} 调用也失败，错误: {fallback_e}")
                    raise
            raise
        except APIError as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "token" in error_msg or "rate limit" in error_msg or "429" in error_msg or "402" in error_msg:
                logger.warning(f"[多模态LLM] 主模型 {target_model} API错误（token耗尽/限流），错误: {e}，尝试降级到 fallback 模型 {fallback_model}")
                try:
                    yield from _stream_model(fallback_model, messages)
                    logger.warning(f"[多模态LLM] fallback 模型 {fallback_model} 调用成功")
                    return
                except Exception as fallback_e:
                    logger.error(f"[多模态LLM] fallback 模型 {fallback_model} 调用也失败，错误: {fallback_e}")
                    raise
            raise
