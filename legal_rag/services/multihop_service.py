"""跨法域对齐多跳问答服务（P1）：A 法域《B》第 a 条 → C 法域对应法规《B′》→ 目标条款。

跳跨法规/法域，靠「对应法规」与实体消歧后的同一概念连接；可靠性写进数据：
similar_law=L1（主题相似，不得表述为同一法规），concept_matrix=L2（同一 canonical_id）。
成本：横向对齐与概念交叉校验全走 ES 与预计算质心，零 LLM，仅最终对比合成调 1 次；
未配置密钥时仍返回完整路径与条款证据（answer 为空）。
"""

import logging
import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from core.config import settings
from core.jurisdiction import normalize_jurisdictions
from core.llm import get_chat_model
from repositories.elasticsearch import ElasticsearchRepository
from services.kg_extraction_service import _RELATION_LABELS
from services.multihop_path import HopStep, ReasoningPath, path_text
from services.similar_law_service import SimilarLawService

logger = logging.getLogger(__name__)

_MAX_SOURCE_CANDIDATES = 6   # 源侧定位时最多尝试的候选条款数（同法规可多条）
_MAX_TARGET_LAWS = 3         # 送入对比的对应法规上限
_MAX_TARGET_ARTICLES = 3     # 每部对应法规的相关条款上限
_MAX_SHARED_CONCEPTS = 5     # 每部对应法规的共享概念上限
_MAX_NAME_LEN = 40           # 超过视为整句描述串，不作概念对齐
_MAX_PATHS = 20              # 候选路径上限（控制 prompt 长度）
_MAX_SNIPPETS = 20           # 回取原文的条款上限
# 法域名是结构性信息，检索条款时是噪声（「在日本和澳门法中」会把跨境条款顶上来）
_JURISDICTION_NAMES = (
    "中国香港", "香港", "中国澳门", "澳门", "台湾", "日本", "韩国", "新加坡", "马来西亚", "泰国", "越南",
    "美国", "欧盟", "英国", "中国大陆", "中国", "内地", "加拿大", "澳大利亚", "印度",
    "菲律宾", "印度尼西亚", "印尼", "柬埔寨", "缅甸", "老挝", "文莱",
)
_JURISDICTION_SUFFIX = r"(?:特别行政区|特区|法域|地区|的法律|法律|法规|法)?"


# 通用指代型「概念」：实体消歧会把它们归并到同一 canonical_id，但对跨法域对齐无信息量
_GENERIC_CONCEPTS = {
    "this act", "the act", "act", "this ordinance", "the ordinance", "ordinance",
    "this regulation", "the regulation", "regulation", "these rules", "this code",
    "the code", "this law", "the law", "本法", "本条例", "本规定", "本办法",
}


class _CrossJurisdictionAnswer(BaseModel):
    """对比合成答案的结构化输出；路径仍只允许引用候选编号。"""

    answer: str = Field(description="回答正文，简体中文，不使用换行符")
    used_path_indexes: List[int] = Field(
        default_factory=list,
        description="实际依据的候选路径编号（从 1 开始，取自【可推理路径】；未使用则留空）",
    )
    cited_articles: List[str] = Field(
        default_factory=list, description="引用条款，格式 law_id::article_number"
    )


class MultiHopService:
    """跨法域对齐链问答"""

    def __init__(
        self,
        repo: Optional[ElasticsearchRepository] = None,
        similar_service: Optional[SimilarLawService] = None,
    ):
        self.repo = repo or ElasticsearchRepository()
        self.similar = similar_service or SimilarLawService(self.repo)
        self.api_key = settings.llm_api_key or ""
        self.model = settings.llm_extraction_model or "deepseek-flash"

    # ================================================================
    #  1. 源侧定位：问题 → A 法域《B》第 a 条
    # ================================================================
    @staticmethod
    def _hit_to_source(meta: Dict, doc: str, score: float, similarity=None) -> Dict:
        # _scores 是 RRF 分（量级 ~0.03，不能当百分比展示）；_similarities 才是真实余弦
        shown = similarity if similarity is not None else score
        return {
            "law_id": meta.get("law_id", ""),
            "title": meta.get("title", ""),
            "jurisdiction": meta.get("jurisdiction", ""),
            "article_number": meta.get("article_number", ""),
            "content": doc or "",
            "score": round(float(shown or 0), 4),
        }

    @staticmethod
    def _by_cosine(hits: List[Dict]) -> List[Dict]:
        """按真实余弦排序（就地返回）：RRF 融合会把词面巧合的跨语种条款顶上来，故保持纯余弦。"""
        hits.sort(key=lambda hit: -float(hit.get("score", 0) or 0))
        return hits

    def _search_in_law(self, question: str, law_id: str, n_results: int = 1) -> List[Dict]:
        """按 law_id 收窄的混合检索（where 同时作用于 KNN 与 BM25 通道）。"""
        res = self.repo.hybrid_search(question, n_results=n_results, where={"law_id": law_id})
        sims = (res.get("_similarities") or [[]])[0]
        out: List[Dict] = []
        for i, (meta, doc, score) in enumerate(zip(
            (res.get("metadatas") or [[]])[0],
            (res.get("documents") or [[]])[0],
            (res.get("_scores") or [[]])[0],
        )):
            similarity = sims[i] if i < len(sims) else None
            out.append(self._hit_to_source(meta, doc, score, similarity))
        return out

    @staticmethod
    def _strip_jurisdiction_terms(question: str) -> str:
        """去掉问题里的法域名：检索的是条款内容，结构性词汇会让「境外/跨境」条款在向量与 BM25 上同时占优。"""
        text = question or ""
        for name in sorted(_JURISDICTION_NAMES, key=len, reverse=True):
            text = re.sub(
                rf"{re.escape(name)}{_JURISDICTION_SUFFIX}", " ", text
            )
        text = re.sub(r"[\s，,、]+", " ", text).strip()
        return text or question

    def _locate_source(
        self, question: str, law_id: Optional[str] = None, article: Optional[str] = None,
        jurisdictions: Optional[List[str]] = None,
    ) -> List[Dict]:
        """返回按优先级排列的源条款候选（同法规允许多条）。

        jurisdictions 非空时只保留这些法域的命中并按问题中的顺序排序，避免被语义更贴近的其他法域抢走；
        每部法规保留前 2 条而非仅检索第一名：top-10 常同时含「跨境」与「安全措施」两类条款，按法规去重会丢掉真正对题的那条。
        """
        if law_id:
            centroid = self.repo.get_law_centroid(law_id) or {}
            hit = (self._by_cosine(self._search_in_law(
                self._strip_jurisdiction_terms(question), law_id, 3
            )) or [None])[0] or {}
            article = article or hit.get("article_number", "")
            content = self.repo.get_article_full_text(law_id, article) if article else None
            return [{
                "law_id": law_id,
                "title": centroid.get("title") or hit.get("title", ""),
                "jurisdiction": centroid.get("jurisdiction") or hit.get("jurisdiction", ""),
                "article_number": article,
                "content": content or hit.get("content", ""),
                "score": hit.get("score", 0.0),
            }]

        cleaned = self._strip_jurisdiction_terms(question)
        res = self.repo.hybrid_search(question, n_results=10)
        metas = (res.get("metadatas") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        scores = (res.get("_scores") or [[]])[0]
        sims = (res.get("_similarities") or [[]])[0]

        preferred = list(jurisdictions or [])
        law_hits: Dict[str, Dict] = {}
        for i, (meta, doc, _score) in enumerate(zip(metas, docs, scores)):
            lid = meta.get("law_id", "")
            if not lid:
                continue
            if preferred and meta.get("jurisdiction", "") not in preferred:
                continue
            law_hits.setdefault(lid, {"meta": meta, "doc": doc,
                                      "similarity": sims[i] if i < len(sims) else None})

        candidates: List[Dict] = []
        seen_pairs = set()
        # 每部候选法规做一次法内检索，最多 6 部：既覆盖问题点名的法域，又避免一次问答放大成十几次 ES 往返。
        for lid, info in list(law_hits.items())[:6]:
            # 同法规内用「去掉法域名的问题」重新排序，取前 2 条作为该法规的源候选
            hits = self._by_cosine(self._search_in_law(cleaned, lid, 6))
            for hit in hits[:2]:
                art = hit.get("article_number", "")
                if not art or (lid, art) in seen_pairs:
                    continue
                seen_pairs.add((lid, art))
                source = dict(hit)
                source["jurisdiction"] = source.get("jurisdiction") or info["meta"].get("jurisdiction", "")
                source["law_score"] = info["similarity"]
                candidates.append(source)
            if len(candidates) >= _MAX_SOURCE_CANDIDATES:
                break

        candidates.sort(key=lambda c: (
            preferred.index(c["jurisdiction"]) if c.get("jurisdiction") in preferred else len(preferred),
            -float(c.get("score", 0) or 0),
        ))
        return candidates[:_MAX_SOURCE_CANDIDATES]

    # ================================================================
    #  2. 横向对齐：A 法域《B》→ C 法域《B′》
    # ================================================================
    @staticmethod
    def _pick_targets(alignment: Dict, source_jurisdiction: str, top_k: int,
                      preferred_jurisdictions: Optional[List[str]] = None) -> List[Dict]:
        laws = [
            law for law in (alignment.get("similar_laws") or [])
            if law.get("law_id")
        ]
        preferred = set(preferred_jurisdictions or [])
        cross = [
            law for law in laws
            if not source_jurisdiction or law.get("jurisdiction") != source_jurisdiction
        ]
        if preferred:
            # 问题点名的法域是唯一候选，否则 top_k 会被未点名的法域占位
            named = [law for law in cross if law.get("jurisdiction") in preferred]
            if named:
                named.sort(key=lambda law: -float(law.get("composite_score", 0) or 0))
                return named[:top_k]
        if preferred:
            # 问题里点名的法域没有对应法规时，退回全部候选但按点名优先排序
            laws.sort(key=lambda law: (
                0 if law.get("jurisdiction") in preferred else 1,
                -float(law.get("composite_score", 0) or 0),
            ))
            return laws[:top_k]
        # 未限定法域：跨法域优先；若全库只有同法域相似法规，仍返回但由 same_jurisdiction 标注
        return (cross or laws)[:top_k]

    # ================================================================
    #  3. 目标侧条款：在《B′》内回答同一问题
    # ================================================================
    def _target_articles(self, question: str, source: Dict, target: Dict) -> List[Dict]:
        # 用「问题 + 源条款原文」作查询：跨语言向量模型能把对应条款拉近，只用问题易命中目标法规的通用程序条款。
        query = question
        if source.get("content"):
            query = f"{question}\n{source['content'][:600]}"
        hits = self._by_cosine(self._search_in_law(
            query, target["law_id"], _MAX_TARGET_ARTICLES * 3
        ))
        articles: List[Dict] = []
        seen = set()
        for hit in hits:
            art = hit.get("article_number", "")
            if not art or art in seen:
                continue
            seen.add(art)
            articles.append({
                "article_number": art,
                "content": hit.get("content", ""),
                "similarity": hit.get("score", 0.0),
                "matched_by": "retrieval",
            })
            if len(articles) >= _MAX_TARGET_ARTICLES:
                break
        # 相似法规评分时命中的条款是另一路证据（主题相似正是由它们贡献）
        for chunk in target.get("top_matched_chunks") or []:
            art = chunk.get("article_number", "")
            if not art or art in seen:
                continue
            seen.add(art)
            articles.append({
                "article_number": art,
                "content": chunk.get("content_preview", ""),
                "similarity": chunk.get("similarity", 0.0),
                "matched_by": "similar_law",
            })
            if len(articles) >= _MAX_TARGET_ARTICLES:
                break
        return articles[:_MAX_TARGET_ARTICLES]

    # ================================================================
    #  4. 概念交叉校验：同一 canonical_id 在两侧条款中出现 → L2 证据
    # ================================================================
    @staticmethod
    def _index_concepts(triples: List[Dict], articles: Optional[set]) -> Dict[str, Dict]:
        """canonical_id → 首次出现信息；articles 非空时只收这些条款的边。"""
        index: Dict[str, Dict] = {}
        for t in triples:
            article = t.get("source_article", "")
            if articles and article not in articles:
                continue
            for side in ("head", "tail"):
                ent = t.get(side) or {}
                cid = ent.get("canonical_id") or ent.get("id")
                name = ent.get("name", "")
                etype = ent.get("type", "")
                if not cid or etype == "Article" or not name or len(name) > _MAX_NAME_LEN:
                    continue
                if name.strip().strip("《》").lower() in _GENERIC_CONCEPTS:
                    continue
                index.setdefault(cid, {
                    "entity_id": cid,
                    "name": name,
                    "type": etype,
                    "article": article,
                    "relation": t.get("relation", ""),
                    "side": side,
                    "confidence": float(t.get("confidence", 0) or 0),
                })
        return index

    def _shared_concepts(
        self, source: Dict, target: Dict, target_articles: List[Dict]
    ) -> List[Dict]:
        src_triples = self.repo.get_kg_triples(source["law_id"])
        tgt_triples = self.repo.get_kg_triples(target["law_id"])
        if not src_triples or not tgt_triples:
            return []

        src_arts = {source["article_number"]} if source.get("article_number") else None
        tgt_arts = {a["article_number"] for a in target_articles} or None

        # 先按具体条款对齐（最强证据），无交集再退回法规级；scope 会写进路径与前端，避免把法规内其他条款伪装成源条款内容。
        src_index = self._index_concepts(src_triples, src_arts)
        tgt_index = self._index_concepts(tgt_triples, tgt_arts)
        common = set(src_index) & set(tgt_index)
        scope = "article"
        if not common:
            src_index = self._index_concepts(src_triples, None)
            tgt_index = self._index_concepts(tgt_triples, None)
            common = set(src_index) & set(tgt_index)
            scope = "law"

        concepts = []
        for cid in common:
            s, t = src_index[cid], tgt_index[cid]
            concepts.append({
                "entity_id": cid,
                "name": s["name"],
                "type": s["type"],
                "scope": scope,
                "source_article": s["article"],
                "source_relation": s["relation"],
                "source_side": s["side"],
                "target_article": t["article"],
                "target_relation": t["relation"],
                "target_side": t["side"],
                "confidence": min(s["confidence"], t["confidence"]),
            })
        # 两侧条款都命中的优先，其次置信度高、名称短的（更可能是真概念）
        concepts.sort(key=lambda c: (
            0 if (c["source_article"] in (src_arts or set()) and
                  c["target_article"] in (tgt_arts or set())) else 1,
            -c["confidence"],
            len(c["name"]),
        ))
        return concepts[:_MAX_SHARED_CONCEPTS]

    # ================================================================
    #  5. 路径收口：横向链（L1）+ 概念链（L2）
    # ================================================================
    @staticmethod
    def _law_ref(title: str, article: str = "") -> str:
        name = title or "未知法规"
        return f"《{name}》{article}".strip() if article else f"《{name}》"

    def _law_path(self, path_id: str, source: Dict, target: Dict) -> ReasoningPath:
        """A 法域条款 → C 法域对应法规 → 目标条款（L1，主题相似）。"""
        source_ref = self._law_ref(source.get("title", ""), source.get("article_number", ""))
        target_law_ref = self._law_ref(target.get("title", ""))
        steps = [
            HopStep(
                hop=1,
                kind="cross_jurisdiction",
                from_ref=source_ref,
                to_ref=target_law_ref,
                relation="对应法规",
                source="similar_law",
                confidence=float(target.get("composite_score", 0) or 0),
                evidence=[{
                    "law_id": target.get("law_id", ""),
                    "article": "",
                    "jurisdiction": target.get("jurisdiction", ""),
                    "similarity": target.get("similarity", 0),
                    "topic_overlap": target.get("topic_overlap", 0),
                    "shared_topics": target.get("shared_topics", []),
                }],
            )
        ]
        articles = target.get("articles") or []
        if articles:
            art = articles[0]
            steps.append(HopStep(
                hop=2,
                kind="target_article",
                from_ref=target_law_ref,
                to_ref=self._law_ref(target.get("title", ""), art["article_number"]),
                relation="相关条款",
                source="retrieval",
                confidence=float(art.get("similarity", 0) or 0),
                evidence=[{
                    "law_id": target.get("law_id", ""),
                    "article": art["article_number"],
                    "jurisdiction": target.get("jurisdiction", ""),
                }],
            ))
        return ReasoningPath(
            path_id=path_id,
            steps=steps,
            # 主链路优先于概念辅链；同分时由路径强度决定
            score=2.0 + float(target.get("composite_score", 0) or 0),
        )

    def _concept_path(self, path_id: str, source: Dict, target: Dict, c: Dict) -> ReasoningPath:
        """源条款 → 共享概念 → 目标条款（L2，同一 canonical_id）。

        scope=law 时概念出现在同法规的其他条款，源端只能是法规名，不能冒充源条款内容。
        """
        source_article = source.get("article_number", "") if c.get("scope") == "article" else ""
        source_ref = self._law_ref(source.get("title", ""), source_article)
        target_ref = self._law_ref(target.get("title", ""), c.get("target_article", ""))
        confidence = float(c.get("confidence", 0) or 0)
        steps = [
            HopStep(
                hop=1,
                kind="concept_link",
                from_ref=source_ref,
                to_ref=c["name"],
                relation=_RELATION_LABELS.get(c.get("source_relation", ""), c.get("source_relation", "")),
                source="concept_matrix",
                confidence=confidence,
                direction="forward" if c.get("source_side") == "head" else "reverse",
                evidence=[{
                    "law_id": source.get("law_id", ""),
                    "article": c.get("source_article", ""),
                    "jurisdiction": source.get("jurisdiction", ""),
                }],
            ),
            HopStep(
                hop=2,
                kind="concept_link",
                from_ref=c["name"],
                to_ref=target_ref,
                relation=_RELATION_LABELS.get(c.get("target_relation", ""), c.get("target_relation", "")),
                source="concept_matrix",
                confidence=confidence,
                direction="forward" if c.get("target_side") == "head" else "reverse",
                evidence=[{
                    "law_id": target.get("law_id", ""),
                    "article": c.get("target_article", ""),
                    "jurisdiction": target.get("jurisdiction", ""),
                }],
            ),
        ]
        return ReasoningPath(
            path_id=path_id,
            steps=steps,
            # 条款级交叉验证强于法规级；法规级只作辅助证据，排在主链路之后
            score=(1.0 + confidence) if c.get("scope") == "article" else (0.5 + confidence * 0.5),
        )

    # ================================================================
    #  6. 证据组装 + 1 次 LLM 对比合成
    # ================================================================
    @staticmethod
    def _build_evidence(source: Dict, targets: List[Dict], paths: List[ReasoningPath]) -> str:
        lines = [
            f"【源条款】《{source.get('title','')}》{source.get('article_number','')}"
            f"（{source.get('jurisdiction','未知法域')}）："
            f"{(source.get('content','') or '')[:600]}"
        ]
        for t in targets:
            lines.append(
                f"\n【对应法规】《{t.get('title','')}》（{t.get('jurisdiction','未知法域')}）"
                f"相似度 {t.get('similarity',0)}，主题重叠 {t.get('topic_overlap',0)}，"
                f"共享主题：{'、'.join(t.get('shared_topics') or []) or '无'}；"
                f"关系强度：{'主题相似（L1）' if not t.get('shared_concepts') else '同一概念可交叉验证（L2）'}"
            )
            if t.get("shared_concepts"):
                names = "、".join(
                    f"{c['name']}（{'条款级' if c.get('scope') == 'article' else '法规级'}）"
                    for c in t["shared_concepts"]
                )
                lines.append(f"  共享概念：{names}")
            for a in t.get("articles") or []:
                lines.append(
                    f"  - 第{a.get('article_number','')}条：{(a.get('content','') or '')[:300]}"
                )
        path_lines = [f"{p.path_id}（{p.level}）: {path_text(p.steps)}" for p in paths]
        if path_lines:
            lines.append(
                "\n【可推理路径】以下路径已由系统校验，作答只能引用这些编号，不得编造：\n"
                + "\n".join(path_lines)
            )
        return "\n".join(lines)

    @staticmethod
    def _build_prompt(question: str, evidence: str) -> str:
        return (
            "你是跨境法律对比助手。请**仅依据下面的源条款、对应法规、共享概念与可推理路径**作答。\n"
            "\n"
            "要求：\n"
            "1. 不得使用材料之外的知识；材料不足时明确说明「依据现有材料无法确定」。\n"
            "2. answer 用简体中文，条理清晰；**不要使用换行符**（用分号或序号分隔）。\n"
            "3. 可靠性口径必须遵守：L1 只代表法规主题相似，只能表述为「对应法规/类似规定」，"
            "严禁表述为「同一法规」「完全相同」「等同适用」；L2 表示实体消歧后为同一概念，"
            "可表述为「同一概念在两法域分别规定」，但仍不得推断两法域法律后果一致。\n"
            "4. 若答案需要跨法域串联，必须依据【可推理路径】作答，并在 used_path_indexes 中"
            "填路径编号（从 1 开始，最多 5 个）；只依据单条材料时留空。\n"
            "5. cited_articles 列出你引用的条款（格式 law_id::article_number）。\n"
            "\n"
            "问题：" + question + "\n\n材料：\n" + evidence
        )

    def answer(
        self,
        question: str,
        source_law_id: Optional[str] = None,
        source_article: Optional[str] = None,
        top_k_laws: int = _MAX_TARGET_LAWS,
        synthesize: bool = True,
        jurisdictions: Optional[List[str]] = None,
    ) -> Dict:
        """跨法域对齐问答主入口。返回路径与证据始终可用，LLM 仅做最终合成。

        jurisdictions: 问题中已识别的目标法域；非空时优先选择这些法域的对应法规。
        """
        question = (question or "").strip()
        if not question:
            return {"found": False, "message": "问题为空", "answer": "",
                    "reasoning_path": [], "cited_articles": [], "targets": []}

        # 法域入参归一：调用方常用旧称（"香港"）或英文名，而索引与质心里存的是规范名
        # （"中国香港"）。不归一时字面比较会全部失配，点名法域优先逻辑静默退化成
        # 纯相似度排序，源侧定位与目标法规选择都会失准。
        jurisdictions = normalize_jurisdictions(jurisdictions)

        candidates = self._locate_source(
            question, source_law_id, source_article, jurisdictions
        )
        if not candidates:
            return {"found": False, "message": "未检索到可作为源侧的条款",
                    "answer": "", "reasoning_path": [], "cited_articles": [], "targets": []}

        source, targets = None, []
        for cand in candidates:
            if not cand.get("law_id"):
                continue
            # 候选池需覆盖全部点名法域：多样性选择每法域只取一条，top_k 太小
            # 会被未点名的法域按评分占满（新增法域入库后澳门曾被挤到第 7 位出局）。
            data = self.similar.get_similar_laws(
                cand["law_id"],
                top_k=max(top_k_laws * 2, 6),
                preferred_jurisdictions=jurisdictions,
            )
            picked = self._pick_targets(
                data, cand.get("jurisdiction", ""), top_k_laws, jurisdictions
            )
            if picked:
                source, targets = cand, picked
                break

        if source is None:
            source = candidates[0]
            return {
                "found": False,
                "message": "已定位源条款，但未找到可用于跨法域对齐的对应法规",
                "answer": "",
                "source": source,
                "targets": [],
                "reasoning_path": [],
                "cited_articles": [],
                "stats": {"source_candidates": len(candidates), "target_count": 0,
                          "path_count": 0, "shared_concept_count": 0},
            }

        # 目标侧条款 + 原文回填（一次 msearch，源/目标共用）
        for target in targets:
            target["same_jurisdiction"] = bool(
                source.get("jurisdiction") and
                target.get("jurisdiction") == source.get("jurisdiction")
            )
            target["articles"] = self._target_articles(question, source, target)

        refs = [{
            "law_id": source.get("law_id", ""),
            "article_number": source.get("article_number", ""),
        }]
        for target in targets:
            refs.extend({
                "law_id": target.get("law_id", ""),
                "article_number": a.get("article_number", ""),
            } for a in target.get("articles") or [])
        snippets = self.repo.get_articles_by_refs(
            refs, max_pairs=_MAX_SNIPPETS, max_chars=600
        )
        source_key = f"{source.get('law_id','')}::{source.get('article_number','')}"
        if snippets.get(source_key):
            source["content"] = snippets[source_key]
        for target in targets:
            for a in target.get("articles") or []:
                key = f"{target.get('law_id','')}::{a.get('article_number','')}"
                if snippets.get(key):
                    a["content"] = snippets[key]
            target["shared_concepts"] = self._shared_concepts(source, target, target.get("articles") or [])

        # 路径收口：主链路优先；条款级概念链全部保留，法规级概念只取最强 2 条作辅助，避免撑爆候选路径。
        paths: List[ReasoningPath] = []
        for target in targets:
            paths.append(self._law_path("", source, target))
            concepts = target.get("shared_concepts") or []
            article_scoped = [c for c in concepts if c.get("scope") == "article"]
            law_scoped = [c for c in concepts if c.get("scope") != "article"][:1]
            for concept in article_scoped + law_scoped:
                if len(paths) >= _MAX_PATHS:
                    break
                paths.append(self._concept_path("", source, target, concept))
        paths = sorted(paths, key=lambda p: -p.score)[:_MAX_PATHS]
        for i, path in enumerate(paths, start=1):
            path.path_id = f"P{i}"

        result = {
            "found": True,
            "message": "",
            "question": question,
            "source": {
                "law_id": source.get("law_id", ""),
                "title": source.get("title", ""),
                "jurisdiction": source.get("jurisdiction", ""),
                "article_number": source.get("article_number", ""),
                "content": (source.get("content", "") or "")[:600],
                "score": source.get("score", 0),
            },
            "targets": [
                {
                    "law_id": t.get("law_id", ""),
                    "title": t.get("title", ""),
                    "jurisdiction": t.get("jurisdiction", ""),
                    "similarity": t.get("similarity", 0),
                    "topic_overlap": t.get("topic_overlap", 0),
                    "composite_score": t.get("composite_score", 0),
                    "shared_topics": t.get("shared_topics", []),
                    "same_jurisdiction": t.get("same_jurisdiction", False),
                    "articles": t.get("articles") or [],
                    "shared_concepts": t.get("shared_concepts") or [],
                }
                for t in targets
            ],
            "reasoning_path": [p.to_dict() for p in paths],
            "answer": "",
            "cited_articles": [],
            "stats": {
                "source_candidates": len(candidates),
                "target_count": len(targets),
                "path_count": len(paths),
                "shared_concept_count": sum(len(t.get("shared_concepts") or []) for t in targets),
            },
        }

        if not synthesize:
            result["message"] = "已按需跳过 LLM 合成，仅返回路径与条款证据"
            return result
        if not self.api_key:
            result["message"] = "未配置 LLM 密钥，仅返回跨法域路径与条款证据"
            return result

        evidence = self._build_evidence(source, targets, paths)
        try:
            llm = get_chat_model(model=self.model, temperature=0.1, max_tokens=2048)
            answer: _CrossJurisdictionAnswer = llm.with_structured_output(
                _CrossJurisdictionAnswer, method="function_calling"
            ).invoke(self._build_prompt(question, evidence))
        except Exception as e:
            logger.warning(f"[MultiHop] LLM 合成失败: {type(e).__name__}: {e}")
            result["message"] = f"生成答案失败: {type(e).__name__}（路径与证据仍可参考）"
            return result

        # 编号 → 真实路径，越界丢弃；引用条款同样只保留实际提供的证据
        selected: List[ReasoningPath] = []
        for idx in list(answer.used_path_indexes or []):
            if not isinstance(idx, int) or idx < 1 or idx > len(paths):
                continue
            path = paths[idx - 1]
            if any(p.path_id == path.path_id for p in selected):
                continue
            selected.append(path)
            if len(selected) >= 5:
                break

        allowed_refs = set(snippets.keys())
        result["answer"] = answer.answer.strip()
        result["cited_articles"] = [
            c for c in (answer.cited_articles or []) if c in allowed_refs
        ]
        result["reasoning_path"] = [p.to_dict() for p in selected] if selected else result["reasoning_path"]

        logger.info(
            "[MultiHop] 源=%s 对应法规=%d 路径=%d 共享概念=%d 被选=%d",
            source.get("law_id", ""), len(targets), len(paths),
            result["stats"]["shared_concept_count"], len(selected),
        )
        return result
