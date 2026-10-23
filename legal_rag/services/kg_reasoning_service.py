"""知识图谱受限多跳问答（GraphRAG）：实体/条款入口 → N 跳子图 → 三元组+条文证据 → 1 次 LLM 生成。

固定跳数控制成本与幻觉面；知识层无命中返回 found=False，由调用方决定是否回退普通 RAG。
"""

import logging
import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from core.config import settings
from core.llm import get_chat_model
from repositories.elasticsearch import ElasticsearchRepository
from services.entity_disambiguation_service import (
    EntityDisambiguationService,
    normalize_name,
    _ALIAS_GROUPS,
)
from services.kg_extraction_service import _RELATION_LABELS
from services.multihop_path import HopStep, ReasoningPath, path_text

logger = logging.getLogger(__name__)

_MAX_EVIDENCE_EDGES = 60   # 子图规模上限（控制 prompt 长度）
_MAX_EVIDENCE_ARTICLES = 20
_MAX_ARTICLE_SEEDS = 6      # 检索条款入口上限：条款是图谱枢纽，过多会挤占路径配额
_MAX_ARTICLE_EDGES = 36     # 条款自身三元组上限：必须给实体邻域展开留边预算
_MAX_ARTICLE_TRIPLES_PER_REF = 6  # 单个条款的边配额（防止高连接度条款独占全部预算）
_MAX_ARTICLE_ENTITY_SEEDS = 12  # 条款三元组中的义务/概念种子上限
_MAX_PATHS = 30            # 送入 prompt 的候选多跳路径上限
_MAX_PATH_LEN = 2           # 固定两跳
_MAX_FANOUT = 12            # 单节点展开的邻居上限（防路径组合爆炸）
_MAX_NODE_NAME_LEN = 40     # 超过视为「描述型串」（整句条文），不参与路径展开
_ARTICLE_PREFIX_RE = re.compile(r"^(?:article|art|sections?|sec)", re.I)  # 条款名前缀
_REVERSE_PENALTY = 0.6      # 逆向遍历降权：方向反转的语义可靠性低于原方向
# 关系类型权重：规范性强的关系（设定义务/禁止/处罚/援引）比释义/相关更能承载答案
_RELATION_WEIGHTS = {
    "imposes": 1.15,
    "prohibits": 1.15,
    "penalizes": 1.20,
    "refers_to": 1.15,
    "grants": 1.00,
    "has_exception": 0.95,
    "defines": 0.90,
    "related_to": 0.85,
}
_DEFAULT_RELATION_WEIGHT = 1.0
# 纵向深挖上限：每轮/总路径数、条款摘录数与单条字数
_MAX_DEEPEN_PER_ROUND = 4
_MAX_DEEPEN_TOTAL_PATHS = 8
_MAX_DEEPEN_SNIPPETS = 4
_MAX_DEEPEN_SNIPPET_CHARS = 200


class _KGAnswer(BaseModel):
    """图谱推理答案的结构化输出：路径只回候选编号，由服务端映射回真实路径，模型无法编造边。"""

    answer: str = Field(description="回答正文，简体中文，不使用换行符")
    used_path_indexes: List[int] = Field(
        default_factory=list,
        description="实际依据的候选路径编号（从 1 开始，取自【可推理路径】列表；未使用则留空）",
    )
    cited_articles: List[str] = Field(
        default_factory=list, description="引用条款，格式 law_id::article_number"
    )


class KGReasoningService:
    """基于知识图谱的受限多跳推理问答"""

    def __init__(self, repo: Optional[ElasticsearchRepository] = None):
        self.repo = repo or ElasticsearchRepository()
        self.api_key = settings.llm_api_key or ""
        self.model = settings.llm_extraction_model or "deepseek-flash"
        self._retriever = None

    # ================================================================
    #  1. 实体链接：把问题映射到实体词典中的规范实体
    # ================================================================
    def _link_entities(self, question: str, top_k: int = 5) -> List[Dict]:
        """实体链接两路互补：L2 别名命中（确定性）+ bge-m3 向量 KNN（跨语言/措辞兜底）。"""
        found: Dict[str, Dict] = {}

        # ---- ① L2 别名表命中 ----
        norm_q = normalize_name(question)
        if norm_q:
            for etype, canonical, aliases in _ALIAS_GROUPS:
                hit = False
                for raw in [canonical] + list(aliases):
                    norm_alias = normalize_name(raw)
                    if len(norm_alias) >= 2 and norm_alias in norm_q:
                        hit = True
                        break
                if hit:
                    canonical_id, display = EntityDisambiguationService.resolve(canonical, etype)
                    found.setdefault(canonical_id, {
                        "entity_id": canonical_id, "name": display, "type": etype, "source": "alias",
                    })

        # ---- ② 向量 KNN ----
        try:
            from core.model_loader import get_embedding_model
            vector = get_embedding_model().encode(
                [question], normalize_embeddings=True, show_progress_bar=False
            )[0]
            for hit in self.repo.search_kg_entities_knn(
                [float(x) for x in vector], k=top_k, exclude_types=["Article"]
            ):
                eid = hit.get("entity_id")
                if eid:
                    found.setdefault(eid, {
                        "entity_id": eid, "name": hit.get("name", ""),
                        "type": hit.get("type", ""), "source": "vector",
                        "score": hit.get("score"),
                    })
        except Exception as e:
            logger.warning(f"[KGReasoning] 向量实体链接失败，仅用别名命中: {type(e).__name__}: {e}")

        return list(found.values())[: max(top_k, 3)]

    # ================================================================
    #  1b. 条款入口：检索命中的条款 → 该条款抽取出的三元组
    # ================================================================
    def _get_retriever(self):
        """延迟构造检索器（与问答链路共用同一 embedding/reranker 实例）。"""
        if self._retriever is None:
            from services.retriever_service import RetrieverService
            self._retriever = RetrieverService()
        return self._retriever

    def _retrieve_article_refs(self, question: str,
                               jurisdictions: Optional[List[str]] = None,
                               limit: int = _MAX_ARTICLE_SEEDS) -> List[Dict]:
        """混合检索命中的条款（按相关性排序），作为图谱的「条款入口」。

        条款节点 canonical_id 由实体名生成、无法由条号反推，故只返回 (law_id, article)。
        """
        jurisdiction_list = [j for j in (jurisdictions or []) if j]
        try:
            results = self._get_retriever().hybrid_search(
                query=question,
                top_k=max(limit * 3, 15),
                jurisdiction=jurisdiction_list[0] if len(jurisdiction_list) == 1 else None,
                _decompose=True,
                # 不开精排：入口只取前 N 条条款作种子，覆盖率比次序更重要；
                # 精排会把目标条款挤出前 N 个入口，次序问题改由路径评分中的入口排名加权解决。
                _rerank=False,
            )
        except Exception as e:
            logger.warning(f"[KGReasoning] 检索条款入口失败，退回纯实体链接: {type(e).__name__}: {e}")
            return []

        allowed = set(jurisdiction_list)
        refs: List[Dict] = []
        seen = set()
        for item in results:
            jurisdiction = str(item.get("jurisdiction") or "").strip()
            if allowed and jurisdiction not in allowed:
                continue
            law_id = (item.get("law_id") or "").strip()
            article = (item.get("article_number") or "").strip()
            if not law_id or not article or (law_id, article) in seen:
                continue
            seen.add((law_id, article))
            refs.append({"law_id": law_id, "article": article, "jurisdiction": jurisdiction})
            if len(refs) >= limit:
                break
        return refs

    def _collect_article_triples(self, refs: List[Dict],
                                 jurisdictions: Optional[List[str]] = None) -> List[Dict]:
        """回取检索条款自身抽取出的三元组，按「检索排名 → 置信度」排序后限量。

        这些边是「问题主题 → 义务/概念」的第一跳证据，限额用于给实体邻域保留边预算。
        """
        if not refs:
            return []
        pairs = [(r["law_id"], r["article"]) for r in refs]
        triples = self.repo.get_kg_triples_by_source_articles(pairs, size=_MAX_ARTICLE_EDGES * 6)
        allowed = set(jurisdictions or [])
        rank = {pair: i for i, pair in enumerate(pairs)}
        grouped: Dict[tuple, List[Dict]] = {}
        for t in triples:
            if allowed and str(t.get("source_jurisdiction") or "").strip() not in allowed:
                continue
            grouped.setdefault((t.get("law_id", ""), t.get("source_article", "")), []).append(t)
        for items in grouped.values():
            items.sort(key=lambda t: -float(t.get("confidence", 0) or 0))

        # 轮转分配配额：按检索排名逐条轮取，避免高连接度条款占满预算、后续条款无边缘。
        ordered: List[Dict] = []
        cursors = {pair: 0 for pair in grouped}
        for items in grouped.values():
            items[:] = items[:_MAX_ARTICLE_TRIPLES_PER_REF]
        while len(ordered) < _MAX_ARTICLE_EDGES:
            advanced = False
            for pair in sorted(grouped, key=lambda p: rank.get(p, len(pairs))):
                items = grouped[pair]
                if cursors[pair] >= len(items):
                    continue
                ordered.append(items[cursors[pair]])
                cursors[pair] += 1
                advanced = True
                if len(ordered) >= _MAX_ARTICLE_EDGES:
                    break
            if not advanced:
                break
        return ordered

    @staticmethod
    def _article_number_from_name(name: str) -> str:
        """从条款节点名解析条号（用于把「路径走到某条款」计入证据）。

        抽取阶段的条款名不统一（35C / 第35C條 / Article 27），统一为规范串；中文数字条号无法可靠还原，返回空。
        """
        text = normalize_name(name or "")
        text = _ARTICLE_PREFIX_RE.sub("", text)
        text = text.strip("第条條款项項章節节")
        if text and any(ch.isdigit() for ch in text):
            return text
        return ""

    @staticmethod
    def _article_triple_seeds(triples: List[Dict]) -> tuple:
        """从条款三元组中提取图种子：(条款节点 ID 列表, 义务/概念等尾实体 ID 列表)。

        条款节点排在前以保证「从检索条款出发」的路径配额；超长描述型实体不参与展开。
        """
        article_ids: List[str] = []
        entity_ids: List[str] = []
        for t in triples:
            for side in ("head", "tail"):
                node = t.get(side) or {}
                cid = node.get("canonical_id") or node.get("id")
                if not cid:
                    continue
                name = node.get("name") or cid
                if node.get("type") == "Article":
                    if cid not in article_ids:
                        article_ids.append(cid)
                elif len(name) <= _MAX_NODE_NAME_LEN and cid not in entity_ids:
                    entity_ids.append(cid)
        return article_ids, entity_ids[:_MAX_ARTICLE_ENTITY_SEEDS]

    # ================================================================
    #  2. 子图检索（多跳）
    # ================================================================
    def _collect_subgraph(self, seed_ids: List[str], hops: int = 2,
                          jurisdictions: Optional[List[str]] = None,
                          priority_triples: Optional[List[Dict]] = None) -> List[Dict]:
        """以种子实体为中心扩 hops 跳，返回去重、限量的三元组。

        先按法域过滤再截断，否则全局边会占满预算；priority_triples（条款入口边）先占位。
        """
        seed_set = set(seed_ids)
        collected: Dict[tuple, Dict] = {}
        priority_keys: List[tuple] = []
        allowed = set(jurisdictions or [])

        def _key(t: Dict) -> tuple:
            return (
                t["head"].get("canonical_id") or t["head"].get("id"),
                t.get("relation"),
                t["tail"].get("canonical_id") or t["tail"].get("id"),
                t.get("source_article"),
            )

        def _merge(triples: List[Dict]) -> None:
            for t in triples:
                if allowed and str(t.get("source_jurisdiction") or "").strip() not in allowed:
                    continue
                key = _key(t)
                if key in collected:
                    continue
                collected[key] = t

        for t in priority_triples or []:
            if allowed and str(t.get("source_jurisdiction") or "").strip() not in allowed:
                continue
            key = _key(t)
            if key not in collected:
                collected[key] = t
            if key not in priority_keys:
                priority_keys.append(key)

        fetch_size = _MAX_EVIDENCE_EDGES * (8 if allowed else 4)
        _merge(self.repo.get_kg_triples_by_entities(seed_ids, size=fetch_size))

        if hops >= 2:
            neighbor_ids = set()
            for t in list(collected.values()):
                for side in ("head", "tail"):
                    cid = t[side].get("canonical_id") or t[side].get("id")
                    if cid and cid not in seed_set:
                        neighbor_ids.add(cid)
            if neighbor_ids:
                _merge(self.repo.get_kg_triples_by_entities(
                    list(neighbor_ids)[:200], size=fetch_size
                ))

        # 先放优先边，再按置信度补齐剩余预算
        priority_set = set(priority_keys)
        ordered = [collected[k] for k in priority_keys if k in collected]
        rest = [t for k, t in collected.items() if k not in priority_set]
        rest.sort(key=lambda t: -float(t.get("confidence", 0) or 0))
        return (ordered + rest)[:_MAX_EVIDENCE_EDGES]

    # ================================================================
    #  2b. 路径枚举（把三元组还原成可读的多跳链）
    # ================================================================
    @staticmethod
    def _node_key(ent: Dict) -> str:
        return ent.get("canonical_id") or ent.get("id") or ""

    @staticmethod
    def _node_name(ent: Dict) -> str:
        return ent.get("name") or KGReasoningService._node_key(ent)

    @staticmethod
    def _triple_key(t: Dict) -> tuple:
        head, tail = t.get("head") or {}, t.get("tail") or {}
        return (
            KGReasoningService._node_key(head),
            t.get("relation"),
            KGReasoningService._node_key(tail),
            t.get("source_article"),
        )

    def _build_adjacency(self, triples: List[Dict]) -> tuple:
        """三元组 → (names, types, adj)；正反双向入邻接表，按置信度降序。"""
        names: Dict[str, str] = {}
        types: Dict[str, str] = {}
        adj: Dict[str, List[tuple]] = {}
        for t in triples:
            head, tail = t.get("head") or {}, t.get("tail") or {}
            hk, tk = self._node_key(head), self._node_key(tail)
            if not hk or not tk or hk == tk:
                continue
            names.setdefault(hk, self._node_name(head))
            names.setdefault(tk, self._node_name(tail))
            types.setdefault(hk, head.get("type", ""))
            types.setdefault(tk, tail.get("type", ""))
            adj.setdefault(hk, []).append((tk, t, "forward"))
            adj.setdefault(tk, []).append((hk, t, "reverse"))
        # 置信度高的边优先展开，避免低置信边先占满扇出额度
        for node_id in adj:
            adj[node_id].sort(key=lambda item: -float(item[1].get("confidence", 0) or 0))
        return names, types, adj

    def _build_paths(self, seed_ids: List[str], triples: List[Dict],
                     entry_rank: Optional[Dict[str, int]] = None) -> List[ReasoningPath]:
        """从子图枚举 1~2 跳简单路径；双向遍历使仅作尾实体的种子也能展开。

        跳数/扇出/路径总数受限且节点不重复，防组合爆炸与回环；路径按编号送入 prompt 以便回查。
        """
        names, types, adj = self._build_adjacency(triples)

        def _usable(node_id: str) -> bool:
            """过滤描述型节点：非条款且名称超长的实体是整句条文，接入路径只会制造噪声。"""
            if types.get(node_id) == "Article":
                return True
            return len(names.get(node_id, "")) <= _MAX_NODE_NAME_LEN

        collected: List[tuple] = []  # (node_ids, steps, score)
        seen: set = set()
        # 每个种子公平分配配额，避免首个种子独占 _MAX_PATHS 把其他种子挤出候选。
        per_seed = max(3, _MAX_PATHS // max(len(seed_ids), 1))

        def _walk(node_id: str, node_ids: List[str], steps: List[HopStep],
                  score: float, sink: List[tuple]) -> None:
            if len(steps) >= _MAX_PATH_LEN or len(sink) >= per_seed:
                return
            for nb, triple, direction in adj.get(node_id, [])[:_MAX_FANOUT]:
                if nb in node_ids or not _usable(nb):
                    continue
                step = HopStep(
                    hop=len(steps) + 1,
                    kind="kg_edge",
                    from_ref=names.get(node_id, node_id),
                    to_ref=names.get(nb, nb),
                    relation=_RELATION_LABELS.get(triple.get("relation", ""), triple.get("relation", "")),
                    source="kg_edge",
                    confidence=float(triple.get("confidence", 0) or 0),
                    direction=direction,
                    evidence=[{
                        "law_id": triple.get("law_id", ""),
                        "article": triple.get("source_article", ""),
                        "jurisdiction": triple.get("source_jurisdiction", ""),
                        # 路径到达/离开的节点若是条款，一并记下其条号：
                        # 引用型边上「被引用条款」本身是这条路径的证据，不能只记来源条款。
                        "from_article": (
                            self._article_number_from_name(names.get(node_id, ""))
                            if types.get(node_id) == "Article" else ""
                        ),
                        "to_article": (
                            self._article_number_from_name(names.get(nb, ""))
                            if types.get(nb) == "Article" else ""
                        ),
                    }],
                )
                new_steps = steps + [step]
                new_ids = node_ids + [nb]
                directional = 1.0 if direction == "forward" else _REVERSE_PENALTY
                relation_weight = _RELATION_WEIGHTS.get(
                    str(triple.get("relation", "")), _DEFAULT_RELATION_WEIGHT
                )
                new_score = score * max(step.confidence, 0.05) * directional * relation_weight
                signature = tuple(new_ids)
                if len(new_steps) >= 2 and signature not in seen:
                    seen.add(signature)
                    # 检索条款入口按排名乘性加权：分组排序会把入口排名靠后的更好路径
                    # 整体压到排名靠前的路径之后，MRR 反而下降。
                    sink.append((new_ids, new_steps, new_score * start_boost))
                    if len(sink) >= per_seed:
                        return
                _walk(nb, new_ids, new_steps, new_score, sink)

        for seed in seed_ids:
            if len(collected) >= _MAX_PATHS:
                break
            if seed in adj:
                rank = (entry_rank or {}).get(seed)
                start_boost = 1.0 + 1.0 / (1.0 + rank) if rank is not None else 1.0
                sink: List[tuple] = []
                _walk(seed, [seed], [], 1.0, sink)
                collected.extend(sink)

        collected.sort(key=lambda item: -item[2])
        return [
            ReasoningPath(path_id=f"P{i}", steps=steps, score=score, node_ids=ids)
            for i, (ids, steps, score) in enumerate(collected[:_MAX_PATHS], start=1)
        ]

    def _deepen_paths(
        self,
        paths: List[ReasoningPath],
        triples: List[Dict],
        rounds: int = 2,
        beam: int = 4,
        fanout: int = 3,
        max_extra_edges: int = 8,
    ) -> tuple:
        """纵向深挖：从高分路径末端沿高置信边再扩 1~2 轮。

        束宽/扇出/每轮新增边数均受限，避免组合爆炸；新增证据可控且不新增 LLM 调用。
        """
        if not paths or rounds <= 0:
            return [], []

        base_count = len(paths)
        signatures = {tuple(p.node_ids) for p in paths if p.node_ids}
        names, types, _ = self._build_adjacency(triples)
        seen_edges = {self._triple_key(t) for t in triples}

        extra_paths: List[ReasoningPath] = []
        extra_triples: List[Dict] = []
        frontier = sorted(paths, key=lambda p: -p.score)[: max(1, beam)]

        for _round in range(rounds):
            terminals: List[str] = []
            for path in frontier:
                if path.node_ids and path.node_ids[-1] not in terminals:
                    terminals.append(path.node_ids[-1])
            if not terminals:
                break

            fetched = self.repo.get_kg_triples_by_entities(
                terminals, size=max(len(terminals) * fanout * 6, 24)
            )
            new_edges: List[Dict] = []
            for t in fetched:
                key = self._triple_key(t)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                new_edges.append(t)
                if len(new_edges) >= max_extra_edges:
                    break
            if not new_edges:
                break

            n_names, n_types, adj = self._build_adjacency(new_edges)
            for key, value in n_names.items():
                names.setdefault(key, value)
            for key, value in n_types.items():
                types.setdefault(key, value)

            round_paths: List[ReasoningPath] = []
            used_edges: Dict[tuple, Dict] = {}
            for path in frontier:
                terminal = path.node_ids[-1]
                for nb, triple, direction in adj.get(terminal, [])[:fanout]:
                    if nb in path.node_ids:
                        continue
                    if types.get(nb) != "Article" and len(names.get(nb, "")) > _MAX_NODE_NAME_LEN:
                        continue
                    signature = tuple(path.node_ids + [nb])
                    if signature in signatures:
                        continue
                    signatures.add(signature)
                    used_edges.setdefault(self._triple_key(triple), triple)
                    step = HopStep(
                        hop=len(path.steps) + 1,
                        kind="kg_edge",
                        from_ref=names.get(terminal, terminal),
                        to_ref=names.get(nb, nb),
                        relation=_RELATION_LABELS.get(
                            triple.get("relation", ""), triple.get("relation", "")
                        ),
                        source="kg_edge",
                        confidence=float(triple.get("confidence", 0) or 0),
                        direction=direction,
                        evidence=[{
                            "law_id": triple.get("law_id", ""),
                            "article": triple.get("source_article", ""),
                            "jurisdiction": triple.get("source_jurisdiction", ""),
                        }],
                    )
                    directional = 1.0 if direction == "forward" else _REVERSE_PENALTY
                    round_paths.append(ReasoningPath(
                        path_id="",
                        steps=path.steps + [step],
                        score=path.score * max(step.confidence, 0.05) * directional,
                        node_ids=path.node_ids + [nb],
                    ))
                    if len(round_paths) >= _MAX_DEEPEN_PER_ROUND:
                        break
                if len(round_paths) >= _MAX_DEEPEN_PER_ROUND:
                    break
            if not round_paths:
                break

            # 只把真正进入路径的边作为深挖证据，避免“取到但没用”的边污染条款摘录
            extra_triples.extend(used_edges.values())
            extra_paths.extend(round_paths)
            frontier = round_paths

        for i, path in enumerate(extra_paths, start=base_count + 1):
            path.path_id = f"P{i}"
        return extra_paths[: _MAX_DEEPEN_TOTAL_PATHS], extra_triples

    # ================================================================
    #  3. 证据组装 + LLM 生成
    # ================================================================
    def _build_evidence(self, triples: List[Dict], snippets: Dict[str, str],
                        law_titles: Dict[str, str],
                        paths: Optional[List[ReasoningPath]] = None,
                        deepen_snippets: Optional[Dict[str, str]] = None) -> str:
        rel_lines = []
        for t in triples:
            head = t["head"].get("name", "")
            tail = t["tail"].get("name", "")
            law = law_titles.get(t.get("law_id", ""), t.get("law_id", ""))
            rel_lines.append(
                f"- ({head}) --{t.get('relation','')}--> ({tail})  ←《{law}》{t.get('source_article','')}"
            )
        evidence = "【知识图谱关系】\n" + "\n".join(rel_lines) if rel_lines else "【知识图谱关系】（无）"

        # 路径段是核心证据：模型看到的是首尾相接的链，且只能按编号引用。
        if paths:
            path_lines = [f"{p.path_id}: {path_text(p.steps)}" for p in paths]
            evidence += (
                "\n\n【可推理路径】以下路径已由系统按图谱连接关系校验，"
                "作答时只能引用这些编号（P1、P2…），不得编造：\n"
                + "\n".join(path_lines)
            )

        if snippets:
            art_lines = []
            for ref, content in snippets.items():
                law_id, _, article = ref.partition("::")
                law = law_titles.get(law_id, law_id)
                art_lines.append(f"- 《{law}》{article}：{content}")
            evidence += "\n\n【条款原文摘录】\n" + "\n".join(art_lines)

        # 深挖到的引用条款单独成段，字数已截断，避免挤占基础证据预算
        if deepen_snippets:
            deep_lines = []
            for ref, content in deepen_snippets.items():
                law_id, _, article = ref.partition("::")
                law = law_titles.get(law_id, law_id)
                deep_lines.append(f"- 《{law}》{article}：{content}")
            evidence += "\n\n【纵向深挖·引用条款摘录】\n" + "\n".join(deep_lines)
        return evidence

    def _build_prompt(self, question: str, evidence: str) -> str:
        return (
            "你是跨境法律知识图谱问答助手。请**仅依据下面提供的知识图谱关系、可推理路径"
            "与条款原文摘录**回答问题。\n"
            "\n"
            "要求：\n"
            "1. 不得使用给材料之外的知识；材料不足时明确说明「依据现有知识图谱无法确定」。\n"
            "2. answer 用简体中文，条理清晰，可跨法域对比；**answer 内不要使用换行符**（用分号或序号分隔）。\n"
            "3. 若答案需要串联两条及以上关系才能得出，必须依据【可推理路径】作答，并在 "
            "used_path_indexes 中填该路径的编号（从 1 开始，最多 5 个）；只依据单条关系时留空。\n"
            "4. cited_articles 列出你引用的条款（格式 law_id::article_number）。\n"
            "5. 【纵向深挖·引用条款摘录】来自路径末端继续追踪到的条款，引用时须与"
            "对应路径编号一致，不得把深挖条款说成源路径本身的内容。\n"
            "\n"
            "问题：" + question + "\n\n"
            "材料：\n" + evidence
        )

    def answer(self, question: str, top_k_entities: int = 5, hops: int = 2,
               deepen: Optional[bool] = None, synthesize: bool = True,
               jurisdictions: Optional[List[str]] = None) -> Dict:
        """图谱多跳推理问答主入口（synthesize=False 时跳过 LLM，仅返回候选路径与证据）。

        deepen=None 时取 settings.multihop_deepen_enabled；jurisdictions 限制三元组法域。
        """
        deepen_enabled = settings.multihop_deepen_enabled if deepen is None else bool(deepen)
        question = (question or "").strip()
        if not question:
            return {"found": False, "message": "问题为空", "answer": "", "used_paths": [],
                    "reasoning_path": [], "cited_articles": []}

        seeds = self._link_entities(question, top_k_entities)
        # 条款入口：检索命中条款 → 其三元组 → 真实条款节点与义务实体种子；
        # 仅靠实体链接时，主题型问题会落到通用实体上，子图接不到目标条款。
        article_refs = self._retrieve_article_refs(question, jurisdictions)
        article_triples = self._collect_article_triples(article_refs, jurisdictions)
        article_node_ids, article_entity_ids = self._article_triple_seeds(article_triples)
        # 条款节点 → 检索排名：路径评分按入口排名加权，命中越靠前的条款产出的路径分越高
        ref_rank = {(r["law_id"], r["article"]): i for i, r in enumerate(article_refs)}
        entry_rank: Dict[str, int] = {}
        for t in article_triples:
            head = t.get("head") or {}
            if head.get("type") != "Article":
                continue
            cid = head.get("canonical_id") or head.get("id")
            rank = ref_rank.get((t.get("law_id", ""), t.get("source_article", "")))
            if cid and rank is not None:
                entry_rank.setdefault(cid, rank)
        if not seeds and not article_triples:
            return {
                "found": False,
                "message": "未能从知识图谱中链接到相关实体或条款（该问题可能不在已抽取的知识范围内）",
                "answer": "", "used_paths": [], "reasoning_path": [], "cited_articles": [],
                "seed_entities": [], "seed_articles": [],
            }

        # 种子顺序：条款节点优先（配额先保证从检索条款出发），义务/概念次之，实体链接垫底。
        seed_ids = list(dict.fromkeys(
            article_node_ids + article_entity_ids + [s["entity_id"] for s in seeds]
        ))
        triples = self._collect_subgraph(
            seed_ids, hops=hops, jurisdictions=jurisdictions,
            priority_triples=article_triples,
        )
        if jurisdictions:
            allowed = set(jurisdictions)
            triples = [
                t for t in triples
                if str(t.get("source_jurisdiction") or "").strip() in allowed
            ]
        if not triples:
            return {
                "found": False,
                "message": "已链接到实体/条款，但知识图谱中没有可用关系"
                           + ("（已按法域过滤）" if jurisdictions else ""),
                "answer": "", "used_paths": [], "reasoning_path": [], "cited_articles": [],
                "seed_entities": seeds, "seed_articles": article_refs,
            }

        # 把子图还原为可读路径；无 2 跳路径时 paths 为空，证据退回散装三元组
        paths = self._build_paths(seed_ids, triples, entry_rank=entry_rank)

        # 从高分路径末端沿高置信边继续深挖（默认关闭；不新增 LLM 调用）
        extra_paths: List[ReasoningPath] = []
        extra_triples: List[Dict] = []
        deepen_snippets: Dict[str, str] = {}
        if deepen_enabled and paths:
            extra_paths, extra_triples = self._deepen_paths(
                paths,
                triples,
                rounds=settings.multihop_deepen_max_rounds,
                beam=settings.multihop_deepen_beam,
                fanout=settings.multihop_deepen_fanout,
                max_extra_edges=settings.multihop_deepen_max_extra_edges,
            )
            paths = paths + extra_paths
            if extra_triples:
                extra_refs = [
                    {"law_id": t.get("law_id", ""), "article_number": t.get("source_article", "")}
                    for t in extra_triples
                ]
                deepen_snippets = self.repo.get_articles_by_refs(
                    extra_refs, max_pairs=_MAX_DEEPEN_SNIPPETS,
                    max_chars=_MAX_DEEPEN_SNIPPET_CHARS,
                )

        # 输出排序：路径分数降序（高置信/深跳优先）
        paths = sorted(paths, key=lambda p: -p.score)

        law_titles = {law.get("law_id"): law.get("title", "") for law in self.repo.get_all_laws()}
        refs = [
            {"law_id": t.get("law_id", ""), "article_number": t.get("source_article", "")}
            for t in triples
        ]
        snippets = self.repo.get_articles_by_refs(refs, max_pairs=_MAX_EVIDENCE_ARTICLES, max_chars=600)

        evidence = self._build_evidence(triples, snippets, law_titles, paths, deepen_snippets)

        subgraph = {
            "edge_count": len(triples),
            "articles": list(snippets.keys()) + list(deepen_snippets.keys()),
            "seed_articles": [f"{s['law_id']}::{s['article']}" for s in article_refs],
            "path_count": len(paths),
            "deepen_enabled": deepen_enabled,
            "deep_path_count": len(extra_paths),
            "deep_edge_count": len(extra_triples),
        }
        if not synthesize:
            return {
                "found": True,
                "message": "已按需跳过 LLM 合成，仅返回候选路径与证据",
                "answer": "",
                "reasoning_path": [p.to_dict() for p in paths],
                "used_paths": [],
                "cited_articles": [],
                "seed_entities": seeds,
                "seed_articles": article_refs,
                "subgraph": subgraph,
            }

        if not self.api_key:
            return {
                "found": False, "message": "未配置 LLM 密钥，无法生成答案",
                "answer": "", "used_paths": [], "reasoning_path": [], "cited_articles": [],
                "seed_entities": seeds, "seed_articles": article_refs,
            }

        try:
            llm = get_chat_model(model=self.model, temperature=0.1, max_tokens=2048)
            # 端点仅支持 function_calling 方式（json_schema 实测返回 400）
            result: _KGAnswer = llm.with_structured_output(
                _KGAnswer, method="function_calling"
            ).invoke(self._build_prompt(question, evidence))
        except Exception as e:
            logger.warning(f"[KGReasoning] LLM 生成失败: {type(e).__name__}: {e}")
            return {
                "found": False, "message": f"生成答案失败: {type(e).__name__}",
                "answer": "", "used_paths": [], "reasoning_path": [], "cited_articles": [],
                "seed_entities": seeds, "seed_articles": article_refs,
            }

        # 编号 → 真实路径（越界编号直接丢弃，模型无法构造不存在的路径）
        selected: List[ReasoningPath] = []
        seen_ids = set()
        for idx in list(result.used_path_indexes or []):
            if not isinstance(idx, int) or idx < 1 or idx > len(paths):
                continue
            path = paths[idx - 1]
            if path.path_id in seen_ids:
                continue
            seen_ids.add(path.path_id)
            selected.append(path)
            if len(selected) >= 5:
                break

        # 旧前端仍读 used_paths：把选中路径的每一步展平为单边结构
        used_paths = []
        for path in selected:
            for step in path.steps:
                ev = (step.evidence or [{}])[0] if step.evidence else {}
                used_paths.append({
                    "head": step.from_ref,
                    "relation": step.relation,
                    "tail": step.to_ref,
                    "article": ev.get("article", ""),
                })

        if paths:
            logger.info(
                "[KGReasoning] 候选路径=%d 深挖路径=%d 深挖边=%d 被选=%d 命中实体=%d 子图边=%d",
                len(paths), len(extra_paths), len(extra_triples),
                len(selected), len(seeds), len(triples),
            )

        return {
            "found": True,
            "message": "",
            "answer": result.answer.strip(),
            "reasoning_path": [p.to_dict() for p in selected],
            "used_paths": used_paths,
            "cited_articles": list(result.cited_articles),
            "seed_entities": seeds,
            "subgraph": subgraph,
        }
