"""LLM 实体关系抽取服务：从法条文本提取 {实体, 关系, 实体} 三元组，构建结构化知识图谱。"""

import logging
import re
import time
from typing import List, Dict, Optional

from pydantic import BaseModel, Field

from core.config import settings
from core.llm import get_chat_model
from repositories.elasticsearch import ElasticsearchRepository
from services.entity_disambiguation_service import EntityDisambiguationService

logger = logging.getLogger(__name__)

# 单批送入 LLM 的条款数，批量预抽取的成本估算据此计算批次数
DEFAULT_BATCH_SIZE = 5

# ---- 标准化本体定义 ----
LEGAL_ONTOLOGY = {
    "entities": {
        "Article":       { "label": "条款",     "description": "法律条文的具体条款" },
        "LegalSubject":  { "label": "法律主体",  "description": "拥有法律权利义务的实体（如数据控制者、监管机构、数据主体）" },
        "Obligation":    { "label": "法律义务",  "description": "法律要求履行的义务（如通知义务、获得同意）" },
        "Right":         { "label": "法律权利",  "description": "法律赋予的权利（如访问权、删除权）" },
        "Penalty":       { "label": "处罚措施",  "description": "违反法律后的处罚（如罚款、刑事责任）" },
        "Concept":       { "label": "法律概念",  "description": "法律中的定义性概念（如个人数据、跨境传输）" },
        "Exception":     { "label": "例外情形",  "description": "法律规定的例外情况（如国家安全例外）" },
    },
    "relations": [
        {"type": "defines",       "label": "定义",         "description": "条款定义了某个概念"},
        {"type": "imposes",       "label": "设定义务",     "description": "条款为主体设定某项义务"},
        {"type": "grants",        "label": "授予权利",     "description": "条款赋予主体某项权利"},
        {"type": "prohibits",     "label": "禁止",         "description": "条款禁止某项行为"},
        {"type": "penalizes",     "label": "处罚",         "description": "条款规定违规后的处罚"},
        {"type": "has_exception", "label": "例外",         "description": "条款包含例外情形"},
        {"type": "refers_to",     "label": "引用",         "description": "条款引用其他条款或法规"},
        {"type": "cross_refers",  "label": "跨法域引用",   "description": "条款跨法域引用其他法规"},
    ]
}


class _ExtractedEntity(BaseModel):
    """抽取出的实体。"""

    name: str = Field(description="实体名称，尽量使用原文中的精确表述")
    type: str = Field(description="实体类型，取自给定实体类型定义")


class _ExtractedTriple(BaseModel):
    """抽取出的三元组。"""

    head: _ExtractedEntity
    relation: str = Field(description="关系类型，取自给定关系类型定义")
    tail: _ExtractedEntity
    article_index: int = Field(default=0, description="该三元组来源法条在输入中的 index")
    confidence: float = Field(default=0.8, description="置信度，0.0-1.0")


class _TripleExtraction(BaseModel):
    """单批次三元组抽取的结构化输出。"""

    triples: List[_ExtractedTriple] = Field(
        default_factory=list, description="抽取到的三元组列表；无可抽取项时为空列表"
    )


class KGExtractionService:
    """基于 LLM 的实体关系抽取服务"""

    def __init__(self):
        self.model = settings.llm_extraction_model or "deepseek-flash"
        self.repo = ElasticsearchRepository()

    def extract_from_articles(self, articles: List[Dict], batch_size: int = 5) -> List[Dict]:
        """批量抽取三元组；单批失败不阻断整体，结果按 head-relation-tail 去重。

        返回项：{head:{id,type,name}, relation, tail:{id,type,name},
                 source_article, source_jurisdiction, confidence}。
        """
        if not articles:
            return []

        all_triples = []
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            try:
                triples = self._extract_batch(batch)
                all_triples.extend(triples)
                logger.info(f"[KGExtraction] 批次 {i//batch_size + 1}: 提取 {len(triples)} 个三元组")
            except Exception as e:
                logger.warning(f"[KGExtraction] 批次 {i//batch_size + 1} 提取失败: {e}")
                continue

        all_triples = self._deduplicate(all_triples)
        logger.info(f"[KGExtraction] 总计提取 {len(all_triples)} 个唯一三元组")
        return all_triples

    def _extract_batch(self, batch: List[Dict]) -> List[Dict]:
        """调用 LLM 抽取单个批次的三元组"""
        try:
            llm = get_chat_model(model=self.model, temperature=0.1, max_tokens=4096)
            # 端点仅支持 function_calling 方式（json_schema 实测返回 400）
            result: _TripleExtraction = llm.with_structured_output(
                _TripleExtraction, method="function_calling"
            ).invoke(self._build_prompt(batch))
        except Exception as e:
            logger.warning(f"[KGExtraction] 结构化抽取失败: {type(e).__name__}: {e}")
            return []

        return self._normalize_triples([t.model_dump() for t in result.triples], batch)

    def _build_prompt(self, batch: List[Dict]) -> str:
        """构建抽取提示词"""
        articles_text = ""
        for i, art in enumerate(batch):
            num = art.get("article_number", "?")
            content = art.get("content", "")
            articles_text += f"<article index=\"{i}\">\n"
            articles_text += f"条款号：{num}\n"
            articles_text += f"内容：{content}\n"
            articles_text += "</article>\n\n"

        # 将 ontology 转为 LLM 易读的格式
        entity_types_str = "\n".join([
            f"  - {k}: {v['label']} — {v['description']}"
            for k, v in LEGAL_ONTOLOGY["entities"].items()
        ])
        relation_types_str = "\n".join([
            f"  - {r['type']}: {r['label']} — {r['description']}"
            for r in LEGAL_ONTOLOGY["relations"]
        ])

        prompt = f"""你是一位法律知识工程专家。请从以下法条中提取法律实体和关系。

实体类型定义：
{entity_types_str}

关系类型定义：
{relation_types_str}

请分析每一条法条，提取其中包含的实体和关系。
每条法条可能产出 1-5 个三元组。

注意：
1. 实体名称尽量使用原文中的精确表述
2. 置信度 (confidence) 在 0.0-1.0 之间，明确匹配给 0.9+，推断匹配给 0.6-0.8
3. 如果法条内容不包含任何可提取的实体关系，triples 返回空列表
4. 同一实体在不同条款中重复出现时，使用相同的实体 name
5. article_index 为该三元组来源法条在输入中的 index

{articles_text}"""
        return prompt

    def _normalize_triples(self, triples: List[Dict], batch: List[Dict]) -> List[Dict]:
        """校验并规范化 LLM 抽取结果（本体类型/关系白名单、法条定位、置信度裁剪）"""
        results = []
        for t in triples:
            if not isinstance(t, dict):
                continue
            head = t.get("head", {})
            tail = t.get("tail", {})
            relation = t.get("relation", "")
            art_idx = t.get("article_index", 0)

            if not head or not tail or not relation:
                continue

            # 非法实体类型降级为 Concept，非法关系直接丢弃
            head_type = head.get("type", "")
            tail_type = tail.get("type", "")
            if head_type not in LEGAL_ONTOLOGY["entities"]:
                head_type = "Concept"
            if tail_type not in LEGAL_ONTOLOGY["entities"]:
                tail_type = "Concept"

            valid_relations = {r["type"] for r in LEGAL_ONTOLOGY["relations"]}
            if relation not in valid_relations:
                continue

            source_art = ""
            source_jur = ""
            if isinstance(art_idx, int) and 0 <= art_idx < len(batch):
                source_art = batch[art_idx].get("article_number", "")
                source_jur = batch[art_idx].get("jurisdiction", "")

            confidence = t.get("confidence", 0.8)
            if not isinstance(confidence, (int, float)):
                confidence = 0.8
            confidence = max(0.0, min(1.0, float(confidence)))

            results.append({
                "head": {
                    "id": self._make_entity_id(head.get("name", ""), head_type),
                    "type": head_type,
                    "name": head.get("name", ""),
                },
                "relation": relation,
                "tail": {
                    "id": self._make_entity_id(tail.get("name", ""), tail_type),
                    "type": tail_type,
                    "name": tail.get("name", ""),
                },
                "source_article": source_art,
                "source_jurisdiction": source_jur,
                "confidence": confidence,
            })

        return results

    @staticmethod
    def _make_entity_id(name: str, etype: str) -> str:
        """生成规范化实体 ID"""
        clean = re.sub(r'[^\w\u4e00-\u9fff]', '_', name.strip())
        clean = re.sub(r'_+', '_', clean).strip('_')
        return f"{etype}::{clean}" if clean else f"{etype}::unknown"

    @staticmethod
    def _deduplicate(triples: List[Dict]) -> List[Dict]:
        """去重：相同 head-relation-tail 只保留置信度最高的"""
        seen = {}
        for t in triples:
            key = (t["head"]["id"], t["relation"], t["tail"]["id"])
            if key not in seen or t["confidence"] > seen[key]["confidence"]:
                seen[key] = t
        return list(seen.values())

    def get_triples_cached(
        self,
        law_id: str,
        articles: List[Dict],
        force_refresh: bool = False,
    ) -> Dict:
        """获取某法规三元组：缓存指纹命中则直接读缓存，否则调 LLM 抽取并落库。

        force_refresh 忽略缓存；抽取为空时不写缓存，故下次请求等价于自动重试。
        """
        repo = self.repo
        fingerprint = repo.compute_law_fingerprint(law_id)

        # 指纹为空说明法规不存在，无法判断失效，直接走实时抽取
        if not force_refresh and fingerprint:
            meta = repo.get_kg_cache_meta(law_id)
            if meta and meta.get("fingerprint") == fingerprint and meta.get("triple_count"):
                logger.info(
                    f"[KGExtraction] 命中知识层缓存: law_id={law_id}, "
                    f"复用 {meta['triple_count']} 条三元组，跳过 LLM 抽取"
                )
                return {
                    "triples": repo.get_kg_triples(law_id),
                    "cache_hit": True,
                    "fingerprint": fingerprint,
                }

        triples = self.extract_from_articles(articles)

        # 抽取为空时不覆盖旧结果：LLM 输出有抖动（实测同一段法条两次分别返回 5 条与 0 条），
        # 指纹未变则判定为抖动并保留旧结果；指纹已变或旧结果不存在才正常落库（含清除）。
        if not triples:
            meta = repo.get_kg_cache_meta(law_id)
            if meta and meta.get("triple_count") and meta.get("fingerprint") == fingerprint:
                logger.warning(
                    f"[KGExtraction] 本次抽取为空但缓存指纹未变，保留已有 "
                    f"{meta['triple_count']} 条三元组: law_id={law_id}"
                )
                return {
                    "triples": repo.get_kg_triples(law_id),
                    "cache_hit": True,
                    "fingerprint": fingerprint,
                }

        jurisdiction = articles[0].get("jurisdiction", "") if articles else ""

        # 实体消歧（别名 + 向量 + 类型约束）：就地补写 canonical_id 并同步实体词典；
        # 失败不阻断落库，canonical_id 退化为原始 id（仅不做归并）。
        try:
            disambiguator = EntityDisambiguationService(self.repo)
            result = disambiguator.canonicalize_and_index(triples, law_id, jurisdiction)
            stats = result["stats"]
            logger.info(
                f"[KGExtraction] 实体消歧完成: law_id={law_id}, "
                f"{stats['raw_entities']} -> {stats['merged_entities']} 个实体"
                f"（L3 向量归并 {stats.get('vector_aligned', 0)} 个，"
                f"候选待人工确认 {stats.get('vector_candidates', 0)} 个）"
            )
        except Exception as e:
            logger.warning(f"[KGExtraction] 实体消歧失败，本次以原始实体落库: {e}")

        saved = repo.save_kg_triples(
            law_id,
            triples,
            jurisdiction=jurisdiction,
            fingerprint=fingerprint,
        )
        if saved < 0:
            logger.warning(f"[KGExtraction] 三元组落库失败，本次结果未缓存: law_id={law_id}")

        return {"triples": triples, "cache_hit": False, "fingerprint": fingerprint}

    # ================================================================
    #  批量预抽取（离线脚本用）：逐个复用 get_triples_cached，指纹命中即跳过 LLM，
    #  因此幂等、中断后可续跑。
    # ================================================================

    @staticmethod
    def _law_data_to_chunks(law_data: Dict) -> List[Dict]:
        """把 repo.get_law_by_id 的返回值转为抽取所需的 chunk 列表"""
        if not law_data:
            return []
        law_id = law_data.get("law_id", "")
        title = law_data.get("title", "")
        jurisdiction = law_data.get("jurisdiction", "")
        chunks = []
        for c in law_data.get("chunks", []):
            chunks.append({
                "law_id": law_id,
                "article_number": c.get("article_number", ""),
                "content": c.get("content", ""),
                "title": title,
                "jurisdiction": jurisdiction,
                "chunk_index": c.get("chunk_index", 0),
            })
        return chunks

    def _plan_law(self, law_id: str) -> Dict:
        """估算单个法规的抽取成本（不调用 LLM），区分 cached/planned/not_found。"""
        law_data = self.repo.get_law_by_id(law_id)
        if not law_data:
            return {"law_id": law_id, "status": "not_found", "articles": 0, "batches": 0}
        article_count = len(law_data.get("chunks", []))
        fingerprint = self.repo.compute_law_fingerprint(law_id)
        meta = self.repo.get_kg_cache_meta(law_id) if fingerprint else None
        cached = bool(
            meta and meta.get("fingerprint") == fingerprint and meta.get("triple_count")
        )
        batches = 0 if cached else (article_count + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE
        return {
            "law_id": law_id,
            "status": "cached" if cached else "planned",
            "articles": article_count,
            "batches": batches,
            "existing_triples": int(meta.get("triple_count", 0)) if meta else 0,
        }

    def _extract_one(self, law_id: str, force_refresh: bool) -> Dict:
        """抽取单个法规（供批量方法复用；异常不抛出，转为 failed 状态）"""
        try:
            law_data = self.repo.get_law_by_id(law_id)
            if not law_data:
                return {"law_id": law_id, "status": "not_found"}
            chunks = self._law_data_to_chunks(law_data)
            if not chunks:
                return {"law_id": law_id, "status": "empty"}
            result = self.get_triples_cached(law_id, chunks, force_refresh=force_refresh)
            return {
                "law_id": law_id,
                "status": "ok",
                "triples": len(result.get("triples") or []),
                "cache_hit": bool(result.get("cache_hit")),
            }
        except Exception as e:
            logger.warning(
                f"[KGExtraction] 法规 {law_id} 批量抽取失败: {type(e).__name__}: {e}"
            )
            return {"law_id": law_id, "status": "failed", "error": f"{type(e).__name__}: {e}"}

    def extract_for_laws(
        self,
        law_ids: List[str],
        workers: int = 1,
        delay: float = 0.0,
        force_refresh: bool = False,
        dry_run: bool = False,
    ) -> Dict:
        """批量预抽取多个法规的知识层（dry_run 仅估算成本，不调 LLM）。

        workers 默认 1：实体词典 merge_kg_entities 是读-并-写，并发写同一实体会丢别名/法域。
        """
        law_ids = [lid for lid in law_ids if lid]
        summary = {
            "total": len(law_ids), "ok": 0, "cache_hit": 0, "failed": 0,
            "not_found": 0, "empty": 0, "triples_total": 0, "llm_calls": 0,
            "failed_laws": [],
        }
        if not law_ids:
            return summary

        # ---- 成本估算模式：不调用 LLM ----
        if dry_run:
            plans = [self._plan_law(lid) for lid in law_ids]
            planned = [p for p in plans if p["status"] == "planned"]
            summary["plans"] = plans
            summary["planned_laws"] = len(planned)
            summary["cached_laws"] = sum(1 for p in plans if p["status"] == "cached")
            summary["planned_articles"] = sum(p["articles"] for p in planned)
            summary["planned_llm_calls"] = sum(p["batches"] for p in planned)
            return summary

        def _handle(item: Dict, index: int, total: int) -> None:
            status = item.get("status")
            if status == "ok":
                summary["ok"] += 1
                summary["triples_total"] += item.get("triples", 0)
                if item.get("cache_hit"):
                    summary["cache_hit"] += 1
                else:
                    summary["llm_calls"] += 1  # 未命中缓存才算一次 LLM 调用
                print(
                    f"[{index}/{total}] {item['law_id']} 完成: "
                    f"{item.get('triples', 0)} 三元组"
                    f"{'（缓存命中）' if item.get('cache_hit') else ''}",
                    flush=True,
                )
            elif status == "failed":
                summary["failed"] += 1
                summary["failed_laws"].append(
                    {"law_id": item["law_id"], "error": item.get("error", "")}
                )
                print(f"[{index}/{total}] {item['law_id']} 失败: {item.get('error', '')}", flush=True)
            else:
                summary[status] = summary.get(status, 0) + 1
                print(f"[{index}/{total}] {item['law_id']} 跳过: {status}", flush=True)

        if workers <= 1:
            for i, law_id in enumerate(law_ids, 1):
                _handle(self._extract_one(law_id, force_refresh), i, len(law_ids))
                if delay:
                    time.sleep(delay)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._extract_one, lid, force_refresh): lid for lid in law_ids
                }
                for fut in as_completed(futures):
                    done += 1
                    _handle(fut.result(), done, len(law_ids))

        return summary

    def rebuild_entities(self, law_ids: List[str]) -> Dict:
        """零 LLM 重建实体词典：读取已落库三元组重新归一并写回 canonical_id，修复历史写入遗漏。

        仅涉及 ES 读写与向量编码，不发起任何 LLM 请求。
        """
        disambiguator = EntityDisambiguationService(self.repo)
        summary = {"total": len(law_ids), "rebuilt": 0, "empty": 0, "failed": 0, "failed_laws": []}

        for i, law_id in enumerate([lid for lid in law_ids if lid], 1):
            try:
                triples = self.repo.get_kg_triples(law_id)
                if not triples:
                    summary["empty"] += 1
                    print(f"[{i}/{summary['total']}] {law_id} 跳过: 无三元组", flush=True)
                    continue
                jurisdiction = triples[0].get("source_jurisdiction", "")
                result = disambiguator.canonicalize_and_index(triples, law_id, jurisdiction)
                # 保留原指纹，避免重建后触发无谓的重抽
                meta = self.repo.get_kg_cache_meta(law_id)
                fingerprint = (meta or {}).get("fingerprint", "")
                self.repo.save_kg_triples(
                    law_id, result["triples"],
                    jurisdiction=jurisdiction, fingerprint=fingerprint,
                )
                summary["rebuilt"] += 1
                print(
                    f"[{i}/{summary['total']}] {law_id} 重建完成: "
                    f"{result['stats']['merged_entities']} 实体",
                    flush=True,
                )
            except Exception as e:
                summary["failed"] += 1
                summary["failed_laws"].append(
                    {"law_id": law_id, "error": f"{type(e).__name__}: {e}"}
                )
                print(f"[{i}/{summary['total']}] {law_id} 重建失败: {type(e).__name__}: {e}", flush=True)

        return summary

    def reconcile_entities(self, law_ids: Optional[List[str]] = None, prune: bool = False) -> Dict:
        """实体归一历史回溯：按当前别名表/本体重算已有三元组的 canonical_id，不调用 LLM。

        law_ids 为 None 时只处理三元组索引中出现过的法规；prune 为 True 时删除已无引用的孤立实体（破坏性）。
        """
        if law_ids is None:
            law_ids = self.repo.get_kg_law_ids()
        summary = self.rebuild_entities(law_ids)
        summary["scope"] = len(law_ids)

        if prune:
            referenced = set()
            for t in self.repo.get_all_kg_triples():
                for side in ("head", "tail"):
                    ent = t.get(side) or {}
                    cid = ent.get("canonical_id") or ent.get("id")
                    if cid:
                        referenced.add(cid)
            summary["pruned"] = self.repo.prune_kg_entities(referenced)
        return summary

    def extract_and_format_for_graph(self, articles: List[Dict]) -> Dict:
        """抽取并格式化为与 get_network_graph() 兼容的图谱结构"""
        return self.format_graph(self.extract_from_articles(articles))

    def format_graph(
        self,
        triples: List[Dict],
        articles: Optional[List[Dict]] = None,
    ) -> Dict:
        """将三元组格式化为前端图谱结构（无副作用）；节点按 canonical_id 合并同一实体。

        展示字段优先取 articles 元信息，缺省退回三元组自身的 source_* 字段。
        """
        # 法条号 -> 展示元信息（标题/法域/正文），供节点补齐字段
        article_meta: Dict[str, Dict] = {}
        for art in articles or []:
            num = art.get("article_number", "")
            if num and num not in article_meta:
                article_meta[num] = {
                    "title": art.get("title", ""),
                    "jurisdiction": art.get("jurisdiction", ""),
                    "content": art.get("content", ""),
                }

        def _node_id(ent: Dict) -> str:
            """规范实体 ID：优先 canonical_id（消歧后），否则退回原始 id"""
            return ent.get("canonical_id") or ent.get("id") or ""

        # ---- 构建节点：同一规范实体只建一个节点 ----
        entity_map: Dict[str, Dict] = {}
        for t in triples:
            for side in ("head", "tail"):
                ent = t[side]
                eid = _node_id(ent)
                if not eid or eid in entity_map:
                    continue
                src_art = t.get("source_article", "")
                meta = article_meta.get(src_art, {})
                title = meta.get("title", "") or ""
                # 保留完整法规名：节点标签由前端缩写，tooltip/详情展示全名
                law_name = title.split(" - ")[0] if " - " in title else title
                entity_map[eid] = {
                    "id": eid,
                    "label": ent["name"],
                    "title": ent["name"],
                    "node_kind": "entity",
                    "entity_type": ent["type"],
                    "entity_label": LEGAL_ONTOLOGY["entities"].get(ent["type"], {}).get("label", "条款"),
                    "entity_color": _ENTITY_COLORS.get(ent["type"], "#6B7280"),
                    "entity_shape": _ENTITY_SHAPES.get(ent["type"], "ellipse"),
                    # 示例来源（该实体在本图中首次出现的位置），供 tooltip / 详情展示
                    "jurisdiction": meta.get("jurisdiction") or t.get("source_jurisdiction", ""),
                    "law_name": law_name,
                    "article_number": src_art,
                    "content_preview": (meta.get("content", "") or "")[:500],
                    "degree": 0,
                    "importance_score": 0,
                    "is_core": False,
                }

        # ---- 构建边：同一 (from, relation, to) 合并，保留置信度最高的一条 ----
        edge_map: Dict[tuple, Dict] = {}
        for t in triples:
            src_id, tgt_id = _node_id(t["head"]), _node_id(t["tail"])
            if not src_id or not tgt_id or src_id == tgt_id:
                continue  # 跳过自环（LLM 偶发的 head==tail 噪声）
            key = (src_id, t["relation"], tgt_id)
            confidence = t.get("confidence", 0.8)
            if key in edge_map and edge_map[key]["similarity"] >= confidence:
                continue
            edge_map[key] = {
                "from": src_id,
                "to": tgt_id,
                "type": "semantic_similar",
                "edge_kind": "entity_relation",
                "relation_type": t["relation"],
                "relation_label": _RELATION_LABELS.get(t["relation"], "关联"),
                "label": _RELATION_LABELS.get(t["relation"], "关联"),
                "similarity": confidence,
                "weight": max(1, round(confidence * 5)),
                "color": "#A78BFA",
                "source_article": t.get("source_article", ""),
                "source_jurisdiction": t.get("source_jurisdiction", ""),
            }
        edge_list = list(edge_map.values())

        # 计算度数
        nodes = list(entity_map.values())
        deg = {n["id"]: 0 for n in nodes}
        for e in edge_list:
            deg[e["from"]] = deg.get(e["from"], 0) + 1
            deg[e["to"]] = deg.get(e["to"], 0) + 1
        for n in nodes:
            n["degree"] = deg.get(n["id"], 0)
            n["importance_score"] = round(min(1.0, n["degree"] / max(max(deg.values(), default=1), 1)), 3)

        return {
            "nodes": nodes,
            "edges": edge_list,
            "node_count": len(nodes),
            "edge_count": len(edge_list),
            "graph_source": "llm",
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edge_list),
                "entity_type_distribution": {
                    et: len([n for n in nodes if n["entity_type"] == et])
                    for et in {n["entity_type"] for n in nodes}
                },
                "schema_version": "extraction-v1",
            },
            "schema_version": "extraction-v1",
        }


# ---- 颜色/形状映射 ----
_ENTITY_COLORS = {
    "Article": "#6B7280", "LegalSubject": "#34A853", "Obligation": "#EA4335",
    "Right": "#0F9D58", "Penalty": "#FF6D00", "Concept": "#FABB05", "Exception": "#795548",
}
_ENTITY_SHAPES = {
    "Article": "ellipse", "LegalSubject": "roundRect", "Obligation": "diamond",
    "Right": "roundRect", "Penalty": "triangle", "Concept": "ellipse", "Exception": "ellipse",
}
_RELATION_LABELS = {
    "defines": "定义", "imposes": "设定义务", "grants": "授予权利",
    "prohibits": "禁止", "penalizes": "处罚", "has_exception": "例外",
    "refers_to": "引用", "cross_refers": "跨法域引用",
}
