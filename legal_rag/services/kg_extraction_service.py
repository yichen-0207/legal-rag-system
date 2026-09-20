"""
LLM 实体关系抽取服务 v1.0
---------------------------
从法条文本中自动提取 {实体类型, 关系类型, 实体类型} 三元组，
用于构建结构化知识图谱，替代纯关键词匹配的建图方式。

使用方式：
    service = KGExtractionService()
    triples = service.extract_from_articles(articles)
"""

import json
import logging
import re
from typing import List, Dict, Optional
from openai import OpenAI
from core.config import settings
from repositories.elasticsearch import ElasticsearchRepository
from services.entity_disambiguation_service import EntityDisambiguationService

logger = logging.getLogger(__name__)

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


class KGExtractionService:
    """基于 LLM 的实体关系抽取服务"""

    def __init__(self):
        self.api_key = settings.llm_api_key or ""
        self.base_url = settings.llm_api_base_url or "https://api.deepseek.com/v1"
        self.model = settings.llm_extraction_model or "deepseek-flash"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.repo = ElasticsearchRepository()

    def extract_from_articles(self, articles: List[Dict], batch_size: int = 5) -> List[Dict]:
        """
        对一批法条批量抽取实体关系三元组

        Args:
            articles: 法条列表，每项需包含：
                - article_number: 条款号
                - content: 条款全文
                - law_id: (可选) 法规ID
                - jurisdiction: (可选) 法域
            batch_size: 每批送入 LLM 的条款数（默认5）

        Returns:
            三元组列表，格式：
            [{
                "head": {"id": str, "type": str, "name": str},
                "relation": str,
                "tail": {"id": str, "type": str, "name": str},
                "source_article": str,
                "source_jurisdiction": str,
                "confidence": float
            }, ...]
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
                # 单个批次失败不影响整体
                continue

        # 去重
        all_triples = self._deduplicate(all_triples)
        logger.info(f"[KGExtraction] 总计提取 {len(all_triples)} 个唯一三元组")
        return all_triples

    def _extract_batch(self, batch: List[Dict]) -> List[Dict]:
        """调用 LLM 抽取单个批次的三元组"""
        prompt = self._build_prompt(batch)

        create_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        }
        extra_body = settings.llm_extra_body()
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        response = self.client.chat.completions.create(**create_kwargs)
        text = response.choices[0].message.content

        return self._parse_response(text, batch)

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

请分析每一条法条，提取其中包含的实体和关系，输出 JSON 格式的三元组列表。
每条法条可能产出 1-5 个三元组。

注意：
1. 实体名称尽量使用原文中的精确表述
2. 置信度 (confidence) 在 0.0-1.0 之间，明确匹配给 0.9+，推断匹配给 0.6-0.8
3. 如果法条内容不包含任何可提取的实体关系，输出空列表 []
4. 同一实体在不同条款中重复出现时，使用相同的实体 name

{articles_text}

请严格按照以下 JSON 格式输出（不要包含 markdown 代码块标记，直接输出 JSON）：
{{
    "triples": [
        {{
            "head": {{"name": "实体名称", "type": "实体类型"}},
            "relation": "关系类型",
            "tail": {{"name": "实体名称", "type": "实体类型"}},
            "article_index": 0,
            "confidence": 0.95
        }}
    ]
}}"""
        return prompt

    def _parse_response(self, text: str, batch: List[Dict]) -> List[Dict]:
        """解析 LLM 返回的 JSON 结果"""
        # 清理可能的 markdown 包裹
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 尝试提取 JSON 部分
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning(f"[KGExtraction] JSON 解析失败，原始响应: {text[:200]}")
                    return []
            else:
                return []

        triples = data.get("triples", [])
        if not isinstance(triples, list):
            return []

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

            # 校验实体类型
            head_type = head.get("type", "")
            tail_type = tail.get("type", "")
            if head_type not in LEGAL_ONTOLOGY["entities"]:
                head_type = "Concept"
            if tail_type not in LEGAL_ONTOLOGY["entities"]:
                tail_type = "Concept"

            # 校验关系类型
            valid_relations = {r["type"] for r in LEGAL_ONTOLOGY["relations"]}
            if relation not in valid_relations:
                continue

            # 找到对应法条
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
        """
        获取某法规的三元组：优先读知识层缓存，未命中才调用 LLM 抽取。

        命中条件为「缓存指纹 == 当前法规指纹」，因此：
          - 新增法规：只对新 law_id 抽取，已有法规零 LLM 调用；
          - 法规修订：只有该法规指纹变化并重抽，其余法规不受影响；
          - 重复访问：直接读 ES，不再触发一次完整抽取。

        说明：抽取结果为空时不写缓存元信息（无文档即无指纹），
        下次请求会重抽。这一行为是有意为之 —— LLM 批次失败或响应异常
        时同样返回空列表，重抽等价于自动重试。

        Args:
            law_id: 法规 ID
            articles: 法条列表（仅在缓存未命中时使用）
            force_refresh: 为 True 时忽略缓存，强制重新抽取（本体变更后使用）

        Returns:
            {"triples": List[Dict], "cache_hit": bool, "fingerprint": str}
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

        # 抽取结果为空时不能直接覆盖已有结果：LLM 输出存在抖动
        # （实测同一段法条两次抽取分别返回 5 条与 0 条），
        # 若此时清空，一次抖动就会把已建好的图谱整体抹掉。
        # 因此：旧结果存在且指纹未变 -> 判定为抖动，保留旧结果；
        #       指纹已变（法规被修订为无可提取关系）或旧结果不存在 -> 正常走落库（含清除）。
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

        # 实体消歧（L1 规范化 + L2 别名表 + L4 类型约束）：
        # 就地补写 head/tail 的 canonical_id，并把规范实体同步到实体词典。
        # 消歧失败不阻断抽取结果落库 —— 此时 canonical_id 退化为原始 id，
        # 检索与图谱展示仍可用，只是未做归并。
        try:
            disambiguator = EntityDisambiguationService(self.repo)
            result = disambiguator.canonicalize_and_index(triples, law_id, jurisdiction)
            logger.info(
                f"[KGExtraction] 实体消歧完成: law_id={law_id}, "
                f"{result['stats']['raw_entities']} -> {result['stats']['merged_entities']} 个实体"
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

    def extract_and_format_for_graph(self, articles: List[Dict]) -> Dict:
        """
        抽取实体关系并格式化为图谱数据结构
        兼容 get_network_graph() 的输出格式
        """
        return self.format_graph(self.extract_from_articles(articles))

    def format_graph(self, triples: List[Dict]) -> Dict:
        """将三元组列表格式化为前端图谱数据结构（无副作用，不发 LLM 请求）"""
        # 构建节点
        entity_map = {}
        for t in triples:
            for side in ["head", "tail"]:
                ent = t[side]
                eid = ent["id"]
                if eid not in entity_map:
                    entity_map[eid] = {
                        "id": eid,
                        "label": ent["name"],
                        "title": ent["name"],
                        "entity_type": ent["type"],
                        "entity_label": LEGAL_ONTOLOGY["entities"].get(ent["type"], {}).get("label", "条款"),
                        "entity_color": _ENTITY_COLORS.get(ent["type"], "#6B7280"),
                        "entity_shape": _ENTITY_SHAPES.get(ent["type"], "ellipse"),
                        "degree": 0,
                        "importance_score": 0,
                        "is_core": False,
                    }

        # 构建边
        edge_list = []
        for t in triples:
            edge_list.append({
                "from": t["head"]["id"],
                "to": t["tail"]["id"],
                "type": "semantic_similar",
                "relation_type": t["relation"],
                "relation_label": _RELATION_LABELS.get(t["relation"], "关联"),
                "label": _RELATION_LABELS.get(t["relation"], "关联"),
                "similarity": t["confidence"],
                "weight": max(1, round(t["confidence"] * 5)),
                "color": "#A78BFA",
                "source_article": t.get("source_article", ""),
                "source_jurisdiction": t.get("source_jurisdiction", ""),
            })

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
