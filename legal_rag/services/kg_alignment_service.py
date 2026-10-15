"""
跨法域概念对齐矩阵服务 v1.0
---------------------------
回答一个跨境合规最关心的问题：「同一个法律概念，在各法域分别由多少条款规定？」

数据来源：法规知识层 legal_kg_triples（已完成实体消歧，实体以 canonical_id 归并），
因此同一概念跨法规、跨法域的出现会被正确聚到一行。

为什么在 Python 侧聚合而非 ES 聚合：
    知识层三元组规模在数千条量级（当前约 6 千），一次取回在内存聚合比多次聚合查询
    更简单、更灵活（可同时统计条文去重数、法域覆盖数并做排序过滤）。
    若未来数据量级上升，再把聚合下沉到 ES。

口径说明：
    - 单元格值 = 该概念在该法域下**去重后的条文数**（distinct source_article），
      比"三元组条数"更能反映"有多少条款在管这件事"。
    - 只纳入「概念型」实体：排除 Article（条款是法规内局部实体），并过滤超长名称
      （英文条款里的整句描述串），否则矩阵会被噪声撑爆、不可读。
"""

import logging
from typing import Dict, List, Optional

from repositories.elasticsearch import ElasticsearchRepository

logger = logging.getLogger(__name__)

# 概念名长度上限：超过视为「描述型串」（整句条文），不参与矩阵
_MAX_NAME_LEN = 40


class KGAlignmentService:
    """跨法域概念对齐：矩阵总览 + 单概念下钻明细"""

    def __init__(self, repo: Optional[ElasticsearchRepository] = None):
        self.repo = repo or ElasticsearchRepository()

    @staticmethod
    def _is_concept_like(name: str, etype: str) -> bool:
        """是否为可用于矩阵的「概念型」实体（排除条款类与超长描述串）"""
        if not name or etype == "Article":
            return False
        return len(name) <= _MAX_NAME_LEN

    def get_concept_matrix(
        self,
        top_n: int = 30,
        min_jurisdictions: int = 2,
        entity_types: Optional[List[str]] = None,
    ) -> Dict:
        """
        构建「概念 × 法域」条文数矩阵。

        Args:
            top_n: 返回的概念行数上限
            min_jurisdictions: 只保留至少出现在这么多个法域的概念（跨域概念才有对比价值）
            entity_types: 只保留指定实体类型（默认全部概念型）

        Returns:
            {
              "jurisdictions": [法域名...],
              "concepts": [{entity_id, name, type, jurisdiction_count, total, counts:{jur:条文数}}],
              "max_count": 最大单元格值（供前端着色），
              "concept_count": 概念行数
            }
        """
        triples = self.repo.get_all_kg_triples()

        # concept_id -> {name, type, jur: {jurisdiction: set(article)}}
        concepts: Dict[str, Dict] = {}
        for t in triples:
            jurisdiction = t.get("source_jurisdiction") or "未知"
            article = t.get("source_article", "")
            for side in ("head", "tail"):
                ent = t.get(side) or {}
                cid = ent.get("canonical_id") or ent.get("id")
                name = ent.get("name", "")
                etype = ent.get("type", "")
                if not cid or not self._is_concept_like(name, etype):
                    continue
                if entity_types and etype not in entity_types:
                    continue
                doc = concepts.get(cid)
                if doc is None:
                    doc = concepts[cid] = {
                        "entity_id": cid, "name": name, "type": etype, "jur": {}
                    }
                doc["jur"].setdefault(jurisdiction, set()).add(article)

        jurisdictions = sorted({j for d in concepts.values() for j in d["jur"]})

        rows = []
        for d in concepts.values():
            counts = {j: len(arts) for j, arts in d["jur"].items()}
            jur_count = sum(1 for v in counts.values() if v > 0)
            if jur_count < min_jurisdictions:
                continue
            rows.append({
                "entity_id": d["entity_id"],
                "name": d["name"],
                "type": d["type"],
                "jurisdiction_count": jur_count,
                "total": sum(counts.values()),
                "counts": counts,
            })

        # 覆盖法域越多越有对比价值；同覆盖数按条文总数降序；再按名称稳定排序
        rows.sort(key=lambda r: (-r["jurisdiction_count"], -r["total"], r["name"]))
        rows = rows[:top_n]

        max_count = max((max(r["counts"].values()) for r in rows), default=0)
        return {
            "jurisdictions": jurisdictions,
            "concepts": rows,
            "max_count": max_count,
            "concept_count": len(rows),
        }

    def get_concept_detail(self, entity_id: str) -> Optional[Dict]:
        """
        单概念下钻：列出各法域中涉及该概念的法规 / 条款号 / 关系类型。

        返回 None 表示该概念在三元组中不存在（前端据此提示）。
        """
        entity_id = (entity_id or "").strip()
        if not entity_id:
            return None

        triples = self.repo.get_all_kg_triples()
        law_titles = {law.get("law_id"): law.get("title", "") for law in self.repo.get_all_laws()}

        name, etype = "", ""
        # jurisdiction -> {(law_id, article): set(relations)}
        grouped: Dict[str, Dict[tuple, set]] = {}
        for t in triples:
            for side in ("head", "tail"):
                ent = t.get(side) or {}
                cid = ent.get("canonical_id") or ent.get("id")
                if cid != entity_id:
                    continue
                if not name:
                    name = ent.get("name", "")
                    etype = ent.get("type", "")
                jurisdiction = t.get("source_jurisdiction") or "未知"
                key = (t.get("law_id", ""), t.get("source_article", ""))
                grouped.setdefault(jurisdiction, {}).setdefault(key, set()).add(t.get("relation", ""))

        if not grouped:
            return None

        jurisdictions = []
        for jur in sorted(grouped, key=lambda j: (-len(grouped[j]), j)):
            articles = [
                {
                    "law_id": law_id,
                    "law_name": law_titles.get(law_id, ""),
                    "article_number": article,
                    "relations": sorted(rel for rel in grouped[jur][(law_id, article)] if rel),
                }
                for (law_id, article) in sorted(grouped[jur])
            ]
            jurisdictions.append({
                "jurisdiction": jur,
                "article_count": len(articles),
                "articles": articles,
            })

        return {
            "entity_id": entity_id,
            "name": name,
            "type": etype,
            "jurisdictions": jurisdictions,
        }
