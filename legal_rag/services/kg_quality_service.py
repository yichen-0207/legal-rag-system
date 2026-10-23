"""知识图谱质量与治理看板：抽取覆盖率、规模、结构健康度、置信度与类型分布。

纯统计计算不调 LLM（零 token、可复现）；与图算法服务（拓扑中心性）职责不重叠。
"""

import logging
from typing import Dict, List

from repositories.elasticsearch import ElasticsearchRepository
from services.kg_extraction_service import LEGAL_ONTOLOGY, _RELATION_LABELS

logger = logging.getLogger(__name__)

# 置信度分桶：含下界、上界开区间；末桶上界取 1.01 以纳入 confidence == 1.0
_CONFIDENCE_BUCKETS = [(0.0, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.01)]


class KGQualityService:
    """知识图谱质量与治理指标计算"""

    def __init__(self, repo: ElasticsearchRepository = None):
        self.repo = repo or ElasticsearchRepository()

    def get_quality_report(self) -> Dict:
        """汇总知识层质量指标（覆盖、规模、结构、置信度、类型分布）。"""
        triples = self.repo.get_all_kg_triples()
        stats = self.repo.get_kg_statistics() or {}
        total_laws = len(self.repo.get_all_laws())

        # ---- 单遍扫描：统计节点/去重边/置信度/类型 ----
        node_types: Dict[str, str] = {}       # canonical_id -> 实体类型
        edge_keys = set()                     # 去重后的 (head, relation, tail)
        relation_counts: Dict[str, int] = {}  # relation -> 去重边数
        bucket_counts = [0] * len(_CONFIDENCE_BUCKETS)
        conf_sum, conf_n = 0.0, 0

        for t in triples:
            h = t.get("head") or {}
            tl = t.get("tail") or {}
            h_id = h.get("canonical_id") or h.get("id")
            t_id = tl.get("canonical_id") or tl.get("id")
            if h_id:
                node_types.setdefault(h_id, h.get("type", "Concept"))
            if t_id:
                node_types.setdefault(t_id, tl.get("type", "Concept"))

            rel = t.get("relation", "")
            if h_id and t_id and h_id != t_id:  # 跳过自环
                key = (h_id, rel, t_id)
                if key not in edge_keys:
                    edge_keys.add(key)
                    relation_counts[rel] = relation_counts.get(rel, 0) + 1

            conf = t.get("confidence")
            if isinstance(conf, (int, float)):
                conf = float(conf)
                conf_sum += conf
                conf_n += 1
                for i, (lo, hi) in enumerate(_CONFIDENCE_BUCKETS):
                    if lo <= conf < hi:
                        bucket_counts[i] += 1
                        break

        node_count = len(node_types)
        edge_count = len(edge_keys)
        entity_count = int(stats.get("entity_count", 0))
        law_count = int(stats.get("law_count", 0))
        triple_count = int(stats.get("triple_count", 0))

        # ---- 结构指标 ----
        avg_degree = round(2 * edge_count / node_count, 3) if node_count else 0.0
        density = (
            round(2 * edge_count / (node_count * (node_count - 1)), 5)
            if node_count > 1 else 0.0
        )
        # 孤立实体：已写入实体词典、却未出现在任何三元组中的实体（无关联边）
        orphan_entities = max(0, entity_count - node_count)
        orphan_rate = round(orphan_entities / entity_count, 4) if entity_count else 0.0

        # ---- 类型分布（按规范实体数）----
        type_dist: Dict[str, int] = {}
        for etype in node_types.values():
            type_dist[etype] = type_dist.get(etype, 0) + 1
        entity_types = [
            {
                "type": etype,
                "label": LEGAL_ONTOLOGY["entities"].get(etype, {}).get("label", "其他"),
                "count": cnt,
            }
            for etype, cnt in sorted(type_dist.items(), key=lambda kv: -kv[1])
        ]
        relations = [
            {
                "relation": rel,
                "label": _RELATION_LABELS.get(rel, rel),
                "count": cnt,
            }
            for rel, cnt in sorted(relation_counts.items(), key=lambda kv: -kv[1])
        ]

        confidence = {
            "avg": round(conf_sum / conf_n, 4) if conf_n else 0.0,
            "buckets": [
                {"range": f"{lo:.1f}-{min(hi, 1.0):.1f}", "count": c}
                for (lo, hi), c in zip(_CONFIDENCE_BUCKETS, bucket_counts)
            ],
        }

        return {
            "scale": {
                "triple_count": triple_count,
                "entity_count": entity_count,
                "law_count": law_count,
                "total_laws": total_laws,
                "coverage_rate": round(law_count / total_laws, 4) if total_laws else 0.0,
            },
            "structure": {
                "node_count": node_count,
                "edge_count": edge_count,
                "avg_degree": avg_degree,
                "density": density,
                "orphan_entities": orphan_entities,
                "orphan_rate": orphan_rate,
            },
            "confidence": confidence,
            "entity_types": entity_types,
            "relations": relations,
        }
