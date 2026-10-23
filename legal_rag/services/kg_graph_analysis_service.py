"""实体关系图的拓扑分析：Louvain 社区、PageRank/度中心性、介数中心性（桥接概念）。

纯拓扑计算不调 LLM；固定 seed 保证簇编号与结果可复现；大图介数改用 k 采样近似。
"""

import logging
from typing import Dict, List

import networkx as nx
from networkx.algorithms.community import louvain_communities

logger = logging.getLogger(__name__)

# 超过该节点数，介数中心性改用 k 采样近似（精确算法在大图上耗时明显）
_BETWEENNESS_EXACT_MAX = 1500
_BETWEENNESS_SAMPLE_K = 200
_DEFAULT_TOP_N = 10


class KGGraphAnalysisService:
    """实体关系图的拓扑分析：社区 / 中心性 / 榜单"""

    def analyze(self, nodes: List[Dict], edges: List[Dict], top_n: int = _DEFAULT_TOP_N) -> Dict:
        """执行社区检测与中心性计算，返回 communities / metrics / top_hubs / bridges。"""
        empty = {"node_count": 0, "edge_count": 0, "communities": [],
                 "metrics": {}, "top_hubs": [], "bridges": []}
        if not nodes:
            return empty

        graph = nx.Graph()
        id2node: Dict[str, Dict] = {}
        for n in nodes:
            nid = n.get("id")
            if not nid:
                continue
            graph.add_node(nid)
            id2node[nid] = n
        for e in edges:
            u, v = e.get("from"), e.get("to")
            # 自环与悬空端点会污染中心性，直接跳过
            if u in id2node and v in id2node and u != v:
                graph.add_edge(u, v)

        if graph.number_of_nodes() == 0:
            return empty

        # ---- 社区检测（确定性 seed）----
        try:
            raw_communities = louvain_communities(graph, seed=42)
        except Exception as e:
            logger.warning(f"[KGAnalysis] Louvain 失败，退化为单社区: {type(e).__name__}")
            raw_communities = [set(graph.nodes())]
        raw_communities = sorted(raw_communities, key=len, reverse=True)

        # ---- PageRank ----
        try:
            pagerank = nx.pagerank(graph, alpha=0.85)
        except Exception as e:
            logger.warning(f"[KGAnalysis] PageRank 失败: {type(e).__name__}")
            pagerank = {nid: 0.0 for nid in graph.nodes()}

        # ---- 介数中心性（大图采样近似）----
        n_nodes = graph.number_of_nodes()
        try:
            if n_nodes <= _BETWEENNESS_EXACT_MAX:
                betweenness = nx.betweenness_centrality(graph)
            else:
                betweenness = nx.betweenness_centrality(
                    graph, k=min(_BETWEENNESS_SAMPLE_K, n_nodes), seed=42
                )
        except Exception as e:
            logger.warning(f"[KGAnalysis] 介数中心性失败: {type(e).__name__}")
            betweenness = {nid: 0.0 for nid in graph.nodes()}

        degrees = dict(graph.degree())

        # ---- 社区编号与命名（簇内最高 PageRank 的实体名为代表）----
        community_of: Dict[str, int] = {}
        communities: List[Dict] = []
        for idx, members in enumerate(raw_communities):
            member_list = list(members)
            representative = max(member_list, key=lambda m: pagerank.get(m, 0.0))
            rep_node = id2node.get(representative, {})
            label = rep_node.get("label") or rep_node.get("name") or representative
            for member in member_list:
                community_of[member] = idx
            communities.append({
                "community_id": idx,
                "size": len(member_list),
                "label": label,
                "representative": representative,
                "members": member_list,
            })

        metrics = {
            nid: {
                "pagerank": round(float(pagerank.get(nid, 0.0)), 6),
                "betweenness": round(float(betweenness.get(nid, 0.0)), 6),
                "degree": int(degrees.get(nid, 0)),
                "community_id": community_of.get(nid, -1),
            }
            for nid in graph.nodes()
        }

        def _brief(nid: str) -> Dict:
            node = id2node.get(nid, {})
            return {
                "id": nid,
                "label": node.get("label") or node.get("name") or nid,
                "entity_type": node.get("entity_type", ""),
                "jurisdiction": node.get("jurisdiction", ""),
            }

        top_hubs = sorted(graph.nodes(), key=lambda m: -pagerank.get(m, 0.0))[:top_n]
        bridges = sorted(graph.nodes(), key=lambda m: -betweenness.get(m, 0.0))[:top_n]

        return {
            "node_count": n_nodes,
            "edge_count": graph.number_of_edges(),
            "communities": communities,
            "metrics": metrics,
            "top_hubs": [
                {**_brief(m), "pagerank": metrics[m]["pagerank"], "degree": metrics[m]["degree"]}
                for m in top_hubs
            ],
            "bridges": [
                {**_brief(m), "betweenness": metrics[m]["betweenness"]}
                for m in bridges
            ],
        }


def enrich_graph_with_analytics(graph: Dict, service: "KGGraphAnalysisService" = None) -> Dict:
    """把社区/中心性指标写入节点，并挂上 communities/top_hubs/bridges，供前端直接渲染。"""
    svc = service or KGGraphAnalysisService()
    analysis = svc.analyze(graph.get("nodes", []), graph.get("edges", []))
    metrics = analysis.get("metrics", {})
    for node in graph.get("nodes", []):
        m = metrics.get(node.get("id"))
        if m:
            node["pagerank"] = m["pagerank"]
            node["betweenness"] = m["betweenness"]
            node["community_id"] = m["community_id"]
    graph["communities"] = analysis.get("communities", [])
    graph["top_hubs"] = analysis.get("top_hubs", [])
    graph["bridges"] = analysis.get("bridges", [])
    return graph
