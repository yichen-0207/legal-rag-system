import logging

from fastapi import APIRouter, Query, Body
from typing import Optional, Dict
from models.schemas import APIResponse
from repositories.elasticsearch import ElasticsearchRepository
from services.structured_analysis_service import StructuredAnalysisService
from services.kg_extraction_service import KGExtractionService
from services.kg_alignment_service import KGAlignmentService
from services.kg_graph_analysis_service import (
    KGGraphAnalysisService,
    enrich_graph_with_analytics,
)
from services.kg_reasoning_service import KGReasoningService

logger = logging.getLogger(__name__)

# 以下路由刻意声明为同步 def：处理器内部是阻塞调用（ES 查询、向量检索、LLM 抽取校验），
# FastAPI 会把 def 路由放进线程池；写成 async def 会让这些调用独占单进程的事件循环。
# 参见 api/v1/search.py。
router = APIRouter(prefix="/kg", tags=["知识图谱"])
repo = ElasticsearchRepository()
analysis_service = StructuredAnalysisService()
# 抽取服务含 ES 客户端与 LLM 客户端，模块级单例复用，避免每请求重建连接
extractor = KGExtractionService()
# 跨法域概念对齐：只读知识层做聚合，无 LLM 调用
alignment_service = KGAlignmentService()
# 图算法（社区/中心性）：纯拓扑计算，无 LLM 调用
graph_analysis = KGGraphAnalysisService()
# 图谱多跳推理问答（含 LLM 调用），独立于既有 qa_service
reasoning_service = KGReasoningService()


def _law_to_chunks(law_data: dict) -> list:
    """将 get_law_by_id 的返回值转为 chunk 列表"""
    if not law_data:
        return []
    chunks = []
    law_id = law_data.get("law_id", "")
    title = law_data.get("title", "")
    jurisdiction = law_data.get("jurisdiction", "")
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


@router.get("/law/{law_id}")
def get_law_graph(
    law_id: str,
    force_refresh: bool = Query(False, description="忽略缓存强制重新抽取知识层"),
):
    """
    获取指定法规的知识图谱。

    优先返回 LLM 抽取的实体关系三元组图（知识层：实体节点 + 类型化关系）；
    当知识层无结果或抽取/读取失败时，自动降级为基于关键词的条款关联图。
    返回体中的 graph_source 标明来源（llm / keyword_fallback）。
    """
    law_data = repo.get_law_by_id(law_id)
    if not law_data:
        return APIResponse(success=False, message="未找到该法规的数据")

    chunks = _law_to_chunks(law_data)

    # ---- 优先：LLM 知识层（命中缓存则不发起 LLM 请求）----
    try:
        result = extractor.get_triples_cached(law_id, chunks, force_refresh=force_refresh)
        triples = result.get("triples") or []
        if triples:
            graph = extractor.format_graph(triples, articles=chunks)
            graph["cache_hit"] = result.get("cache_hit", False)
            # 附上社区划分与中心性，供前端按社区染色 / 按 PageRank 定大小
            enrich_graph_with_analytics(graph, graph_analysis)
            return APIResponse(success=True, data=graph)
        logger.info(f"[KG] 法规 {law_id} 知识层为空，降级为关键词建图")
    except Exception as e:
        logger.warning(
            f"[KG] 法规 {law_id} 知识层构建失败，降级为关键词建图: "
            f"{type(e).__name__}: {e}"
        )

    # ---- 降级：关键词版条款关联图 ----
    graph = analysis_service.get_network_graph(chunks, [], center_law_id=law_id)
    graph["graph_source"] = "keyword_fallback"
    enrich_graph_with_analytics(graph, graph_analysis)
    return APIResponse(success=True, data=graph)


@router.get("/search-entity")
def search_entity(
    q: str = Query(..., description="实体搜索关键词"),
    page_size: int = Query(20, description="返回结果数"),
):
    """搜索实体并构建邻域图谱"""
    from services.retriever_service import RetrieverService
    retriever = RetrieverService()
    results = retriever.hybrid_search(q, page_size)
    if not results:
        return APIResponse(success=False, message="未找到相关结果")

    chunks = []
    for r in results:
        chunks.append({
            "law_id": r.get("law_id", ""),
            "article_number": r.get("article_number", ""),
            "content": r.get("content", ""),
            "title": r.get("title", ""),
            "jurisdiction": r.get("jurisdiction", ""),
            "chunk_index": r.get("chunk_index", 0),
        })

    graph = analysis_service.get_network_graph(chunks, [])
    return APIResponse(success=True, data=graph)


@router.get("/statistics")
def get_kg_statistics():
    """知识层统计：已抽取法规数、三元组总数、实体总数"""
    return APIResponse(success=True, data=repo.get_kg_statistics())


@router.get("/entities")
def list_entities(
    q: Optional[str] = Query(None, description="按规范名或别名模糊匹配"),
    entity_type: Optional[str] = Query(None, description="实体类型过滤"),
    page_size: int = Query(50, description="返回结果数", le=200),
):
    """列出实体词典条目（消歧后的规范实体，含别名与出现法域）"""
    return APIResponse(
        success=True,
        data=repo.list_kg_entities(q=q, entity_type=entity_type, page_size=page_size),
    )


@router.get("/entities/aliases")
def get_alias_statistics():
    """别名表覆盖范围统计（消歧规则的规模）"""
    from services.entity_disambiguation_service import get_alias_statistics as _stats
    return APIResponse(success=True, data=_stats())


@router.get("/extraction/{law_id}")
def get_llm_extraction(
    law_id: str,
    force_refresh: bool = Query(False, description="忽略缓存强制重新抽取（本体变更后使用）"),
):
    """
    用 LLM 对该法规进行实体关系抽取（返回三元组格式的知识图谱）

    抽取结果持久化在 legal_kg_triples 索引中：
      - 命中缓存（法规指纹未变）时不调用 LLM，直接读 ES；
      - 法规修订后指纹变化，仅该法规自动重抽，其余法规不受影响。
    """
    law_data = repo.get_law_by_id(law_id)
    if not law_data:
        return APIResponse(success=False, message="未找到该法规的数据")

    result = extractor.get_triples_cached(
        law_id, _law_to_chunks(law_data), force_refresh=force_refresh
    )
    graph = extractor.format_graph(result["triples"])
    graph["cache_hit"] = result["cache_hit"]
    return APIResponse(success=True, data=graph)


@router.get("/entity-graph")
def get_entity_graph(
    q: Optional[str] = Query(None, description="实体名称关键词（规范名或别名）"),
    entity_id: Optional[str] = Query(None, description="规范实体 ID（精确命中，优先于 q）"),
    depth: int = Query(1, ge=1, le=2, description="邻域扩展跳数"),
    limit: int = Query(50, ge=1, le=200, description="种子实体数与返回规模上限"),
):
    """
    按实体构建邻域知识图谱（实体节点 + 类型化关系边）。

    - entity_id：精确命中一个规范实体；
    - q：按规范名/别名模糊匹配，取前 limit 个作为种子。
    邻域扩展 1~2 跳；无命中时返回空图，由前端降级到条款检索建图。
    仅依赖实体消歧后的 canonical_id 关联，因此可跨法规/跨法域连通同一实体。
    """
    seed_ids: list = []
    if entity_id:
        seed_ids = [entity_id]
    elif q:
        seeds = repo.list_kg_entities(q=q, page_size=limit)
        seed_ids = [s.get("entity_id", "") for s in seeds if s.get("entity_id")]

    if not seed_ids:
        return APIResponse(
            success=True,
            data={"nodes": [], "edges": [], "graph_source": "llm", "seed_entities": []},
        )

    triples = repo.get_kg_triples_by_entities(seed_ids, size=limit * 40)

    # 2 跳：把一跳邻居实体并入种子再查一次（受 limit 约束，控制规模）
    if depth >= 2 and triples:
        seed_set = set(seed_ids)
        neighbor_ids = set()
        for t in triples:
            for side in ("head", "tail"):
                cid = t[side].get("canonical_id") or t[side].get("id")
                if cid and cid not in seed_set:
                    neighbor_ids.add(cid)
        if neighbor_ids:
            more = repo.get_kg_triples_by_entities(
                list(neighbor_ids)[: limit * 4], size=limit * 40
            )
            seen = {
                (
                    t["head"].get("canonical_id") or t["head"].get("id"),
                    t.get("relation"),
                    t["tail"].get("canonical_id") or t["tail"].get("id"),
                    t.get("source_article"),
                )
                for t in triples
            }
            for t in more:
                key = (
                    t["head"].get("canonical_id") or t["head"].get("id"),
                    t.get("relation"),
                    t["tail"].get("canonical_id") or t["tail"].get("id"),
                    t.get("source_article"),
                )
                if key not in seen:
                    seen.add(key)
                    triples.append(t)

    graph = extractor.format_graph(triples)
    graph["seed_entities"] = seed_ids
    enrich_graph_with_analytics(graph, graph_analysis)
    return APIResponse(success=True, data=graph)


@router.get("/analytics")
def get_analytics(
    law_id: Optional[str] = Query(None, description="指定则分析单个法规的实体图，否则分析全局"),
    top_n: int = Query(10, ge=5, le=30, description="榜单长度"),
):
    """
    知识图谱拓扑分析（社区检测 + PageRank + 介数中心性），零 LLM 调用。

    返回社区划分、核心枢纽榜单（PageRank）、桥接概念榜单（介数中心性）。
    不传 law_id 时对全局实体图分析（可用于"全局核心实体/主题簇"总览）。
    """
    if law_id:
        law_data = repo.get_law_by_id(law_id)
        if not law_data:
            return APIResponse(success=False, message="未找到该法规的数据")
        triples = repo.get_kg_triples(law_id)
        graph = extractor.format_graph(triples, articles=_law_to_chunks(law_data))
    else:
        graph = extractor.format_graph(repo.get_all_kg_triples())

    return APIResponse(
        success=True,
        data=graph_analysis.analyze(
            graph.get("nodes", []), graph.get("edges", []), top_n=top_n
        ),
    )


@router.get("/concept-matrix")
def get_concept_matrix(
    top_n: int = Query(30, ge=5, le=100, description="返回的概念行数上限"),
    min_jurisdictions: int = Query(2, ge=1, le=8, description="概念至少出现在这么多个法域"),
):
    """
    跨法域概念对齐矩阵：概念 × 法域 的条文数总览（热力图）。

    单元格值 = 该概念在该法域下去重后的条文数。默认只返回跨 ≥2 法域的跨域概念
    （跨域概念才有对比价值），按法域覆盖数与条文总数排序。
    纯知识层聚合，无 LLM 调用。
    """
    return APIResponse(
        success=True,
        data=alignment_service.get_concept_matrix(
            top_n=top_n, min_jurisdictions=min_jurisdictions
        ),
    )


@router.get("/concept-detail")
def get_concept_detail(
    entity_id: str = Query(..., description="概念的规范实体 ID"),
):
    """单概念下钻：各法域中涉及该概念的法规 / 条款号 / 关系类型。"""
    data = alignment_service.get_concept_detail(entity_id)
    if not data:
        return APIResponse(success=False, message="未找到该概念的相关条款")
    return APIResponse(success=True, data=data)


@router.post("/reasoning-qa")
def reasoning_qa(payload: Dict = Body(...)):
    """
    图谱多跳推理问答（GraphRAG 局部检索）。

    请求体：{"question": "...", "top_k_entities": 5, "hops": 2}
    流程：实体链接 → 扩 2 跳子图 → 组装「三元组 + 条款原文」证据 → 1 次 LLM 生成，
    返回答案与可溯源的关系路径（used_paths）和引用条款（cited_articles）。

    仅对已完成知识层抽取的法规有效；无命中时返回 found=False，调用方可回退普通问答。
    """
    question = (payload or {}).get("question", "")
    try:
        top_k = int((payload or {}).get("top_k_entities", 5) or 5)
    except (TypeError, ValueError):
        top_k = 5
    try:
        hops = int((payload or {}).get("hops", 2) or 2)
    except (TypeError, ValueError):
        hops = 2
    hops = max(1, min(2, hops))
    return APIResponse(
        success=True,
        data=reasoning_service.answer(question, top_k_entities=top_k, hops=hops),
    )

