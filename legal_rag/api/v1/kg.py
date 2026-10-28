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
from services.kg_quality_service import KGQualityService
from services.multihop_service import MultiHopService

logger = logging.getLogger(__name__)

# 以下路由刻意声明为同步 def：处理器内部是阻塞调用（ES 查询、向量检索、LLM 抽取校验），
# FastAPI 会把 def 路由放进线程池；写成 async def 会独占单进程的事件循环（参见 api/v1/search.py）。
router = APIRouter(prefix="/kg", tags=["知识图谱"])
repo = ElasticsearchRepository()
analysis_service = StructuredAnalysisService()
# 抽取服务含 ES 客户端与 LLM 客户端，模块级单例复用，避免每请求重建连接
extractor = KGExtractionService()
# 跨法域概念对齐：只读知识层聚合，无 LLM 调用
alignment_service = KGAlignmentService()
# 图算法（社区/中心性）：纯拓扑计算，无 LLM 调用
graph_analysis = KGGraphAnalysisService()
# 图谱多跳推理问答（含 LLM 调用），独立于既有 qa_service
reasoning_service = KGReasoningService()
# 质量与治理指标：只读知识层统计，无 LLM 调用
quality_service = KGQualityService()
# 跨法域对齐多跳问答：检索 + 预计算质心 + 概念交叉校验，仅最终合成 1 次 LLM
multihop_service = MultiHopService()


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
    extract: bool = Query(
        False, description="允许在无缓存时调用 LLM 抽取（默认关闭，避免浏览即产生费用）"
    ),
):
    """获取法规知识图谱：优先 LLM 三元组知识层，为空或失败时降级为关键词条款关联图。

    默认**只读缓存**：未抽取过的法规直接返回关键词图，并把 graph_source 标为
    `not_extracted`。此前这里无条件调用 `get_triples_cached`，打开一部法规图谱页
    就会触发几十次 LLM 调用（如澳大利亚单部法规约 58 次、约 17 万 tokens），
    在大批量数据入库后属于不可控成本。需要按需补抽时显式传 `extract=true`，
    批量补抽走 `run_kg_extraction.py`。
    """
    law_data = repo.get_law_by_id(law_id)
    if not law_data:
        return APIResponse(success=False, message="未找到该法规的数据")

    chunks = _law_to_chunks(law_data)

    # ---- 优先：LLM 知识层（命中缓存则不发起 LLM 请求）----
    try:
        if extract or force_refresh:
            result = extractor.get_triples_cached(
                law_id, chunks, force_refresh=force_refresh
            )
            triples = result.get("triples") or []
            cache_hit = result.get("cache_hit", False)
        else:
            triples = repo.get_kg_triples(law_id)
            cache_hit = True
        if triples:
            graph = extractor.format_graph(triples, articles=chunks)
            graph["cache_hit"] = cache_hit
            graph["graph_source"] = "knowledge"
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
    graph["kg_extracted"] = False
    graph["hint"] = "该法规知识层尚未抽取，当前为关键词关联图；可用 extract=true 按需抽取"
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


@router.get("/quality")
def get_kg_quality():
    """知识图谱质量与治理指标（覆盖率/平均度/密度/孤立实体率/类型分布），无 LLM 调用。"""
    return APIResponse(success=True, data=quality_service.get_quality_report())


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


@router.get("/review-queue")
def get_review_queue(
    actionable_only: bool = Query(True, description="仅返回可行动候选（放宽相似度阈值即可归并）"),
    blocked_by: Optional[str] = Query(None, description="按拦截原因过滤，如 similarity_below_threshold"),
    min_similarity: float = Query(0.0, ge=0, le=1, description="相似度下限"),
    page_size: int = Query(50, ge=1, le=200, description="返回条数上限"),
):
    """实体合并候选复核队列（只读）：列出被 L3 守卫拦下的高相似实体对，不改写别名表。"""
    return APIResponse(
        success=True,
        data=repo.list_kg_review_candidates(
            actionable_only=actionable_only,
            blocked_by=blocked_by,
            min_similarity=min_similarity,
            page_size=page_size,
        ),
    )


@router.get("/extraction/{law_id}")
def get_llm_extraction(
    law_id: str,
    force_refresh: bool = Query(False, description="忽略缓存强制重新抽取（本体变更后使用）"),
):
    """用 LLM 抽取该法规的实体关系三元组；结果按法规指纹缓存，修订后仅该法规重抽。"""
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
    """按实体构建 1~2 跳邻域知识图谱（entity_id 精确命中优先，q 按规范名/别名模糊匹配取种子）。"""
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
    """知识图谱拓扑分析（社区检测 + PageRank + 介数中心性），零 LLM 调用。"""
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
    """跨法域概念对齐矩阵（概念 × 法域条文数，热力图），默认只返回跨 ≥2 法域的跨域概念。"""
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
    """图谱多跳推理问答（GraphRAG 局部检索）：实体链接→2 跳子图→路径校验→1 次 LLM 生成。"""
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
    deepen_raw = (payload or {}).get("deepen")
    deepen = None if deepen_raw is None else bool(deepen_raw)
    synthesize = bool((payload or {}).get("synthesize", True))
    jur_raw = (payload or {}).get("jurisdictions")
    jurisdictions = (
        [str(j).strip() for j in jur_raw if str(j).strip()]
        if isinstance(jur_raw, list) else None
    )
    return APIResponse(
        success=True,
        data=reasoning_service.answer(
            question, top_k_entities=top_k, hops=hops,
            deepen=deepen, synthesize=synthesize, jurisdictions=jurisdictions,
        ),
    )


@router.post("/multihop-qa")
def multihop_qa(payload: Dict = Body(...)):
    """跨法域对齐多跳问答：源条款→对应法规→目标条款→共享概念交叉校验→1 次 LLM 合成。"""
    question = ((payload or {}).get("question") or "").strip()
    source_law_id = (payload or {}).get("source_law_id") or None
    source_article = (payload or {}).get("source_article") or None
    try:
        top_k_laws = int((payload or {}).get("top_k_laws", 3) or 3)
    except (TypeError, ValueError):
        top_k_laws = 3
    top_k_laws = max(1, min(5, top_k_laws))
    synthesize = bool((payload or {}).get("synthesize", True))
    jur_raw = (payload or {}).get("jurisdictions")
    jurisdictions = (
        [str(j).strip() for j in jur_raw if str(j).strip()]
        if isinstance(jur_raw, list) else None
    )
    return APIResponse(
        success=True,
        data=multihop_service.answer(
            question,
            source_law_id=source_law_id,
            source_article=source_article,
            top_k_laws=top_k_laws,
            synthesize=synthesize,
            jurisdictions=jurisdictions,
        ),
    )

