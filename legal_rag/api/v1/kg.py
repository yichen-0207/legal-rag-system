from fastapi import APIRouter, Query
from typing import Optional
from models.schemas import APIResponse
from repositories.elasticsearch import ElasticsearchRepository
from services.structured_analysis_service import StructuredAnalysisService
from services.kg_extraction_service import KGExtractionService

# 以下路由刻意声明为同步 def：处理器内部是阻塞调用（ES 查询、向量检索、LLM 抽取校验），
# FastAPI 会把 def 路由放进线程池；写成 async def 会让这些调用独占单进程的事件循环。
# 参见 api/v1/search.py。
router = APIRouter(prefix="/kg", tags=["知识图谱"])
repo = ElasticsearchRepository()
analysis_service = StructuredAnalysisService()
# 抽取服务含 ES 客户端与 LLM 客户端，模块级单例复用，避免每请求重建连接
extractor = KGExtractionService()


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
def get_law_graph(law_id: str):
    """
    获取指定法规内部条款的知识图谱（实体类型标注 + 同法规关联）
    """
    law_data = repo.get_law_by_id(law_id)
    if not law_data:
        return APIResponse(success=False, message="未找到该法规的数据")

    chunks_a = _law_to_chunks(law_data)
    graph = analysis_service.get_network_graph(chunks_a, [], center_law_id=law_id)
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
