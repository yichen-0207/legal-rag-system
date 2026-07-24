from fastapi import APIRouter, Query
from typing import Optional
from models.schemas import APIResponse
from repositories.elasticsearch import ElasticsearchRepository
from services.structured_analysis_service import StructuredAnalysisService

router = APIRouter(prefix="/kg", tags=["知识图谱"])
repo = ElasticsearchRepository()
analysis_service = StructuredAnalysisService()


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
async def get_law_graph(law_id: str):
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
async def search_entity(
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


@router.get("/extraction/{law_id}")
async def get_llm_extraction(law_id: str):
    """
    用 LLM 对该法规进行实体关系抽取（返回三元组格式的知识图谱）
    """
    law_data = repo.get_law_by_id(law_id)
    if not law_data:
        return APIResponse(success=False, message="未找到该法规的数据")

    from services.kg_extraction_service import KGExtractionService
    extractor = KGExtractionService()
    graph = extractor.extract_and_format_for_graph(_law_to_chunks(law_data))
    return APIResponse(success=True, data=graph)
