from fastapi import APIRouter, Query
from typing import List, Optional
from core.jurisdiction import normalize_jurisdiction
from models.schemas import APIResponse, SearchResult, DocumentMetadata, SimilarLawResponse, SimilarChunkResponse
from services.retriever_service import RetrieverService
from services.similar_law_service import SimilarLawService
from services.summary_service import SummaryService
from repositories.elasticsearch import ElasticsearchRepository, is_excluded_topic
from utils.law_catalog import LawCatalog
from utils.law_parts import enrich_parts

router = APIRouter(prefix="/search", tags=["搜索"])
retriever = RetrieverService()
repo = ElasticsearchRepository()
similar_service = SimilarLawService()
summary_service = SummaryService()
catalog = LawCatalog()


@router.get("")
@router.get("/")
def search(
    query: str = Query("", description="搜索关键词（为空时仅按筛选条件检索）"),
    top_k: int = Query(50, description="返回结果数量"),
    jurisdiction: str = Query(None, description="法域筛选（澳门/新加坡等）"),
    topics: Optional[str] = Query(None, description="主题筛选（多选用逗号分隔，如：网络安全,金融监管）")
):
    topic_list = None
    if topics:
        topic_list = [t.strip() for t in topics.split(",") if t.strip()]
        if not topic_list:
            topic_list = None

    results = retriever.hybrid_search(query, top_k, jurisdiction, topic_list)

    search_results = []
    for result in results:
        search_results.append(SearchResult(
            content=result['content'],
            metadata=DocumentMetadata(
                law_id=result['law_id'],
                jurisdiction=result['jurisdiction'],
                title=result['title'],
                article_number=result['article_number'],
                chunk_index=result['chunk_index'],
                topics=result.get('topics', []),
                topic_labels=result.get('topic_labels', [])
            ),
            similarity=result['score'],
            retrieval_hop=result.get('retrieval_hop'),
            referenced_by=result.get('referenced_by'),
            cross_law=result.get('cross_law'),
            referenced_law_id=result.get('referenced_law_id'),
            referenced_law_name=result.get('referenced_law_name'),
            match_score=result.get('match_score'),
        ))

    return APIResponse(success=True, data=search_results)


@router.get("/laws")
def get_laws(
    jurisdiction: str = Query(None, description="法域筛选"),
    topics: str = Query(None, description="主题筛选，逗号分隔")
):
    """获取法规列表：以 LawCatalog 为权威数据源，ES 命中时补充实时 chunk/article 统计。"""
    cat_data = catalog.load()
    all_laws = [l for l in cat_data.get("laws", []) if l.get("status") != "deleted"]

    if jurisdiction:
        target = normalize_jurisdiction(jurisdiction)
        all_laws = [l for l in all_laws if normalize_jurisdiction(l.get("jurisdiction")) == target]

    # 主题筛选（多选 OR，支持中文标签）
    topic_filter = None
    if topics:
        topic_filter = set(t.strip() for t in topics.split(",") if t.strip())
        if topic_filter:
            all_laws = [l for l in all_laws
                        if any(t in (l.get("topics") or []) for t in topic_filter)]

    # 可选：从 ES 拉取实际索引统计（仅命中 ES 的法规补充 chunk_count）
    es_stats: dict = {}
    try:
        es_laws = repo.get_all_laws()
        es_stats = {law["law_id"]: law for law in es_laws if law.get("law_id")}
    except Exception:
        es_stats = {}

    # 转换为前端期望的响应字段名（保持向后兼容）
    result = []
    for law in all_laws:
        law_id = law.get("law_id", "")
        es_info = es_stats.get(law_id, {})
        result.append({
            "law_id": law_id,
            "title": law.get("law_name_zh") or law.get("law_name_original") or "",
            "jurisdiction": law.get("jurisdiction", ""),
            "chunk_count": es_info.get("chunk_count") or law.get("total_chunks", 0),
            "article_count": es_info.get("article_count") or law.get("total_articles", 0),
            # 额外字段（前端可选使用）
            "passing_date": law.get("passing_date", ""),
            "effective_date": law.get("effective_date", ""),
            "publication_date": law.get("publication_date", ""),
            "topics": law.get("topics", []),
            "status": law.get("status", "active"),
            "indexed_in_es": bool(es_info),
        })

    return APIResponse(success=True, data=result)


@router.get("/law/{law_id}")
def get_law(
    law_id: str,
    page: int = Query(1, ge=1, description="页码（从 1 开始）"),
    page_size: int = Query(1000, ge=1, le=2000, description="每页文档块数"),
    outline: bool = Query(False, description="是否返回完整条款目录（供前端生成目录导航）")
):
    """根据law_id获取法规详情（支持分页与条款目录）"""
    try:
        law = repo.get_law_by_id(law_id, page=page, page_size=page_size, include_outline=outline)
    except ValueError as e:
        return APIResponse(success=False, message=str(e))
    if law is None:
        return APIResponse(success=False, message="法规不存在")

    # 合并法规目录（catalog）中的元数据：ES 文档仅存块级字段，法条级元信息以 catalog 为准
    entry = catalog.find_by_law_id(law_id) or {}
    if entry:
        law["law_name_zh"] = entry.get("law_name_zh") or law.get("title", "")
        law["title_original"] = entry.get("law_name_original") or law.get("title_original", "")
        law["law_type"] = entry.get("law_type", "")
        law["status"] = entry.get("status", "active")
        law["passing_date"] = entry.get("passing_date") or law.get("passing_date", "")
        law["effective_date"] = entry.get("effective_date") or law.get("effective_date", "")
        law["publication_date"] = entry.get("publication_date") or law.get("publication_date", "")
        law["total_articles"] = entry.get("total_articles") or law.get("total_articles", 0)
        law["chunk_count"] = entry.get("total_chunks") or law.get("total_chunks", 0)
        if entry.get("topics"):
            law["topics"] = entry["topics"]
    else:
        law["law_name_zh"] = law.get("title", "")
        law["law_type"] = ""
        law["status"] = "active"
        law["total_articles"] = law.get("total_articles", 0)
        law["chunk_count"] = law.get("total_chunks", 0)

    # 剔除已废弃的粗粒度主题，避免在法规详情页继续展示
    if isinstance(law.get("topics"), list):
        law["topics"] = [t for t in law["topics"] if t and not is_excluded_topic(t)]

    # 补全章节名：ES 中的 part_title 可能只有「第一章」这类裸序号
    if law.get("outline"):
        law["outline"] = enrich_parts(law_id, law["outline"])
    if law.get("chunks"):
        law["chunks"] = enrich_parts(law_id, law["chunks"])

    return APIResponse(success=True, data=law)


@router.get("/jurisdictions")
@router.get("/jurisdictions/")
def get_jurisdictions():
    """获取所有法域列表"""
    jurisdictions = repo.get_all_jurisdictions()
    return APIResponse(success=True, data=jurisdictions)


@router.get("/topics")
@router.get("/topics/")
def get_topics():
    """获取所有主题列表（含统计信息，用于筛选下拉框）"""
    topics = repo.get_all_topics()
    return APIResponse(success=True, data=topics)


@router.get("/topics/simple")
@router.get("/topics/simple/")
def get_topics_simple():
    """获取简单主题标签列表（仅名称，用于轻量级场景）"""
    topics = repo.get_topic_filter_values()
    return APIResponse(success=True, data=topics)


@router.get("/law/{law_id}/similar")
def get_similar_laws(
    law_id: str,
    top_k: int = Query(5, ge=1, le=20, description="推荐数量（1-20）"),
    jurisdiction: str = Query(None, description="法域过滤（如：澳门/新加坡），不填则跨法域推荐")
):
    """相似法规推荐：法规所有 chunk 求平均向量做 KNN，排除自身并按 law_id 聚合去重。"""
    result = similar_service.get_similar_laws(
        query_law_id=law_id,
        top_k=top_k,
        jurisdiction_filter=jurisdiction
    )
    return APIResponse(success=True, data=result)


@router.get("/chunk/{chunk_id}/similar")
def get_similar_chunks(
    chunk_id: str,
    top_k: int = Query(10, ge=1, le=30, description="推荐条款数量（1-30）"),
    jurisdiction: str = Query(None, description="法域过滤")
):
    """相似条款推荐：直接用该 chunk 的向量检索，不做平均。"""
    result = similar_service.get_similar_chunks(
        query_chunk_id=chunk_id,
        top_k=top_k,
        jurisdiction_filter=jurisdiction
    )
    return APIResponse(success=True, data=result)


@router.get("/law/{law_id}/summary")
def get_law_summary(law_id: str):
    """生成法规结构化摘要（带缓存；法规数据变化后缓存自动失效并重新生成）。"""
    try:
        import traceback
        summary = summary_service.generate_summary_with_cache(law_id)
        return APIResponse(success=True, data=summary)
    except ValueError as e:
        return APIResponse(success=False, message=f"法规不存在：{str(e)}")
    except Exception as e:
        error_type = type(e).__name__
        error_detail = str(e)[:200]
        print(f"[摘要API错误] {error_type}: {error_detail}")
        if "timeout" in str(e).lower() or "timed out" in str(e).lower():
            return APIResponse(success=False, message="摘要生成超时，请稍后重试")
        elif "connection" in str(e).lower():
            return APIResponse(success=False, message="无法连接到AI服务")
        else:
            return APIResponse(success=False, message=f"生成摘要失败：{error_type}")


@router.get("/summary/cache/info")
def get_summary_cache_info():
    """获取摘要缓存统计信息（缓存条目数、有效条目数等）。"""
    from services.summary_cache import summary_cache
    info = summary_cache.get_cache_info()
    return APIResponse(success=True, data=info)


@router.delete("/summary/cache/{law_id}")
def invalidate_summary_cache(law_id: str):
    """清除指定法规的摘要缓存（法规数据变化后调用，下次访问会重新生成）。"""
    from services.summary_cache import summary_cache
    summary_cache.invalidate(law_id)
    return APIResponse(success=True, message=f"已清除法规 {law_id} 的摘要缓存")


@router.delete("/summary/cache")
def invalidate_all_summary_cache():
    """清除所有法规摘要缓存（批量更新法规数据后使用）。"""
    from services.summary_cache import summary_cache
    summary_cache.invalidate_all()
    return APIResponse(success=True, message="已清除所有法规摘要缓存")
