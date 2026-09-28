from fastapi import APIRouter, Query
from typing import List, Optional
from models.schemas import APIResponse, SearchResult, DocumentMetadata, SimilarLawResponse, SimilarChunkResponse
from services.retriever_service import RetrieverService
from services.similar_law_service import SimilarLawService
from services.summary_service import SummaryService
from repositories.elasticsearch import ElasticsearchRepository
from utils.law_catalog import LawCatalog

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
    # 将逗号分隔的主题字符串转为列表
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
        ))

    return APIResponse(success=True, data=search_results)


@router.get("/laws")
def get_laws(
    jurisdiction: str = Query(None, description="法域筛选"),
    topics: str = Query(None, description="主题筛选，逗号分隔")
):
    """获取所有法规列表

    以 LawCatalog (data/law_catalog.json) 为权威数据源，保证返回所有已入库的法规元信息。
    仅当 ES 中存在对应索引数据时，附加实时 chunk_count/article_count 统计。
    """
    # 1) 从 catalog 获取所有非删除状态的法规
    cat_data = catalog.load()
    all_laws = [l for l in cat_data.get("laws", []) if l.get("status") != "deleted"]

    # 2) 法域筛选（catalog 中 jurisdiction 已为中文）
    if jurisdiction:
        all_laws = [l for l in all_laws if l.get("jurisdiction") == jurisdiction]

    # 3) 主题筛选（多选 OR，支持中文标签）
    topic_filter = None
    if topics:
        topic_filter = set(t.strip() for t in topics.split(",") if t.strip())
        if topic_filter:
            all_laws = [l for l in all_laws
                        if any(t in (l.get("topics") or []) for t in topic_filter)]

    # 4) 可选：从 ES 拉取实际索引统计（仅命中 ES 的法规补充 chunk_count）
    es_stats: dict = {}
    try:
        es_laws = repo.get_all_laws()
        es_stats = {law["law_id"]: law for law in es_laws if law.get("law_id")}
    except Exception:
        es_stats = {}

    # 5) 转换为前端期望的响应字段名（保持向后兼容）
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
def get_law(law_id: str):
    """根据law_id获取法规详情"""
    law = repo.get_law_by_id(law_id)
    if law is None:
        return APIResponse(success=False, message="法规不存在")
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
    """获取所有主题列表（含统计信息，用于筛选下拉框）

    返回格式:
    [
        {
            "id": "cybersecurity",
            "label_zh": "网络安全",
            "count": 15,
            "by_jurisdiction": {"澳门": 10, "新加坡": 5}
        },
        ...
    ]
    """
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
    """
    相似法规推荐接口

    根据当前浏览的法规，推荐最相似的 Top-N 部法规。

    内部流程：
    1. 获取该法规所有 chunk 的向量，计算平均向量作为"核心语义"
    2. 用平均向量在 ES 中执行 KNN 相似搜索
    3. 排除自身，按 law_id 聚合去重
    4. 返回去重后的法规级推荐列表
    """
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
    """
    相似条款推荐接口

    根据当前浏览的某一条具体条款，推荐最相似的条款片段。
    直接使用该 chunk 的向量进行搜索，不做平均。
    """
    result = similar_service.get_similar_chunks(
        query_chunk_id=chunk_id,
        top_k=top_k,
        jurisdiction_filter=jurisdiction
    )
    return APIResponse(success=True, data=result)


@router.get("/law/{law_id}/summary")
def get_law_summary(law_id: str):
    """
    法规摘要生成接口（带缓存）
    
    根据法规ID生成该法规的结构化摘要，帮助用户快速了解法规核心内容。
    支持缓存机制：
    - 第二次访问同一法规时直接从缓存读取，减少响应时间
    - 当法规数据发生变化时自动清除缓存并重新生成
    
    返回格式：
    {
      "law_name": "法规全称",
      "purpose": "立法目的（1-2句话）",
      "scope": "适用范围（1-2句话）",
      "core_points": ["要点1", "要点2", "要点3", "要点4", "要点5"],
      "chapters": [
        {"title": "章节名", "articles": "第X-Y条", "summary": "一句话概括"}
      ],
      "keywords": ["关键词1", "关键词2", ...],
      "from_cache": true/false  // 新增字段：是否来自缓存
    }
    """
    try:
        import traceback
        summary = summary_service.generate_summary_with_cache(law_id)
        return APIResponse(success=True, data=summary)
    except ValueError as e:
        return APIResponse(success=False, message=f"法规不存在：{str(e)}")
    except Exception as e:
        error_type = type(e).__name__
        error_detail = str(e)[:200]
        # 记录详细错误到控制台
        print(f"[摘要API错误] {error_type}: {error_detail}")
        # 根据错误类型返回不同的提示
        if "timeout" in str(e).lower() or "timed out" in str(e).lower():
            return APIResponse(success=False, message="摘要生成超时，请稍后重试")
        elif "connection" in str(e).lower():
            return APIResponse(success=False, message="无法连接到AI服务")
        else:
            return APIResponse(success=False, message=f"生成摘要失败：{error_type}")


@router.get("/summary/cache/info")
def get_summary_cache_info():
    """
    获取摘要缓存统计信息
    
    返回缓存的总体状态，包括缓存条目数、有效条目数等。
    """
    from services.summary_cache import summary_cache
    info = summary_cache.get_cache_info()
    return APIResponse(success=True, data=info)


@router.delete("/summary/cache/{law_id}")
def invalidate_summary_cache(law_id: str):
    """
    清除指定法规的摘要缓存
    
    当法规数据发生变化时调用此接口，下次访问时将重新生成摘要。
    
    Args:
        law_id: 法规ID
    """
    from services.summary_cache import summary_cache
    summary_cache.invalidate(law_id)
    return APIResponse(success=True, message=f"已清除法规 {law_id} 的摘要缓存")


@router.delete("/summary/cache")
def invalidate_all_summary_cache():
    """
    清除所有法规摘要缓存
    
    适用于批量更新法规数据后，需要重新生成所有摘要的场景。
    """
    from services.summary_cache import summary_cache
    summary_cache.invalidate_all()
    return APIResponse(success=True, message="已清除所有法规摘要缓存")
