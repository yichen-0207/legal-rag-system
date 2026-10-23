from fastapi import APIRouter, Query, Body
from fastapi.responses import StreamingResponse
from models.schemas import APIResponse
from services.structured_analysis_service import StructuredAnalysisService
import json, asyncio

# /stream 之外的以下路由刻意声明为同步 def：处理器内部是阻塞调用（ES 检索、LLM 解读、
# 图计算），FastAPI 会把 def 路由放进线程池；写成 async def 会独占单进程的事件循环。
# /stream 的生成器已用 run_in_executor 显式卸载阻塞迭代，必须保持 async。
router = APIRouter(prefix="/structured-analysis", tags=["结构化专题分析"])
service = StructuredAnalysisService()


@router.get("/topics")
def list_topics():
    """获取所有预设对比专题列表"""
    try:
        topics = service.list_topics()
        return APIResponse(success=True, data=topics)
    except Exception as e:
        return APIResponse(success=False, message=f"获取专题列表失败: {str(e)}")


@router.get("/stream")
async def run_analysis_stream(
    topic_id: str = Query(..., description="预设专题ID"),
    jurisdiction_a: str = Query(..., description="法域A"),
    jurisdiction_b: str = Query(..., description="法域B"),
    top_n: int = Query(10, ge=3, le=20, description="每个法域检索条款数"),
    include_summary: bool = Query(True, description="是否生成LLM差异总结")
):
    """流式执行跨法域结构化专题分析（SSE，含 dashboard_done / network_done 事件）。"""
    async def generate():
        loop = asyncio.get_event_loop()
        _SENTINEL = object()
        iterator = iter(service.analyze_stream(
            topic_id=topic_id,
            jurisdiction_a=jurisdiction_a,
            jurisdiction_b=jurisdiction_b,
            top_n_per_jurisdiction=top_n,
            include_summary=include_summary
        ))
        while True:
            # 用 sentinel 替代 StopIteration，避免 Future 传播异常
            event = await loop.run_in_executor(None, next, iterator, _SENTINEL)
            if event is _SENTINEL:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            # 强制事件循环处理挂起的写操作（刷新 socket 缓冲区）
            await asyncio.sleep(0)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================
#  升级接口：仪表盘 / AI解读 / 网络图
# ============================================================

@router.post("/dashboard")
def get_dashboard(
    analysis_result: dict = Body(..., description="已有的结构化分析结果")
):
    """基于已有分析结果生成仪表盘数据（保护力度评分 / 雷达图得分 / 关键差异点）。"""
    try:
        dashboard = service.generate_dashboard(analysis_result)
        return APIResponse(success=True, data=dashboard)
    except Exception as e:
        return APIResponse(success=False, message=f"生成仪表盘失败: {str(e)}")


@router.post("/ai-interpret")
def ai_interpret(
    request_data: dict = Body(..., description="解读参数：dimension_label/detail_a/detail_b/jurisdiction_a/jurisdiction_b/topic_name")
):
    """AI 深度解读某个差异维度；条款正文较长，故经请求体传入以免 URL 超长 414。"""
    try:
        interpretation = service.ai_interpret(
            dimension_label=request_data.get("dimension_label", ""),
            detail_a=request_data.get("detail_a", ""),
            detail_b=request_data.get("detail_b", ""),
            jurisdiction_a=request_data.get("jurisdiction_a", ""),
            jurisdiction_b=request_data.get("jurisdiction_b", ""),
            topic_name=request_data.get("topic_name", "")
        )
        return APIResponse(success=True, data={"interpretation": interpretation})
    except Exception as e:
        return APIResponse(success=False, message=f"AI解读失败: {str(e)}")


@router.get("/network-graph")
def get_network_graph(
    topic_id: str = Query(..., description="专题ID"),
    jurisdiction_a: str = Query(..., description="法域A"),
    jurisdiction_b: str = Query(..., description="法域B"),
    center_law_id: str = Query(None, description="中心法规ID（可选，以此为中心展开子图）"),
    top_n: int = Query(10, description="每法域检索条款数"),
    min_similarity: float = Query(0.6, ge=0.0, le=1.0, description="跨法域语义对齐的最低相似度"),
    cross_top_k: int = Query(2, ge=1, le=5, description="每个条款最多保留的跨域对齐数")
):
    """关联网络图：节点=法条，边=同法规关联/跨法域语义对齐（bge-m3 向量 + 每节点 Top-K 稀疏化）。"""
    try:
        queries = service.get_topic_queries(topic_id)
        if not queries:
            return APIResponse(success=False, message=f"未找到专题: {topic_id}")

        chunks_a = service._search_topic(queries, jurisdiction_a, top_n)
        chunks_b = service._search_topic(queries, jurisdiction_b, top_n)

        graph = service.get_network_graph(
            chunks_a, chunks_b,
            center_law_id=center_law_id or None,
            similarity_threshold=min_similarity,
            cross_top_k=cross_top_k,
        )
        return APIResponse(success=True, data=graph)
    except Exception as e:
        return APIResponse(success=False, message=f"获取网络图失败: {str(e)}")


@router.get("/alignment-matrix")
def get_alignment_matrix(
    topic_id: str = Query(..., description="专题ID"),
    jurisdiction_a: str = Query(..., description="法域A"),
    jurisdiction_b: str = Query(..., description="法域B"),
    top_n: int = Query(10, description="每法域检索条款数"),
    min_similarity: float = Query(0.6, ge=0.0, le=1.0, description="语义对齐的最低相似度"),
):
    """法规级对齐矩阵：行=法域A法规、列=法域B法规，格值=语义对口的条款对数量。"""
    try:
        data = service.get_alignment_matrix(
            topic_id, jurisdiction_a, jurisdiction_b,
            top_n=top_n, min_similarity=min_similarity,
        )
        if data is None:
            return APIResponse(success=False, message="未找到可对齐的法规数据")
        return APIResponse(success=True, data=data)
    except Exception as e:
        return APIResponse(success=False, message=f"生成对齐矩阵失败: {str(e)}")


@router.get("/alignment-detail")
def get_alignment_detail(
    topic_id: str = Query(..., description="专题ID"),
    jurisdiction_a: str = Query(..., description="法域A"),
    jurisdiction_b: str = Query(..., description="法域B"),
    law_a: str = Query(..., description="法域A的法规ID"),
    law_b: str = Query(..., description="法域B的法规ID"),
    top_n: int = Query(10, description="每法域检索条款数"),
    min_similarity: float = Query(0.6, ge=0.0, le=1.0, description="语义对齐的最低相似度"),
):
    """下钻：两部法规之间语义对口的条款对明细（按相似度降序）。"""
    try:
        data = service.get_alignment_detail(
            topic_id, jurisdiction_a, jurisdiction_b, law_a, law_b,
            top_n=top_n, min_similarity=min_similarity,
        )
        if data is None:
            return APIResponse(success=False, message="未找到该法规对的对齐明细")
        return APIResponse(success=True, data=data)
    except Exception as e:
        return APIResponse(success=False, message=f"获取对齐明细失败: {str(e)}")


@router.get("/cache-status")
def get_cache_status(
    cache_type: str = Query("analysis", description="缓存类型: analysis 或 compare"),
):
    """查询当前缓存索引中的所有记录（调试用）"""
    try:
        cached = service.repo.get_analysis_cache("__all__", "__", "__", cache_type=cache_type)
        if isinstance(cached, list):
            return APIResponse(success=True, data={
                "total": len(cached),
                "records": [
                    {
                        "doc_id": r.get("_id", ""),
                        "cache_type": r.get("cache_type", ""),
                        "topic_id": r.get("topic_id", ""),
                        "jurisdictions": f"{r.get('jurisdiction_a', '')} vs {r.get('jurisdiction_b', '')}",
                        "fingerprint": r.get("fingerprint", "")[:16] + "...",
                        "updated_at": r.get("updated_at", ""),
                    }
                    for r in cached
                ]
            })
        return APIResponse(success=True, data={"total": 0, "records": []})
    except Exception as e:
        return APIResponse(success=False, message=f"查询失败: {str(e)}")


# ============================================================
#  最短路径分析接口
# ============================================================

@router.post("/shortest-path")
def find_shortest_path(
    request_data: dict = Body(..., description="最短路径查询参数"),
):
    """在知识图谱中 BFS 查找两个法条节点的最短关联路径（可选同时生成 AI 解读）。"""
    try:
        nodes = request_data.get("nodes", [])
        edges = request_data.get("edges", [])
        source_id = request_data.get("source_id")
        target_id = request_data.get("target_id")
        max_depth = request_data.get("max_depth", 6)
        do_interpret = request_data.get("interpret", False)

        if not source_id or not target_id:
            return APIResponse(
                success=False,
                message="缺少必要参数：source_id 和 target_id 均为必填"
            )

        result = service.find_shortest_path(
            nodes=nodes,
            edges=edges,
            source_id=source_id,
            target_id=target_id,
            max_depth=max_depth
        )

        if do_interpret and result.get("found"):
            interpretation = service.interpret_path(result)
            result["interpretation"] = interpretation

        return APIResponse(success=True, data=result)

    except Exception as e:
        import traceback
        logger_instance = __import__("logging").getLogger(__name__)
        logger_instance.error(f"[Shortest-Path] 异常: {traceback.format_exc()}")
        return APIResponse(success=False, message=f"最短路径搜索失败: {str(e)}")
