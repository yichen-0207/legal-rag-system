from fastapi import APIRouter, Query, Body
from fastapi.responses import StreamingResponse
from models.schemas import APIResponse
from services.structured_analysis_service import StructuredAnalysisService
import json, asyncio

router = APIRouter(prefix="/structured-analysis", tags=["结构化专题分析"])
service = StructuredAnalysisService()


@router.get("/topics")
async def list_topics():
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
    """
    流式执行跨法域结构化专题分析（SSE）
    新增事件类型：dashboard_done（仪表盘数据）、network_done（网络图数据）
    """
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
async def get_dashboard(
    analysis_result: dict = Body(..., description="已有的结构化分析结果")
):
    """
    基于已有分析结果，生成仪表盘数据：
    - 保护力度评分（1-5分）
    - 雷达图各维度得分
    - 关键差异点列表
    """
    try:
        dashboard = service.generate_dashboard(analysis_result)
        return APIResponse(success=True, data=dashboard)
    except Exception as e:
        return APIResponse(success=False, message=f"生成仪表盘失败: {str(e)}")


@router.post("/ai-interpret")
async def ai_interpret(
    dimension_label: str = Query(..., description="差异维度名称"),
    detail_a: str = Query("", description="法域A的具体规定"),
    detail_b: str = Query("", description="法域B的具体规定"),
    jurisdiction_a: str = Query(..., description="法域A名称"),
    jurisdiction_b: str = Query(..., description="法域B名称"),
    topic_name: str = Query(..., description="专题名称")
):
    """
    AI 深度解读某个具体差异维度。
    用通俗语言解释实务影响、合规策略建议。
    """
    try:
        interpretation = service.ai_interpret(
            dimension_label=dimension_label,
            detail_a=detail_a,
            detail_b=detail_b,
            jurisdiction_a=jurisdiction_a,
            jurisdiction_b=jurisdiction_b,
            topic_name=topic_name
        )
        return APIResponse(success=True, data={"interpretation": interpretation})
    except Exception as e:
        return APIResponse(success=False, message=f"AI解读失败: {str(e)}")


@router.get("/network-graph")
async def get_network_graph(
    topic_id: str = Query(..., description="专题ID"),
    jurisdiction_a: str = Query(..., description="法域A"),
    jurisdiction_b: str = Query(..., description="法域B"),
    center_law_id: str = Query(None, description="中心法规ID（可选，以此为中心展开子图）"),
    top_n: int = Query(10, description="每法域检索条款数")
):
    """
    获取关联网络图数据。
    节点=法条，边=同法规关联/跨法域相似关系。
    可指定 center_law_id 以某法条为中心展开子图。
    """
    try:
        queries = service.get_topic_queries(topic_id)
        if not queries:
            return APIResponse(success=False, message=f"未找到专题: {topic_id}")

        chunks_a = service._search_topic(queries, jurisdiction_a, top_n)
        chunks_b = service._search_topic(queries, jurisdiction_b, top_n)

        graph = service.get_network_graph(
            chunks_a, chunks_b,
            center_law_id=center_law_id or None
        )
        return APIResponse(success=True, data=graph)
    except Exception as e:
        return APIResponse(success=False, message=f"获取网络图失败: {str(e)}")


@router.get("/cache-status")
async def get_cache_status(
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
async def find_shortest_path(
    request_data: dict = Body(..., description="最短路径查询参数"),
):
    """
    最短路径查找接口

    在知识图谱中执行BFS搜索，找出两个法条节点之间的最短关联路径。

    请求体：
    {
        "nodes": [...],           // 图谱节点列表（来自network_graph数据）
        "edges": [...],           // 图谱边列表
        "source_id": "node_0",    // 起始节点ID
        "target_id": "node_5",    // 目标节点ID
        "max_depth": 6,          // 最大搜索深度（可选，默认6）
        "interpret": false       // 是否同时生成AI解读（可选，默认false）
    }

    返回：
    {
        "found": true,
        "path_nodes": [...],      // 路径上的节点
        "path_edges": [...],      // 路径上的边（含关系详情）
        "hops": 3,                // 跳数
        "path_description": "...", // 文字描述
        "interpretation": "..."   // AI解读（仅当interpret=true时）
    }
    """
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

        # 执行BFS最短路径搜索
        result = service.find_shortest_path(
            nodes=nodes,
            edges=edges,
            source_id=source_id,
            target_id=target_id,
            max_depth=max_depth
        )

        # 可选：生成AI解读
        if do_interpret and result.get("found"):
            interpretation = service.interpret_path(result)
            result["interpretation"] = interpretation

        return APIResponse(success=True, data=result)

    except Exception as e:
        import traceback
        logger_instance = __import__("logging").getLogger(__name__)
        logger_instance.error(f"[Shortest-Path] 异常: {traceback.format_exc()}")
        return APIResponse(success=False, message=f"最短路径搜索失败: {str(e)}")
