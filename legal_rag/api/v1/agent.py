from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from typing import Optional
import json

from services.agent.agent_service import AgentService
from api.v1.qa import _parse_history

router = APIRouter(prefix="/agent", tags=["Agent"])

# 模块级单例（与 api/v1/qa.py 一致）
agent_service = AgentService()


@router.get("/stream")
async def agent_stream(
    question: str = Query(..., description="用户问题"),
    top_k: int = Query(5, description="最终证据条数"),
    jurisdictions: str = Query(None, description="法域筛选，多个用逗号分隔"),
    history: str = Query(None, description="对话历史，JSON数组格式：[{role,content}]"),
    mode: str = Query("auto", description="路由模式：auto=按复杂度自动；agent=强制多步探索；fast=强制快速 RAG"),
):
    """迭代式工具调用问答（SSE）。

    LEGAL_AGENT_ENABLED 关闭时自动降级为既有 RAG；mode=auto 按复杂度路由，
    简单查询直接走快速 RAG。事件：route / status / step / observation / sources / content / end。
    """
    jurisdiction_list = None
    if jurisdictions:
        jurisdiction_list = [j.strip() for j in jurisdictions.split(",") if j.strip()]

    history_list = _parse_history(history)

    def generate():
        for chunk in agent_service.ask_stream(
            question, top_k, jurisdiction_list, history_list, route_mode=mode
        ):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/stream")
async def agent_stream_post(request: Request):
    """迭代式工具调用问答（SSE，POST 版本），历史放入 body 避免 URL 过长。"""
    try:
        data = await request.json()
    except Exception:
        data = {}

    question = str(data.get("question") or "").strip()
    top_k = int(data.get("top_k") or 5)

    jurisdictions = data.get("jurisdictions")
    jurisdiction_list = None
    if isinstance(jurisdictions, list):
        jurisdiction_list = [str(j).strip() for j in jurisdictions if str(j).strip()]
    elif isinstance(jurisdictions, str) and jurisdictions:
        jurisdiction_list = [j.strip() for j in jurisdictions.split(",") if j.strip()]

    history = data.get("history")
    if isinstance(history, list):
        history_list = history
    elif isinstance(history, str):
        history_list = _parse_history(history)
    else:
        history_list = None

    mode = data.get("mode") or "auto"

    def generate():
        if not question:
            yield f"data: {json.dumps({'type': 'error', 'data': '问题不能为空'}, ensure_ascii=False)}\n\n"
            return
        for chunk in agent_service.ask_stream(
            question, top_k, jurisdiction_list, history_list, route_mode=mode
        ):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
