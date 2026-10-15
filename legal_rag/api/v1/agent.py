from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from typing import Optional
import json

from services.agent.agent_service import AgentService
from api.v1.qa import _parse_history

router = APIRouter(prefix="/agent", tags=["Agent"])

# 模块级单例：AgentService 构造仅持有轻量对象（RetrieverService/QAService 均为单例或轻量），
# 与 api/v1/qa.py 的做法保持一致。
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

    开关 LEGAL_AGENT_ENABLED 关闭时，本接口自动降级为既有 RAG 问答，端点始终可用。
    mode=auto（默认）时按复杂度路由：仅复杂查询走多步探索，简单/定向查询直接走既有
    快速 RAG（与 /qa/stream 同路径），避免用简单查询的延迟换复杂查询的收益。

    事件类型：route / status / step / observation / sources / content / end；其中
    route 给出本次实际走的分支与原因，end.trace 给出完整工具调用轨迹，便于前端展示
    「这次为什么走这条路」与「推理过程」。
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
