from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from fastapi import Request
from typing import Optional
from models.schemas import APIResponse
from services.qa_service import QAService
import json

router = APIRouter(prefix="/qa", tags=["问答"])
qa_service = QAService()


def _parse_history(history_str: Optional[str]) -> Optional[list]:
    """解析对话历史JSON字符串"""
    if not history_str or not history_str.strip():
        return None
    try:
        parsed = json.loads(history_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return None


@router.get("/stream")
async def ask_stream(
    question: str = Query(..., description="用户问题"),
    top_k: int = Query(5, description="检索结果数量"),
    jurisdictions: str = Query(None, description="法域筛选，多个用逗号分隔"),
    history: str = Query(None, description="对话历史，JSON数组格式：[{role,content}]")
):
    """流式问答接口（支持追问/多轮对话）"""
    jurisdiction_list = None
    if jurisdictions:
        jurisdiction_list = [j.strip() for j in jurisdictions.split(",")]

    history_list = _parse_history(history)

    def generate():
        for chunk in qa_service.ask_stream(question, top_k, jurisdiction_list, history_list):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/stream")
async def ask_stream_multimodal(request: Request):
    """
    支持图片的多模态流式问答（POST接口）

    请求体格式：
    {
        "question": "用户问题（必填）",
        "top_k": 5,
        "jurisdictions": ["澳门", "香港"],
        "history": [{"role": "user", "content": "..."}],
        "images": [
            {
                "data": "base64编码的图片数据",
                "mime_type": "image/jpeg"
            }
        ]
    }
    """
    try:
        data = await request.json()
    except Exception:
        return APIResponse(success=False, message="无效的JSON数据")

    question = data.get("question", "")
    if not question or not question.strip():
        return APIResponse(success=False, message="问题不能为空")

    top_k = data.get("top_k", 5)
    jurisdictions = data.get("jurisdictions")
    history = data.get("history")
    images = data.get("images", [])  # 新增：图片列表

    jurisdiction_list = None
    if jurisdictions:
        if isinstance(jurisdictions, list):
            jurisdiction_list = [j.strip() for j in jurisdictions if j]
        elif isinstance(jurisdictions, str):
            jurisdiction_list = [j.strip() for j in jurisdictions.split(",") if j.strip()]

    history_list = None
    if history and isinstance(history, list) and len(history) > 0:
        history_list = history

    def generate():
        for chunk in qa_service.ask_stream_with_image(
            question=question,
            top_k=top_k,
            jurisdictions=jurisdiction_list,
            history=history_list,
            images=images
        ):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
