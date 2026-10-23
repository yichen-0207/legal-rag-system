from fastapi import APIRouter, Query, Body
from models.schemas import APIResponse, TopicAnalysisResponse
from services.topic_analysis_service import TopicAnalysisService

# 同步 def：内部为双法域检索 + LLM 报告的阻塞流程，交由线程池执行，避免独占事件循环
router = APIRouter(prefix="/topic-analysis", tags=["跨法域专题分析"])
analysis_service = TopicAnalysisService()


@router.post("")
def run_topic_analysis(
    query: str = Query(..., description="分析主题（如：个人数据跨境转移的规定）"),
    jurisdiction_a: str = Query(..., description="法域A（如：澳门）"),
    jurisdiction_b: str = Query(..., description="法域B（如：新加坡）"),
    top_n: int = Query(8, ge=3, le=20, description="每个法域检索条款数（3-20）")
):
    """跨法域专题分析：双法域向量检索 → 按法规聚合 → LLM 五维对比报告。"""
    try:
        result = analysis_service.analyze(
            query=query,
            jurisdiction_a=jurisdiction_a,
            jurisdiction_b=jurisdiction_b,
            top_n_per_jurisdiction=top_n
        )
        return APIResponse(success=True, data=result)
    except Exception as e:
        return APIResponse(success=False, message=f"专题分析失败: {str(e)}")


@router.post("/follow-up")
def follow_up(
    request_data: dict = Body(..., description="追问参数：original_analysis（原始分析结果）+ follow_up_question（追问问题）")
):
    """专题分析追问：基于已生成报告回答后续问题（参数走请求体，避免 414）。"""
    try:
        result = analysis_service.follow_up(
            original_analysis=request_data.get("original_analysis", {}),
            follow_up_question=request_data.get("follow_up_question", "")
        )
        return APIResponse(success=True, data=result)
    except Exception as e:
        return APIResponse(success=False, message=f"追问失败: {str(e)}")
