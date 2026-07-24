from fastapi import APIRouter, Query, Body
from models.schemas import APIResponse, TopicAnalysisResponse
from services.topic_analysis_service import TopicAnalysisService

router = APIRouter(prefix="/topic-analysis", tags=["跨法域专题分析"])
analysis_service = TopicAnalysisService()


@router.post("")
async def run_topic_analysis(
    query: str = Query(..., description="分析主题（如：个人数据跨境转移的规定）"),
    jurisdiction_a: str = Query(..., description="法域A（如：澳门）"),
    jurisdiction_b: str = Query(..., description="法域B（如：新加坡）"),
    top_n: int = Query(8, ge=3, le=20, description="每个法域检索条款数（3-20）")
):
    """
    跨法域专题分析（完整流程，一步到位）

    三阶段流水线：
    1. 语义向量检索：分别对两个法域执行 BGE-M3 向量搜索，各取 Top-N 条款
       （分开搜索确保每个法域都有足够素材，避免被另一个法域"淹没"）
    2. 结果聚合：按 law_id 分组去重 → chunk_index 排序 → 拼接为法规级结构化文本
    3. LLM 对比报告：基于聚合后的两个法域条款，从5个固定维度生成结构化对比分析：
       - 适用范围、核心权利/义务、例外情形、法律责任/处罚、总体差异总结表格
    """
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
async def follow_up(
    original_analysis: dict = Body(..., description="原始分析结果"),
    follow_up_question: str = Query(..., description="用户追问问题")
):
    """
    专题分析追问接口：基于已生成的分析报告，回答用户的后续问题。
    
    核心对比报告已经生成后，用户可以在此基础上继续追问，获得补充信息。
    """
    try:
        result = analysis_service.follow_up(
            original_analysis=original_analysis,
            follow_up_question=follow_up_question
        )
        return APIResponse(success=True, data=result)
    except Exception as e:
        return APIResponse(success=False, message=f"追问失败: {str(e)}")
