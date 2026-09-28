from fastapi import APIRouter, Query
from pydantic import BaseModel
from models.schemas import APIResponse
from services.compare_service import CompareService
from core.config import settings

router = APIRouter(prefix="/compare", tags=["法规对比"])
compare_service = CompareService()


class DeepAnalysisRequest(BaseModel):
    """AI深度分析请求体"""
    jurisdiction_a: str
    jurisdiction_b: str
    topic: str
    context_a: str  # 法域A的法规信息摘要
    context_b: str  # 法域B的法规信息摘要


@router.get("")
def compare_laws(
    jurisdiction_a: str = Query(..., description="法域A（如：澳门）"),
    jurisdiction_b: str = Query(..., description="法域B（如：新加坡）"),
    topic: str = Query(..., description="对比主题/关键词（如：数据保护）"),
    top_k: int = Query(5, ge=1, le=20, description="每个法域检索条款数（1-20）")
):
    """
    法规对比（轻量版，不调用LLM，秒级响应）

    分别检索两个法域的相关条款 → 按法规聚合去重 → 结构化表格展示 + 自动总结。
    如需 LLM 深度分析报告，请使用「跨法域专题分析」功能或点击「AI深度对比分析」按钮。
    """
    try:
        result = compare_service.compare(
            jurisdiction_a=jurisdiction_a,
            jurisdiction_b=jurisdiction_b,
            topic=topic,
            top_k_per_jurisdiction=top_k
        )
        return APIResponse(success=True, data=result)
    except Exception as e:
        return APIResponse(success=False, message=f"对比失败: {str(e)}")


@router.post("/deep-analysis")
def deep_analysis(req: DeepAnalysisRequest):
    """
    AI深度对比分析（调用LLM，流式输出）

    基于两个法域的检索结果，由LLM生成专业的跨法域深度对比报告，流式输出。
    """
    try:
        from openai import OpenAI
        from fastapi.responses import StreamingResponse
        import json

        prompt = f"""你是跨境法律合规专家。对比{req.jurisdiction_a}和{req.jurisdiction_b}在「{req.topic}」主题下的法规差异。

## {req.jurisdiction_a}
{req.context_a[:2000]}

## {req.jurisdiction_b}
{req.context_b[:2000]}

请严格按以下格式输出（Markdown），不要输出其他内容：

### 差异总览
一段话概括核心差异。

### 逐维度对比表
| 维度 | {req.jurisdiction_a} | {req.jurisdiction_b} |
|------|------|------|
| 立法目的 | | |
| 适用范围 | | |
| 核心义务 | | |
| 监管机构 | | |
| 处罚力度 | | |

### 关键差异要点
1. **差异点1**：描述 → 实务影响
2. **差异点2**：描述 → 实务影响
3. **差异点3**：描述 → 实务影响

### 合规建议
3条具体建议，针对跨法域运营主体。
"""

        base_url = settings.llm_api_base_url or "https://api.deepseek.com/v1"
        api_key = settings.llm_api_key or ""
        model = settings.llm_comparison_model or "deepseek-flash"

        client = OpenAI(api_key=api_key, base_url=base_url)
        create_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            # 用全局 max_tokens：原先硬编码 2500 会被思考 token 吃满，导致 content 为空（报告空白）
            "max_tokens": settings.max_tokens,
            "stream": True
        }
        extra_body = settings.llm_extra_body()
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        stream = client.chat.completions.create(**create_kwargs)

        def generate():
            for chunk in stream:
                if not chunk.choices or len(chunk.choices) == 0:
                    continue
                delta = chunk.choices[0].delta
                if not delta:
                    continue
                content = delta.content or ""
                if content:
                    yield f"data: {json.dumps({'type': 'chunk', 'data': content})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        import json
        from fastapi.responses import StreamingResponse
        def generate_error():
            yield f"data: {json.dumps({'type': 'error', 'data': f'深度分析失败: {str(e)}'})}\n\n"
        return StreamingResponse(generate_error(), media_type="text/event-stream")


class CacheAIRequest(BaseModel):
    """AI分析结果回写缓存请求"""
    topic: str
    jurisdiction_a: str
    jurisdiction_b: str
    ai_analysis: str


@router.post("/cache-ai")
def cache_ai_analysis(req: CacheAIRequest):
    """
    将前端生成的 AI 深度分析报告回写到 ES 缓存中。
    前端在首次生成 AI 报告后异步调用此接口，后续缓存命中时直接返回。
    """
    try:
        # 查询现有缓存
        cached = compare_service.repo.get_analysis_cache(
            req.topic, req.jurisdiction_a, req.jurisdiction_b, cache_type="compare"
        )
        if not cached or not isinstance(cached, dict):
            return APIResponse(success=False, message="缓存不存在，无法回写")

        # 在现有 report_data 中追加 ai_analysis 字段
        report_data = cached.get("report_data", {})
        if not isinstance(report_data, dict):
            report_data = {}
        report_data["ai_analysis"] = req.ai_analysis

        # 获取当前指纹重新写入
        fingerprint = cached.get("fingerprint", "")
        compare_service.repo.save_analysis_cache(
            req.topic, req.jurisdiction_a, req.jurisdiction_b,
            fingerprint, report_data, cache_type="compare"
        )
        return APIResponse(success=True, message="AI 报告已写入缓存")

    except Exception as e:
        return APIResponse(success=False, message=f"缓存回写失败: {str(e)}")
