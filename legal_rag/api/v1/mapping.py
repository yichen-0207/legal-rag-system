from fastapi import APIRouter, Query
from models.schemas import APIResponse
from services.mapping_service import LawMappingService

router = APIRouter(prefix="/mapping", tags=["法规条款映射"])
mapping_service = LawMappingService()


@router.get("/laws")
async def list_laws(
    jurisdiction: str = Query(..., description="法域名称（如：澳门、新加坡）")
):
    """获取指定法域下的所有法规列表（用于前端下拉选择源法律）"""
    try:
        laws = mapping_service.get_laws_by_jurisdiction(jurisdiction)
        return APIResponse(success=True, data=laws)
    except Exception as e:
        return APIResponse(success=False, message=f"获取法规列表失败: {str(e)}")


@router.post("")
async def map_laws(
    source_jurisdiction: str = Query(..., description="源法域"),
    source_law_id: str = Query(..., description="源法规ID"),
    target_jurisdiction: str = Query(..., description="目标法域"),
    threshold: float = Query(0.75, ge=0.5, le=1.0, description="相似度匹配阈值（默认0.75）")
):
    """
    法规条款映射（纯向量KNN，不依赖LLM）

    对源法律的每个条款，在目标法域中找到最相似的对应条款，返回映射对照表。
    """
    try:
        result = mapping_service.map_laws(
            source_jurisdiction=source_jurisdiction,
            source_law_id=source_law_id,
            target_jurisdiction=target_jurisdiction,
            similarity_threshold=threshold
        )
        if "error" in result:
            return APIResponse(success=False, message=result["error"])
        return APIResponse(success=True, data=result)
    except Exception as e:
        return APIResponse(success=False, message=f"映射失败: {str(e)}")
