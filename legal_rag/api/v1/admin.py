from fastapi import APIRouter, HTTPException, Depends, Query, Header
from typing import Dict, Optional, Any, List
from core.config import settings
from models.schemas import APIResponse
from services.law_management_service import LawManagementService
from repositories.elasticsearch import ElasticsearchRepository
from utils.law_catalog import LawCatalog

router = APIRouter(prefix="/admin", tags=["管理员"])

law_service = LawManagementService()
repo = ElasticsearchRepository()
catalog = LawCatalog()

# 管理员Token，从环境变量 LEGAL_ADMIN_TOKEN 读取（见 core/config.py 的 admin_token）
# 默认值仅供本地开发使用，生产环境务必在 .env 中覆盖
ADMIN_TOKEN = settings.admin_token


def verify_admin_token(authorization: Optional[str] = Header(None)) -> bool:
    """验证管理员Token"""
    if not authorization:
        raise HTTPException(status_code=401, detail="缺少认证令牌", headers={"WWW-Authenticate": "Bearer"})
    
    # 支持Bearer token格式
    if authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    else:
        token = authorization.strip()
    
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="无效的认证令牌")
    
    return True


@router.post("/law", summary="新增法规")
async def create_law(
    data: Dict[str, Any],
    _: bool = Depends(verify_admin_token)
):
    try:
        result = law_service.create_law(data)
        return APIResponse(success=True, data=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建法规失败: {str(e)}")


@router.post("/laws/batch", summary="批量导入法规")
async def batch_import_laws(
    files_data: List[Dict[str, Any]],
    _: bool = Depends(verify_admin_token)
):
    """批量导入法规，支持创建新法规和更新已有法规"""
    try:
        success_count = 0
        failed_count = 0
        failed_list = []
        
        for file_item in files_data:
            filename = file_item.get("filename", "unknown")
            data = file_item.get("data", {})
            
            try:
                law_id = data.get("law_id")
                
                if law_id:
                    existing_data = law_service.get_law_raw_data(law_id)
                    if existing_data:
                        result = law_service.update_law(law_id, data, overwrite=True)
                    else:
                        result = law_service.create_law(data)
                else:
                    result = law_service.create_law(data)
                
                success_count += 1
            except Exception as e:
                failed_count += 1
                failed_list.append({
                    "filename": filename,
                    "error": str(e)
                })
        
        return APIResponse(success=True, data={
            "success_count": success_count,
            "failed_count": failed_count,
            "failed": failed_list
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量导入失败: {str(e)}")


@router.delete("/law/{law_id}", summary="删除法规")
async def delete_law(
    law_id: str,
    _: bool = Depends(verify_admin_token)
):
    try:
        result = law_service.delete_law(law_id)
        return APIResponse(success=True, data=result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除法规失败: {str(e)}")


@router.put("/law/{law_id}", summary="修改法规")
async def update_law(
    law_id: str,
    updates: Dict[str, Any],
    overwrite: bool = Query(False, description="是否完全覆盖原有数据"),
    _: bool = Depends(verify_admin_token)
):
    """修改法规，支持合并更新和完全覆盖两种模式"""
    try:
        result = law_service.update_law(law_id, updates, overwrite=overwrite)
        return APIResponse(success=True, data=result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新法规失败: {str(e)}")


@router.get("/law/{law_id}", summary="获取法规原始数据")
async def get_law_raw(
    law_id: str,
    _: bool = Depends(verify_admin_token)
):
    try:
        data = law_service.get_law_raw_data(law_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"法规不存在: {law_id}")
        return APIResponse(success=True, data=data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取法规数据失败: {str(e)}")


@router.get("/laws", summary="获取所有法规列表")
async def get_all_laws(
    jurisdiction: Optional[str] = None,
    _: bool = Depends(verify_admin_token)
):
    try:
        laws = repo.get_all_laws(jurisdiction)
        return APIResponse(success=True, data=laws)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取法规列表失败: {str(e)}")


@router.post("/sync", summary="同步索引与文件系统")
async def sync_index(
    _: bool = Depends(verify_admin_token)
):
    try:
        result = law_service.sync_index()
        return APIResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"同步失败: {str(e)}")


# ==================== 法律目录 (Law Catalog) API ====================

@router.get("/catalog", summary="获取法律目录")
async def get_catalog(
    jurisdiction: Optional[str] = Query(None, description="法域筛选"),
    status: Optional[str] = Query(None, description="状态筛选 (active/amended/repealed/deleted)"),
    _: bool = Depends(verify_admin_token)
):
    """
    获取法律目录列表
    
    支持按法域和状态筛选
    返回所有法律的元信息（不含全文）
    """
    try:
        laws = catalog.get_all_laws(jurisdiction=jurisdiction, status=status)
        stats = catalog.get_statistics()
        return APIResponse(success=True, data={
            "laws": laws,
            "statistics": stats,
            "catalog_path": str(catalog.catalog_path),
            "last_updated": catalog.load().get("last_updated")
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取法律目录失败: {str(e)}")


@router.get("/catalog/statistics", summary="获取法律目录统计信息")
async def get_catalog_statistics(
    _: bool = Depends(verify_admin_token)
):
    """
    获取详细的统计信息
    
    包括：总法规数、按法域分布、按年份分布、按状态分布等
    """
    try:
        stats = catalog.get_statistics()
        return APIResponse(success=True, data=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.get("/catalog/{law_id}", summary="获取单个法律目录记录")
async def get_catalog_record(
    law_id: str,
    _: bool = Depends(verify_admin_token)
):
    """根据 law_id 获取单个法律的目录记录"""
    try:
        record = catalog.find_by_law_id(law_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"未找到法律 {law_id} 的目录记录")
        return APIResponse(success=True, data=record)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取目录记录失败: {str(e)}")


@router.delete("/catalog/{law_id}", summary="删除/标记删除法律目录记录")
async def delete_catalog_record(
    law_id: str,
    _: bool = Depends(verify_admin_token)
):
    """
    删除或标记删除法律目录记录
    
    注意：此操作仅标记为 deleted，不会物理删除（便于审计）
    如需完全清除，请使用 /admin/catalog/rebuild 从 ES 重建
    """
    try:
        success = catalog.delete(law_id)
        if success:
            return APIResponse(success=True, message=f"已标记删除 {law_id}")
        else:
            raise HTTPException(status_code=404, detail=f"未找到法律 {law_id} 的目录记录")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除目录记录失败: {str(e)}")


@router.post("/catalog/rebuild", summary="从 ES 重建法律目录")
async def rebuild_catalog(
    _: bool = Depends(verify_admin_token)
):
    """
    从 Elasticsearch 重建 law_catalog.json
    
    使用场景：
    - catalog 文件丢失或损坏
    - 需要修复数据不一致问题
    - 首次初始化时从现有数据生成
    
    此操作会覆盖现有的 catalog 数据！
    """
    try:
        stats = catalog.rebuild_from_es(repo)
        return APIResponse(success=True, data={
            "message": "Catalog 已从 ES 重建完成",
            "statistics": stats
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重建 catalog 失败: {str(e)}")


@router.put("/catalog/{law_id}", summary="手动更新法律目录记录")
async def update_catalog_record(
    law_id: str,
    updates: Dict[str, Any],
    _: bool = Depends(verify_admin_token)
):
    """
    手动更新指定法律的目录记录
    
    可更新的字段：
    - law_name_zh, law_name_original
    - law_type, passing_date, effective_date
    - topics (list), status
    - total_articles, total_chunks
    """
    try:
        existing = catalog.find_by_law_id(law_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"未找到法律 {law_id}")

        # 合并更新字段
        updated_record = catalog.add_or_update(
            law_id=law_id,
            law_name_zh=updates.get("law_name_zh", existing.get("law_name_zh", "")),
            jurisdiction=existing.get("jurisdiction", ""),
            total_articles=updates.get("total_articles", existing.get("total_articles", 0)),
            total_chunks=updates.get("total_chunks", existing.get("total_chunks", 0)),
            law_name_original=updates.get("law_name_original", existing.get("law_name_original", "")),
            law_type=updates.get("law_type", existing.get("law_type", "")),
            passing_date=updates.get("passing_date", existing.get("passing_date", "")),
            effective_date=updates.get("effective_date", existing.get("effective_date", "")),
            topics=updates.get("topics", existing.get("topics", [])),
            source_file=existing.get("source_file", ""),
            status=updates.get("status", existing.get("status", "active"))
        )

        return APIResponse(success=True, data=updated_record)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新目录记录失败: {str(e)}")