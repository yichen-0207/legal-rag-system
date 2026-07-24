from typing import Dict, Optional
from repositories.elasticsearch import ElasticsearchRepository

class SummaryCache:
    """
    法规摘要缓存服务（基于 Elasticsearch）
    
    功能：
    1. 缓存法规摘要，减少重复生成
    2. 使用指纹检测法规数据变化，数据不变时直接返回缓存
    3. 无过期时间，仅在数据变化时失效
    4. 支持缓存清除
    """
    
    def __init__(self):
        self._repo = ElasticsearchRepository()
    
    def get(self, law_id: str) -> Optional[Dict]:
        """
        从缓存获取摘要
        
        返回：
        - 如果缓存存在且法规数据未变化，返回摘要
        - 如果缓存不存在或法规数据已变化，返回 None
        """
        return self._repo.get_summary_cache(law_id)
    
    def set(self, law_id: str, summary: Dict):
        """
        将摘要存入缓存
        
        Args:
            law_id: 法规ID
            summary: 摘要数据
        """
        self._repo.save_summary_cache(law_id, summary)
    
    def invalidate(self, law_id: str):
        """
        清除指定法规的缓存
        
        Args:
            law_id: 法规ID
        """
        self._repo.delete_summary_cache(law_id)
    
    def invalidate_all(self):
        """清除所有摘要缓存"""
        self._repo.delete_all_summary_cache()
    
    def get_cache_info(self) -> Dict:
        """获取缓存统计信息"""
        try:
            cached = self._repo.get_analysis_cache("__all__", "", "", cache_type="summary")
            total_entries = len(cached) if isinstance(cached, list) else 0
            return {
                "total_entries": total_entries,
                "valid_entries": total_entries,
                "expired_entries": 0,
                "expire_seconds": None,
                "cache_type": "elasticsearch"
            }
        except Exception:
            return {
                "total_entries": 0,
                "valid_entries": 0,
                "expired_entries": 0,
                "expire_seconds": None,
                "cache_type": "elasticsearch"
            }
    
    def get_cached_law_ids(self) -> list:
        """获取所有已缓存的法规ID列表"""
        try:
            cached = self._repo.get_analysis_cache("__all__", "", "", cache_type="summary")
            if isinstance(cached, list):
                return [item.get("law_id", "") for item in cached if item.get("law_id")]
        except Exception:
            pass
        return []

summary_cache = SummaryCache()