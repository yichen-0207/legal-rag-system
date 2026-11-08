import threading
from typing import Dict, Optional
from repositories.elasticsearch import ElasticsearchRepository

class SummaryCache:
    """法规摘要缓存（基于 Elasticsearch）：按 law_id 缓存，无过期时间，仅法规数据指纹变化时失效。"""
    
    def __init__(self):
        # 只保存状态，不建连：本模块在导入期就会被实例化（summary_cache 是模块级单例），
        # 而 compose start 时 ES 可能尚未就绪；建连推迟到首次真正读写缓存时。
        self._repo = None
        self._repo_lock = threading.Lock()

    @property
    def _repository(self) -> ElasticsearchRepository:
        """惰性构造仓储（构造本身已不建连，索引创建同样推迟到首次使用）。"""
        if self._repo is None:
            with self._repo_lock:
                if self._repo is None:
                    self._repo = ElasticsearchRepository()
        return self._repo
    
    def get(self, law_id: str) -> Optional[Dict]:
        """读取缓存；缓存不存在或法规数据已变化时返回 None。"""
        return self._repository.get_summary_cache(law_id)
    
    def set(self, law_id: str, summary: Dict):
        """写入摘要缓存。"""
        self._repository.save_summary_cache(law_id, summary)
    
    def invalidate(self, law_id: str):
        """清除指定法规的缓存。"""
        self._repository.delete_summary_cache(law_id)
    
    def invalidate_all(self):
        """清除所有摘要缓存"""
        self._repository.delete_all_summary_cache()
    
    def get_cache_info(self) -> Dict:
        """获取缓存统计信息"""
        try:
            cached = self._repository.get_analysis_cache("__all__", "", "", cache_type="summary")
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
            cached = self._repository.get_analysis_cache("__all__", "", "", cache_type="summary")
            if isinstance(cached, list):
                return [item.get("law_id", "") for item in cached if item.get("law_id")]
        except Exception:
            pass
        return []

summary_cache = SummaryCache()