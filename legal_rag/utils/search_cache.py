# -*- coding: utf-8 -*-
"""检索结果缓存 / 问答答案缓存。

按「归一化查询 + 筛选条件」精确命中，命中时跳过整条检索（精排占单次耗时约 93%）。
语义命中已实测否决：良性改写与危险近似的相似度区间重叠（0.94 vs 0.93），
阈值无法同时接住改写、拦住法律含义不同的表述，法条场景错答代价更高。
失效：law_catalog.json mtime 变化即整体清空；路由在线程池执行，所有操作加锁。
"""
import copy
import logging
import os
import re
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from core.config import settings
from repositories.elasticsearch import zh_convert

logger = logging.getLogger(__name__)


def normalize_query_for_cache(query: str) -> str:
    """缓存键归一化：压缩空白 + 小写 + 繁→简，检索与问答缓存共用。"""
    if not query:
        return ""
    collapsed = re.sub(r"\s+", " ", query).strip().lower()
    return zh_convert(collapsed, "t2s")


# 缓存键：(筛选条件标识, 归一化查询)
CacheKey = Tuple[str, str]


class SearchResultCache:
    """有界 LRU 缓存（按法规目录版本失效）。

    检索缓存与问答答案缓存各持一个实例，互不共享条目与命中统计；
    存取均走深拷贝，避免调用方原地修改污染缓存。
    """

    def __init__(self, max_entries: int = 500, catalog_path: Optional[str] = None):
        self._lock = threading.RLock()
        self._entries: "OrderedDict[CacheKey, List[Dict]]" = OrderedDict()
        self._max_entries = max(1, int(max_entries))
        self._catalog_path = catalog_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "law_catalog.json",
        )
        self._version: Optional[float] = None
        self._hits = 0
        self._misses = 0

    def get(self, key: CacheKey) -> Optional[List[Dict]]:
        """命中返回结果副本（按最近使用置尾），未命中返回 None"""
        with self._lock:
            self._refresh_locked()
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            logger.info(f"[检索缓存] 命中: {key[1][:40]}")
            return copy.deepcopy(entry)

    def set(self, key: CacheKey, results: List[Dict]) -> None:
        with self._lock:
            self._refresh_locked()
            self._entries[key] = copy.deepcopy(results)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
            }

    def _catalog_version(self) -> Optional[float]:
        try:
            return os.path.getmtime(self._catalog_path)
        except OSError:
            return None

    def _refresh_locked(self) -> None:
        """法规目录版本变化即整体清空；首次调用只记录版本，不做清空日志"""
        version = self._catalog_version()
        if version == self._version:
            return
        if self._version is not None and self._entries:
            logger.info(f"[检索缓存] 法规目录已变更，清空 {len(self._entries)} 条缓存")
        self._entries.clear()
        self._version = version


_cache: Optional[SearchResultCache] = None
_cache_lock = threading.Lock()


def get_search_cache() -> SearchResultCache:
    """进程内单例（惰性创建），条数上限来自 settings"""
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = SearchResultCache(max_entries=settings.search_cache_max_entries)
    return _cache


_qa_cache: Optional[SearchResultCache] = None
_qa_cache_lock = threading.Lock()


def get_qa_answer_cache() -> SearchResultCache:
    """问答答案缓存（进程内单例，与检索缓存相互独立）。

    值形态：{"answer", "sources", "retrieved_count", "jurisdictions"}。
    """
    global _qa_cache
    if _qa_cache is None:
        with _qa_cache_lock:
            if _qa_cache is None:
                _qa_cache = SearchResultCache(max_entries=settings.qa_answer_cache_max_entries)
    return _qa_cache
