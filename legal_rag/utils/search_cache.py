# -*- coding: utf-8 -*-
"""检索结果缓存 / 问答答案缓存。

检索链路里 CPU 精排占单次耗时约 93%（16s 量级），而前端在页面刷新、切换标签后
会按设计重新发起同一查询（只保存查询条件、不保存结果快照）。本模块缓存
「查询 + 筛选条件 → 最终结果」，命中时整条检索链路（含精排）直接跳过。

查询归一化（压缩空白 + 繁→简）后比对，因此「澳門個人資料保護法」与
「澳门个人资料保护法」落到同一键，同一问题不因输入法差异重复支付一次完整检索。

关于「语义命中」为何不做（实测依据，勿轻易重启该方向）：
    用 bge-m3 实测两组查询对的余弦相似度 ——
      良性改写（同一问题换说法）：「处罚规定 / 罚则是什么」0.9397、「解雇规定 / 辞退的条款」0.8423
      危险近似（法律含义不同）  ：「通知义务 / 告知义务」0.9257、「最高一年徒刑 / 最高二年徒刑」0.9156
    两组区间重叠：良性最高 0.9397 与危险最高 0.9257 仅差 0.014，不存在既能接住改写、
    又能拦住法律含义不同表述的阈值。阈值取 0.98 则几乎永不触发（实测改写对只到 0.94），
    等同死代码；取 0.93 则「通知义务」会被判成「告知义务」并直接返回错误答案。
    法条场景返回错误答案的代价远高于省一次检索，故语义命中整体移除。
    附带收益：省去每次请求额外一次查询编码（几十毫秒）。

问答答案缓存复用本模块的同一套机制（第二个独立实例，见 get_qa_answer_cache）：
键仍是「归一化问题 + 筛选条件」的精确比对，因此同样不涉及上述语义阈值问题 ——
语义判定在上面的实测中已被否决，问答链路不会重蹈该方向。

失效：以 law_catalog.json 的 mtime 作为数据版本，版本变化时整体清空。
线程安全：路由已由线程池执行，多个请求会并发读写，故所有操作加锁。
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
    """缓存键的查询归一化：压缩空白 + 统一小写 + 繁→简。

    繁→简保证「澳門 勞動關係法」与「澳门劳动关系法」落到同一个缓存键，
    避免同一问题因输入法差异重复支付一次完整检索。

    检索缓存与问答答案缓存共用本函数：两者若各自实现，一旦归一化口径漂移，
    「同一问题」在两条链路上会落到不同的键，缓存收益与命中统计都会失真。
    """
    if not query:
        return ""
    collapsed = re.sub(r"\s+", " ", query).strip().lower()
    return zh_convert(collapsed, "t2s")


# 缓存键：(筛选条件标识, 归一化查询)
CacheKey = Tuple[str, str]


class SearchResultCache:
    """有界的「缓存键 → 值」缓存（LRU 淘汰 + 法规目录版本失效）。

    进程内存在两个独立实例，互不干扰：
      - 检索结果缓存（get_search_cache）：值形态为 List[Dict]（检索结果列表）；
      - 问答答案缓存（get_qa_answer_cache）：值形态为 Dict（答案正文 + 来源载荷）。
    两者共用本类与同一套归一化/失效逻辑，但不共享条目 —— 否则问答答案会占用
    检索缓存的条数配额，且同一问题在两条链路上的命中统计会相互污染。

    结果以深拷贝进出，避免调用方（检索服务 / 问答服务）后续对列表或字典的原地修改
    污染缓存内容。
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

    # ---------- 读 ----------

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

    # ---------- 写 ----------

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

    # ---------- 内部 ----------

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
    """问答答案缓存（进程内单例，惰性创建）。

    与检索缓存相互独立：同一问题的检索结果与问答答案是两次不同的开销
    （前者是精排 CPU，后者是 LLM 生成），分开计数才能看清单条链路的收益。

    缓存值的形态为 Dict：{"answer": 答案正文, "sources": 来源载荷,
    "retrieved_count": int, "jurisdictions": List[str]}。
    """
    global _qa_cache
    if _qa_cache is None:
        with _qa_cache_lock:
            if _qa_cache is None:
                _qa_cache = SearchResultCache(max_entries=settings.qa_answer_cache_max_entries)
    return _qa_cache
