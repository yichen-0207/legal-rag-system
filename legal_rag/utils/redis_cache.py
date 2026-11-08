# -*- coding: utf-8 -*-
"""答案层 Redis 缓存（Redis 不可用时自动退回进程内实现）。

定位：进程内 LRU 是底座，Redis 只解决「跨 backend 重启不丢缓存」。
语义近似命中已实测否决（危险近似与良性改写相似度区间重叠），因此只做精确命中。

redis 客户端在函数内导入：镜像未安装 redis 包时不得让模块导入期失败，
此时自动退回进程内实现，行为与未接入 Redis 完全一致。
"""
import hashlib
import json
import logging
import threading
import time
from typing import Any, Dict, Optional

from core.config import settings

logger = logging.getLogger(__name__)

_DATA_VER_KEY = "legal:qa:data_ver"
_STATS_KEY = "legal:qa:stats"
_INDEX_KEY = "legal:qa:index"
# 连接失败后的退避时间：期内一律不再尝试，避免每次请求都等超时
_FAIL_COOLDOWN = 60.0

_module_lock = threading.Lock()
_client_lock = threading.Lock()
_redis_module: Any = None
_redis_module_checked = False
_client: Any = None
_next_retry_at = 0.0
_local_data_version = 0
_connect_failures = 0


def _load_redis_module() -> Any:
    """函数内导入 redis；缺失时返回 None 并只探测一次。"""
    global _redis_module, _redis_module_checked
    if _redis_module_checked:
        return _redis_module
    with _module_lock:
        if not _redis_module_checked:
            try:
                import redis as _redis  # 局部导入：不在导入期硬依赖
                _redis_module = _redis
            except Exception as e:
                _redis_module = None
                logger.info("[Redis缓存] 客户端包不可用，退回进程内缓存: %s", e)
            _redis_module_checked = True
    return _redis_module


def _get_client() -> Any:
    """惰性建连；开关关闭、包缺失、连接失败都返回 None（调用方走进程内兜底）。"""
    global _client, _next_retry_at, _connect_failures
    if not settings.redis_cache_enabled:
        return None
    if _client is not None:
        return _client
    module = _load_redis_module()
    if module is None:
        return None
    with _client_lock:
        if _client is not None:
            return _client
        if time.time() < _next_retry_at:
            return None
        try:
            client = module.Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=settings.redis_connect_timeout,
                socket_timeout=settings.redis_socket_timeout,
                health_check_interval=30,
            )
            client.ping()
            _client = client
            logger.info("[Redis缓存] 已连接 %s", settings.redis_url)
            return _client
        except Exception as e:
            _connect_failures += 1
            _next_retry_at = time.time() + _FAIL_COOLDOWN
            logger.warning("[Redis缓存] 连接失败，%.0fs 内 bypass 回退进程内缓存: %s",
                           _FAIL_COOLDOWN, e)
            return None


def _drop_client(error: Exception) -> None:
    """运行期出错即断开并进入退避期，后续请求直接走进程内实现。"""
    global _client, _next_retry_at, _connect_failures
    with _client_lock:
        _client = None
        _next_retry_at = time.time() + _FAIL_COOLDOWN
        _connect_failures += 1
    logger.warning("[Redis缓存] 运行期异常，%.0fs 内 bypass 回退进程内缓存: %s",
                   _FAIL_COOLDOWN, error)


def _item_key(key: tuple) -> str:
    """条目键：全维度 filter_key + 归一化查询一起散列，避免超长键与原串泄露。"""
    raw = str(key[0]) + "\x00" + str(key[1])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return f"{settings.redis_key_prefix}item:{digest}"


def get_data_version() -> int:
    """当前数据版本；Redis 可用时读计数器，否则用进程内计数器。"""
    client = _get_client()
    if client is not None:
        try:
            return int(client.get(_DATA_VER_KEY) or 0)
        except Exception as e:
            _drop_client(e)
    return _local_data_version


def bump_data_version() -> int:
    """法规数据变更时递增版本（供 admin 写端点调用），返回新版本号。"""
    global _local_data_version
    client = _get_client()
    if client is not None:
        try:
            return int(client.incr(_DATA_VER_KEY))
        except Exception as e:
            _drop_client(e)
    _local_data_version += 1
    return _local_data_version


class RedisAnswerCache:
    """答案层缓存：Redis 为主、进程内 LRU 为兜底，接口与 SearchResultCache 兼容。

    双写是刻意的：Redis 中途挂掉时未过期的条目仍能从进程内命中，
    请求不会因为缓存后端故障而失败或变慢到重算整条问答链路。
    """

    def __init__(self, fallback):
        self._fallback = fallback
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._errors = 0
        self._redis_hits = 0
        self._redis_misses = 0

    def get(self, key: tuple) -> Optional[Dict]:
        client = _get_client()
        if client is not None:
            try:
                raw = client.get(_item_key(key))
                if raw is not None:
                    with self._lock:
                        self._redis_hits += 1
                        self._hits += 1
                    self._incr_stat(client, "hits")
                    logger.info("[答案缓存] Redis 命中: %s", str(key[1])[:40])
                    return json.loads(raw)
                with self._lock:
                    self._redis_misses += 1
            except Exception as e:
                with self._lock:
                    self._errors += 1
                self._incr_stat(client, "errors")
                _drop_client(e)

        data = self._fallback.get(key)
        with self._lock:
            if data is not None:
                self._hits += 1
            else:
                self._misses += 1
        self._incr_stat(client, "hits" if data is not None else "misses")
        return data

    def set(self, key: tuple, value: Dict) -> None:
        self._fallback.set(key, value)
        client = _get_client()
        if client is None:
            return
        try:
            item_key = _item_key(key)
            client.set(item_key, json.dumps(value, ensure_ascii=False),
                       ex=settings.redis_answer_cache_ttl)
            client.zadd(_INDEX_KEY, {item_key: time.time()})
            # 容量上限：超出部分按写入时间淘汰（条目本身也受 TTL 约束）
            overflow = int(client.zcard(_INDEX_KEY)) - settings.redis_answer_cache_max_entries
            if overflow > 0:
                popped = client.zpopmin(_INDEX_KEY, overflow)
                stale = [member for member, _score in popped] if popped else []
                if stale:
                    client.delete(*stale)
        except Exception as e:
            with self._lock:
                self._errors += 1
            self._incr_stat(client, "errors")
            _drop_client(e)

    def clear(self) -> None:
        self._fallback.clear()

    @staticmethod
    def _incr_stat(client: Any, field: str) -> None:
        """Redis 侧持久计数（best-effort）：backend 重启后仍能看到累计命中。"""
        if client is None:
            return
        try:
            client.hincrby(_STATS_KEY, field, 1)
        except Exception:
            pass

    def stats(self) -> Dict[str, Any]:
        client = _get_client()
        try:
            local_stats = self._fallback.stats()
        except Exception:
            local_stats = {}
        with self._lock:
            counters = {
                "hits": self._hits,
                "misses": self._misses,
                "errors": self._errors,
                "redis_hits": self._redis_hits,
                "redis_misses": self._redis_misses,
            }
        payload: Dict[str, Any] = {
            "backend": "redis" if client is not None else "memory",
            "enabled": settings.redis_cache_enabled,
            "ttl_seconds": settings.redis_answer_cache_ttl,
            "max_entries": settings.redis_answer_cache_max_entries,
            "counters": counters,
            "connection_failures": _connect_failures,
            "local": local_stats,
        }
        if client is not None:
            try:
                payload["redis"] = {
                    "entries": int(client.zcard(_INDEX_KEY)),
                    "counters": client.hgetall(_STATS_KEY),
                    "data_version": int(client.get(_DATA_VER_KEY) or 0),
                }
            except Exception as e:
                payload["redis"] = {"error": str(e)}
                _drop_client(e)
        return payload


def build_answer_cache(fallback):
    """开关打开且客户端可用时返回 Redis 实现，否则原样返回进程内实例。"""
    if not settings.redis_cache_enabled:
        return fallback
    if _load_redis_module() is None:
        return fallback
    return RedisAnswerCache(fallback)
