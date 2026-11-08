"""法规数据统计 API：提供可视化看板所需的总览 / 法域 / 主题 / 时间趋势聚合数据。"""

from fastapi import APIRouter, Query, HTTPException
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

try:  # orjson 解析大 JSON 比标准库快约 1.5-2 倍；缺失时自动回退
    import orjson as _orjson
except Exception:  # pragma: no cover
    _orjson = None

from schemas import APIResponse
from repositories.elasticsearch import ElasticsearchRepository, _to_label_zh, _jurisdiction_to_zh, is_excluded_topic
from utils.law_catalog import LawCatalog

router = APIRouter(prefix="/stats", tags=["数据统计"])

repo = ElasticsearchRepository()
catalog = LawCatalog()

# 进程内短缓存（雷达图）：数据源是 data/raw 下全部法域目录的 JSON，结果只随语料变化。
# 键含数据版本号；管理员增删改法规会 bump_data_version，缓存随版本自然失效。
_RADAR_CACHE: Dict[str, Any] = {}
_RADAR_TTL = 300.0

# 原始语料索引：只保存「法域 → 主题 → chunk 数」，不保留 chunk 正文/向量。
# 目录 mtime + 数据版本未变时整目录复用；否则按每个 JSON 的 (size, mtime) 指纹增量解析。
# 解析逻辑变更时必须把版本号 +1，否则会沿用旧口径的结果。
# 落盘位置由 LEGAL_RADAR_CACHE_DIR 指定（compose 命名卷 /var/cache/legal-radar），
# 容器 recreate 后索引仍在；未配置或目录不可写时退回系统临时目录（旧行为）。
_RADAR_INDEX_VERSION = 3
_RADAR_INDEX_FILENAME = "legal_radar_index_v%d.json" % _RADAR_INDEX_VERSION


def _resolve_radar_index_file() -> str:
    """解析索引文件路径：优先持久目录，不可用时回退 /tmp 并记录原因。"""
    from core.config import settings

    configured = (settings.radar_cache_dir or "").strip()
    if configured:
        try:
            os.makedirs(configured, exist_ok=True)
            probe = os.path.join(configured, ".write_probe")
            with open(probe, "w", encoding="utf-8") as fh:
                fh.write("1")
            os.remove(probe)
            logger.info("[雷达索引] 持久目录可用: %s", configured)
            return os.path.join(configured, _RADAR_INDEX_FILENAME)
        except OSError as e:
            logger.warning("[雷达索引] 持久目录不可写，回退临时目录: %s (%s)", configured, e)
    else:
        logger.warning("[雷达索引] 未配置 LEGAL_RADAR_CACHE_DIR，索引落临时目录：容器重建后会丢失")
    return os.path.join(tempfile.gettempdir(), _RADAR_INDEX_FILENAME)


_RADAR_INDEX_FILE = _resolve_radar_index_file()


def _data_version_key() -> str:
    """取当前数据版本号（复用 redis_cache 的 bump_data_version 机制）。"""
    try:
        from utils.redis_cache import get_data_version
        return str(get_data_version())
    except Exception:
        return "static"


# ==================== 数据获取辅助函数（不包含路由参数） ====================

def _get_best_date(law: Dict) -> str:
    """取法规最佳可视化日期（passing_date → effective_date → publication_date → ingest_date）"""
    for field in ("passing_date", "effective_date", "publication_date", "ingest_date"):
        date_str = law.get(field, "")
        if not date_str or date_str == "0000-00-00":
            continue
        date_str = str(date_str).strip()
        if len(date_str) < 4:
            continue
        if len(date_str) == 4 and date_str.isdigit():
            return f"{date_str}-01-01"
        if len(date_str) >= 10:
            return date_str[:10]
    return ""


def _fetch_overview_data() -> dict:
    """获取总览数据"""
    cat_data = catalog.load()
    laws = [l for l in cat_data.get("laws", []) if l.get("status") != "deleted"]
    
    total_laws = len(laws)
    total_chunks = sum(l.get("total_chunks", 0) for l in laws)
    total_articles = sum(l.get("total_articles", 0) for l in laws)

    # 法域去重（转换为中文）
    jurisdictions = set(_jurisdiction_to_zh(l.get("jurisdiction", "")) for l in laws)
    
    # 时间范围
    dates = []
    for law in laws:
        date_str = _get_best_date(law)
        if date_str:
            try:
                dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
            except:
                pass
    
    date_range = {}
    if dates:
        date_range["earliest"] = min(dates).strftime("%Y-%m-%d")
        date_range["latest"] = max(dates).strftime("%Y-%m-%d")
    
    # 最近更新
    latest_ingest = None
    if laws:
        sorted_laws = sorted(laws, key=lambda x: x.get("last_updated", ""), reverse=True)
        latest_law = sorted_laws[0]
        latest_ingest = {
            "law_name": latest_law.get("law_name_zh", ""),
            "date": latest_law.get("last_updated", ""),
            "law_id": latest_law.get("law_id", "")
        }
    
    # 收集所有主题（剔除已废弃的粗粒度主题）
    all_topics = set()
    for law in laws:
        topics = law.get("topics", [])
        if isinstance(topics, list):
            all_topics.update(t for t in topics if t and not is_excluded_topic(t))
    
    return {
        "total_laws": total_laws,
        "total_chunks": total_chunks,
        "total_articles": total_articles,
        "jurisdictions_count": len(jurisdictions),
        "topics_count": len(all_topics),
        "jurisdictions": list(jurisdictions),
        "date_range": date_range,
        "latest_ingest": latest_ingest,
        "last_catalog_update": cat_data.get("last_updated", "")
    }


def _fetch_jurisdiction_data() -> list:
    """获取法域统计数据"""
    cat_data = catalog.load()
    laws = [l for l in cat_data.get("laws", []) if l.get("status") != "deleted"]
    
    # 按 jurisdiction 分组统计
    jurisdiction_stats: Dict[str, Dict] = {}
    
    for law in laws:
        jur = _jurisdiction_to_zh(law.get("jurisdiction", "未知"))
        
        if jur not in jurisdiction_stats:
            jurisdiction_stats[jur] = {
                "jurisdiction": jur,
                "law_count": 0,
                "chunk_count": 0,
                "article_count": 0,
                "laws": []
            }
        
        jurisdiction_stats[jur]["law_count"] += 1
        jurisdiction_stats[jur]["chunk_count"] += law.get("total_chunks", 0)
        jurisdiction_stats[jur]["article_count"] += law.get("total_articles", 0)
        jurisdiction_stats[jur]["laws"].append({
            "law_id": law.get("law_id"),
            "law_name_zh": law.get("law_name_zh"),
            "passing_date": law.get("passing_date"),
            "effective_date": law.get("effective_date"),
            "publication_date": law.get("publication_date"),
            "display_date": _get_best_date(law),
            "total_chunks": law.get("total_chunks"),
            "total_articles": law.get("total_articles"),
            "status": law.get("status")
        })
    
    result = list(jurisdiction_stats.values())
    result.sort(key=lambda x: x["law_count"], reverse=True)
    
    return result


def _fetch_topic_data(limit: int = 50) -> list:
    """获取主题统计数据"""
    # 优先用 ES 的更细主题统计，无数据时回退到 catalog
    topic_stats = repo.get_topic_statistics(limit=limit)
    
    if topic_stats and len(topic_stats) > 0:
        return topic_stats
    
    # 回退到从 catalog 统计
    cat_data = catalog.load()
    laws = [l for l in cat_data.get("laws", []) if l.get("status") != "deleted"]
    
    topic_distribution: Dict[str, Dict] = {}
    
    for law in laws:
        topics = law.get("topics", [])
        jur = _jurisdiction_to_zh(law.get("jurisdiction", "未知"))
        
        if not isinstance(topics, list):
            continue
        
        for topic in topics:
            if not topic or topic == "":
                continue
            if is_excluded_topic(topic):
                continue

            if topic not in topic_distribution:
                topic_distribution[topic] = {
                    "topic": topic,
                    "label_zh": _to_label_zh(topic),
                    "count": 0,
                    "by_jurisdiction": {},
                    "laws": []
                }
            
            topic_distribution[topic]["count"] += 1
            
            if jur not in topic_distribution[topic]["by_jurisdiction"]:
                topic_distribution[topic]["by_jurisdiction"][jur] = 0
            topic_distribution[topic]["by_jurisdiction"][jur] += 1
            
            if len(topic_distribution[topic]["laws"]) < 5:
                topic_distribution[topic]["laws"].append({
                    "law_id": law.get("law_id"),
                    "law_name_zh": law.get("law_name_zh")
                })
    
    result = list(topic_distribution.values())
    result.sort(key=lambda x: x["count"], reverse=True)
    
    return result[:limit]


def _fetch_time_data(group_by: str = "year") -> dict:
    """获取时间统计数据（按年或月分组，附各法域累计值）"""
    cat_data = catalog.load()
    laws = [l for l in cat_data.get("laws", []) if l.get("status") != "deleted"]
    
    time_stats: Dict[str, Dict] = {}
    
    for law in laws:
        date_str = _get_best_date(law)
        
        if not date_str:
            continue
        
        try:
            dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
            
            if group_by == "year":
                time_key = str(dt.year)
            else:
                time_key = f"{dt.year}-{dt.month:02d}"
            
            if time_key not in time_stats:
                time_stats[time_key] = {
                    "period": time_key,
                    "total": 0,
                    "by_jurisdiction": {},
                    "laws": []
                }
            
            time_stats[time_key]["total"] += 1
            
            jur = _jurisdiction_to_zh(law.get("jurisdiction", "未知"))
            if jur not in time_stats[time_key]["by_jurisdiction"]:
                time_stats[time_key]["by_jurisdiction"][jur] = 0
            time_stats[time_key]["by_jurisdiction"][jur] += 1
            
            time_stats[time_key]["laws"].append({
                "law_id": law.get("law_id"),
                "law_name_zh": law.get("law_name_zh"),
                "jurisdiction": jur,
                "display_date": date_str[:10],
                "passing_date": law.get("passing_date"),
                "effective_date": law.get("effective_date"),
                "publication_date": law.get("publication_date")
            })
            
        except ValueError:
            continue
    
    result = list(time_stats.values())
    result.sort(key=lambda x: x["period"])

    # 累计值按「实际存在的法域」动态生成：
    # 原先硬编码 macau/singapore/hongkong/other 三类，新入库的澳大利亚、中国大陆、
    # 韩国等法域全部被并进 other，看板上的累计曲线无法反映真实构成。
    cumulative_by_jur: Dict[str, int] = {}
    cumulative_total = 0

    for item in result:
        cumulative_total += item["total"]
        item["cumulative_total"] = cumulative_total

        for jur, count in item["by_jurisdiction"].items():
            cumulative_by_jur[jur] = cumulative_by_jur.get(jur, 0) + count

        item["cumulative_by_jurisdiction"] = dict(cumulative_by_jur)
        # 兼容旧字段：三个早期法域的累计值单独保留（前端旧版本可能读取）
        item["cumulative_macau"] = cumulative_by_jur.get("中国澳门", 0)
        item["cumulative_singapore"] = cumulative_by_jur.get("新加坡", 0)
        item["cumulative_hongkong"] = cumulative_by_jur.get("中国香港", 0)

    return {
        "group_by": group_by,
        "data": result,
        "jurisdictions": list(cumulative_by_jur.keys()),
        "summary": {
            "earliest_period": result[0]["period"] if result else "",
            "latest_period": result[-1]["period"] if result else "",
            "total_periods": len(result)
        }
    }


def _fetch_timeline_data(jurisdiction: Optional[str] = None) -> dict:
    """获取时间轴数据"""
    cat_data = catalog.load()
    laws = [l for l in cat_data.get("laws", []) if l.get("status") != "deleted"]
    
    if jurisdiction:
        laws = [l for l in laws if l.get("jurisdiction") == jurisdiction]
    
    # 过滤有效日期并排序
    valid_laws = []
    for law in laws:
        date_str = _get_best_date(law)
        if date_str:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                valid_laws.append({
                    **law,
                    "year": dt.year,
                    "sort_date": dt.strftime("%Y-%m-%d")
                })
            except ValueError:
                continue
    
    valid_laws.sort(key=lambda x: x["sort_date"])
    
    return {
        "timeline": valid_laws,
        "count": len(valid_laws),
        "jurisdiction_filter": jurisdiction
    }


def _chunk_topic_labels(chunk: Dict) -> List[str]:
    """解析一条原始 chunk 的主题中文标签，兼容两种上游格式。

    - 新版：``topics_details`` = [{"id": ..., "label_zh": ...}]，有中文标签，优先使用；
    - 旧版/其它语料：``topics`` = ["data_protection", ...] 字符串列表，无 label_zh，
      回退按 topic_id 转中文标签；若 ``topics`` 本身是 dict 列表也照样取 label_zh。
    """
    details = chunk.get("topics_details")
    labels: List[str] = []
    if isinstance(details, list) and details:
        for item in details:
            if isinstance(item, dict):
                label = item.get("label_zh") or item.get("label_en") or ""
                if not label:
                    tid = item.get("id") or item.get("topic_id") or ""
                    label = _to_label_zh(tid) if tid else ""
            elif isinstance(item, str):
                label = _to_label_zh(item)
            else:
                continue
            if label:
                labels.append(label)
        if labels:
            return labels

    topics = chunk.get("topics")
    if isinstance(topics, list):
        for item in topics:
            if isinstance(item, dict):
                label = item.get("label_zh") or item.get("label_en") or ""
            elif isinstance(item, str):
                label = _to_label_zh(item)
            else:
                continue
            if label:
                labels.append(label)
    return labels


def _load_raw_json(fpath: str):
    """读取原始 chunk JSON；优先 orjson，缺失时回退标准库。"""
    if _orjson is not None:
        with open(fpath, "rb") as fh:
            return _orjson.loads(fh.read())
    with open(fpath, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _summarize_raw_file(fpath: str) -> Dict[str, Any]:
    """统计单个原始 JSON 的「法域 → 主题 → chunk 数」，结果可写入索引复用。"""
    try:
        chunks = _load_raw_json(fpath)
    except Exception:
        return {"error": True}
    if not isinstance(chunks, list):
        return {"error": True}

    jur_topics: Dict[str, Dict[str, int]] = {}
    jur_chunks: Dict[str, int] = {}
    no_jur = 0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        jur = _jurisdiction_to_zh(chunk.get("jurisdiction") or "")
        if not jur:
            no_jur += 1
            continue
        jur_chunks[jur] = jur_chunks.get(jur, 0) + 1
        bucket = jur_topics.setdefault(jur, {})
        # 同一条 chunk 内同一主题只计一次，避免重复计入覆盖度
        for label in set(_chunk_topic_labels(chunk)):
            if is_excluded_topic(label):
                continue
            bucket[label] = bucket.get(label, 0) + 1
    return {"jur_topics": jur_topics, "jur_chunks": jur_chunks, "no_jur": no_jur}


def _read_radar_index() -> Dict[str, Any]:
    """读取原始语料索引；版本不符或损坏时返回空索引。"""
    try:
        with open(_RADAR_INDEX_FILE, "r", encoding="utf-8") as fh:
            idx = json.load(fh)
        if idx.get("version") == _RADAR_INDEX_VERSION and isinstance(idx.get("files"), dict):
            return idx
    except Exception:
        pass
    return {}


def _write_radar_index(index: Dict[str, Any]) -> None:
    """原子写入索引；失败不影响本次响应（下次只是重新解析）。"""
    try:
        tmp = _RADAR_INDEX_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(index, fh, ensure_ascii=False)
        os.replace(tmp, _RADAR_INDEX_FILE)
    except Exception:
        pass


def _scan_region_dirs(raw_base: str) -> Dict[str, int]:
    """返回 data/raw 下所有法域目录的 mtime 指纹（名 → mtime_ns），按名称排序。

    只取目录自身 mtime：新增/删除/改名文件都会更新它，成本是「目录数」次 stat 而非「文件数」次。
    文件内容原地改写不会更新目录 mtime，那种情况由数据版本号兜底。
    """
    dir_mtimes: Dict[str, int] = {}
    try:
        with os.scandir(raw_base) as it:
            entries = sorted(it, key=lambda e: e.name)
    except OSError:
        return dir_mtimes
    for entry in entries:
        if entry.name.startswith("."):
            continue
        try:
            if not entry.is_dir():
                continue
            dir_mtimes[entry.name] = entry.stat().st_mtime_ns
        except OSError:
            continue
    return dir_mtimes


def _fetch_radar_data(min_chunk_threshold: int = 3, max_dimensions: int = 10) -> dict:
    """获取雷达图数据：从原始 JSON 动态统计各法域的主题覆盖度并归一化到 0-100。

    低于 min_chunk_threshold 的维度不展示，最多取 max_dimensions 个。

    - 法域目录不写死：扫描 data/raw 下所有子目录（跳过隐藏目录），并按名称排序遍历，
      与文件名排序一起保证同一份语料每次返回相同的维度与法域顺序（图表不会刷新一次变一次）。
    - 主题解析同时兼容 ``topics_details``（dict 列表，含 label_zh）与 ``topics``（字符串列表）。
    - 响应区分「该法域无数据」与「整体算不出来」：``has_data`` / ``reason`` /
      ``empty_jurisdictions`` / ``jurisdictions_without_raw_data``，不再静默返回空数组。

    全量语料约 128MB / 540 个 JSON，单次解析代价高，分两层缓存：
      1) 文件级索引（跨进程/重启有效）：目录 mtime 未变且数据版本未变时直接复用索引，
         不再逐个 stat/解析文件；目录 mtime 变化时只重新解析该目录下指纹变化的文件；
      2) 进程内 300s 结果缓存，键含「数据版本 + 目录 mtime 指纹」，目录增删文件会立即失效。
    """
    import time as _time
    from collections import defaultdict, Counter

    raw_base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
    cur_data_version = _data_version_key()
    dir_mtimes = _scan_region_dirs(raw_base)
    region_dirs = list(dir_mtimes.keys())
    # 目录指纹既是缓存键的一部分，也是索引「是否可整目录信任」的依据
    _dir_fingerprint = "|".join("%s:%d" % (k, v) for k, v in dir_mtimes.items())

    _cache_key = "%s:%d:%d:%s" % (cur_data_version, min_chunk_threshold, max_dimensions, _dir_fingerprint)
    _hit = _RADAR_CACHE.get(_cache_key)
    if _hit is not None and _time.time() - _hit[0] < _RADAR_TTL:
        return _hit[1]

    _t0 = _time.time()

    _cache_parse_hits = 0
    _reparsed_files = 0

    # ---- 第一步：从原始 JSON 文件收集 (jurisdiction, label_zh) → chunk 数量 ----
    jur_topic_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    raw_chunk_counts: Dict[str, int] = defaultdict(int)

    scanned_files = 0
    failed_files = 0
    no_jur_chunks = 0

    _index = _read_radar_index()
    cached_files = _index.get("files") or {}
    cached_dir_mtimes = _index.get("dir_mtimes") or {}
    # 数据版本变了，说明可能发生过原地改文件（不改目录 mtime），此时逐文件校验指纹
    _version_unchanged = _index.get("data_version") == cur_data_version

    new_files: Dict[str, Any] = {}
    index_dirty = False

    for region_dir in region_dirs:
        region_path = os.path.join(raw_base, region_dir)
        _trust_dir = bool(_version_unchanged) and cached_dir_mtimes.get(region_dir) == dir_mtimes[region_dir]
        try:
            names = sorted(os.listdir(region_path))
        except OSError:
            continue
        for fname in names:
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(region_path, fname)
            rel = region_dir + "/" + fname

            entry = cached_files.get(rel)
            if _trust_dir and isinstance(entry, dict) and "error" not in entry:
                # 目录 mtime 与数据版本都没变 → 文件不可能新增/删除/改写，直接用索引
                _cache_parse_hits += 1
            else:
                try:
                    st = os.stat(fpath)
                    size, mtime = st.st_size, st.st_mtime_ns
                except OSError:
                    continue
                if (
                    isinstance(entry, dict)
                    and entry.get("size") == size
                    and entry.get("mtime") == mtime
                    and "error" not in entry
                ):
                    _cache_parse_hits += 1
                else:
                    entry = {"size": size, "mtime": mtime, **_summarize_raw_file(fpath)}
                    index_dirty = True
                    _reparsed_files += 1

            new_files[rel] = entry
            if entry.get("error"):
                failed_files += 1
                continue
            scanned_files += 1
            no_jur_chunks += entry.get("no_jur", 0)
            for jur, cnt in (entry.get("jur_chunks") or {}).items():
                raw_chunk_counts[jur] += cnt
            for jur, topics in (entry.get("jur_topics") or {}).items():
                bucket = jur_topic_counts[jur]
                for label, cnt in topics.items():
                    bucket[label] += cnt

    # 目录或文件集合变化也要让索引收敛
    if set(cached_files) != set(new_files):
        index_dirty = True
    if index_dirty:
        _write_radar_index({
            "version": _RADAR_INDEX_VERSION,
            "data_version": cur_data_version,
            "dir_mtimes": dir_mtimes,
            "files": new_files,
        })

    # ---- 第二步：统计全局各主题的总 chunk 数，筛选出有效维度 ----
    global_topic_counter: Counter = Counter()
    for jur_topics in jur_topic_counts.values():
        for label, cnt in jur_topics.items():
            global_topic_counter[label] += cnt

    # 过滤低于阈值的主题；同票数时按名称排序，保证维度顺序稳定
    ordered_topics = sorted(global_topic_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    dimensions = [label for label, cnt in ordered_topics if cnt >= min_chunk_threshold][:max_dimensions]

    # 若没有任何维度达到阈值，回退到全部主题（上限 max_dimensions）
    if not dimensions:
        dimensions = [label for label, _ in ordered_topics[:max_dimensions]]

    # ---- 第三步：获取法域列表（原始数据里的 + catalog 里的，并集后排序）----
    jurisdictions = set(jur_topic_counts.keys())
    catalog_jurs = {_jurisdiction_to_zh(j["jurisdiction"]) for j in _fetch_jurisdiction_data()}
    jurisdictions.update(j for j in catalog_jurs if j)

    # ---- 第四步：计算每个法域在每个维度的得分 ----
    radar_data: Dict[str, List[int]] = {}
    for jur in sorted(jurisdictions):
        topic_counter = jur_topic_counts.get(jur, {})
        radar_data[jur] = [topic_counter.get(dim, 0) for dim in dimensions]

    # ---- 第五步：归一化到 0-100 分 ----
    all_scores = [s for scores in radar_data.values() for s in scores]
    max_score = max(all_scores) if all_scores else 0
    if max_score == 0:
        normalized = {jur: [0.0 for _ in scores] for jur, scores in radar_data.items()}
    else:
        normalized = {
            jur: [round((s / max_score) * 100, 1) for s in scores]
            for jur, scores in radar_data.items()
        }

    # ---- 第六步：区分「该法域无数据」与「整体算不出来」----
    has_data = bool(dimensions) and max_score > 0
    if not scanned_files:
        reason = "no_raw_files"
    elif not global_topic_counter:
        reason = "no_topics"
    elif not dimensions:
        reason = "all_topics_below_threshold"
    else:
        reason = ""

    empty_jurisdictions = sorted(
        jur for jur, scores in radar_data.items() if not any(scores)
    )
    jurisdictions_without_raw_data = sorted(
        jur for jur in radar_data if raw_chunk_counts.get(jur, 0) == 0
    )

    _payload = {
        "dimensions": dimensions,
        "jurisdictions": normalized,
        "has_data": has_data,
        "reason": reason,
        "empty_jurisdictions": empty_jurisdictions,
        "jurisdictions_without_raw_data": jurisdictions_without_raw_data,
        "raw_chunk_counts": {jur: raw_chunk_counts.get(jur, 0) for jur in radar_data},
        "_meta": {
            "dimension_source": "dynamic_from_topics_details",
            "total_topics_found": len(global_topic_counter),
            "dimensions_after_filter": len(dimensions),
            "min_chunk_threshold": min_chunk_threshold,
            "scan_dirs": region_dirs,
            "scanned_files": scanned_files,
            "failed_files": failed_files,
            "chunks_without_jurisdiction": no_jur_chunks,
            "index_reused_files": _cache_parse_hits,
            "index_reparsed_files": _reparsed_files,
            "scan_seconds": round(_time.time() - _t0, 3),
            "all_available_topics": dict(global_topic_counter.most_common())
        }
    }
    # 只保留当前数据版本下的缓存项，避免版本切换/目录变更后旧键堆积
    for _k in [k for k in _RADAR_CACHE if not k.startswith(cur_data_version + ":")]:
        _RADAR_CACHE.pop(_k, None)
    _RADAR_CACHE[_cache_key] = (_time.time(), _payload)
    return _payload


# ==================== API 路由函数 ====================

@router.get("", summary="获取统计数据总入口")
@router.get("/", summary="获取统计数据总入口")
async def get_stats():
    """获取统计数据总入口（兼容前端直接调用 /api/v1/stats），返回完整看板数据。"""
    try:
        overview = _fetch_overview_data()
        jur_list = _fetch_jurisdiction_data()
        topic_list = _fetch_topic_data(limit=30)
        time_trend = _fetch_time_data(group_by="year")

        # 转换 jurisdiction 为前端期望的 dict 格式
        law_counts = [{"jurisdiction": j["jurisdiction"], "count": j["law_count"]} for j in jur_list]
        clause_counts = [{"jurisdiction": j["jurisdiction"], "count": j["article_count"]} for j in jur_list]

        dashboard_data = {
            "overview": {
                "total_laws": overview.get("total_laws", 0),
                "total_clauses": overview.get("total_articles", 0),
                "jurisdiction_count": overview.get("jurisdictions_count", 0),
                "topic_count": overview.get("topics_count", 0)
            },
            "jurisdiction": {
                "law_counts": law_counts,
                "clause_counts": clause_counts
            },
            "topics": topic_list,
            "time_trend": time_trend,
            "generated_at": datetime.now().isoformat()
        }
        return APIResponse(success=True, data=dashboard_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计数据失败: {str(e)}")


@router.get("/overview", summary="获取数据总览")
async def get_overview():
    """获取数据库总览数字（用于首页卡片展示）"""
    try:
        data = _fetch_overview_data()
        return APIResponse(success=True, data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取总览数据失败: {str(e)}")


@router.get("/by_jurisdiction", summary="按法域统计")
async def get_by_jurisdiction():
    """获取各法域的法律和条款数量统计"""
    try:
        data = _fetch_jurisdiction_data()
        return APIResponse(success=True, data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取法域统计失败: {str(e)}")


@router.get("/by_topic", summary="按主题统计")
async def get_by_topic(
    limit: int = Query(50, description="返回的主题数量上限")
):
    """获取主题分类及其在各法域的分布"""
    try:
        data = _fetch_topic_data(limit=limit)
        return APIResponse(success=True, data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取主题统计失败: {str(e)}")


@router.get("/by_time", summary="按时间统计")
async def get_by_time(
    group_by: str = Query("year", enum=["year", "month"], description="分组方式")
):
    """获取法规通过时间的分布趋势"""
    try:
        data = _fetch_time_data(group_by=group_by)
        return APIResponse(success=True, data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取时间统计失败: {str(e)}")


@router.get("/timeline", summary="获取法律时间轴")
async def get_timeline(
    jurisdiction: Optional[str] = Query(None, description="法域筛选")
):
    """获取法律的时间线列表"""
    try:
        data = _fetch_timeline_data(jurisdiction=jurisdiction)
        return APIResponse(success=True, data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取时间轴数据失败: {str(e)}")


@router.get("/radar", summary="法域对比雷达图数据")
async def get_radar_data():
    """获取用于绘制法域对比雷达图的归一化数据"""
    try:
        data = _fetch_radar_data()
        return APIResponse(success=True, data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取雷达图数据失败: {str(e)}")


@router.get("/full_dashboard", summary="获取完整看板数据")
async def get_full_dashboard():
    """一次性获取所有可视化所需数据（减少前端请求次数）"""
    try:
        # 直接调用内部数据获取函数，不调用路由函数
        dashboard_data = {
            "overview": _fetch_overview_data(),
            "by_jurisdiction": _fetch_jurisdiction_data(),
            "by_topic": _fetch_topic_data(limit=30),
            "by_time": _fetch_time_data(group_by="year"),
            "timeline": _fetch_timeline_data(),
            "radar": _fetch_radar_data(),
            "generated_at": datetime.now().isoformat(),
            "cache_ttl": 300  # 建议缓存5分钟
        }
        
        return APIResponse(success=True, data=dashboard_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取完整看板数据失败: {str(e)}")