"""
法规数据统计 API 接口

提供可视化看板所需的各种聚合统计数据：
- 总览数据（数字卡片）
- 按法域分布
- 按主题分类
- 按时间趋势
"""

from fastapi import APIRouter, Query, HTTPException
from typing import List, Dict, Optional, Any
from datetime import datetime
import json

from models.schemas import APIResponse
from repositories.elasticsearch import ElasticsearchRepository, _to_label_zh, _jurisdiction_to_zh
from utils.law_catalog import LawCatalog

router = APIRouter(prefix="/stats", tags=["数据统计"])

repo = ElasticsearchRepository()
catalog = LawCatalog()


# ==================== 数据获取辅助函数（不包含路由参数） ====================

def _get_best_date(law: Dict) -> str:
    """
    获取法规的最佳可视化日期。
    优先级：passing_date → effective_date → publication_date → ingest_date
    支持年份格式（如 "2011"）和完整日期格式（如 "2020-06-01"）。
    """
    for field in ("passing_date", "effective_date", "publication_date", "ingest_date"):
        date_str = law.get(field, "")
        if not date_str or date_str == "0000-00-00":
            continue
        date_str = str(date_str).strip()
        if len(date_str) < 4:
            continue
        # 年份格式（如 "2011"）补全为 "2011-01-01"
        if len(date_str) == 4 and date_str.isdigit():
            return f"{date_str}-01-01"
        # 完整日期格式（如 "2020-06-01"）
        if len(date_str) >= 10:
            return date_str[:10]
    return ""


def _fetch_overview_data() -> dict:
    """获取总览数据（内部函数）"""
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
    
    # 收集所有主题
    all_topics = set()
    for law in laws:
        topics = law.get("topics", [])
        if isinstance(topics, list):
            all_topics.update(topics)
    
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
    """获取法域统计数据（内部函数）"""
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
    """获取主题统计数据（内部函数）"""
    # 尝试从 ES 获取更详细的主题数据
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
    """获取时间统计数据（内部函数）"""
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
    
    # 按时间排序
    result = list(time_stats.values())
    result.sort(key=lambda x: x["period"])
    
    # 计算累计值
    cumulative = {"macau": 0, "singapore": 0, "hongkong": 0, "other": 0}
    cumulative_total = 0
    
    for item in result:
        cumulative_total += item["total"]
        item["cumulative_total"] = cumulative_total
        
        for jur, count in item["by_jurisdiction"].items():
            jur_lower = jur.lower()
            if "澳门" in jur or "macau" in jur_lower:
                cumulative["macau"] += count
                item["cumulative_macau"] = cumulative["macau"]
            elif "新加坡" in jur or "singapore" in jur_lower:
                cumulative["singapore"] += count
                item["cumulative_singapore"] = cumulative["singapore"]
            elif "香港" in jur or "hongkong" in jur_lower:
                cumulative["hongkong"] += count
                item["cumulative_hongkong"] = cumulative["hongkong"]
            else:
                cumulative["other"] += count
                item["cumulative_other"] = cumulative["other"]
    
    return {
        "group_by": group_by,
        "data": result,
        "summary": {
            "earliest_period": result[0]["period"] if result else "",
            "latest_period": result[-1]["period"] if result else "",
            "total_periods": len(result)
        }
    }


def _fetch_timeline_data(jurisdiction: Optional[str] = None) -> dict:
    """获取时间轴数据（内部函数）"""
    cat_data = catalog.load()
    laws = [l for l in cat_data.get("laws", []) if l.get("status") != "deleted"]
    
    # 筛选法域
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
    
    # 按日期排序
    valid_laws.sort(key=lambda x: x["sort_date"])
    
    return {
        "timeline": valid_laws,
        "count": len(valid_laws),
        "jurisdiction_filter": jurisdiction
    }


def _fetch_radar_data(min_chunk_threshold: int = 3, max_dimensions: int = 10) -> dict:
    """获取雷达图数据（内部函数）

    动态从原始数据的 topics.label_zh 统计各法域在各维度的覆盖度。
    新法规入库后会自动纳入新的重要维度，无需手动维护关键词映射。

    参数:
        min_chunk_threshold: 维度至少包含的 chunk 数量门槛（低于此值的不展示）
        max_dimensions: 雷达图最多展示的维度数量（取 topN）
    """
    import os
    from collections import defaultdict, Counter

    # ---- 第一步：从原始 JSON 文件收集 (jurisdiction, label_zh) → chunk 数量 ----
    jur_topic_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    raw_base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")

    for region_dir in ("macau", "hongkong", "singapore"):
        region_path = os.path.join(raw_base, region_dir)
        if not os.path.isdir(region_path):
            continue
        for fname in os.listdir(region_path):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(region_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                if not isinstance(chunks, list):
                    continue
                for chunk in chunks:
                    jur = chunk.get("jurisdiction", "")
                    if not jur:
                        continue
                    for topic_obj in chunk.get("topics", []):
                        if isinstance(topic_obj, dict):
                            label = topic_obj.get("label_zh", "")
                            if label:
                                jur_topic_counts[jur][label] += 1
            except Exception:
                continue

    # ---- 第二步：统计全局各主题的总 chunk 数，筛选出有效维度 ----
    global_topic_counter: Counter = Counter()
    for jur_topics in jur_topic_counts.values():
        for label, cnt in jur_topics.items():
            global_topic_counter[label] += cnt

    # 过滤低于阈值的主题，并取前 max_dimensions 个作为雷达图维度
    dimensions = [
        label for label, _ in global_topic_counter.most_common()
        if global_topic_counter[label] >= min_chunk_threshold
    ][:max_dimensions]

    # 若没有任何维度达到阈值，回退到全部主题（上限 max_dimensions）
    if not dimensions:
        dimensions = [label for label, _ in global_topic_counter.most_common(max_dimensions)]

    # ---- 第三步：获取实际有数据的法域列表 ----
    jurisdictions = sorted(jur_topic_counts.keys())

    # 若原始文件未覆盖，补充 catalog 中的法域
    catalog_jurs = {j["jurisdiction"] for j in _fetch_jurisdiction_data()}
    for j in catalog_jurs:
        if j not in jurisdictions:
            jurisdictions.append(j)

    # ---- 第四步：计算每个法域在每个维度的得分 ----
    radar_data: Dict[str, List[int]] = {}
    for jur in jurisdictions:
        scores = []
        topic_counter = jur_topic_counts.get(jur, {})
        for dim in dimensions:
            scores.append(topic_counter.get(dim, 0))
        radar_data[jur] = scores

    # ---- 第五步：归一化到 0-100 分 ----
    all_scores = [s for scores in radar_data.values() for s in scores]
    max_score = max(all_scores) if all_scores else 1
    if max_score == 0:
        max_score = 1

    normalized = {
        jur: [round((s / max_score) * 100, 1) for s in scores]
        for jur, scores in radar_data.items()
    }

    return {
        "dimensions": dimensions,
        "jurisdictions": normalized,
        "_meta": {
            "dimension_source": "dynamic_from_topics",
            "total_topics_found": len(global_topic_counter),
            "dimensions_after_filter": len(dimensions),
            "min_chunk_threshold": min_chunk_threshold,
            "all_available_topics": dict(global_topic_counter.most_common())
        }
    }


# ==================== API 路由函数 ====================

@router.get("/", summary="获取统计数据总入口")
async def get_stats():
    """
    获取统计数据总入口（兼容前端直接调用 /api/v1/stats）
    返回完整看板数据
    """
    try:
        # 获取原始数据
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
    """
    一次性获取所有可视化所需的数据（减少前端请求次数）
    
    适用于首次加载时批量获取全部数据
    """
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