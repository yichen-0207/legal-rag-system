"""零 token 探针：多跳链路串行 ES 查询点量化 + 并行改造前后的逐条等价性与耗时对比（不构造 LLM）。

用法（容器内，零 LLM）：
  docker compose run --rm -T --no-deps backend python /app/evals/scripts/probe_multihop_serial.py

测量口径：
  1) ES 请求层：包装 ElasticsearchRepository._client，记录每次 search/msearch/get/count 的耗时与是否报错；
  2) 服务阶段层：包装 MultiHopService._locate_source / _target_articles / _shared_concepts
     与 SimilarLawService.get_similar_laws，记录每次调用耗时与期间 ES 请求数；
  3) 三类路径各跑：cross / entity / citation，逐条记录总墙钟；
  4) 等价性：
     a. 路径级：串行 vs 并行（同进程交替跑）对比 (law_id, article, 顺序)，并与改造前基线文件对比；
     b. 仓库级：同一查询 + 多 law_id 过滤，hybrid_search 逐个调用 vs hybrid_search_same_query
        对比 ids/documents/_scores/_similarities 逐字段。
"""
import json
import logging
import os
import statistics
import time
from collections import defaultdict

from _bootstrap import DATASETS_DIR, RESULTS_DIR

from core.config import settings

# 零 token：禁止任何 LLM 路径；关闭缓存避免复用上一次结果掩盖真实耗时
settings.search_cache_enabled = False
settings.multihop_decompose_llm_enabled = False
settings.multihop_deepen_enabled = False
settings.reranker_enabled = False  # 多跳链路本身不精排，关掉可省 ~1GB 内存与加载时间

from repositories.elasticsearch import ElasticsearchRepository  # noqa: E402
from services.kg_reasoning_service import KGReasoningService  # noqa: E402
from services.multihop_service import MultiHopService  # noqa: E402
from services.retriever_service import RetrieverService  # noqa: E402
from services.similar_law_service import SimilarLawService  # noqa: E402

DATA_PATH = os.path.join(DATASETS_DIR, "multihop_dataset.json")
OUT_JSON = os.path.join(RESULTS_DIR, "probe_multihop_serial_ab.json")
ORDER_JSON = os.path.join(RESULTS_DIR, "probe_multihop_serial_baseline_order.json")
PREV_ORDER_JSON = ORDER_JSON + ".prev.json"   # 改造前串行基线（由旧版探针落盘，改名保留）

ES_RECORDS = []
STAGE_RECORDS = []
WARNINGS = []
REPEATS = 3


def _now():
    return time.perf_counter()


def _req_label(kind, args, kwargs):
    if kind in ("msearch", "get", "count"):
        return kind
    body = kwargs.get("query") if isinstance(kwargs.get("query"), dict) else kwargs.get("body")
    if not isinstance(body, dict):
        return kind
    if "aggs" in body:
        return "aggs"
    if "knn" in body:
        return "knn"
    if "sort" in body and "query" in body:
        return "search-sorted"
    if "size" in body and "query" in body:
        return "search"
    return kind


class TimingClient:
    """透明代理：记录 ES 请求耗时后原样返回结果；异常仍按原样抛出。"""

    _TRACKED = {"search", "msearch", "get", "count", "update", "delete_by_query", "bulk"}

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr) or name not in self._TRACKED:
            return attr

        def wrapped(*args, **kwargs):
            t0 = _now()
            ok = True
            try:
                return attr(*args, **kwargs)
            except Exception:
                ok = False
                raise
            finally:
                ES_RECORDS.append({
                    "kind": name,
                    "label": _req_label(name, args, kwargs),
                    "ms": (_now() - t0) * 1000,
                    "ok": ok,
                    "t0": t0,
                })

        return wrapped


class _WarnCapture(logging.Handler):
    def emit(self, record):
        WARNINGS.append(f"{record.name}:{record.levelname}:{record.getMessage()[:160]}")


def trace_stage(obj, name):
    orig = getattr(obj, name)

    def wrapped(*args, **kwargs):
        t0 = _now()
        n0 = len(ES_RECORDS)
        try:
            return orig(*args, **kwargs)
        finally:
            t1 = _now()
            STAGE_RECORDS.append({
                "stage": name,
                "ms": (t1 - t0) * 1000,
                "es": [r for r in ES_RECORDS[n0:] if r["t0"] <= t1],
            })

    setattr(obj, name, wrapped)


def ref_key(law_id, article):
    return f"{law_id}::{article}" if law_id and article else None


def cross_order(data):
    """与 run_multihop_path_eval.cross_ranked 同一口径：末端条款优先，最后补源条款。"""
    ranked = []
    for target in data.get("targets") or []:
        for article in target.get("articles") or []:
            key = ref_key(target.get("law_id"), article.get("article_number"))
            if key and key not in ranked:
                ranked.append(key)
    source = data.get("source") or {}
    key = ref_key(source.get("law_id"), source.get("article_number"))
    if key and key not in ranked:
        ranked.append(key)
    return ranked


def entity_order(data):
    ranked = []
    for path in data.get("reasoning_path") or []:
        articles = []
        for step in path.get("steps") or []:
            ev = ((step.get("evidence") or [{}])[0]) or {}
            for field in ("article", "to_article", "from_article"):
                key = ref_key(ev.get("law_id"), ev.get(field))
                if key and key not in articles:
                    articles.append(key)
        for key in reversed(articles):
            if key not in ranked:
                ranked.append(key)
    return ranked


def citation_order(repo, item):
    source = item.get("source") or {}
    law_id, article = source.get("law_id"), source.get("article")
    content = repo.get_article_full_text(law_id, article)
    if not content:
        return []
    seeds = [(law_id, article, content)]
    rounds = []
    for _ in range(max(1, int(item.get("hops") or 2))):
        docs = repo.get_referenced_articles(seeds, max_refs=10)
        if not docs:
            break
        keys = []
        for doc in docs:
            key = ref_key(doc.get("law_id"), doc.get("article_number"))
            if key and key not in keys:
                keys.append(key)
        rounds.append(keys)
        seeds = [(d.get("law_id", ""), d.get("article_number", ""), d.get("content", "") or "")
                 for d in docs]
    ranked = []
    for group in reversed(rounds):
        for key in group:
            if key not in ranked:
                ranked.append(key)
    return ranked


def summarize_stages(rows):
    by_stage = defaultdict(list)
    for row in rows:
        for rec in row["stages"]:
            by_stage[rec["stage"]].append(rec)
    out = {}
    for stage, recs in by_stage.items():
        per_kind = defaultdict(list)
        for r in recs:
            for e in r["es"]:
                per_kind[e["kind"] + "/" + e["label"]].append(e["ms"])
        out[stage] = {
            "calls": len(recs),
            "calls_per_item": round(len(recs) / max(len(rows), 1), 2),
            "single_median_ms": round(statistics.median([r["ms"] for r in recs]), 1),
            "single_max_ms": round(max(r["ms"] for r in recs), 1),
            "serial_total_ms": round(sum(r["ms"] for r in recs), 1),
            "es_req_median": round(statistics.median([len(r["es"]) for r in recs]), 2),
            "es_req_total": sum(len(r["es"]) for r in recs),
            "es_by_kind_median_ms": {
                k: round(statistics.median(v), 1) for k, v in sorted(per_kind.items())
            },
        }
    return out


def compare_orders(base, other, label, diffs):
    """逐条对比有序结果（law_id::article 顺序），只报告差异条数与明细。"""
    bad = 0
    for item_id, keys in (base or {}).items():
        other_keys = (other or {}).get(item_id)
        if other_keys is None:
            diffs.append({"item": item_id, "check": label, "why": "missing"})
            bad += 1
            continue
        if keys != other_keys:
            bad += 1
            diffs.append({"item": item_id, "check": label,
                          "serial": keys, "other": other_keys})
    return bad


def repo_level_equivalence(repo, items, diffs):
    """仓库级逐字段等价：逐个 hybrid_search（串行） vs hybrid_search_same_query（并发，编码一次）。"""
    checked = 0
    for item in items:
        law_ids = []
        for key in (item.get("_order_keys") or []):
            lid = key.split("::", 1)[0]
            if lid and lid not in law_ids:
                law_ids.append(lid)
        law_ids = law_ids[:4]
        if len(law_ids) < 2:
            continue
        query = item["query"]
        serial = [repo.hybrid_search(query, n_results=6, where={"law_id": lid})
                  for lid in law_ids]
        parallel = repo.hybrid_search_same_query(
            query, [{"law_id": lid} for lid in law_ids], 6, max_workers=4
        )
        checked += 1
        for lid, s, p in zip(law_ids, serial, parallel):
            same_ids = s.get("ids") == p.get("ids")
            same_docs = s.get("documents") == p.get("documents")
            same_scores = s.get("_scores") == p.get("_scores")
            same_sims = s.get("_similarities") == p.get("_similarities")
            if not (same_ids and same_docs and same_scores and same_sims):
                diffs.append({
                    "item": item["id"], "check": "repo", "law_id": lid,
                    "same_ids": same_ids, "same_docs": same_docs,
                    "same_scores": same_scores, "same_sims": same_sims,
                    "serial_scores": (s.get("_scores") or [[]])[0][:5],
                    "parallel_scores": (p.get("_scores") or [[]])[0][:5],
                })
    return checked


def main():
    items = json.load(open(DATA_PATH, encoding="utf-8"))["items"]

    for logger_name in ("urllib3", "elastic_transport", "elasticsearch", "services.multihop_service"):
        lg = logging.getLogger(logger_name)
        lg.addHandler(_WarnCapture())
        lg.setLevel(logging.WARNING)

    prev = None
    if os.path.exists(PREV_ORDER_JSON):
        prev = json.load(open(PREV_ORDER_JSON, encoding="utf-8"))
    elif os.path.exists(ORDER_JSON):
        prev = json.load(open(ORDER_JSON, encoding="utf-8"))

    repo = ElasticsearchRepository()
    _ = repo.client
    repo._client = TimingClient(repo._client)

    retriever = RetrieverService()
    retriever.repo = repo
    multihop = MultiHopService(repo=repo, similar_service=SimilarLawService(repo))
    kg = KGReasoningService(repo=repo)
    kg._retriever = retriever

    for obj, name in (
        (multihop, "_locate_source"),
        (multihop, "_search_laws_same_query"),
        (multihop, "_target_articles_from_hits"),
        (multihop, "_shared_concepts"),
        (multihop.similar, "get_similar_laws"),
        (kg, "_link_entities"),
        (kg, "_retrieve_article_refs"),
        (kg, "_collect_article_triples"),
        (kg, "_collect_subgraph"),
    ):
        trace_stage(obj, name)

    order = {"serial": {"cross": {}, "cross_oracle": {}, "entity": {}, "citation": {}},
             "parallel": {"cross": {}, "cross_oracle": {}, "entity": {}, "citation": {}}}
    rows = []
    diffs = []

    print("=" * 96)
    print(f"  多跳串行 ES 查询点 / 并行 A/B（零 LLM）：{len(items)} 条 × {REPEATS} 次 × 2 模式")
    print("=" * 96)

    # 预热（模型加载不计入）
    multihop.answer(question=items[0]["query"], top_k_laws=3, synthesize=False)

    for mode, flag in (("serial", False), ("parallel", True)):
        settings.multihop_parallel_search_enabled = flag
        for idx, item in enumerate(items, start=1):
            typ = item["type"]
            for rep in range(REPEATS):
                STAGE_RECORDS.clear()
                ES_RECORDS.clear()
                t0 = _now()
                if typ == "cross_jurisdiction":
                    data = multihop.answer(
                        question=item["query"], top_k_laws=3, synthesize=False,
                        jurisdictions=item.get("jurisdictions"),
                    )
                    ordered = cross_order(data)
                elif typ == "entity_navigation":
                    data = kg.answer(
                        item["query"], top_k_entities=5, hops=2, deepen=False,
                        synthesize=False, jurisdictions=item.get("jurisdictions"),
                    )
                    ordered = entity_order(data)
                else:
                    ordered = citation_order(repo, item)
                    data = {}
                wall = (_now() - t0) * 1000
                rows.append({
                    "mode": mode, "id": item["id"], "type": typ, "rep": rep,
                    "wall_ms": round(wall, 1),
                    "es_requests": len(ES_RECORDS),
                    "es_errors": sum(1 for r in ES_RECORDS if not r["ok"]),
                    "es_total_ms": round(sum(r["ms"] for r in ES_RECORDS), 1),
                    "stages": [{"stage": r["stage"], "ms": round(r["ms"], 1),
                                "es": [{"kind": e["kind"], "label": e["label"],
                                        "ms": round(e["ms"], 1)} for e in r["es"]]}
                               for r in STAGE_RECORDS],
                })
                if rep == 0:
                    if typ == "cross_jurisdiction":
                        order[mode]["cross"][item["id"]] = ordered
                        oracle = multihop.answer(
                            question=item["query"], top_k_laws=3, synthesize=False,
                            jurisdictions=item.get("jurisdictions"),
                            source_law_id=(item.get("source") or {}).get("law_id"),
                            source_article=(item.get("source") or {}).get("article"),
                        )
                        order[mode]["cross_oracle"][item["id"]] = cross_order(oracle)
                    elif typ == "entity_navigation":
                        order[mode]["entity"][item["id"]] = ordered
                    else:
                        order[mode]["citation"][item["id"]] = ordered

    # ---------- 等价性 ----------
    # 1) 改造前串行基线 vs 改造后串行（应完全一致）
    if prev:
        compare_orders(prev.get("cross"), order["serial"]["cross"], "prev-vs-now(serial) cross", diffs)
        compare_orders(prev.get("cross_oracle"), order["serial"]["cross_oracle"],
                       "prev-vs-now(serial) cross_oracle", diffs)
        compare_orders(prev.get("entity"), order["serial"]["entity"], "prev-vs-now(serial) entity", diffs)
        compare_orders(prev.get("citation"), order["serial"]["citation"], "prev-vs-now(serial) citation", diffs)
    # 2) 同进程串行 vs 并行（核心等价性证据）
    compare_orders(order["serial"]["cross"], order["parallel"]["cross"], "serial-vs-parallel cross", diffs)
    compare_orders(order["serial"]["cross_oracle"], order["parallel"]["cross_oracle"],
                   "serial-vs-parallel cross_oracle", diffs)
    compare_orders(order["serial"]["entity"], order["parallel"]["entity"], "serial-vs-parallel entity", diffs)
    compare_orders(order["serial"]["citation"], order["parallel"]["citation"],
                   "serial-vs-parallel citation", diffs)
    # 3) 仓库级逐字段
    cross_items = [i for i in items if i["type"] == "cross_jurisdiction"]
    for item in cross_items:
        item["_order_keys"] = order["serial"]["cross"].get(item["id"]) or []
    repo_checked = repo_level_equivalence(repo, cross_items, diffs)

    # ---------- 汇总 ----------
    per_type = {}
    for typ in ("cross_jurisdiction", "entity_navigation", "citation_chain"):
        per_type[typ] = {}
        for mode in ("serial", "parallel"):
            subset = [r for r in rows if r["type"] == typ and r["mode"] == mode]
            per_type[typ][mode] = {
                "items": len(subset) // REPEATS,
                "wall_median_ms": round(statistics.median([r["wall_ms"] for r in subset]), 1),
                "wall_max_ms": round(max(r["wall_ms"] for r in subset), 1),
                "es_requests_median": round(statistics.median([r["es_requests"] for r in subset]), 1),
                "es_errors": sum(r["es_errors"] for r in subset),
                "es_total_ms_median": round(statistics.median([r["es_total_ms"] for r in subset]), 1),
                "stages": summarize_stages(subset),
            }

    print("\n  === 分类型耗时（中位数，含 embedding） ===")
    for typ, modes in per_type.items():
        s, p = modes["serial"], modes["parallel"]
        save = s["wall_median_ms"] - p["wall_median_ms"]
        print(f"  [{typ}] n={s['items']}  serial={s['wall_median_ms']:>7.1f}ms  "
              f"parallel={p['wall_median_ms']:>7.1f}ms  省={save:>7.1f}ms "
              f"({(save / s['wall_median_ms'] * 100 if s['wall_median_ms'] else 0):.1f}%)")
    print("\n  === 跨法域阶段分解（串行模式，单次调用中位数 / 每项调用次数） ===")
    for stage, st in per_type["cross_jurisdiction"]["serial"]["stages"].items():
        print(f"      {stage:<26} calls/item={st['calls_per_item']:<5} "
              f"单次中位={st['single_median_ms']:>7}ms 串行合计={st['serial_total_ms']:>8}ms "
              f"ES/次={st['es_req_median']}")

    print(f"\n  === 等价性 === 差异条数={len(diffs)}  仓库级对比组合={repo_checked}")
    odd = [d for d in diffs if d.get("item", "").startswith("MH-")]
    for d in diffs[:12]:
        print("    ", json.dumps(d, ensure_ascii=False)[:400])

    errs = sum(r["es_errors"] for r in rows)
    pool_warn = [w for w in WARNINGS if "pool" in w.lower() or "connection" in w.lower()]
    print(f"\n  ES 报错={errs}  连接池/连接类告警={len(pool_warn)}  其他告警={len(WARNINGS) - len(pool_warn)}")
    for w in (pool_warn + WARNINGS)[:6]:
        print("    ", w)

    payload = {
        "meta": {
            "zero_llm": True,
            "es_hosts": settings.es_hosts,
            "repeats": REPEATS,
            "switch": "settings.multihop_parallel_search_enabled",
            "note": "wall_ms 含 BGE-M3 CPU 编码；stages 为串行累计值，可视为并发改造收益上限",
        },
        "by_type": per_type,
        "equivalence": {
            "diff_count": len(diffs),
            "repo_level_checked": repo_checked,
            "diffs": diffs,
        },
        "es_errors": errs,
        "warnings": WARNINGS[:200],
        "orders": order,
        "rows": rows,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"\n  已写入：{OUT_JSON}")


if __name__ == "__main__":
    main()
