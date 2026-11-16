"""四策略对比评测：仅BM25 / 仅向量 / 混合(RRF) / 混合+重排序（分两阶段跑，避免显存峰值）。

v2 数据集支持双层 gold：
- 法规级：gold_law_ids，覆盖全部查询（v2 无空法规级 gold）；
- 条款级：relevant，只覆盖有条款 gold 的查询，空 gold 的查询单列、不进均值。
v2 的 id 形如 TCN_F1_01_A，基础查询 id 一律取 query_id 字段；缺失时才按末尾
_A.._D 变体后缀回退一次。解析不认识的条目直接抛错，绝不静默跳过。
"""
import os, json, gc, csv, argparse, time

from _bootstrap import DATASETS_DIR, RESULTS_DIR
# NDCG 口径与计算统一在 eval_metrics 中实现（逐 k 截断；与 v1 历史 @5/@10 不可直接比较）
from eval_metrics import NDCG_DEFINITION, NDCG_KS, average_valid, ndcg_by_k
from core.jurisdiction import normalize_jurisdiction

EVAL_DATASET_PATH = os.path.join(DATASETS_DIR, "eval_dataset_clause.json")
TOP_K = 20
VARIANTS = ("A", "B", "C", "D")


class EvalQuery:
    def __init__(self, id="", query="", jurisdiction=None, relevant=None,
                 law_gold=None, base_query="", subset=""):
        self.id = id
        self.query = query
        self.jurisdiction = jurisdiction
        self.relevant = relevant or []
        self.law_gold = [x for x in (law_gold or []) if x]
        self.base_query = base_query or id
        self.subset = subset or ""

    def get_relevant_articles(self): return {(r["law_id"], r.get("article", "")) for r in self.relevant}
    def get_gain(self, law_id, article):
        for r in self.relevant:
            if r["law_id"] == law_id and r.get("article", "") == article:
                return (2 ** r.get("relevance", 1)) - 1
        return 0


def load_queries(dataset_path):
    """读数据集并做严格校验；任何条目解析不了都抛错，不静默丢弃。"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list) or not items:
        raise ValueError(f"数据集为空或不是 list: {dataset_path}")

    queries, seen, problems = [], set(), []
    for idx, item in enumerate(items):
        qid = str(item.get("id") or "").strip()
        text = item.get("query") or ""
        if not qid or not text:
            problems.append(f"第 {idx} 条缺 id/query: id={qid!r}")
            continue
        if qid in seen:
            problems.append(f"id 重复: {qid}")
        seen.add(qid)
        base = str(item.get("query_id") or "").strip()
        if not base:
            head, sep, tail = qid.rpartition("_")
            base = head if (sep and tail in VARIANTS and head) else qid
        queries.append(EvalQuery(
            id=qid, query=text,
            jurisdiction=item.get("jurisdiction"),
            relevant=item.get("relevant") or [],
            law_gold=item.get("gold_law_ids") or [],
            base_query=base,
            subset=item.get("subset") or "",
        ))
    if problems:
        raise ValueError("数据集解析失败（拒绝静默跳过）：\n  " + "\n  ".join(problems))
    if len(queries) != len(items):
        raise ValueError(f"条目数不一致: 读入 {len(items)}，解析 {len(queries)}")
    return queries


def compute_clause_metrics(retrieved_ids, gold_set, eval_query):
    """条款级指标；条款 gold 为空返回 None（不参与均值）。"""
    if not gold_set:
        return None
    mrr = 0.0
    for i, (lid, art) in enumerate(retrieved_ids):
        if (lid, art) in gold_set: mrr = 1.0 / (i + 1); break
    rel_count = len(gold_set)
    recall, precision = {}, {}
    for k in [1, 3, 5, 10, 20]:
        topk = retrieved_ids[:k]
        hits = sum(1 for lid, art in topk if (lid, art) in gold_set)
        recall[k] = hits / max(rel_count, 1); precision[k] = hits / max(k, 1)
    R = min(rel_count, 20)
    r_hits = sum(1 for lid, art in retrieved_ids[:R] if (lid, art) in gold_set)
    r_prec = r_hits / max(R, 1)
    # NDCG 逐 k 标准定义见 eval_metrics（单一实现）；gold 为空时该查询记 None
    dcg = ndcg_by_k(retrieved_ids, eval_query.get_gain, list(gold_set))
    return {"mrr": round(mrr, 4),
            "hit_any": 1.0 if mrr > 0 else 0.0,
            "rank1": 1.0 if mrr >= 0.999 else 0.0,
            "r_precision": round(r_prec, 4),
            "recall": {k: round(v, 4) for k, v in recall.items()},
            "precision": {k: round(v, 4) for k, v in precision.items()},
            "ndcg": {k: (round(v, 4) if v is not None else None) for k, v in dcg.items()},
            "relevant_count": rel_count, "retrieved_count": len(retrieved_ids)}


def _dedup(seq):
    out = []
    for x in seq:
        if x and x not in out:
            out.append(x)
    return out


def compute_law_metrics(retrieved_laws, gold_laws):
    """法规级指标：按 law_id 去重后算，增益取二元（命中=1）。

    顶层 20 条条款常集中在少数几部法，去重才能反映「法规」层面的排名质量。
    """
    gold = {x for x in (gold_laws or []) if x}
    if not gold:
        return None
    seen = _dedup(retrieved_laws)
    mrr = 0.0
    for i, lid in enumerate(seen):
        if lid in gold: mrr = 1.0 / (i + 1); break
    recall = {k: sum(1 for lid in seen[:k] if lid in gold) / len(gold) for k in (1, 5, 10, 20)}
    dcg = ndcg_by_k([(lid, "") for lid in seen],
                    lambda lid, _art: 1 if lid in gold else 0,
                    [(lid, "") for lid in gold])
    return {"mrr": round(mrr, 4),
            "hit_any": 1.0 if mrr > 0 else 0.0,
            "rank1": 1.0 if mrr >= 0.999 else 0.0,
            "recall": {k: round(v, 4) for k, v in recall.items()},
            "ndcg": {k: (round(v, 4) if v is not None else None) for k, v in dcg.items()},
            "relevant_count": len(gold), "retrieved_count": len(seen)}


def summarize_clause(rows):
    rows = [r["clause"] for r in rows if r.get("clause")]
    if not rows:
        return {}
    n = len(rows)
    out = {"total_queries": n, "ndcg_definition": NDCG_DEFINITION}
    out["MRR"] = round(sum(r["mrr"] for r in rows) / n, 4)
    out["Hit@20"] = round(sum(r["hit_any"] for r in rows) / n, 4)
    out["Rank1_Accuracy"] = round(sum(r["rank1"] for r in rows) / n, 4)
    out["R-Precision"] = round(sum(r["r_precision"] for r in rows) / n, 4)
    for k in [1, 3, 5, 10, 20]: out[f"Recall@{k}"] = round(sum(r["recall"][k] for r in rows) / n, 4)
    for k in NDCG_KS:
        out[f"Precision@{k}"] = round(sum(r["precision"][k] for r in rows) / n, 4)
        out[f"NDCG@{k}"], valid_n = average_valid(r["ndcg"][k] for r in rows)
        out[f"NDCG@{k}_valid_queries"] = valid_n
    return out


def summarize_law(rows):
    rows = [r["law"] for r in rows if r.get("law")]
    if not rows:
        return {}
    n = len(rows)
    out = {"total_queries": n, "ndcg_definition": NDCG_DEFINITION}
    out["MRR"] = round(sum(r["mrr"] for r in rows) / n, 4)
    out["Hit@20"] = round(sum(r["hit_any"] for r in rows) / n, 4)
    out["Rank1_Accuracy"] = round(sum(r["rank1"] for r in rows) / n, 4)
    for k in [1, 5, 10, 20]: out[f"Recall@{k}"] = round(sum(r["recall"][k] for r in rows) / n, 4)
    for k in NDCG_KS:
        out[f"NDCG@{k}"], valid_n = average_valid(r["ndcg"][k] for r in rows)
        out[f"NDCG@{k}_valid_queries"] = valid_n
    return out


def _fmt(v, width=8):
    """None（无有效查询）打印 n/a，避免格式化崩溃。"""
    return f"{'n/a':>{width}}" if v is None else f"{v:>{width}.4f}"


def print_tables(label, summaries, law_level):
    print(f"\n  --- {label} ---")
    if law_level:
        print(f"  {'策略':<26} {'MRR':>8} {'Hit@20':>8} {'Rank1':>8} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@20':>8} {'NDCG@20':>10}")
    else:
        print(f"  {'策略':<26} {'MRR':>8} {'Hit@20':>8} {'Rank1':>8} {'R-Prec':>8} {'R@5':>8} {'R@10':>8} {'R@20':>8} {'P@5':>8} {'NDCG@20':>10}")
    print(f"  {'-'*110}")
    for sk, s in summaries.items():
        if not s:
            print(f"  {sk:<26} {'n/a':>8}")
            continue
        if law_level:
            print(f"  {sk:<26} {_fmt(s['MRR'])} {_fmt(s['Hit@20'])} {_fmt(s['Rank1_Accuracy'])} "
                  f"{_fmt(s['Recall@1'])} {_fmt(s['Recall@5'])} {_fmt(s['Recall@10'])} "
                  f"{_fmt(s['Recall@20'])} {_fmt(s['NDCG@20'], 10)}")
        else:
            print(f"  {sk:<26} {_fmt(s['MRR'])} {_fmt(s['Hit@20'])} {_fmt(s['Rank1_Accuracy'])} "
                  f"{_fmt(s['R-Precision'])} {_fmt(s['Recall@5'])} {_fmt(s['Recall@10'])} "
                  f"{_fmt(s['Recall@20'])} {_fmt(s['Precision@5'])} {_fmt(s['NDCG@20'], 10)}")
    print(f"  NDCG 口径: {NDCG_DEFINITION}（IDCG@k 逐 k 截断；本仓库 NDCG_KS={list(NDCG_KS)}）")


# 策略名 → viz overall 键 / CSV 策略列取值（与历史文件保持一致）
STRATEGY_KEYS = {
    "仅BM25": ("bm25", "仅BM25检索"),
    "仅向量检索": ("vector", "纯向量检索"),
    "混合检索(BM25+向量+RRF)": ("hybrid", "混合检索(RRF)"),
    "混合检索+重排序": ("hybrid+reranker", "混合+Re-ranker"),
}
CSV_HEADER = ["strategy", "query_id", "query_text", "jurisdiction", "subset", "variant",
              "base_query", "clause_gold_empty", "relevant_count", "retrieved_count",
              "MRR", "R@5", "R@10", "R@20", "P@5",
              "law_relevant_count", "law_MRR", "law_R@5", "law_R@20", "law_NDCG@20",
              "clause_rank1", "law_rank1"]


def _csv_row(label, r):
    c = r.get("clause") or {}
    l = r.get("law") or {}
    return {
        "strategy": label, "query_id": r["query_id"], "query_text": r["query_text"],
        "jurisdiction": r["jurisdiction"], "subset": r["subset"],
        "variant": r["variant"], "base_query": r["base_query"],
        "clause_gold_empty": "1" if r.get("clause") is None else "0",
        "relevant_count": c.get("relevant_count", ""), "retrieved_count": c.get("retrieved_count", ""),
        "MRR": c.get("mrr", ""), "R@5": c["recall"][5] if c else "",
        "R@10": c["recall"][10] if c else "", "R@20": c["recall"][20] if c else "",
        "P@5": c["precision"][5] if c else "",
        "law_relevant_count": l.get("relevant_count", ""), "law_MRR": l.get("mrr", ""),
        "law_R@5": l["recall"][5] if l else "", "law_R@20": l["recall"][20] if l else "",
        "law_NDCG@20": l["ndcg"][20] if l else "",
        "clause_rank1": c.get("rank1", ""), "law_rank1": l.get("rank1", ""),
    }


def summarize_by_subset(rows):
    """按 template / legacy 分开汇总；legacy 仅 176 条有条款 gold。"""
    out = {}
    for subset in ("template", "legacy"):
        sub = [r for r in rows if r.get("subset") == subset]
        if not sub:
            continue
        out[subset] = {"law": summarize_law(sub), "clause": summarize_clause(sub)}
    return out


def save_outputs(strategy_rows, law_summaries, clause_summaries, prefix="", meta=None):
    """增量写出双层 overall 汇总与逐查询 CSV。"""
    viz_path = os.path.join(RESULTS_DIR, f"eval_viz_data{prefix}.json")
    viz_data = {}
    if os.path.exists(viz_path):
        viz_data = json.load(open(viz_path, encoding="utf-8"))
    overall_law = viz_data.get("overall_law", {})
    overall_clause = viz_data.get("overall_clause", {})
    for name, (key, _) in STRATEGY_KEYS.items():
        if law_summaries.get(name):
            overall_law[key] = law_summaries[name]
        if clause_summaries.get(name):
            overall_clause[key] = clause_summaries[name]
    viz_data["overall_law"] = overall_law
    viz_data["overall_clause"] = overall_clause
    by_subset = viz_data.get("by_subset", {})
    for name, (key, _) in STRATEGY_KEYS.items():
        if strategy_rows.get(name):
            by_subset[key] = summarize_by_subset(strategy_rows[name])
    viz_data["by_subset"] = by_subset
    viz_data["subset_definition"] = {
        "template": "template gold（276 变体，全部有条款 gold）",
        "legacy": "legacy 诊断 gold（320 变体；仅 176 条有条款 gold）",
    }
    viz_data["metric_semantics"] = {
        "Hit@20": "返回列表前 20 条中至少命中一条 gold；旧字段 Top1_Accuracy 的实际含义",
        "Rank1_Accuracy": "返回列表第 1 条即命中 gold（MRR==1.0）",
        "Top1_Accuracy": "已废弃：旧文件中的该字段等价于 Hit@20，不得当作 rank-1",
        "NDCG@20": NDCG_DEFINITION,
    }
    viz_data["ndcg_definition"] = NDCG_DEFINITION
    if meta:
        viz_data.setdefault("meta", {}).update(meta)
    json.dump(viz_data, open(viz_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  已保存: {viz_path}")

    csv_path = os.path.join(RESULTS_DIR, f"eval_per_query{prefix}.csv")
    labels = {label for _, label in STRATEGY_KEYS.values()}
    old_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            old_rows = [r for r in csv.DictReader(f) if r.get("strategy") not in labels]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for r in old_rows:
            w.writerow({k: r.get(k, "") for k in CSV_HEADER})
        for name, (_, label) in STRATEGY_KEYS.items():
            for r in strategy_rows.get(name, []):
                w.writerow(_csv_row(label, r))
    print(f"  已保存: {csv_path}")


def _describe(q):
    head, sep, tail = q.id.rpartition("_")
    variant = tail if (sep and tail in VARIANTS) else ""
    return variant


def main():
    ap = argparse.ArgumentParser(description="四策略对比评测（支持 v2 双层 gold）")
    ap.add_argument("--limit", type=int, default=0, help="仅跑前 N 条（0=全量），用于试跑")
    ap.add_argument("--ids", default="", help="仅跑指定 query id（逗号分隔），用于定点复核")
    ap.add_argument("--out-prefix", default="", help="输出文件后缀（试跑用 _pilot，避免覆盖正式结果）")
    ap.add_argument("--dataset", default=EVAL_DATASET_PATH,
                    help="gold 数据集路径；相对路径按 legal_rag/evals/datasets 解析（如 v2/eval_dataset_v2.json）")
    args = ap.parse_args()
    dataset_path = args.dataset if os.path.isabs(args.dataset) else os.path.join(DATASETS_DIR, args.dataset)

    import torch
    from services.retriever_service import RetrieverService
    from core.config import settings

    print("=" * 70)
    print("  四策略对比评测：BM25 / 向量 / 混合(RRF) / 混合+重排序")
    print("=" * 70)

    queries = load_queries(dataset_path)
    if args.ids:
        want = [x.strip() for x in args.ids.split(",") if x.strip()]
        got = {q.id: q for q in queries}
        missing = [x for x in want if x not in got]
        if missing:
            raise ValueError(f"--ids 中这些 id 不在数据集里: {missing}")
        queries = [got[x] for x in want]
    if args.limit:
        queries = queries[:args.limit]

    clause_empty = [q for q in queries if not q.relevant]
    empty_bases = sorted({q.base_query for q in clause_empty})
    base_ids = {q.base_query for q in queries}
    print(f"\n  数据集: {dataset_path}")
    print(f"  查询变体: {len(queries)}（基础查询: {len(base_ids)}）")
    print(f"  法规级 gold 有效: {sum(1 for q in queries if q.law_gold)}")
    print(f"  条款级 gold 有效: {sum(1 for q in queries if q.relevant)}，"
          f"空条款 gold: {len(clause_empty)} 条变体 / {len(empty_bases)} 条基础查询（单列，不进均值）")
    if empty_bases:
        print(f"  空条款 gold 基础 ID: {','.join(empty_bases)}")

    # Phase 1: BM25 + 向量 + 混合（只需 Retriever）
    print("\n  Phase 1: 运行 BM25 / 向量 / 混合...")
    settings.reranker_enabled = False
    retriever = RetrieverService()

    strategy_rows = {"仅BM25": [], "仅向量检索": [], "混合检索(BM25+向量+RRF)": []}
    elapsed = {k: 0.0 for k in ["仅BM25", "仅向量检索", "混合检索(BM25+向量+RRF)", "混合检索+重排序"]}

    def flush(name):
        """单个策略跑完立即落盘：中途失败时已完成的策略结果不丢。"""
        save_outputs({name: strategy_rows[name]},
                     {name: summarize_law(strategy_rows[name])},
                     {name: summarize_clause(strategy_rows[name])}, args.out_prefix)
        print(f"  [{name}] 已落盘 {len(strategy_rows[name])} 条")

    def row_of(q, ids):
        return {"query_id": q.id, "query_text": q.query,
                "jurisdiction": q.jurisdiction or "跨法域",
                "subset": q.subset, "variant": _describe(q), "base_query": q.base_query,
                "clause": compute_clause_metrics(ids, q.get_relevant_articles(), q),
                "law": compute_law_metrics([x[0] for x in ids], q.law_gold)}

    # 三个策略各自跑完整一轮，跑完一个写一个
    for name in ["仅BM25", "仅向量检索", "混合检索(BM25+向量+RRF)"]:
        for idx, q in enumerate(queries):
            jur = None if q.jurisdiction in (None, "", "跨法域") else normalize_jurisdiction(q.jurisdiction)
            t0 = time.perf_counter()
            if name == "仅BM25":
                try:
                    res = retriever.bm25_search(query=q.query, top_k=TOP_K, jurisdiction=jur)
                except Exception:
                    res = []
            elif name == "仅向量检索":
                res = retriever.search(query=q.query, top_k=TOP_K, jurisdiction=jur)
            else:
                res = retriever.hybrid_search(query=q.query, top_k=TOP_K, jurisdiction=jur)
            elapsed[name] += time.perf_counter() - t0
            ids = [(r["law_id"], r.get("article_number", "")) for r in res]
            strategy_rows[name].append(row_of(q, ids))
            if (idx + 1) % 100 == 0:
                print(f"    [{name}] {idx+1}/{len(queries)}")
        flush(name)

    # 释放 Retriever
    print("  释放 Retriever...")
    del retriever; gc.collect(); torch.cuda.empty_cache()

    # Phase 2: 混合+重排序（需要 Reranker）
    print("\n  Phase 2: 运行 混合+重排序...")
    settings.reranker_enabled = True
    retriever2 = RetrieverService()

    strategy_rows["混合检索+重排序"] = []
    for idx, q in enumerate(queries):
        jur = None if q.jurisdiction in (None, "", "跨法域") else normalize_jurisdiction(q.jurisdiction)
        t0 = time.perf_counter()
        rr_res = retriever2.hybrid_search(query=q.query, top_k=TOP_K, jurisdiction=jur)
        elapsed["混合检索+重排序"] += time.perf_counter() - t0
        rr_ids = [(r["law_id"], r.get("article_number", "")) for r in rr_res]
        strategy_rows["混合检索+重排序"].append(row_of(q, rr_ids))
        if (idx + 1) % 100 == 0:
            flush("混合检索+重排序")
            print(f"    [{idx+1}/{len(queries)}] 检查点已落盘")

        if (idx + 1) % 50 == 0:
            print(f"    [{idx+1}/{len(queries)}] 完成 MRR(law) so far: "
                  f"{sum((r['law'] or {}).get('mrr', 0) for r in strategy_rows['混合检索+重排序'])/(idx+1):.4f}")

    law_summaries = {n: summarize_law(rows) for n, rows in strategy_rows.items()}
    clause_summaries = {n: summarize_clause(rows) for n, rows in strategy_rows.items()}

    print("\n" + "=" * 70)
    print("  四策略对比结果")
    print("=" * 70)
    print_tables(f"法规级（全部 {len(queries)} 条变体，按 law_id 去重）", law_summaries, True)
    print_tables(f"条款级（仅条款 gold 非空，{sum(1 for q in queries if q.relevant)} 条）", clause_summaries, False)

    print("\n  NDCG 有效查询数（gold 为空记缺失，不参与均值）:")
    for name in law_summaries:
        ls, cs = law_summaries.get(name) or {}, clause_summaries.get(name) or {}
        print(f"  {name:<26} 法规级 " + " ".join(
            f"@{k}={ls.get(f'NDCG@{k}_valid_queries')}" for k in NDCG_KS) +
            "  条款级 " + " ".join(f"@{k}={cs.get(f'NDCG@{k}_valid_queries')}" for k in NDCG_KS))

    print("\n  增量提升（法规级 MRR）:")
    base = "仅BM25"
    for name in ["仅向量检索", "混合检索(BM25+向量+RRF)", "混合检索+重排序"]:
        if law_summaries.get(base) and law_summaries.get(name):
            d = law_summaries[name]["MRR"] - law_summaries[base]["MRR"]
            print(f"    {name:<26} vs {base}: MRR={d:+.4f}")

    print("\n  耗时（秒，含检索与精排，不含模型加载）:")
    total = sum(elapsed.values())
    for k, v in elapsed.items():
        print(f"    {k:<26} {v:>8.1f}  ({v/len(queries):.2f}s/条)" if len(queries) else f"    {k}: {v:.1f}")
    print(f"    {'合计':<26} {total:>8.1f}")

    meta = {"dataset": os.path.basename(dataset_path), "n_variants": len(queries),
            "n_clause_valid": sum(1 for q in queries if q.relevant),
            "n_clause_empty": len(clause_empty), "clause_empty_bases": empty_bases,
            "elapsed_seconds": {k: round(v, 1) for k, v in elapsed.items()},
            "elapsed_total": round(total, 1)}
    save_outputs(strategy_rows, law_summaries, clause_summaries, args.out_prefix, meta)

    print("\n  完成！")


if __name__ == "__main__":
    main()
