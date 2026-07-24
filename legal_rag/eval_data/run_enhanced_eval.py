"""
P0+P1: 增强评估脚本（独立版本，不依赖 evaluate.py）
===================================================
"""
import sys, os
import json, math, csv
from collections import defaultdict
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DATASET_PATH = os.path.join(BASE_DIR, "eval_data", "eval_dataset_clause.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "eval_data")
TOP_K = 20
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 简单的 EvalQuery 替代
# ============================================================
class EvalQuery:
    def __init__(self, id="", query="", jurisdiction=None, relevant=None):
        self.id = id
        self.query = query
        self.jurisdiction = jurisdiction
        self.relevant = relevant or []

    def get_relevant_articles(self) -> set:
        """返回 {(law_id, article)}"""
        return {(r["law_id"], r.get("article", "")) for r in self.relevant}

    def get_gain(self, law_id, article):
        for r in self.relevant:
            if r["law_id"] == law_id and r.get("article", "") == article:
                rel = r.get("relevance", 1)
                return (2 ** rel) - 1
        return 0


# ============================================================
# 策略函数（独立实现）
# ============================================================
def strategy_bm25(retriever, query_q, top_k=20):
    """仅 BM25 全文检索"""
    jur = None if getattr(query_q, 'jurisdiction', None) == "跨法域" else getattr(query_q, 'jurisdiction', None)
    results = retriever.bm25_search(query=query_q.query, top_k=top_k, jurisdiction=jur)
    return [(r["law_id"], r.get("article_number", "")) for r in results]


def strategy_vector(retriever, query_q, top_k=20):
    """纯向量检索"""
    from core.config import settings
    settings.reranker_enabled = False
    jur = None if getattr(query_q, 'jurisdiction', None) == "跨法域" else getattr(query_q, 'jurisdiction', None)
    results = retriever.search(query=query_q.query, top_k=top_k, jurisdiction=jur)
    return [(r["law_id"], r.get("article_number", "")) for r in results]


def strategy_hybrid(retriever, query_q, top_k=20):
    """混合检索(RRF) - 含路由检测"""
    from core.config import settings
    settings.reranker_enabled = False
    jur = None if getattr(query_q, 'jurisdiction', None) == "跨法域" else getattr(query_q, 'jurisdiction', None)
    results = retriever.hybrid_search(query=query_q.query, top_k=top_k, jurisdiction=jur)
    return [(r["law_id"], r.get("article_number", "")) for r in results]


def strategy_hybrid_reranker(retriever, query_q, top_k=20):
    """混合+Re-ranker - 含路由检测 + 候选窗口扩展"""
    from core.config import settings
    settings.reranker_enabled = True
    jur = None if getattr(query_q, 'jurisdiction', None) == "跨法域" else getattr(query_q, 'jurisdiction', None)
    results = retriever.hybrid_search(query=query_q.query, top_k=top_k, jurisdiction=jur)
    return [(r["law_id"], r.get("article_number", "")) for r in results]


STRATEGIES = {
    "bm25": {"name": "仅BM25检索", "run": strategy_bm25},
    "vector": {"name": "纯向量检索", "run": strategy_vector},
    "hybrid": {"name": "混合检索(RRF)", "run": strategy_hybrid},
    "hybrid+reranker": {"name": "混合+Re-ranker", "run": strategy_hybrid_reranker},
}


# ============================================================
# 指标计算
# ============================================================
def compute_metrics(retrieved_ids, gold_set, gold_articles, eval_query):
    """计算单条 query 的完整指标"""
    # MRR
    mrr = 0.0
    for i, (lid, art) in enumerate(retrieved_ids):
        if (lid, art) in gold_set:
            mrr = 1.0 / (i + 1)
            break

    # Top-1 Accuracy
    top1_acc = 1.0 if mrr > 0 else 0.0

    # Recall & Precision
    rel_count = len(gold_set)
    recall = {}
    precision = {}
    for k in [1, 3, 5, 10, 20]:
        topk = retrieved_ids[:k]
        hits = sum(1 for lid, art in topk if (lid, art) in gold_set)
        recall[k] = hits / max(rel_count, 1)
        precision[k] = hits / max(k, 1)

    # R-Precision
    R = min(rel_count, 20)
    r_hits = sum(1 for lid, art in retrieved_ids[:R] if (lid, art) in gold_set)
    r_prec = r_hits / max(R, 1)

    # NDCG
    gains = sorted([eval_query.get_gain(lid, art) for lid, art in gold_articles], reverse=True)
    ideal_dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:20]))
    dcg = {}
    for k in [5, 10, 20]:
        dcg_k = 0.0
        for i, (lid, art) in enumerate(retrieved_ids[:k]):
            gain = eval_query.get_gain(lid, art)
            dcg_k += gain / math.log2(i + 2)
        dcg[k] = dcg_k / max(ideal_dcg, 1e-10) if k <= len(gains) else 0.0

    return {
        "mrr": round(mrr, 4),
        "top1_acc": round(top1_acc, 4),
        "r_precision": round(r_prec, 4),
        "recall": {k: round(v, 4) for k, v in recall.items()},
        "precision": {k: round(v, 4) for k, v in precision.items()},
        "ndcg": {k: round(v, 4) for k, v in dcg.items()},
        "relevant_count": rel_count,
        "retrieved_count": len(retrieved_ids),
    }


def compute_summary(results: List[Dict]) -> Dict:
    """汇总多条 query 的平均指标"""
    if not results:
        return {}
    n = len(results)
    s = {}
    for key in ["mrr", "top1_acc", "r_precision"]:
        s[key.capitalize().replace("_", "-")] = round(sum(r[key] for r in results) / n, 4)

    # Manually build the dict to handle nested keys
    out = {"total_queries": n}
    
    # Calculate averages
    mrr_avg = sum(r["mrr"] for r in results) / n
    top1_avg = sum(r["top1_acc"] for r in results) / n
    rprec_avg = sum(r["r_precision"] for r in results) / n
    out["MRR"] = round(mrr_avg, 4)
    out["Top1_Accuracy"] = round(top1_avg, 4)
    out["R-Precision"] = round(rprec_avg, 4)

    for k in [1, 3, 5, 10, 20]:
        out[f"Recall@{k}"] = round(sum(r["recall"][k] for r in results) / n, 4)
    for k in [5, 10, 20]:
        out[f"Precision@{k}"] = round(sum(r["precision"][k] for r in results) / n, 4)
        out[f"NDCG@{k}"] = round(sum(r["ndcg"][k] for r in results) / n, 4)

    return out


def print_summary_block(label: str, results: List[Dict]):
    if not results:
        print(f"\n  [{label}] 无数据")
        return
    s = compute_summary(results)
    print(f"\n  --- {label} (n={s['total_queries']}) ---")
    header = f"  {'MRR':>8} {'Top1':>8} {'R-Prec':>8} {'R@5':>8} {'R@10':>8} {'R@20':>8} {'P@5':>8} {'NDCG@10':>10}"
    print(header)
    print(f"  {'-'*76}")
    print(f"  {s['MRR']:>8.4f} {s['Top1_Accuracy']:>8.4f} {s['R-Precision']:>8.4f} "
          f"{s['Recall@5']:>8.4f} {s['Recall@10']:>8.4f} {s['Recall@20']:>8.4f} "
          f"{s['Precision@5']:>8.4f} {s['NDCG@10']:>10.4f}")


def print_comparison_table(all_summaries: Dict):
    print(f"\n  {'策略':<22} {'MRR':>8} {'R@5':>8} {'R@10':>8} {'R@20':>8} {'P@5':>8} {'NDCG@10':>10} {'NDCG@20':>10}")
    print(f"  {'-'*84}")
    for sk, s in all_summaries.items():
        name = STRATEGIES[sk]["name"]
        print(f"  {name:<22} {s['MRR']:>8.4f} {s['Recall@5']:>8.4f} {s['Recall@10']:>8.4f} "
              f"{s['Recall@20']:>8.4f} {s['Precision@5']:>8.4f} {s['NDCG@10']:>10.4f} {s['NDCG@20']:>10.4f}")


def export_per_query_csv(strategy_names, strategy_results, output_path):
    rows = []
    for sk, name in strategy_names.items():
        results = strategy_results[sk]
        for r in results:
            rows.append({
                "strategy": name,
                "query_id": r["query_id"],
                "query_text": r["query_text"],
                "jurisdiction": r["jurisdiction"],
                "variant": r["variant"],
                "base_query": r["base_query"],
                "relevant_count": r["relevant_count"],
                "retrieved_count": r["retrieved_count"],
                "MRR": r["mrr"],
                "R@5": r["recall"][5],
                "R@10": r["recall"][10],
                "R@20": r["recall"][20],
                "P@5": r["precision"][5],
            })
    if not rows:
        return
    fieldnames = ["strategy", "query_id", "query_text", "jurisdiction", "variant",
                  "base_query", "relevant_count", "retrieved_count",
                  "MRR", "R@5", "R@10", "R@20", "P@5"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV: {output_path} ({len(rows)} 行)")


def export_viz_json(all_summaries, jur_summaries, var_summaries, output_path):
    data = {
        "overall": all_summaries,
        "by_jurisdiction": {j: s for j, s in jur_summaries.items()},
        "by_variant": {v: s for v, s in var_summaries.items()},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Viz: {output_path}")


# ============================================================
# 主流程
# ============================================================
def main():
    from services.retriever_service import RetrieverService

    print("=" * 60)
    print("  P0+P1: 条款级评估")
    print("=" * 60)

    # 1. 加载数据
    if not os.path.exists(EVAL_DATASET_PATH):
        print(f"\n  [ERROR] 未找到: {EVAL_DATASET_PATH}")
        return

    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
    queries = []
    for item in items:
        q = EvalQuery(id=item["id"], query=item["query"],
                       jurisdiction=item.get("jurisdiction"),
                       relevant=item.get("relevant", []))
        queries.append(q)

    active = [q for q in queries if q.relevant]
    print(f"\n  数据集: {len(queries)}（有 gold: {len(active)}）")
    if not active:
        return

    retriever = RetrieverService()
    strategy_keys = ["bm25", "vector", "hybrid", "hybrid+reranker"]

    # 2. 运行策略
    all_strategy_results = {}
    all_summaries = {}
    strategy_names = {k: STRATEGIES[k]["name"] for k in strategy_keys}

    for sk in strategy_keys:
        print(f"\n  >>> {strategy_names[sk]}")
        results = []
        for q in active:
            gold_set = q.get_relevant_articles()
            gold_articles = list(gold_set)

            retrieved_ids = STRATEGIES[sk]["run"](retriever, q, TOP_K)
            metrics = compute_metrics(retrieved_ids, gold_set, gold_articles, q)

            var = q.id.split("_")[-1] if "_" in q.id else ""
            base = q.id.rsplit("_", 1)[0] if "_" in q.id else q.id
            jur = q.jurisdiction or "跨法域"

            results.append({
                **metrics,
                "query_id": q.id,
                "query_text": q.query[:100],
                "jurisdiction": jur,
                "variant": var,
                "base_query": base,
                "strategy_key": sk,
            })
        all_strategy_results[sk] = results
        all_summaries[sk] = compute_summary(results)
        print(f"    完成: MRR={all_summaries[sk]['MRR']:.4f}")

    # 3. 总体对比
    print("\n  === 总体对比 ===")
    print_comparison_table(all_summaries)

    print("\n  === 额外指标 ===")
    print(f"  {'策略':<22} {'Top1-Acc':>10} {'R-Precision':>12}")
    print(f"  {'-'*46}")
    for sk in strategy_keys:
        s = all_summaries[sk]
        print(f"  {strategy_names[sk]:<22} {s['Top1_Accuracy']:>10.4f} {s['R-Precision']:>12.4f}")

    # 4. 按法域细分
    print("\n  === 按法域细分 (混合检索) ===")
    jur_summaries = {}
    jur_groups = defaultdict(list)
    for r in all_strategy_results["hybrid"]:
        jur_groups[r["jurisdiction"]].append(r)
    for jur in sorted(jur_groups.keys()):
        s = compute_summary(jur_groups[jur])
        jur_summaries[jur] = s
        print_summary_block(jur, jur_groups[jur])

    # 5. 按查询变体细分
    print("\n  === 按查询变体细分 (混合检索) ===")
    var_summaries = {}
    var_groups = defaultdict(list)
    var_names = {"A": "Keyword(zh)", "B": "NL(zh)", "C": "Mixed(zh-en)", "D": "Pure(en)"}
    for r in all_strategy_results["hybrid"]:
        var_groups[r["variant"]].append(r)
    for var in sorted(var_groups.keys()):
        vn = var_names.get(var, var)
        s = compute_summary(var_groups[var])
        var_summaries[var] = s
        print_summary_block(f"{vn}", var_groups[var])

    # 6. 导出
    export_per_query_csv(strategy_names, all_strategy_results,
                         os.path.join(OUTPUT_DIR, "eval_per_query.csv"))
    export_viz_json(all_summaries, jur_summaries, var_summaries,
                    os.path.join(OUTPUT_DIR, "eval_viz_data.json"))

    print("\n" + "=" * 60)
    print("  完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
