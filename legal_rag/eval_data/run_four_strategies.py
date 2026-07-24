"""
四策略对比评测：仅BM25 / 仅向量 / 混合(RRF) / 混合+重排序
==========================================================
"""
import sys, os, json, math, gc
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DATASET_PATH = os.path.join(BASE_DIR, "eval_data", "eval_dataset_clause.json")
TOP_K = 20


class EvalQuery:
    def __init__(self, id="", query="", jurisdiction=None, relevant=None):
        self.id = id; self.query = query; self.jurisdiction = jurisdiction; self.relevant = relevant or []
    def get_relevant_articles(self): return {(r["law_id"], r.get("article", "")) for r in self.relevant}
    def get_gain(self, law_id, article):
        for r in self.relevant:
            if r["law_id"] == law_id and r.get("article", "") == article:
                return (2 ** r.get("relevance", 1)) - 1
        return 0


def compute_metrics(retrieved_ids, gold_set, gold_articles, eval_query):
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
    gains = sorted([eval_query.get_gain(lid, art) for lid, art in gold_articles], reverse=True)
    ideal_dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:20]))
    dcg = {}
    for k in [5, 10, 20]:
        dcg_k = sum(eval_query.get_gain(lid, art) / math.log2(i + 2) for i, (lid, art) in enumerate(retrieved_ids[:k]))
        dcg[k] = dcg_k / max(ideal_dcg, 1e-10) if k <= len(gains) else 0.0
    return {"mrr": round(mrr, 4), "top1_acc": 1.0 if mrr > 0 else 0.0,
            "r_precision": round(r_prec, 4),
            "recall": {k: round(v, 4) for k, v in recall.items()},
            "precision": {k: round(v, 4) for k, v in precision.items()},
            "ndcg": {k: round(v, 4) for k, v in dcg.items()},
            "relevant_count": rel_count, "retrieved_count": len(retrieved_ids)}


def compute_summary(results):
    if not results: return {}
    n = len(results)
    out = {"total_queries": n}
    out["MRR"] = round(sum(r["mrr"] for r in results) / n, 4)
    out["Top1_Accuracy"] = round(sum(r["top1_acc"] for r in results) / n, 4)
    out["R-Precision"] = round(sum(r["r_precision"] for r in results) / n, 4)
    for k in [1, 3, 5, 10, 20]: out[f"Recall@{k}"] = round(sum(r["recall"][k] for r in results) / n, 4)
    for k in [5, 10, 20]:
        out[f"Precision@{k}"] = round(sum(r["precision"][k] for r in results) / n, 4)
        out[f"NDCG@{k}"] = round(sum(r["ndcg"][k] for r in results) / n, 4)
    return out


def print_comparison_table(all_summaries):
    print(f"\n  {'策略':<26} {'MRR':>8} {'Top1':>8} {'R-Prec':>8} {'R@5':>8} {'R@10':>8} {'R@20':>8} {'P@5':>8} {'NDCG@20':>10}")
    print(f"  {'-'*94}")
    for sk, s in all_summaries.items():
        print(f"  {sk:<26} {s['MRR']:>8.4f} {s['Top1_Accuracy']:>8.4f} {s['R-Precision']:>8.4f} "
              f"{s['Recall@5']:>8.4f} {s['Recall@10']:>8.4f} {s['Recall@20']:>8.4f} "
              f"{s['Precision@5']:>8.4f} {s['NDCG@20']:>10.4f}")


def main():
    import torch
    from services.retriever_service import RetrieverService
    from core.config import settings

    print("=" * 70)
    print("  四策略对比评测：BM25 / 向量 / 混合(RRF) / 混合+重排序")
    print("=" * 70)

    # 加载数据集
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
    queries = [EvalQuery(id=item["id"], query=item["query"],
                         jurisdiction=item.get("jurisdiction"),
                         relevant=item.get("relevant", [])) for item in items]
    active = [q for q in queries if q.relevant]
    print(f"\n  数据集: {len(queries)}（有 gold: {len(active)}）")

    # Phase 1: BM25 + 向量 + 混合（只需 Retriever）
    print("\n  Phase 1: 运行 BM25 / 向量 / 混合...")
    settings.reranker_enabled = False
    retriever = RetrieverService()

    # 收集每种策略的结果
    strategy_results = {
        "仅BM25": [],
        "仅向量检索": [],
        "混合检索(BM25+向量+RRF)": [],
    }

    for q in active:
        jur = None if q.jurisdiction == "跨法域" else q.jurisdiction
        gold_set = q.get_relevant_articles()
        gold_articles = list(gold_set)

        # BM25
        try:
            bm25_res = retriever.bm25_search(query=q.query, top_k=TOP_K, jurisdiction=jur)
            bm25_ids = [(r["law_id"], r.get("article_number", "")) for r in bm25_res]
        except Exception:
            bm25_ids = []
        m = compute_metrics(bm25_ids, gold_set, gold_articles, q)
        m.update({"query_id": q.id, "jurisdiction": q.jurisdiction or "跨法域"})
        strategy_results["仅BM25"].append(m)

        # 向量
        vec_res = retriever.search(query=q.query, top_k=TOP_K, jurisdiction=jur)
        vec_ids = [(r["law_id"], r.get("article_number", "")) for r in vec_res]
        m = compute_metrics(vec_ids, gold_set, gold_articles, q)
        m.update({"query_id": q.id, "jurisdiction": q.jurisdiction or "跨法域"})
        strategy_results["仅向量检索"].append(m)

        # 混合
        hybrid_res = retriever.hybrid_search(query=q.query, top_k=TOP_K, jurisdiction=jur)
        hybrid_ids = [(r["law_id"], r.get("article_number", "")) for r in hybrid_res]
        m = compute_metrics(hybrid_ids, gold_set, gold_articles, q)
        m.update({"query_id": q.id, "jurisdiction": q.jurisdiction or "跨法域"})
        strategy_results["混合检索(BM25+向量+RRF)"].append(m)

    # 释放 Retriever
    print("  释放 Retriever...")
    del retriever; gc.collect(); torch.cuda.empty_cache()

    # Phase 2: 混合+重排序（需要 Reranker）
    print("\n  Phase 2: 运行 混合+重排序...")
    settings.reranker_enabled = True
    retriever2 = RetrieverService()

    strategy_results["混合检索+重排序"] = []
    for q in active:
        jur = None if q.jurisdiction == "跨法域" else q.jurisdiction
        gold_set = q.get_relevant_articles()
        gold_articles = list(gold_set)

        rr_res = retriever2.hybrid_search(query=q.query, top_k=TOP_K, jurisdiction=jur)
        rr_ids = [(r["law_id"], r.get("article_number", "")) for r in rr_res]
        m = compute_metrics(rr_ids, gold_set, gold_articles, q)
        m.update({"query_id": q.id, "jurisdiction": q.jurisdiction or "跨法域"})
        strategy_results["混合检索+重排序"].append(m)

    # 3. 汇总
    print("\n" + "=" * 70)
    print("  四策略对比结果")
    print("=" * 70)
    all_summaries = {}
    for name, results in strategy_results.items():
        all_summaries[name] = compute_summary(results)
    print_comparison_table(all_summaries)

    # 4. 输出增量对比
    print("\n  === 增量提升 === ")
    base = "仅BM25"
    for name in ["仅向量检索", "混合检索(BM25+向量+RRF)", "混合检索+重排序"]:
        delta_mrr = (all_summaries[name]["MRR"] - all_summaries[base]["MRR"]) / max(all_summaries[base]["MRR"], 1e-10) * 100
        delta_r20 = (all_summaries[name]["Recall@20"] - all_summaries[base]["Recall@20"]) / max(all_summaries[base]["Recall@20"], 1e-10) * 100
        delta_p5 = (all_summaries[name]["Precision@5"] - all_summaries[base]["Precision@5"]) / max(all_summaries[base]["Precision@5"], 1e-10) * 100
        print(f"  {name:<28} vs {base:<12}: MRR={delta_mrr:+.1f}%  R@20={delta_r20:+.1f}%  P@5={delta_p5:+.1f}%")

    print(f"  {'混合检索+重排序':<28} vs {'仅向量检索':<12}: MRR={all_summaries['混合检索+重排序']['MRR']-all_summaries['仅向量检索']['MRR']:+.4f}")

    # 5. 保存
    viz_path = os.path.join(BASE_DIR, "eval_data", "eval_viz_data.json")
    if os.path.exists(viz_path):
        viz_data = json.load(open(viz_path, encoding="utf-8"))
        viz_data["overall_four"] = all_summaries
        json.dump(viz_data, open(viz_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n  已保存: {viz_path}")

    print("\n  完成！")


if __name__ == "__main__":
    main()
