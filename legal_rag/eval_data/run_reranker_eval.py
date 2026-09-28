"""
第三阶段：单独运行 Re-ranker 策略（手动加载模型绕开 mmap）
=====================================================
使用 torch.load 加载权重（不经过 safetensors mmap，避免页面文件不足）
"""
import sys, os, json, math, gc
from collections import defaultdict
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 本脚本与数据集同处 eval_data/ 目录，BASE_DIR 已是该目录，不能再拼一层 "eval_data"。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DATASET_PATH = os.path.join(BASE_DIR, "eval_dataset_clause.json")
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
    for k in ["mrr", "top1_acc", "r_precision"]:
        out[{"mrr": "MRR", "top1_acc": "Top1_Accuracy", "r_precision": "R-Precision"}[k]] = round(sum(r[k] for r in results) / n, 4)
    for k in [1, 3, 5, 10, 20]: out[f"Recall@{k}"] = round(sum(r["recall"][k] for r in results) / n, 4)
    for k in [5, 10, 20]:
        out[f"Precision@{k}"] = round(sum(r["precision"][k] for r in results) / n, 4)
        out[f"NDCG@{k}"] = round(sum(r["ndcg"][k] for r in results) / n, 4)
    return out


def load_reranker_direct():
    """加载 reranker（页面文件已扩大，可直接 from_pretrained）"""
    import torch
    from core.config import settings
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_path = settings.reranker_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  加载 reranker: {model_path} -> {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    print(f"  模型加载完成: {sum(p.numel() for p in model.parameters())/1e6:.0f}M 参数")
    return tokenizer, model, device


def batch_rerank(query, candidates, tokenizer, model, device, top_k=20):
    """使用已加载的模型重排"""
    import torch

    pairs = []
    valid = []
    for doc in candidates:
        content = doc.get("content", "")
        if not content: continue
        pairs.append([query, content[:2000]])
        valid.append(doc)

    if not pairs: return candidates[:top_k]

    all_scores = []
    batch_size = 8
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=2048, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().tolist()
            all_scores.extend(scores if isinstance(scores, list) else [scores])

    for doc, score in zip(valid, all_scores):
        doc["rerank_score"] = round(score, 4)
    valid.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return valid[:top_k]


def main():
    import torch

    print("=" * 60)
    print("  P1b: Re-ranker 策略评估（手动加载）")
    print("=" * 60)

    # 1. 加载数据集
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
    queries = [EvalQuery(id=item["id"], query=item["query"],
                         jurisdiction=item.get("jurisdiction"),
                         relevant=item.get("relevant", [])) for item in items]
    active = [q for q in queries if q.relevant]
    print(f"\n  数据集: {len(queries)}（有 gold: {len(active)}）")

    # Phase 1: 生成混合检索候选
    print("\n  Phase 1: 生成混合检索候选...")
    from services.retriever_service import RetrieverService
    from core.config import settings

    settings.reranker_enabled = False
    retriever = RetrieverService()
    candidates_map = {}
    for q in active:
        jur = None if q.jurisdiction == "跨法域" else q.jurisdiction
        results = retriever.hybrid_search(query=q.query, top_k=TOP_K, jurisdiction=jur)
        candidates_map[q.id] = [(r["law_id"], r.get("article_number", "")) for r in results]

    # 释放 Retriever
    print("  释放 Retriever...")
    del retriever; gc.collect(); torch.cuda.empty_cache()
    print(f"  显存: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # Phase 2: 加载 Reranker
    print("\n  Phase 2: 加载 Re-ranker...")
    tokenizer, model, device = load_reranker_direct()

    # Phase 3: 重排 + 评分
    print("\n  Phase 3: 重排 + 评分...")
    from repositories.elasticsearch import ElasticsearchRepository
    repo = ElasticsearchRepository()

    all_results = []
    for idx, q in enumerate(active):
        candidates = candidates_map[q.id]
        if not candidates:
            all_results.append(compute_metrics([], q.get_relevant_articles(),
                                                list(q.get_relevant_articles()), q))
            continue

        # 补齐 text
        enriched = []
        for lid, art in candidates:
            text = repo.get_article_full_text(lid, art)[:2000]
            enriched.append({"law_id": lid, "article_number": art, "content": text or ""})

        # 重排
        reranked = batch_rerank(q.query, enriched, tokenizer, model, device, TOP_K)
        retrieved_ids = [(r["law_id"], r["article_number"]) for r in reranked]

        gold_set = q.get_relevant_articles()
        gold_articles = list(gold_set)
        metrics = compute_metrics(retrieved_ids, gold_set, gold_articles, q)
        metrics.update({"query_id": q.id, "query_text": q.query[:100],
                       "jurisdiction": q.jurisdiction or "跨法域",
                       "variant": q.id.split("_")[-1] if "_" in q.id else "",
                       "base_query": q.id.rsplit("_", 1)[0] if "_" in q.id else q.id})
        all_results.append(metrics)

        if (idx + 1) % 40 == 0:
            print(f"    [{idx+1}/{len(active)}] MRR so far: {sum(r['mrr'] for r in all_results)/len(all_results):.4f}")

    # 汇总
    summary = compute_summary(all_results)
    print(f"\n  === 混合+Re-ranker 结果 (n={summary['total_queries']}) ===")
    print(f"  {'MRR':>8} {'Top1':>8} {'R-Prec':>8} {'R@5':>8} {'R@10':>8} {'R@20':>8} {'P@5':>8} {'NDCG@10':>10}")
    print(f"  {'-'*76}")
    print(f"  {summary['MRR']:>8.4f} {summary['Top1_Accuracy']:>8.4f} {summary['R-Precision']:>8.4f} "
          f"{summary['Recall@5']:>8.4f} {summary['Recall@10']:>8.4f} {summary['Recall@20']:>8.4f} "
          f"{summary['Precision@5']:>8.4f} {summary['NDCG@10']:>10.4f}")

    # 追加到 viz JSON
    viz_path = os.path.join(BASE_DIR, "eval_viz_data.json")
    if os.path.exists(viz_path):
        viz_data = json.load(open(viz_path, encoding="utf-8"))
        viz_data["overall"]["hybrid+reranker"] = summary
        json.dump(viz_data, open(viz_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n  已保存: {viz_path}")

    # 追加到 CSV
    csv_path = os.path.join(BASE_DIR, "eval_per_query.csv")
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for r in all_results:
                w.writerow(["混合+Re-ranker", r["query_id"], r["query_text"],
                           r["jurisdiction"], r["variant"], r["base_query"],
                           r["relevant_count"], r["retrieved_count"],
                           r["mrr"], r["recall"][5], r["recall"][10],
                           r["recall"][20], r["precision"][5]])
        print(f"  已保存: {csv_path}")

    print("\n  完成！")


if __name__ == "__main__":
    main()
