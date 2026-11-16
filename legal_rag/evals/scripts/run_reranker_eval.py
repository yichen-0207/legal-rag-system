"""第三阶段：单独运行 Re-ranker 策略（Phase 1 先生成混合检索候选，Phase 2 再加载模型重排，避免显存峰值）。

与 run_four_strategies.py 共用同一套数据集解析、指标与落盘逻辑，避免两处口径分叉。
"""
import os, json, gc, argparse, time

from _bootstrap import DATASETS_DIR
# NDCG 口径与计算统一在 eval_metrics 中实现（逐 k 截断；与 v1 历史 @5/@10 不可直接比较）
from eval_metrics import NDCG_DEFINITION, NDCG_KS
from core.jurisdiction import normalize_jurisdiction
from run_four_strategies import (TOP_K, compute_clause_metrics, compute_law_metrics,
                                 load_queries, save_outputs, summarize_clause,
                                 summarize_law, _describe)

STRATEGY_NAME = "混合检索+重排序"


def load_reranker_direct():
    """加载 reranker（直接 from_pretrained）。"""
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
    ap = argparse.ArgumentParser(description="独立精排评测（支持 v2 双层 gold）")
    ap.add_argument("--limit", type=int, default=0, help="仅跑前 N 条（0=全量）")
    ap.add_argument("--ids", default="", help="仅跑指定 query id（逗号分隔）")
    ap.add_argument("--out-prefix", default="", help="输出文件后缀，避免覆盖正式结果")
    ap.add_argument("--dataset", required=True, help="gold 数据集路径（相对路径按 evals/datasets 解析）")
    args = ap.parse_args()
    dataset_path = args.dataset if os.path.isabs(args.dataset) else os.path.join(DATASETS_DIR, args.dataset)

    import torch

    print("=" * 60)
    print("  P1b: Re-ranker 策略评估（手动加载）")
    print("=" * 60)

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
    clause_valid = sum(1 for q in queries if q.relevant)
    print(f"\n  数据集: {dataset_path}")
    print(f"  查询变体: {len(queries)}（法规级 gold {sum(1 for q in queries if q.law_gold)}，"
          f"条款级 gold {clause_valid}，空条款 gold {len(queries) - clause_valid}）")
    print(f"  NDCG 口径: {NDCG_DEFINITION}（IDCG@k 逐 k 截断；NDCG_KS={list(NDCG_KS)}）")

    # Phase 1: 生成混合检索候选
    print("\n  Phase 1: 生成混合检索候选...")
    from services.retriever_service import RetrieverService
    from core.config import settings

    settings.reranker_enabled = False
    retriever = RetrieverService()
    candidates_map = {}
    phase1_sec = 0.0
    for q in queries:
        jur = None if q.jurisdiction in (None, "", "跨法域") else normalize_jurisdiction(q.jurisdiction)
        t0 = time.perf_counter()
        results = retriever.hybrid_search(query=q.query, top_k=TOP_K, jurisdiction=jur)
        phase1_sec += time.perf_counter() - t0
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

    rows = []
    rerank_sec = 0.0
    for idx, q in enumerate(queries):
        t0 = time.perf_counter()
        candidates = candidates_map[q.id]
        retrieved_ids = []
        if candidates:
            # 补齐 text
            enriched = []
            for lid, art in candidates:
                text = repo.get_article_full_text(lid, art)[:2000]
                enriched.append({"law_id": lid, "article_number": art, "content": text or ""})
            # 重排
            reranked = batch_rerank(q.query, enriched, tokenizer, model, device, TOP_K)
            retrieved_ids = [(r["law_id"], r["article_number"]) for r in reranked]
        rerank_sec += time.perf_counter() - t0

        rows.append({
            "query_id": q.id, "query_text": q.query[:100],
            "jurisdiction": q.jurisdiction or "跨法域",
            "subset": q.subset, "variant": _describe(q), "base_query": q.base_query,
            "clause": compute_clause_metrics(retrieved_ids, q.get_relevant_articles(), q),
            "law": compute_law_metrics([x[0] for x in retrieved_ids], q.law_gold),
        })

        if (idx + 1) % 40 == 0:
            avg = sum((r["law"] or {}).get("mrr", 0) for r in rows) / len(rows)
            print(f"    [{idx+1}/{len(rows)}] MRR(law) so far: {avg:.4f}")
        if (idx + 1) % 100 == 0:
            # 检查点落盘：进程中途被杀也只丢最近 100 条
            save_outputs({STRATEGY_NAME: rows}, {STRATEGY_NAME: summarize_law(rows)},
                         {STRATEGY_NAME: summarize_clause(rows)}, args.out_prefix)
            print(f"    [{idx+1}/{len(rows)}] 检查点已落盘")

    law_summary = summarize_law(rows)
    clause_summary = summarize_clause(rows)

    print(f"\n  === 混合+Re-ranker 结果 ===")
    if law_summary:
        print(f"  法规级 (n={law_summary['total_queries']}): MRR={law_summary['MRR']:.4f} "
              f"Hit@20={law_summary['Hit@20']:.4f} Rank1={law_summary['Rank1_Accuracy']:.4f} "
              f"R@5={law_summary['Recall@5']:.4f} "
              f"R@20={law_summary['Recall@20']:.4f} NDCG@20={law_summary['NDCG@20']:.4f}")
        print("    NDCG 有效条数: " + " ".join(
            f"@{k}={law_summary[f'NDCG@{k}_valid_queries']}" for k in NDCG_KS))
    if clause_summary:
        print(f"  条款级 (n={clause_summary['total_queries']}): MRR={clause_summary['MRR']:.4f} "
              f"Hit@20={clause_summary['Hit@20']:.4f} Rank1={clause_summary['Rank1_Accuracy']:.4f} "
              f"P@5={clause_summary['Precision@5']:.4f} "
              f"R@20={clause_summary['Recall@20']:.4f} NDCG@20={clause_summary['NDCG@20']:.4f}")
        print("    NDCG 有效条数: " + " ".join(
            f"@{k}={clause_summary[f'NDCG@{k}_valid_queries']}" for k in NDCG_KS))

    meta = {"rerank_dataset": os.path.basename(dataset_path),
            "rerank_phase1_seconds": round(phase1_sec, 1),
            "rerank_phase3_seconds": round(rerank_sec, 1)}
    save_outputs({STRATEGY_NAME: rows}, {STRATEGY_NAME: law_summary},
                 {STRATEGY_NAME: clause_summary}, args.out_prefix, meta)
    print("\n  完成！")


if __name__ == "__main__":
    main()
