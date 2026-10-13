"""多跳 B（查询分解）受影响查询的 A/B 快速回归。

目的：验证开启查询分解后，会走分解路径的那部分查询是否发生指标退化。

做法：
  - 仅跑 hybrid+reranker 策略（分解只作用于该路径）；
  - 关闭 LLM 兜底 → 规则切分确定、可复现，且不引入网络波动；
  - 关闭检索缓存 → 两次跑互不污染；
  - 先筛出「关 LLM 兜底时 decompose_query 非空」的查询，作为受影响集合；
  - 同一 retriever 分别以 multihop_decompose_enabled=False/True 各跑一遍，
    对比 MRR / R@10 / NDCG@10 等，并逐条给出变好/变差/持平。

用法（在 backend 容器内，工作目录 /app）：
    python eval_data/run_multihop_ab.py
"""
import importlib.util
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/app")
sys.path.insert(0, BASE_DIR)

# 复用 run_enhanced_eval 的 EvalQuery / 策略 / 指标实现，避免重复维护
_spec = importlib.util.spec_from_file_location(
    "ree", os.path.join(BASE_DIR, "run_enhanced_eval.py")
)
ree = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ree)

from core.config import settings  # noqa: E402
from services.retriever_service import RetrieverService  # noqa: E402
from services.query_decomposer import decompose_query  # noqa: E402


def load_queries():
    with open(ree.EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
    queries = [
        ree.EvalQuery(
            id=it["id"],
            query=it["query"],
            jurisdiction=it.get("jurisdiction"),
            relevant=it.get("relevant", []),
        )
        for it in items
    ]
    return [q for q in queries if q.relevant]


def run_once(queries, enabled, retriever):
    settings.multihop_decompose_enabled = enabled
    rows = []
    for q in queries:
        gold = q.get_relevant_articles()
        ids = ree.strategy_hybrid_reranker(retriever, q, ree.TOP_K)
        m = ree.compute_metrics(ids, gold, list(gold), q)
        rows.append({
            "query_id": q.id,
            "query_text": q.query,
            "mrr": m["mrr"],
            "r10": m["recall"][10],
            "r20": m["recall"][20],
            "ndcg10": m["ndcg"][10],
            "ndcg20": m["ndcg"][20],
        })
    return rows


def fmt(label, rows):
    n = max(len(rows), 1)

    def avg(key):
        return sum(r[key] for r in rows) / n

    top1 = sum(1.0 for r in rows if r["mrr"] > 0) / n
    print(f"  {label:<6} MRR={avg('mrr'):.4f}  Top1={top1:.4f}  "
          f"R@10={avg('r10'):.4f}  R@20={avg('r20'):.4f}  "
          f"NDCG@10={avg('ndcg10'):.4f}  NDCG@20={avg('ndcg20'):.4f}")


def main():
    settings.search_cache_enabled = False
    settings.multihop_decompose_llm_enabled = False  # 规则确定、避免网络波动
    settings.reranker_enabled = True

    all_q = load_queries()
    affected = [q for q in all_q if decompose_query(q.query)]
    print("=" * 78)
    print(f"  多跳 B 查询分解 A/B 回归：总查询 {len(all_q)}，受影响（会触发分解）{len(affected)}")
    print("=" * 78)

    retriever = RetrieverService()

    # 让路跨法域路由后，部分查询会被守卫拦下：统计修复后真正仍走分解的条数，
    # 用于解释 ON 侧为何不再对「比较类」查询生效。
    settings.multihop_decompose_enabled = True
    still = [q for q in affected if retriever._decompose_query(q.query)]
    print(f"\n  让路路由守卫后，仍会走分解的查询：{len(still)} / {len(affected)}")

    print("\n  >>> 跑 OFF（multihop_decompose_enabled=False）")
    off = run_once(affected, False, retriever)
    print("  >>> 跑 ON （multihop_decompose_enabled=True）")
    on = run_once(affected, True, retriever)

    print("\n  === 汇总（受影响子集） ===")
    fmt("OFF", off)
    fmt("ON", on)

    # 逐条对比
    on_map = {r["query_id"]: r for r in on}
    better, worse, same = [], [], []
    for r in off:
        o = on_map[r["query_id"]]
        d = round(o["mrr"] - r["mrr"], 4)
        rec = (r["query_text"], r["mrr"], o["mrr"], d)
        if d > 1e-9:
            better.append(rec)
        elif d < -1e-9:
            worse.append(rec)
        else:
            same.append(rec)

    print(f"\n  === 逐条 MRR 对比 ===  变好 {len(better)} / 变差 {len(worse)} / 持平 {len(same)}")
    if worse:
        print("\n  [变差清单]")
        for t, a, b, d in sorted(worse, key=lambda x: x[3]):
            print(f"    {d:+.4f}  {a:.4f} -> {b:.4f}  {t}")
    if better:
        print("\n  [变好清单]")
        for t, a, b, d in sorted(better, key=lambda x: -x[3]):
            print(f"    {d:+.4f}  {a:.4f} -> {b:.4f}  {t}")

    print("\n" + "=" * 78)
    print("  完成")
    print("=" * 78)


if __name__ == "__main__":
    main()
