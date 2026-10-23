"""多跳路径检索级 A/B（零 LLM）：对比混合检索基线（向量+BM25+RRF+精排/引用扩展）与跨法域对齐链、实体图多跳、引用扩展在 multihop_dataset 上的条款覆盖。

指标以 terminal gold（路径末端、须靠第二/第三跳才能拿到的条款）为主，另附 all-gold 口径；全程只做向量/ES/正则计算。
"""
import csv
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/app")
sys.path.insert(0, BASE_DIR)

from core.config import settings  # noqa: E402
from repositories.elasticsearch import ElasticsearchRepository  # noqa: E402
from services.kg_reasoning_service import KGReasoningService  # noqa: E402
from services.multihop_service import MultiHopService  # noqa: E402
from services.retriever_service import RetrieverService  # noqa: E402

DATA_PATH = os.path.join(BASE_DIR, "multihop_dataset.json")
OUT_JSON = os.path.join(BASE_DIR, "multihop_ab_results.json")
OUT_CSV = os.path.join(BASE_DIR, "multihop_ab_results.csv")
TOP_K = 20
KS = (5, 10, 20)


def norm_article(value):
    """条号规范化：各法域写法不一（35C / 第35C條 / Article 27 / 52.204-13），去前缀、中文标记与标点后再比较。"""
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"^(?:article|art|sections?|sec)", "", text)
    text = text.strip("第条條款项項章節节 ")
    return re.sub(r"[^\w]", "", text)


def ref_key(law_id, article):
    if not law_id or not article:
        return None
    key = norm_article(article)
    if not key:
        return None
    return f"{law_id}::{key}"


def gold_sets(item):
    """返回 (all_gold, terminal_gold)；terminal 取每条 gold path 最后一个带 article 的节点，是多跳的增量价值。"""
    all_gold = set()
    for gold in item.get("gold_articles") or []:
        if "::" not in gold:
            continue
        law_id, article = gold.split("::", 1)
        key = ref_key(law_id, article)
        if key:
            all_gold.add(key)
    terminal = set()
    paths = item.get("gold_paths") or [{"steps": item.get("gold_path") or []}]
    for path in paths:
        arts = []
        for step in path.get("steps") or []:
            for side in ("from", "to"):
                ref = step.get(side) or {}
                key = ref_key(ref.get("law_id"), ref.get("article"))
                if key:
                    arts.append(key)
        if arts:
            terminal.add(arts[-1])
    return all_gold, terminal


def compute_metrics(ranked, gold, ks=KS):
    gold = set(gold)
    n = len(gold)
    out = {"mrr": 0.0}
    for i, key in enumerate(ranked, start=1):
        if key in gold:
            out["mrr"] = 1.0 / i
            break
    for k in ks:
        hits = len(set(ranked[:k]) & gold)
        out[f"recall@{k}"] = hits / n if n else 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, key in enumerate(ranked[:10]) if key in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(n, 10)))
    out["ndcg@10"] = dcg / ideal if ideal else 0.0
    out["full@10"] = 1.0 if gold and gold <= set(ranked[:10]) else 0.0
    return out


def baseline_ranked(retriever, item):
    jur = None if item.get("jurisdiction") == "跨法域" else item.get("jurisdiction")
    results = retriever.hybrid_search(
        query=item["query"], top_k=TOP_K, jurisdiction=jur,
        _decompose=True, _rerank=True,
    )
    ranked = []
    for r in results:
        key = ref_key(r.get("law_id"), r.get("article_number"))
        if key and key not in ranked:
            ranked.append(key)
    return ranked


def cross_ranked(service, item, oracle=False):
    kwargs = dict(
        question=item["query"], top_k_laws=3, synthesize=False,
        jurisdictions=item.get("jurisdictions"),
    )
    if oracle:
        # 诊断口径：给定正确源条款，只衡量「跨法域对齐 + 目标条款定位」能力，
        # 不把源侧定位误差混进多跳指标。
        kwargs["source_law_id"] = item["source"]["law_id"]
        kwargs["source_article"] = item["source"]["article"]
    data = service.answer(**kwargs)
    ranked = []
    # 末端条款优先：路径输出不是普通检索列表，先给「必须靠第二跳才能拿到」的目标
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


def entity_ranked(service, item):
    data = service.answer(
        item["query"], top_k_entities=5, hops=2, deepen=False, synthesize=False,
        jurisdictions=item.get("jurisdictions"),
    )
    ranked = []
    for path in data.get("reasoning_path") or []:
        articles = []
        for step in path.get("steps") or []:
            ev = ((step.get("evidence") or [{}])[0]) or {}
            # 引用型边上「被引用条款」也是这条路径的证据（例如 52.204-7 → 52.204-13），
            # 只记来源条款会漏掉路径真正走到的目标条款。
            for field in ("article", "to_article", "from_article"):
                key = ref_key(ev.get("law_id"), ev.get(field))
                if key and key not in articles:
                    articles.append(key)
        # 深跳条款优先（路径末端承载答案），再补路径前缀
        for key in reversed(articles):
            if key not in ranked:
                ranked.append(key)
    return ranked


def citation_ranked(repo, item):
    source = item.get("source") or {}
    law_id = source.get("law_id")
    article = source.get("article")
    content = repo.get_article_full_text(law_id, article)
    if not content:
        return []
    seeds = [(law_id, article, content)]
    rounds = []
    for _ in range(max(1, int(item.get("hops") or 2))):
        # 评测关注 2~3 跳路径能否走通：单轮补入上限放宽到 10，
        # 避免第一跳的多个分支把后续目标挤出候选（生产 follow_references 默认 5）。
        docs = repo.get_referenced_articles(seeds, max_refs=10)
        if not docs:
            break
        round_keys = []
        for doc in docs:
            key = ref_key(doc.get("law_id"), doc.get("article_number"))
            if key and key not in round_keys:
                round_keys.append(key)
        rounds.append(round_keys)
        seeds = [
            (d.get("law_id", ""), d.get("article_number", ""), d.get("content", "") or "")
            for d in docs
        ]
    # 深跳优先：最后一轮的被引条款才是 multi-hop 的增量证据
    ranked = []
    for group in reversed(rounds):
        for key in group:
            if key not in ranked:
                ranked.append(key)
    return ranked


def summarize(rows, variant, metric_keys):
    n = max(len(rows), 1)
    return {m: round(sum(r[variant][m] for r in rows) / n, 4) for m in metric_keys}


def fmt(label, summary, metric_keys):
    parts = [f"{m}={summary[m]:.4f}" for m in metric_keys]
    print(f"  {label:<10} " + "  ".join(parts))


def main():
    settings.search_cache_enabled = False
    settings.multihop_decompose_llm_enabled = False  # 规则分解，避免网络波动
    settings.reranker_enabled = True

    items = json.load(open(DATA_PATH, encoding="utf-8"))["items"]
    retriever = RetrieverService()
    repo = retriever.repo
    multihop = MultiHopService(repo=repo)
    kg = KGReasoningService(repo=repo)

    rows = []
    print("=" * 88)
    print(f"  多跳路径检索级 A/B：{len(items)} 条（零 LLM）")
    print("=" * 88)

    for idx, item in enumerate(items, start=1):
        all_gold, terminal = gold_sets(item)
        t0 = time.perf_counter()
        base = baseline_ranked(retriever, item)
        t_base = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        oracle = None
        if item["type"] == "cross_jurisdiction":
            mh = cross_ranked(multihop, item)
            oracle = cross_ranked(multihop, item, oracle=True)
        elif item["type"] == "entity_navigation":
            mh = entity_ranked(kg, item)
        else:
            mh = citation_ranked(repo, item)
        t_mh = (time.perf_counter() - t0) * 1000

        row = {
            "id": item["id"],
            "type": item["type"],
            "hops": item["hops"],
            "query": item["query"],
            "terminal_gold": sorted(terminal),
            "baseline": compute_metrics(base, terminal),
            "multihop": compute_metrics(mh, terminal),
            "baseline_all": compute_metrics(base, all_gold),
            "multihop_all": compute_metrics(mh, all_gold),
            "baseline_top5": base[:5],
            "multihop_top5": mh[:5],
            "baseline_ms": round(t_base, 1),
            "multihop_ms": round(t_mh, 1),
        }
        if oracle is not None:
            row["multihop_oracle"] = compute_metrics(oracle, terminal)
            row["multihop_oracle_top5"] = oracle[:5]
        row["delta_mrr"] = round(row["multihop"]["mrr"] - row["baseline"]["mrr"], 4)
        row["delta_recall@10"] = round(
            row["multihop"]["recall@10"] - row["baseline"]["recall@10"], 4
        )
        rows.append(row)
        oracle_note = ""
        if oracle is not None:
            oracle_note = f"  oracle R@10={row['multihop_oracle']['recall@10']:.3f}"
        print(
            f"  [{idx:02d}/{len(items)}] {item['id']} {item['type']:<20} "
            f"MRR {row['baseline']['mrr']:.3f}->{row['multihop']['mrr']:.3f}  "
            f"R@10 {row['baseline']['recall@10']:.3f}->{row['multihop']['recall@10']:.3f}"
            + oracle_note
        )

    metric_keys = ["mrr", "recall@5", "recall@10", "recall@20", "ndcg@10", "full@10"]
    summary = {
        "items": len(rows),
        "baseline": summarize(rows, "baseline", metric_keys),
        "multihop": summarize(rows, "multihop", metric_keys),
    }
    summary["delta"] = {
        m: round(summary["multihop"][m] - summary["baseline"][m], 4) for m in metric_keys
    }
    oracle_rows = [r for r in rows if r.get("multihop_oracle")]
    if oracle_rows:
        summary["multihop_oracle"] = summarize(oracle_rows, "multihop_oracle", metric_keys)

    by_type = {}
    for typ in ("cross_jurisdiction", "entity_navigation", "citation_chain"):
        subset = [r for r in rows if r["type"] == typ]
        if not subset:
            continue
        by_type[typ] = {
            "items": len(subset),
            "baseline": summarize(subset, "baseline", metric_keys),
            "multihop": summarize(subset, "multihop", metric_keys),
        }
        sub_oracle = [r for r in subset if r.get("multihop_oracle")]
        if sub_oracle:
            by_type[typ]["multihop_oracle"] = summarize(
                sub_oracle, "multihop_oracle", metric_keys
            )

    print("\n  === 汇总（terminal gold：只看必须靠多跳才能拿到的末端条款） ===")
    fmt("baseline", summary["baseline"], metric_keys)
    fmt("multihop", summary["multihop"], metric_keys)
    if summary.get("multihop_oracle"):
        fmt("mh-oracle", summary["multihop_oracle"], metric_keys)
    print("\n  === 分类型 ===")
    for typ, data in by_type.items():
        print(f"  [{typ}] n={data['items']}")
        fmt("baseline", data["baseline"], metric_keys)
        fmt("multihop", data["multihop"], metric_keys)
        if data.get("multihop_oracle"):
            fmt("mh-oracle", data["multihop_oracle"], metric_keys)

    # 落盘
    payload = {
        "meta": {
            "dataset": "multihop_dataset.json",
            "top_k": TOP_K,
            "zero_llm": True,
            "note": "terminal gold 为每条路径末端条款；all_gold 口径见每行 baseline_all/multihop_all；"
                    "multihop_oracle 仅对跨法域条目给出，表示给定正确源条款时的对齐能力上限",
        },
        "summary": summary,
        "by_type": by_type,
        "rows": rows,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    csv_keys = [
        "id", "type", "hops", "query",
        "baseline_mrr", "multihop_mrr", "delta_mrr",
        "baseline_recall@10", "multihop_recall@10", "delta_recall@10",
        "baseline_recall@5", "multihop_recall@5",
        "baseline_ndcg@10", "multihop_ndcg@10",
        "baseline_full@10", "multihop_full@10",
        "oracle_mrr", "oracle_recall@10", "oracle_full@10",
        "baseline_top5", "multihop_top5", "oracle_top5",
    ]
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "id": r["id"], "type": r["type"], "hops": r["hops"], "query": r["query"],
                "baseline_mrr": r["baseline"]["mrr"], "multihop_mrr": r["multihop"]["mrr"],
                "delta_mrr": r["delta_mrr"],
                "baseline_recall@10": r["baseline"]["recall@10"],
                "multihop_recall@10": r["multihop"]["recall@10"],
                "delta_recall@10": r["delta_recall@10"],
                "baseline_recall@5": r["baseline"]["recall@5"],
                "multihop_recall@5": r["multihop"]["recall@5"],
                "baseline_ndcg@10": r["baseline"]["ndcg@10"],
                "multihop_ndcg@10": r["multihop"]["ndcg@10"],
                "baseline_full@10": r["baseline"]["full@10"],
                "multihop_full@10": r["multihop"]["full@10"],
                "oracle_mrr": (r.get("multihop_oracle") or {}).get("mrr", ""),
                "oracle_recall@10": (r.get("multihop_oracle") or {}).get("recall@10", ""),
                "oracle_full@10": (r.get("multihop_oracle") or {}).get("full@10", ""),
                "baseline_top5": " | ".join(r["baseline_top5"]),
                "multihop_top5": " | ".join(r["multihop_top5"]),
                "oracle_top5": " | ".join(r.get("multihop_oracle_top5") or []),
            })

    print(f"\n  已写入：{OUT_JSON}")
    print(f"  已写入：{OUT_CSV}")
    print("=" * 88)


if __name__ == "__main__":
    main()
