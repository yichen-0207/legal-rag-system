"""Agent vs RAG 离线对比：同一批查询、同一 judge 口径下比较迭代式 Agent 与单跳 RAG 的问答质量。

唯一变量为「是否多步探索」（两者共用合成 prompt 与 judge 模型）；HasAns@5 / Faithfulness / Relevance 口径与 run_qa_eval.py 一致，参考条款一律回取完整正文再判。
"""
import argparse
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/app")
sys.path.insert(0, BASE_DIR)

# 复用 run_qa_eval 的数据加载与 judge 实现，避免重复维护
_spec = importlib.util.spec_from_file_location("rqe", os.path.join(BASE_DIR, "run_qa_eval.py"))
rqe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rqe)

from core.config import settings  # noqa: E402
from services.qa_service import QAService  # noqa: E402
from services.agent.agent_service import AgentService  # noqa: E402

TOP_K = 5
OUTPUT_PATH = os.path.join(BASE_DIR, "eval_agent_vs_rag.json")


def _load_subset_ids(path: str) -> str:
    """读取固定回归子集（JSON 字符串数组），返回逗号分隔的 query_id（与 run_ragas_eval --subset 同口径）。"""
    p = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
    with open(p, "r", encoding="utf-8") as f:
        ids = json.load(f)
    return ",".join(str(x) for x in ids)


def _pick_queries(limit: int, stratify: bool, query_ids: str = "") -> list:
    """挑选评估查询；query_ids 非空时按其顺序取数（A/B 跑同一批查询），stratify 时按法域均匀抽样。"""
    queries = rqe.load_queries()
    gold_map = rqe.load_gold()
    queries = [q for q in queries if gold_map.get(q["query_id"])]

    if query_ids:
        want = [x.strip() for x in query_ids.split(",") if x.strip()]
        by_id = {q["query_id"]: q for q in queries}
        picked = [by_id[qid] for qid in want if qid in by_id]
        missing = [qid for qid in want if qid not in by_id]
        if missing:
            print(f"  [warn] 以下 query_id 无 gold 或不存在，已跳过: {missing}")
        return picked, gold_map

    if limit and limit < len(queries):
        if stratify:
            random.seed(42)  # 固定种子：同参数下多次运行抽到同一批查询
            by_jur = defaultdict(list)
            for q in queries:
                by_jur[q["jurisdiction"]].append(q)
            per = max(1, limit // max(len(by_jur), 1))
            picked = []
            for _, lst in sorted(by_jur.items()):
                random.shuffle(lst)
                picked.extend(lst[:per])
            # 不足则从剩余项补齐
            rest = [q for q in queries if q not in picked]
            random.shuffle(rest)
            picked.extend(rest[: max(0, limit - len(picked))])
            queries = picked[:limit]
        else:
            random.seed(42)
            queries = random.sample(queries, limit)
    return queries, gold_map


def _full_contents(repo, sources: list) -> list:
    """把 sources 的 (law_id, article_number) 回取完整正文：sources 预览仅 100 字，直接判分会误判忠实度。"""
    out = []
    for s in sources:
        law_id = s.get("law_id", "")
        article = s.get("article_number", "")
        content = s.get("content", "") or ""
        try:
            full = repo.get_article_full_text(law_id, article)
            if full:
                content = full
        except Exception:
            pass
        out.append((law_id, article, content))
    return out


def _collect_events(gen) -> dict:
    """消费一次问答生成器，收集 sources / 答案正文 / trace / 错误。"""
    sources, parts, trace, err = [], [], [], None
    try:
        for ev in gen:
            t = ev.get("type")
            if t == "sources":
                sources = ev.get("sources", []) or []
            elif t == "content":
                parts.append(ev.get("data", ""))
            elif t == "end":
                trace = ev.get("trace", []) or []
            elif t == "error":
                err = ev.get("data")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    return {"answer": "".join(parts), "sources": sources, "trace": trace, "error": err}


def _evaluate(query: str, gold: list, run: dict, qa_service, repo, mode: str) -> dict:
    """基于一次问答产出计算 HasAns 与两个 judge 分数。"""
    gold_set = {(g["law_id"], g.get("article", "")) for g in gold}
    matched = [
        {"law_id": s.get("law_id", ""), "article": s.get("article_number", "")}
        for s in run["sources"]
        if (s.get("law_id", ""), s.get("article_number", "")) in gold_set
    ]
    has_ans = len(matched) > 0

    rec = {
        "mode": mode,
        "has_ans": has_ans,
        "gold_matched": matched,
        "evidence_count": len(run["sources"]),
        "answer": run["answer"],
        "error": run["error"],
        "faithfulness_score": 0,
        "relevance_score": 0,
        "tool_calls": sum(1 for t in run["trace"] if t.get("tool")),
        # 检索到的完整条款正文，供 Ragas 侧计算忠实度/上下文精确率使用
        "contexts": [],
    }

    if run["error"] and not run["answer"]:
        return rec
    if not run["answer"]:
        rec["error"] = rec["error"] or "空答案"
        return rec

    refs = _full_contents(repo, run["sources"])
    rec["contexts"] = [content for _, _, content in refs]
    try:
        rec["faithfulness_score"] = rqe.judge_faithfulness(query, run["answer"], refs, qa_service)
    except Exception as e:
        rec["error"] = (rec["error"] or "") + f" | faithful judge 失败: {e}"
    try:
        rec["relevance_score"] = rqe.judge_relevance(query, run["answer"], qa_service)
    except Exception as e:
        rec["error"] = (rec["error"] or "") + f" | relevance judge 失败: {e}"
    return rec


def _summary(rows: list) -> dict:
    n = max(len(rows), 1)
    faith = [r["faithfulness_score"] for r in rows if r["faithfulness_score"] > 0]
    rel = [r["relevance_score"] for r in rows if r["relevance_score"] > 0]
    return {
        "n": len(rows),
        "has_ans": round(sum(1 for r in rows if r["has_ans"]) / n, 4),
        "faithfulness": round(sum(faith) / max(len(faith), 1), 2) if faith else 0.0,
        "relevance": round(sum(rel) / max(len(rel), 1), 2) if rel else 0.0,
        "avg_tool_calls": round(sum(r["tool_calls"] for r in rows) / n, 2),
        "avg_latency": round(sum(r["latency"] for r in rows) / n, 1),
        "errors": sum(1 for r in rows if r["error"] and not r["answer"]),
    }


def _fmt(label: str, s: dict) -> str:
    return (f"  {label:<6} n={s['n']:<3} HasAns={s['has_ans']:.2%}  "
            f"Faith={s['faithfulness']:.2f}  Rel={s['relevance']:.2f}  "
            f"工具调用={s['avg_tool_calls']:.1f}  耗时={s['avg_latency']:.0f}s  错误={s['errors']}")


def _warmup(qa_service) -> None:
    """预热编码与精排，避免首次前向开销污染第一条样本的延迟。"""
    print("  预热中（编码 + 精排首次前向）...", flush=True)
    try:
        qa_service.repo.query(["warmup 劳动关系 数据保护"], n_results=1)
    except Exception as e:
        print(f"  [warn] 检索预热跳过: {type(e).__name__}: {e}")
    try:
        if settings.reranker_enabled:
            from services.reranker_service import ReRankerService
            ReRankerService().rerank(
                query="warmup 劳动关系",
                candidates=[{"content": "劳动合同与劳动关系条款。" * 20, "id": "w"}],
                top_k=1,
            )
    except Exception as e:
        print(f"  [warn] 精排预热跳过: {type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Agent vs RAG 离线效果对比")
    ap.add_argument("--limit", type=int, default=0, help="评估条数（0=全部）")
    ap.add_argument("--stratify", action="store_true", help="按法域分层抽样")
    ap.add_argument("--query-ids", default="", help="指定 query_id（逗号分隔），用于前后跑同一批查询")
    ap.add_argument("--subset", default="", help="固定回归子集文件（query_id 字符串数组），与 run_ragas_eval 口径一致")
    ap.add_argument("--mode", choices=["both", "rag", "agent"], default="both")
    ap.add_argument("--tag", default="", help="输出文件标签，避免覆盖既有结果（如 before/after）")
    args = ap.parse_args()

    query_ids = _load_subset_ids(args.subset) if args.subset else args.query_ids
    queries, gold_map = _pick_queries(args.limit, args.stratify, query_ids)
    print("=" * 78)
    print(f"  Agent vs RAG 离线评估：{len(queries)} 条查询，top_k={TOP_K}，mode={args.mode}")
    print(f"  法域分布：{dict(sorted(Counter(q['jurisdiction'] for q in queries).items()))}")
    print("=" * 78)

    qa_service = QAService()
    repo = qa_service.repo
    rag_rows, agent_rows = [], []

    # 基线 RAG 与 Agent 用同一 QAService 实例；Agent 仅在 agent 模式下置位开关
    agent_service = AgentService() if args.mode in ("both", "agent") else None

    # 预热：编码与精排首次前向的开销若落在正式样本上，会把检索耗时抬高，使延迟对比失真。
    _warmup(qa_service)

    for i, q in enumerate(queries, 1):
        gold = gold_map.get(q["query_id"], [])
        jur = None if q["jurisdiction"] == "跨法域" else q["jurisdiction"]
        jur_list = [jur] if jur else None
        print(f"  [{i}/{len(queries)}] {q['query_id']} ({q['jurisdiction']}): {q['query'][:44]}...", flush=True)

        if args.mode in ("both", "rag"):
            t0 = time.time()
            run = _collect_events(qa_service.ask_stream(q["query"], TOP_K, jur_list, None))
            rec = _evaluate(q["query"], gold, run, qa_service, repo, "rag")
            rec["latency"] = time.time() - t0
            rec["query_id"] = q["query_id"]
            rec["jurisdiction"] = q["jurisdiction"]
            rag_rows.append(rec)
            print(f"      RAG   HasAns={rec['has_ans']} Faith={rec['faithfulness_score']} "
                  f"Rel={rec['relevance_score']} ({rec['latency']:.0f}s)", flush=True)

        if args.mode in ("both", "agent"):
            settings.agent_enabled = True  # 仅进程内开启，不落配置
            t0 = time.time()
            run = _collect_events(agent_service.ask_stream(q["query"], TOP_K, jur_list, None))
            rec = _evaluate(q["query"], gold, run, qa_service, repo, "agent")
            rec["latency"] = time.time() - t0
            rec["query_id"] = q["query_id"]
            rec["jurisdiction"] = q["jurisdiction"]
            agent_rows.append(rec)
            print(f"      Agent HasAns={rec['has_ans']} Faith={rec['faithfulness_score']} "
                  f"Rel={rec['relevance_score']} tools={rec['tool_calls']} ({rec['latency']:.0f}s)", flush=True)

    print("\n" + "=" * 78)
    print("  汇总")
    print("=" * 78)
    summary = {}
    if rag_rows:
        summary["rag"] = _summary(rag_rows)
        print(_fmt("RAG", summary["rag"]))
    if agent_rows:
        summary["agent"] = _summary(agent_rows)
        print(_fmt("Agent", summary["agent"]))

    # 逐条 has_ans 变化（在两条都跑的查询上）
    if rag_rows and agent_rows:
        rag_map = {r["query_id"]: r for r in rag_rows}
        better = worse = 0
        for a in agent_rows:
            r = rag_map.get(a["query_id"])
            if not r:
                continue
            if a["has_ans"] and not r["has_ans"]:
                better += 1
            elif r["has_ans"] and not a["has_ans"]:
                worse += 1
        print(f"\n  HasAns 变化：Agent 变好 {better} / 变差 {worse}")

    out_path = OUTPUT_PATH if not args.tag else OUTPUT_PATH.replace(
        ".json", f"_{args.tag}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rag": rag_rows, "agent": agent_rows},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  明细已写入: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
