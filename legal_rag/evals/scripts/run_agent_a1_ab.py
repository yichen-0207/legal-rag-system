"""A1 单变量 A/B：search_laws 观测字段（total_chars/truncated/omitted_chars）开 vs 关。

对照口径：同一批 query_id、同一 AgentService、缓存全关、agent 模式、不跑 judge；
逐题交替两臂顺序（i 为奇数时先 on 后 off），抵消 ES 缓存升温与时间漂移带来的顺序偏差。

落盘内容（本脚本的核心产出，后续评测可直接复用）：
- `trace`：每次工具调用的 step/tool/args/summary；search_laws 额外带 `hits`
  （每条结果的 rank/law_id/article），可直接算命中排名分布，不再丢数据；
- 逐题 token：读 `core.llm_usage` 当日 NDJSON 的字节增量（含规划/决策/合成全部 LLM 调用）；
- 逐题派生指标：get_article_text 次数与去重条款数、重复率、stop_reason、排名分布。

用法（容器内，源码 bind-mount 到 /app）：
    python evals/scripts/run_agent_a1_ab.py --tag a1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

# 缓存必须在导入 settings 之前关闭：答案缓存会让第 2 题起直接复用历史答案（评测失真）
os.environ["LEGAL_SEARCH_CACHE_ENABLED"] = "false"
os.environ["LEGAL_QA_ANSWER_CACHE_ENABLED"] = "false"
os.environ["LEGAL_REDIS_CACHE_ENABLED"] = "false"

from _bootstrap import DATASETS_DIR, RESULTS_DIR  # noqa: E402

from core.config import settings  # noqa: E402
from core.llm_usage import current_log_path  # noqa: E402
from services.agent.agent_service import AgentService, ROUTE_MODE_AGENT  # noqa: E402
from services.qa_service import QAService  # noqa: E402

# P3 同批 query_id 的 10 题子集：4 cross / 3 entity / 3 citation
DEFAULT_QUERY_IDS = ",".join([
    "MH-C01", "MH-C02", "MH-C03", "MH-C04",
    "MH-E01", "MH-E02", "MH-E05",
    "MH-R01", "MH-R04", "MH-R05",
])
DATASET_PATH = os.path.join(DATASETS_DIR, "multihop_dataset.json")

ARM_ON = "on"    # agent_search_obs_meta_enabled=True（带三个字段 + 描述提示）
ARM_OFF = "off"  # False（输出与改动前逐字一致）


# ---------------------------------------------------------------- 数据 / 账本
def _load_items(query_ids: str) -> list:
    """按指定顺序读取 multihop 题目；缺 id 直接失败（避免悄悄少跑几条）。"""
    want = [x.strip() for x in (query_ids or "").split(",") if x.strip()]
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_id = {it["id"]: it for it in (data.get("items") or [])}
    missing = [qid for qid in want if qid not in by_id]
    if missing:
        raise SystemExit(f"数据集中缺少 query_id: {missing}")
    return [by_id[qid] for qid in want]


def _ledger_offset() -> int:
    p = current_log_path()
    try:
        return p.stat().st_size if p.exists() else 0
    except OSError:
        return 0


def _ledger_delta(offset: int) -> dict:
    """读取账本自 offset 起的追加行，汇总 token 与调用次数（账本不可读时返回零值并标注）。"""
    p = current_log_path()
    bucket = {"tokens": 0, "calls": 0, "errors": 0, "by_purpose": {}, "ledger_ok": True}
    try:
        if not p.exists():
            bucket["ledger_ok"] = False
            return bucket
        size = p.stat().st_size
        if size < offset:  # 跨日切文件
            offset = 0
        with p.open("rb") as fp:
            fp.seek(offset)
            for raw in fp:
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                bucket["calls"] += 1
                bucket["tokens"] += int(row.get("total_tokens") or 0)
                if not row.get("ok", True):
                    bucket["errors"] += 1
                purpose = str(row.get("purpose") or "unknown")
                bucket["by_purpose"][purpose] = bucket["by_purpose"].get(purpose, 0) + 1
    except OSError:
        bucket["ledger_ok"] = False
    return bucket


# ---------------------------------------------------------------- 单次运行
def _run_one(agent: AgentService, item: dict, arm: str, top_k: int) -> dict:
    """跑一题一臂，返回含完整 trace / 逐题 token / 派生指标的记录。"""
    settings.agent_search_obs_meta_enabled = (arm == ARM_ON)
    jur = item.get("jurisdiction") or ""
    jur_list = None if jur in ("跨法域", "") else [jur]

    offset = _ledger_offset()
    t0 = time.perf_counter()
    answer_parts, trace, sources = [], [], []
    stop_reason, tool_calls, err, timing = "", 0, None, {}
    try:
        for ev in agent.ask_stream(item["query"], top_k, jur_list, None,
                                   route_mode=ROUTE_MODE_AGENT):
            t = (ev or {}).get("type")
            if t == "content":
                answer_parts.append(ev.get("data", ""))
            elif t == "sources":
                sources = ev.get("sources", []) or []
            elif t == "end":
                stop_reason = ev.get("stop_reason", "") or ""
                tool_calls = int(ev.get("tool_calls") or 0)
                trace = ev.get("trace", []) or []
                timing = ev.get("timing", {}) or {}
            elif t == "error":
                err = ev.get("data")
    except Exception as e:  # noqa: BLE001 - 单题失败不中断整批
        err = f"{type(e).__name__}: {e}"
    wall_s = round(time.perf_counter() - t0, 2)
    usage = _ledger_delta(offset)
    answer = "".join(answer_parts)

    rec = {
        "query_id": item["id"],
        "type": item.get("type", ""),
        "arm": arm,
        "stop_reason": stop_reason,
        "tool_calls": tool_calls,
        "tool_sequence": [t.get("tool") for t in trace if t.get("tool")],
        "tokens": usage["tokens"],
        "llm_calls": usage["calls"],
        "llm_errors": usage["errors"],
        "llm_by_purpose": usage["by_purpose"],
        "ledger_ok": usage["ledger_ok"],
        "wall_s": wall_s,
        "timing": timing,
        "answer_chars": len(answer),
        "answer": answer,
        "sources": [
            {"law_id": s.get("law_id", ""), "article": str(s.get("article_number", ""))}
            for s in sources
        ],
        "trace": trace,
        "error": err,
    }
    rec["metrics"] = _row_metrics(trace)
    return rec


def _row_metrics(trace: list) -> dict:
    """从 trace 派生：get_article_text 次数/去重条款/重复率，以及其相对最近一次检索的排名。"""
    gat_pairs, gat_ranks, search_all_ranks = [], [], []
    last_hits = []
    for t in trace:
        name = t.get("tool")
        if name == "search_laws":
            last_hits = t.get("hits") or []
            search_all_ranks.extend(h.get("rank") for h in last_hits)
        elif name == "get_article_text":
            args = t.get("args") or {}
            pair = (args.get("law_id", ""), str(args.get("article_number", "")))
            gat_pairs.append(pair)
            hit = next((h for h in last_hits
                        if h.get("law_id") == pair[0]
                        and str(h.get("article", "")) == pair[1]), None)
            # 未出现在「上一次」检索命中里 → 记为 miss（可能来自更早检索/引用/图谱）
            gat_ranks.append(int(hit["rank"]) if hit else 0)
    unique_pairs = {(a, b) for a, b in gat_pairs}
    n = len(gat_pairs)
    return {
        "gat_calls": n,
        "gat_unique": len(unique_pairs),
        "gat_dup": n - len(unique_pairs),
        "gat_dup_rate": round((n - len(unique_pairs)) / n, 4) if n else 0.0,
        "gat_rank_dist": dict(Counter(gat_ranks)),
        "search_result_rank_dist": dict(Counter(search_all_ranks)),
        "search_hit_total": len(search_all_ranks),
    }


def _annotate_visibility(rows: list, repo) -> None:
    """标注每次 get_article_text 的条款在「上一次检索」里是否已被完整展示。

    visible_complete = 命中上一次检索且全文 ≤ agent_obs_max_chars（开臂会看到 truncated=false）；
    visible_truncated = 命中但片段被截断；not_in_last_hits = rank=0；unknown_len = 取不到正文。
    比较两臂的 visible_complete 次数，即可判断模型是否真的因 truncated=false 而少取全文。
    """
    cap = max(1, int(settings.agent_obs_max_chars))
    length_cache: dict = {}

    def _length(law_id: str, article: str) -> int:
        key = (law_id, article)
        if key not in length_cache:
            try:
                length_cache[key] = len(repo.get_article_full_text(law_id, article) or "")
            except Exception:
                length_cache[key] = -1
        return length_cache[key]

    for row in rows:
        last_hits = []
        c = {"visible_complete": 0, "visible_truncated": 0,
             "not_in_last_hits": 0, "unknown_len": 0}
        for t in row["trace"]:
            name = t.get("tool")
            if name == "search_laws":
                last_hits = t.get("hits") or []
                continue
            if name != "get_article_text":
                continue
            args = t.get("args") or {}
            law_id, article = str(args.get("law_id", "")), str(args.get("article_number", ""))
            hit = next((h for h in last_hits
                        if h.get("law_id") == law_id
                        and str(h.get("article", "")) == article), None)
            if hit is None:
                c["not_in_last_hits"] += 1
                continue
            n = _length(law_id, article)
            if n < 0:
                c["unknown_len"] += 1
            elif n <= cap:
                c["visible_complete"] += 1
            else:
                c["visible_truncated"] += 1
        row["metrics"].update(c)


# ---------------------------------------------------------------- 汇总
def _arm_summary(rows: list) -> dict:
    n = max(len(rows), 1)
    gat = sum(r["metrics"]["gat_calls"] for r in rows)
    gat_unique = sum(r["metrics"]["gat_unique"] for r in rows)
    rank_dist = Counter()
    for r in rows:
        rank_dist.update(r["metrics"]["gat_rank_dist"])
    return {
        "n": len(rows),
        "gat_calls": gat,
        "gat_unique_pairs": gat_unique,
        "gat_dup": gat - gat_unique,
        "gat_dup_rate": round((gat - gat_unique) / gat, 4) if gat else 0.0,
        "gat_calls_per_query": round(gat / n, 2),
        "gat_per_search": round(gat / max(1, sum(len([x for x in r["tool_sequence"]
                                                    if x == "search_laws"]) for r in rows)), 2),
        "stop_dist": dict(Counter(r["stop_reason"] for r in rows)),
        "tokens_total": sum(r["tokens"] for r in rows),
        "tokens_per_query": round(sum(r["tokens"] for r in rows) / n, 1),
        "tokens_by_query": {r["query_id"]: r["tokens"] for r in rows},
        "wall_avg_s": round(sum(r["wall_s"] for r in rows) / n, 1),
        "errors": sum(1 for r in rows if r["error"] and not r["answer"]),
        # 0 = 该条 get_article_text 的条款不在最近一次检索命中里
        "gat_rank_dist": dict(sorted(rank_dist.items())),
        # 开臂看得到 truncated=false 的「已完整展示」条款被重复取全文的次数
        "visible_complete": sum(r["metrics"].get("visible_complete", 0) for r in rows),
        "visible_truncated": sum(r["metrics"].get("visible_truncated", 0) for r in rows),
        "not_in_last_hits": sum(r["metrics"].get("not_in_last_hits", 0) for r in rows),
        "unknown_len": sum(r["metrics"].get("unknown_len", 0) for r in rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="A1 观测字段开/关单变量 A/B")
    ap.add_argument("--query-ids", default=DEFAULT_QUERY_IDS, help="逗号分隔的 multihop id")
    ap.add_argument("--arms", default=f"{ARM_OFF},{ARM_ON}", help="臂顺序，如 off,on")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--tag", default="a1", help="结果文件标签")
    ap.add_argument("--max-tokens", type=int, default=600000,
                    help="累计 token 硬上限（默认 600k ≈ ¥1.98，超限立即停跑并落盘部分结果）")
    args = ap.parse_args()

    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    bad = [a for a in arms if a not in (ARM_ON, ARM_OFF)]
    if bad:
        raise SystemExit(f"未知臂: {bad}")

    items = _load_items(args.query_ids)
    print("=" * 78)
    print(f"  A1 A/B：{len(items)} 题 × {len(arms)} 臂 = {len(items) * len(arms)} 次 Agent 运行")
    print(f"  query_ids={[i['id'] for i in items]}")
    print(f"  arms={arms}  缓存=关  route_mode=agent   judge=不跑")
    print("=" * 78, flush=True)

    # 预热：编码/精排首次前向与 ES 连接建立不应计入任何一臂
    qa = QAService()
    try:
        qa.repo.query(["warmup 数据保护"], n_results=1)
        print("  预热完成", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] 预热失败: {type(e).__name__}: {e}", flush=True)

    agent = AgentService()
    rows = []
    cum_tokens = 0
    aborted = None
    for i, item in enumerate(items):
        order = arms if i % 2 == 0 else list(reversed(arms))
        for arm in order:
            print(f"  [{i + 1}/{len(items)}] {item['id']} arm={arm} ...", flush=True)
            rec = _run_one(agent, item, arm, args.top_k)
            rows.append(rec)
            cum_tokens += rec["tokens"]
            m = rec["metrics"]
            print(f"      stop={rec['stop_reason']} tools={rec['tool_calls']} "
                  f"gat={m['gat_calls']}(去重{m['gat_unique']}) rank={m['gat_rank_dist']} "
                  f"tok={rec['tokens']} 累计={cum_tokens} wall={rec['wall_s']}s", flush=True)
            if cum_tokens > args.max_tokens:
                aborted = {"reason": "token_guard", "cum_tokens": cum_tokens,
                           "max_tokens": args.max_tokens}
                print(f"[ABORT] 累计 {cum_tokens} 超上限 {args.max_tokens}，停跑并落盘已完成部分",
                      flush=True)
                break
        if aborted:
            break

    _annotate_visibility(rows, qa.repo)
    summary = {arm: _arm_summary([r for r in rows if r["arm"] == arm]) for arm in arms}
    print("\n" + "=" * 78)
    print("  汇总")
    print("=" * 78)
    for arm in arms:
        s = summary[arm]
        print(f"  {arm:<3} gat={s['gat_calls']}(去重{s['gat_unique_pairs']}, "
              f"重复率{s['gat_dup_rate']:.1%}) stop={s['stop_dist']} "
              f"tok/题={s['tokens_per_query']} tok合计={s['tokens_total']} wall={s['wall_avg_s']}s "
              f"gat排名={s['gat_rank_dist']} 已完整展示仍取全文={s['visible_complete']} "
              f"截断仍取={s['visible_truncated']} 不在上次命中={s['not_in_last_hits']}")

    out_path = os.path.join(RESULTS_DIR, f"agent_a1_ab_{args.tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "query_ids": [i["id"] for i in items],
                "arms": arms,
                "top_k": args.top_k,
                "aborted": aborted,
                "tokens_total": cum_tokens,
                "llm_log_path": str(current_log_path()),
                "search_cache_enabled": settings.search_cache_enabled,
                "qa_answer_cache_enabled": settings.qa_answer_cache_enabled,
                "agent_obs_max_chars": settings.agent_obs_max_chars,
                "agent_article_text_chars": settings.agent_article_text_chars,
                "gat_rank_dist_note": "rank=该条在最近一次 search_laws 命中中的名次；0=不在该次命中",
            },
            "summary": summary,
            "rows": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  明细已写入: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
