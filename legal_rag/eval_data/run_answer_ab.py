"""答案级 A/B：multihop（多跳集双方案作答）、regress（12 条单跳，baseline=快速 RAG vs Agent auto 路由，检验非劣化）、judge（仅对分歧条目两两裁判）。

指标：surf_cov 链路拿到条款命中 terminal gold 的比例；cited_cov 答案明确引用命中比例；answered 答案非空且未声明材料不足；full = cited_cov 达 1 且 answered。统计：成对 bootstrap 95% 置信区间 + 符号检验（零 token 成本）。
"""
import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, "/app")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from pydantic import BaseModel  # noqa: E402

from core.config import settings  # noqa: E402
from core.llm import get_chat_model  # noqa: E402
from repositories.elasticsearch import ElasticsearchRepository  # noqa: E402
from services.agent.agent_service import AgentService  # noqa: E402
from services.kg_reasoning_service import KGReasoningService  # noqa: E402
from services.multihop_service import MultiHopService  # noqa: E402
from services.qa_service import QAService  # noqa: E402

MULTIHOP_PATH = os.path.join(BASE_DIR, "multihop_dataset.json")
SINGLE_PATH = os.path.join(BASE_DIR, "eval_dataset_clause.json")
OUT_JSON = os.path.join(BASE_DIR, "answer_ab_results.json")
OUT_CSV = os.path.join(BASE_DIR, "answer_ab_results.csv")
CACHE_PATH = os.path.join(BASE_DIR, "answer_ab_cache.json")
USAGE_DIR = "/app/logs"
REGRESS_ITEMS = 12
SOURCE_REF_RE = re.compile(r"【来源(\d+)】")
EVIDENCE_REF_RE = re.compile(r"【证据(\d+)】")
INSUFFICIENT_RE = re.compile(r"材料不足|无法确定|未能找到|没有找到|证据不足")
# 答案正文里的条款表述：第11条 / Article 28 / section 35C / FAR 52.204-30
ARTICLE_TEXT_RES = (
    re.compile(r"第\s*([0-9A-Za-z][0-9A-Za-z.\-]*)\s*[条條款项]"),
    re.compile(r"(?:article|art\.?|section|sec\.?)\s*([0-9A-Za-z][0-9A-Za-z.\-]*)", re.I),
    re.compile(r"\b(\d{2}\.\d{3,4}-\d+)\b"),
)


# ============================================================
# 基础工具
# ============================================================
def norm_article(value) -> str:
    """条号规范化：35C / 第35C條 / Article 27 / 52.204-13 统一后比较。"""
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"^(?:article|art|sections?|sec)", "", text)
    text = text.strip("第条條款项項章節节 ")
    return re.sub(r"[^\w]", "", text)


def ref_key(law_id, article) -> Optional[str]:
    if not law_id or not article:
        return None
    key = norm_article(article)
    return f"{law_id}::{key}" if key else None


def terminal_gold(item) -> set:
    """每条 gold path 末端的条款（必须靠第二/第三跳才能拿到）。"""
    out = set()
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
            out.add(arts[-1])
    return out


def usage_snapshot() -> Dict[str, int]:
    """读取当天 LLM 用量账本，用于汇报真实 token 消耗。"""
    path = os.path.join(USAGE_DIR, f"llm_usage_{datetime.now().strftime('%Y-%m-%d')}.jsonl")
    total = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
    if not os.path.exists(path):
        return total
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("ok", True):
                continue
            total["prompt_tokens"] += int(rec.get("prompt_tokens") or 0)
            total["completion_tokens"] += int(rec.get("completion_tokens") or 0)
            total["requests"] += 1
    return total


def extract_cited(answer: str, surfaced: set) -> set:
    """从答案正文提取「实际引用到的已提供条款」：只看正文表述，且只认 surfaced 内的条款（防编造得分）。"""
    text = unicodedata.normalize("NFKC", answer or "")
    tokens = set()
    for pattern in ARTICLE_TEXT_RES:
        for match in pattern.finditer(text):
            token = norm_article(re.sub(r"\(.*?\)", "", match.group(1)))
            if token:
                tokens.add(token)
    cited = set()
    for key in surfaced:
        _, _, article = key.partition("::")
        if article and article in tokens:
            cited.add(key)
    return cited


def load_cache() -> Dict:
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        return json.load(open(CACHE_PATH, encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache: Dict) -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def cached_arm(cache: Dict, key: str, build) -> Dict:
    """答案缓存：同一 id+arm 只调用一次 LLM，后续调整指标不再产生 token。"""
    hit = cache.get(key)
    if hit:
        return {"answer": hit["answer"], "surfaced": set(hit["surfaced"]),
                "route": hit.get("route", ""), "cached": True}
    arm = build()
    cache[key] = {"answer": arm["answer"], "surfaced": sorted(s for s in arm["surfaced"] if s),
                  "route": arm.get("route", "")}
    return arm


def consume_stream(gen) -> Dict:
    """消费问答流，聚合答案正文与来源。"""
    answer, sources, route = [], [], ""
    for ev in gen:
        kind = ev.get("type")
        if kind == "content":
            answer.append(ev.get("data") or "")
        elif kind == "sources":
            sources = ev.get("sources") or []
        elif kind == "route":
            route = ev.get("mode") or ""
        elif kind == "error":
            answer.append(f"[ERROR]{ev.get('data') or ''}")
    return {"answer": "".join(answer).strip(), "sources": sources, "route": route}


# ============================================================
# 各方案的作答
# ============================================================
def baseline_answer(qa: QAService, item) -> Dict:
    """生产快速 RAG：法域交给链路自动检测，与线上默认行为一致。"""
    res = consume_stream(qa.ask_stream(item["query"], top_k=5))
    surfaced = {ref_key(s.get("law_id"), s.get("article_number")) for s in res["sources"]}
    explicit = {
        ref_key(res["sources"][int(n) - 1].get("law_id"),
                res["sources"][int(n) - 1].get("article_number"))
        for n in SOURCE_REF_RE.findall(res["answer"])
        if 1 <= int(n) <= len(res["sources"])
    }
    cited = explicit | extract_cited(res["answer"], surfaced)
    return {"answer": res["answer"], "surfaced": surfaced, "cited": cited, "route": "fast"}


def multihop_answer(multihop: MultiHopService, kg: KGReasoningService, item) -> Dict:
    juris = item.get("jurisdictions")
    if item["type"] == "cross_jurisdiction":
        data = multihop.answer(item["query"], top_k_laws=3, synthesize=True,
                               jurisdictions=juris)
        surfaced, cited = set(), set()
        source = data.get("source") or {}
        key = ref_key(source.get("law_id"), source.get("article_number"))
        if key:
            surfaced.add(key)
        for target in data.get("targets") or []:
            for art in target.get("articles") or []:
                k = ref_key(target.get("law_id"), art.get("article_number"))
                if k:
                    surfaced.add(k)
        for ref in data.get("cited_articles") or []:
            law_id, _, art = str(ref).partition("::")
            k = ref_key(law_id, art)
            if k:
                cited.add(k)
        answer = data.get("answer") or ""
        cited |= extract_cited(answer, surfaced)
        return {"answer": answer, "surfaced": surfaced,
                "cited": cited, "route": "cross"}
    if item["type"] == "entity_navigation":
        data = kg.answer(item["query"], top_k_entities=5, hops=2, deepen=False,
                         synthesize=True, jurisdictions=juris)
        surfaced, cited = set(), set()
        for path in data.get("reasoning_path") or []:
            for step in path.get("steps") or []:
                ev = (step.get("evidence") or [{}])[0] or {}
                for field in ("article", "from_article", "to_article"):
                    k = ref_key(ev.get("law_id"), ev.get(field))
                    if k:
                        surfaced.add(k)
        for ref in data.get("cited_articles") or []:
            law_id, _, art = str(ref).partition("::")
            k = ref_key(law_id, art)
            if k:
                cited.add(k)
        answer = data.get("answer") or ""
        cited |= extract_cited(answer, surfaced)
        return {"answer": answer, "surfaced": surfaced,
                "cited": cited, "route": "entity"}
    return citation_answer(multihop.repo, item)


def citation_answer(repo: ElasticsearchRepository, item) -> Dict:
    """引用链：沿正文明确援引做确定性扩展，再把证据交给 LLM 合成（1 次调用）。"""
    source = item.get("source") or {}
    law_id, article = source.get("law_id"), source.get("article")
    content = repo.get_article_full_text(law_id, article)
    if not content:
        return {"answer": "[ERROR]源条款原文缺失", "surfaced": set(), "cited": set(),
                "route": "citation"}
    seeds = [(law_id, article, content)]
    refs: List[tuple] = [(law_id, article, content)]
    for _ in range(max(1, int(item.get("hops") or 2))):
        docs = repo.get_referenced_articles(seeds, max_refs=10)
        if not docs:
            break
        for d in docs:
            refs.append((d.get("law_id", ""), d.get("article_number", ""),
                         d.get("content", "") or ""))
        seeds = [(d.get("law_id", ""), d.get("article_number", ""),
                  d.get("content", "") or "") for d in docs]

    # 证据条数与字数上限：与生产 follow_references 口径接近，避免 prompt 失控
    evidence, seen = [], set()
    for lid, art, text in refs:
        key = (lid, art)
        if key in seen or not lid or not art:
            continue
        seen.add(key)
        evidence.append({"law_id": lid, "article": art, "text": (text or "")[:400]})
        if len(evidence) >= 8:
            break

    lines = [f"【证据{i}】{e['law_id']} 第{e['article']}条\n{e['text']}"
             for i, e in enumerate(evidence, start=1)]
    prompt = (
        "你是法律检索助手。只能依据下列条款片段回答问题，不得编造。\n\n"
        f"问题：{item['query']}\n\n" + "\n\n".join(lines) + "\n\n"
        "要求：\n1) 先给出直接结论；\n2) 每个结论后用【证据N】标注依据；\n"
        "3) 证据不足以回答时明确写「材料不足」，不要推测。"
    )
    try:
        llm = get_chat_model(model=settings.llm_extraction_model, temperature=0.1,
                             max_tokens=1024)
        answer = (llm.invoke(prompt).content or "").strip()
    except Exception as e:
        answer = f"[ERROR]{type(e).__name__}"
    cited = set()
    for n in EVIDENCE_REF_RE.findall(answer):
        idx = int(n) - 1
        if 0 <= idx < len(evidence):
            cited.add(ref_key(evidence[idx]["law_id"], evidence[idx]["article"]))
    surfaced = {ref_key(e["law_id"], e["article"]) for e in evidence}
    cited |= extract_cited(answer, surfaced)
    return {"answer": answer, "surfaced": surfaced, "cited": cited,
            "route": "citation"}


def agent_answer(agent: AgentService, item) -> Dict:
    """生产默认链路（Agent auto 路由）：用于单跳非劣化检验。"""
    res = consume_stream(agent.ask_stream(item["query"], top_k=5,
                                          route_mode="auto"))
    surfaced = {ref_key(s.get("law_id"), s.get("article_number")) for s in res["sources"]}
    explicit = {
        ref_key(res["sources"][int(n) - 1].get("law_id"),
                res["sources"][int(n) - 1].get("article_number"))
        for n in SOURCE_REF_RE.findall(res["answer"])
        if 1 <= int(n) <= len(res["sources"])
    }
    cited = explicit | extract_cited(res["answer"], surfaced)
    return {"answer": res["answer"], "surfaced": surfaced, "cited": cited,
            "route": res["route"] or "fast"}


# ============================================================
# 指标与统计
# ============================================================
def score_arm(arm: Dict, gold: set) -> Dict:
    answer = arm["answer"] or ""
    plain = answer.strip()
    # 「材料不足」只对某一法域如实说明时不应判为整篇未作答：
    # 短答案（<300 字）出现该表述视为未作答，长答案里有局部说明仍算作答。
    answered = 1 if len(plain) >= 80 else 0
    if answered and len(plain) < 300 and INSUFFICIENT_RE.search(plain):
        answered = 0
    surf_cov = len(gold & arm["surfaced"]) / len(gold) if gold else 0.0
    cite_cov = len(gold & arm["cited"]) / len(gold) if gold else 0.0
    return {
        "surf_cov": round(surf_cov, 4),
        "cite_cov": round(cite_cov, 4),
        "answered": answered,
        "full": 1 if cite_cov >= 1.0 and answered else 0,
    }


def paired_bootstrap(deltas: List[float], rounds: int = 2000, seed: int = 42):
    if not deltas:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(rng.choice(deltas) for _ in range(n)) / n for _ in range(rounds))
    return (round(sum(deltas) / n, 4),
            round(means[int(rounds * 0.025)], 4),
            round(means[int(rounds * 0.975) - 1], 4))


def sign_test(deltas: List[float]):
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return pos, neg, round(min(1.0, p), 4)


class _Verdict(BaseModel):
    winner: str
    reason: str


def judge_pair(question: str, gold: List[str], ans_a: str, ans_b: str) -> Dict:
    """两两裁判：只判条款准确性、结论完整性、无编造。A=baseline，B=multihop。"""
    prompt = (
        "你是法律问答评审。请比较两份回答在「条款准确性、结论完整性、是否编造」上的优劣。\n"
        f"问题：{question}\n标准答案应覆盖的条款：{', '.join(gold) or '（未标注）'}\n\n"
        f"【回答A】\n{ans_a[:1500]}\n\n【回答B】\n{ans_b[:1500]}\n\n"
        "输出 winner（A/B/tie）与一句理由。只选项，不要解释标准。"
    )
    try:
        llm = get_chat_model(model=settings.llm_extraction_model, temperature=0,
                             max_tokens=200)
        verdict = llm.with_structured_output(_Verdict, method="function_calling").invoke(prompt)
        winner = (verdict.winner or "").strip().upper()
        if winner not in ("A", "B", "TIE"):
            winner = "TIE"
        return {"winner": winner.lower(), "reason": (verdict.reason or "")[:120]}
    except Exception as e:
        return {"winner": "error", "reason": f"{type(e).__name__}"}


# ============================================================
# 数据集抽样
# ============================================================
def sample_single_hop(n: int = REGRESS_ITEMS, retriever=None) -> List[Dict]:
    """按法域分层抽单跳问题，并用零成本检索预筛只保留 gold 能进 top-5 的条目（否则测不出劣化）。"""
    items = json.load(open(SINGLE_PATH, encoding="utf-8"))
    by_jur: Dict[str, List[Dict]] = {}
    for it in items:
        if not it.get("relevant"):
            continue
        by_jur.setdefault(it.get("jurisdiction") or "未知", []).append(it)
    picked: List[Dict] = []
    jur_order = sorted(by_jur, key=lambda j: -len(by_jur[j]))
    round_idx = 0
    while len(picked) < n and round_idx < max(len(b) for b in by_jur.values()):
        for jur in jur_order:
            bucket = by_jur[jur]
            if round_idx >= len(bucket):
                continue
            it = bucket[round_idx]
            gold = {
                ref_key(r.get("law_id"), r.get("article"))
                for r in it.get("relevant") or [] if int(r.get("relevance", 1)) >= 2
            }
            if not gold:
                continue
            if retriever is not None:
                jur_filter = None if jur == "跨法域" else jur
                hits = retriever.hybrid_search(
                    query=it["query"], top_k=5, jurisdiction=jur_filter,
                    _decompose=True, _rerank=False,
                )
                surfaced = {
                    ref_key(h.get("law_id"), h.get("article_number")) for h in hits
                }
                if not (gold & surfaced):
                    continue
            it["_gold"] = gold
            picked.append(it)
            if len(picked) >= n:
                break
        round_idx += 1
    return picked


# ============================================================
# 主流程
# ============================================================
def run_multihop(judge: bool, max_judge: int, limit: int = 0,
                 only_type: str = "all") -> List[Dict]:
    items = json.load(open(MULTIHOP_PATH, encoding="utf-8"))["items"]
    if only_type and only_type != "all":
        items = [i for i in items if i["type"] == only_type]
    if limit:
        items = items[:limit]
    qa = QAService()
    repo = ElasticsearchRepository()
    multihop = MultiHopService(repo=repo)
    kg = KGReasoningService(repo=repo)

    rows = []
    cache = load_cache()
    for idx, item in enumerate(items, start=1):
        gold = terminal_gold(item)
        t0 = time.perf_counter()
        base = cached_arm(cache, f"{item['id']}|base", lambda: baseline_answer(qa, item))
        t_base = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        mh = cached_arm(cache, f"{item['id']}|mh",
                        lambda: multihop_answer(multihop, kg, item))
        t_mh = (time.perf_counter() - t0) * 1000
        base["cited"] = extract_cited(base["answer"], base["surfaced"]) | (base.get("cited") or set())
        mh["cited"] = extract_cited(mh["answer"], mh["surfaced"]) | (mh.get("cited") or set())

        s_base, s_mh = score_arm(base, gold), score_arm(mh, gold)
        row = {
            "id": item["id"], "type": item["type"], "hops": item["hops"],
            "query": item["query"], "gold": sorted(gold),
            "base": s_base, "mh": s_mh,
            "base_answer": base["answer"][:600], "mh_answer": mh["answer"][:600],
            "base_route": base["route"], "mh_route": mh["route"],
            "base_ms": round(t_base, 1), "mh_ms": round(t_mh, 1),
        }
        rows.append(row)
        print(f"  [{idx:02d}/{len(items)}] {item['id']:<6} {item['type']:<20} "
              f"cited {s_base['cite_cov']:.2f}->{s_mh['cite_cov']:.2f}  "
              f"full {s_base['full']}->{s_mh['full']}")

    # 分歧裁判：任一项指标不同才判，最多 max_judge 条；已有判决直接复用（不重复计费）
    prev_judges = {}
    if os.path.exists(OUT_JSON):
        try:
            prev_judges = {
                r["id"]: r.get("judge")
                for r in json.load(open(OUT_JSON, encoding="utf-8")).get("rows", [])
                if (r.get("judge") or {}).get("winner") not in (None, "error")
            }
        except (json.JSONDecodeError, OSError):
            prev_judges = {}
    judged = 0
    if judge:
        for row in rows:
            if row["id"] in prev_judges:
                row["judge"] = prev_judges[row["id"]]
                continue
            if judged >= max_judge:
                break
            b, m = row["base"], row["mh"]
            if b["cite_cov"] == m["cite_cov"] and b["answered"] == m["answered"]:
                continue
            verdict = judge_pair(row["query"], row["gold"],
                                 row["base_answer"], row["mh_answer"])
            row["judge"] = verdict
            judged += 1
            print(f"      judge {row['id']}: {verdict['winner']} ({verdict['reason']})")
    save_cache(cache)
    return rows


def run_regress(limit: int = 0) -> List[Dict]:
    from services.retriever_service import RetrieverService
    items = sample_single_hop(retriever=RetrieverService())
    if limit:
        items = items[:limit]
    qa = QAService()
    agent = AgentService()
    rows = []
    cache = load_cache()
    print(f"\n  单跳回归：{len(items)} 条（baseline 快速 RAG vs Agent auto 路由）")
    for idx, item in enumerate(items, start=1):
        gold = item["_gold"]
        t0 = time.perf_counter()
        base = cached_arm(cache, f"{item['id']}|base", lambda: baseline_answer(qa, item))
        t_base = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        treat = cached_arm(cache, f"{item['id']}|agent", lambda: agent_answer(agent, item))
        t_treat = (time.perf_counter() - t0) * 1000
        base["cited"] = extract_cited(base["answer"], base["surfaced"]) | (base.get("cited") or set())
        treat["cited"] = extract_cited(treat["answer"], treat["surfaced"]) | (treat.get("cited") or set())
        s_base, s_treat = score_arm(base, gold), score_arm(treat, gold)
        rows.append({
            "id": item["id"], "type": "single_hop", "hops": 1,
            "jurisdiction": item.get("jurisdiction", ""),
            "gold": sorted(gold), "base": s_base, "mh": s_treat,
            "base_answer": base["answer"][:600], "mh_answer": treat["answer"][:600],
            "mh_route": treat["route"],
            "base_ms": round(t_base, 1), "mh_ms": round(t_treat, 1),
        })
        print(f"  [{idx:02d}/{len(items)}] {item['id']:<10} {item.get('jurisdiction',''):<5} "
              f"cited {s_base['cite_cov']:.2f}->{s_treat['cite_cov']:.2f}  "
              f"route={treat['route']}")
    save_cache(cache)
    return rows


def summarize(rows: List[Dict], label: str) -> Dict:
    keys = ["surf_cov", "cite_cov", "answered", "full"]
    out = {"label": label, "n": len(rows)}
    for arm in ("base", "mh"):
        out[arm] = {k: round(sum(r[arm][k] for r in rows) / len(rows), 4) for k in keys}
    out["delta_cite"] = _delta_block(rows, "cite_cov")
    out["delta_full"] = _delta_block(rows, "full")
    return out


def _delta_block(rows: List[Dict], metric: str) -> Dict:
    deltas = [r["mh"][metric] - r["base"][metric] for r in rows]
    mean, lo, hi = paired_bootstrap(deltas)
    pos, neg, p = sign_test(deltas)
    return {"mean": mean, "ci95": [lo, hi], "sign": [pos, neg, p]}


def print_summary(summaries: List[Dict]) -> None:
    print("\n" + "=" * 88)
    for s in summaries:
        print(f"\n  [{s['label']}] n={s['n']}")
        print(f"    baseline  surf={s['base']['surf_cov']:.3f} cite={s['base']['cite_cov']:.3f} "
              f"answered={s['base']['answered']:.3f} full={s['base']['full']:.3f}")
        print(f"    multihop  surf={s['mh']['surf_cov']:.3f} cite={s['mh']['cite_cov']:.3f} "
              f"answered={s['mh']['answered']:.3f} full={s['mh']['full']:.3f}")
        for key, name in (("delta_cite", "引用覆盖"), ("delta_full", "完整作答")):
            d = s[key]
            print(f"    Δ{name}: {d['mean']:+.4f}  95%CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  "
                  f"胜/负={d['sign'][0]}/{d['sign'][1]}  p={d['sign'][2]}")
    print("=" * 88)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["multihop", "regress", "all"], default="all")
    parser.add_argument("--type", default="all",
                        help="只重跑某类多跳题（cross_jurisdiction / entity_navigation / citation_chain）")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--max-judge", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="调试用：只跑前 N 条")
    args = parser.parse_args()

    # 评测口径固化：关缓存、关 LLM 查询分解（规则分解足够，省 token 且降方差）
    settings.qa_answer_cache_enabled = False
    settings.search_cache_enabled = False
    settings.multihop_decompose_llm_enabled = False

    before = usage_snapshot()
    started = time.time()
    rows: List[Dict] = []
    summaries: List[Dict] = []

    print("=" * 88)
    print("  答案级 A/B（方案 B）")
    print("=" * 88)

    if args.suite in ("multihop", "all"):
        mh_rows = run_multihop(judge=not args.no_judge,
                               max_judge=args.max_judge, limit=args.limit,
                               only_type=args.type)
        rows.extend(mh_rows)
        for t in sorted({r["type"] for r in mh_rows}):
            summaries.append(summarize([r for r in mh_rows if r["type"] == t], t))
        summaries.append(summarize(mh_rows, "多跳总体"))

    if args.suite in ("regress", "all"):
        rg_rows = run_regress(limit=args.limit)
        rows.extend(rg_rows)
        summaries.append(summarize(rg_rows, "单跳回归"))

    after = usage_snapshot()
    delta = {k: after[k] - before[k] for k in before}

    # 只重跑部分题型时，用已有结果补齐其余条目，保证汇总仍是完整口径
    if os.path.exists(OUT_JSON):
        try:
            prev = json.load(open(OUT_JSON, encoding="utf-8")).get("rows", [])
            by_id = {r["id"]: r for r in prev}
            prev_ids = [r["id"] for r in prev]
            new_only = [r["id"] for r in rows if r["id"] not in set(prev_ids)]
            for r in rows:
                by_id[r["id"]] = r
            rows = [by_id[i] for i in prev_ids + new_only if i in by_id]
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    summaries = []
    mh_all = [r for r in rows if r["type"] != "single_hop"]
    single_all = [r for r in rows if r["type"] == "single_hop"]
    for t in sorted({r["type"] for r in mh_all}):
        summaries.append(summarize([r for r in mh_all if r["type"] == t], t))
    if mh_all:
        summaries.append(summarize(mh_all, "多跳总体"))
    if single_all:
        summaries.append(summarize(single_all, "单跳回归"))

    print_summary(summaries)

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "model": settings.llm_model,
            "decompose_llm": settings.multihop_decompose_llm_enabled,
            "judge": not args.no_judge,
        },
        "rows": rows,
        "summary": summaries,
        "usage": delta,
        "elapsed_s": round(time.time() - started, 1),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "type", "gold", "base_cite", "mh_cite", "base_full",
                         "mh_full", "base_answered", "mh_answered", "judge", "mh_route"])
        for r in rows:
            writer.writerow([r["id"], r["type"], "|".join(r["gold"]),
                             r["base"]["cite_cov"], r["mh"]["cite_cov"],
                             r["base"]["full"], r["mh"]["full"],
                             r["base"]["answered"], r["mh"]["answered"],
                             (r.get("judge") or {}).get("winner", ""), r.get("mh_route", "")])

    print(f"\n  实际消耗：prompt={delta['prompt_tokens']} completion={delta['completion_tokens']} "
          f"requests={delta['requests']}  耗时={result['elapsed_s']}s")
    print(f"  已写入：{OUT_JSON}")
    print(f"  已写入：{OUT_CSV}")


if __name__ == "__main__":
    main()
