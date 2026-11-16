#!/usr/bin/env python3
"""Gold v2 数据集构建器。

规则版本：gold-v2.0，设计说明见 docs/eval_gold_v2_design.md。
只读 raw_v2 / law_catalog / taxonomy_merged / 旧评测文件，只写
legal_rag/evals/datasets/v2/，不读任何检索结果。

用法：
    python build_eval_v2.py --stage stats
    python build_eval_v2.py --stage build
    python build_eval_v2.py --stage check
    python build_eval_v2.py --stage audit
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

LEGAL_RAG = Path(__file__).resolve().parents[3]
DATA = LEGAL_RAG / "data"
DATASETS = LEGAL_RAG / "evals" / "datasets"
OUT_DIR = DATASETS / "v2"

RULES_VERSION = "gold-v2.2"
SEED = 20261114
MIN_SCORE = 6.0
R2_SCORE = 9.0
R2_STRONG = 2
MAX_TEXT_LEN = 3000
QUERY_CAP = {"F1": 8, "F2": 5, "F3": 12, "legacy": 10}
PER_LAW_CAP = {"F1": 2, "F2": 5, "F3": 2, "legacy": 2}
MIN_CELL_GOLD = 4

# 按语料条款数降序，同时作为选取顺序，保证确定性
JURIS = [
    "中国大陆", "中国澳门", "澳大利亚", "美国", "日本", "中国香港",
    "泰国", "韩国", "新加坡", "越南", "马来西亚",
]
JUR_CODE = {
    "中国大陆": "CN", "中国香港": "HK", "中国澳门": "MO", "美国": "US",
    "日本": "JP", "新加坡": "SG", "泰国": "TH", "越南": "VN",
    "马来西亚": "MY", "澳大利亚": "AU", "韩国": "KR",
}
JUR_EN = {
    "中国大陆": "mainland China", "中国香港": "Hong Kong", "中国澳门": "Macao",
    "美国": "the United States", "日本": "Japan", "新加坡": "Singapore",
    "泰国": "Thailand", "越南": "Vietnam", "马来西亚": "Malaysia",
    "澳大利亚": "Australia", "韩国": "South Korea",
}
LEGACY_JUR = {"香港": "中国香港", "澳门": "中国澳门"}
LEGACY_UNMAPPED = {
    "数据保护与企业合规", "贸易竞争与市场监管", "宪法与基本权利",
    "电信与信息技术", "电信与数字通信",
}

COMMENCEMENT_RE = re.compile(
    r"^第[一二三四五六七八九十百千零〇\d]+条\s*[^。]{0,60}(自|从)[^。]{0,40}(起)?(施行|实施)"
)
DEFINITION_RE = re.compile(r"下列用语的含义|本办法所称|本条例所称|本法所称")
BOILERPLATE_MARKS = ("首页", "关于我们", "联系我们", "版权所有", "网站地图", "为您服务")
SHORT_TITLE_RE = re.compile(
    r"(?i)^\s*\d*\.?\s*(this act is the|this act may be cited as|"
    r"this (regulation|order|code|notification) (is|may be cited as))")
MAX_TITLE_LEN = 160
REFERRAL_SENT_RE = re.compile(
    r"[^。；;]{0,60}(构成犯罪|涉嫌犯罪|追究[^。；;]{0,6}刑事责任)[^。；;]{0,60}[。；;]?"
)
TITLE_SUFFIX_RE = re.compile(r"\s*\(title(?:,\s*\+?\d+)?\)\s*$", re.I)
WEAK_SUFFIX_RE = re.compile(r"\s*\(weak\)\s*$", re.I)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def normalize_keyword(kw: str) -> str:
    kw = TITLE_SUFFIX_RE.sub("", kw)
    kw = WEAK_SUFFIX_RE.sub("", kw)
    return kw.strip().lower()


def strong_terms(matched) -> set:
    return {normalize_keyword(k) for k in matched if "(weak)" not in k}


def referral_only(text: str, terms: set) -> bool:
    """主题强词只出现在「构成犯罪/追究刑事责任」引用句中。"""
    if not text or not terms or not REFERRAL_SENT_RE.search(text):
        return False
    stripped = REFERRAL_SENT_RE.sub("", text).lower()
    return not any(t and t in stripped for t in terms)


def load_taxonomy():
    raw = json.loads((DATA / "taxonomy_merged.json").read_text(encoding="utf-8"))
    topics = {t["id"]: t for t in raw["topics"]}
    groups = defaultdict(list)
    for tid, t in topics.items():
        groups[t["label_zh"]].append(tid)
    for label in groups:
        groups[label].sort()
    return raw, topics, groups


def load_catalog():
    raw = json.loads((DATA / "law_catalog.json").read_text(encoding="utf-8"))
    return raw, {x["law_id"]: x for x in raw["laws"]}


# 外部法规清单 K 里的截断 ID：新加坡法规抽取时把「This Act is the …」当成 law_id。
# 形如 SG_ACT_This_Act_is_the_Pers_2012，前缀 token + 年份可唯一反查语料 law_name。
K_TRUNC_RE = re.compile(r"^([A-Z]{2})_ACT_This_Act_is_the_([A-Za-z]+)_(\d{4})$")


def repair_k_id(law_id, catalog, law_names):
    """返回语料中唯一匹配的 law_id；无法唯一匹配时返回 None。"""
    m = K_TRUNC_RE.match(law_id)
    if not m:
        return None
    code, token, year = m.groups()
    hits = []
    for lid, name in law_names.items():
        if lid not in catalog or not lid.startswith(code + "_") or not name:
            continue
        low = re.sub(r"\s+", " ", name.lower())
        if token.lower() in low and re.search(rf"(?<!\d){year}(?!\d)", low):
            hits.append(lid)
    return hits[0] if len(hits) == 1 else None


def repair_k_set(ids, catalog, law_names):
    """K 的截断 ID 修复后去重；无法修复或不在目录内的条目丢弃。"""
    out = []
    for lid in ids:
        if lid in catalog:
            out.append(lid)
        else:
            fixed = repair_k_id(lid, catalog, law_names)
            if fixed:
                out.append(fixed)
    return set(out)


def exclusion_reason(rec, text: str):
    """R1：返回命中原因；None 表示保留。顺序即优先级，供 stats 逐项计数。"""
    if not rec.get("is_substantive"):
        return "not_substantive"
    if rec.get("chunk_type") != "substantive":
        return "chunk_type"
    if str(rec.get("article_number", "")).lower().startswith("sch"):
        return "schedule"
    n = len(text)
    if n > MAX_TEXT_LEN:
        return "too_long"
    if n > 800 and "附件" in text:
        return "annex"
    if n > 800 and COMMENCEMENT_RE.match(text):
        return "commencement"
    if sum(1 for m in BOILERPLATE_MARKS if m in text) >= 3:
        return "boilerplate"
    if DEFINITION_RE.search(text[:120]):
        return "definition"
    if n <= MAX_TITLE_LEN and SHORT_TITLE_RE.match(text):
        return "short_title"
    return None


def law_name_ok(name: str) -> bool:
    name = (name or "").strip()
    if len(name) < 4 or name.endswith("。") or name.startswith("第"):
        return False
    if "制定本法律" in name or "首页" in name:
        return False
    return True


class Corpus:
    """按 (法域, 主题) 索引条款候选，只依赖语料字段。"""

    def __init__(self, topics):
        self.topics = topics
        self.by_cell = defaultdict(list)      # (jur, topic_id) -> [cand]
        self.articles = {}                     # (law_id, article) -> rec
        self.law_articles = defaultdict(set)   # law_id -> {article}
        self.articles_per_jur = Counter()
        self.laws = set()
        self.law_names = {}                    # law_id -> 语料 law_name
        self.exclusion_counts = Counter()
        self.pairs_score_pass = 0              # R1–R3 后的「条款×主题」对
        self.pairs_top_only = 0                # R4 后的对
        self.raw_label_by_topic = defaultdict(set)
        self.max_cleaned_at = ""
        self._load()

    def _load(self):
        root = DATA / "raw_v2"
        for path in sorted(root.glob("*/*.json")):
            records = json.loads(path.read_text(encoding="utf-8"))
            for rec in records:
                jur = rec["jurisdiction"]
                law_id = rec["law_id"]
                art = rec["article_number"]
                text = rec.get("text_zh") or rec.get("text_en") or ""
                self.articles_per_jur[jur] += 1
                self.laws.add(law_id)
                self.law_articles[law_id].add(art)
                if law_id not in self.law_names and rec.get("law_name"):
                    self.law_names[law_id] = rec["law_name"]
                if rec.get("cleaned_at") and rec["cleaned_at"] > self.max_cleaned_at:
                    self.max_cleaned_at = rec["cleaned_at"]
                reason = exclusion_reason(rec, text)
                if reason:
                    self.exclusion_counts[reason] += 1
                    continue
                self.articles[(law_id, art)] = rec
                text_sha = sha256_text(normalize_text(text)) if text else ""
                tds = []
                for td in rec.get("topics_details") or []:
                    tid = td.get("id")
                    if tid not in self.topics:
                        continue
                    score = float(td.get("score") or 0)
                    if score < MIN_SCORE:
                        continue
                    self.raw_label_by_topic[tid].add(td.get("label_zh"))
                    self.pairs_score_pass += 1
                    tds.append((td, score))
                if not tds:
                    continue
                top_score = max(score for _, score in tds)
                for td, score in tds:
                    if score < top_score:
                        continue
                    tid = td.get("id")
                    self.pairs_top_only += 1
                    name = rec.get("law_name") or law_id
                    ok = law_name_ok(name)
                    terms = strong_terms(td.get("matched_keywords") or [])
                    self.by_cell[(jur, tid)].append({
                        "jurisdiction": jur,
                        "law_id": law_id,
                        "law_name": name if ok else law_id,
                        "law_name_ok": ok,
                        "article": art,
                        "text": text,
                        "text_sha": text_sha,
                        "topic_id": tid,
                        "score": score,
                        "n_strong": len(terms),
                        "terms": terms,
                        "referral_only": referral_only(text, terms),
                        "matched": list(td.get("matched_keywords") or []),
                    })

    def cell_candidates(self, jur, topic_ids):
        cands = []
        for tid in topic_ids:
            cands.extend(self.by_cell.get((jur, tid), []))
        return cands


def rank_key(c):
    return (c["referral_only"], -c["n_strong"], -c["score"], c["law_id"], c["article"])


def select_gold(cands, family):
    cap = QUERY_CAP[family]
    per_law = PER_LAW_CAP[family]
    out, used_text, per_law_count = [], set(), Counter()
    for c in sorted(cands, key=rank_key):
        if len(out) >= cap:
            break
        if per_law_count[c["law_id"]] >= per_law:
            continue
        if c["text_sha"] and c["text_sha"] in used_text:
            continue
        used_text.add(c["text_sha"])
        per_law_count[c["law_id"]] += 1
        out.append(c)
    return out


def relevance_of(c) -> int:
    if c["referral_only"]:
        return 1
    return 2 if (c["n_strong"] >= R2_STRONG and c["score"] >= R2_SCORE) else 1


def fmt_gold(cands):
    return [{
        "law_id": c["law_id"],
        "law_name": c["law_name"],
        "article": c["article"],
        "relevance": relevance_of(c),
        "topic_id": c["topic_id"],
        "n_strong": c["n_strong"],
        "score": round(c["score"], 2),
    } for c in cands]


def strip_brackets(name: str) -> str:
    return name.strip().strip("《》").strip()


def make_variants(family, jur=None, topic_zh="", topic_en="", law_name="",
                  jurs_extra=None):
    if family == "F1":
        return {
            "A_keyword_zh": f"{jur} {topic_zh} 法律规定",
            "B_nl_zh": f"{jur}法律体系中关于「{topic_zh}」的核心规定有哪些？",
            "C_mixed_zhen": f"{jur} {topic_en} 相关法律规定",
            "D_pure_en": f"What are the key legal provisions on {topic_en} in {JUR_EN[jur]}?",
        }
    if family == "F2":
        name = strip_brackets(law_name)
        return {
            "A_keyword_zh": f"{name} {topic_zh} 规定",
            "B_nl_zh": f"《{name}》对「{topic_zh}」有哪些具体规定？",
            "C_mixed_zhen": f"{name} 中 {topic_en} provisions",
            "D_pure_en": f"What does {name} provide regarding {topic_en}?",
        }
    if family == "F3":
        jur_a, jur_b = jurs_extra
        return {
            "A_keyword_zh": f"{jur_a} {jur_b} {topic_zh} 对比",
            "B_nl_zh": f"{jur_a}与{jur_b}在「{topic_zh}」方面的立法规定分别是什么？",
            "C_mixed_zhen": f"{jur_a} vs {jur_b} {topic_en} comparison",
            "D_pure_en": (f"How do {JUR_EN[jur_a]} and {JUR_EN[jur_b]} "
                          f"regulate {topic_en}?"),
        }
    raise ValueError(family)


def build_cells(corpus, topics, groups):
    """返回 (jur, label_zh) 的可行单元统计。"""
    cells = {}
    for jur in JURIS:
        for label, ids in groups.items():
            cands = corpus.cell_candidates(jur, ids)
            if not cands:
                continue
            gold = select_gold(cands, "F1")
            if len(gold) >= MIN_CELL_GOLD:
                cells[(jur, label)] = {
                    "topic_ids": ids,
                    "label_zh": label,
                    "label_en": topics[ids[0]]["label_en"],
                    "gold": gold,
                    "n_laws": len({c["law_id"] for c in gold}),
                }
    return cells


def plan_queries(corpus, topics, groups, cells):
    feasible_jurs = defaultdict(set)
    for (jur, label) in cells:
        feasible_jurs[label].add(jur)
    tier = {}
    for label, js in feasible_jurs.items():
        n = len(js)
        tier[label] = ("rare" if n <= 2 else ("mid" if n <= 5 else "common"))

    used_f1 = Counter()
    plan = []

    # F1：每法域最多 4 条，先按层各取 1，再按 gold 数补齐
    for jur in JURIS:
        pool = [(label, d) for (j, label), d in cells.items() if j == jur]
        selected = []
        for want_tier in ("rare", "mid", "common"):
            cands = [x for x in pool if tier[x[0]] == want_tier
                     and x[0] not in {s[0] for s in selected}]
            if not cands:
                continue
            cands.sort(key=lambda x: (used_f1[x[0]], -len(x[1]["gold"]), x[0]))
            selected.append(cands[0])
        rest = [x for x in pool if x[0] not in {s[0] for s in selected}]
        rest.sort(key=lambda x: (used_f1[x[0]], -len(x[1]["gold"]), x[0]))
        selected.extend(rest[:max(0, 4 - len(selected))])
        selected = selected[:4]
        for label, data in selected:
            used_f1[label] += 1
            plan.append({
                "family": "F1", "jurisdiction": jur, "jurisdictions": [jur],
                "topic_label": label, "topic_ids": data["topic_ids"],
                "topic_en": data["label_en"],
                "gold": select_gold(corpus.cell_candidates(jur, data["topic_ids"]), "F1"),
            })

    # F2：每法域 2 条，F1 不足时补到最多 4 条；优先未在 F1 用过的主题
    for jur in JURIS:
        f1_topics = {p["topic_label"] for p in plan
                     if p["family"] == "F1" and p["jurisdiction"] == jur}
        pairs = []
        for label, ids in groups.items():
            for law_id in sorted({c["law_id"] for tid in ids
                                  for c in corpus.by_cell.get((jur, tid), [])}):
                cands = [c for tid in ids for c in corpus.by_cell.get((jur, tid), [])
                         if c["law_id"] == law_id]
                gold = select_gold(cands, "F2")
                if len(gold) >= MIN_CELL_GOLD and gold[0]["law_name_ok"]:
                    pairs.append({
                        "label": label, "topic_ids": ids, "law_id": law_id,
                        "law_name": gold[0]["law_name"], "gold": gold,
                    })
        pairs.sort(key=lambda p: (p["label"] in f1_topics, -len(p["gold"]),
                                  p["law_id"]))
        need = 6 - sum(1 for p in plan if p["jurisdiction"] == jur)
        need = min(max(need, 2), 4)
        for p in pairs[:need]:
            plan.append({
                "family": "F2", "jurisdiction": jur, "jurisdictions": [jur],
                "topic_label": p["label"], "topic_ids": p["topic_ids"],
                "topic_en": topics[p["topic_ids"][0]]["label_en"],
                "law_id": p["law_id"], "law_name": p["law_name"],
                "gold": p["gold"],
            })

    # F3：跨法域，主题在 >=3 个法域可行，取前 8
    cross = [(label, js) for label, js in feasible_jurs.items() if len(js) >= 3]
    cross.sort(key=lambda x: (-len(x[1]), x[0]))
    for label, js in cross[:8]:
        ranked = sorted(js, key=lambda j: (-len(cells[(j, label)]["gold"]),
                                           JURIS.index(j)))
        chosen = ranked[:2]
        cands = [c for j in chosen
                 for tid in groups[label]
                 for c in corpus.by_cell.get((j, tid), [])]
        gold = select_gold(cands, "F3")
        if len(gold) < MIN_CELL_GOLD:
            continue
        plan.append({
            "family": "F3", "jurisdiction": "跨法域",
            "jurisdictions": chosen, "topic_label": label,
            "topic_ids": groups[label],
            "topic_en": topics[groups[label][0]]["label_en"],
            "gold": gold,
        })
    return plan, feasible_jurs, tier


def build_template_queries(plan, topics):
    counter = defaultdict(int)
    queries, dataset = [], []
    for item in plan:
        fam = item["family"]
        code = "X" if fam == "F3" else JUR_CODE[item["jurisdiction"]]
        counter[(code, fam)] += 1
        qid = f"T{code}_{fam}_{counter[(code, fam)]:02d}"
        if fam == "F3":
            variants = make_variants(fam, topic_zh=item["topic_label"],
                                     topic_en=item["topic_en"],
                                     jurs_extra=item["jurisdictions"])
        elif fam == "F2":
            variants = make_variants(fam, jur=item["jurisdiction"],
                                     topic_zh=item["topic_label"],
                                     topic_en=item["topic_en"],
                                     law_name=item["law_name"])
        else:
            variants = make_variants(fam, jur=item["jurisdiction"],
                                     topic_zh=item["topic_label"],
                                     topic_en=item["topic_en"])
        gold = fmt_gold(item["gold"])
        gold_laws = sorted({g["law_id"] for g in gold})
        base = {
            "query_id": qid, "subset": "template", "family": fam,
            "jurisdiction": item["jurisdiction"],
            "jurisdictions": item["jurisdictions"],
            "topic": item["topic_label"], "topic_id": item["topic_ids"][0],
            "topic_ids": item["topic_ids"], "core_concept": item["topic_label"],
            "is_core": False, "gold_law_source": "derived",
            "gold_clause_status": "derived", "clause_gold_empty": False,
        }
        if fam == "F2":
            base["anchor_law_id"] = item["law_id"]
        base.update(variants)
        queries.append(base)
        for v in "ABCD":
            text = variants[{
                "A": "A_keyword_zh", "B": "B_nl_zh",
                "C": "C_mixed_zhen", "D": "D_pure_en"}[v]]
            dataset.append({
                "id": f"{qid}_{v}", "query_id": qid, "query": text,
                "subset": "template", "family": fam,
                "jurisdiction": item["jurisdiction"], "topic": item["topic_label"],
                "topic_id": item["topic_ids"][0],
                "gold_law_ids": gold_laws, "gold_law_source": "derived",
                "gold_clause_status": "derived", "clause_gold_empty": False,
                "relevant": gold,
            })
    return queries, dataset


def load_legacy_queries():
    raw = json.loads((DATASETS / "eval_queries.json").read_text(encoding="utf-8"))
    return raw


def load_k_column():
    import openpyxl
    wb = openpyxl.load_workbook(DATASETS / "eval_annotation_template.xlsx",
                                data_only=True)
    ws = wb["标注总表"]
    out = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not row[0]:
            continue
        qid = str(row[0]).strip()
        ids = [x.strip() for x in str(row[10] or "").replace("\n", ";").split(";")
               if x.strip()]
        out[qid] = ids
    return out


def build_legacy(corpus, topics, groups, catalog):
    raw_queries = load_legacy_queries()
    k_col = load_k_column()
    queries, dataset = [], []
    stats = {"k_entries": 0, "k_distinct": 0, "k_in_catalog_entries": 0,
             "k_in_catalog_distinct": 0, "k_repaired": [], "k_deduped": [],
             "k_dropped": [], "unmappable": [], "empty_clause": []}
    all_k = []
    final_k = set()
    for q in raw_queries["queries"]:
        qid = q["query_id"]
        topic_label = q["topic"]
        jur = LEGACY_JUR.get(q["jurisdiction"], q["jurisdiction"])
        ids = groups.get(topic_label, [])
        k_ids = []
        for raw_id in k_col.get(qid, []):
            stats["k_entries"] += 1
            all_k.append(raw_id)
            law_id = raw_id
            if law_id not in catalog:
                fixed = repair_k_id(raw_id, catalog, corpus.law_names)
                if fixed:
                    stats["k_repaired"].append(
                        {"query_id": qid, "from": raw_id, "to": fixed})
                    law_id = fixed
            if law_id not in catalog:
                stats["k_dropped"].append({"query_id": qid, "law_id": raw_id})
                continue
            stats["k_in_catalog_entries"] += 1
            if law_id in k_ids:
                stats["k_deduped"].append({"query_id": qid, "law_id": law_id})
                continue
            k_ids.append(law_id)
            final_k.add(law_id)
        if not ids:
            gold, status = [], "unmappable_topic"
            stats["unmappable"].append(qid)
        else:
            scope = JURIS if jur == "跨法域" else [jur]
            cands = [c for j in scope for tid in ids
                     for c in corpus.by_cell.get((j, tid), [])]
            gold = select_gold(cands, "legacy")
            status = "derived" if gold else "empty_no_topic_match"
            if not gold:
                stats["empty_clause"].append(qid)
        base = {
            "query_id": qid, "subset": "legacy", "family": "legacy",
            "jurisdiction": q["jurisdiction"], "jurisdictions": [jur],
            "topic": topic_label,
            "topic_id": ids[0] if ids else None, "topic_ids": ids,
            "core_concept": q["core_concept"], "is_core": q["is_core"],
            "gold_law_source": "external_k_list",
            "gold_clause_status": status,
            "clause_gold_diagnostic_only": True,
            "clause_gold_empty": not gold,
        }
        for v, k in (("A", "A_keyword_zh"), ("B", "B_nl_zh"),
                     ("C", "C_mixed_zhen"), ("D", "D_pure_en")):
            base[k] = q[k]
        queries.append(base)
        gold_payload = fmt_gold(gold)
        for v, k in (("A", "A_keyword_zh"), ("B", "B_nl_zh"),
                     ("C", "C_mixed_zhen"), ("D", "D_pure_en")):
            dataset.append({
                "id": f"{qid}_{v}", "query_id": qid, "query": q[k],
                "subset": "legacy", "family": "legacy",
                "jurisdiction": q["jurisdiction"], "topic": topic_label,
                "topic_id": ids[0] if ids else None,
                "gold_law_ids": k_ids, "gold_law_source": "external_k_list",
                "gold_clause_status": status,
                "clause_gold_diagnostic_only": True,
                "clause_gold_empty": not gold_payload,
                "relevant": gold_payload,
            })
    stats["k_distinct"] = len(set(all_k))
    stats["k_in_catalog_distinct"] = len(final_k)
    return queries, dataset, stats


def corpus_digest():
    root = DATA / "raw_v2"
    per_file = []
    for path in sorted(root.glob("*/*.json")):
        per_file.append(f"{path.relative_to(root).as_posix()}:{sha256_file(path)}")
    return sha256_text("\n".join(per_file)), len(per_file)


def known_defects():
    return [
        {"id": "D1", "title": "查询主题标签与条款主题标注共用 taxonomy 词表",
         "impact": "指标度量主题可达性，非完整语义相关性"},
        {"id": "D2", "title": "topics_details.score 未归一化且与条文长度正相关",
         "impact": "长条文仍更易进入 gold；已用 n_strong 排序与长度上限压制"},
        {"id": "D3", "title": "主题标注漏检不可度量",
         "impact": "金币只有正例，Recall 绝对上限未知"},
        {"id": "D4", "title": "gold 条数上限造成 recall 天花板",
         "impact": "F1 8 / F2 5 / F3 12 / legacy 10"},
        {"id": "D5", "title": "低资源法域样本过少",
         "impact": "新加坡 4 条、越南 4 条、马来西亚 6 条基础查询，分法域指标不可比"},
        {"id": "D6", "title": "legacy 查询概念粒度粗于主题",
         "impact": "条款级 gold 错配，legacy 仅用于法规级指标与新旧对照"},
        {"id": "D7", "title": "主题标签重复/近义",
         "impact": "政府采购双 ID 合并，金融监管与金融服务不合并"},
        {"id": "D8", "title": "条文重复与异常编号",
         "impact": "按文本哈希去重；article_number 存在 Sch1/44ZZ/63F 等"},
        {"id": "D9", "title": "中国大陆/澳门法规在 D 变体中保留中文/葡文原名",
         "impact": "语料无官方英译，D 变体实为混合语言"},
        {"id": "D10", "title": "中国大陆、澳大利亚、韩国无法规级外部清单 gold",
         "impact": "其 gold_law_source 为 derived，无外部法规清单背书"},
        {"id": "D11", "title": "主题标注器的召回不足",
         "impact": ("日本等英文译本条款大量缺少 topics_details；新条款 gold 只覆盖 K 的 "
                    "36/128 部法规（28.1%），50/80 条 legacy 查询零覆盖（见 selfcheck）")},
        {"id": "D12", "title": "过滤规则损失部分 K 法规在检索候选中出现过的条款",
         "impact": ("长度>3000 等排除规则使 15.6% 的 K 法规候选条款不进入候选池；"
                    "换来的是避免长附件/修订附表霸榜")},
        {"id": "D13", "title": "两个主题标签在标注与出题时不同名",
         "impact": ("raw_v2 的 topics_details.label_zh 为「金融服务」「电信」，"
                    "taxonomy 2.5 为「金融服务与监管」「电信与通讯」；"
                    "gold 按 topic_id 选取不受影响，但查询词与条款词表的词汇匹配变难，"
                    "属跨系统比较的混杂因素")},
        {"id": "D14", "title": "K 列 1 个目录外 ID 无法修复",
         "impact": ("Q006 的 US_CFR_T48_P1_1983 在语料中不存在（FAR 只按 Part 拆分入库），"
                    "该法规不计入 Q006 的法规级 gold；其余 9 个目录外 ID 已修复或去重")},
        {"id": "D15", "title": "模板查询只有正例、无负例",
         "impact": "条款级指标只能算 Recall / NDCG，不能算精确率"},
    ]


def cmd_stats(corpus, topics, groups, cells):
    print(f"articles_total: {sum(corpus.articles_per_jur.values())}")
    print(f"articles_retained: {len(corpus.articles)}")
    for reason, n in corpus.exclusion_counts.most_common():
        print(f"  excluded[{reason}]: {n}")
    print(f"pairs_after_R1_R3: {corpus.pairs_score_pass}")
    print(f"pairs_after_R4: {corpus.pairs_top_only}")
    print(f"pairs_dropped_by_R4: {corpus.pairs_score_pass - corpus.pairs_top_only}")
    print(f"topic-article pairs: {sum(len(v) for v in corpus.by_cell.values())}")
    feasible = defaultdict(int)
    for (jur, label) in cells:
        feasible[label] += 1
    print(f"feasible cells: {len(cells)}")
    print(f"topics with >=1 feasible cell: {len(feasible)} / {len(groups)}")
    for jur in JURIS:
        n = sum(1 for (j, _) in cells if j == jur)
        print(f"  cells {jur}\t{n}")
    for label, n in sorted(feasible.items(), key=lambda x: -x[1]):
        print(f"  {label}\t{n}")


def topic_label_drift(corpus, topics, queries):
    """R13：raw_v2 的 topics_details.label_zh 与 taxonomy 2.5 不同名的主题。"""
    out = []
    for tid, raw_labels in sorted(corpus.raw_label_by_topic.items()):
        old = sorted(l for l in raw_labels if l and l != topics[tid]["label_zh"])
        if not old:
            continue
        out.append({
            "topic_id": tid,
            "taxonomy_label": topics[tid]["label_zh"],
            "raw_v2_labels": old,
            "affected_queries": sorted(
                q["query_id"] for q in queries
                if tid in (q.get("topic_ids") or [])),
        })
    return out


def cmd_build(corpus, topics, groups, catalog):
    cells = build_cells(corpus, topics, groups)
    plan, feasible_jurs, tier = plan_queries(corpus, topics, groups, cells)
    tpl_queries, tpl_dataset = build_template_queries(plan, topics)
    lg_queries, lg_dataset, lg_stats = build_legacy(corpus, topics, groups, catalog)
    queries = lg_queries + tpl_queries
    dataset = lg_dataset + tpl_dataset
    tax_version = json.loads(
        (DATA / "taxonomy_merged.json").read_text(encoding="utf-8"))["taxonomy_version"]
    drift = topic_label_drift(corpus, topics, queries)

    base_entries = [d for d in dataset if d["id"].endswith("_A")]
    empty_base = [d for d in base_entries if not d["relevant"]]
    empty_manifest = [{
        "query_id": d["query_id"], "subset": d["subset"],
        "jurisdiction": d["jurisdiction"], "topic": d["topic"],
        "gold_law_ids": d["gold_law_ids"],
        "gold_clause_status": d["gold_clause_status"],
        "reason": ("主题标签无法映射到 taxonomy" if d["gold_clause_status"]
                   == "unmappable_topic" else "该主题在查询法域内无 score≥6.0 的合格条款"),
    } for d in empty_base]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eval_gold_empty_queries.json").write_text(
        json.dumps({
            "note": (f"这 {len(empty_base)} 条基础查询（{len(empty_base) * 4} 条变体）"
                     f"的条款级 gold 为空，条款级指标分母应为 "
                     f"{len(dataset)}-{len(empty_base) * 4}="
                     f"{len(dataset) - len(empty_base) * 4}；法规级 gold 仍保留。"),
            "empty_base_queries": len(empty_base),
            "empty_variants": len(empty_base) * 4,
            "items": empty_manifest,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    q_doc = {
        "meta": {
            "dataset_version": "v2.0",
            "rules_version": RULES_VERSION,
            "total_base_queries": len(queries),
            "total_variants": len(dataset),
            "subset_counts": {
                "legacy": sum(1 for q in queries if q["subset"] == "legacy"),
                "template": sum(1 for q in queries if q["subset"] == "template"),
            },
            "family_counts": dict(Counter(q["family"] for q in queries)),
            "variants": {
                "A_keyword_zh": "纯中文关键词查询",
                "B_nl_zh": "纯中文自然语言查询",
                "C_mixed_zhen": "中英混合查询",
                "D_pure_en": "纯英文查询",
            },
            "annotation_level": ("法规级(外部法规清单 K / 派生) + "
                                 "条款级(机械派生, relevance 1/2)"),
            "topic_source": f"taxonomy_merged.json v{tax_version}",
            "topic_label_drift": drift,
            "seed": SEED,
        },
        "queries": queries,
    }
    (OUT_DIR / "eval_queries_v2.json").write_text(
        json.dumps(q_doc, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT_DIR / "eval_dataset_v2.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=1), encoding="utf-8")

    old_files = [DATASETS / "eval_dataset.json", DATASETS / "eval_dataset_clause.json",
                 DATASETS / "eval_queries.json",
                 DATASETS / "eval_annotation_template.xlsx"]
    corpus_hash, corpus_files = corpus_digest()
    manifest = {
        "rules_version": RULES_VERSION,
        "seed": SEED,
        "dataset_sha256": canonical_hash(dataset),
        "queries_sha256": canonical_hash(q_doc),
        "corpus": {
            "articles": sum(corpus.articles_per_jur.values()),
            "articles_retained": len(corpus.articles),
            "laws": len(corpus.laws),
            "jurisdictions": len(corpus.articles_per_jur),
            "max_cleaned_at": corpus.max_cleaned_at,
            "files": corpus_files,
            "digest_sha256": corpus_hash,
            "articles_per_jurisdiction": dict(corpus.articles_per_jur),
            "exclusions": dict(corpus.exclusion_counts),
            "pairs_after_R1_R3": corpus.pairs_score_pass,
            "pairs_after_R4": corpus.pairs_top_only,
        },
        "taxonomy": {
            "version": tax_version,
            "file_sha256": sha256_file(DATA / "taxonomy_merged.json"),
            "topics": len(topics),
            "label_drift": drift,
        },
        "inputs": {p.name: sha256_file(p) for p in old_files},
        "model": None,
        "counts": {
            "base_queries": len(queries),
            "variants": len(dataset),
            "base_queries_with_clause_gold": len(base_entries) - len(empty_base),
            "clause_gold_coverage": f"{len(base_entries) - len(empty_base)}/{len(base_entries)}",
            "empty_base_queries": len(empty_base),
            "empty_gold_variants": sum(1 for d in dataset if not d["relevant"]),
            "gold_clauses": sum(len(d["relevant"]) for d in dataset),
            "gold_clauses_base": sum(len(d["relevant"]) for d in base_entries),
            "gold_clauses_unique": len({
                (g["law_id"], g["article"]) for d in dataset for g in d["relevant"]}),
            "per_jurisdiction_template": dict(Counter(
                d["jurisdiction"] for d in dataset
                if d["id"].endswith("_A") and d["subset"] == "template")),
            "per_jurisdiction_legacy": dict(Counter(
                d["jurisdiction"] for d in dataset
                if d["id"].endswith("_A") and d["subset"] == "legacy")),
            "per_family": dict(Counter(d["family"] for d in dataset
                                       if d["id"].endswith("_A"))),
            "legacy_k": lg_stats,
        },
        "known_defects": known_defects(),
    }
    (OUT_DIR / "eval_gold_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    # 时间戳单独存放：dataset / queries / manifest 三者逐字节可复现
    (OUT_DIR / "eval_gold_build_info.json").write_text(
        json.dumps({
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "rules_version": RULES_VERSION,
            "dataset_sha256": manifest["dataset_sha256"],
            "queries_sha256": manifest["queries_sha256"],
            "manifest_sha256": canonical_hash(manifest),
            "python": f"{__import__('sys').version.split()[0]}",
            "note": ("manifest 不含时间戳；本文件不参与复现校验，"
                     "仅记录本次构建时间与三份产物的哈希。"),
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    budget = {
        "feasible_cells": len(cells),
        "topics_covered_f1": len({p["topic_label"] for p in plan
                                  if p["family"] == "F1"}),
        "tier": dict(Counter(tier.values())),
        "plan_by_family": dict(Counter(p["family"] for p in plan)),
        "topic_label_drift": drift,
    }
    print(json.dumps(budget, ensure_ascii=False, indent=1))
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=1))
    print(f"written to {OUT_DIR}")


def parse_candidate_list(cell: str):
    """解析工作簿 L 列：`LAW_ID (article); ...`。"""
    out = []
    for chunk in str(cell or "").replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", chunk)
        if m:
            out.append((m.group(1).strip(), m.group(2).strip()))
    return out


def load_workbook_annotations():
    """读工作簿 K/L/M 三列。

    M 不是独立判断：工作簿的 M 列由 `run_eval_pipeline.auto_annotate` 生成，
    规则为 `2 if is_core and law_id in K else 1 if law_id in K else 0`，
    即 M 是 `f(K, L, is_core)` 的确定性输出。本函数只把它当作 K 的法规投影使用。
    """
    import openpyxl
    wb = openpyxl.load_workbook(DATASETS / "eval_annotation_template.xlsx",
                                data_only=True)
    ws = wb["标注总表"]
    out = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not row[0]:
            continue
        qid = str(row[0]).strip()
        labels = [int(x) for x in str(row[12] or "").split(",") if x.strip()]
        out[qid] = {
            "K": [x.strip() for x in str(row[10] or "").replace("\n", ";").split(";")
                  if x.strip()],
            "L": parse_candidate_list(row[11]),
            "M": labels,
        }
    return out


def pct(x, n):
    return f"{x / n * 100:.1f}%" if n else "n/a"


def cmd_check(corpus, topics, groups, catalog):
    ds = json.loads((OUT_DIR / "eval_dataset_v2.json").read_text(encoding="utf-8"))
    qdoc = json.loads((OUT_DIR / "eval_queries_v2.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT_DIR / "eval_gold_manifest.json").read_text(encoding="utf-8"))
    old_clause = {x["id"]: x for x in json.loads(
        (DATASETS / "eval_dataset_clause.json").read_text(encoding="utf-8"))}
    old_law = {x["id"]: x for x in json.loads(
        (DATASETS / "eval_dataset.json").read_text(encoding="utf-8"))}
    bench = load_workbook_annotations()
    build_info = json.loads(
        (OUT_DIR / "eval_gold_build_info.json").read_text(encoding="utf-8"))

    lines = []
    add = lines.append
    add("# Gold v2 自检报告")
    add("")
    add(f"- 规则版本：`{manifest['rules_version']}`")
    add(f"- 数据集：`eval_dataset_v2.json`（{len(ds)} 条变体），"
        f"`dataset_sha256={manifest['dataset_sha256'][:16]}…`")
    add(f"- 生成时间：{build_info['generated_at']}（单独记录，不参与复现校验）")
    add(f"- 语料：{manifest['corpus']['articles']} 条，保留 "
        f"{manifest['corpus']['articles_retained']} 条；"
        f"cleaned_at ≤ {manifest['corpus']['max_cleaned_at']}")
    add("")

    # 1 结构
    missing, dup, bad_rel, out_of_bounds = [], [], [], []
    for d in ds:
        keys = [(g["law_id"], g["article"]) for g in d["relevant"]]
        if len(keys) != len(set(keys)):
            dup.append(d["id"])
        for g in d["relevant"]:
            if g["relevance"] not in (1, 2):
                bad_rel.append((d["id"], g["law_id"], g["article"]))
            if (g["law_id"], g["article"]) not in corpus.articles:
                missing.append((d["id"], g["law_id"], g["article"]))
            if g["article"] not in corpus.law_articles.get(g["law_id"], set()):
                out_of_bounds.append((d["id"], g["law_id"], g["article"]))
    base = [x for x in ds if x["id"].endswith("_A")]
    empty = [x for x in base if not x["relevant"]]
    add("## 1. 结构校验")
    add("")
    add("| 检查项 | 结果 |")
    add("|---|---|")
    add(f"| gold 条款总条数（含变体重复） | {sum(len(d['relevant']) for d in ds)} |")
    add(f"| gold 条款总条数（按基础查询计） | "
        f"{sum(len(d['relevant']) for d in base)} |")
    add(f"| 唯一 (law_id, article) 对 | "
        f"{len({(g['law_id'], g['article']) for d in ds for g in d['relevant']})} |")
    add(f"| 不存在于 raw_v2 保留集的 gold | {len(missing)} |")
    add(f"| 条款号越界 | {len(out_of_bounds)} |")
    add(f"| 查询内重复 (law_id, article) | {len(dup)} |")
    add(f"| relevance 非法取值 | {len(bad_rel)} |")
    add(f"| 条款级 gold 覆盖 | {len(base) - len(empty)} / {len(base)} 条基础查询"
        f"（{len(ds) - len(empty) * 4} / {len(ds)} 条变体） |")
    add(f"| 空 gold 基础查询 | {len(empty)} / {len(base)} |")
    add(f"| 空 gold 变体 | {len(empty) * 4} / {len(ds)} |")
    add("")
    if empty:
        add("空 gold 明细（条款级不可派生，法规级仍保留）：")
        add("")
        add("| query_id | 子集 | 法域 | 主题 | 状态 |")
        add("|---|---|---|---|---|")
        for x in empty:
            add(f"| {x['query_id']} | {x['subset']} | {x['jurisdiction']} | "
                f"{x['topic']} | {x['gold_clause_status']} |")
        add("")

    # 2 覆盖
    tpl = [q for q in qdoc["queries"] if q["subset"] == "template"]
    tpl_topics = Counter(q["topic"] for q in tpl)
    lg = [q for q in qdoc["queries"] if q["subset"] == "legacy"]
    add("## 2. 覆盖情况")
    add("")
    add(f"- 主题覆盖：模板查询覆盖 {len(tpl_topics)} / {len(groups)} 个主题标签"
        f"（38 个主题 ID 中 `government_procurement` 与 `government_contracting` "
        f"同标签合并后为 {len(groups)} 组）")
    add(f"- 法域覆盖：{len({q['jurisdiction'] for q in tpl if q['family'] != 'F3'})} "
        f"个法域 + 跨法域（下表 12 行）")
    add(f"- 基础查询：{len(qdoc['queries'])}（legacy {len(lg)} + 模板 {len(tpl)}）；"
        f"变体 {len(ds)}")
    add(f"- 条款级 gold 覆盖：{len(base) - len(empty)} / {len(base)} 条基础查询；"
        f"空 gold 见 §1 明细与 `eval_gold_empty_queries.json`")
    drift = qdoc["meta"].get("topic_label_drift") or []
    if drift:
        aff = sorted({q for d in drift for q in d["affected_queries"]})
        add(f"- 主题标签漂移：{len(drift)} 个主题（"
            + "、".join(f"{d['topic_id']} {d['raw_v2_labels']}→{d['taxonomy_label']}"
                        for d in drift)
            + f"），影响 {len(aff)} 条基础查询；gold 按 topic_id 选取不受影响")
    add("")
    add("| 法域 | 模板查询 | F1 | F2 | F3 | 平均条款 gold |")
    add("|---|---|---|---|---|---|")
    for jur in JURIS + ["跨法域"]:
        qs = [q for q in tpl if q["jurisdiction"] == jur]
        if not qs:
            continue
        fam = Counter(q["family"] for q in qs)
        golds = [len(d["relevant"]) for d in base
                 if d["subset"] == "template" and d["jurisdiction"] == jur]
        add(f"| {jur} | {len(qs)} | {fam.get('F1', 0)} | {fam.get('F2', 0)} | "
            f"{fam.get('F3', 0)} | {sum(golds) / len(golds):.1f} |")
    add("")
    add("主题标签分布（模板查询，F1/F2 计一次，F3 计一次）：")
    add("")
    add("| 主题 | 查询数 |")
    add("|---|---|")
    for label, n in tpl_topics.most_common():
        add(f"| {label} | {n} |")
    add("")

    # 3 与旧 gold 重合
    old_law_sets, new_law_sets = {}, {}
    for q in lg:
        qid = q["query_id"]
        new_law_sets[qid] = set(next(d["gold_law_ids"] for d in ds
                                     if d["query_id"] == qid))
        old_law_sets[qid] = {g["law_id"] for g in old_law.get(f"{qid}_A", {})
                             .get("relevant", [])}
    jac = []
    for qid, new in new_law_sets.items():
        old = old_law_sets[qid]
        inter = len(old & new)
        union = len(old | new)
        jac.append({"qid": qid, "old": old, "new": new,
                    "jaccard": inter / union if union else 1.0,
                    "inter": inter})
    mean_jac = sum(x["jaccard"] for x in jac) / len(jac)
    exact = sum(1 for x in jac if x["old"] == x["new"])
    # 同一比较，但右侧用未修复的原始 K，用于说明 R12 修复的效果
    jac_raw = []
    for qid, new in new_law_sets.items():
        old = old_law_sets[qid]
        kraw = set(bench.get(qid, {}).get("K", []))
        jac_raw.append(len(old & kraw) / len(old | kraw) if (old | kraw) else 1.0)
    mean_jac_raw = sum(jac_raw) / len(jac_raw)

    clause_stats = {"new": 0, "old": 0, "inter": 0, "new_n": 0, "old_n": 0}
    inter_labels = Counter()
    old_only_labels = Counter()
    old_variant_sets = defaultdict(list)
    k_pos_total = k_pos_hit = 0
    k_law_total = k_law_hit = k_zero_cover = 0
    lab_by_qid = {}
    for d in ds:
        if d["subset"] != "legacy":
            continue
        new_keys = {(g["law_id"], g["article"]) for g in d["relevant"]}
        old = old_clause.get(d["id"], {"relevant": []})
        old_keys = {(g["law_id"], g["article"]) for g in old["relevant"]}
        old_variant_sets[d["query_id"]].append(frozenset(old_keys))
        clause_stats["new"] += len(new_keys)
        clause_stats["old"] += len(old_keys)
        clause_stats["inter"] += len(new_keys & old_keys)
        clause_stats["new_n"] += 1
        if old_keys:
            clause_stats["old_n"] += 1
        lab = dict(zip(bench.get(d["query_id"], {}).get("L", []),
                       bench.get(d["query_id"], {}).get("M", [])))
        for key in new_keys & old_keys:
            inter_labels[lab.get(key, "unlabeled")] += 1
        for key in old_keys - new_keys:
            old_only_labels[lab.get(key, "unlabeled")] += 1
        if d["id"].endswith("_A"):
            for key, label in lab.items():
                if label >= 1:
                    k_pos_total += 1
                    if key in new_keys:
                        k_pos_hit += 1
            krep = repair_k_set(bench.get(d["query_id"], {}).get("K", []),
                                catalog, corpus.law_names)
            new_laws = {g["law_id"] for g in d["relevant"]}
            covered = len(krep & new_laws)
            k_law_total += len(krep)
            k_law_hit += covered
            if krep and not covered:
                k_zero_cover += 1
    same_across_variants = sum(
        1 for sets in old_variant_sets.values() if len(set(sets)) == 1)

    add("## 3. 与旧 gold 的重合度")
    add("")
    add(f"- 法规级（legacy 80 条，新法规级 = 外部法规清单 K）：旧 gold 法规集与 K"
        f"（R12 修复后）平均 Jaccard = **{mean_jac:.3f}**，完全一致 {exact} / {len(jac)}")
    add(f"- 同一比较用未修复的原始 K 时 Jaccard = **{mean_jac_raw:.3f}**。旧 gold 的法规集"
        "取自「K 中每部法规在 ES 里实际取到的条款」，且新法规级 gold 本身就是 K，两侧同源。"
        "**该数字衡量的是 R12 修复把截断 ID 找回来的效果，不是规则与外部判断的一致性校验。**")
    add("- 自查记录：`0.9938` 曾被误当作外部校验证据引用，已于 `gold-v2.2` 更正。")
    add(f"- 条款级：新 gold {clause_stats['new']} 条（4 变体合计），旧 gold "
        f"{clause_stats['old']} 条，交集 {clause_stats['inter']} 条；"
        f"交集占新 gold 的 {pct(clause_stats['inter'], clause_stats['new'])}，"
        f"占旧 gold 的 {pct(clause_stats['inter'], clause_stats['old'])}")
    add(f"- 交集条目在工作簿 L/M 中的机器标注分布：{dict(inter_labels)}"
        "（`unlabeled` = 该条不在 L 列候选表内；M 的取值由 K 与 L 决定）")
    add(f"- 旧 gold 有、新 gold 无的条目机器标注分布：{dict(old_only_labels)}")
    add(f"- M≥1 的候选共 {k_pos_total} 条（等于 ES top-20 候选中 `law_id` 属于 K 的条款数），"
        f"其中被新条款 gold 覆盖 {k_pos_hit} 条（仅作对照，不作指标）；"
        "M 是 `f(K, L, is_core)` 的确定性输出，不代表任何外部判断")
    add(f"- 法规级覆盖率：新条款 gold 覆盖 K 的 **{k_law_hit} / {k_law_total}** 部法规"
        f"（{pct(k_law_hit, k_law_total)}）；**{k_zero_cover} / {len(jac)}** 条 legacy 查询里，"
        "K 的法规全部未被覆盖。该口径不依赖 L，是衡量新旧两套机械判据差异的主指标")
    add(f"- legacy 80 条查询中，4 个变体的旧 gold 集合完全相同的查询："
        f"{same_across_variants} 条")
    add("")
    add("法规级差异最大的 10 条：")
    add("")
    add("| query_id | K 列（在目录内） | 新法规级 | Jaccard |")
    add("|---|---|---|---|")
    for x in sorted(jac, key=lambda x: x["jaccard"])[:10]:
        add(f"| {x['qid']} | {len(x['old'])} 部 | {len(x['new'])} 部 | "
            f"{x['jaccard']:.3f} |")
    add("")

    # 4 与 K 法规投影的分歧（全量口径：80 条 legacy 全部计入）
    zero_qids = [qid for qid, v in bench.items()
                 if not any(lab in (1, 2) for lab in v["M"])]
    contra, inter_l, zero_new = 0, 0, []
    all_contra, all_inter, all_q = 0, 0, []
    for qid in zero_qids:
        d = next(x for x in base if x["query_id"] == qid)
        lab = dict(zip(bench[qid]["L"], bench[qid]["M"]))
        keys = {(g["law_id"], g["article"]) for g in d["relevant"]}
        hits = [k for k in keys if k in lab]
        inter_l += len(hits)
        contra += sum(1 for k in hits if lab[k] == 0)
        zero_new.append((qid, len(d["relevant"]), len(hits),
                         sum(1 for k in hits if lab[k] == 0)))
    for d in base:
        if d["subset"] != "legacy":
            continue
        lab = dict(zip(bench.get(d["query_id"], {}).get("L", []),
                       bench.get(d["query_id"], {}).get("M", [])))
        keys = {(g["law_id"], g["article"]) for g in d["relevant"]}
        hits = [k for k in keys if k in lab]
        c = sum(1 for k in hits if lab[k] == 0)
        all_inter += len(hits)
        all_contra += c
        all_q.append((d["query_id"], len(d["relevant"]), len(hits), c))
    add("## 4. 与 K 法规投影的分歧")
    add("")
    add("口径：工作簿 M 列是 `f(K, L, is_core)` 的确定性输出，M=0 等价于该候选的 "
        "`law_id` 不在 K 中。因此「分歧」= 新 gold 条款落在 ES top-20 候选内、且其法规"
        "不在 K 里。这是两套机器判据的差异，与外部判断无关。")
    add("")
    add("| 口径 | 查询数 | 新 gold ∩ ES 候选 | 其中法规不在 K | 分歧率 |")
    add("|---|---|---|---|---|")
    add(f"| 全量（80 条 legacy） | {len(all_q)} | {all_inter} | {all_contra} | "
        f"{pct(all_contra, all_inter)} |")
    add(f"| 其中：M 全零的 {len(zero_qids)} 条 | {len(zero_new)} | {inter_l} | "
        f"{contra} | {pct(contra, inter_l)} |")
    add("")
    add("| query_id | 新 gold 条数 | 落在 ES 候选内 | 其中法规不在 K |")
    add("|---|---|---|---|")
    for row in zero_new:
        add(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    add("")
    add(f"M 全零的 {len(zero_qids)} 条查询见上表（M 全零等价于该查询的 K 法规与 ES "
        "top-20 候选零交集）；全量口径下每条 legacy 查询的交集与分歧数见 "
        "`eval_gold_selfcheck.json` 的 `contradiction_full` 字段。")
    add("")

    # 5 完整性
    add("## 5. 文件完整性")
    add("")
    add("| 文件 | manifest 记录 SHA-256 | 当前 SHA-256 | 一致 |")
    add("|---|---|---|---|")
    for name, digest in manifest["inputs"].items():
        now = sha256_file(DATASETS / name)
        add(f"| {name} | {digest[:16]}… | {now[:16]}… | "
            f"{'是' if now == digest else '否'} |")
    add("")

    md_path = OUT_DIR / "eval_gold_selfcheck.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "missing": len(missing), "out_of_bounds": len(out_of_bounds),
        "dup": len(dup), "empty_base_queries": len(empty),
        "law_level_mean_jaccard_vs_old": round(mean_jac, 4),
        "law_level_mean_jaccard_vs_k_raw": round(mean_jac_raw, 4),
        "law_level_exact_vs_old": exact,
        "clause_inter": clause_stats["inter"],
        "clause_new": clause_stats["new"], "clause_old": clause_stats["old"],
        "inter_label_dist": dict(inter_labels),
        "old_only_label_dist": dict(old_only_labels),
        "k_list_positive_total": k_pos_total,
        "k_list_positive_covered": k_pos_hit,
        "k_law_coverage": f"{k_law_hit}/{k_law_total}",
        "k_law_zero_cover_queries": k_zero_cover,
        "zero_qids": len(zero_qids), "zero_intersection": inter_l,
        "zero_contradiction": contra,
        "contradiction_full_scope": {
            "queries": len(all_q), "intersection": all_inter,
            "contradiction": all_contra,
            "rate": round(all_contra / all_inter, 4) if all_inter else None,
            "per_query": [
                {"query_id": x[0], "gold": x[1], "intersection": x[2],
                 "contradiction": x[3]} for x in all_q],
        },
        "topics_covered": len(tpl_topics),
        "clause_gold_coverage": f"{len(base) - len(empty)}/{len(base)}",
        "empty_gold_variants": len(empty) * 4,
    }
    (OUT_DIR / "eval_gold_selfcheck.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"written to {md_path}")


def cmd_audit(corpus, topics, groups, catalog):
    """生成 25 条条款级抽检清单：5 条来自 11 条 M 全零查询 + 20 条分层模板。"""
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill

    ds = json.loads((OUT_DIR / "eval_dataset_v2.json").read_text(encoding="utf-8"))
    base = [d for d in ds if d["id"].endswith("_A")]
    base_by = {d["query_id"]: d for d in base}
    bench = load_workbook_annotations()

    def ev(law_id, article, topic_id):
        rec = corpus.articles.get((law_id, article))
        if not rec:
            return None, None, ""
        best = None
        for td in rec.get("topics_details") or []:
            if td.get("id") != topic_id:
                continue
            if best is None or float(td.get("score") or 0) > float(best.get("score") or 0):
                best = td
        if not best:
            return None, None, ""
        kws = best.get("matched_keywords") or []
        return (float(best.get("score") or 0), len(strong_terms(kws)),
                ", ".join(str(x) for x in kws))

    rows = []

    # 5 条 M 全零查询：优先「新 gold 非空且与 ES 候选有交集」，其余按 gold 数降序
    zero_qids = [qid for qid, v in bench.items()
                 if not any(lab in (1, 2) for lab in v["M"])]

    def zero_rank(qid):
        d = base_by.get(qid)
        if not d:
            return (3, 0, 0, qid)
        lab = dict(zip(bench[qid]["L"], bench[qid]["M"]))
        keys = {(g["law_id"], g["article"]) for g in d["relevant"]}
        inter = len(keys & set(lab))
        return (0 if d["relevant"] else 1, -inter, -len(d["relevant"]), qid)

    for qid in sorted(zero_qids, key=zero_rank)[:5]:
        rows.append((base_by.get(qid), "M 全零查询", qid))

    # 11 条 F1：每个法域各 1 条（按 JURIS 顺序取该法域 query_id 最小者）
    tpl = [d for d in base if d["subset"] == "template"]
    f1 = sorted([d for d in tpl if d["family"] == "F1"],
                key=lambda d: (JURIS.index(d["jurisdiction"]), d["query_id"]))
    seen = set()
    for d in f1:
        if d["jurisdiction"] in seen:
            continue
        seen.add(d["jurisdiction"])
        rows.append((d, "模板 F1（逐法域）", d["query_id"]))

    # 5 条 F2：按法域序号 0/2/4/7/10 各取 1（兼顾大法域与低资源法域），法域内取 gold 最多者
    f2 = sorted([d for d in tpl if d["family"] == "F2"],
                key=lambda d: (JURIS.index(d["jurisdiction"]),
                               -len(d["relevant"]), d["query_id"]))
    by_jur = {}
    for d in f2:
        by_jur.setdefault(d["jurisdiction"], []).append(d)
    for j in [JURIS[i] for i in (0, 2, 4, 7, 10)]:
        if by_jur.get(j):
            rows.append((by_jur[j][0], "模板 F2（逐法规）", by_jur[j][0]["query_id"]))

    # 4 条 F3：按 query_id 排序后隔一取一
    f3 = sorted([d for d in tpl if d["family"] == "F3"], key=lambda d: d["query_id"])
    for d in f3[::2][:4]:
        rows.append((d, "模板 F3（跨法域）", d["query_id"]))

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "抽检清单"
    header = ["序号", "抽样层", "query_id", "子集", "族", "法域", "主题",
              "查询文本（A 关键词）", "查询文本（B 自然语言）",
              "gold 条数", "law_id", "law_name", "article", "relevance",
              "主题 score", "n_strong", "matched_keywords",
              "条款原文（前 600 字）", "人工判定", "判定理由"]
    ws.append(header)
    for i, (d, layer, qid) in enumerate(rows, 1):
        if not d:
            ws.append([i, layer, qid, "", "", "", "", "", "", 0,
                       "", "", "", "", "", "", "", "（规则判定应为空 gold，抽检目标是确认「该主题在本法域确实查不到」成立）", "", ""])
            continue
        gold = d["relevant"]
        g = gold[0] if gold else None
        score = n_strong = matched = snippet = ""
        if g:
            snippet = (corpus.articles.get((g["law_id"], g["article"])) or {})
            snippet = (snippet.get("text_zh") or snippet.get("text_en") or "")[:600]
            score, n_strong, matched = ev(g["law_id"], g["article"], g["topic_id"])
        ws.append([
            i, layer, qid, d["subset"], d["family"], d["jurisdiction"], d["topic"],
            d.get("query", ""), d.get("B_nl_zh", ""), len(gold),
            g["law_id"] if g else "", g["law_name"] if g else "",
            g["article"] if g else "", g["relevance"] if g else "",
            score, n_strong, matched, snippet, "", "",
        ])

    widths = {"A": 5, "B": 16, "C": 14, "D": 8, "E": 8, "F": 10, "G": 14,
              "H": 30, "I": 34, "J": 8, "K": 24, "L": 30, "M": 8,
              "N": 9, "O": 9, "P": 8, "Q": 30, "R": 60, "S": 10, "T": 24}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDEBF7")
    ws.freeze_panes = "A2"

    ws2 = wb.create_sheet("判定标准")
    ws2.append(["项", "内容"])
    for k, v in [
        ("目的", "验证条款级 gold 的首位条款是否实质规定查询主题。"),
        ("判定粒度", "每条只判 1 个 clause（gold 排序第 1 位），结论只代表首位，不代表整组 gold。"),
        ("通过", "条款正文直接规定该主题的权利义务/程序/标准，查询方可用它回答问题。"),
        ("不通过", "纯定义条款、纯附件/修订附表、纯「构成犯罪的依法追究刑事责任」引用、"
                  "仅顺带出现主题词而规范对象是别的主题。"),
        ("存疑", "需要领域知识才能判断，或条款跨多个主题、无法归入单一主题。"),
        ("通过门槛", "25 条中 ≥20 条通过；F1/F2/F3 每层通过率 ≥75%；"
                    "不通过项中「纯定义/纯附件/纯引用」≥3 条则触发 R1 复核；"
                    "「与主题无关」≥5 条则条款级 gold 需重订阈值。"),
        ("记录方式", "在「人工判定」列填 通过/不通过/存疑，在「判定理由」列写一句话依据。"),
        ("工时", "25 条 × 5–7 分钟，约 2–3 小时。"),
        ("边界", "抽检不覆盖 Recall：gold 之外的同主题条款是否遗漏，无法由本表回答。"),
    ]:
        ws2.append([k, v])
    ws2.column_dimensions["A"].width = 14
    ws2.column_dimensions["B"].width = 100
    for row in ws2.iter_rows(min_row=1, max_row=ws2.max_row):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for cell in ws2[1]:
        cell.font = Font(bold=True)

    ws3 = wb.create_sheet("抽样说明")
    ws3.append(["项", "内容"])
    for k, v in [
        ("全零查询 5 条", "M 列全为 0 的 11 条查询中，优先取「新 gold 非空且与 ES 候选有交集」，"
                        "再按 gold 数降序；11 条里只有 4 条新 gold 非空，第 5 条为应判空查询。"),
        ("模板 F1 11 条", "11 个法域各取 1 条，取该法域内 query_id 最小者；保证法域全覆盖。"),
        ("模板 F2 5 条", "按法域序号 0/2/4/7/10 各取 1 条（中国大陆、澳大利亚、日本、韩国、马来西亚），"
                       "法域内取 gold 条数最多者，兼顾大法域与低资源法域。"),
        ("模板 F3 4 条", "8 条跨法域查询按 query_id 排序后隔一取一。"),
        ("确定性", "抽样只依赖 query_id、法域顺序、gold 数，不含随机数；重跑结果一致。"),
        ("数据来源", "只读 raw_v2 与工作簿；不读取任何检索结果或 rerank 分数。"),
        ("与法规级抽检的区别", "本清单抽的是条款（25 条）。法规级另有一套：从 K 的 128 部法规中抽 30 部"
                            "对原文，用于确认外部法规清单本身是否指向正确的法规，两者不要混用。"),
    ]:
        ws3.append([k, v])
    ws3.column_dimensions["A"].width = 16
    ws3.column_dimensions["B"].width = 100
    for row in ws3.iter_rows(min_row=1, max_row=ws3.max_row):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for cell in ws3[1]:
        cell.font = Font(bold=True)

    path = OUT_DIR / "eval_gold_audit_25.xlsx"
    wb.save(path)
    print(f"audit rows: {len(rows)}")
    print(f"written to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["stats", "build", "check", "audit", "all"])
    args = ap.parse_args()

    raw_tax, topics, groups = load_taxonomy()
    _, catalog = load_catalog()
    corpus = Corpus(topics)
    cells = build_cells(corpus, topics, groups)

    if args.stage in ("stats", "all"):
        cmd_stats(corpus, topics, groups, cells)
    if args.stage in ("build", "all"):
        cmd_build(corpus, topics, groups, catalog)
    if args.stage in ("check", "all"):
        cmd_check(corpus, topics, groups, catalog)
    if args.stage in ("audit", "all"):
        cmd_audit(corpus, topics, groups, catalog)


if __name__ == "__main__":
    main()
