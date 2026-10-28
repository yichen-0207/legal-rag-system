# -*- coding: utf-8 -*-
"""入库前校验与最小修复：data/<新增法域> → data/raw/<法域>。

校验口径对齐 phase1_clean_data.py：一文一 law_id、doc_id 唯一、条款号唯一、
正文可召回、法域字段规范、UTF-8 可解析。默认只读；--apply 才落盘。

修复范围仅限身份字段与不可召回条目：
- australia: 依据 article_number 重建截断的 doc_id（原 205 组重复）；
- china: 拆分 3 组同名 law_id、统一法域、剔除新闻摘要/附件正文；
- japan: SMP_JP_001 新旧合并，其余用上游新版覆盖；
- korea: 剔除 deleted/omitted 空正文；
- hk/macau: 既有 raw 数据法域改名为 中国香港/中国澳门。
"""

import argparse
import collections
import datetime
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RAW = DATA / "raw"
REPORT = DATA / "pre_ingest_report.json"

NEW_DIRS = ("australia", "china", "japan", "korea")
CANONICAL_JURISDICTION = {
    "australia": "澳大利亚",
    "china": "中国大陆",
    "japan": "日本",
    "korea": "韩国",
}
LEGACY_RENAME = {"hk": ("香港", "中国香港"), "macau": ("澳门", "中国澳门")}

REQUIRED_FIELDS = (
    "doc_id", "law_id", "jurisdiction", "law_name", "article_number",
    "chunk_type", "text_zh", "text_pt", "text_en",
)
EXCLUDED_ROLES = {"SUMMARY_OR_NEWS", "BODY_IN_ATTACHMENT"}
JP_NEW_FIELDS = ("promulgating_authority", "related_laws", "status")


def item_text(item: Dict) -> str:
    return item.get("text_zh") or item.get("text_pt") or item.get("text_en") or ""


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def validate_items(items: List[Dict], source: str) -> List[str]:
    """单项文件的结构校验，返回问题描述列表。"""
    errors = []
    if not isinstance(items, list) or not items:
        return [f"{source}: 顶层不是非空数组"]
    for field in REQUIRED_FIELDS:
        missing = sum(1 for it in items if field not in it)
        if missing:
            errors.append(f"{source}: 缺失字段 {field} × {missing}")
    law_ids = {it.get("law_id") for it in items}
    if len(law_ids) != 1:
        errors.append(f"{source}: 文件内 law_id 不唯一 {sorted(law_ids)}")
    doc_ids = [it.get("doc_id") for it in items]
    dup_docs = [k for k, v in collections.Counter(doc_ids).items() if v > 1]
    if dup_docs:
        errors.append(f"{source}: doc_id 重复 {len(dup_docs)} 组")
    arts = [str(it.get("article_number") or "") for it in items]
    dup_arts = [k for k, v in collections.Counter(arts).items() if v > 1]
    if dup_arts:
        errors.append(f"{source}: 条款号重复 {dup_arts[:5]}")
    return errors


def normalize_au_article(article: str) -> str:
    """把数字部分补足 3 位并保留字母后缀：32BA → 032BA，End0 → End0。"""
    match = re.fullmatch(r"(\d+)([A-Za-z]*)", article)
    if not match:
        return article
    return match.group(1).zfill(3) + match.group(2)


def repair_australia(files: List[Tuple[str, List[Dict]]]) -> Tuple[List[Tuple[str, List[Dict]]], Dict]:
    out, dropped, renamed = [], [], 0
    for name, items in files:
        kept = []
        for it in items:
            if not item_text(it):
                dropped.append({"file": name, "doc_id": it.get("doc_id"),
                                "article_number": it.get("article_number")})
                continue
            article = str(it.get("article_number") or "").strip()
            normalized = normalize_au_article(article)
            new_doc_id = f"{it['law_id']}_ART_{normalized}"
            if new_doc_id != it.get("doc_id"):
                renamed += 1
            it["article_number_normalized"] = normalized
            it["doc_id"] = new_doc_id
            kept.append(it)
        if kept:
            out.append((name, kept))
    stats = {"doc_id_rebuilt": renamed, "empty_text_removed": dropped}
    return out, stats


def resolve_law_id_collisions(files: List[Tuple[str, List[Dict]]]) -> Tuple[List[Tuple[str, List[Dict]]], Dict]:
    """同一 law_id 出现在多个文件时，保留文件名靠前者，其余改用 CN_DOC_CN_<编号>。"""
    groups: Dict[str, List[str]] = collections.defaultdict(list)
    by_name = dict(files)
    for name, items in files:
        groups[items[0]["law_id"]].append(name)

    renames = []
    used = set(groups)
    for law_id, names in sorted(groups.items()):
        if len(names) <= 1:
            continue
        for name in sorted(names)[1:]:
            match = re.search(r"CN_(\d+)", name)
            if not match:
                raise ValueError(f"{name}: 无法从文件名生成新 law_id")
            new_law_id = f"CN_DOC_CN_{match.group(1)}"
            if new_law_id in used:
                raise ValueError(f"{name}: 新 law_id 冲突 {new_law_id}")
            used.add(new_law_id)
            for it in by_name[name]:
                old_prefix = f"{law_id}_"
                if str(it.get("doc_id") or "").startswith(old_prefix):
                    it["doc_id"] = new_law_id + it["doc_id"][len(law_id):]
                it["law_id"] = new_law_id
            renames.append({"file": name, "old_law_id": law_id, "new_law_id": new_law_id})
    return files, {"law_id_renamed": renames}


def repair_china(files: List[Tuple[str, List[Dict]]]) -> Tuple[List[Tuple[str, List[Dict]]], Dict]:
    kept_files, excluded, jurisdiction_fixed = [], [], 0
    for name, items in files:
        kept_items = []
        for it in items:
            if it.get("content_role") in EXCLUDED_ROLES:
                excluded.append({"file": name, "doc_id": it.get("doc_id"),
                                 "content_role": it.get("content_role"),
                                 "law_name": it.get("law_name")})
                continue
            if it.get("jurisdiction") != CANONICAL_JURISDICTION["china"]:
                jurisdiction_fixed += 1
            it["jurisdiction"] = CANONICAL_JURISDICTION["china"]
            kept_items.append(it)
        if kept_items:
            kept_files.append((name, kept_items))

    kept_files, collision_stats = resolve_law_id_collisions(kept_files)
    stats = {
        "excluded_non_legal_files": excluded,
        "jurisdiction_normalized": jurisdiction_fixed,
        **collision_stats,
    }
    return kept_files, stats


def merge_japan_001(new_items: List[Dict], old_items: List[Dict]) -> List[Dict]:
    """以旧版 103 条为主体，用新版同条款替换并补齐新字段。"""
    new_by_art = {str(it.get("article_number")): it for it in new_items}
    merged, used = [], set()
    for old in old_items:
        art = str(old.get("article_number"))
        if art in new_by_art and art not in used:
            merged.append(new_by_art[art])
            used.add(art)
            continue
        for field in JP_NEW_FIELDS:
            old.setdefault(field, None)
        merged.append(old)
    for art, it in new_by_art.items():
        if art not in used:
            merged.append(it)
    for index, it in enumerate(merged):
        it["chunk_index"] = index
    return merged


def repair_japan(files: List[Tuple[str, List[Dict]]]) -> Tuple[List[Tuple[str, List[Dict]]], Dict]:
    out, dropped, merged_log = [], [], []
    for name, items in files:
        kept = [it for it in items if item_text(it)]
        for it in items:
            if not item_text(it):
                dropped.append({"file": name, "doc_id": it.get("doc_id"),
                                "article_number": it.get("article_number")})
        old_path = RAW / "japan" / name
        old_items = load_json(old_path) if old_path.exists() else []
        old_arts = {str(it.get("article_number")) for it in old_items if item_text(it)}
        kept_arts = {str(it.get("article_number")) for it in kept}

        if name == "SMP_JP_001_cleaned.json" and old_items:
            merged = merge_japan_001(kept, old_items)
            merged_log.append({"file": name, "new_articles": len(kept), "old_articles": len(old_items),
                               "merged_articles": len(merged), "lost_articles": sorted(old_arts - kept_arts)})
            out.append((name, merged))
            continue

        lost = sorted(old_arts - kept_arts)
        if lost:
            raise AssertionError(f"{name}: 新版缺失旧版条款 {lost[:10]}，拒绝覆盖")
        out.append((name, kept))
    return out, {"empty_text_removed": dropped, "japan_001_merge": merged_log}


def repair_korea(files: List[Tuple[str, List[Dict]]]) -> Tuple[List[Tuple[str, List[Dict]]], Dict]:
    out, dropped = [], []
    for name, items in files:
        kept = []
        for it in items:
            if not item_text(it):
                dropped.append({"file": name, "doc_id": it.get("doc_id"),
                                "article_number": it.get("article_number"),
                                "chunk_type": it.get("chunk_type")})
                continue
            kept.append(it)
        if kept:
            out.append((name, kept))
    return out, {"empty_text_removed": dropped}


def collect_source(src_dir: Path) -> List[Tuple[str, List[Dict]]]:
    return [(path.name, load_json(path)) for path in sorted(src_dir.glob("*.json"))]


def verify_before_write(all_files: Dict[str, List[Tuple[str, List[Dict]]]]) -> Dict:
    """落盘前的全局唯一性校验：law_id 按文件、doc_id 按条目。"""
    law_ids = collections.Counter()
    doc_ids = collections.Counter()
    for jurisdiction, files in all_files.items():
        for name, items in files:
            errors = validate_items(items, f"{jurisdiction}/{name}")
            if errors:
                raise AssertionError("；".join(errors))
            law_ids[items[0]["law_id"]] += 1
            for it in items:
                doc_ids[it["doc_id"]] += 1
    dup_laws = [k for k, v in law_ids.items() if v > 1]
    dup_docs = [k for k, v in doc_ids.items() if v > 1]
    if dup_laws or dup_docs:
        raise AssertionError(f"全局唯一性失败: law_id={dup_laws[:5]} doc_id={dup_docs[:5]}")
    return {
        "distinct_law_ids": len(law_ids),
        "distinct_doc_ids": len(doc_ids),
        "total_articles": sum(len(items) for _, items in
                              (f for fs in all_files.values() for f in fs)),
        "per_jurisdiction": {
            j: {
                "files": len(fs),
                "articles": sum(len(items) for _, items in fs),
                "distinct_law_ids": len({items[0]["law_id"] for _, items in fs}),
            } for j, fs in all_files.items()
        },
        "duplicate_law_ids": dup_laws,
        "duplicate_doc_ids": dup_docs,
    }


def rename_existing_jurisdiction() -> Dict:
    """把既有 raw/hk、raw/macau 的法域字段改为中国香港/中国澳门（文本替换保留原格式）。"""
    result = {}
    for dir_name, (old, new) in LEGACY_RENAME.items():
        dir_path = RAW / dir_name
        if not dir_path.is_dir():
            continue
        files, items = 0, 0
        for path in sorted(dir_path.glob("*.json")):
            text = path.read_text(encoding="utf-8")
            count = text.count(f'"jurisdiction": "{old}"')
            if not count:
                continue
            path.write_text(text.replace(f'"jurisdiction": "{old}"', f'"jurisdiction": "{new}"'),
                            encoding="utf-8")
            files += 1
            items += count
        result[dir_name] = {"old": old, "new": new, "files": files, "items": items}
    return result


def check_raw_conflicts(all_files: Dict[str, List[Tuple[str, List[Dict]]]]) -> Dict:
    """新增数据与既有 raw 的 ID 冲突；japan 为版本升级，单独标注。"""
    old_laws, old_docs = set(), set()
    for jur_dir in RAW.iterdir():
        if not jur_dir.is_dir() or jur_dir.name.startswith("."):
            continue
        for path in jur_dir.glob("*.json"):
            try:
                data = load_json(path)
            except Exception:
                continue
            if isinstance(data, list):
                for it in data:
                    old_laws.add(it.get("law_id"))
                    old_docs.add(it.get("doc_id"))
    result = {}
    for jurisdiction, files in all_files.items():
        laws = {it["law_id"] for _, items in files for it in items}
        docs = {it["doc_id"] for _, items in files for it in items}
        result[jurisdiction] = {
            "law_id_conflicts": sorted(laws & old_laws),
            "doc_id_conflict_count": len(docs & old_docs),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="新增法域语料入库前校验与最小修复")
    parser.add_argument("--apply", action="store_true", help="写入 data/raw 并输出报告；默认只读")
    args = parser.parse_args()

    missing = [d for d in NEW_DIRS if not (DATA / d).is_dir()]
    if missing:
        sys.exit(f"源目录不存在: {missing}")

    all_files, stats = {}, {}
    for jurisdiction in NEW_DIRS:
        files = collect_source(DATA / jurisdiction)
        if jurisdiction == "australia":
            all_files[jurisdiction], stats[jurisdiction] = repair_australia(files)
        elif jurisdiction == "china":
            all_files[jurisdiction], stats[jurisdiction] = repair_china(files)
        elif jurisdiction == "japan":
            all_files[jurisdiction], stats[jurisdiction] = repair_japan(files)
        else:
            all_files[jurisdiction], stats[jurisdiction] = repair_korea(files)

    verification = verify_before_write(all_files)
    conflicts = check_raw_conflicts(all_files)

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "mode": "apply" if args.apply else "check",
        "transforms": stats,
        "verification": verification,
        "raw_conflicts": conflicts,
        "warnings": [
            "china 保留 2 条 REVIEW_REQUIRED（CN_120、CN_150），需人工复核",
            "japan 42 个文件用上游新版覆盖；SMP_JP_001 为旧版 103 条与新版合并",
            "hk/macau 改名仅在 raw 执行；raw_v2 与 ES 需重建后生效",
        ],
    }

    print("=" * 66)
    for jurisdiction, info in verification["per_jurisdiction"].items():
        print(f"[{jurisdiction}] 文件 {info['files']} | 条款 {info['articles']} | 唯一 law_id {info['distinct_law_ids']}")
    print(f"全局: 条款 {verification['total_articles']} | law_id {verification['distinct_law_ids']} | "
          f"doc_id {verification['distinct_doc_ids']} | 重复 0")
    for jurisdiction, s in stats.items():
        print(f"[{jurisdiction} 修复] {json.dumps(s, ensure_ascii=False)[:300]}")

    if not args.apply:
        print("\n只读模式，未写入。确认后执行: python data_tools/pre_ingest_check.py --apply")
        return

    for jurisdiction, files in all_files.items():
        for name, items in files:
            dump_json(RAW / jurisdiction / name, items)
    report["raw_legacy_rename"] = rename_existing_jurisdiction()
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n已写入 data/raw，报告:", REPORT)


if __name__ == "__main__":
    main()
