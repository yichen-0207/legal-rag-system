# -*- coding: utf-8 -*-
"""
阶段 1 离线数据清洗（高配方案 1a / 1b / 1c / 1d）

读：legal_rag/data/raw             —— 只读，本脚本绝不修改原始语料
写：legal_rag/data/raw_v2          —— 派生语料，每次运行全量重建（可复现、可回滚）
写：legal_rag/data/phase1_report.json —— 审计报告（放在 data/ 下而非 raw_v2 内，
                                     避免被 ingest 的目录扫描当成法规语料）

变换清单
  1a 泰国 law_id 拆分：TH_ACT_1979 拆出 Export And Import Of Goods Act, 1979
                       TH_MIN_1999 拆出 No.22/23/24/25/26/27 六个部级规章
                       同步重写 doc_id 前缀（原始数据中仅 law_id / doc_id 两个字段含该串）
  1b 空文本条目盘点：只记录清单，不改数据（本机无原始语料，sub_chunks 亦无内容，无法补洗）
  1c USA / 新加坡单字符行合并：还原被换行拆散的括号标记（"(\\na\\n)" → "(a)"）
  1d 日期元数据回填：按文件取各日期字段的首个非空值，填齐同文件所有条目
  1f 清除与合并（用户确认范围）：
       1f-1 剔除三个文本字段全空的条目（不可召回，仅污染索引与统计）
       1f-2 同文件内 doc_id 重复时保留首条
       1f-3 新加坡 4 部法的片段 ID 文件并入规范 law_id（片段 ID 由正文首句截断生成，
            属清洗缺陷产物，同一部法因此被拆成两个 law_id）；合并后同条款号
            重复（新旧两代抓取并存）保留正文更长者

用法：
  python phase1_clean_data.py            # 全量重建 raw_v2 并输出报告
  python phase1_clean_data.py --dry-run  # 只统计与校验，不写任何文件
"""

import argparse
import collections
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent          # legal_rag/
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "raw_v2"
REPORT = BASE / "data" / "phase1_report.json"

SKIP_DIRS = {".uploads", "__pycache__", ".git"}

# ---------------------------------------------------------------- 1a 泰国 ID 拆分
# 键 = 文件名；值 = (期望的旧 law_id, 新 law_id, 法规名，用于断言)
THAI_ID_SPLIT = {
    "SMP_TH_003_cleaned.json": ("TH_ACT_1979", "TH_ACT_1979_EXIM",
                                "Export And Import Of Goods Act, 1979"),
    "SMP_TH_014_cleaned.json": ("TH_MIN_1999", "TH_MIN_1999_022",
                                "Ministerial Regulations No. 22 (B.E. 2542)"),
    "SMP_TH_015_cleaned.json": ("TH_MIN_1999", "TH_MIN_1999_024",
                                "Ministerial Regulations No. 24 (B.E. 2542)"),
    "SMP_TH_016_cleaned.json": ("TH_MIN_1999", "TH_MIN_1999_025",
                                "Ministerial Regulations No. 25 (B.E. 2542)"),
    "SMP_TH_017_cleaned.json": ("TH_MIN_1999", "TH_MIN_1999_026",
                                "Ministerial Regulations No. 26 (B.E. 2542)"),
    "SMP_TH_018_cleaned.json": ("TH_MIN_1999", "TH_MIN_1999_027",
                                "Ministerial Regulations No. 27 (B.E. 2542)"),
    "SMP_TH_020_cleaned.json": ("TH_MIN_1999", "TH_MIN_1999_023",
                                "Ministerial Regulations No. 23 (B.E. 2542)"),
}

# ---------------------------------------------------------------- 1c 换行碎字合并
LINE_MERGE_JURISDICTIONS = {"usa", "singapore"}
TEXT_FIELDS = ("text_zh", "text_pt", "text_en")
SHORT_LINE_MAX = 1          # 非空且长度 <= 1 的行视为被拆散的标记片段

# ---------------------------------------------------------------- 1d 日期回填
DATE_FIELDS = ("passing_date", "effective_date", "publication_date", "last_amended_date")

# ---------------------------------------------------------------- 1f 清除与合并
# 范围来自用户确认：① 空文本条目 ② 重复 doc_id ③ 新加坡重复 ID 副本 18 条。
# 原则：只改写/剔除「重复副本」，不删除任何仅存一份的正文
#（旧代块状条目是否剔除见报告 inventory_pending_decision 的 P1-SG-BLOCK）。
EXCLUDE_EMPTY_TEXT = True
DEDUP_DOC_ID = True

# 片段文件名 → (片段 law_id, 规范 law_id, 合并后的落盘文件)
# 片段 ID 形如 SG_ACT_This_Act_is_the_Pers_2012，由正文首句截断生成，是清洗脚本缺陷产物；
# 规范 ID 为立法编号式命名（与同法的旧代抓取文件保持一致）。
SG_ID_MERGE = {
    "SG_36164195_Personal_Data_Protection_Act_2012_cleaned.json":
        ("SG_ACT_This_Act_is_the_Pers_2012", "SG_ACT_22_2016",
         "SG_30981088_Personal_Data_Protection_Act_2012_cleaned.json"),
    "SG_38989252_Financial_Services_and_Markets_Act_2022_cleaned.json":
        ("SG_ACT_This_Act_is_the_Fina_2022", "SG_ACT_18_2022",
         "SG_19582447_Financial_Services_And_Markets_Act_2022_cleaned.json"),
    "SG_9954904_Payment_Services_Act_2019_cleaned.json":
        ("SG_ACT_This_Act_is_the_Paym_2019", "SG_ACT_1_2021",
         "SG_17345705_Payment_Services_Act_2019_cleaned.json"),
    "SG_93567056_Spam_Control_Act_2007_cleaned.json":
        ("SG_ACT_This_Act_is_the_Spam_2007", "SG_ACT_40_2020",
         "SG_57922111_Spam_Control_Act_2007_cleaned.json"),
}


def merge_short_lines(text: str) -> str:
    """把连续的单字符行拼回同一行（"(\\na\\n)\\nGeneral." → "(a)\\nGeneral."）。

    空行保留为段落分隔，普通行原样保留（仅去尾部空白）。
    幂等：已合并的文本再次处理结果不变。
    """
    if not text:
        return text
    out = []
    buf = ""
    for line in text.split("\n"):
        s = line.strip()
        if s == "":
            if buf:
                out.append(buf)
                buf = ""
            out.append("")
        elif len(s) <= SHORT_LINE_MAX:
            buf += s
        else:
            if buf:
                out.append(buf)
                buf = ""
            out.append(line.rstrip())
    if buf:
        out.append(buf)
    return "\n".join(out)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj):
    """按原始语料格式写回：UTF-8、indent=2、CRLF。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def apply_thai_split(items, filename):
    """1a：重写单文件内的 law_id 与 doc_id 前缀。"""
    old_lid, new_lid, expect_name = THAI_ID_SPLIT[filename]
    first = items[0] if items else {}
    actual_lid = first.get("law_id", "")
    actual_name = (first.get("law_name") or "").strip()
    if actual_lid != old_lid:
        # 已处理过（幂等）或数据与预期不符 → 断言失败即中止，避免误改
        if actual_lid == new_lid:
            return 0
        raise AssertionError(f"{filename}: 期望 law_id={old_lid}，实际={actual_lid}")
    if expect_name and expect_name not in actual_name:
        raise AssertionError(f"{filename}: 期望法规名含 {expect_name!r}，实际={actual_name!r}")

    changed = 0
    for it in items:
        it["law_id"] = new_lid
        doc_id = it.get("doc_id") or ""
        if doc_id.startswith(old_lid):
            it["doc_id"] = new_lid + doc_id[len(old_lid):]
        changed += 1
    return changed


def apply_line_merge(items):
    """1c：对 USA / 新加坡语料的三个文本字段做单字符行合并。"""
    stat = collections.Counter()
    for it in items:
        for field in TEXT_FIELDS:
            txt = it.get(field)
            if not txt:
                continue
            merged = merge_short_lines(txt)
            if merged != txt:
                stat[field + "_chars"] += len(txt) - len(merged)
                stat[field + "_items"] += 1
                it[field] = merged
    return stat


def apply_date_backfill(items):
    """1d：同文件内取各日期字段的首个非空值，回填所有空值条目。"""
    filled = 0
    for field in DATE_FIELDS:
        value = ""
        for it in items:
            if (it.get(field) or "").strip():
                value = it[field]
                break
        if not value:
            continue
        for it in items:
            if not (it.get(field) or "").strip():
                it[field] = value
                filled += 1
    return filled


def item_text(it) -> str:
    """条目正文（三个文本字段取首个非空）。"""
    return it.get("text_zh") or it.get("text_pt") or it.get("text_en") or ""


def merge_sg_items(items, fragments, old_id, new_id) -> int:
    """1f-3：把片段条目改写 law_id / doc_id 前缀后并入目标条目列表。"""
    moved = 0
    for it in fragments:
        it["law_id"] = new_id
        did = it.get("doc_id") or ""
        if did.startswith(old_id):
            it["doc_id"] = new_id + did[len(old_id):]
        items.append(it)
        moved += 1
    return moved


def dedup_same_article(items):
    """1f-3：同一 law_id + 条款号出现多条时保留正文更长者，返回被剔除条目清单。

    合并后新旧两代抓取并存：旧代块状条目把多条规定压成一条且条款号为解析异常值
    （如 2020/25/31），与逐条版本在同一部法内共存时只保留正文更完整的一条。
    """
    best, dropped = {}, []
    for it in items:
        art = str(it.get("article_number") or "").strip()
        key = (it.get("law_id"), art if art else it.get("doc_id"))
        cur = best.get(key)
        if cur is None:
            best[key] = it
            continue
        a, b = item_text(it), item_text(cur)
        keep, drop = (it, cur) if len(a) > len(b) else (cur, it)
        best[key] = keep
        dropped.append({"law_id": keep.get("law_id"), "article_number": art,
                        "kept_doc_id": keep.get("doc_id"), "kept_chars": len(item_text(keep)),
                        "dropped_doc_id": drop.get("doc_id"), "dropped_chars": len(item_text(drop))})
    if dropped:
        items[:] = list(best.values())
    return dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只统计校验，不写文件")
    args = ap.parse_args()

    if not RAW.is_dir():
        sys.exit(f"原始语料目录不存在: {RAW}")

    if OUT.exists() and not args.dry_run:
        shutil.rmtree(OUT)          # 全量重建，保证可复现
    if not args.dry_run:
        OUT.mkdir(parents=True, exist_ok=True)

    report = {
        "phase": 1,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(RAW),
        "output_dir": str(OUT),
        "dry_run": args.dry_run,
        "transforms": {},
        "inventory_pending_decision": [],
        "verification": {},
    }
    split_files, merge_stat = [], collections.Counter()
    date_files, date_filled = 0, 0
    raw_stats = collections.Counter()      # (jur, "files"/"articles")
    out_stats = collections.Counter()
    raw_laws, out_laws = set(), set()
    empty_text_items = []
    sg_pending, sg_target_map = {}, {}     # 1f-3：片段文件内容与目标文件映射
    removed_by_jur = collections.Counter()      # (jur, 规则) -> 剔除条数，供校验用
    removed_files_by_jur = collections.Counter()
    sg_merge_log, sg_dedup_log, dup_doc_log = [], [], []

    # ---- 1f-3 预读：片段 ID 文件在排序上位于目标文件之后，先读入内存等待并入
    for fname, (old_id, new_id, target) in SG_ID_MERGE.items():
        fitems = load_json(RAW / "singapore" / fname)
        assert fitems and fitems[0].get("law_id") == old_id, \
            f"{fname}: 期望片段 law_id={old_id}，实际={fitems[0].get('law_id')}"
        sg_pending.setdefault(target, []).append((fname, fitems))
        sg_target_map[target] = (old_id, new_id)
        removed_files_by_jur["singapore"] += 1

    for jur_dir in sorted(p for p in RAW.iterdir() if p.is_dir() and p.name not in SKIP_DIRS):
        jur = jur_dir.name
        for src in sorted(jur_dir.glob("*.json")):
            items = load_json(src)
            raw_stats[(jur, "files")] += 1
            raw_stats[(jur, "articles")] += len(items)
            for it in items:                       # 变换前的原始 law_id，用于验证拆分效果
                if it.get("law_id"):
                    raw_laws.add(it["law_id"])

            # ---- 1f-3：片段 ID 文件的内容已并入目标文件，主循环中不再单独落盘
            if jur == "singapore" and src.name in SG_ID_MERGE:
                continue

            # ---- 1f-3：把片段条目并入规范 law_id（须在 1c/1d 之前，保证一并做换行合并）
            if jur == "singapore" and src.name in sg_target_map:
                old_id, new_id = sg_target_map[src.name]
                moved = 0
                for fname, fitems in sg_pending.pop(src.name, []):
                    moved += merge_sg_items(items, fitems, old_id, new_id)
                sg_merge_log.append({"target_file": src.name, "old_law_id": old_id,
                                     "new_law_id": new_id, "items_moved": moved})

            # ---- 1a
            if jur == "thailand" and src.name in THAI_ID_SPLIT:
                n = apply_thai_split(items, src.name)
                if n:
                    split_files.append({
                        "file": src.name,
                        "old_law_id": THAI_ID_SPLIT[src.name][0],
                        "new_law_id": THAI_ID_SPLIT[src.name][1],
                        "articles": n,
                    })
            # ---- 1b（只盘点，不改数据）
            for it in items:
                if not (it.get("text_zh") or it.get("text_pt") or it.get("text_en")):
                    empty_text_items.append({
                        "jurisdiction": jur,
                        "file": src.name,
                        "doc_id": it.get("doc_id"),
                        "article_number": it.get("article_number"),
                        "sub_chunks": len(it.get("sub_chunks") or []),
                        "source_file": it.get("source_file"),
                        "source_url": it.get("source_url"),
                    })
            # ---- 1c
            if jur in LINE_MERGE_JURISDICTIONS:
                merge_stat.update(apply_line_merge(items))
            # ---- 1f-1：剔除三字段全空的条目（不可召回，仅占索引与统计口径）
            if EXCLUDE_EMPTY_TEXT:
                kept = [it for it in items if item_text(it)]
                removed_by_jur[(jur, "empty_text")] += len(items) - len(kept)
                items = kept
            # ---- 1f-3：合并后同条款号重复（新旧两代抓取并存）保留正文更长者。
            # 必须先于 1f-2：片段条目改写 doc_id 后会与旧代块状条目同名（如 ..._ART_002），
            # 若先按 doc_id 去重，会「保留首条」把正文更完整的片段版本误删。
            if jur == "singapore" and src.name in sg_target_map:
                dropped = dedup_same_article(items)
                for d in dropped:
                    d["file"] = src.name
                sg_dedup_log.extend(dropped)
                removed_by_jur[(jur, "sg_same_article")] += len(dropped)
            # ---- 1f-2：同文件内 doc_id 重复时保留首条
            if DEDUP_DOC_ID:
                seen_doc, kept = set(), []
                for it in items:
                    did = it.get("doc_id")
                    if did in seen_doc:
                        dup_doc_log.append({"jurisdiction": jur, "file": src.name, "doc_id": did})
                        removed_by_jur[(jur, "dup_doc_id")] += 1
                        continue
                    seen_doc.add(did)
                    kept.append(it)
                items = kept
            # ---- 1d：日期回填放在 1f 之后 —— 只在保留条目之间取首个非空值，
            # 避免已被去重剔除的旧代块状条目日期（如修订版 2021-12-31）污染保留条目
            n_filled = apply_date_backfill(items)
            if n_filled:
                date_files += 1
                date_filled += n_filled
            # ---- 输出（先记录内存侧统计，写入后回读再校验一次）
            for it in items:
                if it.get("law_id"):
                    out_laws.add(it["law_id"])
            out_stats[(jur, "files")] += 1
            out_stats[(jur, "articles")] += len(items)
            if not args.dry_run:
                dump_json(OUT / jur / src.name, items)

    # ---------------------------------------------------------- 校验
    disk_stats = collections.Counter()
    if not args.dry_run:
        for jur_dir in sorted(p for p in OUT.iterdir() if p.is_dir() and p.name not in SKIP_DIRS):
            for f in jur_dir.glob("*.json"):
                disk_stats[(jur_dir.name, "files")] += 1
                disk_stats[(jur_dir.name, "articles")] += len(load_json(f))

    assert not sg_pending, f"片段文件未并入目标文件: {list(sg_pending)}"
    jur_list = sorted({k[0] for k in raw_stats})
    per_jur = {}
    ok = True
    disk_ok = True
    for jur in jur_list:
        a_raw, a_out = raw_stats[(jur, "articles")], out_stats[(jur, "articles")]
        f_raw, f_out = raw_stats[(jur, "files")], out_stats[(jur, "files")]
        # 1f 的剔除与合并会让条数/文件数下降，校验口径改为「实际差值 == 逐条记录的剔除数」
        exp_removed = sum(v for (j, _r), v in removed_by_jur.items() if j == jur)
        exp_files_removed = removed_files_by_jur.get(jur, 0)
        same = (a_raw - a_out == exp_removed) and (f_raw - f_out == exp_files_removed)
        ok = ok and same
        entry = {"files_raw": f_raw, "files_v2": f_out,
                 "articles_raw": a_raw, "articles_v2": a_out,
                 "articles_removed": exp_removed, "files_removed": exp_files_removed,
                 "match": same}
        if not args.dry_run:
            d_art, d_file = disk_stats[(jur, "articles")], disk_stats[(jur, "files")]
            entry["articles_on_disk"] = d_art
            entry["disk_readback_match"] = (d_art == a_out and d_file == f_out)
            disk_ok = disk_ok and entry["disk_readback_match"]
        per_jur[jur] = entry

    report["transforms"]["1a_law_id_split"] = {
        "files": split_files,
        "articles": sum(x["articles"] for x in split_files),
        "note": "拆分后 law_id 全局唯一，doc_id 前缀同步改写",
    }
    report["transforms"]["1b_empty_text_inventory"] = {
        "total": len(empty_text_items),
        "recoverable_from_sub_chunks": sum(1 for x in empty_text_items if x["sub_chunks"]),
        "note": "本机无原始语料（.txt/.pdf 源文件不在仓库内），sub_chunks 亦无内容，无法补洗；"
                "建议保留清单并在评测分母中对不可召回项显式剔除",
        "items": empty_text_items,
    }
    report["transforms"]["1c_line_merge"] = {
        "jurisdictions": sorted(LINE_MERGE_JURISDICTIONS),
        "items_touched": {k: v for k, v in merge_stat.items() if k.endswith("_items")},
        "chars_removed": {k: v for k, v in merge_stat.items() if k.endswith("_chars")},
    }
    report["transforms"]["1d_date_backfill"] = {
        "files_touched": date_files, "values_filled": date_filled,
        "fields": list(DATE_FIELDS),
    }
    report["transforms"]["1f_cleanup"] = {
        "empty_text_removed": {
            "total": sum(v for (_j, r), v in removed_by_jur.items() if r == "empty_text"),
            "by_jurisdiction": {j: v for (j, r), v in removed_by_jur.items() if r == "empty_text"},
            "note": "条目清单见 1b 的 items 字段（含来源文件、doc_id、是否可恢复）",
        },
        "dup_doc_id_removed": {
            "total": len(dup_doc_log),
            "items": dup_doc_log,
            "note": "澳门 SMP_MO_061 的重复 doc_id 两条均无正文，已被空文本剔除覆盖",
        },
        "sg_id_merge": {
            "files_removed": len(SG_ID_MERGE),
            "items_moved": sum(x["items_moved"] for x in sg_merge_log),
            "pairs": sg_merge_log,
            "same_article_dedup": sg_dedup_log,
            "note": "片段 law_id 由正文首句截断生成（清洗缺陷产物），统一并入立法编号式规范 ID，"
                    "doc_id 前缀同步改写；合并后同条款号重复保留正文更长者",
        },
    }
    report["verification"] = {
        "distinct_law_id_raw": len(raw_laws),
        "distinct_law_id_v2": len(out_laws),
        "per_jurisdiction": per_jur,
        "article_count_match": ok,
        "disk_readback_match": disk_ok if not args.dry_run else None,
    }
    report["inventory_pending_decision"] = [
        {"id": "P1-MACAU-NAME", "desc": "澳门 29 个文件首条 law_name 为正文片段（如「制定本法律。」），"
                                        "另有 5 个为空；ES 的 title 直接取该字段，需决定回填来源"
                                        "（law_name_original / source_url / 正文首条）", "severity": "中"},
        {"id": "P1-SG-STUB", "desc": "新加坡语料为摘录级（每法 1~8 条，共 584 块），"
                                     "非全文；无本地源文件可补，需决定是否重新抓取", "severity": "高"},
        {"id": "P1-SG-BLOCK", "desc": "新加坡 4 部法合并后仍各有旧代块状条目（article_number 为解析异常值 "
                                      "2020/25/31，正文为逐条版本的部分覆盖，实测覆盖率 0.585/0.722/0.901/0.942）；"
                                      "本次仅做同条款号去重、未删除整条，是否剔除待决策", "severity": "低",
         "items": ["SG_ACT_18_2022_ART_025", "SG_ACT_1_2021_ART_2020",
                   "SG_ACT_1_2021_ART_031", "SG_ACT_40_2020_ART_2020"]},
        {"id": "P1-DATE-MISSING", "desc": "40/309 个文件（占 13.9% 的块）全文件无 passing_date，"
                                          "且文件内取值一致、无部分缺失 → 属数据源缺失，回填无解；"
                                          "需决定是否从 law_id 派生粗粒度年份字段，或时间筛选功能声明受限",
         "severity": "中"},
    ]
    report["resolved_actions"] = [
        {"id": "P1-SG-DUP-ID",
         "resolution": "1f-3：4 个片段 law_id 的 18 条并入规范 ID（doc_id 前缀同步改写），"
                       "4 个片段文件不再单独落盘；同条款号重复保留正文更长者 1 条；"
                       "评测集/评测脚本中的片段 ID 引用另行同步"},
        {"id": "P1-MO-DUP-DOCID",
         "resolution": "1f-1：重复的两条 MAC_LEI_17_2020_ART_SCH001 均无正文，随空文本剔除一并消除"},
    ]

    if not args.dry_run:
        with open(REPORT, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    # ---------------------------------------------------------- 控制台摘要
    print("原始 law_id 数: %d  →  清洗后: %d" % (len(raw_laws), len(out_laws)))
    print("泰国拆分文件: %d 个，迁移条款 %d 条" % (len(split_files), sum(x["articles"] for x in split_files)))
    print("1b 空文本条目: %d 条（可恢复 0 条）" % len(empty_text_items))
    print("1c 换行合并: 受影响条目 %s，移除多余字符 %s" % (
        {k: v for k, v in merge_stat.items() if k.endswith("_items")},
        {k: v for k, v in merge_stat.items() if k.endswith("_chars")}))
    print("1d 日期回填: 触及文件 %d 个，填值 %d 处" % (date_files, date_filled))
    print("1f 清除: 空文本 %d 条，重复 doc_id %d 条" % (
        sum(v for (_j, r), v in removed_by_jur.items() if r == "empty_text"), len(dup_doc_log)))
    print("1f 合并: 片段文件 %d 个 → 规范 law_id，迁移条款 %d 条，同条款号去重 %d 条" % (
        len(SG_ID_MERGE), sum(x["items_moved"] for x in sg_merge_log), len(sg_dedup_log)))
    print("逐语域条目数一致性: %s" % ("通过" if ok else "不一致！"))
    for jur, v in per_jur.items():
        if not v["match"]:
            print("   不一致: %s %s" % (jur, v))
    print("报告: %s" % (REPORT if not args.dry_run else "(dry-run 未写)"))


if __name__ == "__main__":
    main()