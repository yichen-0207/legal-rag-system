# -*- coding: utf-8 -*-
"""回填 ES 顶层 chunk_type 字段：data/raw_v2 → {doc_id: chunk_type}，ES _id 剥出 parent_doc_id 后写入。

只写顶层 chunk_type，不改 _id / content / embedding / article_number；无需重嵌入、无 LLM 调用。
"""

import argparse
import collections
import json
from pathlib import Path

from core.config import settings
from repositories.elasticsearch import ElasticsearchRepository

BASE = Path(__file__).resolve().parent.parent          # legal_rag/
RAW_V2 = BASE / "data" / "raw_v2"
SKIP_DIRS = {".uploads", "__pycache__", ".git"}

# 已知的源数据解析遗留片段（人工逐条核对确认）：
# 内容为附表罚款表碎片或页眉碎片，被误编成条文号，无有效条文标题。
# 标记为 fragment 后仅从法条序列与目录中移出，内容仍保留在索引中可被检索。
FRAGMENT_RULES = {
    "HK_CAP_200": {"1", "2", "3", "4", "5", "6", "13"},   # 附表2 罚款表碎片 + "父母 第XII部" 页眉
    "HK_CAP_528": {"500"},                                  # 内容仅"附表"的误编号
}


def load_chunk_type_map():
    """遍历 raw_v2，返回 {doc_id: chunk_type}（仅非空）与文件数；命中 FRAGMENT_RULES 者改写为 fragment。"""
    mapping = {}
    files = 0
    for jur_dir in sorted(p for p in RAW_V2.iterdir() if p.is_dir() and p.name not in SKIP_DIRS):
        for src in sorted(jur_dir.glob("*.json")):
            try:
                items = json.loads(src.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(items, list):
                continue
            files += 1
            for it in items:
                doc_id = it.get("doc_id")
                ct = (it.get("chunk_type") or "").strip()
                if not doc_id or not ct:
                    continue
                nums = FRAGMENT_RULES.get(it.get("law_id"))
                if nums and str(it.get("article_number") or "") in nums:
                    ct = "fragment"
                mapping[doc_id] = ct
    return mapping, files


def parent_of(es_id: str) -> str:
    """ES _id → 原始 doc_id：剥离尾部 _sub_{n}。"""
    return es_id.rsplit("_sub_", 1)[0] if "_sub_" in es_id else es_id


def main():
    ap = argparse.ArgumentParser(description="回填 ES 顶层 chunk_type")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写入")
    ap.add_argument("--batch", type=int, default=2000, help="scroll 批大小")
    args = ap.parse_args()

    repo = ElasticsearchRepository()
    client = repo.client
    index = settings.es_index_name

    ct_map, files = load_chunk_type_map()
    print(f"[源] raw_v2 文件 {files} 个，带 chunk_type 条目 {len(ct_map)} 条，类型分布 {dict(collections.Counter(ct_map.values()))}")

    resp = client.search(index=index, body={
        "query": {"match_all": {}}, "size": args.batch, "_source": ["chunk_type"],
    }, scroll="5m")
    scroll_id = resp.get("_scroll_id")

    scanned = matched = to_update = unmatched = 0
    updates, samples = [], []
    try:
        while True:
            hits = resp["hits"]["hits"]
            if not hits:
                break
            for h in hits:
                scanned += 1
                es_id = h["_id"]
                cur = ((h.get("_source") or {}).get("chunk_type") or "").strip()
                want = ct_map.get(parent_of(es_id))
                if not want:
                    unmatched += 1
                    continue
                matched += 1
                if cur != want:
                    to_update += 1
                    updates.append({"_op_type": "update", "_index": index, "_id": es_id,
                                    "doc": {"chunk_type": want}})
                    if len(samples) < 15:
                        samples.append((es_id, cur, want))
            resp = client.scroll(scroll_id=scroll_id, scroll="5m")
            scroll_id = resp.get("_scroll_id")
    finally:
        if scroll_id:
            client.clear_scroll(scroll_id=scroll_id)

    print(f"[扫] ES {index} 共扫描 {scanned} 条")
    print(f"[配] 命中 raw_v2 映射 {matched} 条，其中待更新 {to_update} 条；未命中 {unmatched} 条")
    for s in samples:
        print(f"    样本 {s[0]}: '{s[1]}' -> '{s[2]}'")

    if args.dry_run:
        print("dry-run：未写入任何数据")
        return

    if not updates:
        print("无需更新")
        return
    from elasticsearch.helpers import bulk
    ok, errs = bulk(client, updates, stats_only=False, raise_on_error=False)
    client.indices.refresh(index=index)
    print(f"[写] 已更新 {ok} 条，失败 {len(errs) if isinstance(errs, list) else errs} 条")


if __name__ == "__main__":
    main()
