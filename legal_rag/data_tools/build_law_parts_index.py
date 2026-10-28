# -*- coding: utf-8 -*-
"""生成章节目录索引 data/law_parts_index.json。

背景：ES 索引里的 metadata.part_title 由 chunker 写入，历史数据只有「第一章」这类
裸序号；章节名（chapter_title）留在原始清洗数据里没有入库。详情页目录需要
「第一章 总则」这样的可读章节名，故离线抽取一份轻量索引，供 API 补全，
避免每次请求扫描原始文件或重建索引。

用法（在 legal_rag 目录下）:
    python data_tools/build_law_parts_index.py
"""

import json
import os
import re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_v2")
OUT_PATH = os.path.join(BASE_DIR, "data", "law_parts_index.json")

# 章节名长度上限：超过通常是把正文首句误当标题
MAX_TITLE_LEN = 24


def normalize_title(chapter: str, chapter_title: str) -> str:
    """拼接「章节序号 + 章节名」，并过滤明显噪声。"""
    chapter = str(chapter or "").strip()
    title = str(chapter_title or "").strip()

    if title:
        # 源数据常见「总 则」，去掉中文字符之间的空格
        title = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", title)
        title = re.sub(r"\s+", " ", title)
        if len(title) > MAX_TITLE_LEN or title[0].islower():
            title = ""

    return f"{chapter} {title}".strip() if title else chapter


def build_law_parts(items) -> list:
    """按 chunk_index 顺序抽取章节序列：返回 [[起始条款号, 章节名], ...]。"""
    parts = []
    current = ""

    ordered = sorted(items, key=lambda x: x.get("chunk_index") or 0)
    for item in ordered:
        if (item.get("chunk_type") or "") in ("schedule", "fragment"):
            continue
        raw_chapter = str(item.get("chapter") or "").strip()
        name = normalize_title(raw_chapter, item.get("chapter_title"))
        if not name or name == current:
            continue
        current = name
        parts.append([str(item.get("article_number") or ""), name])

    return parts


def main():
    laws = {}
    scanned = 0

    for root, _dirs, files in os.walk(RAW_DIR):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[跳过] {fname}: {e}")
                continue
            if not isinstance(data, list) or not data:
                continue

            law_id = str(data[0].get("law_id") or "")
            if not law_id:
                continue

            scanned += 1
            parts = build_law_parts(data)
            if parts:
                laws[law_id] = parts

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "data/raw_v2",
        "laws": laws,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    total_parts = sum(len(v) for v in laws.values())
    print(f"扫描法规 {scanned} 部，产出章节索引 {len(laws)} 部 / {total_parts} 个章节节点 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
