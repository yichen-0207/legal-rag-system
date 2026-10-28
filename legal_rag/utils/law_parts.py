# -*- coding: utf-8 -*-
"""章节目录补全：用离线索引 data/law_parts_index.json 给 outline 补上可读章节名。

ES 中的 metadata.part_title 只有「第一章」这类裸序号（历史数据），章节名留在原始
清洗数据里。索引由 data_tools/build_law_parts_index.py 生成，缺失时自动退化为
不补全，不影响接口可用性。
"""

import json
import os
from typing import Dict, List, Optional

_INDEX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "law_parts_index.json"
)
_LAWS_CACHE: Optional[Dict[str, List[List[str]]]] = None


def _load_laws() -> Dict[str, List[List[str]]]:
    global _LAWS_CACHE
    if _LAWS_CACHE is not None:
        return _LAWS_CACHE
    try:
        with open(_INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _LAWS_CACHE = data.get("laws", {}) if isinstance(data, dict) else {}
    except Exception:
        _LAWS_CACHE = {}
    return _LAWS_CACHE


def enrich_parts(law_id: str, items: List[Dict]) -> List[Dict]:
    """按索引顺序为条款列表（outline 或 chunks）写入 part_title。

    以「章节起始条款号」为锚点：条款号命中锚点即切换章节，未命中的条目沿用当前章节。
    附表与解析遗留片段不参与章节归并，保持原值。
    """
    parts = _load_laws().get(str(law_id or ""))
    if not parts or not items:
        return items

    titles = [title for _num, title in parts]
    starts = {str(num): idx for idx, (num, _title) in enumerate(parts)}

    current = None
    for item in items:
        if str(item.get("chunk_type") or "").lower() in ("schedule", "fragment"):
            continue
        idx = starts.get(str(item.get("article_number") or ""))
        if idx is not None:
            current = idx
        item["part_title"] = titles[current] if current is not None else ""

    return items
