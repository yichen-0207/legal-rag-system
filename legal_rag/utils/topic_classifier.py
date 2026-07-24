"""
主题分类器：基于 taxonomy_merged.json 的关键词匹配实现。

输出格式：
- topics: topic id 字符串数组
- topic_labels: 中文标签数组
- topics_details: 包含 id/label_zh/score/matched_keywords 的对象数组
- taxonomy_version: 词表版本号
"""

import json
import os
from typing import List, Dict, Any


def _default_taxonomy_path() -> str:
    """返回默认词表路径"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", "taxonomy_merged.json")


class TopicClassifier:
    """基于关键词匹配的主题分类器"""

    def __init__(self, taxonomy_path: str = None):
        taxonomy_path = taxonomy_path or _default_taxonomy_path()
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.topics = data.get("topics", [])
        self.taxonomy_version = data.get("taxonomy_version", "unknown")

        # 预处理：构建 (keyword_lower, keyword_original, weight, topic_id) 列表
        self._keyword_index = []
        for topic in self.topics:
            tid = topic["id"]
            for kw in topic.get("keywords", []):
                self._keyword_index.append((kw.lower(), kw, 1.0, tid))
            for kw in topic.get("keywords_en", []):
                self._keyword_index.append((kw.lower(), kw, 1.0, tid))
            for kw in topic.get("keywords_weak", []):
                self._keyword_index.append((kw.lower(), kw, 0.3, tid))

    def classify(self, text: str) -> List[Dict[str, Any]]:
        """对单条文本进行分类，返回 topics_details 列表。"""
        if not text:
            return []

        text_lower = text.lower()
        topic_scores: Dict[str, Dict[str, Any]] = {}

        for kw_lower, kw_orig, weight, tid in self._keyword_index:
            if kw_lower in text_lower:
                if tid not in topic_scores:
                    topic = next(t for t in self.topics if t["id"] == tid)
                    topic_scores[tid] = {
                        "id": tid,
                        "label_zh": topic.get("label_zh", tid),
                        "score": 0.0,
                        "matched_keywords": set(),
                    }
                topic_scores[tid]["score"] += weight
                topic_scores[tid]["matched_keywords"].add(kw_orig)

        results = []
        for tid, info in topic_scores.items():
            topic = next(t for t in self.topics if t["id"] == tid)
            # 应用 exclude_if 排除规则
            excluded = False
            for ex in topic.get("exclude_if", []):
                if ex.lower() in text_lower:
                    excluded = True
                    break
            if excluded:
                continue
            # 仅保留 score >= 1 的主题（至少命中一个强关键词，或多个弱关键词）
            if info["score"] >= 1.0:
                results.append({
                    "id": tid,
                    "label_zh": info["label_zh"],
                    "score": round(info["score"], 2),
                    "matched_keywords": sorted(info["matched_keywords"]),
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def classify_document(self, text: str) -> Dict[str, Any]:
        """
        对文档进行分类，返回完整的主题字段集合。

        返回：
        {
            "topics": [id1, id2, ...],
            "topic_labels": [label1, label2, ...],
            "topics_details": [...],
            "taxonomy_version": "x.x"
        }
        """
        details = self.classify(text)
        return {
            "topics": [d["id"] for d in details],
            "topic_labels": [d["label_zh"] for d in details],
            "topics_details": details,
            "taxonomy_version": self.taxonomy_version,
        }
