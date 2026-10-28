"""把 taxonomy.json 合并进系统实际使用的 taxonomy_merged.json。

合并口径（已与用户确认）：
- 以 taxonomy_merged.json 为主体，保留其 38 个主题，不恢复 legal_definitions；
- 不引入 keywords_by_jurisdiction 字段；
- 公共主题的关键词取并集（目标在前，来源去重追加）；
- jurisdiction_topic_index 只补目标缺失的法域，已有法域保持原样；
- taxonomy.json 不含新法域专属关键词（实测新增关键词 0 个），故
  taxonomy_version 保持 2.5，不触发全量重标注。

用法：python data_tools/merge_taxonomy.py [--dry-run]
"""

import argparse
import json
import os
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(BASE, "data", "taxonomy.json")
TARGET = os.path.join(BASE, "data", "taxonomy_merged.json")
NEW_VERSION = None  # 关键词零差异，保持原 taxonomy_version 不变
LIST_FIELDS = ("keywords", "keywords_en", "keywords_weak", "exclude_if")


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def union(target: list, source: list) -> list:
    """目标列表在前，来源中未出现的元素追加在后。"""
    merged = list(target or [])
    for item in source or []:
        if item not in merged:
            merged.append(item)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 taxonomy.json 到 taxonomy_merged.json")
    parser.add_argument("--dry-run", action="store_true", help="只打印差异，不写文件")
    args = parser.parse_args()

    source = load(SOURCE)
    target = load(TARGET)
    src_topics = {t["id"]: t for t in source.get("topics", [])}

    added_keywords = {}
    for topic in target.get("topics", []):
        src = src_topics.get(topic["id"])
        if not src:
            continue
        added = 0
        for field in LIST_FIELDS:
            before = topic.get(field) or []
            after = union(before, src.get(field))
            added += len(after) - len(before)
            topic[field] = after
        if added:
            added_keywords[topic["id"]] = added

    target_index = target.get("jurisdiction_topic_index", {})
    added_jurisdictions = []
    for jur, block in source.get("jurisdiction_topic_index", {}).items():
        if jur not in target_index:
            target_index[jur] = block
            added_jurisdictions.append(jur)
    target["jurisdiction_topic_index"] = target_index

    skipped_topics = [tid for tid in src_topics if tid not in {t["id"] for t in target["topics"]}]

    if NEW_VERSION:
        target["taxonomy_version"] = NEW_VERSION
    target["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    changelog = target.get("changelog", [])
    if isinstance(changelog, str):
        changelog = [changelog]
    changelog.append(
        f"[元数据] 从 taxonomy.json 合并法域索引 {', '.join(added_jurisdictions) or '无'}；"
        f"公共主题关键词零差异（新增 {sum(added_keywords.values())} 个），"
        f"不恢复 legal_definitions 与 keywords_by_jurisdiction；版本保持 {target.get('taxonomy_version')}，不触发重标注"
    )
    target["changelog"] = changelog

    print(f"合并后主题数: {len(target['topics'])}（保持）")
    print(f"新增关键词的主题: {len(added_keywords)} 个, 合计 {sum(added_keywords.values())} 个关键词")
    for tid, n in sorted(added_keywords.items(), key=lambda x: -x[1]):
        print(f"  {tid}: +{n}")
    print(f"新增法域索引: {', '.join(added_jurisdictions) or '无'}")
    print(f"来源独有、未合并的主题: {', '.join(skipped_topics) or '无'}")
    print(f"taxonomy_version: {target.get('taxonomy_version')}")

    if args.dry_run:
        print("dry-run：未写入文件")
        return

    with open(TARGET, "w", encoding="utf-8") as f:
        json.dump(target, f, ensure_ascii=False, indent=2)
    print(f"已写入 {TARGET}")


if __name__ == "__main__":
    main()
