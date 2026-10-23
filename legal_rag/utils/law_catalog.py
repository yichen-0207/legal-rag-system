"""法律目录管理（law_catalog.json）：维护法律级元信息，与 ES 数据保持同步。

记录字段：law_id / 名称 / 法域 / 类型 / 日期 / 条款与分块数 / 主题 / 状态 / 来源文件。
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any


class LawCatalog:
    """法律目录管理器"""

    def __init__(self, catalog_path: str = None):
        """catalog_path 缺省为 data/law_catalog.json。"""
        if catalog_path:
            self.catalog_path = Path(catalog_path)
        else:
            # 默认路径：与 data 目录同级
            self.catalog_path = Path(__file__).parent.parent / "data" / "law_catalog.json"

        self._ensure_catalog_exists()

    def _ensure_catalog_exists(self):
        """确保 catalog 文件存在"""
        if not self.catalog_path.exists():
            self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
            initial_data = {
                "version": "1.0",
                "last_updated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "total_laws": 0,
                "laws": []
            }
            with open(self.catalog_path, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, ensure_ascii=False, indent=2)

    def load(self) -> Dict:
        """加载 catalog 数据"""
        try:
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  加载 law_catalog.json 失败: {e}")
            return {"version": "1.0", "last_updated": "", "total_laws": 0, "laws": []}

    def save(self, data: Dict):
        """保存 catalog 数据"""
        data["last_updated"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        data["total_laws"] = len(data.get("laws", []))

        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def find_by_law_id(self, law_id: str) -> Optional[Dict]:
        """根据 law_id 查找法律记录"""
        catalog = self.load()
        for law in catalog.get("laws", []):
            if law.get("law_id") == law_id:
                return law
        return None

    def add_or_update(
        self,
        law_id: str,
        law_name_zh: str,
        jurisdiction: str,
        total_articles: int,
        total_chunks: int,
        law_name_original: str = "",
        law_type: str = "",
        passing_date: str = "",
        effective_date: str = "",
        publication_date: str = "",
        topics: List[str] = None,
        source_file: str = "",
        status: str = "active"
    ) -> Dict:
        """新增或更新法律记录；状态变更时归档 previous_versions。"""
        catalog = self.load()
        now = datetime.now().strftime("%Y-%m-%d")

        existing_record = self.find_by_law_id(law_id)

        new_record = {
            "law_id": law_id,
            "law_name_zh": law_name_zh,
            "law_name_original": law_name_original,
            "jurisdiction": jurisdiction,
            "law_type": law_type,
            "passing_date": passing_date,
            "effective_date": effective_date,
            "publication_date": publication_date,
            "total_articles": total_articles,
            "total_chunks": total_chunks,
            "topics": topics or [],
            "ingest_date": now,
            "last_updated": now,
            "status": status,
            "source_file": source_file
        }

        if existing_record:
            index = next(i for i, law in enumerate(catalog["laws"]) if law.get("law_id") == law_id)

            # 状态变化（如 active→amended）时归档旧版本
            if existing_record.get("status") != status and status != "deleted":
                if "previous_versions" not in existing_record:
                    existing_record["previous_versions"] = []
                
                previous_version = {
                    "version_date": existing_record.get("last_updated", ""),
                    "status": existing_record.get("status"),
                    "total_articles": existing_record.get("total_articles", 0),
                    "total_chunks": existing_record.get("total_chunks", 0),
                }
                existing_record["previous_versions"].append(previous_version)

            # 合并新数据
            new_record["ingest_date"] = existing_record.get("ingest_date", now)  # 首次入库日期不变
            catalog["laws"][index] = new_record

            print(f"📝 更新 catalog 记录: {law_id} ({law_name_zh})")
        else:
            catalog["laws"].append(new_record)
            print(f"➕ 新增 catalog 记录: {law_id} ({law_name_zh})")

        self.save(catalog)
        return new_record

    def update_from_chunks(self, chunks_data: List[Dict]) -> Dict:
        """从同一法律的 chunks 汇总元信息并写入 catalog。"""
        if not chunks_data:
            return {}

        first_chunk = chunks_data[0]

        law_id = first_chunk.get("_meta", {}).get("law_id", "") or first_chunk.get("metadata", {}).get("law_id", "")
        law_name_zh = first_chunk.get("_meta", {}).get("law_name", "") or first_chunk.get("metadata", {}).get("title", "")
        jurisdiction = first_chunk.get("_meta", {}).get("jurisdiction", "") or first_chunk.get("metadata", {}).get("jurisdiction", "")
        source_file = first_chunk.get("_meta", {}).get("source_file", "").split("\\")[-1] or ""
        
        # 统计条款数（按 article_number 去重）
        article_numbers = set()
        for chunk in chunks_data:
            article_num = chunk.get("article_number", "")
            if article_num:
                article_numbers.add(article_num)
        total_articles = len(article_numbers)
        
        total_chunks = len(chunks_data)
        
        # 收集所有 topics 并去重
        all_topics = []
        for chunk in chunks_data:
            topics = chunk.get("metadata", {}).get("topic", [])
            if isinstance(topics, list):
                all_topics.extend(topics)
            elif topics:
                all_topics.append(topics)
        unique_topics = list(set(all_topics))
        
        # 其他可选字段
        passing_date = first_chunk.get("_meta", {}).get("passing_date", "") or first_chunk.get("metadata", {}).get("passing_date", "")
        effective_date = first_chunk.get("_meta", {}).get("effective_date", "") or first_chunk.get("metadata", {}).get("effective_date", "")
        publication_date = first_chunk.get("_meta", {}).get("publication_date", "") or first_chunk.get("metadata", {}).get("publication_date", "")

        return self.add_or_update(
            law_id=law_id,
            law_name_zh=law_name_zh,
            jurisdiction=jurisdiction,
            total_articles=total_articles,
            total_chunks=total_chunks,
            source_file=source_file,
            passing_date=passing_date,
            effective_date=effective_date,
            publication_date=publication_date,
            topics=unique_topics[:10],  # 最多保留10个主题标签
            status="active"
        )

    def delete(self, law_id: str) -> bool:
        """标记删除（软删除，保留审计记录）。"""
        catalog = self.load()
        original_count = len(catalog["laws"])
        
        # 标记为 deleted 而非物理删除（便于审计）
        for i, law in enumerate(catalog["laws"]):
            if law.get("law_id") == law_id:
                catalog["laws"][i]["status"] = "deleted"
                catalog["laws"][i]["last_updated"] = datetime.now().strftime("%Y-%m-%d")
                print(f"🗑️  标记删除 catalog 记录: {law_id}")
                self.save(catalog)
                return True
        
        print(f"⚠️  未找到 catalog 记录: {law_id}")
        return False

    def get_all_laws(self, jurisdiction: str = None, status: str = None) -> List[Dict]:
        """按法域 / 状态筛选法律记录。"""
        catalog = self.load()
        laws = catalog.get("laws", [])

        if jurisdiction:
            laws = [l for l in laws if l.get("jurisdiction") == jurisdiction]

        if status:
            laws = [l for l in laws if l.get("status") == status]

        return laws

    def get_statistics(self) -> Dict:
        """汇总法规数、分块数及按法域 / 状态 / 年份的分布。"""
        catalog = self.load()
        laws = [l for l in catalog.get("laws", []) if l.get("status") != "deleted"]

        stats = {
            "total_laws": len(laws),
            "by_jurisdiction": {},
            "by_status": {},
            "by_year": {},
            "total_chunks": sum(l.get("total_chunks", 0) for l in laws),
            "total_articles": sum(l.get("total_articles", 0) for l in laws),
        }

        for law in laws:
            jur = law.get("jurisdiction", "未知")
            stats["by_jurisdiction"][jur] = stats["by_jurisdiction"].get(jur, 0) + 1

            status_val = law.get("status", "未知")
            stats["by_status"][status_val] = stats["by_status"].get(status_val, 0) + 1

            year = law.get("passing_date", "")[:4] if law.get("passing_date") else "未知"
            stats["by_year"][year] = stats["by_year"].get(year, 0) + 1

        return stats

    def rebuild_from_es(self, es_repo) -> Dict:
        """从 ES 重建 catalog（用于修复损坏或丢失的目录）。"""
        print("🔄 开始从 ES 重建 law_catalog.json...")
        
        all_laws = es_repo.get_all_laws()
        catalog = {"version": "1.0", "last_updated": "", "total_laws": 0, "laws": []}
        now = datetime.now().strftime("%Y-%m-%d")

        for law_info in all_laws:
            law_id = law_info.get("law_id")
            
            law_detail = es_repo.get_law_by_id(law_id)
            if law_detail:
                record = {
                    "law_id": law_id,
                    "law_name_zh": law_detail.get("title", ""),
                    "law_name_original": "",
                    "jurisdiction": law_detail.get("jurisdiction", ""),
                    "law_type": "",
                    "passing_date": law_detail.get("passing_date", ""),
                    "effective_date": "",
                    "total_articles": law_info.get("article_count", 0),
                    "total_chunks": law_info.get("article_count", 0),
                    "topics": [],
                    "ingest_date": now,
                    "last_updated": now,
                    "status": "active",
                    "source_file": ""
                }
                catalog["laws"].append(record)

        self.save(catalog)
        stats = self.get_statistics()

        print(f"✅ Catalog 重建完成:")
        print(f"   - 总法规数: {stats['total_laws']}")
        print(f"   - 总文档块数: {stats['total_chunks']}")
        
        return stats


# 全局单例实例
_catalog_instance = None

def get_catalog() -> LawCatalog:
    """获取全局 catalog 实例"""
    global _catalog_instance
    if _catalog_instance is None:
        _catalog_instance = LawCatalog()
    return _catalog_instance
