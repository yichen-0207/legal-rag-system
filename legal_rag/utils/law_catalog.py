"""
法律目录管理模块 (law_catalog.json)

功能：
- 维护法律级别的元信息索引文件
- 支持新增、更新、查询法律目录记录
- 与 ES 数据保持同步

数据结构：
{
  "version": "1.0",
  "last_updated": "2024-01-15T10:30:00",
  "total_laws": 21,
  "laws": [
    {
      "law_id": "MAC_LEI_13_2019",
      "law_name_zh": "網絡安全法",
      "law_name_original": "",
      "jurisdiction": "澳门",
      "law_type": "法律",
      "passing_date": "2019-06-06",
      "effective_date": "2019-08-05",
      "total_articles": 28,
      "total_chunks": 28,
      "topics": ["网络安全", "关键基础设施"],
      "ingest_date": "2024-01-15",
      "last_updated": "2024-01-15",
      "status": "active",
      "source_file": "MO_13196741_cleaned.json"
    }
  ]
}
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any


class LawCatalog:
    """法律目录管理器"""

    def __init__(self, catalog_path: str = None):
        """
        初始化目录管理器

        参数:
            catalog_path: catalog 文件路径，默认为 data/law_catalog.json
        """
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
        """
        新增或更新法律目录记录

        参数:
            law_id: 法律ID
            law_name_zh: 中文名称
            jurisdiction: 法域
            total_articles: 总条款数
            total_chunks: 总文档块数
            law_name_original: 原始名称
            law_type: 法律类型
            passing_date: 通过日期
            effective_date: 生效日期
            publication_date: 发布日期
            topics: 主题标签列表
            source_file: 来源文件名
            status: 状态 (active/amended/repealed/deleted/missing)

        返回:
            更新后的法律记录
        """
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
            # 更新现有记录，保留历史版本信息
            index = next(i for i, law in enumerate(catalog["laws"]) if law.get("law_id") == law_id)
            
            # 如果状态变化（如从 active 变为 amended），保留旧版本信息
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
            # 新增记录
            catalog["laws"].append(new_record)
            print(f"➕ 新增 catalog 记录: {law_id} ({law_name_zh})")

        self.save(catalog)
        return new_record

    def update_from_chunks(self, chunks_data: List[Dict]) -> Dict:
        """
        从 chunks 数据中提取元信息并更新 catalog

        参数:
            chunks_data: 同一法律的多个 chunk 列表

        返回:
            更新后的法律记录
        """
        if not chunks_data:
            return {}

        first_chunk = chunks_data[0]
        
        # 提取基础字段
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
        """
        删除或标记删除法律记录

        参数:
            law_id: 要删除的法律ID

        返回:
            是否成功删除
        """
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
        """
        获取所有法律记录（支持筛选）

        参数:
            jurisdiction: 法域筛选（可选）
            status: 状态筛选（可选）

        返回:
            法律记录列表
        """
        catalog = self.load()
        laws = catalog.get("laws", [])

        if jurisdiction:
            laws = [l for l in laws if l.get("jurisdiction") == jurisdiction]

        if status:
            laws = [l for l in laws if l.get("status") == status]

        return laws

    def get_statistics(self) -> Dict:
        """
        获取统计信息

        返回:
            包含各种统计数据的字典
        """
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
        """
        从 ES 重建 catalog（用于修复损坏或丢失的 catalog）

        参数:
            es_repo: ElasticsearchRepository 实例

        返回:
            重建后的统计数据
        """
        print("🔄 开始从 ES 重建 law_catalog.json...")
        
        all_laws = es_repo.get_all_laws()
        catalog = {"version": "1.0", "last_updated": "", "total_laws": 0, "laws": []}
        now = datetime.now().strftime("%Y-%m-%d")

        for law_info in all_laws:
            law_id = law_info.get("law_id")
            
            # 获取该法律的详细信息
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
