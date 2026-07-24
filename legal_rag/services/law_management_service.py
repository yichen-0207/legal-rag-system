import json
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from core.config import settings
from core.jurisdiction import get_jurisdiction_info, JURISDICTION_INFO
from repositories.elasticsearch import ElasticsearchRepository
from services.abstract_service import AbstractService
from utils.chunker import chunk_text, chunk_from_structure


class LawManagementService:
    def __init__(self):
        self.repo = ElasticsearchRepository()
        self.data_dir = Path(settings.data_dir)
        self._dir_prefix_cache: Dict[str, str] = {}
        self._init_dir_prefix_cache()
    
    def _init_dir_prefix_cache(self):
        """扫描现有目录，建立目录名到ID前缀的映射"""
        if not self.data_dir.exists():
            return
        
        for item in self.data_dir.iterdir():
            if item.is_dir() and item.name != "recycle_bin":
                json_files = list(item.glob("*.json"))
                if json_files:
                    for json_file in json_files:
                        stem = json_file.stem
                        if stem.startswith("SMP_"):
                            parts = stem.split("_")
                            if len(parts) >= 2:
                                # 如果目录名是中文，转换为英文目录名
                                if item.name in JURISDICTION_INFO:
                                    dir_name, prefix_code = JURISDICTION_INFO[item.name]
                                    self._dir_prefix_cache[dir_name.lower()] = f"SMP_{prefix_code}"
                                else:
                                    self._dir_prefix_cache[item.name.lower()] = f"SMP_{parts[1]}"
                                break
    
    def _find_or_create_dir(self, jurisdiction: str) -> Tuple[Path, str]:
        """
        查找或创建法域目录，返回(目录路径, ID前缀)
        自动从现有目录识别，或为新法域创建目录
        """
        jur_lower = jurisdiction.lower().strip()
        
        # 使用法域信息获取标准英文目录名和ID前缀
        dir_name, prefix_code = get_jurisdiction_info(jurisdiction)
        prefix = f"SMP_{prefix_code}"
        
        # 检查缓存中是否有该目录
        if dir_name.lower() in self._dir_prefix_cache:
            prefix = self._dir_prefix_cache[dir_name.lower()]
        
        dir_path = self.data_dir / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            self._dir_prefix_cache[dir_name.lower()] = prefix
        
        return dir_path, prefix
    
    def _generate_law_id(self, jurisdiction: str) -> str:
        """生成唯一的law_id，确保ID连续递增"""
        dir_path, prefix = self._find_or_create_dir(jurisdiction)
        
        existing_ids = []
        if dir_path.exists():
            for json_file in dir_path.glob("*.json"):
                stem = json_file.stem
                if stem.startswith(prefix):
                    existing_ids.append(stem)
        
        if existing_ids:
            nums = []
            for id_str in existing_ids:
                try:
                    num = int(id_str.split("_")[-1])
                    nums.append(num)
                except:
                    pass
            if nums:
                return f"{prefix}_{str(max(nums) + 1).zfill(3)}"
        
        return f"{prefix}_001"
    
    def _get_dir_for_law_id(self, law_id: str) -> Optional[Path]:
        """根据law_id找到对应的目录"""
        if not law_id:
            return None
        
        parts = law_id.split("_")
        if len(parts) >= 2:
            prefix_code = parts[1].lower()
            
            for dir_name, prefix in self._dir_prefix_cache.items():
                if prefix.lower().endswith(prefix_code):
                    return self.data_dir / dir_name
        
        for item in self.data_dir.iterdir():
            if item.is_dir() and item.name != "recycle_bin":
                for json_file in item.glob("*.json"):
                    if json_file.stem == law_id or json_file.stem.startswith(f"{law_id}_"):
                        return item
        
        return None
    
    def _save_json_file(self, data: Dict) -> Path:
        """保存JSON文件到对应法域目录"""
        jurisdiction = data.get("jurisdiction", "")
        law_id = data.get("law_id", "")
        
        dir_path, _ = self._find_or_create_dir(jurisdiction)
        file_path = dir_path / f"{law_id}.json"
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return file_path
    
    def _process_and_index(self, law_id: str):
        """处理法规并增量添加到Elasticsearch，支持新旧两种格式"""
        dir_path = self._get_dir_for_law_id(law_id)
        if not dir_path:
            raise FileNotFoundError(f"无法找到法规文件目录: {law_id}")

        file_path = dir_path / f"{law_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"法规文件不存在: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 检测数据格式
        is_list_format = isinstance(data, list)

        documents = []
        metadatas = []
        ids = []

        if is_list_format:
            # 新格式：数组（cleaned文件）
            # 从第一个元素提取公共元数据
            law_id_from_data = data[0].get('law_id', '') if data else ''
            title = data[0].get('law_name', '') if data else ''
            jurisdiction = data[0].get('jurisdiction', '') if data else ''
            passing_date = data[0].get('passing_date', '') if data else ''

            structure_chunks = chunk_from_structure(data)

            for i, chunk in enumerate(structure_chunks):
                doc_id = chunk.get('_meta', {}).get('doc_id', f"{law_id}_chunk_{i}")
                article_num = chunk["article_number"] if chunk["article_number"] is not None else ""

                metadata = {
                    "law_id": law_id_from_data or law_id,
                    "jurisdiction": jurisdiction,
                    "title": title,
                    "passing_date": passing_date,
                    "source_file": str(file_path),
                    "chunk_index": i,
                    "article_number": article_num,
                    "part_title": chunk.get("part_title", ""),
                    "article_title": chunk.get('_meta', {}).get('article_title', ''),
                    "effective_date": chunk.get('_meta', {}).get('effective_date', '')
                }
                documents.append(chunk["text"])
                metadatas.append(metadata)
                ids.append(doc_id)

        else:
            # 旧格式：字典
            title = data.get("title_zh", "") or data.get("title_en", "") or data.get("title_pt", "")
            full_text = data.get("text_zh", "") or data.get("text_pt", "") or data.get("text_en", "")

            structure_chunks = chunk_from_structure(data)

            if structure_chunks:
                for i, chunk in enumerate(structure_chunks):
                    doc_id = f"{law_id}_chunk_{i}"
                    article_num = chunk["article_number"] if chunk["article_number"] is not None else ""
                    metadata = {
                        "law_id": law_id,
                        "jurisdiction": data.get("jurisdiction", ""),
                        "title": title,
                        "law_number": data.get("law_number", "") or "",
                        "passing_date": data.get("passing_date", "") or "",
                        "publisher": data.get("publisher", "") or "",
                        "source_file": str(file_path),
                        "chunk_index": i,
                        "article_number": article_num,
                        "part_title": chunk.get("part_title", "")
                    }
                    documents.append(chunk["text"])
                    metadatas.append(metadata)
                    ids.append(doc_id)
            elif full_text:
                for i, chunk in enumerate(chunk_text(full_text)):
                    doc_id = f"{law_id}_chunk_{i}"
                    metadata = {
                        "law_id": law_id,
                        "jurisdiction": data.get("jurisdiction", ""),
                        "title": title,
                        "law_number": data.get("law_number", "") or "",
                        "passing_date": data.get("passing_date", "") or "",
                        "publisher": data.get("publisher", "") or "",
                        "source_file": str(file_path),
                        "chunk_index": i
                    }
                    documents.append(chunk)
                    metadatas.append(metadata)
                    ids.append(doc_id)

        if documents:
            self.repo.add_documents(documents, metadatas, ids)

        return len(documents)
    
    def create_law(self, data: Dict) -> Dict:
        """新增法规，支持新旧两种数据格式"""
        
        # 检测数据格式
        is_list_format = isinstance(data, list)
        
        if is_list_format:
            # 新cleaned格式（数组）
            if not data:
                raise ValueError("数据不能为空")
            
            first_item = data[0]
            jurisdiction = first_item.get("jurisdiction", "")
            law_name = first_item.get("law_name", "")
            
            if not jurisdiction:
                raise ValueError("必须指定法域（jurisdiction）")
            if not law_name:
                raise ValueError("必须指定法规名称（law_name）")
            
            # 从数据中提取或生成law_id
            law_id_from_data = first_item.get("law_id", "")
            
            dir_path, _ = self._find_or_create_dir(jurisdiction)
            
            if law_id_from_data:
                law_id = law_id_from_data
            else:
                law_id = self._generate_law_id(jurisdiction)
                # 更新数据中的law_id
                for item in data:
                    item["law_id"] = law_id
            
            # 构建保存的数据结构
            save_data = data
            save_data_source_file = str(dir_path / f"{law_id}.json")
            
        else:
            # 旧格式（字典）
            if "jurisdiction" not in data or not data["jurisdiction"]:
                raise ValueError("必须指定法域（jurisdiction）")
            
            if "title_zh" not in data and "title_en" not in data and "title_pt" not in data:
                raise ValueError("必须提供至少一个标题（title_zh/title_en/title_pt）")
            
            jurisdiction = data["jurisdiction"]
            dir_path, _ = self._find_or_create_dir(jurisdiction)
            
            if "law_id" in data and data["law_id"]:
                law_id = data["law_id"]
            else:
                law_id = self._generate_law_id(jurisdiction)
            
            data["law_id"] = law_id
            save_data = data
            save_data_source_file = str(dir_path / f"{law_id}.json")
        
        # 统一保存文件
        if is_list_format:
            file_path = dir_path / f"{law_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
        else:
            self._save_json_file(save_data)
        
        # 处理并索引
        chunk_count = self._process_and_index(law_id)
        
        # 后台异步生成摘要
        self._start_abstract_generation(law_id)
        
        return {
            "law_id": law_id,
            "message": "法规创建成功，摘要将在后台生成",
            "chunk_count": chunk_count
        }
    
    def delete_law(self, law_id: str) -> Dict:
        """删除法规"""
        dir_path = self._get_dir_for_law_id(law_id)
        if not dir_path:
            raise FileNotFoundError(f"无法找到法规文件: {law_id}")
        
        file_path = dir_path / f"{law_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"法规文件不存在: {law_id}")
        
        deleted_count = self.repo.delete_by_law_id(law_id)
        file_path.unlink()
        
        return {
            "law_id": law_id,
            "message": "法规删除成功",
            "deleted_chunks": deleted_count
        }
    
    def update_law(self, law_id: str, updates: Dict, overwrite: bool = False) -> Dict:
        """修改法规
        
        Args:
            law_id: 法规ID
            updates: 更新数据
            overwrite: 是否完全覆盖原有数据（True: 用updates完全替换原数据; False: 合并更新）
        """
        dir_path = self._get_dir_for_law_id(law_id)
        if not dir_path:
            raise FileNotFoundError(f"无法找到法规文件: {law_id}")
        
        file_path = dir_path / f"{law_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"法规文件不存在: {law_id}")
        
        # 读取原始数据以获取必要元数据
        with open(file_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)
        
        # 根据overwrite参数决定如何处理数据
        if overwrite:
            # 检测更新数据的格式
            is_list_format = isinstance(updates, list)
            
            if is_list_format:
                # 新格式（数组）：直接使用，但确保law_id一致
                data = updates
                for item in data:
                    item["law_id"] = law_id
            else:
                # 旧格式（字典）：保留必要的系统字段
                data = updates.copy()
                data["law_id"] = law_id
                # 保留原始的source_file等系统字段
                if "source_file" in original_data and "source_file" not in data:
                    data["source_file"] = original_data["source_file"]
        else:
            # 合并更新：读取原数据并合并
            data = original_data.copy()
            
            if isinstance(updates, dict):
                data.update(updates)
            elif isinstance(updates, list):
                # 如果更新数据是数组格式，直接替换
                data = updates
                for item in data:
                    item["law_id"] = law_id
        
        # 删除旧索引，保存新数据，重新索引
        self.repo.delete_by_law_id(law_id)
        
        # 统一保存逻辑
        if isinstance(data, list):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            self._save_json_file(data)
        
        chunk_count = self._process_and_index(law_id)
        
        # 后台异步重新生成摘要（强制覆盖旧摘要）
        self._start_abstract_generation(law_id, force=True)
        
        return {
            "law_id": law_id,
            "message": "法规更新成功，摘要将在后台重新生成",
            "chunk_count": chunk_count,
            "overwrite": overwrite
        }
    
    def get_law_raw_data(self, law_id: str) -> Optional[Dict]:
        """获取原始JSON数据"""
        dir_path = self._get_dir_for_law_id(law_id)
        if not dir_path:
            return None
        
        file_path = dir_path / f"{law_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def sync_index(self) -> Dict:
        """
        同步 Elasticsearch 索引与文件系统
        清理文件不存在的孤立索引
        """
        # 获取 Elasticsearch 中所有法规
        es_laws = self.repo.get_all_laws()
        es_law_ids = {law["law_id"] for law in es_laws}
        
        # 获取文件系统中所有法规
        file_law_ids = set()
        if self.data_dir.exists():
            for item in self.data_dir.iterdir():
                if item.is_dir() and item.name != "recycle_bin":
                    for json_file in item.glob("*.json"):
                        file_law_ids.add(json_file.stem)
        
        # 找出孤立的索引（ES中有但文件中没有）
        orphan_ids = es_law_ids - file_law_ids
        
        deleted_count = 0
        for law_id in orphan_ids:
            count = self.repo.delete_by_law_id(law_id)
            deleted_count += count
        
        return {
            "es_law_count": len(es_law_ids),
            "file_law_count": len(file_law_ids),
            "orphan_count": len(orphan_ids),
            "deleted_chunks": deleted_count,
            "orphan_ids": list(orphan_ids)
        }
    
    def _start_abstract_generation(self, law_id: str, force: bool = False):
        """在后台线程中启动摘要生成
        
        Args:
            law_id: 法规ID
            force: 是否强制重新生成
        """
        def _generate_abstract_async():
            try:
                abstract_service = AbstractService()
                abstract_service.generate_for_law(law_id, force=force)
            except Exception as e:
                print(f"[LawManagementService] 后台生成摘要异常: {e}")
        
        thread = threading.Thread(target=_generate_abstract_async, daemon=True)
        thread.start()