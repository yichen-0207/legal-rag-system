import json
import sys
import argparse
import time
from pathlib import Path
from typing import Iterator, Tuple, List, Dict

from core.config import settings
from utils.chunker import chunk_text, chunk_from_structure
from repositories.elasticsearch import ElasticsearchRepository, _to_label_zh
from utils.law_catalog import LawCatalog
from utils.topic_classifier import TopicClassifier
from services.abstract_service import AbstractService


def process_json_file(json_file: Path, repo, skip_existing: bool = True) -> Iterator[Tuple[str, Dict, str]]:
    """
    处理单个JSON文件，支持新旧两种格式

    旧格式: 字典 {"law_id": ..., "structure": {...}, ...}
    新格式: 数组 [{"doc_id": ..., "text_zh": ...}, ...] (cleaned文件)

    参数:
        skip_existing: 如果为True，当数据库中已有相同law_id时跳过
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 检测数据格式
    is_list_format = isinstance(data, list)

    if is_list_format:
        # 新格式：数组（cleaned文件）
        if not data or len(data) == 0:
            print(f"⚠️  文件 {json_file.name} 为空数组，跳过")
            return

        # 从第一个元素提取公共元数据
        first_item = data[0]
        law_id = first_item.get('law_id', '')
        title = first_item.get('law_name', '')
        jurisdiction = first_item.get('jurisdiction', '')
        passing_date = first_item.get('passing_date', '')
        effective_date = first_item.get('effective_date', '')
        publication_date = first_item.get('publication_date', '')

        # 验证关键字段（从元素级别提取）
        if not law_id or law_id == 'null':
            # 尝试从doc_id推断（格式: MAC_LEI_XX_XXXX_ART_XXX 或 ART_XXX）
            sample_doc_id = first_item.get('doc_id', '')
            if sample_doc_id and '_' in sample_doc_id:
                # 从 "MAC_LEI_14_2001_ART_001" 提取 "MAC_LEI_14_2001"
                parts = sample_doc_id.split('_ART_')[0] if '_ART_' in sample_doc_id else sample_doc_id
                if parts and not parts.startswith('ART'):
                    law_id = parts
                    print(f"ℹ️  文件 {json_file.name} law_id为空，从doc_id推断: {law_id}")
                else:
                    # 使用文件名作为law_id（去掉_cleaned后缀）
                    law_id = json_file.stem.replace('_cleaned', '')
                    print(f"ℹ️  文件 {json_file.name} 无法推断law_id，使用文件名: {law_id}")
            else:
                # 最后的备选方案
                law_id = json_file.stem.replace('_cleaned', '')
                print(f"ℹ️  文件 {json_file.name} law_id为空，使用文件名: {law_id}")

        # jurisdiction 可选，如果缺失则使用目录名推断
        if not jurisdiction:
            # 尝试从父目录名推断法域
            parent_dir = json_file.parent.name
            jurisdiction = parent_dir
            print(f"ℹ️  文件 {json_file.name} 缺少 jurisdiction，使用目录名: {jurisdiction}")

        # 去重检查：如果数据库中已有该law_id，则跳过
        if skip_existing and law_id and repo.exists_law_id(law_id):
            existing_count = repo.get_law_chunk_count(law_id)
            print(f"⏭️  跳过 {json_file.name} (law_id={law_id}) - 数据库中已存在 ({existing_count} 个文档块)")
            return

        print(f"✅ 处理新格式文件: {json_file.name}")
        print(f"   法规ID: {law_id}, 标题: {title}, 条款数: {len(data)}")

        # 使用新的处理函数
        structure_chunks = chunk_from_structure(data)

        if not structure_chunks:
            print(f"⚠️  文件 {json_file.name} 未生成任何chunks")
            return

        for i, chunk in enumerate(structure_chunks):
            doc_id = chunk.get('_meta', {}).get('doc_id', f"{law_id}_chunk_{i}")
            article_num = chunk["article_number"] if chunk["article_number"] is not None else ""

            # 提取 topics 信息
            topics = chunk.get('topics', [])
            topic_labels = chunk.get('topic_labels', [])
            # 确保 topic_labels 全部为中文（将英文ID转换为中文）
            topic_labels = [_to_label_zh(t) for t in topic_labels]
            # topic 字段：用于 ES keyword 筛选（取第一个或合并）
            primary_topic = topic_labels[0] if topic_labels else ''

            metadata = {
                "law_id": law_id,
                "jurisdiction": jurisdiction,
                "title": title,
                "passing_date": passing_date or chunk.get('_meta', {}).get('passing_date', ''),
                "effective_date": effective_date or chunk.get('_meta', {}).get('effective_date', ''),
                "publication_date": publication_date or chunk.get('_meta', {}).get('publication_date', ''),
                "source_file": str(json_file),
                "chunk_index": i,
                "article_number": article_num,
                "part_title": chunk.get("part_title", ""),
                # 额外保留cleaned数据的元数据
                "article_title": chunk.get('_meta', {}).get('article_title', ''),
                # 主题信息（支持筛选）
                "topic": primary_topic,
                "topics": topics,
                "topic_labels": topic_labels
            }
            yield chunk["text"], metadata, doc_id

    else:
        # 旧格式：字典
        law_id = data.get("law_id", "")
        title = data.get("title_zh", "") or data.get("title_en", "") or data.get("title_pt", "")
        full_text = data.get("text_zh", "") or data.get("text_pt", "") or data.get("text_en", "")

        if not law_id:
            print(f"⚠️  文件 {json_file.name} 缺少 law_id 字段，跳过")
            return

        # 去重检查：如果数据库中已有该law_id，则跳过
        if skip_existing and repo.exists_law_id(law_id):
            existing_count = repo.get_law_chunk_count(law_id)
            print(f"⏭️  跳过 {json_file.name} (law_id={law_id}) - 数据库中已存在 ({existing_count} 个文档块)")
            return

        print(f"✅ 处理旧格式文件: {json_file.name}")

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
                    "effective_date": data.get("effective_date", "") or "",
                    "publication_date": data.get("publication_date", "") or "",
                    "publisher": data.get("publisher", "") or "",
                    "source_file": str(json_file),
                    "chunk_index": i,
                    "article_number": article_num,
                    "part_title": chunk.get("part_title", "")
                }
                yield chunk["text"], metadata, doc_id
        elif full_text:
            for i, chunk in enumerate(chunk_text(full_text)):
                doc_id = f"{law_id}_chunk_{i}"
                metadata = {
                    "law_id": law_id,
                    "jurisdiction": data.get("jurisdiction", ""),
                    "title": title,
                    "law_number": data.get("law_number", "") or "",
                    "passing_date": data.get("passing_date", "") or "",
                    "effective_date": data.get("effective_date", "") or "",
                    "publication_date": data.get("publication_date", "") or "",
                    "publisher": data.get("publisher", "") or "",
                    "source_file": str(json_file),
                    "chunk_index": i
                }
                yield chunk, metadata, doc_id


def load_json_files(data_dir_str: str, repo, skip_existing: bool = True) -> Iterator[Tuple[str, Dict, str]]:
    data_dir = Path(data_dir_str)
    json_files = []

    for jurisdiction_dir in data_dir.iterdir():
        if not jurisdiction_dir.is_dir():
            continue
        if jurisdiction_dir.name in ["chroma_db", ".chroma", "__pycache__"]:
            continue
        for json_file in jurisdiction_dir.glob("*.json"):
            json_files.append(json_file)

    print(f"找到 {len(json_files)} 个 JSON 文件待处理:")
    for f in json_files:
        print(f"   - {f.name}")

    for json_file in json_files:
        yield from process_json_file(json_file, repo, skip_existing)


def ingest_data(batch_size: int = 32, clear_index: bool = False, force: bool = False):
    """
    入库数据到Elasticsearch

    参数:
        batch_size: 批量处理大小
        clear_index: 是否清空现有索引
        force: 是否强制覆盖已有数据（忽略去重检查）
    """
    print("开始加载数据...")

    repo = ElasticsearchRepository()
    catalog = LawCatalog()

    # 清空现有索引
    if clear_index:
        print("清空现有索引...")
        repo.clear()
        
        # 同时重建空的 catalog（保留结构）
        print("重建 law_catalog.json...")
        initial_data = {
            "version": "1.0",
            "last_updated": "",
            "total_laws": 0,
            "laws": []
        }
        from utils.law_catalog import json as json_module
        from datetime import datetime as dt
        initial_data["last_updated"] = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
        with open(catalog.catalog_path, "w", encoding="utf-8") as f:
            json_module.dump(initial_data, f, ensure_ascii=False, indent=2)
        print("索引和 catalog 已清空")

    # 统计信息
    total_chunks = 0
    skipped_count = 0
    processed_laws = set()  # 已处理的law_id集合
    
    # 用于收集每个法律的 chunks 信息（用于更新 catalog）
    law_chunks_map: Dict[str, List[Dict]] = {}

    documents_batch: List[str] = []
    metadatas_batch: List[Dict] = []
    ids_batch: List[str] = []

    for doc, meta, doc_id in load_json_files(settings.data_dir, repo, skip_existing=not force):
        # 跳过标记（由process_json_file内部处理）
        if doc is None and meta is None and doc_id is None:
            skipped_count += 1
            continue

        documents_batch.append(doc)
        metadatas_batch.append(meta)
        ids_batch.append(doc_id)

        # 记录已处理的law_id
        law_id = meta.get('law_id', '')
        if law_id:
            processed_laws.add(law_id)
            
            # 收集 chunk 信息用于 catalog 更新
            if law_id not in law_chunks_map:
                law_chunks_map[law_id] = []
            law_chunks_map[law_id].append({
                "article_number": meta.get('article_number', ''),
                "metadata": meta,
                "_meta": {
                    "law_id": law_id,
                    "law_name": meta.get('title', ''),
                    "jurisdiction": meta.get('jurisdiction', ''),
                    "passing_date": meta.get('passing_date', ''),
                    "source_file": meta.get('source_file', ''),
                }
            })

        if len(documents_batch) >= batch_size:
            print(f"处理第 {total_chunks + 1} - {total_chunks + len(documents_batch)} 个文档块...")
            repo.add_documents(documents_batch, metadatas_batch, ids_batch)
            total_chunks += len(documents_batch)
            documents_batch = []
            metadatas_batch = []
            ids_batch = []

    if documents_batch:
        print(f"处理剩余 {total_chunks + 1} - {total_chunks + len(documents_batch)} 个文档块...")
        repo.add_documents(documents_batch, metadatas_batch, ids_batch)
        total_chunks += len(documents_batch)

    # 更新 law_catalog.json
    print("\n📋 更新法律目录 (law_catalog.json)...")
    catalog_updated_count = 0
    for law_id, chunks_data in law_chunks_map.items():
        try:
            catalog.update_from_chunks(chunks_data)
            catalog_updated_count += 1
        except Exception as e:
            print(f"⚠️  更新 {law_id} 的 catalog 记录失败: {e}")

    # 输出统计结果
    print("\n" + "=" * 60)
    print("📊 入库完成统计")
    print("=" * 60)
    print(f"✅ 成功入库文档块数: {total_chunks}")
    if skipped_count > 0:
        print(f"⏭️  跳过已存在法规数: {skipped_count}")
    print(f"📚 新增/更新法规数量: {len(processed_laws)}")
    print(f"📝 Catalog 更新记录数: {catalog_updated_count}")
    
    if processed_laws:
        print(f"\n已入库的法规ID:")
        for law_id in sorted(processed_laws):
            count = repo.get_law_chunk_count(law_id)
            # 从 catalog 获取更详细的信息
            cat_record = catalog.find_by_law_id(law_id)
            title = cat_record.get('law_name_zh', '') if cat_record else ''
            jur = cat_record.get('jurisdiction', '') if cat_record else ''
            print(f"   • {law_id} | {title} ({jur}) | {count} 个文档块")

    # 显示 catalog 统计信息
    stats = catalog.get_statistics()
    print(f"\n📊 Catalog 统计:")
    print(f"   总法规数: {stats['total_laws']}")
    print(f"   总文档块数: {stats['total_chunks']}")
    print(f"   总条款数: {stats['total_articles']}")
    if stats['by_jurisdiction']:
        print(f"   按法域分布:")
        for jur, count in sorted(stats['by_jurisdiction'].items()):
            print(f"      • {jur}: {count} 部法律")

    count = repo.count()
    print(f"\n🎉 完成！向量数据库中共有 {count} 个文档块")
    print(f"📁 法律目录已保存至: {catalog.catalog_path}")

    # 为新增的法规批量生成摘要
    if processed_laws:
        print(f"\n📝 开始为 {len(processed_laws)} 个法规生成摘要...")
        abstract_service = AbstractService()
        abstract_service.generate_for_laws(list(processed_laws), delay=2.0)

    # 自动重新标注旧数据
    print(f"\n🔄 开始重新标注所有数据...")
    reannotate_all_data(repo)


def reannotate_all_data(repo: ElasticsearchRepository):
    """重新标注所有数据"""
    print("初始化主题分类器...")
    classifier = TopicClassifier()
    print(f"加载词表版本: {classifier.taxonomy_version}, 主题数: {len(classifier.topics)}")
    
    total_docs = repo.count()
    print(f"开始处理 {total_docs} 条数据...")
    
    # 批量读取所有文档
    batch_size = 100
    processed = 0
    labeled = 0
    
    # 使用 scroll 游标遍历全部文档, 避免 from+size 超过 10000 限制
    results = repo.client.search(
        index=settings.es_index_name,
        body={"query": {"match_all": {}}, "size": batch_size},
        scroll="2m"
    )
    scroll_id = results.get("_scroll_id")

    try:
        while True:
            hits = results["hits"]["hits"]
            if not hits:
                break

            # 批量更新这批文档
            bulk_actions = []
            for hit in hits:
                doc_id = hit["_id"]
                source = hit["_source"]

                # 进行主题分类
                text = source.get("content", "")
                topics_info = classifier.classify_document(text)

                # 构建更新操作
                update_action = {
                    "_op_type": "update",
                    "_index": settings.es_index_name,
                    "_id": doc_id,
                    "doc": {
                        "topics": topics_info["topics"],
                        "topic_labels": topics_info["topic_labels"],
                        "topics_details": topics_info["topics_details"],
                        "taxonomy_version": topics_info["taxonomy_version"]
                    }
                }
                bulk_actions.append(update_action)

                # 统计信息
                if topics_info["topics"]:
                    labeled += 1

            # 执行批量更新
            if bulk_actions:
                from elasticsearch.helpers import bulk
                bulk(repo.client, bulk_actions)
                repo.client.indices.refresh(index=settings.es_index_name)

            processed += len(hits)
            print(f"  已处理 {processed}/{total_docs} 条数据...")

            # 获取下一批
            results = repo.client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = results.get("_scroll_id")
    finally:
        if scroll_id:
            repo.client.clear_scroll(scroll_id=scroll_id)
    
    print(f"重新标注完成: {processed} 条数据, 有主题标签: {labeled} 条")
    
    # 显示主题分布
    aggs = {
        "size": 0,
        "aggs": {
            "by_topic": {
                "terms": {"field": "topics", "size": 20}
            }
        }
    }
    resp = repo.client.search(index=settings.es_index_name, body=aggs)
    print("\n主题分布 TOP20:")
    for bucket in resp['aggregations']['by_topic']['buckets']:
        print(f"  {bucket['key']}: {bucket['doc_count']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="法规数据入库脚本")
    parser.add_argument("--clear", action="store_true", help="清空现有索引后重新入库")
    parser.add_argument("--force", action="store_true", help="强制覆盖已有数据（不跳过重复）")
    parser.add_argument("--batch-size", type=int, default=32, help="批量处理大小")

    args = parser.parse_args()
    ingest_data(batch_size=args.batch_size, clear_index=args.clear, force=args.force)
