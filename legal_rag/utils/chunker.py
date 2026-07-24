from typing import List, Dict, Optional, Union
from core.config import settings


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """纯文本切片"""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        if end < text_length:
            chunk = chunk[:-chunk_overlap]

        chunks.append(chunk)
        start = end - chunk_overlap

    return chunks


def _resplit_long_text(text: str, chunk_size: int = None, chunk_overlap: int = None,
                       min_chunk_size: int = 300) -> List[str]:
    """
    对超长文本进行二次切分，避免单条 embedding 序列过长。
    文本长度未超过 chunk_size 时直接返回；切分后若末尾 chunk 过短则合并到前一段。
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if len(text) <= chunk_size:
        return [text]

    raw_chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not raw_chunks:
        return [text]

    # 合并过短的末尾 chunk，避免产生无意义的小碎片
    if len(raw_chunks) >= 2 and len(raw_chunks[-1]) < min_chunk_size:
        merged = raw_chunks[-2] + '\n' + raw_chunks[-1]
        raw_chunks[-2] = merged
        raw_chunks = raw_chunks[:-1]

    return raw_chunks


def process_pre_chunked_data(data_list: List[Dict]) -> List[Dict]:
    """
    处理已经预分块的数据（cleaned格式）
    输入: 数组格式的条款列表
    输出: 标准化的chunks列表
    """
    chunks = []

    for item in data_list:
        # 提取文本内容（优先中文）
        text = item.get('text_zh', '') or item.get('text_pt', '') or item.get('text_en', '')

        if not text:
            continue

        # 提取 topics 信息（用于主题筛选）
        # 优先从 topics_details 获取（含中文 label_zh），其次从 topics（字符串ID）映射
        raw_topics_details = item.get('topics_details', [])
        raw_topics = item.get('topics', [])
        topics = []

        # 优先使用 topics_details（已有中文标签）
        if isinstance(raw_topics_details, list) and raw_topics_details:
            for t in raw_topics_details:
                if isinstance(t, dict):
                    topics.append({
                        'id': t.get('id', ''),
                        'label_zh': t.get('label_zh', ''),
                    })
        # 回退：topics 是字符串数组，需要映射为中文标签
        elif isinstance(raw_topics, list) and raw_topics:
            from repositories.elasticsearch import _to_label_zh
            for t in raw_topics:
                if isinstance(t, dict):
                    topics.append({
                        'id': t.get('id', ''),
                        'label_zh': t.get('label_zh', ''),
                    })
                elif isinstance(t, str):
                    topics.append({'id': t, 'label_zh': _to_label_zh(t)})

        # 对超长文本二次切分，生成 sub_chunk
        parent_doc_id = item.get('doc_id')
        sub_texts = _resplit_long_text(text)

        for sub_idx, sub_text in enumerate(sub_texts):
            # 每个 sub_chunk 使用独立 doc_id，确保 ES 文档唯一
            sub_doc_id = f"{parent_doc_id}_sub_{sub_idx}" if parent_doc_id else None

            chunk = {
                'text': sub_text,
                'article_number': item.get('article_number', ''),
                'part_title': item.get('chapter', ''),  # chapter对应原来的part_title
                # 额外保留原始元数据
                '_meta': {
                    'doc_id': sub_doc_id,
                    'parent_doc_id': parent_doc_id,
                    'sub_chunk_index': sub_idx,
                    'sub_chunk_total': len(sub_texts),
                    'article_title': item.get('article_title'),
                    'chunk_index': item.get('chunk_index'),
                    'law_name': item.get('law_name'),
                    'passing_date': item.get('passing_date'),
                    'effective_date': item.get('effective_date'),
                    'publication_date': item.get('publication_date'),
                    'source_file': item.get('source_file')
                },
                # 主题信息（用于筛选）
                'topics': topics,
                'topic_labels': [t['label_zh'] for t in topics if t.get('label_zh')]
            }

            chunks.append(chunk)

    return chunks


def chunk_from_structure(data: Union[Dict, List]) -> List[Dict]:
    """
    从结构化数据提取chunks - 支持多种格式

    支持的格式:
    1. 新格式: 数组 [ {...}, {...} ] (cleaned文件)
    2. 旧格式1: data['structure']['parts'][...]['chapters'][...]['articles'][...]
    3. 旧格式2: data['chapters'][...]['articles'][...]
    4. 旧格式3: data['articles'][...]
    """

    # 格式0: 新清洗数据的数组格式 [ {...}, {...} ]
    if isinstance(data, list):
        return process_pre_chunked_data(data)

    chunks = []
    structure = data.get('structure', {})

    # 格式1: data['structure']['parts'][...]['articles'][...]
    if 'parts' in structure:
        for part in structure['parts']:
            part_title = part.get('title', '')
            if 'chapters' in part:
                for chapter in part['chapters']:
                    chapter_title = chapter.get('title', '')
                    if 'articles' in chapter:
                        for article in chapter['articles']:
                            article_num = article.get('article_number', '')
                            article_text = article.get('content', '')  # 注意：content 不是 text

                            if article_text:
                                chunks.append({
                                    'text': article_text,
                                    'article_number': article_num,
                                    'part_title': f"{part_title} - {chapter_title}" if chapter_title else part_title
                                })
            elif 'articles' in part:
                for article in part['articles']:
                    article_num = article.get('article_number', '')
                    article_text = article.get('content', '')

                    if article_text:
                        chunks.append({
                            'text': article_text,
                            'article_number': article_num,
                            'part_title': part_title
                        })

    # 格式2: data['chapters'][...]['articles'][...]
    elif 'chapters' in data:
        for chapter in data['chapters']:
            chapter_title = chapter.get('title', '')

            if 'articles' in chapter:
                for article in chapter['articles']:
                    article_num = article.get('article_number', '')
                    article_text = article.get('text', '')

                    if article_text:
                        chunks.append({
                            'text': article_text,
                            'article_number': article_num,
                            'part_title': chapter_title
                        })

    # 格式3: data['articles'][...]
    elif 'articles' in data:
        for article in data['articles']:
            article_num = article.get('article_number', '')
            article_text = article.get('text', '')

            if article_text:
                chunks.append({
                    'text': article_text,
                    'article_number': article_num,
                    'part_title': ''
                })

    return chunks