import re
from typing import List, Dict, Optional, Union
from core.config import settings


# 滑窗切分的优先落刀位置（句读/换行）；长条款结构化切分另用编号行作边界
_BOUNDARY_CHARS = "。！？；;.!?\n"
_STRUCT_LINE_RE = re.compile(
    r"^\s*(?:[（(]\s*[0-9a-zA-Z]{1,4}\s*[)）]|[0-9]{1,3}\s*[\.、)]|"
    r"第[〇零一二三四五六七八九十百千0-9]+[条節节款项])"
)


def _find_boundary_cut(text: str, start: int, hard_end: int) -> int:
    """在窗口后半段自后向前找句读/换行作为切点（切点字符归属前一块）。"""
    floor = start + (hard_end - start) // 2
    for i in range(hard_end - 1, floor - 1, -1):
        if text[i] in _BOUNDARY_CHARS:
            return i + 1
    return hard_end


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """真滑动窗口切片：相邻窗口实际重叠 chunk_overlap 个字符，切点优先落在句读/换行边界。"""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    if chunk_size <= 0:
        return [text] if text else []
    # 重叠不超过窗口一半，避免步长过小导致块数爆炸
    chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))

    text_length = len(text)
    chunks: List[str] = []
    start = 0
    while start < text_length:
        hard_end = min(start + chunk_size, text_length)
        if hard_end >= text_length:
            chunks.append(text[start:])
            break
        cut = _find_boundary_cut(text, start, hard_end)
        if cut <= start + chunk_overlap:  # 保证每次必然前进
            cut = hard_end
        chunks.append(text[start:cut])
        start = cut - chunk_overlap
    return chunks


def _split_into_blocks(text: str) -> List[str]:
    """按「段落 + 编号行」切块，避免把一条编号规定从中间劈开。"""
    blocks: List[str] = []
    buf: List[str] = []

    def flush():
        if buf:
            blocks.append("\n".join(buf))
            buf.clear()

    for line in text.split("\n"):
        s = line.strip()
        if not s:
            flush()
            continue
        if _STRUCT_LINE_RE.match(s) and buf:
            flush()
        buf.append(line.rstrip())
    flush()
    return blocks


def _pack_blocks(blocks: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """把结构块贪心装包成不超过 chunk_size 的窗口；跨窗口时保留上块尾部做重叠。"""
    chunks: List[str] = []
    cur = ""
    for blk in blocks:
        if len(blk) > chunk_size:
            # 单个结构块本身超窗（如整段无编号的长段落）：退回真滑窗切分
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(chunk_text(blk, chunk_size, chunk_overlap))
            continue
        candidate = blk if not cur else f"{cur}\n{blk}"
        if len(candidate) <= chunk_size:
            cur = candidate
            continue
        chunks.append(cur)
        tail = cur[-chunk_overlap:] if (chunk_overlap > 0 and len(cur) > chunk_overlap) else ""
        if tail and len(tail) + 1 + len(blk) <= chunk_size:
            cur = f"{tail}\n{blk}"
        else:
            cur = blk
    if cur:
        chunks.append(cur)
    return chunks


def _resplit_long_text(text: str, chunk_size: int = None, chunk_overlap: int = None,
                       min_chunk_size: int = 300) -> List[str]:
    """超长条款二次切分：先做结构化切分，超窗段退回滑窗；末尾过短并入前块。"""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if len(text) <= chunk_size:
        return [text]

    blocks = _split_into_blocks(text)
    raw_chunks = _pack_blocks(blocks, chunk_size, chunk_overlap)
    if not raw_chunks:
        return [text]

    # 合并过短的末尾 chunk，避免产生无意义的小碎片
    if len(raw_chunks) >= 2 and len(raw_chunks[-1]) < min_chunk_size:
        merged = raw_chunks[-2] + '\n' + raw_chunks[-1]
        raw_chunks[-2] = merged
        raw_chunks = raw_chunks[:-1]

    return raw_chunks


def _build_embed_prefix(item: Dict) -> str:
    """子块 embedding 标题前缀（仅进向量化输入，不写入 content）。"""
    parts = [
        str(item.get("law_name") or "").strip(),
        str(item.get("article_number") or "").strip(),
        str(item.get("article_title") or "").strip(),
    ]
    parts = [p for p in parts if p]
    if not parts:
        return ""
    return ("【" + " ".join(parts) + "】")[:100]


def _compose_part_title(chapter: str, chapter_title: str) -> str:
    """合并「第一章」与「总则」，过滤把正文首句误当章节名的噪声。"""
    chapter = str(chapter or "").strip()
    title = str(chapter_title or "").strip()

    if title:
        # 源数据常见「总 则」，去掉中文字符之间的空格
        title = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", title)
        title = re.sub(r"\s+", " ", title)
        if len(title) > 24 or title[0].islower():
            title = ""

    return f"{chapter} {title}".strip() if title else chapter


def process_pre_chunked_data(data_list: List[Dict]) -> List[Dict]:
    """把预分块的 cleaned 条款列表标准化为 chunks。"""
    chunks = []

    for item in data_list:
        # 提取文本内容（优先中文）
        text = item.get('text_zh', '') or item.get('text_pt', '') or item.get('text_en', '')

        if not text:
            continue

        # 主题信息：优先 topics_details（含中文标签），其次 topics 字符串 ID 映射
        raw_topics_details = item.get('topics_details', [])
        raw_topics = item.get('topics', [])
        topics = []

        if isinstance(raw_topics_details, list) and raw_topics_details:
            for t in raw_topics_details:
                if isinstance(t, dict):
                    topics.append({
                        'id': t.get('id', ''),
                        'label_zh': t.get('label_zh', ''),
                    })
        # 回退：字符串 ID 映射为中文标签
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

        # 剔除已废弃的粗粒度主题，避免重新入库后再次出现在看板
        from repositories.elasticsearch import is_excluded_topic
        topics = [t for t in topics
                  if not (is_excluded_topic(t.get('id')) or is_excluded_topic(t.get('label_zh')))]

        # 对超长文本二次切分，生成 sub_chunk
        parent_doc_id = item.get('doc_id')
        sub_texts = _resplit_long_text(text)

        # 子块 embedding 标题前缀（只进向量化输入，不进 content）
        prefix = _build_embed_prefix(item)

        for sub_idx, sub_text in enumerate(sub_texts):
            # 每个 sub_chunk 使用独立 doc_id，确保 ES 文档唯一
            sub_doc_id = f"{parent_doc_id}_sub_{sub_idx}" if parent_doc_id else None

            chunk = {
                'text': sub_text,
                'article_number': item.get('article_number', ''),
                # 条目类型（substantive/schedule 等），供检索筛选与分区
                'chunk_type': item.get('chunk_type', ''),
                'part_title': _compose_part_title(item.get('chapter', ''), item.get('chapter_title', '')),
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
            if prefix:
                chunk['embed_text'] = f"{prefix}\n{sub_text}"

            chunks.append(chunk)

    return chunks


def chunk_from_structure(data: Union[Dict, List]) -> List[Dict]:
    """从结构化数据提取 chunks，兼容数组新格式与三种旧嵌套格式。"""

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