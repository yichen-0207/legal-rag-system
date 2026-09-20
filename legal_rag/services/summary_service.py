from typing import Dict, Optional, List
import json
from datetime import datetime
from openai import OpenAI
from core.config import settings
from repositories.elasticsearch import ElasticsearchRepository
from services.summary_cache import summary_cache

class SummaryService:
    """法规摘要服务 - 增强版"""

    def __init__(self):
        self.repo = ElasticsearchRepository()
        # LLM 上下文窗口限制（字符数）
        self.MAX_CONTEXT_LENGTH = 15000
        # 智能压缩后保留的长度
        self.COMPRESSED_LENGTH = 10000

    def _get_db_abstract_with_metadata(self, law_id: str) -> Optional[Dict]:
        """从数据库获取摘要内容及元数据（单次查询）"""
        query = {
            "query": {"term": {"law_id": law_id}},
            "_source": ["abstract", "title", "jurisdiction", "passing_date", "article_number"],
            "size": 100
        }
        try:
            resp = self.repo.client.search(index=settings.es_index_name, body=query)
            hits = resp.get("hits", {}).get("hits", [])
            if not hits:
                return None
            
            # 获取第一条的摘要和元数据
            first_hit = hits[0].get("_source", {})
            abstract = first_hit.get("abstract")
            if not isinstance(abstract, dict) or not abstract.get("text"):
                return None
            
            # 提取文章数量
            article_numbers = set()
            for hit in hits:
                art_num = hit.get("_source", {}).get("article_number", "")
                if art_num:
                    article_numbers.add(art_num)
            
            return {
                "abstract_text": abstract["text"],
                "title": first_hit.get("title", ""),
                "jurisdiction": first_hit.get("jurisdiction", ""),
                "passing_date": first_hit.get("passing_date", ""),
                "article_count": len(article_numbers)
            }
        except Exception:
            return None

    def generate_summary_with_cache(self, law_id: str) -> Dict:
        """
        生成法规摘要（带缓存，优先使用数据库已有摘要）

        流程：
        1. 先从内存缓存获取摘要
        2. 如果内存缓存不存在，检查数据库中是否有 abstract 字段
        3. 如果数据库已有摘要，构建结构化摘要并缓存到内存
        4. 如果数据库也没有，调用 LLM 生成新摘要
        5. 返回摘要

        Args:
            law_id: 法规ID

        Returns:
            结构化摘要数据
        """
        # 尝试从内存缓存获取
        cached_summary = summary_cache.get(law_id)
        if cached_summary:
            return cached_summary

        # 检查数据库中是否已有摘要（单次查询获取摘要+元数据）
        db_data = self._get_db_abstract_with_metadata(law_id)
        if db_data:
            abstract_text = db_data["abstract_text"]
            
            # 将摘要文本拆分为purpose和scope两部分
            sentences = abstract_text.split("。")
            purpose_part = ""
            scope_part = ""
            
            if len(sentences) >= 2:
                # 前半部分作为立法目的
                mid = len(sentences) // 2
                purpose_part = "。".join(sentences[:mid]) + "。"
                scope_part = "。".join(sentences[mid:]) + "。"
            else:
                purpose_part = abstract_text
                scope_part = abstract_text
            
            # 构建结构化摘要
            summary = {
                "basic_info": {
                    "law_name": db_data.get("title", ""),
                    "jurisdiction": db_data.get("jurisdiction", ""),
                    "law_type": "待确定",
                    "status": "现行有效",
                    "passing_date": db_data.get("passing_date", ""),
                    "article_count": db_data.get("article_count", 0)
                },
                "purpose": purpose_part.strip(),
                "scope": scope_part.strip(),
                "compliance_points": {
                    "core_obligations": [],
                    "obligation_count": 0,
                    "penalties": "",
                    "key_requirements": []
                },
                "chapters": [],
                "keywords": [],
                "from_cache": True,
                "is_db_abstract": True
            }
            
            # 存入内存缓存
            summary_cache.set(law_id, summary)
            return summary

        # 数据库也没有，生成新摘要
        summary = self.generate_summary(law_id)

        # 存入缓存（仅在生成成功时）
        if summary and summary.get("purpose") != "解析失败":
            summary_cache.set(law_id, summary)

        return summary

    def _build_enhanced_summary_prompt(self, full_text: str, metadata: Dict) -> str:
        """构建增强版摘要生成的 Prompt - 单次调用提取所有信息"""
        jurisdiction = metadata.get("jurisdiction", "")
        passing_date = metadata.get("passing_date", "")
        publisher = metadata.get("publisher", "")
        law_number = metadata.get("law_number", "")

        return f"""你是一位资深的法律文本分析专家，擅长从法律文本中提炼关键信息。请根据以下法律全文和元数据，生成一份全面的结构化摘要。

## 法律元数据
- 法域：{jurisdiction}
- 通过日期：{passing_date}
- 发布机构：{publisher}
- 法律编号：{law_number}

## 要求
请严格按照以下 JSON 格式返回（不要输出其他内容），每个字段用中文撰写：

{{
  "basic_info": {{
    "law_name": "法规全称",
    "jurisdiction": "{jurisdiction}",
    "law_type": "法律类型（如：法律/行政法规/部门规章/地方性法规等）",
    "status": "现行状态（如：现行有效/已废止/已修订等）",
    "passing_date": "{passing_date}",
    "article_count": 条款总数（数字）
  }},
  "purpose": "立法目的（2-3句话，说明制定本法的核心目标和价值）",
  "scope": "适用范围（2-3句话，明确适用的主体、地域、对象等）",
  "compliance_points": {{
    "core_obligations": ["核心义务1：简要描述", "核心义务2：简要描述", ...],
    "obligation_count": 核心义务数量,
    "penalties": "违规后果概述（包括行政处罚、刑事责任、罚款范围等）",
    "key_deadlines": ["时限1：如24小时内报告", "时限2：如年度评估", ...]
  }},
  "legal_effect": {{
    "law_type_detail": "法律层级详细说明（如：最高层级法律/上位法等）",
    "current_status": "当前效力状态",
    "years_in_effect": "生效至今约X年（估算）",
    "hierarchy_position": "在法律体系中的位置"
  }},
  "chapters": [
    {{"title": "章节名", "articles": "第X-Y条", "summary": "一句话概括该章核心内容", "key_articles": ["第X条：核心条款简述"]}}
  ],
  "keywords": {{
    "core": ["核心概念词1", "核心概念词2", ...],
    "subjects": ["主体/机构名1", "主体/机构名2", ...],
    "responsibilities": ["责任/义务相关词1", "责任/义务相关词2", ...]
  }}
}}

## 分析重点提示
1. **企业合规要点**：重点识别企业需要履行的具体义务、不合规的法律后果、以及关键的时间节点要求
2. **关键词分类**：
   - core：本法的核心概念和专业术语
   - subjects：涉及的主要监管主体、运营者、机构名称
   - responsibilities：关于责任、处罚、义务的相关词汇
3. **主要章节**：识别每章的核心条款，特别是定义条款、义务条款、处罚条款

## 法律全文如下：
{full_text}"""

    def _compress_text(self, chunks: List[Dict]) -> str:
        """智能压缩文本，保留核心内容 - 优化版"""
        compressed_chunks = []

        for chunk in chunks:
            content = chunk.get("content", "")
            article_number = chunk.get("article_number", "")

            # 保留每条的前两句（比原来更多，确保信息完整）
            sentences = [s.strip() for s in content.split('。') if s.strip()]
            if sentences:
                # 取前两句或全部（如果少于2句）
                selected_sentences = sentences[:min(2, len(sentences))]
                first_part = '。'.join(selected_sentences)
                if not first_part.endswith('。'):
                    first_part += '。'

                # 只取条款文本，去掉元数据噪音
                if article_number:
                    compressed_chunks.append(f"{article_number} {first_part}")
                else:
                    compressed_chunks.append(first_part)

        return "\n".join(compressed_chunks)

    def _get_full_text(self, law_id: str) -> tuple:
        """获取法规的完整文本和元数据"""
        law = self.repo.get_law_by_id(law_id)
        if not law:
            raise ValueError("法规不存在")

        chunks = law.get("chunks", [])
        # 按 chunk_index 排序
        chunks.sort(key=lambda x: x.get("chunk_index", 0))

        # 拼接完整文本
        full_text = ""
        for chunk in chunks:
            article_number = chunk.get("article_number", "")
            content = chunk.get("content", "")
            if article_number:
                full_text += f"{article_number} {content}\n\n"
            else:
                full_text += f"{content}\n\n"

        # 提取元数据
        metadata = {
            "jurisdiction": law.get("jurisdiction", ""),
            "passing_date": law.get("passing_date", ""),
            "publisher": law.get("publisher", ""),
            "law_number": law.get("law_number", ""),
            "title": law.get("title", "")
        }

        return full_text.strip(), chunks, metadata

    def _call_llm(self, prompt: str) -> str:
        """调用远程 LLM API（带超时和重试）"""
        base_url = settings.llm_api_base_url or "https://api.deepseek.com/v1"
        api_key = settings.llm_api_key or ""
        model = settings.llm_model or "deepseek-flash"

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=100.0,  # 增加客户端超时到100秒
            max_retries=2   # 自动重试2次
        )

        create_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": settings.max_tokens,
            "stream": False
        }
        extra_body = settings.llm_extra_body()
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        response = client.chat.completions.create(**create_kwargs)
        return response.choices[0].message.content.strip()

    def _calculate_years_in_effect(self, passing_date: str) -> str:
        """计算法规生效至今的年数"""
        if not passing_date:
            return "未知"
        try:
            # 支持多种日期格式
            for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y年%m月%d日"]:
                try:
                    date_obj = datetime.strptime(passing_date.split('T')[0], fmt)
                    years = (datetime.now() - date_obj).days / 365.25
                    if years >= 1:
                        return f"约{years:.1f}年"
                    else:
                        months = years * 12
                        return f"约{months:.0f}个月"
                except ValueError:
                    continue
            return "未知"
        except Exception:
            return "未知"

    def _extract_article_count(self, chunks: List[Dict]) -> int:
        """从chunks中提取文章数量"""
        article_numbers = set()
        for chunk in chunks:
            art_num = chunk.get("article_number", "")
            if art_num:
                article_numbers.add(art_num)
        return len(article_numbers)

    def generate_summary(self, law_id: str) -> Dict:
        """
        生成法规的结构化摘要 - 增强版

        流程：
        1. 从 ES 获取该法律的所有 chunk 和元数据
        2. 按chunk_index排序拼接成完整原文
        3. 如果原文太长，做智能压缩
        4. 将原文+元数据+增强模板指令发给LLM（单次调用）
        5. 解析并丰富返回的结构化摘要
        """
        # 获取完整文本和元数据
        full_text, chunks, metadata = self._get_full_text(law_id)

        # 提取基础统计信息
        article_count = self._extract_article_count(chunks)
        years_in_effect = self._calculate_years_in_effect(metadata.get("passing_date", ""))

        # 检查文本长度，决定是否需要压缩
        if len(full_text) > self.MAX_CONTEXT_LENGTH:
            full_text = self._compress_text(chunks)
            # 如果压缩后仍然太长，进一步截断
            if len(full_text) > self.COMPRESSED_LENGTH:
                full_text = full_text[:self.COMPRESSED_LENGTH]

        # 构建 Prompt（包含元数据）
        prompt = self._build_enhanced_summary_prompt(full_text, metadata)

        # 调用 LLM（单次调用获取所有信息）
        response = self._call_llm(prompt)

        # 解析 JSON 响应
        try:
            summary = json.loads(response.strip())

            # 补充和修正基础信息
            if "basic_info" in summary:
                basic_info = summary["basic_info"]
                # 使用ES数据的准确值覆盖LLM可能的估算
                if not basic_info.get("law_name") or basic_info["law_name"] == "法规全称":
                    basic_info["law_name"] = metadata.get("title", "")
                basic_info["jurisdiction"] = metadata.get("jurisdiction", basic_info.get("jurisdiction", ""))
                basic_info["passing_date"] = metadata.get("passing_date", basic_info.get("passing_date", ""))
                # 使用实际计算的文章数
                if not basic_info.get("article_count") or basic_info["article_count"] == "条款总数（数字）":
                    basic_info["article_count"] = article_count
            else:
                # 如果LLM没有返回basic_info结构，创建一个
                summary["basic_info"] = {
                    "law_name": metadata.get("title", ""),
                    "jurisdiction": metadata.get("jurisdiction", ""),
                    "law_type": summary.get("law_type", "待确定"),
                    "status": "现行有效",
                    "passing_date": metadata.get("passing_date", ""),
                    "article_count": article_count
                }

            # 补充法律效力信息
            if "legal_effect" in summary:
                legal_effect = summary["legal_effect"]
                # 使用计算出的准确年限（安全处理各种类型）
                years_val = legal_effect.get("years_in_effect")
                years_str = str(years_val) if years_val is not None else ""
                if not years_str or "X年" in years_str or years_str.isdigit():
                    legal_effect["years_in_effect"] = years_in_effect

                # 确保所有字段都是字符串类型（防止类型错误）
                for key in ['law_type_detail', 'current_status', 'hierarchy_position']:
                    val = legal_effect.get(key)
                    if val is not None and not isinstance(val, str):
                        legal_effect[key] = str(val)
            else:
                summary["legal_effect"] = {
                    "law_type_detail": "待确定",
                    "current_status": "现行有效",
                    "years_in_effect": years_in_effect,
                    "hierarchy_position": "待确定"
                }

            # 移除核心要点字段（用户要求去除）
            if "core_points" in summary:
                del summary["core_points"]

            # 安全处理 basic_info 中的字段类型
            if "basic_info" in summary:
                basic_info = summary["basic_info"]
                for key in ['law_name', 'jurisdiction', 'law_type', 'status', 'passing_date']:
                    val = basic_info.get(key)
                    if val is not None and not isinstance(val, str):
                        basic_info[key] = str(val)

                # article_count 确保是整数
                ac = basic_info.get("article_count")
                if isinstance(ac, str) and ac.isdigit():
                    basic_info["article_count"] = int(ac)
                elif not isinstance(ac, int):
                    basic_info["article_count"] = article_count

            return summary

        except json.JSONDecodeError:
            # 如果解析失败，返回包含原始响应的错误信息和基础元数据
            return {
                "basic_info": {
                    "law_name": metadata.get("title", ""),
                    "jurisdiction": metadata.get("jurisdiction", ""),
                    "law_type": "待确定",
                    "status": "解析失败",
                    "passing_date": metadata.get("passing_date", ""),
                    "article_count": article_count
                },
                "purpose": "解析失败",
                "scope": "解析失败",
                "compliance_points": {
                    "core_obligations": ["解析失败"],
                    "obligation_count": 0,
                    "penalties": "解析失败",
                    "key_deadlines": []
                },
                "legal_effect": {
                    "law_type_detail": "解析失败",
                    "current_status": "解析失败",
                    "years_in_effect": years_in_effect,
                    "hierarchy_position": "解析失败"
                },
                "chapters": [],
                "keywords": {
                    "core": [],
                    "subjects": [],
                    "responsibilities": []
                },
                "raw_response": response.strip()
            }