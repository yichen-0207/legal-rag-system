from typing import List, Dict, Optional
import time
from collections import defaultdict

from repositories.elasticsearch import ElasticsearchRepository
from services.qa_service import QAService


class TopicAnalysisService:
    """跨法域专题分析服务"""

    # 对比分析维度（固定框架，要求详细展开）
    ANALYSIS_DIMENSIONS = [
        "适用范围：哪些主体/行为受该条款约束？请分别列出两个法域的适用对象、管辖边界。",
        "核心权利/义务：法律赋予了什么权利或规定了什么义务？请逐条引用原文说明。",
        "例外情形：是否有豁免或例外规定？具体条件和范围是什么？",
        "法律责任/处罚：违反规定的后果是什么？罚款金额、监禁期限等具体处罚措施。",
        "程序性要求：是否有申报、审批、备案等程序要求？",
        "总体差异总结：用对比表格清晰呈现两个法域在上述各维度的核心异同点。"
    ]

    def __init__(self):
        self.repo = ElasticsearchRepository()
        self.qa_service = QAService()

    def _search_by_jurisdiction(
        self,
        query: str,
        jurisdiction: str,
        top_n: int = 8
    ) -> List[Dict]:
        """
        阶段一：对指定法域执行语义向量搜索，保证返回 top_n 条结果。

        分开搜索确保每个法域都有足够素材，不会被另一个法域"淹没"。
        """
        results = self.repo.query([query], n_results=top_n, where={"jurisdiction": jurisdiction})
        chunks = []

        if results.get('ids') and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                chunks.append({
                    "_id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "law_id": meta.get("law_id", ""),
                    "title": meta.get("title", ""),
                    "jurisdiction": meta.get("jurisdiction", jurisdiction),
                    "article_number": meta.get("article_number", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "similarity": round(1 / (1 + distance), 4)
                })

        return chunks

    def _aggregate_chunks_to_laws(self, chunks: List[Dict]) -> List[Dict]:
        """
        阶段二：将碎片化的 chunk 聚合为法规级别的结构化文本。

        按 law_id 分组 → 同一法规的 chunk 按 chunk_index 排序 → 拼接为完整条款上下文。
        """
        laws_map: Dict[str, Dict] = defaultdict(lambda: {
            "chunks": [],
            "article_numbers": set(),
            "title": "",
            "passing_date": ""
        })

        for chunk in chunks:
            lid = chunk["law_id"]
            laws_map[lid]["chunks"].append(chunk)
            laws_map[lid]["article_numbers"].add(chunk["article_number"])
            if not laws_map[lid]["title"]:
                laws_map[lid]["title"] = chunk["title"]

        aggregated = []
        for lid, info in laws_map.items():
            # 按 chunk_index 排序后拼接
            sorted_chunks = sorted(info["chunks"], key=lambda c: c["chunk_index"])

            # 构建结构化文本块
            parts = [f"## {info['title']}"]
            for ch in sorted_chunks:
                art_num = ch.get("article_number", "")
                content = ch.get("content", "").strip()
                sim = ch.get("similarity", 0)
                header = f"### {art_num}" if art_num else "### 条款"
                parts.append(f"{header}\n{content}\n*(匹配度: {sim:.2%})*")

            # 提取去重后的条款编号列表
            article_list = sorted(
                [a for a in info["article_numbers"] if a],
                key=lambda x: x.lstrip("第").rstrip("条")[:5] if x else ""
            )

            aggregated.append({
                "law_id": lid,
                "title": info['title'],
                "article_numbers": article_list,
                "chunk_count": len(sorted_chunks),
                "context_text": "\n\n".join(parts),
                "max_similarity": max(c["similarity"] for c in sorted_chunks) if sorted_chunks else 0
            })

        # 按最高相似度降序排列
        aggregated.sort(key=lambda x: x["max_similarity"], reverse=True)
        return aggregated

    def _build_analysis_prompt(
        self,
        query: str,
        jurisdiction_a_name: str,
        context_a: str,
        jurisdiction_b_name: str,
        context_b: str
    ) -> str:
        """
        阶段三：构建详细版 Prompt（要求深度分析）。
        """
        dimension_lines = "\n".join("   {}. {}".format(i + 1, dim) for i, dim in enumerate(self.ANALYSIS_DIMENSIONS))

        prompt = (
            "你是一位资深的比较法研究专家，拥有20年以上的跨境法律实务经验。\n"
            "你需要基于提供的法律条款原文，给出客观、准确、结构化的深度对比分析报告。\n"
            "\n"
            "**重要原则：**\n"
            "- 必须严格基于提供的法律条款原文进行分析和引用，不得编造\n"
            "- 每个分析维度都要分别对" + jurisdiction_a_name + "和" + jurisdiction_b_name + "进行独立阐述\n"
            "- 引用条款时请注明法规名称和条款编号\n"
            "- 如果某个法域在某方面没有明确规定，明确标注\"未找到相关规定\"\n"
            "- 分析要深入具体，不要泛泛而谈，尽量引用原文中的关键措辞和数据\n"
            "\n"
            "---\n"
            "\n"
            "**用户查询主题：** " + query + "\n"
            "\n"
            "**对比法域：** " + jurisdiction_a_name + " vs " + jurisdiction_b_name + "\n"
            "\n"
            "---\n"
            "\n"
            "## 分析框架（请逐项展开，每项不少于150字）\n"
            "\n" +
            dimension_lines +
            "\n"
            "---\n"
            "\n"
            "### " + jurisdiction_a_name + "相关法律条款原文\n"
            "\n" +
            context_a +
            "\n"
            "---\n"
            "\n"
            "### " + jurisdiction_b_name + "相关法律条款原文\n"
            "\n" +
            context_b +
            "\n"
            "---\n"
            "\n"
            "## 输出格式要求\n"
            "\n"
            "请按以下结构输出完整的对比分析报告：\n"
            "\n"
            "# " + query + " —— " + jurisdiction_a_name + "与" + jurisdiction_b_name + "法规对比分析报告\n"
            "\n"
            "## 一、适用范围对比\n"
            "（分别阐述两地的适用主体、管辖范围...）\n"
            "\n"
            "## 二、核心权利与义务对比\n"
            "（分别阐述两地的权利义务体系，引用具体条文...）\n"
            "\n"
            "## 三、例外情形对比\n"
            "（分别阐述两地的豁免条件...）\n"
            "\n"
            "## 四、法律责任与处罚对比\n"
            "（分别列举具体的处罚措施、金额、刑期...）\n"
            "\n"
            "## 五、程序性要求对比\n"
            "（如有申报、审批等程序要求，分别说明...）\n"
            "\n"
            "## 六、总体差异总结表格\n"
            "\n"
            "| 对比维度 | " + jurisdiction_a_name + " | " + jurisdiction_b_name + " |\n"
            "|---------|------|------|\n"
            "| 适用范围 | ... | ... |\n"
            "| 核心权利/义务 | ... | ... |\n"
            "| 例外情形 | ... | ... |\n"
            "| 法律责任/处罚 | ... | ... |\n"
            "| 程序要求 | ... | ... |\n"
            "\n"
            "## 七、实践建议\n"
            "（基于以上分析，给用户的实用建议）"
        )

        return prompt

    def analyze(
        self,
        query: str,
        jurisdiction_a: str,
        jurisdiction_b: str,
        top_n_per_jurisdiction: int = 8
    ) -> Dict:
        """
        执行完整的跨法域专题分析流程。

        Args:
            query: 用户输入的查询主题（如"个人数据跨境转移的规定"）
            jurisdiction_a: 法域A名称（如"澳门"）
            jurisdiction_b: 法域B名称（如"新加坡"）
            top_n_per_jurisdiction: 每个法域检索的条款数量

        Returns:
            包含检索结果聚合 + LLM 分析报告的完整响应
        """
        t_total = time.time()

        # ===== 阶段一：分别对两个法域进行语义向量搜索 =====
        t1 = time.time()
        chunks_a = self._search_by_jurisdiction(query, jurisdiction_a, top_n_per_jurisdiction)
        chunks_b = self._search_by_jurisdiction(query, jurisdiction_b, top_n_per_jurisdiction)
        t_search = time.time() - t1

        # ===== 阶段二：聚合为法规级别结构化文本 =====
        t2 = time.time()
        laws_a = self._aggregate_chunks_to_laws(chunks_a)
        laws_b = self._aggregate_chunks_to_laws(chunks_b)

        context_a = "\n\n---\n\n".join(law["context_text"] for law in laws_a)
        context_b = "\n\n---\n\n".join(law["context_text"] for law in laws_b)
        t_aggregate = time.time() - t2

        # ===== 阶段三：LLM 生成对比分析报告 =====
        t3 = time.time()
        prompt = self._build_analysis_prompt(query, jurisdiction_a, context_a, jurisdiction_b, context_b)
        analysis_report = self.qa_service._call_llm(prompt)
        t_llm = time.time() - t3

        total_time = time.time() - t_total

        print(
            f"【专题分析计时】总耗时: {total_time:.4f}s | "
            f"语义检索: {t_search:.4f}s | "
            f"结果聚合: {t_aggregate:.4f}s | "
            f"LLM生成: {t_llm:.4f}s"
        )

        return {
            "query": query,
            "jurisdiction_a": jurisdiction_a,
            "jurisdiction_b": jurisdiction_b,

            # 阶段一原始检索结果
            "retrieval": {
                "chunks_a_count": len(chunks_a),
                "chunks_b_count": len(chunks_b),
                "top_n_per_jurisdiction": top_n_per_jurisdiction
            },

            # 阶段二聚合后的法规级结果
            "aggregated": {
                "laws_a": [
                    {
                        "law_id": l["law_id"],
                        "title": l["title"],
                        "article_numbers": l["article_numbers"],
                        "chunk_count": l["chunk_count"],
                        "max_similarity": l["max_similarity"]
                    }
                    for l in laws_a
                ],
                "laws_b": [
                    {
                        "law_id": l["law_id"],
                        "title": l["title"],
                        "article_numbers": l["article_numbers"],
                        "chunk_count": l["chunk_count"],
                        "max_similarity": l["max_similarity"]
                    }
                    for l in laws_b
                ],
                "laws_a_count": len(laws_a),
                "laws_b_count": len(laws_b)
            },

            # 阶段三 LLM 生成的分析报告
            "analysis_report": analysis_report,

            "timing": {
                "semantic_search": f"{t_search:.4f}s",
                "aggregation": f"{t_aggregate:.4f}s",
                "llm_generation": f"{t_llm:.4f}s",
                "total": f"{total_time:.4f}s"
            }
        }

    def analyze_stream(self, query: str, jurisdiction_a: str, jurisdiction_b: str,
                       top_n_per_jurisdiction: int = 8):
        """
        流式版本：阶段一二同步返回，LLM部分流式输出。
        用于前端实时展示进度。
        """
        t_total = time.time()

        # 阶段一：语义检索
        t1 = time.time()
        chunks_a = self._search_by_jurisdiction(query, jurisdiction_a, top_n_per_jurisdiction)
        chunks_b = self._search_by_jurisdiction(query, jurisdiction_b, top_n_per_jurisdiction)
        t_search = time.time() - t1

        yield {"type": "retrieval_done", "data": {
            "chunks_a_count": len(chunks_a),
            "chunks_b_count": len(chunks_b)
        }}

        # 阶段二：聚合
        t2 = time.time()
        laws_a = self._aggregate_chunks_to_laws(chunks_a)
        laws_b = self._aggregate_chunks_to_laws(chunks_b)

        context_a = "\n\n---\n\n".join(law["context_text"] for law in laws_a)
        context_b = "\n\n---\n\n".join(law["context_text"] for law in laws_b)
        t_aggregate = time.time() - t2

        yield {"type": "aggregation_done", "data": {
            "laws_a": [{"law_id": l["law_id"], "title": l["title"],
                        "article_numbers": l["article_numbers"], "max_similarity": l["max_similarity"]}
                       for l in laws_a],
            "laws_b": [{"law_id": l["law_id"], "title": l["title"],
                        "article_numbers": l["article_numbers"], "max_similarity": l["max_similarity"]}
                       for l in laws_b]
        }}

        # 阶段三：流式 LLM 输出
        t3 = time.time()
        prompt = self._build_analysis_prompt(query, jurisdiction_a, context_a, jurisdiction_b, context_b)

        try:
            for chunk in self.qa_service._call_llm_stream(prompt):
                yield {"type": "content", "data": chunk}
        except Exception as e:
            yield {"type": "error", "data": f"生成分析报告时出错: {str(e)}"}
            return

        t_llm = time.time() - t3
        total_time = time.time() - t_total

        yield {"type": "done", "timing": {
            "semantic_search": f"{t_search:.4f}s",
            "aggregation": f"{t_aggregate:.4f}s",
            "llm_generation": f"{t_llm:.4f}s",
            "total": f"{total_time:.4f}s"
        }}

    def follow_up(
        self,
        original_analysis: Dict,
        follow_up_question: str
    ) -> Dict:
        """
        追问功能：基于已生成的分析报告，回答用户的后续问题。
        
        Args:
            original_analysis: 原始分析结果（包含 analysis_report、context_a、context_b 等）
            follow_up_question: 用户的追问问题
            
        Returns:
            包含追问回答的结果字典
        """
        t_start = time.time()
        
        # 从原始分析中提取必要信息
        analysis_report = original_analysis.get("analysis_report", "")
        jurisdiction_a = original_analysis.get("jurisdiction_a", "")
        jurisdiction_b = original_analysis.get("jurisdiction_b", "")
        
        # 构建追问的 prompt
        prompt = self._build_follow_up_prompt(
            follow_up_question,
            jurisdiction_a,
            jurisdiction_b,
            analysis_report
        )
        
        # 调用 LLM 生成回答
        answer = self.qa_service._call_llm(prompt)
        t_llm = time.time() - t_start
        
        print(f"【专题分析追问计时】LLM调用: {t_llm:.4f}s")
        
        return {
            "follow_up_question": follow_up_question,
            "answer": answer,
            "timing": {
                "llm_generation": f"{t_llm:.4f}s",
                "total": f"{t_llm:.4f}s"
            }
        }

    def _build_follow_up_prompt(
        self,
        question: str,
        jurisdiction_a: str,
        jurisdiction_b: str,
        original_report: str
    ) -> str:
        """
        构建追问的 Prompt。
        """
        prompt = (
            "你是一位资深的比较法研究专家，正在处理用户对已生成分析报告的追问。\n"
            "\n"
            "**重要原则：**\n"
            "- 必须基于以下已有的分析报告进行回答\n"
            "- 如果追问涉及报告中未涵盖的内容，可以适当扩展但需注明\n"
            "- 保持回答简洁，直接针对用户的问题\n"
            "- 引用条款时请注明法规名称和条款编号\n"
            "\n"
            "---\n"
            "\n"
            "**原始分析报告（" + jurisdiction_a + " vs " + jurisdiction_b + "）：**\n"
            "\n" +
            original_report +
            "\n"
            "---\n"
            "\n"
            "**用户追问：** " + question + "\n"
            "\n"
            "请基于以上分析报告，对用户的追问给出专业、准确的回答：\n"
        )
        
        return prompt
