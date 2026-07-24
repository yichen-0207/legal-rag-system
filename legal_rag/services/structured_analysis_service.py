import json
import time
import re
import hashlib
import logging
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

from repositories.elasticsearch import ElasticsearchRepository
from services.qa_service import QAService
from services.topics_config import get_topic_by_id, TOPICS
from core.config import settings

# 繁→简 常见法规标题映射（覆盖本项目实际数据）
_TRADITIONAL_TO_SIMPLIFIED = {
    "網絡安全法": "网络安全法",
    "個人資料保護法": "个人资料保护法",
    "電子服務法": "电子服务法",
    "電信綱要法": "电信纲要法",
    "網絡安全": "网络安全",
    "個人資料": "个人资料",
    "電子服務": "电子服务",
    "電信": "电信",
    "網絡": "网络",
    "資料": "资料",
    "規定": "规定",
    "機構": "机构",
    "監管": "监管",
    "違反": "违反",
    "處罰": "处罚",
    "義務": "义务",
    "權利": "权利",
    "責任": "责任",
    "許可": "许可",
    "申請": "申请",
    "審批": "审批",
    "通知": "通知",
    "條件": "条件",
    "標準": "标准",
    "要求": "要求",
    "措施": "措施",
    "系統": "系统",
    "資訊": "信息",
    "傳輸": "传输",
    "跨境": "跨境",
    "出境": "出境",
    "同意": "同意",
    "刪除": "删除",
    "修正": "修正",
    "查詢": "查询",
    "可攜": "可携",
    "限制": "限制",
    "處理": "处理",
    "蒐集": "收集",
    "利用": "利用",
    "公開": "公开",
    "保護": "保护",
    "安全": "安全",
    "風險": "风险",
    "評估": "评估",
    "報告": "报告",
    "記錄": "记录",
    "檔案": "档案",
    "保存": "保存",
    "期限": "期限",
    "逾期": "逾期",
    "損害": "损害",
    "賠償": "赔偿",
    "訴訟": "诉讼",
    "管轄": "管辖",
    "適用": "适用",
    "範圍": "范围",
    "對象": "对象",
    "主體": "主体",
    "控制者": "控制者",
    "處理者": "处理者",
    "受託人": "受托人",
}


def to_simplified(text: str) -> str:
    """将文本中的繁体中文转换为简体中文"""
    if not text:
        return text
    result = text
    for trad, simp in _TRADITIONAL_TO_SIMPLIFIED.items():
        result = result.replace(trad, simp)
    return result


class StructuredAnalysisService:
    """
    跨法域专题结构化对比分析服务

    核心流程（三步）：
    1. 检索：用预设检索词在两个法域分别搜索相关条款
    2. LLM结构化提取：让LLM从条款中提取JSON格式的结构化信息（不是生成回答）
    3. 返回两个法域的JSON + 对比维度定义，前端用模板渲染表格

    与QA的本质区别：
    - QA: 用户自由提问 → LLM生成自然语言回答
    - 本服务: 选预设专题 → LLM做信息提取(输出JSON) → 前端模板渲染对比表
    """

    _qa_service_instance = None

    def __init__(self):
        self.repo = ElasticsearchRepository()

    @property
    def qa_service(self):
        if StructuredAnalysisService._qa_service_instance is None:
            StructuredAnalysisService._qa_service_instance = QAService()
        return StructuredAnalysisService._qa_service_instance

    # ================================================================
    #  缓存指纹机制
    # ================================================================

    def _compute_fingerprint(self, topic: Dict, jurisdiction_a: str, jurisdiction_b: str) -> str:
        """
        生成专题分析的综合指纹（MD5）。

        指纹仅基于稳定输入参数（topic_id + jurisdictions + dimensions），
        不依赖 ES 检索结果（向量检索每次返回的 chunk 列表可能不同）。
        数据变化检测通过外部手动清缓存或版本号机制处理。
        """
        raw = json.dumps({
            "topic_id": topic.get("id", ""),
            "jurisdiction_a": jurisdiction_a,
            "jurisdiction_b": jurisdiction_b,
            "dimensions": sorted([(d["key"], d["label"]) for d in topic.get("dimensions", [])]),
            "search_queries": topic.get("search_queries", []),
        }, sort_keys=True, ensure_ascii=False)
        fp = hashlib.md5(raw.encode("utf-8")).hexdigest()
        logger.info(f"[Fingerprint] computed={fp[:16]}..., topic={topic.get('id')}, {jurisdiction_a} vs {jurisdiction_b}")
        return fp

    def _try_get_cached(self, topic_id: str, jur_a: str, jur_b: str, current_fp: str) -> Optional[Dict]:
        """
        尝试从 ES 缓存获取有效结果。
        返回缓存的 report_data 或 None（缓存不存在/已失效）。
        """
        logger.info(f"[Cache-Check] 查询缓存: {topic_id}, {jur_a} vs {jur_b}, fp={current_fp[:16]}...")
        cached = self.repo.get_analysis_cache(topic_id, jur_a, jur_b)
        if not cached or not isinstance(cached, dict):
            logger.info(f"[Cache-Check] 缓存不存在或格式异常")
            return None
        cached_fp = cached.get("fingerprint", "")
        report_data = cached.get("report_data")
        if not isinstance(report_data, dict):
            logger.warning(f"[Cache-Check] report_data 格式异常: {type(report_data)}")
            return None
        if cached_fp != current_fp:
            logger.info(
                f"[Cache-Check] 指纹不匹配，缓存失效。"
                f" cached={cached_fp[:16]}... current={current_fp[:16]}..."
            )
            return None
        logger.info(
            f"[Cache-Check] 命中！topic={topic_id}, {jur_a} vs {jur_b}, "
            f"updated_at={cached.get('updated_at')}"
        )

        # 反序列化 topic 字段（写入时转为JSON字符串）
        import json
        if isinstance(report_data, dict):
            topic_raw = report_data.get("topic")
            if isinstance(topic_raw, str):
                try:
                    report_data["topic"] = json.loads(topic_raw)
                    logger.info(f"[Cache-Check] topic字段已反序列化为字典")
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"[Cache-Check] topic字段反序列化失败，保持原值")

        return report_data

    def _search_topic(self, queries: List[str], jurisdiction: str, top_n: int) -> List[Dict]:
        """
        用多个查询词在指定法域中检索，合并去重后返回。
        多个查询词通过线程池并行搜索，大幅缩短检索耗时。
        所有查询词一次性批量编码，避免重复的 CPU embedding 计算。
        双重过滤：ES where 条件 + Python 后置校验（防止 ES knn+filter 兼容性问题）。
        """
        if not queries:
            return []

        # 查询去重：如果一条查询是另一条的语义子串（包含关系），只保留较长的
        # 例如 "数据保护 跨境传输 合规要求" 已覆盖 "跨境传输"
        unique = list(set(queries))
        unique.sort(key=len, reverse=True)  # 长查询优先
        kept = []
        for q in unique:
            if not any(q in k or k in q for k in kept):
                kept.append(q)
        queries = kept if kept else unique

        all_chunks = []
        seen_ids = set()

        # 一次性批量编码所有查询词（关键优化：避免 N 次独立 CPU 推理）
        batch_results = self.repo.hybrid_search_batch(
            queries, n_results=top_n, where={"jurisdiction": jurisdiction}
        )

        # 合并去重（跨查询词之间去重）
        for result in batch_results:
            if not result.get('ids') or not result['ids'][0]:
                continue
            for i in range(len(result['ids'][0])):
                cid = result['ids'][0][i]
                if cid not in seen_ids:
                    meta = result['metadatas'][0][i]
                    actual_jur = meta.get("jurisdiction", "")
                    if actual_jur != jurisdiction:
                        continue
                    seen_ids.add(cid)
                    all_chunks.append({
                        "content": result['documents'][0][i],
                        "law_id": meta.get("law_id", ""),
                        "title": to_simplified(meta.get("title", "")),
                        "article_number": meta.get("article_number", ""),
                        "chunk_index": meta.get("chunk_index", 0),
                        "jurisdiction": jurisdiction,
                    })

        # 按 law_id + chunk_index 排序
        all_chunks.sort(key=lambda c: (c["law_id"], c["chunk_index"]))
        return all_chunks

    def _collect_matched_laws(self, chunks: List[Dict]) -> List[Dict]:
        """从检索结果中提取命中的法规列表（去重，含标题和命中条款数）"""
        law_map = {}
        for ch in chunks:
            lid = ch["law_id"]
            if lid and lid not in law_map:
                law_map[lid] = {
                    "law_id": lid,
                    "title": ch.get("title", ""),
                    "chunk_count": 0
                }
            if lid in law_map:
                law_map[lid]["chunk_count"] += 1
        return sorted(law_map.values(), key=lambda x: x["chunk_count"], reverse=True)

    def _format_chunks_for_llm(self, chunks: List[Dict], jurisdiction: str) -> str:
        """
        将检索到的 chunks 格式化为供 LLM 阅读的文本。
        优化：截断过长内容（每条最多500字），减少 LLM 输入 token 数。
        """
        if not chunks:
            return "[未检索到相关条款，该法域数据库中可能不存在此专题的相关规定]"

        lines = []

        # 顶部：标注本次检索涉及哪些法规
        laws_in_scope = self._collect_matched_laws(chunks)
        law_names = [f"《{l['title']}》({l['chunk_count']}条)" for l in laws_in_scope]
        lines.append(f"[本次检索范围 - {jurisdiction}法域共命中以下法规]:")
        lines.append("  " + " | ".join(law_names))
        lines.append("")
        lines.append("--- 各法规具体条款原文 ---")
        lines.append("")

        current_law = None
        for ch in chunks:
            lid = ch["law_id"]
            title = to_simplified(ch["title"])  # 确保标题是简体
            art = ch["article_number"] or f"片段{ch['chunk_index']}"
            content = ch["content"]

            # 截断过长内容，控制总输入长度（每条最多400字）
            if len(content) > 400:
                content = content[:400] + "...(已截断)"

            if lid != current_law:
                current_law = lid
                lines.append(f"\n=== 《{title}》 ===")

            lines.append(f"【{art}】{content}")

        return "\n".join(lines)

    def _build_json_schema(self, dimensions: List[Dict]) -> str:
        """根据维度定义构建JSON schema示例"""
        fields = []
        for dim in dimensions:
            key = dim["key"]
            label = dim["label"]
            if key == "source_articles":
                fields.append(f'      "{key}": ["《法规名称》第X条", "《另一法规》Section Y", ...]')
            else:
                fields.append(f'      "{key}": "关于「{label}」的提取结果"')

        inner = ",\n".join(fields)
        return "{\n" + inner + "\n    }"

    def _build_extraction_prompt(
        self,
        topic_name: str,
        jurisdiction: str,
        dimensions: List[Dict],
        json_schema: str,
        context_text: str
    ) -> str:
        """构建LLM结构化提取Prompt——强调：内容字段必须写具体规定，不能只列法条编号"""
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"[PromptBuild] topic_name={topic_name!r}, jurisdiction={jurisdiction!r}, dimensions={len(dimensions)}, json_schema_len={len(json_schema)}, context_len={len(context_text)}")

        # 构建每个字段的详细说明 + 示例
        dim_details = []
        for d in dimensions:
            if d["key"] == "source_articles":
                dim_details.append(
                    f'- **{d["key"]}** ({d["label"]})：仅记录你引用了哪些法条，格式如 `["《法规名》第X条"]`'
                )
            else:
                dim_details.append(
                    f'- **{d["key"]}** ({d["label"]})：用2-4句话**概括该法域在此维度的具体规定内容**。'
                    f' 必须写出"谁、做什么、怎么做"，不要只列法条编号！'
                )

        # 使用纯字符串拼接，彻底避免与法规原文中 {} 冲突
        dim_details_text = "\n".join(dim_details)

        prompt = (
            "你是法律信息提取引擎。任务：从法律条款中提取结构化信息。\n"
            "\n"
            "## 专题与法域\n"
            "- 专题：" + topic_name + "\n"
            "- 法域：" + jurisdiction + "\n"
            "\n"
            "## 提取字段（共 " + str(len(dimensions)) + " 个）\n"
            "\n" +
            dim_details_text +
            "\n"
            "## ⚠️ 最重要规则（违反则结果无效）\n"
            "\n"
            "### 规则1：内容字段必须写具体规定\n"
            "对于 source_articles 以外的所有字段，你必须写出**具体的法律规定内容**：\n"
            '- ✅ 正确："关键基础设施运营者必须在事件发生后X小时内向监管机构报告，报告内容包括事件类型、影响范围和已采取的应对措施"\n'
            '- ❌ 错误："《CYBERSECURITYACT 2018》第16I、16L条"（这只是法条编号，不是内容！）\n'
            '- ❌ 错误："未在检索结果中找到相关规定"（如果条款中有相关信息就不能这样写）\n'
            "\n"
            "### 规则2：source_articles 只写法条引用\n"
            "- source_articles 字段专门用于列出你引用的法条编号\n"
            "- 其他字段绝对不能只填法条编号\n"
            "\n"
            "### 规则3：输出格式\n"
            "- 纯JSON，无markdown标记，无解释文字\n"
            "- 全部使用简体中文\n"
            "- 找不到信息时填：\"数据库中未找到此方面的规定\"\n"
            "\n"
            "## JSON 结构示例\n"
            "```json\n" +
            json_schema +
            "\n```\n"
            "\n"
            "## 法律条款原文\n" +
            jurisdiction +
            "的相关法律条款：\n"
            "\n" +
            context_text +
            "\n"
            "\n直接输出JSON："
        )

        logger.info(f"[PromptBuild] 生成prompt长度={len(prompt)}")
        if len(prompt) < 100:
            logger.error(f"[PromptBuild] prompt过短！内容: {prompt!r}")

        return prompt

    def _parse_json_extraction(self, raw_text: str, dimensions: List[Dict]) -> Dict:
        """解析LLM返回的结构化JSON，容错处理"""
        if not raw_text:
            return {d["key"]: "提取失败" for d in dimensions}

        # 尝试提取JSON（可能被包裹在 ```json ... ``` 中）
        cleaned = raw_text.strip()
        if "```" in cleaned:
            cleaned = re.sub(r'```(?:json)?\s*', '', cleaned).rstrip('`').strip()

        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned)
        if json_match:
            cleaned = json_match.group(0)

        try:
            parsed = json.loads(cleaned)
            result = {}
            for dim in dimensions:
                key = dim["key"]
                val = parsed.get(key)
                if val is None:
                    result[key] = "数据库中未找到此方面的规定"
                elif isinstance(val, list):
                    result[key] = val if val else []
                else:
                    result[key] = str(val) if val else "数据库中未找到此方面的规定"
            return result
        except (json.JSONDecodeError, Exception):
            # 终极兜底：逐维度用正则从原始文本中提取
            fallback = {}
            for dim in dimensions:
                key = dim["key"]
                label = dim["label"]
                found_val = None
                # 尝试多种匹配模式（用字符串拼接避免f-string与正则花括号冲突）
                esc_key = re.escape(key)
                esc_label = re.escape(label)
                patterns = [
                    r'(?:["\']?' + esc_key + r'["\']?|' + esc_label + r')\s*[:：]\s*"?(.+?)"?(?:[,}\n])',
                    r'"' + esc_key + r'"\s*:\s*"?(.+?)"?(?:[,}\n])',
                ]
                for pat in patterns:
                    m = re.search(pat, raw_text)
                    if m and m.group(1).strip():
                        found_val = m.group(1).strip().rstrip(',').rstrip('"')
                        break
                fallback[key] = found_val or f"[解析异常] {raw_text[:150]}..."
            return fallback

    def _generate_summary(
        self,
        topic: Dict,
        jur_a: str,
        jur_b: str,
        data_a: Dict,
        data_b: Dict
    ) -> str:
        """可选：基于已提取的两个JSON，让LLM写一段简短差异总结（100-200字）"""
        dim_lines = []
        for dim in topic["dimensions"]:
            if dim["key"] == "source_articles":
                continue
            key = dim["key"]
            label = dim["label"]
            val_a = data_a.get(key, "")
            val_b = data_b.get(key, "")
            dim_lines.append(f"- **{label}**: {jur_a}→{str(val_a)[:80]}; {jur_b}→{str(val_b)[:80]}")

        comparison_data = "\n".join(dim_lines)

        prompt = (
            "基于以下两个法域在「" + topic["name"] + "」专题上的结构化提取数据，写一段100-200字的客观差异总结。\n"
            "\n"
            "只总结最关键的3-5个差异点，不要重复已有数据。语气专业、简洁。\n"
            "\n"
            "## 提取数据对比\n"
            "\n" +
            comparison_data +
            "\n"
            "\n## 总结（100-200字）："
        )

        try:
            # 使用 qwen3.6-flash 进行摘要生成，追求速度
            summary_model = settings.llm_dashboard_model or "qwen3.6-flash"
            return self.qa_service._call_llm_with_model(prompt, model=summary_model)
        except Exception:
            return ""

    def list_topics(self) -> List[Dict]:
        """获取所有可用专题列表"""
        from services.topics_config import get_all_topics
        return get_all_topics()

    def get_topic_queries(self, topic_id: str) -> List[str]:
        """根据专题ID获取检索查询词列表"""
        from services.topics_config import get_topic_by_id
        topic = get_topic_by_id(topic_id)
        return topic["search_queries"] if topic else []

    def analyze_stream(
        self,
        topic_id: str,
        jurisdiction_a: str,
        jurisdiction_b: str,
        top_n_per_jurisdiction: int = 10,
        include_summary: bool = True
    ):
        """
        流式执行专题结构化分析，逐步 yield 事件。

        事件类型（并行生成优化版）：
          {"type": "search_done", "data": {...}}          — 检索完成
          {"type": "extracted", "jurisdiction": "澳门", "data": {}} — 单个法域提取完成
          {"type": "extraction_done", "data": {}}         — 两个法域都提取完成（对比表数据就绪）
          {"type": "dashboard_done", "data": {}}          — 仪表盘数据生成完成（优先展示）
          {"type": "summary_chunk", "data": "总结片段"}    — 总结文字流
          {"type": "done", "data": {完整结果}}             — 全部完成
          {"type": "error", "data": "错误信息"}
          {"type": "cache_hit", "data": {缓存命中时的完整结果}}
          {"type": "timing_update", "data": {...}}         — 各模块耗时更新
        """
        t_total = time.time()
        timing = {}

        topic = get_topic_by_id(topic_id)
        if not topic:
            yield {"type": "error", "data": f"未找到专题: {topic_id}"}
            return

        # ===== 步骤1：计算指纹 + 检查缓存（在ES检索之前，避免不必要的搜索开销）=====
        current_fingerprint = self._compute_fingerprint(topic, jurisdiction_a, jurisdiction_b)
        cached_result = self._try_get_cached(topic_id, jurisdiction_a, jurisdiction_b, current_fingerprint)
        if cached_result is not None:
            cache_response = {
                **cached_result,
                "_cache_hit": True,
                "_cache_fingerprint": current_fingerprint,
            }
            cache_response.setdefault("timing", {})["cache_status"] = "HIT"
            cache_response.setdefault("timing", {})["search"] = "0.00s (缓存命中，跳过检索)"
            cache_response.setdefault("timing", {})["total"] = f"{time.time() - t_total:.2f}s"
            yield {"type": "cache_hit", "data": cache_response}
            yield {"type": "done", "data": cache_response}
            return

        logger.info(f"[Cache-Stream] 未命中，开始生成。topic={topic_id}")

        # ===== 步骤2：检索 =====
        t1 = time.time()

        # 两个法域并行检索
        with ThreadPoolExecutor(max_workers=2) as executor:
            f_a = executor.submit(self._search_topic, topic["search_queries"], jurisdiction_a, top_n_per_jurisdiction)
            f_b = executor.submit(self._search_topic, topic["search_queries"], jurisdiction_b, top_n_per_jurisdiction)
            chunks_a = f_a.result()
            chunks_b = f_b.result()
        t_search = time.time() - t1
        timing["search"] = f"{t_search:.2f}s"
        yield {"type": "timing_update", "data": timing}

        matched_laws_a = self._collect_matched_laws(chunks_a)
        matched_laws_b = self._collect_matched_laws(chunks_b)

        yield {
            "type": "search_done",
            "data": {
                f"{jurisdiction_a}_chunks": len(chunks_a),
                f"{jurisdiction_b}_chunks": len(chunks_b),
                f"{jurisdiction_a}_laws": len(matched_laws_a),
                f"{jurisdiction_b}_laws": len(matched_laws_b),
            }
        }

        context_a = self._format_chunks_for_llm(chunks_a, jurisdiction_a)
        context_b = self._format_chunks_for_llm(chunks_b, jurisdiction_b)

        # ===== 步骤3：LLM提取 + 网络图生成 并行执行 =====
        # 网络图只需要 chunks 数据，可以与 LLM 提取同时开始
        t2 = time.time()
        json_schema = self._build_json_schema(topic["dimensions"])
        extraction_prompt_a = self._build_extraction_prompt(
            topic_name=topic["name"],
            jurisdiction=jurisdiction_a,
            dimensions=topic["dimensions"],
            json_schema=json_schema,
            context_text=context_a
        )
        extraction_prompt_b = self._build_extraction_prompt(
            topic_name=topic["name"],
            jurisdiction=jurisdiction_b,
            dimensions=topic["dimensions"],
            json_schema=json_schema,
            context_text=context_b
        )

        _extract_results = {}
        _parallel_results = {}
        # 流式提取使用配置的结构化提取模型
        _stream_model = settings.llm_extraction_model or "qwen3.6-plus"
        _stream_fallback = settings.llm_extraction_fallback_model or "qwen3.5-omni-plus"

        # ----- 流式提取：每个维度完成时立即 yield -----
        import queue as _queue
        import threading as _threading

        def _stream_extract(prompt, jur, dims, q, model_name, fallback_name):
            """在后台线程中流式调用 LLM，每检测到一个完整维度就放入队列"""
            raw = ""
            seen_keys = set()
            try:
                for chunk in self.qa_service._call_llm_stream(prompt, model=model_name, fallback_model=fallback_name):
                    raw += chunk
                    # 实时检测已完成的关键字段 "key": "value"
                    for dim in dims:
                        key = dim["key"]
                        if key in seen_keys:
                            continue
                        pattern = '"' + re.escape(key) + r'"\s*:\s*"((?:[^"\\]|\\.)*)"'
                        m = re.search(pattern, raw)
                        if m:
                            value = m.group(1)
                            seen_keys.add(key)
                            q.put(("dim", jur, key, value))
                # 完整解析兜底（补全可能被 regex 遗漏的字段）
                parsed = self._parse_json_extraction(raw, dims)
                for dim in dims:
                    key = dim["key"]
                    if key not in seen_keys and key in parsed and parsed.get(key):
                        q.put(("dim", jur, key, parsed[key]))
                _extract_results[jur] = (raw, parsed)
                q.put(("done", jur, parsed))
            except Exception as e:
                logger.warning(f"[StreamExtract] 法域{jur}提取失败: {e}")
                _extract_results[jur] = (raw, None)
                q.put(("done", jur, None))

        # 启动两个流式提取线程（两个法域并行）
        result_queue = _queue.Queue()
        extract_threads = []
        for _prompt, _jur in [(extraction_prompt_a, jurisdiction_a), (extraction_prompt_b, jurisdiction_b)]:
            t = _threading.Thread(target=_stream_extract, args=(_prompt, _jur, topic["dimensions"], result_queue, _stream_model, _stream_fallback))
            t.daemon = True
            t.start()
            extract_threads.append(t)

        # 消费队列：每完成一个维度立即 yield，前端逐行渲染
        dashboard_future = None
        done_count = 0
        while done_count < 2:
            typ, jur, *rest = result_queue.get()
            if typ == "dim":
                key, value = rest
                yield {"type": "extraction_dimension", "jurisdiction": jur, "key": key, "value": value, "dimensions": topic["dimensions"]}
            elif typ == "done":
                parsed = rest[0]
                done_count += 1
                if parsed is not None:
                    yield {"type": "extracted", "jurisdiction": jur, "data": parsed, "dimensions": topic["dimensions"]}
                else:
                    yield {"type": "error", "data": f"法域{jur}提取失败"}
                    yield {"type": "extracted", "jurisdiction": jur, "data": {}, "dimensions": topic["dimensions"]}

                # 两个法域都提取完成 → 后台启动仪表盘生成
                if done_count == 2:
                    for et in extract_threads:
                        et.join(timeout=2)

                    intermediate_result = {
                        "topic": {
                            "id": topic["id"],
                            "name": topic["name"],
                            "description": topic.get("description", ""),
                            "dimensions": topic["dimensions"]
                        },
                        "jurisdictions": [jurisdiction_a, jurisdiction_b],
                        "extraction": {
                            jurisdiction_a: _extract_results.get(jurisdiction_a, (None, {}))[1] or {},
                            jurisdiction_b: _extract_results.get(jurisdiction_b, (None, {}))[1] or {}
                        }
                    }

                    def _gen_dashboard(result):
                        t_start = time.time()
                        try:
                            summary_text, dashboard_data = self.generate_summary_and_dashboard(result)
                            _parallel_results["summary"] = summary_text
                            timing["llm_dashboard"] = f"{time.time() - t_start:.2f}s"
                            return dashboard_data
                        except Exception as e:
                            timing["llm_dashboard"] = f"{time.time() - t_start:.2f}s (失败)"
                            logger.warning(f"[Parallel] dashboard 生成失败: {e}")
                            return None

                    # 用独立线程池运行仪表盘（不阻塞当前生成器）
                    _dash_executor = ThreadPoolExecutor(max_workers=1)
                    dashboard_future = _dash_executor.submit(_gen_dashboard, intermediate_result)
                    _dash_executor.shutdown(wait=False)

        t_extract = time.time() - t2
        timing["llm_extraction"] = f"{t_extract:.2f}s"
        yield {"type": "timing_update", "data": timing}

        raw_result_a = _extract_results.get(jurisdiction_a, ("", {}))[0]
        raw_result_b = _extract_results.get(jurisdiction_b, ("", {}))[0]

        # 标记对比表数据就绪（仪表盘可能在后台正在生成）
        yield {
            "type": "extraction_done",
            "data": {
                "jurisdictions": [jurisdiction_a, jurisdiction_b],
                "dimensions": topic["dimensions"],
            }
        }

        # ===== 步骤4：获取仪表盘结果（等待后台任务）=====
        if dashboard_future:
            dashboard_data = dashboard_future.result()
        else:
            dashboard_data = None
        _parallel_results["dashboard"] = dashboard_data
        if dashboard_data:
            yield {"type": "dashboard_done", "data": dashboard_data}
        yield {"type": "timing_update", "data": timing}

        # 单独处理总结（如果需要流式输出）
        summary = _parallel_results.get("summary", "")
        if include_summary and summary:
            # 流式输出总结片段
            chunk_size = 30
            for i in range(0, len(summary), chunk_size):
                yield {"type": "summary_chunk", "data": summary[i:i+chunk_size]}

        # ===== 构建完整结果 =====
        result = {
            "topic": {
                "id": topic["id"],
                "name": topic["name"],
                "description": topic["description"],
                "dimensions": topic["dimensions"]
            },
            "jurisdictions": [jurisdiction_a, jurisdiction_b],
            "retrieval_stats": {
                f"{jurisdiction_a}_chunks": len(chunks_a),
                f"{jurisdiction_b}_chunks": len(chunks_b),
                f"{jurisdiction_a}_laws": len(matched_laws_a),
                f"{jurisdiction_b}_laws": len(matched_laws_b),
            },
            "matched_laws": {
                jurisdiction_a: matched_laws_a,
                jurisdiction_b: matched_laws_b,
            },
            "extraction": {
                jurisdiction_a: _extract_results.get(jurisdiction_a, (None, {}))[1] or {},
                jurisdiction_b: _extract_results.get(jurisdiction_b, (None, {}))[1] or {},
                "_raw_response_a": raw_result_a,
                "_raw_response_b": raw_result_b,
            },
            "summary": summary,
            "dashboard": _parallel_results.get("dashboard"),
            "_cache_hit": False,
            "_cache_fingerprint": current_fingerprint,
            "timing": {
                **timing,
                "total": f"{time.time() - t_total:.2f}s",
                "cache_status": "MISS"
            }
        }

        # 写入缓存（异常不影响返回结果）
        try:
            self.repo.save_analysis_cache(
                topic_id=topic_id,
                jurisdiction_a=jurisdiction_a,
                jurisdiction_b=jurisdiction_b,
                fingerprint=current_fingerprint,
                report_data=result,
            )
        except Exception as e:
            logger.warning(f"[Cache] 写入缓存失败（不影响结果）: {e}")

        yield {"type": "done", "data": result}

    def _build_summary_prompt(self, topic, jur_a, jur_b, data_a, data_b) -> str:
        """构建差异总结 prompt（已废弃，由 generate_summary_and_dashboard 统一生成）"""
        dim_lines = []
        for dim in topic["dimensions"]:
            if dim["key"] == "source_articles":
                continue
            key = dim["key"]
            label = dim["label"]
            val_a = data_a.get(key, "")
            val_b = data_b.get(key, "")
            dim_lines.append(f"- **{label}**: {jur_a}→{str(val_a)[:80]}; {jur_b}→{str(val_b)[:80]}")

        comparison_data = "\n".join(dim_lines)

        return (
            "基于以下两个法域在「" + topic["name"] + "」专题上的结构化提取数据，写一段100-200字的客观差异总结。\n"
            "\n只总结最关键的3-5个差异点，不要重复已有数据。语气专业、简洁。\n"
            "\n## 提取数据对比\n\n" + comparison_data + "\n\n## 总结（100-200字）："
        )

    # ================================================================
    #  升版能力：一次性生成 总结 + 仪表盘数据（1次LLM调用替代2次）
    # ================================================================

    def generate_summary_and_dashboard(self, analysis_result: Dict) -> tuple:
        """
        合并调用：基于已有的结构化分析结果，同时生成：
        - 差异总结文字（100-200字）
        - 仪表盘数据（评分/雷达图/差异点）

        Returns: (summary_text, dashboard_dict)
        比分别调用 _generate_summary + generate_dashboard 节省约 50% 时间。
        """
        topic = analysis_result.get("topic", {})
        jurs = analysis_result.get("jurisdictions", ["", ""])
        jur_a, jur_b = jurs[0], jurs[1]
        extraction = analysis_result.get("extraction", {})
        data_a = extraction.get(jur_a, {})
        data_b = extraction.get(jur_b, {})
        dimensions = [d for d in topic.get("dimensions", []) if d["key"] != "source_articles"]

        # 构建精简的对比数据
        comparison_lines = []
        for dim in dimensions:
            key = dim["key"]
            label = dim["label"]
            val_a = str(data_a.get(key, ""))[:150]
            val_b = str(data_b.get(key, ""))[:150]
            comparison_lines.append(f"【{label}】\n  {jur_a}: {val_a}\n  {jur_b}: {val_b}")
        comparison_text = "\n".join(comparison_lines)
        dim_list = "\n".join(f"  {i+1}. {d['label']}" for i, d in enumerate(dimensions))

        combined_prompt = (
            "你是法律合规评估专家。基于以下两个法域在「" + topic.get("name", "") + "」"
            "专题上的法规条款对比信息，完成两项任务。\n\n"

            "## 任务1：写一段100-200字的客观差异总结\n"
            "只总结最关键的3-5个差异点，不要重复已有数据。语气专业、简洁。\n\n"

            "## 任务2：输出仪表盘JSON数据\n"
            "{\n"
            '  "scores": {\n'
            f'    "{jur_a}": <1-5数字>,\n'
            f'    "{jur_b}": <1-5数字>\n'
            '  },\n'
            '  "score_reasons": {\n'
            f'    "{jur_a}": "<一句话>",\n'
            f'    "{jur_b}": "<一句话>"\n'
            "  },\n"
            '  "radar": {\n'
            f'    "{jur_a}": [<按维度顺序的1-5分数组>],\n'
            f'    "{jur_b}": [<同上>]\n'
            "  },\n"
            '  "key_differences": [\n'
            '    {"dimension":"<维度名>","summary":"<一句话>","severity":"major|moderate|minor",'
            f'"{jur_a}_detail":"<摘要>","{jur_b}_detail":"<摘要>"}}\n'
            "  ]\n"
            "}\n\n"

            "**维度顺序**:\n" + dim_list +
            "\n\n**对比数据**:\n\n" + comparison_text +
            "\n\n请先输出总结文字（用 ===SUMMARY=== 标记结尾），然后紧接输出JSON（用 ===DASHBOARD=== 标记结尾）：\n\n"
        )

        # 使用 qwen3.6-flash 生成仪表盘数据，追求速度优先
        dashboard_model = settings.llm_dashboard_model or "qwen3.6-flash"
        raw = self.qa_service._call_llm_with_model(combined_prompt, model=dashboard_model)
        return self._parse_combined_output(raw, dimensions, jur_a, jur_b)

    def _parse_combined_output(self, raw: str, dimensions: list, jur_a: str, jur_b: str) -> tuple:
        """解析合并输出的 总结+仪表盘 JSON"""
        import re
        summary = ""
        dashboard_raw = ""

        # 尝试按标记拆分
        if "===SUMMARY===" in raw and "===DASHBOARD===" in raw:
            parts = raw.split("===SUMMARY===")
            summary_part = parts[1].split("===DASHBOARD===")[0] if len(parts) > 1 else ""
            summary = summary_part.strip()
            dashboard_raw = raw.split("===DASHBOARD===")[1] if "===DASHBOARD===" in raw else ""
        elif "===DASHBOARD===" in raw:
            # 只有仪表盘部分
            summary = raw.split("===DASHBOARD===")[0].strip()
            dashboard_raw = raw.split("===DASHBOARD===")[1]
        else:
            # 无标记，尝试智能拆分：找最后一个 {
            last_brace = raw.rfind("{")
            if last_brace > 100:
                summary = raw[:last_brace].strip()
                dashboard_raw = raw[last_brace:]
            else:
                summary = raw[:500]  # 取前500字作为总结
                dashboard_raw = ""

        # 解析仪表盘 JSON
        parsed_dash = self._parse_dashboard_json(dashboard_raw, dimensions, jur_a, jur_b)

        dashboard = {
            "topic_name": "",
            "jurisdictions": [jur_a, jur_b],
            "scores": parsed_dash.get("scores", {}),
            "score_reasons": parsed_dash.get("score_reasons", {}),
            "radar": parsed_dash.get("radar", {}),
            "radar_labels": [d["label"] for d in dimensions],
            "key_differences": parsed_dash.get("key_differences", []),
        }

        return (summary or "（暂无总结）", dashboard)

    # ================================================================
    #  仪表盘数据（保留单独接口供前端按需调用）
    # ================================================================

    def generate_dashboard(self, analysis_result: Dict) -> Dict:
        """
        基于已有的结构化分析结果，生成仪表盘所需的数据：
        - 保护力度评分（每个法域 1-5 分）
        - 关键差异点列表
        - 雷达图各维度得分
        """
        topic = analysis_result.get("topic", {})
        jur_a = analysis_result.get("jurisdictions", ["", ""])[0]
        jur_b = analysis_result.get("jurisdictions", ["", ""])[1]
        extraction = analysis_result.get("extraction", {})
        data_a = extraction.get(jur_a, {})
        data_b = extraction.get(jur_b, {})
        dimensions = [d for d in topic.get("dimensions", []) if d["key"] != "source_articles"]

        comparison_lines = []
        for dim in dimensions:
            key = dim["key"]
            label = dim["label"]
            val_a = data_a.get(key, "")
            val_b = data_b.get(key, "")
            comparison_lines.append(f"【{label}】\n  {jur_a}: {val_a}\n  {jur_b}: {val_b}")

        comparison_text = "\n".join(comparison_lines)

        dashboard_prompt = (
            "你是法律合规评估专家。基于以下两个法域在「" + topic.get("name", "") + "」"
            "专题上的法规条款对比信息，输出严格的 JSON（不要 markdown 包裹）：\n\n"
            "{\n"
            '  "scores": {\n'
            f'    "{jur_a}": <1-5的数字>,\n'
            f'    "{jur_b}": <1-5的数字>\n'
            '  },\n'
            '  "score_reasons": {\n'
            f'    "{jur_a}": "<一句话>",\n'
            f'    "{jur_b}": "<一句话>"\n'
            "  },\n"
            '  "radar": {\n'
            f'    "{jur_a}": [<每个维度1-5分的数字数组，顺序与下方维度一致>],\n'
            f'    "{jur_b}": [<同上>]\n'
            "  },\n"
            '  "key_differences": [\n'
            '    {"dimension":"<维度名>","summary":"<一句话>","severity":"major|moderate|minor",'
            f'"{jur_a}_detail":"<摘要>","{jur_b}_detail":"<摘要>"}}\n'
            "  ]\n"
            "}\n\n"
            "**维度顺序**:\n"
            + "\n".join(f"  {i+1}. {d['label']}" for i, d in enumerate(dimensions))
            + "\n\n**对比原始数据**:\n\n"
            + comparison_text
            + "\n\n直接输出 JSON:"
        )

        # 使用 qwen3.6-flash 生成仪表盘数据，追求速度优先
        dashboard_model = settings.llm_dashboard_model or "qwen3.6-flash"
        raw = self.qa_service._call_llm_with_model(dashboard_prompt, model=dashboard_model)
        parsed = self._parse_dashboard_json(raw, dimensions, jur_a, jur_b)

        return {
            "topic_name": topic.get("name", ""),
            "jurisdictions": [jur_a, jur_b],
            "scores": parsed.get("scores", {}),
            "score_reasons": parsed.get("score_reasons", {}),
            "radar": parsed.get("radar", {}),
            "radar_labels": [d["label"] for d in dimensions],
            "key_differences": parsed.get("key_differences", []),
        }

    def _parse_dashboard_json(self, raw: str, dimensions: list, jur_a: str, jur_b: str) -> dict:
        """解析 LLM 返回的仪表盘 JSON，容错处理"""
        import re
        cleaned = raw.strip()
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if json_match:
            cleaned = json_match.group(0)
        try:
            parsed = json.loads(cleaned)
            # 确保字段存在
            parsed.setdefault("scores", {jur_a: 3.0, jur_b: 3.0})
            parsed.setdefault("score_reasons", {jur_a: "", jur_b: ""})
            parsed.setdefault("radar", {jur_a: [3.0] * len(dimensions), jur_b: [3.0] * len(dimensions)})
            parsed.setdefault("key_differences", [])
            return parsed
        except (json.JSONDecodeError, Exception):
            n = len(dimensions)
            return {
                "scores": {jur_a: 3.0, jur_b: 3.0},
                "score_reasons": {jur_a: "解析异常，使用默认值", jur_b: "解析异常，使用默认值"},
                "radar": {jur_a: [3.0] * n, jur_b: [3.0] * n},
                "key_differences": []
            }

    # ================================================================
    #  升级能力2：AI 解读（按需调用）
    # ================================================================

    def ai_interpret(
        self,
        dimension_label: str,
        detail_a: str,
        detail_b: str,
        jurisdiction_a: str,
        jurisdiction_b: str,
        topic_name: str
    ) -> str:
        """
        对某个具体差异维度进行 AI 深度解读。
        用通俗语言解释实务影响。
        """
        prompt = (
            "你是资深跨境法律顾问。请对以下法律差异进行通俗易懂的深度解读。\n\n"
            "**专题**：" + topic_name + "\n"
            "**对比法域**：" + jurisdiction_a + " vs " + jurisdiction_b + "\n"
            "**差异维度**：" + dimension_label + "\n\n"
            "---\n\n"
            "**" + jurisdiction_a + "的规定**：\n" + detail_a + "\n\n"
            "**" + jurisdiction_b + "的规定**：\n" + detail_b + "\n\n"
            "---\n\n"
            "请从以下几个角度进行解读（200-400字）：\n"
            "1. 这个差异的本质是什么？\n"
            "2. 对企业在两地的合规策略有何不同影响？\n"
            "3. 有什么实操建议？\n\n"
            "用通俗的语言回答，避免过于学术化的表述。"
        )
        return self.qa_service._call_llm(prompt)

    # ================================================================
    #  升级能力3：关联网络图数据
    # ================================================================

    def get_network_graph(
        self,
        chunks_a: list,
        chunks_b: list,
        center_law_id: Optional[str] = None,
        similarity_threshold: float = 0.75
    ) -> Dict:
        """
        基于检索到的两个法域的法条，构建关联网络图数据。

        升级版 v2.0：
        - 主题分类（基于关键词自动归类）
        - 节点重要性评分（度数+核心性+中心性）
        - 边标签增强（相似度百分比、关系类型）
        - 最短路径算法支持
        - 保护力度评分（颜色深浅编码）
        """
        import hashlib
        import re

        # ========== 主题分类词库 ==========
        THEME_KEYWORDS = {
            "数据主体权利": ["同意", "知情", "访问", "更正", "删除", "携带", "反对", "拒绝",
                            "consent", "right", "access", "rectification", "erasure", "portability"],
            "数据处理义务": ["处理", "收集", "使用", "存储", "保留", "目的限制", "最小化",
                           "process", "collect", "store", "purpose", "minimise"],
            "安全保障": ["安全", "保护措施", "加密", "泄露", "通报", "事故", "技术",
                       "security", "breach", "encrypt", "incident"],
            "跨境传输": ["传输", "转移", "出境", "境外", "跨境", " adequacy", "safeguard",
                        "transfer", "cross-border", "outbound"],
            "监管与处罚": ["处罚", "罚款", "责任", "违规", "刑事", "民事", "监管", "执法",
                         "penalty", "fine", "offence", "liability", "enforcement"],
            "定义与范围": ["定义", "指", "是指", "包括", "适用", "范围", "含义",
                          "definition", "means", "includes", "scope"],
        }

        # 保护力度评分关键词（用于节点颜色深浅）
        PROTECTION_STRONG = ["必须", "应当", "严禁", "禁止", "不得", "强制", "shall", "must", "prohibited"]
        PROTECTION_WEAK = ["可以", "可", "建议", "鼓励", "may", "should", "encourage"]

        def _normalize_jurisdiction(jur: str, law_id: str) -> str:
            jur_map = {
                "us": "美国", "usa": "美国", "united states": "美国",
                "hk": "香港", "hong kong": "香港",
                "sg": "新加坡", "singapore": "新加坡",
                "mac": "澳门", "macau": "澳门", "mo": "澳门",
                "tw": "台湾", "taiwan": "台湾",
            }
            if jur:
                normalized = jur_map.get(jur.lower(), jur)
                if normalized != jur:
                    return normalized
            if not jur:
                lid = law_id or ""
                if any(lid.startswith(prefix) for prefix in ["MAC", "MO"]):
                    jur = "澳门"
                elif lid.startswith("SG"):
                    jur = "新加坡"
                elif lid.startswith("HK"):
                    jur = "香港"
                elif lid.startswith("TW"):
                    jur = "台湾"
                elif lid.startswith("US"):
                    jur = "美国"
            return jur or "未知"

        all_chunks = []
        for ch in chunks_a:
            jur = (ch.get("jurisdiction") or ch.get("_jurisdiction") or "").strip()
            jur = _normalize_jurisdiction(jur, ch.get("law_id", ""))
            ch["_jurisdiction"] = jur
            all_chunks.append({**ch, "_side": "A"})
        for ch in chunks_b:
            jur = (ch.get("jurisdiction") or ch.get("_jurisdiction") or "").strip()
            jur = _normalize_jurisdiction(jur, ch.get("law_id", ""))
            ch["_jurisdiction"] = jur
            all_chunks.append({**ch, "_side": "B"})

        # 去重 + 构建节点（增强版）
        # 关键修复：同一法条可能有多个chunk（ES分块存储），需要按chunk_index排序后拼接完整内容
        chunk_groups = {}  # key -> list of chunks
        for ch in all_chunks:
            key = f"{ch.get('law_id','')}_{ch.get('article_number','')}"
            if not key or key.endswith("_"):
                continue
            if key not in chunk_groups:
                chunk_groups[key] = []
            chunk_groups[key].append(ch)

        nodes = []
        node_map = {}
        for key, group in chunk_groups.items():
            # 按chunk_index排序，确保内容顺序正确
            group.sort(key=lambda c: c.get("chunk_index", 0))
            # 取第一个chunk的元数据，拼接所有chunk的内容
            ch = group[0]
            content = "\n".join(c.get("content", "") for c in group if c.get("content", ""))
            node_id = f"node_{len(nodes)}"
            title = to_simplified(ch.get("title", ""))
            art = ch.get("article_number", "")
            law_name = title.split(" - ")[0] if " - " in title else title[:16]
            content_lower = content.lower()

            # 判断是否为核心法条
            is_definition = any(kw in content_lower for kw in ["定义", "指", "是指", "包括", "meaning", "definition", "refers to"])
            is_penalty = any(kw in content_lower for kw in ["处罚", "罚款", "刑事责任", "penalty", "fine", "offence", "违例"])
            is_core = is_definition or is_penalty

            # 主题分类
            detected_themes = []
            for theme_name, kws in THEME_KEYWORDS.items():
                match_count = sum(1 for kw in kws if kw in content_lower)
                if match_count >= 1:  # 至少匹配1个关键词
                    detected_themes.append(theme_name)
            primary_theme = detected_themes[0] if detected_themes else "其他"

            # 保护力度评分 (0-100)
            strong_count = sum(1 for kw in PROTECTION_STRONG if kw in content)
            weak_count = sum(1 for kw in PROTECTION_WEAK if kw in content)
            protection_score = min(100, max(20, 50 + strong_count * 15 - weak_count * 10))

            nodes.append({
                "id": node_id,
                "label": f"{art}" if art else f"{law_name[:8]}",
                "title": f"{title} - {art}",
                "full_title": title,
                "law_name": law_name,
                "jurisdiction": ch.get("_jurisdiction", ""),
                "_side": ch.get("_side", ""),
                "law_id": ch.get("law_id", ""),
                "article_number": art,
                "content": content,                    # 完整内容（用于详情展示）
                "content_preview": content[:500],      # 预览内容（适当增加长度）
                "is_core": is_core,
                "node_type": "definition" if is_definition else ("penalty" if is_penalty else "normal"),
                # ---- v6.0: 实体类型标注 ----
                "entity_type": self._classify_entity_type(content),
                "entity_label": self.ENTITY_TYPE_RULES.get(self._classify_entity_type(content), {}).get("label", "条款"),
                "entity_color": self.ENTITY_TYPE_RULES.get(self._classify_entity_type(content), {}).get("color", "#6B7280"),
                "entity_shape": self.ENTITY_TYPE_RULES.get(self._classify_entity_type(content), {}).get("shape", "ellipse"),
                "degree": 0,
                "cluster": ch.get("_side", ""),
                # 新增字段
                "themes": detected_themes,
                "primary_theme": primary_theme,
                "protection_score": protection_score,
                "importance_score": 0,  # 后续计算综合重要性
            })
            node_map[key] = node_id

        # 构建边（增强版）
        edges = []
        edge_set = set()

        # ===== 边类型1：同法规内关联（连续条款）=====
        by_law = {}
        for i, ch in enumerate(all_chunks):
            lid = ch.get("law_id", "")
            if lid not in by_law:
                by_law[lid] = []
            by_law[lid].append(i)

        for lid, indices in by_law.items():
            sorted_idxs = sorted(indices, key=lambda idx: all_chunks[idx].get("chunk_index", 0))
            for j in range(len(sorted_idxs) - 1):
                ch1 = all_chunks[sorted_idxs[j]]
                ch2 = all_chunks[sorted_idxs[j + 1]]
                k1 = f"{ch1.get('law_id','')}_{ch1.get('article_number','')}"
                k2 = f"{ch2.get('law_id','')}_{ch2.get('article_number','')}"
                if k1 in node_map and k2 in node_map:
                    eid = tuple(sorted([node_map[k1], node_map[k2]]))
                    if eid not in edge_set:
                        edge_set.add(eid)
                        edges.append({
                            "from": eid[0], "to": eid[1],
                            "type": "same_law",
                            "label": "同法规关联",
                            "description": "同一法律中的连续或相关条款",
                            "weight": 2,
                            "style": "solid",
                            "color": "#94A3B8",
                            "width": 1,
                        })

        # ===== 边类型2：跨法域语义关联边（多策略匹配）=====
        cross_edges = []

        def extract_features(content, law_name, article_num):
            text = content or ""
            chars = set(re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text))
            text_clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
            bigrams = set(text_clean[i:i+2] for i in range(len(text_clean)-1)) if len(text_clean) > 1 else set()
            legal_keywords = set()
            for kw_list_item in ["数据", "个人", "信息", "保护", "隐私", "安全", "处罚",
                                 "责任", "义务", "权利", "同意", "披露", "传输", "存储",
                                 "处理", "收集", "使用", "删除", "更正", "访问", "泄露",
                                 "通知", "监管", "合规", "违规", "罚款", "刑事", "民事",
                                 "定义", "范围", "适用", "主体", "控制者", "当事人"]:
                if kw_list_item in text:
                    legal_keywords.add(kw_list_item)
            return {"chars": chars, "bigrams": bigrams, "keywords": legal_keywords}

        features_a = [extract_features(ch.get("content",""), ch.get("title",""), ch.get("article_number","")) for ch in chunks_a]
        features_b = [extract_features(ch.get("content",""), ch.get("title",""), ch.get("article_number","")) for ch in chunks_b]

        for i, (ch_a, feat_a) in enumerate(zip(chunks_a, features_a)):
            for j, (ch_b, feat_b) in enumerate(zip(chunks_b, features_b)):
                scores = []

                if feat_a["chars"] and feat_b["chars"]:
                    char_overlap = len(feat_a["chars"] & feat_b["chars"]) / min(len(feat_a["chars"]), len(feat_b["chars"]))
                    scores.append(char_overlap * 0.8)

                if feat_a["bigrams"] and feat_b["bigrams"]:
                    bg_overlap = len(feat_a["bigrams"] & feat_b["bigrams"]) / min(len(feat_a["bigrams"]), len(feat_b["bigrams"]))
                    scores.append(bg_overlap * 1.0)

                if feat_a["keywords"] and feat_b["keywords"]:
                    kw_overlap = len(feat_a["keywords"] & feat_b["keywords"]) / max(len(feat_a["keywords"] | feat_b["keywords"]), 1)
                    scores.append(kw_overlap * 1.5)

                if feat_a.get("article_num") and feat_b.get("article_num"):
                    if feat_a["article_num"] == feat_b["article_num"]:
                        scores.append(0.3)

                final_score = sum(scores) / max(len(scores), 1)

                if final_score >= 0.08:
                    ka = f"{ch_a.get('law_id','')}_{ch_a.get('article_number','')}"
                    kb = f"{ch_b.get('law_id','')}_{ch_b.get('article_number','')}"
                    if ka in node_map and kb in node_map:
                        eid = tuple(sorted([node_map[ka], node_map[kb]]))
                        if eid not in edge_set:
                            edge_set.add(eid)

                            # 判断关系类型
                            rel_type = "semantic_similar"
                            if final_score >= 0.25:
                                rel_type = "strong_match"
                                edge_label = f"高度相似 {final_score:.0%}"
                                edge_color = "#EF4444"  # 红色-强关联
                                edge_style = "solid"
                                edge_width = 3  # 必须是整数（ES字段类型为long）
                            elif final_score >= 0.15:
                                rel_type = "moderate_match"
                                edge_label = f"语义相关 {final_score:.0%}"
                                edge_color = "#F59E0B"  # 橙色-中等
                                edge_style = "solid"
                                edge_width = 2  # 必须是整数（ES字段类型为long）
                            else:
                                rel_type = "weak_match"
                                edge_label = f"弱关联 {final_score:.0%}"
                                edge_color = "#D97706"  # 深橙-弱关联
                                edge_style = "dash"
                                edge_width = 1  # 必须是整数（ES字段类型为long）

                            edge_data = {
                                "from": eid[0], "to": eid[1],
                                "type": "cross_jurisdiction",
                                "relation_type": rel_type,
                                "label": edge_label,
                                "similarity": round(final_score, 3),
                                "weight": min(5, max(1, int(final_score * 12))),
                                "style": edge_style,
                                "color": edge_color,
                                "width": edge_width,
                            }
                            edges.append(edge_data)
                            cross_edges.append(edge_data)

        # ===== 计算图指标 =====
        # 度数
        degree_count = {n["id"]: 0 for n in nodes}
        for e in edges:
            degree_count[e["from"]] = degree_count.get(e["from"], 0) + 1
            degree_count[e["to"]] = degree_count.get(e["to"], 0) + 1

        # 综合重要性评分（度数归一化 + 核心加分 + 保护力度加权）
        max_degree = max(degree_count.values()) if degree_count else 1
        for n in nodes:
            deg = degree_count.get(n["id"], 0)
            deg_norm = deg / max(max_degree, 1)
            core_bonus = 0.3 if n.get("is_core") else 0
            prot_factor = n.get("protection_score", 50) / 100 * 0.2
            n["degree"] = deg
            n["importance_score"] = round(deg_norm * 0.5 + core_bonus + prot_factor, 3)

        # 构建邻接表（用于路径查找）
        adjacency = {n["id"]: [] for n in nodes}
        for e in edges:
            adjacency[e["from"]].append(e["to"])
            adjacency[e["to"]].append(e["from"])

        # BFS最短路径算法（供前端调用）
        def find_shortest_path(start_id, end_id):
            """BFS查找两节点间最短路径"""
            if start_id not in adjacency or end_id not in adjacency:
                return None
            visited = {start_id}
            queue = [[start_id]]
            while queue:
                path = queue.pop(0)
                node = path[-1]
                if node == end_id:
                    return path
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_path = path + [neighbor]
                        queue.append(new_path)
            return None

        # 预计算所有跨法域的最短路径对（供前端使用）
        path_cache = {}
        nodes_a_ids = [n["id"] for n in nodes if n.get("_side") == "A"]
        nodes_b_ids = [n["id"] for n in nodes if n.get("_side") == "B"]
        for na in nodes_a_ids[:10]:  # 限制计算量
            for nb in nodes_b_ids[:10]:
                p = find_shortest_path(na, nb)
                if p and len(p) <= 6:  # 只缓存合理长度的路径
                    path_cache[f"{na}->{nb}"] = p

        # 如果指定了中心节点，过滤为子图
        if center_law_id and center_law_id in [n["law_id"] for n in nodes]:
            center_nodes = {n["id"] for n in nodes if n["law_id"] == center_law_id}
            neighbor_ids = set(center_nodes)
            for e in edges:
                if e["from"] in center_nodes:
                    neighbor_ids.add(e["to"])
                if e["to"] in center_nodes:
                    neighbor_ids.add(e["from"])
            nodes = [n for n in nodes if n["id"] in neighbor_ids]
            edges = [e for e in edges if e["from"] in neighbor_ids and e["to"] in neighbor_ids]

        # ========== [v6.0] 抗杂乱过滤 ==========
        MAX_NODES = 40
        MAX_EDGES_PER_NODE = 8

        # 记录原始节点数（供前端展示截断提示）—— 保留完整数据由前端控制显示
        _raw_node_count = len(nodes)

        # 1. 节点数限制（已换由前端控制，仅统计标记不断）
        # 保留每节点最大边数限制（减少视觉杂乱）

        # 2. 每节点最大边数限制
        node_edge_count = {}
        filtered_edges = []
        for e in edges:
            src, tgt = e["from"], e["to"]
            if node_edge_count.get(src, 0) >= MAX_EDGES_PER_NODE and node_edge_count.get(tgt, 0) >= MAX_EDGES_PER_NODE:
                continue
            filtered_edges.append(e)
            node_edge_count[src] = node_edge_count.get(src, 0) + 1
            node_edge_count[tgt] = node_edge_count.get(tgt, 0) + 1
        edges = filtered_edges

        # 3. 给跨域边添加语义关系类型
        for e in edges:
            if e.get("type") != "cross_jurisdiction":
                continue
            # 查找源节点和目标节点内容
            src_node = next((n for n in nodes if n["id"] == e["from"]), None)
            tgt_node = next((n for n in nodes if n["id"] == e["to"]), None)
            src_content = (src_node.get("content", "") or "") if src_node else ""
            tgt_content = (tgt_node.get("content", "") or "") if tgt_node else ""
            combined = src_content + " " + tgt_content

            # 检测语义关系
            rel = self._detect_relation_type(combined)
            if rel:
                rel_info = self.RELATION_TYPE_RULES.get(rel, {})
                e["relation_type"] = rel
                e["relation_label"] = rel_info.get("label", "关联")
                e["relation_color"] = rel_info.get("color", "#9B8EA3")

        # 4. 重新计算节点度数（过滤后）
        degree_count = {n["id"]: 0 for n in nodes}
        for e in edges:
            degree_count[e["from"]] = degree_count.get(e["from"], 0) + 1
            degree_count[e["to"]] = degree_count.get(e["to"], 0) + 1
        for n in nodes:
            n["degree"] = degree_count.get(n["id"], 0)

        # 统计信息（增强版）
        entity_type_distribution = {}
        for n in nodes:
            et = n.get("entity_type", "Article")
            entity_type_distribution[et] = entity_type_distribution.get(et, 0) + 1

        stats = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "cross_jurisdiction_edges": len([e for e in edges if e.get("type") == "cross_jurisdiction"]),
            "same_law_edges": len([e for e in edges if e.get("type") == "same_law"]),
            "core_nodes": len([n for n in nodes if n.get("is_core")]),
            "max_degree": max((n.get("degree", 0) for n in nodes), default=0),
            "avg_degree": round(sum(n.get("degree", 0) for n in nodes) / max(len(nodes), 1), 1),
            "jurisdictions": list(set(n.get("jurisdiction", "") for n in nodes)),
            "themes_available": list(set(n.get("primary_theme", "") for n in nodes)),
            "has_cross_edges": len([e for e in edges if e.get("type") == "cross_jurisdiction"]) > 0,
            "path_cache_size": len(path_cache),
            # [v6.0] 新增
            "entity_type_distribution": entity_type_distribution,
            "node_types_available": list(entity_type_distribution.keys()),
            "max_nodes_limit": MAX_NODES,
            "max_edges_per_node": MAX_EDGES_PER_NODE,
            "schema_version": "6.0",
        }

        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "raw_node_count": _raw_node_count,
            "stats": stats,
            "adjacency": adjacency,
            "path_cache": path_cache,
            "has_cross_edges": len([e for e in edges if e.get("type") == "cross_jurisdiction"]) > 0,
            "schema_version": "6.0",
        }

    # ================================================================
    #  升级能力4：多法域通用知识图谱构建 v6.0
    #  支持用户自主选择2-4个法域进行专题分析
    #  新增：实体类型分类（5类）、语义关系边、抗杂乱机制
    # ================================================================

    # ---- 实体类型检测规则 ----
    ENTITY_TYPE_RULES = {
        "LegalSubject": {
            "label": "法律主体",
            "keywords": ["控制者", "处理者", "受託人", "监管机构", "主体", "当事人",
                        "controller", "processor", "supervisory authority", "data subject",
                        "数据主体", "用户", "消费者", "机构", "组织", "企业", "公司",
                        "主管机关", "监管", "当局", "委员会", "政府部门"],
            "color": "#34A853",
            "shape": "roundRect",
        },
        "Obligation": {
            "label": "法律义务",
            "keywords": ["应当", "必须", "有义务", "负责", "承担", "确保", "应",
                        "shall", "must", "obligation", "duty", "required",
                        "采取", "提供", "通知", "告知", "取得", "获得同意",
                        "保存", "保留", "限制", "禁止", "不得"],
            "color": "#EA4335",
            "shape": "diamond",
        },
        "Right": {
            "label": "法律权利",
            "keywords": ["有权", "权利", "可以要求", "有权要求", "自由",
                        "right", "entitled", "have the right",
                        "访问权", "更正权", "删除权", "反对权", "携带权",
                        "access", "rectification", "erasure", "portability"],
            "color": "#0F9D58",
            "shape": "roundRect",
        },
        "Penalty": {
            "label": "处罚措施",
            "keywords": ["处罚", "罚款", "刑事责任", "罚金", "违例", "罪",
                        "penalty", "fine", "offence", "liable", "sanction",
                        "处以", "判处", "承担法律责任", "民事赔偿", "损害赔偿"],
            "color": "#FF6D00",
            "shape": "triangle",
        },
        "Concept": {
            "label": "法律概念",
            "keywords": ["定义", "指", "是指", "包括", "含义", "meaning", "definition",
                        "refers to", "means", "includes",
                        "个人数据", "个人资料", "个人信息", "敏感数据", "敏感资料",
                        "处理", "跨境传输", "自动化决策", "画像分析"],
            "color": "#FABB05",
            "shape": "ellipse",
        },
    }

    # ---- 关系类型检测规则 ----
    RELATION_TYPE_RULES = {
        "defines": {"keywords": ["定义", "是指", "meaning", "definition", "refers to"], "label": "定义", "color": "#FABB05"},
        "imposes": {"keywords": ["应当", "必须", "有义务", "shall", "must", "required to"], "label": "设定义务", "color": "#EA4335"},
        "grants": {"keywords": ["有权", "权利", "right to", "entitled"], "label": "授予权利", "color": "#0F9D58"},
        "prohibits": {"keywords": ["不得", "禁止", "严禁", "prohibited", "shall not"], "label": "禁止", "color": "#FF6D00"},
        "penalizes": {"keywords": ["罚款", "处罚", "刑事责任", "penalty", "fine", "liable"], "label": "处罚", "color": "#DC2626"},
        "exception": {"keywords": ["除外", "例外", "不适用", "exception", "unless", "provided that"], "label": "例外", "color": "#795548"},
    }

    def _classify_entity_type(self, content: str) -> str:
        """基于关键词规则检测法条的实体类型——优先返回最具体的类型"""
        if not content:
            return "Article"
        for etype, rules in self.ENTITY_TYPE_RULES.items():
            match_count = sum(1 for kw in rules["keywords"] if kw in content)
            if match_count >= 2:
                return etype
        return "Article"

    def _detect_relation_type(self, content: str) -> Optional[str]:
        """检测法条内容的主要关系类型"""
        if not content:
            return None
        best_type = None
        best_count = 0
        for rtype, rules in self.RELATION_TYPE_RULES.items():
            count = sum(1 for kw in rules["keywords"] if kw in content.lower())
            if count > best_count:
                best_count = count
                best_type = rtype
        return best_type if best_count >= 1 else None

    def get_network_graph_v5(
        self,
        jurisdiction_chunks: Dict[str, list],
        similarity_threshold: float = 0.75,
        center_law_id: Optional[str] = None
    ) -> Dict:
        """
        多法域通用知识图谱构建器 v5.0

        功能：
        - 支持2-4个法域动态选择
        - 自动两两计算跨法域语义关联
        - 智能多簇布局参数生成
        - 完整的节点/边/统计信息输出

        Args:
            jurisdiction_chunks: 法域名称到chunks列表的字典
                例如：{
                    "澳门": [chunk1, chunk2, ...],
                    "新加坡": [chunk3, chunk4, ...],
                    "香港": [chunk5, ...]
                }
            similarity_threshold: 语义相似度阈值（默认0.75）
            center_law_id: 可选，指定中心法律ID（用于子图过滤）

        Returns:
            包含nodes、edges、stats等完整图谱数据的字典

        Raises:
            ValueError: 如果法域数量不在2-4之间
        """
        import hashlib
        import re

        # ========== 参数校验 ==========
        if not 2 <= len(jurisdiction_chunks) <= 4:
            raise ValueError(
                f"法域数量必须在2-4之间，当前传入 {len(jurisdiction_chunks)} 个法域: "
                f"{list(jurisdiction_chunks.keys())}"
            )

        jurisdictions = list(jurisdiction_chunks.keys())
        n_jurisdictions = len(jurisdictions)

        # ========== 主题分类词库 ==========
        THEME_KEYWORDS = {
            "数据主体权利": ["同意", "知情", "访问", "更正", "删除", "携带", "反对", "拒绝",
                            "consent", "right", "access", "rectification", "erasure", "portability"],
            "数据处理义务": ["处理", "收集", "使用", "存储", "保留", "目的限制", "最小化",
                           "process", "collect", "store", "purpose", "minimise"],
            "安全保障": ["安全", "保护措施", "加密", "泄露", "通报", "事故", "技术",
                       "security", "breach", "encrypt", "incident"],
            "跨境传输": ["传输", "转移", "出境", "境外", "跨境", "adequacy", "safeguard",
                        "transfer", "cross-border", "outbound"],
            "监管与处罚": ["处罚", "罚款", "责任", "违规", "刑事", "民事", "监管", "执法",
                         "penalty", "fine", "offence", "liability", "enforcement"],
            "定义与范围": ["定义", "指", "是指", "包括", "适用", "范围", "含义",
                          "definition", "means", "includes", "scope"],
        }

        # 保护力度评分关键词
        PROTECTION_STRONG = ["必须", "应当", "严禁", "禁止", "不得", "强制", "shall", "must", "prohibited"]
        PROTECTION_WEAK = ["可以", "可", "建议", "鼓励", "may", "should", "encourage"]

        def _normalize_jurisdiction(jur: str, law_id: str, default_jur: str) -> str:
            jur_map = {
                "us": "美国", "usa": "美国", "united states": "美国",
                "hk": "香港", "hong kong": "香港",
                "sg": "新加坡", "singapore": "新加坡",
                "mac": "澳门", "macau": "澳门", "mo": "澳门",
                "tw": "台湾", "taiwan": "台湾",
            }
            if jur:
                normalized = jur_map.get(jur.lower(), jur)
                if normalized != jur:
                    return normalized
            if not jur:
                lid = law_id or ""
                if any(lid.startswith(prefix) for prefix in ["MAC", "MO"]):
                    jur = "澳门"
                elif lid.startswith("SG"):
                    jur = "新加坡"
                elif lid.startswith("HK"):
                    jur = "香港"
                elif lid.startswith("TW"):
                    jur = "台湾"
                elif lid.startswith("US"):
                    jur = "美国"
            return jur or default_jur

        # ========== 1. 数据预处理与标签分配 ==========
        SIDE_LABELS = ['A', 'B', 'C', 'D']  # 最多支持4个法域
        all_chunks = []
        jurisdiction_side_map = {}  # 法域名 -> side label映射

        for idx, (jur_name, chunks) in enumerate(jurisdiction_chunks.items()):
            side_label = SIDE_LABELS[idx]
            jurisdiction_side_map[jur_name] = side_label

            for ch in chunks:
                jur = (ch.get("jurisdiction") or ch.get("_jurisdiction") or "").strip()
                jur = _normalize_jurisdiction(jur, ch.get("law_id", ""), jur_name)
                ch["_jurisdiction"] = jur
                ch["_side"] = side_label
                all_chunks.append({**ch})

        # ========== 2. 构建节点（支持N个法域）==========
        # 关键修复：同一法条可能有多个chunk（ES分块存储），需要按chunk_index排序后拼接完整内容
        chunk_groups = {}  # key -> list of chunks
        for ch in all_chunks:
            key = f"{ch.get('law_id','')}_{ch.get('article_number','')}"
            if not key or key.endswith("_"):
                continue
            if key not in chunk_groups:
                chunk_groups[key] = []
            chunk_groups[key].append(ch)

        nodes = []
        node_map = {}

        for key, group in chunk_groups.items():
            # 按chunk_index排序，确保内容顺序正确
            group.sort(key=lambda c: c.get("chunk_index", 0))
            # 取第一个chunk的元数据，拼接所有chunk的内容
            ch = group[0]
            content = "\n".join(c.get("content", "") for c in group if c.get("content", ""))
            node_id = f"node_{len(nodes)}"
            title = to_simplified(ch.get("title", ""))
            art = ch.get("article_number", "")
            law_name = title.split(" - ")[0] if " - " in title else title[:16]
            content_lower = content.lower()

            # 判断核心条款
            is_definition = any(kw in content_lower for kw in ["定义", "指", "是指", "包括", "meaning", "definition", "refers to"])
            is_penalty = any(kw in content_lower for kw in ["处罚", "罚款", "刑事责任", "penalty", "fine", "offence", "违例"])
            is_core = is_definition or is_penalty

            # 主题分类
            detected_themes = []
            for theme_name, kws in THEME_KEYWORDS.items():
                match_count = sum(1 for kw in kws if kw in content_lower)
                if match_count >= 1:
                    detected_themes.append(theme_name)
            primary_theme = detected_themes[0] if detected_themes else "其他"

            # 保护力度评分 (0-100)
            strong_count = sum(1 for kw in PROTECTION_STRONG if kw in content)
            weak_count = sum(1 for kw in PROTECTION_WEAK if kw in content)
            protection_score = min(100, max(20, 50 + strong_count * 15 - weak_count * 10))

            nodes.append({
                "id": node_id,
                "label": f"{art}" if art else f"{law_name[:8]}",
                "title": f"{title} - {art}",
                "full_title": title,
                "law_name": law_name,
                "jurisdiction": ch.get("_jurisdiction", ""),
                "_side": ch.get("_side", ""),
                "law_id": ch.get("law_id", ""),
                "article_number": art,
                "content": content,                    # 完整内容（用于详情展示）
                "content_preview": content[:500],      # 预览内容（适当增加长度）
                "is_core": is_core,
                "node_type": "definition" if is_definition else ("penalty" if is_penalty else "normal"),
                # ---- v6.0: 实体类型标注 ----
                "entity_type": self._classify_entity_type(content),
                "entity_label": self.ENTITY_TYPE_RULES.get(self._classify_entity_type(content), {}).get("label", "条款"),
                "entity_color": self.ENTITY_TYPE_RULES.get(self._classify_entity_type(content), {}).get("color", "#6B7280"),
                "entity_shape": self.ENTITY_TYPE_RULES.get(self._classify_entity_type(content), {}).get("shape", "ellipse"),
                "degree": 0,
                "cluster": ch.get("_side", ""),  # 用于布局分组
                "themes": detected_themes,
                "primary_theme": primary_theme,
                "protection_score": protection_score,
                "importance_score": 0,
            })
            node_map[key] = node_id

        # ========== 3. 构建边（多法域版本）==========
        edges = []
        edge_set = set()
        cross_edges = []

        # ===== 边类型1：同法规内关联 =====
        by_law = {}
        for i, ch in enumerate(all_chunks):
            lid = ch.get("law_id", "")
            if lid not in by_law:
                by_law[lid] = []
            by_law[lid].append(i)

        for lid, indices in by_law.items():
            sorted_idxs = sorted(indices, key=lambda idx: all_chunks[idx].get("chunk_index", 0))
            for j in range(len(sorted_idxs) - 1):
                ch1 = all_chunks[sorted_idxs[j]]
                ch2 = all_chunks[sorted_idxs[j + 1]]
                k1 = f"{ch1.get('law_id','')}_{ch1.get('article_number','')}"
                k2 = f"{ch2.get('law_id','')}_{ch2.get('article_number','')}"
                if k1 in node_map and k2 in node_map:
                    eid = tuple(sorted([node_map[k1], node_map[k2]]))
                    if eid not in edge_set:
                        edge_set.add(eid)
                        edges.append({
                            "from": eid[0], "to": eid[1],
                            "type": "same_law",
                            "label": "同法规关联",
                            "description": "同一法律中的连续或相关条款",
                            "weight": 2,
                            "style": "solid",
                            "color": "#BDC3C7",
                            "width": 1,
                        })

        # ===== 边类型2：跨法域语义关联（两两比较所有法域对）=====
        def extract_features(content, law_name, article_num):
            text = content or ""
            chars = set(re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text))
            text_clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
            bigrams = set(text_clean[i:i+2] for i in range(len(text_clean)-1)) if len(text_clean) > 1 else set()
            legal_keywords = set()
            for kw_list_item in ["数据", "个人", "信息", "保护", "隐私", "安全", "处罚",
                                 "责任", "义务", "权利", "同意", "披露", "传输", "存储",
                                 "处理", "收集", "使用", "删除", "更正", "访问", "泄露",
                                 "通知", "监管", "合规", "违规", "罚款", "刑事", "民事",
                                 "定义", "范围", "适用", "主体", "控制者", "当事人"]:
                if kw_list_item in text:
                    legal_keywords.add(kw_list_item)
            return {"chars": chars, "bigrams": bigrams, "keywords": legal_keywords}

        # 为每个法域预计算特征
        jurisdiction_features = {}
        for jur_name, chunks in jurisdiction_chunks.items():
            jurisdiction_features[jur_name] = [
                extract_features(ch.get("content",""), ch.get("title",""), ch.get("article_number",""))
                for ch in chunks
            ]

        # 两两比较所有法域对
        jurisdiction_pairs = [
            (jurisdictions[i], jurisdictions[j])
            for i in range(n_jurisdictions)
            for j in range(i + 1, n_jurisdictions)
        ]

        for jur_a, jur_b in jurisdiction_pairs:
            chunks_a = jurisdiction_chunks[jur_a]
            chunks_b = jurisdiction_chunks[jur_b]
            features_a = jurisdiction_features[jur_a]
            features_b = jurisdiction_features[jur_b]

            for i, (ch_a, feat_a) in enumerate(zip(chunks_a, features_a)):
                for j, (ch_b, feat_b) in enumerate(zip(chunks_b, features_b)):
                    scores = []

                    # 字符集重叠
                    if feat_a["chars"] and feat_b["chars"]:
                        char_overlap = len(feat_a["chars"] & feat_b["chars"]) / min(len(feat_a["chars"]), len(feat_b["chars"]))
                        scores.append(char_overlap * 0.8)

                    # 2-gram重叠
                    if feat_a["bigrams"] and feat_b["bigrams"]:
                        bg_overlap = len(feat_a["bigrams"] & feat_b["bigrams"]) / min(len(feat_a["bigrams"]), len(feat_b["bigrams"]))
                        scores.append(bg_overlap * 0.6)

                    # 法律关键词重叠
                    if feat_a["keywords"] and feat_b["keywords"]:
                        kw_overlap = len(feat_a["keywords"] & feat_b["keywords"]) / max(len(feat_a["keywords"]), len(feat_b["keywords"]))
                        scores.append(kw_overlap * 1.0)

                    final_score = sum(scores) / len(scores) if scores else 0

                    if final_score >= similarity_threshold:
                        ka = f"{ch_a.get('law_id','')}_{ch_a.get('article_number','')}"
                        kb = f"{ch_b.get('law_id','')}_{ch_b.get('article_number','')}"

                        if ka in node_map and kb in node_map:
                            na_id, nb_id = node_map[ka], node_map[kb]
                            eid = tuple(sorted([na_id, nb_id]))

                            if eid not in edge_set:
                                edge_set.add(eid)
                                edge_data = {
                                    "from": na_id, "to": nb_id,
                                    "type": "semantic_similar",
                                    "label": f"{final_score:.0%}",
                                    "description": f"{jur_a}↔{jur_b} 语义相似",
                                    "similarity_score": round(final_score, 4),
                                    "source_jurisdiction": jur_a,
                                    "target_jurisdiction": jur_b,
                                    "pair_label": f"{jur_a}↔{jur_b}",
                                    "weight": round(final_score * 5, 2),
                                    "style": "solid",
                                    "color": "#9B8EA3",
                                    "width": max(2, int(final_score * 5)),
                                }
                                edges.append(edge_data)
                                cross_edges.append(edge_data)

                                # 更新节点度数
                                nodes_dict = {n["id"]: n for n in nodes}
                                if na_id in nodes_dict:
                                    nodes_dict[na_id]["degree"] += 1
                                if nb_id in nodes_dict:
                                    nodes_dict[nb_id]["degree"] += 1

        # ===== 边类型3：主题关联边（可选增强）=====
        # （可根据需要添加基于topics字段交集的边）

        # ========== 4. 计算综合重要性评分 ==========
        if nodes:
            degrees = [n.get("degree", 0) for n in nodes]
            max_degree = max(degrees) if degrees else 1

            for node in nodes:
                degree_norm = node.get("degree", 0) / max(max_degree, 1)
                core_bonus = 0.3 if node.get("is_core") else 0
                prot_norm = node.get("protection_score", 50) / 100.0

                importance = (
                    degree_norm * 0.45 +
                    core_bonus * 0.25 +
                    prot_norm * 0.30
                )
                node["importance_score"] = round(importance, 4)

        # ========== 5. 构建邻接表 ==========
        adjacency = {}
        for e in edges:
            src, tgt = e["from"], e["to"]
            if src not in adjacency:
                adjacency[src] = set()
            if tgt not in adjacency:
                adjacency[tgt] = set()
            adjacency[src].add(tgt)
            adjacency[tgt].add(src)

        # ========== 6. 最短路径算法（BFS）==========
        def find_shortest_path(start, end):
            if start == end:
                return [start]
            queue = [[start]]
            visited = {start}
            while queue:
                path = queue.pop(0)
                node = path[-1]
                if node in adjacency:
                    for neighbor in sorted(adjacency[node]):
                        if neighbor == end:
                            return path + [neighbor]
                        if neighbor not in visited:
                            visited.add(neighbor)
                            new_path = path + [neighbor]
                            queue.append(new_path)
            return None

        # 预计算跨法域最短路径缓存
        path_cache = {}
        jurisdiction_node_groups = {}
        for node in nodes:
            jur = node.get("_side", "")
            if jur not in jurisdiction_node_groups:
                jurisdiction_node_groups[jur] = []
            jurisdiction_node_groups[jur].append(node["id"])

        # 两两计算不同法域间的最短路径
        sides = list(jurisdiction_node_groups.keys())
        for i in range(len(sides)):
            for j in range(i + 1, len(sides)):
                side_a, side_b = sides[i], sides[j]
                nodes_a = jurisdiction_node_groups[side_a][:8]  # 限制计算量
                nodes_b = jurisdiction_node_groups[side_b][:8]

                for na in nodes_a:
                    for nb in nodes_b:
                        p = find_shortest_path(na, nb)
                        if p and len(p) <= 7:
                            path_cache[f"{na}->{nb}"] = p

        # ========== 7. 中心节点子图过滤（可选）==========
        if center_law_id and center_law_id in [n["law_id"] for n in nodes]:
            center_nodes = {n["id"] for n in nodes if n["law_id"] == center_law_id}
            neighbor_ids = set(center_nodes)
            for e in edges:
                if e["from"] in center_nodes:
                    neighbor_ids.add(e["to"])
                if e["to"] in center_nodes:
                    neighbor_ids.add(e["from"])
            nodes = [n for n in nodes if n["id"] in neighbor_ids]
            edges = [e for e in edges if e["from"] in neighbor_ids and e["to"] in neighbor_ids]

        # ========== 7b. [v6.0] 抗杂乱过滤 ==========
        MAX_NODES = 40
        MAX_EDGES_PER_NODE = 8

        # 记录原始节点数（供前端展示截断提示）—— 保留完整数据由前端控制显示
        _raw_node_count = len(nodes)

        # 7b.1: 节点数限制（已换由前端控制，仅统计标记不断）
        # 7b.2: 每节点最大边数限制（保留权重最高的）
        node_edge_count = {}
        edge_priority = []  # (edge_idx, min_priority)
        for idx, e in enumerate(edges):
            src_pri = next((n.get("importance_score", 0) for n in nodes if n["id"] == e["from"]), 0)
            tgt_pri = next((n.get("importance_score", 0) for n in nodes if n["id"] == e["to"]), 0)
            min_pri = min(src_pri, tgt_pri)
            edge_priority.append((idx, min_pri))

        filtered_edges = []
        for e in edges:
            src = e["from"]
            tgt = e["to"]
            if node_edge_count.get(src, 0) >= MAX_EDGES_PER_NODE and node_edge_count.get(tgt, 0) >= MAX_EDGES_PER_NODE:
                continue
            filtered_edges.append(e)
            node_edge_count[src] = node_edge_count.get(src, 0) + 1
            node_edge_count[tgt] = node_edge_count.get(tgt, 0) + 1
        edges = filtered_edges

        # 7b.3: 给边添加语义关系类型
        for e in edges:
            # 查找边对应的源节点内容
            src_node = next((n for n in nodes if n["id"] == e["from"]), None)
            tgt_node = next((n for n in nodes if n["id"] == e["to"]), None)
            src_content = (src_node.get("content", "") or "") if src_node else ""
            tgt_content = (tgt_node.get("content", "") or "") if tgt_node else ""

            if e.get("type") == "same_law":
                e["relation_type"] = "same_law"
                e["relation_label"] = "同法规"
                e["relation_color"] = "#94A3B8"
            elif e.get("type") == "semantic_similar":
                # 检测两条法条之间的语义关系类型
                src_rel = self._detect_relation_type(src_content)
                tgt_rel = self._detect_relation_type(tgt_content)
                # 取更具体的关系类型
                if src_rel and tgt_rel:
                    e["relation_type"] = src_rel if src_rel != tgt_rel else src_rel
                elif src_rel:
                    e["relation_type"] = src_rel
                elif tgt_rel:
                    e["relation_type"] = tgt_rel
                else:
                    e["relation_type"] = "cross_reference"
                rel_info = self.RELATION_TYPE_RULES.get(e["relation_type"], {})
                e["relation_label"] = rel_info.get("label", "关联")
                e["relation_color"] = rel_info.get("color", "#9B8EA3")

        # ========== 8. 统计信息汇总 ==========
        # 实体类型分布
        entity_type_distribution = {}
        for n in nodes:
            et = n.get("entity_type", "Article")
            entity_type_distribution[et] = entity_type_distribution.get(et, 0) + 1

        stats = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "cross_jurisdiction_edges": len([e for e in edges if e.get("type") == "semantic_similar"]),
            "same_law_edges": len([e for e in edges if e.get("type") == "same_law"]),
            "core_nodes": len([n for n in nodes if n.get("is_core")]),
            "max_degree": max((n.get("degree", 0) for n in nodes), default=0),
            "avg_degree": round(sum(n.get("degree", 0) for n in nodes) / max(len(nodes), 1), 1),
            "jurisdictions": jurisdictions,
            "n_jurisdictions": n_jurisdictions,
            "themes_available": list(set(n.get("primary_theme", "") for n in nodes)),
            "has_cross_edges": len([e for e in edges if e.get("type") == "semantic_similar"]) > 0,
            "path_cache_size": len(path_cache),
            "jurisdiction_pairs": [
                f"{p[0]} ↔ {p[1]}" for p in jurisdiction_pairs
            ],
            "cross_pair_count": len(jurisdiction_pairs),
            # [v6.0] 新增统计
            "entity_type_distribution": entity_type_distribution,
            "node_types_available": list(entity_type_distribution.keys()),
            "max_nodes_limit": MAX_NODES,
            "max_edges_per_node": MAX_EDGES_PER_NODE,
            "layout_config": {
                "n_clusters": n_jurisdictions,
                "cluster_labels": [SIDE_LABELS[i] for i in range(n_jurisdictions)],
                "cluster_names": jurisdictions,
            }
        }

        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "raw_node_count": _raw_node_count,
            "stats": stats,
            "adjacency": adjacency,
            "path_cache": path_cache,
            "has_cross_edges": len([e for e in edges if e.get("type") == "semantic_similar"]) > 0,
            "jurisdiction_side_map": jurisdiction_side_map,
            "schema_version": "6.0"
        }

    # ================================================================
    #  最短路径分析（独立功能模块）
    # ================================================================

    def find_shortest_path(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        source_id: str,
        target_id: str,
        max_depth: int = 6,
        max_visits: int = 5000
    ) -> Dict:
        """
        BFS最短路径查找算法（增强版）

        在知识图谱中查找两个节点之间的最短路径，返回完整的路径信息，
        包括路径上的每个节点、每条边及其关系详情。

        Args:
            nodes: 图谱节点列表
            edges: 图谱边列表
            source_id: 起始节点ID
            target_id: 目标节点ID
            max_depth: 最大搜索深度（默认6跳）
            max_visits: 最大访问节点数上限（防止性能问题，默认5000）

        Returns:
            {
                "found": bool,               # 是否找到路径
                "path_nodes": [Dict],         # 路径上的节点列表
                "path_edges": [Dict],         # 路径上的边列表（含关系详情）
                "path_length": int,           # 路径长度（节点数）
                "hops": int,                  # 跳数（边数）
                "depth_reached": int,         # 实际搜索深度
                "nodes_visited": int,         # 实际访问节点数
                "path_description": str,      # 路径文字描述
                "error": str | None           # 错误信息（如有）
            }
        """
        # ---- 边界情况检查 ----
        if not nodes or not edges:
            return {
                "found": False, "path_nodes": [], "path_edges": [],
                "path_length": 0, "hops": 0, "depth_reached": 0,
                "nodes_visited": 0, "path_description": "",
                "error": "图谱数据为空，无法执行路径搜索"
            }

        # 同一节点检查
        if source_id == target_id:
            node = next((n for n in nodes if n["id"] == source_id), None)
            label = f"第{node.get('article_number', '?')}条" if node else source_id
            return {
                "found": True, "path_nodes": [node] if node else [],
                "path_edges": [], "path_length": 1, "hops": 0,
                "depth_reached": 0, "nodes_visited": 1,
                "path_description": f"起点与终点相同：{label}",
                "error": None
            }

        # 节点存在性检查
        valid_ids = {n["id"] for n in nodes}
        if source_id not in valid_ids:
            return {
                "found": False, "path_nodes": [], "path_edges": [],
                "path_length": 0, "hops": 0, "depth_reached": 0,
                "nodes_visited": 0, "path_description": "",
                "error": f"起始节点 '{source_id}' 不在图谱中"
            }
        if target_id not in valid_ids:
            return {
                "found": False, "path_nodes": [], "path_edges": [],
                "path_length": 0, "hops": 0, "depth_reached": 0,
                "nodes_visited": 0, "path_description": "",
                "error": f"目标节点 '{target_id}' 不在图谱中"
            }

        # ---- 构建邻接表（含边信息）----
        # adjacency[node_id] = [(neighbor_id, edge_dict), ...]
        adjacency: Dict[str, list] = {n["id"]: [] for n in nodes}
        edge_map: Dict[tuple, Dict] = {}
        for e in edges:
            src, tgt = e["from"], e["to"]
            if src in adjacency and tgt in adjacency:
                adjacency[src].append((tgt, e))
                adjacency[tgt].append((src, e))
                edge_key = tuple(sorted([src, tgt]))
                if edge_key not in edge_map:
                    edge_map[edge_key] = e

        # ---- BFS最短路径搜索 ----
        from collections import deque

        visited = set()
        queue = deque()
        queue.append(([source_id], 0))  # (当前路径, 当前深度)
        visited.add(source_id)
        nodes_visited = 1
        best_path = None

        while queue and nodes_visited < max_visits:
            current_path, depth = queue.popleft()

            # 超过最大深度，跳过
            if depth >= max_depth:
                continue

            current_node = current_path[-1]

            # 找到目标！
            if current_node == target_id:
                best_path = current_path
                break

            # 遍历邻居
            for neighbor, _edge_info in adjacency.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    nodes_visited += 1
                    queue.append((current_path + [neighbor], depth + 1))

        # ---- 构建结果 ----
        if not best_path:
            from_node = next((n for n in nodes if n["id"] == source_id), None)
            to_node = next((n for n in nodes if n["id"] == target_id), None)
            from_label = self._format_node_label(from_node) if from_node else source_id
            to_label = self._format_node_label(to_node) if to_node else target_id

            return {
                "found": False, "path_nodes": [], "path_edges": [],
                "path_length": 0, "hops": 0,
                "depth_reached": min(depth, max_depth) if 'depth' in dir() else max_depth,
                "nodes_visited": nodes_visited,
                "path_description": "",
                "error": (
                    f"在 {max_depth} 跳内未发现「{from_label}」与「{to_label}」之间的关联路径。"
                    f"\n可能原因：两个条款分属不同主题领域，或图谱中缺少中间连接节点。"
                    f"\n建议：尝试扩大搜索深度，或选择主题更相近的条款。"
                )
            }

        # 提取路径上的节点和边
        path_nodes = []
        path_edges = []
        for i, node_id in enumerate(best_path):
            node = next((n for n in nodes if n["id"] == node_id), None)
            if node:
                path_nodes.append(node)

            # 提取相邻节点之间的边
            if i < len(best_path) - 1:
                next_id = best_path[i + 1]
                edge_key = tuple(sorted([node_id, next_id]))
                edge_info = edge_map.get(edge_key)
                if edge_info:
                    path_edges.append({**edge_info, "direction": f"{node_id} → {next_id}"})

        # 生成路径文字描述
        path_description = self._build_path_description(path_nodes, path_edges)

        return {
            "found": True,
            "path_nodes": path_nodes,
            "path_edges": path_edges,
            "path_length": len(path_nodes),
            "hops": len(path_edges),
            "depth_reached": len(best_path) - 1,
            "nodes_visited": nodes_visited,
            "path_description": path_description,
            "error": None
        }

    def interpret_path(self, path_result: Dict) -> str:
        """
        调用LLM对最短路径结果进行自然语言解读

        将路径上的节点和关系转化为一句通俗易懂的法律含义解释。

        Args:
            path_result: find_shortest_path() 返回的结果字典

        Returns:
            自然语言解读文本
        """
        if not path_result.get("found") or not path_result.get("path_nodes"):
            return "无法解读：未找到有效路径。"

        nodes = path_result["path_nodes"]
        edges = path_result.get("path_edges", [])

        # 构建路径信息供LLM解读
        path_segments = []
        for i, node in enumerate(nodes):
            jur = node.get("jurisdiction", "")
            law = node.get("law_name", "")
            art = node.get("article_number", "?")
            theme = node.get("primary_theme", "")
            content_preview = (node.get("content_preview", "") or "")[:200]

            segment = f"节点{i+1}：{jur}《{law}》第{art}条（主题：{theme}）\n内容摘要：{content_preview}"
            path_segments.append(segment)

            # 添加边信息
            if i < len(edges):
                edge = edges[i]
                rel_type = edge.get("relation_type", edge.get("type", "关联"))
                label = edge.get("label", "")
                sim = edge.get("similarity")
                sim_str = f"，相似度{sim:.0%}" if sim else ""
                path_segments.append(f"  ↓ [{rel_type}] {label}{sim_str}")

        path_text = "\n".join(path_segments)

        # 首尾节点信息
        first_node = nodes[0]
        last_node = nodes[-1]
        hops = path_result.get("hops", 0)

        prompt = (
            "你是一位资深的比较法研究专家。\n\n"
            "以下是在跨法域法规知识图谱中找到的一条连接两个法条的路径：\n\n"
            f"{path_text}\n\n"
            f"这条路径从 {first_node.get('jurisdiction', '')}《{first_node.get('law_name', '')}"
            f"第{first_node.get('article_number', '?')}条 出发，经过 {hops} 步到达 "
            f"{last_node.get('jurisdiction', '')}《{last_node.get('law_name', '')}"
            f"第{last_node.get('article_number', '?')}条。\n\n"
            "请用一段话（100-200字）总结这条路径揭示的法律关联含义：\n"
            "- 说明这两个法条通过什么概念/主题链条连接起来\n"
            "- 指出这种关联的强度（强/中/弱）\n"
            "- 给出实务层面的启示（如合规建议、风险提示等）\n\n"
            "请直接输出解读文字，不要加前缀标题。"
        )

        try:
            # 使用 qwen3.6-plus 进行知识图谱路径解读，保证推理和输出质量
            extraction_model = settings.llm_extraction_model or "qwen3.6-plus"
            interpretation = self.qa_service._call_llm_with_model(prompt, model=extraction_model)
            return interpretation or "AI解读生成失败，请稍后重试。"
        except Exception as e:
            logger.error(f"[Path-Interpret] LLM调用失败: {e}")
            return f"解读生成异常: {str(e)}"

    @staticmethod
    def _format_node_label(node: Dict) -> str:
        """格式化节点的可读标签"""
        if not node:
            return "未知节点"
        jur = node.get("jurisdiction", "")
        art = node.get("article_number", "?")
        law = node.get("law_name", "")
        return f"{jur}_{law}_第{art}条"

    @staticmethod
    def _build_path_description(path_nodes: List[Dict], path_edges: List[Dict]) -> str:
        """
        构建路径的文字描述

        例如：澳门_PDPL_11 → (语义相似, 0.92) → 新加坡_PDPA_22 → (主题关联) → 新加坡_PDPA_15
        """
        if not path_nodes:
            return ""

        parts = []
        for i, node in enumerate(path_nodes):
            jur = node.get("jurisdiction", "")
            art = node.get("article_number", "?")
            label = f"{jur}_第{art}条"

            if i == 0:
                parts.append(label)
            else:
                # 前一条边的信息
                edge = path_edges[i - 1] if i <= len(path_edges) else None
                if edge:
                    rel_type = edge.get("relation_type", edge.get("type", "关联"))
                    edge_label = edge.get("label", "")
                    sim = edge.get("similarity")
                    sim_str = f", {sim:.2f}" if sim is not None else ""
                    edge_desc = f"({edge_label or rel_type}{sim_str})"
                else:
                    edge_desc = "(关联)"

                parts.append(f"{edge_desc}\n   ↓\n   {label}")

        return "\n".join(parts)
