"""Agent 工具集：把现有检索 / 引用 / 图谱能力封装为 LLM 可调用的函数工具。

设计要点：
  - **只封装、不改造**：三个执行器全部转调既有服务（RetrieverService / repo /
    QAService），不新增检索能力，因此工具本身无回归风险；
  - **成本可控**：search_laws 固定 `_rerank=False`——本机精排占检索延迟约 93%，
    探索步走 RRF 顺序换取速度，精排由 AgentService 在「最终证据集」上统一做一次；
  - **同请求内去重**：相同参数的工具调用在单次提问内命中内存 memo，不重复打 ES；
  - **失败不抛出**：任何异常都转成 `{"error": ...}` 观察值回灌 LLM，让模型自行
    决定重试或换策略，绝不中断探索循环。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from core.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# 工具 JSON Schema（OpenAI function-calling 格式）
# ============================================================
SEARCH_LAWS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_laws",
        "description": (
            "按关键词/语义检索法规条文，返回最相关的条文（含法域、法规名、条号、正文片段）。"
            "适合先定位与问题相关的法条；可指定法域缩小范围。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索查询，建议使用法律术语，可包含法域、主题词"},
                "jurisdiction": {
                    "type": "string",
                    "description": "可选，限定法域（如 澳门 / 香港 / 新加坡 / 日本 / 美国 / 马来西亚 / 泰国 / 越南）",
                },
                "top_k": {"type": "integer", "description": "返回条数，默认 5，最大 10"},
            },
            "required": ["query"],
        },
    },
}

FOLLOW_REFERENCES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "follow_references",
        "description": (
            "追溯某条款正文中引用到的其他条款（同法规内「第X条」及可解析的跨法规引用），"
            "用于补全被引用规则的具体内容。参数取自 search_laws 结果中的 law_id 与 article_number。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "law_id": {"type": "string", "description": "引用方条款所属法规 ID"},
                "article_number": {"type": "string", "description": "引用方条款号（如 2）"},
            },
            "required": ["law_id", "article_number"],
        },
    },
}

QUERY_GRAPH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_graph",
        "description": (
            "查询法律知识图谱：返回与问题相关的实体关系（义务/权利/处罚/定义/例外等）"
            "及来源条款摘录，适合需要跨条文关联推理时使用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "需要查询关联关系的问题或主题"},
            },
            "required": ["question"],
        },
    },
}

TOOL_SCHEMAS: List[Dict] = [SEARCH_LAWS_SCHEMA, FOLLOW_REFERENCES_SCHEMA, QUERY_GRAPH_SCHEMA]

# 允许的参数名白名单：防止 LLM 传入未知字段导致 **kwargs 报错
_ALLOWED_ARGS = {
    "search_laws": {"query", "jurisdiction", "top_k"},
    "follow_references": {"law_id", "article_number"},
    "query_graph": {"question"},
}


class AgentTools:
    """工具执行器：持有检索服务与 QAService（延迟注入，避免构造期重活）。"""

    def __init__(self, retriever, qa_service_provider, jurisdictions=None):
        """
        Args:
            retriever: RetrieverService 实例
            qa_service_provider: 无参可调用对象，返回 QAService（用于复用图谱证据收集）
            jurisdictions: 本次提问已识别的法域（中文名列表）。用于把图谱证据限定在
                相关法域内，避免跨法域无关证据（如问泰国金融却注入美国 HIPAA、问新加坡
                电子交易却注入澳门法规）污染答案。None/空表示未识别，图谱证据不过滤。
        """
        self.retriever = retriever
        self._qa_provider = qa_service_provider
        self._jurisdictions = list(jurisdictions) if jurisdictions else None
        # 单次提问内的 memo：键为 (tool, 规范化参数) → 结果，避免重复外部调用。
        # 本执行器按请求新建（见 AgentService.ask_stream），memo 天然是请求级、
        # 无需额外 reset，也不会被并发请求串用。
        self._memo: Dict[str, Dict] = {}

    # ---------------- 对外入口 ----------------
    def execute(self, name: str, args: Optional[Dict[str, Any]]) -> Dict:
        """执行工具；异常一律转成 error 观察值，不向上抛出。"""
        safe_args = {
            k: v for k, v in (args or {}).items() if k in _ALLOWED_ARGS.get(name, set())
        }
        memo_key = f"{name}:{json.dumps(safe_args, ensure_ascii=False, sort_keys=True)}"
        if memo_key in self._memo:
            cached = dict(self._memo[memo_key])
            cached["_memo"] = True  # 提示 LLM/日志：该结果本次提问内已查过
            return cached

        try:
            if name == "search_laws":
                result = self._search_laws(**safe_args)
            elif name == "follow_references":
                result = self._follow_references(**safe_args)
            elif name == "query_graph":
                result = self._query_graph(**safe_args)
            else:
                result = {"error": f"未知工具: {name}"}
        except Exception as e:  # 工具失败不影响循环，回灌错误让模型自行决策
            logger.warning("[Agent] 工具 %s 执行失败: %s: %s", name, type(e).__name__, e)
            result = {"error": f"{type(e).__name__}: {e}"}

        self._memo[memo_key] = result
        return result

    # ---------------- 具体工具实现 ----------------
    def _search_laws(self, query: str = "", jurisdiction: Optional[str] = None,
                     top_k: Optional[int] = None) -> Dict:
        """检索法规条文。固定跳过精排（`_rerank=False`）以换取速度。"""
        q = (query or "").strip()
        if not q:
            return {"error": "query 不能为空", "count": 0, "results": []}
        k = settings.agent_tool_top_k
        try:
            k = min(int(top_k), 10) if top_k else settings.agent_tool_top_k
        except (TypeError, ValueError):
            k = settings.agent_tool_top_k
        k = max(1, k)

        results = self.retriever.hybrid_search(
            query=q,
            top_k=k,
            jurisdiction=jurisdiction or None,
            _decompose=True,
            _rerank=False,  # 探索步跳过精排：精排留给最终证据集统一做一次
        )
        out = []
        for r in results[:k]:
            out.append({
                "law_id": r.get("law_id", ""),
                "title": r.get("title", ""),
                "article_number": r.get("article_number", ""),
                "jurisdiction": r.get("jurisdiction", ""),
                "content": (r.get("content", "") or "")[: settings.agent_obs_max_chars],
                "score": round(float(r.get("score") or 0.0), 4),
            })
        return {"count": len(out), "results": out}

    def _follow_references(self, law_id: str = "", article_number: str = "") -> Dict:
        """追溯条款引用：先取引用方原文，再解析并回取被引条款。"""
        law_id = (law_id or "").strip()
        article_number = str(article_number or "").strip()
        if not law_id or not article_number:
            return {"error": "law_id 与 article_number 均不能为空", "count": 0, "referenced": []}

        content = self.retriever.repo.get_article_full_text(law_id, article_number)
        if not content:
            return {"count": 0, "referenced": [], "note": "未找到该条款原文，无法追溯引用"}

        docs = self.retriever.repo.get_referenced_articles(
            [(law_id, article_number, content)],
            settings.reference_expand_max,
        )
        out = []
        for d in docs:
            out.append({
                "law_id": d.get("law_id", ""),
                "title": d.get("title", ""),
                "article_number": d.get("article_number", ""),
                "jurisdiction": d.get("jurisdiction", ""),
                "referenced_by": d.get("referenced_by", ""),
                "cross_law": bool(d.get("cross_law")),
                "content": (d.get("content", "") or "")[: settings.agent_obs_max_chars],
            })
        return {"count": len(out), "referenced": out}

    def _query_graph(self, question: str = "") -> Dict:
        """查询知识图谱关联（复用 QAService 的图谱证据收集，含条款摘录）。"""
        q = (question or "").strip()
        if not q:
            return {"error": "question 不能为空", "count": 0, "evidence": ""}
        qa = self._qa_provider()
        text = qa._collect_kg_evidence(q, self._jurisdictions)
        if not text:
            return {"count": 0, "evidence": "", "note": "知识图谱中未找到与该问题相关的实体关系"}
        return {"count": 1, "evidence": text[:1500]}
