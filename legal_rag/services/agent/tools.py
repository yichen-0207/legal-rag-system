"""Agent 工具集：把现有检索 / 引用 / 图谱能力封装为 LangChain 工具。

设计要点：
  - **框架声明工具**：用 `StructuredTool` + pydantic 参数模型声明工具，工具 schema
    （名称 / 描述 / 参数类型）由类型自动生成，不再手写 JSON Schema 与参数白名单——
    手写版既冗余又易与执行器签名漂移；
  - **只封装、不改造**：三个执行器全部转调既有服务（RetrieverService / repo /
    QAService），不新增检索能力，因此工具本身无回归风险；
  - **成本可控**：search_laws 固定 `_rerank=False`——本机精排占检索延迟约 93%，
    探索步走 RRF 顺序换取速度，精排由 AgentService 在「最终证据集」上统一做一次；
  - **请求级隔离**：工具运行时持有「单次提问」的去重 memo 与法域，必须按请求新建；
    若挂在进程级单例上，并发请求会互相清空/串用对方的缓存；
  - **失败不抛出**：任何异常都转成 `{"error": ...}` 观察值回灌 LLM，让模型自行
    决定重试或换策略，绝不中断探索循环。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from core.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# 工具参数模型（schema 由 pydantic 自动生成，字段描述即工具选择依据）
# ============================================================
class SearchLawsArgs(BaseModel):
    """检索法规条文的入参。"""

    query: str = Field(..., description="检索查询，建议使用法律术语，可包含法域、主题词")
    jurisdiction: Optional[str] = Field(
        None,
        description="可选，限定法域（如 澳门 / 香港 / 新加坡 / 日本 / 美国 / 马来西亚 / 泰国 / 越南）",
    )
    top_k: Optional[int] = Field(None, description="返回条数，默认 5，最大 10")


class FollowReferencesArgs(BaseModel):
    """追溯条款引用的入参。"""

    law_id: str = Field(..., description="引用方条款所属法规 ID")
    article_number: str = Field(..., description="引用方条款号（如 2）")


class QueryGraphArgs(BaseModel):
    """查询知识图谱的入参。"""

    question: str = Field(..., description="需要查询关联关系的问题或主题")


# LangChain 工具需要字符串化的观测值；此处统一用 JSON 文本，
# AgentService 消费 ToolMessage 时再解析回 dict 以累积证据（见 ask_stream）。
_SEARCH_LAWS_DESC = (
    "按关键词/语义检索法规条文，返回最相关的条文（含法域、法规名、条号、正文片段）。"
    "适合先定位与问题相关的法条；可指定法域缩小范围。"
)
_FOLLOW_REFERENCES_DESC = (
    "追溯某条款正文中引用到的其他条款（同法规内「第X条」及可解析的跨法规引用），"
    "用于补全被引用规则的具体内容。参数取自 search_laws 结果中的 law_id 与 article_number。"
)
_QUERY_GRAPH_DESC = (
    "查询法律知识图谱：返回与问题相关的实体关系（义务/权利/处罚/定义/例外等）"
    "及来源条款摘录，适合需要跨条文关联推理时使用。"
)


class AgentToolkit:
    """单次提问的工具运行时：持有检索服务、请求级去重 memo 与已识别法域。

    生命周期与一次提问一致（AgentService.ask_stream 内构造），因此 memo 天然是
    请求级、并发请求之间互相隔离，无需加锁或显式清理。
    """

    def __init__(self, retriever, qa_service_provider: Callable[[], Any],
                 jurisdictions: Optional[List[str]] = None):
        """
        Args:
            retriever: RetrieverService 实例。
            qa_service_provider: 无参可调用对象，返回 QAService（用于复用图谱证据收集）。
            jurisdictions: 本次提问已识别的法域（中文名列表）。用于把图谱证据限定在
                相关法域内，避免跨法域无关证据（如问泰国金融却注入美国 HIPAA、问新加坡
                电子交易却注入澳门法规）污染答案。None/空表示未识别，图谱证据不过滤。
        """
        self.retriever = retriever
        self._qa_provider = qa_service_provider
        self._jurisdictions = list(jurisdictions) if jurisdictions else None
        # 键为「工具名 + 规范化参数」→ 结果，避免同一参数在单次提问内重复打 ES。
        self._memo: Dict[str, Dict] = {}

    # ---------------- 对外入口 ----------------
    def build_tools(self) -> List[BaseTool]:
        """把三个执行器包装为 LangChain 工具（各自绑定本请求的 memo 与法域）。"""
        return [
            self._as_tool("search_laws", _SEARCH_LAWS_DESC, SearchLawsArgs, self._search_laws),
            self._as_tool("follow_references", _FOLLOW_REFERENCES_DESC,
                          FollowReferencesArgs, self._follow_references),
            self._as_tool("query_graph", _QUERY_GRAPH_DESC, QueryGraphArgs, self._query_graph),
        ]

    def _as_tool(self, name: str, description: str, args_schema: type[BaseModel],
                 impl: Callable[..., Dict]) -> BaseTool:
        """把执行器包装成工具；返回值为 JSON 文本（LangChain 工具需字符串观测值）。"""

        def _run(**kwargs: Any) -> str:
            return json.dumps(self._execute(name, impl, kwargs), ensure_ascii=False)

        return StructuredTool.from_function(
            name=name,
            description=description,
            args_schema=args_schema,
            func=_run,
        )

    def _execute(self, name: str, impl: Callable[..., Dict], args: Dict[str, Any]) -> Dict:
        """执行工具：命中请求级 memo 直接复用；异常一律转成 error 观察值，不向上抛出。"""
        memo_key = f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
        if memo_key in self._memo:
            cached = dict(self._memo[memo_key])
            cached["_memo"] = True  # 提示 LLM/日志：该结果本次提问内已查过
            return cached

        try:
            result = impl(**args)
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
