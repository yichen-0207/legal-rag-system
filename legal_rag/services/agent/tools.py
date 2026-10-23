"""Agent 工具集：把现有检索 / 引用 / 图谱能力封装为 LangChain 工具。

用 `StructuredTool` + pydantic 声明工具（schema 由类型自动生成）；只封装、不改造：
前三个执行器转调既有服务，get_evidence_status 只读台账不触发检索；search_laws 固定
`_rerank=False`，精排由 AgentService 对最终证据集统一做一次；工具运行时按请求新建以
保证 memo 并发隔离；任何异常都转成 `{"error": ...}` 观察值回灌 LLM，绝不中断探索循环。
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Dict, List, Literal, Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from core.config import settings

logger = logging.getLogger(__name__)

# 单工具超时的共享执行池（工具本身无状态，无需请求级隔离）。max_workers=8 与 ES msearch
# 并发上限同量级，可承接 ToolNode 的一轮并发扇出；超时只解除主流程等待，被卡线程靠底层
# HTTP 客户端的 socket 超时（秒级~30s）自行回收，8 是留出余量后的取值。
_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="agent-tool")


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
    """查询知识图谱多跳路径的入参。"""

    question: str = Field(..., description="需要查询关联关系的问题或主题")
    mode: Literal["auto", "cross", "entity"] = Field(
        "auto",
        description=(
            "auto=按问题自动选择；cross=跨法域对齐链（源条款→对应法规→目标条款）；"
            "entity=法域内实体关系多跳路径"
        ),
    )
    deepen: Optional[bool] = Field(
        None,
        description="仅 entity 模式有效：是否从路径末端纵向深挖 1~2 轮（默认关闭）",
    )


class EvidenceStatusArgs(BaseModel):
    """证据覆盖状态查询：无入参（只读，返回当前证据池的覆盖台账）。"""
    pass


# 工具观测值统一为 JSON 文本；AgentService 消费 ToolMessage 时再解析回 dict 累积证据。
_SEARCH_LAWS_DESC = (
    "按关键词/语义检索法规条文，返回最相关的条文（含法域、法规名、条号、正文片段）。"
    "适合先定位与问题相关的法条；可指定法域缩小范围。"
)
_FOLLOW_REFERENCES_DESC = (
    "追溯某条款正文中引用到的其他条款（同法规内「第X条」及可解析的跨法规引用），"
    "用于补全被引用规则的具体内容。参数取自 search_laws 结果中的 law_id 与 article_number。"
)
_QUERY_GRAPH_DESC = (
    "查询法律知识图谱的多跳路径（只返回系统校验过的路径与条款证据，不生成答案）。"
    "mode=cross 做跨法域对齐：源条款→对应法规→目标条款，并给出可靠性等级"
    "（L1 主题相似 / L2 同一概念）；mode=entity 做法域内实体关系多跳，"
    "可设 deepen=true 纵向深挖 1~2 轮；mode=auto 由系统按问题自动选择。"
    "适合需要跨条文、跨法域关联推理时使用。"
)

# 跨步工作记忆入口：四个工具中唯一只读、不改变证据池的一个，供模型自评覆盖与缺口。
EVIDENCE_STATUS_TOOL = "get_evidence_status"
_EVIDENCE_STATUS_DESC = (
    "查询当前已收集证据的覆盖情况：已覆盖的法域及条款数、仍缺口的目标法域、"
    "已检索过的条文清单。在判断「证据是否已足够、还需补充哪个法域」时调用，无参数。"
)

# auto 模式下判定是否优先走跨法域链：多法域/对比类信号；识别到 ≥2 个法域时直接走 cross。
_CROSS_GRAPH_RE = re.compile(
    r"跨境|跨法域|各法域|不同法域|多地|各国|海外|境外|域外|对比|比较|区别|差异"
)
# 单次 query_graph 返回给 LLM 的路径条数与证据字数上限（控制观测值膨胀）
_GRAPH_MAX_PATHS = 8
_GRAPH_MAX_EVIDENCE_CHARS = 1600


class AgentToolkit:
    """单次提问的工具运行时：持有检索服务、请求级去重 memo 与已识别法域。

    生命周期与一次提问一致（在 AgentService.ask_stream 内构造），memo 天然请求级隔离。
    """

    def __init__(self, retriever, qa_service_provider: Callable[[], Any],
                 jurisdictions: Optional[List[str]] = None,
                 evidence: Optional[Dict[str, Dict]] = None):
        """Args: retriever=RetrieverService；qa_service_provider=返回 QAService 的无参可调用；
        jurisdictions=本次识别法域（限定图谱证据范围，None 表示不过滤）；evidence=AgentService
        的证据池引用（get_evidence_status 据此实时汇报，传引用才能反映增量）。
        """
        self.retriever = retriever
        self._qa_provider = qa_service_provider
        self._jurisdictions = list(jurisdictions) if jurisdictions else None
        self._evidence_ref = evidence
        # 键为「工具名 + 规范化参数」→ 结果，避免同一参数在单次提问内重复打 ES。
        self._memo: Dict[str, Dict] = {}
        # 图谱多跳服务按请求惰性构造（复用 retriever 的 ES 客户端），失败负缓存避免每轮重试。
        self._multihop = None
        self._kg_reasoning = None

    # ---------------- 对外入口 ----------------
    def build_tools(self) -> List[BaseTool]:
        """把四个执行器包装为 LangChain 工具（各自绑定本请求的 memo 与法域）。"""
        return [
            self._as_tool("search_laws", _SEARCH_LAWS_DESC, SearchLawsArgs, self._search_laws),
            self._as_tool("follow_references", _FOLLOW_REFERENCES_DESC,
                          FollowReferencesArgs, self._follow_references),
            self._as_tool("query_graph", _QUERY_GRAPH_DESC, QueryGraphArgs, self._query_graph),
            self._as_tool(EVIDENCE_STATUS_TOOL, _EVIDENCE_STATUS_DESC,
                          EvidenceStatusArgs, self._evidence_status),
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
        """执行工具：命中请求级 memo 直接复用；异常/超时一律转成 error 观察值，不向上抛出。

        超时保护：把 impl 提交到共享线程池并限时等待，单工具卡死不再拖累端到端预算。
        成功与必然性失败（参数错/ES 拒绝）写 memo；超时不写（偶发抖动重试可能成功）。
        """
        # get_evidence_status 返回值随探索实时变化，缓存会返回过期台账，故不做 memo
        # （纯内存操作，也无需超时保护）。
        memoizable = name != EVIDENCE_STATUS_TOOL
        memo_key = f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
        if memoizable and memo_key in self._memo:
            cached = dict(self._memo[memo_key])
            cached["_memo"] = True  # 提示 LLM/日志：该结果本次提问内已查过
            return cached

        write_memo = True
        if memoizable:
            future = _TOOL_EXECUTOR.submit(impl, **args)
            try:
                result = future.result(timeout=settings.agent_tool_timeout)
            except FuturesTimeoutError:
                future.cancel()
                logger.warning("[Agent] 工具 %s 超时（%.1fs），转 error 观察值",
                               name, settings.agent_tool_timeout)
                result = {"error": f"timeout: 超过 {settings.agent_tool_timeout}s 未返回"}
                write_memo = False  # 超时不固化，保留重试机会
            except Exception as e:  # 工具失败不影响循环，回灌错误让模型自行决策
                logger.warning("[Agent] 工具 %s 执行失败: %s: %s", name, type(e).__name__, e)
                result = {"error": f"{type(e).__name__}: {e}"}
        else:
            try:
                result = impl(**args)
            except Exception as e:
                logger.warning("[Agent] 工具 %s 执行失败: %s: %s", name, type(e).__name__, e)
                result = {"error": f"{type(e).__name__}: {e}"}

        if memoizable and write_memo:
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

    # ---------------- 图谱多跳 ----------------
    def _get_multihop(self):
        """惰性构造跨法域对齐服务；失败负缓存，避免每轮重试。"""
        if self._multihop is None:
            try:
                from services.multihop_service import MultiHopService
                self._multihop = MultiHopService(repo=self.retriever.repo)
            except Exception as e:
                logger.warning("[Agent] 初始化跨法域多跳服务失败: %s", e)
                self._multihop = False
        return self._multihop or None

    def _get_kg_reasoning(self):
        """惰性复用 QAService 的 KGReasoningService（共享 ES 仓储）。"""
        if self._kg_reasoning is None:
            try:
                self._kg_reasoning = self._qa_provider()._get_kg_reasoning()
            except Exception as e:
                logger.warning("[Agent] 初始化图谱推理服务失败: %s", e)
                self._kg_reasoning = False
        return self._kg_reasoning or None

    def _resolve_graph_mode(self, question: str, mode: str) -> str:
        """auto 路由：识别到多法域/对比信号→cross，否则→entity。纯规则、零 LLM。"""
        if mode in ("cross", "entity"):
            return mode
        if len(self._jurisdictions or []) >= 2:
            return "cross"
        return "cross" if _CROSS_GRAPH_RE.search(question) else "entity"

    def _graph_cross(self, question: str) -> Dict:
        """跨法域对齐链：源条款→对应法规→目标条款 / 同一概念链。"""
        svc = self._get_multihop()
        if svc is None:
            return {"mode": "cross", "count": 0, "paths": [], "evidence": "",
                    "note": "跨法域多跳服务不可用"}
        data = svc.answer(
            question, top_k_laws=3, synthesize=False,
            jurisdictions=self._jurisdictions,
        )
        paths = data.get("reasoning_path") or []
        if not data.get("found") or not paths:
            return {"mode": "cross", "count": 0, "paths": [], "evidence": "",
                    "note": data.get("message") or "未找到跨法域对齐依据"}

        source = data.get("source") or {}
        targets = data.get("targets") or []
        lines = ["【跨法域对齐路径（系统校验，不含模型生成）】"]
        for p in paths[:_GRAPH_MAX_PATHS]:
            lines.append(f"{p.get('path_id', '')}（{p.get('level', '')}）: {p.get('text', '')}")
        lines.append(
            f"源条款：《{source.get('title', '')}》{source.get('article_number', '')}"
            f"（{source.get('jurisdiction', '')}）"
        )
        for t in targets:
            arts = "、".join(
                str(a.get("article_number", "")) for a in (t.get("articles") or [])
            )
            lines.append(
                f"对应法规：《{t.get('title', '')}》（{t.get('jurisdiction', '')}，"
                f"相似度 {t.get('similarity', 0)}）；相关条款：{arts or '无'}"
            )
            for a in (t.get("articles") or [])[:2]:
                content = re.sub(r"\s+", " ", str(a.get("content") or "")).strip()
                if content:
                    lines.append(f"  {a.get('article_number', '')}条：{content[:200]}")
            if t.get("shared_concepts"):
                lines.append("  共享概念：" + "、".join(
                    str(c.get("name", "")) for c in t["shared_concepts"]
                ))

        cited = set()
        if source.get("law_id") and source.get("article_number"):
            cited.add(f"{source['law_id']}::{source['article_number']}")
        for t in targets:
            for a in (t.get("articles") or []):
                if t.get("law_id") and a.get("article_number"):
                    cited.add(f"{t['law_id']}::{a['article_number']}")

        shown = paths[:_GRAPH_MAX_PATHS]
        # 观测值只保留路径文本与轻量元数据，避免决策与合成各读一遍完整结构导致 token 翻倍。
        return {
            "mode": "cross",
            "count": len(shown),
            "total_paths": len(paths),
            "truncated": len(paths) > len(shown),
            "source": {
                "title": source.get("title", ""),
                "article_number": source.get("article_number", ""),
                "jurisdiction": source.get("jurisdiction", ""),
            },
            "target_laws": [
                {
                    "title": t.get("title", ""),
                    "jurisdiction": t.get("jurisdiction", ""),
                    "similarity": t.get("similarity", 0),
                    "articles": [a.get("article_number", "") for a in (t.get("articles") or [])],
                    "shared_concepts": [
                        c.get("name", "") for c in (t.get("shared_concepts") or [])
                    ],
                }
                for t in targets
            ],
            "cited_articles": sorted(cited),
            "evidence": "\n".join(lines)[:_GRAPH_MAX_EVIDENCE_CHARS],
        }

    def _graph_entity(self, question: str, deepen: Optional[bool]) -> Dict:
        """法域内实体关系多跳，不生成答案。"""
        svc = self._get_kg_reasoning()
        if svc is None:
            return {"mode": "entity", "count": 0, "paths": [], "evidence": "",
                    "note": "图谱推理服务不可用"}
        data = svc.answer(
            question, top_k_entities=5, hops=2, deepen=deepen, synthesize=False,
            jurisdictions=self._jurisdictions,
        )
        paths = data.get("reasoning_path") or []
        if not data.get("found") or not paths:
            return {"mode": "entity", "count": 0, "paths": [], "evidence": "",
                    "note": data.get("message") or "知识图谱中未找到多跳路径"}

        seeds = data.get("seed_entities") or []
        lines = ["【图谱实体多跳路径（系统校验，不含模型生成）】"]
        for p in paths[:_GRAPH_MAX_PATHS]:
            lines.append(f"{p.get('path_id', '')}（{p.get('level', '')}）: {p.get('text', '')}")
        if seeds:
            lines.append("命中实体：" + "、".join(
                str(s.get("name") or s.get("entity_id") or "") for s in seeds
            ))

        cited = set()
        for p in paths[:_GRAPH_MAX_PATHS]:
            for step in (p.get("steps") or []):
                ev = ((step.get("evidence") or [{}])[0]) or {}
                if ev.get("law_id") and ev.get("article"):
                    cited.add(f"{ev['law_id']}::{ev['article']}")

        shown = paths[:_GRAPH_MAX_PATHS]
        subgraph = data.get("subgraph") or {}
        return {
            "mode": "entity",
            "count": len(shown),
            "total_paths": len(paths),
            "truncated": len(paths) > len(shown),
            "seed_entities": [
                s.get("name", "") for s in seeds
            ],
            "subgraph": {
                "path_count": subgraph.get("path_count", 0),
                "deepen_enabled": bool(subgraph.get("deepen_enabled")),
                "deep_path_count": subgraph.get("deep_path_count", 0),
            },
            "cited_articles": sorted(cited),
            "evidence": "\n".join(lines)[:_GRAPH_MAX_EVIDENCE_CHARS],
        }

    def _graph_flat(self, question: str) -> Dict:
        """兜底：沿用既有扁平图谱证据（实体关系 + 条款摘录）。"""
        qa = self._qa_provider()
        text = qa._collect_kg_evidence(question, self._jurisdictions)
        return {
            "mode": "flat",
            "count": 1 if text else 0,
            "evidence": text[:1500],
            "cited_articles": [],
            "note": "" if text else "知识图谱中未找到与该问题相关的实体关系",
        }

    def _query_graph(self, question: str = "", mode: str = "auto",
                     deepen: Optional[bool] = None) -> Dict:
        """查询图谱多跳路径：auto 规则路由 → 失败逐级回退，绝不阻断 Agent。"""
        q = (question or "").strip()
        if not q:
            return {"error": "question 不能为空", "count": 0, "evidence": ""}

        resolved = self._resolve_graph_mode(q, mode)
        result = self._graph_cross(q) if resolved == "cross" else self._graph_entity(q, deepen)
        if not result.get("count") and mode == "auto":
            # auto 下首选的模式无命中，换另一种模式再试一次（零 LLM）
            other = self._graph_entity(q, deepen) if resolved == "cross" else self._graph_cross(q)
            if other.get("count"):
                result = other
        if not result.get("count"):
            fallback = self._graph_flat(q)
            if fallback.get("count"):
                fallback["note"] = (result.get("note") or "") + "；已回退扁平图谱证据"
                return fallback
        return result

    def _evidence_status(self) -> Dict:
        """跨步工作记忆：汇报证据池覆盖台账（只读，不改动证据池）。

        证据池 key 为 `law_id::article`，value 含 metadata；据此派生已覆盖/缺口法域与已
        检索条文，让模型拿到确定性事实而非凭记忆推断；返回值随探索变化，故不做 memo。
        """
        evidence = self._evidence_ref if isinstance(self._evidence_ref, dict) else {}
        by_jur: Dict[str, int] = {}
        articles: List[str] = []
        for ctx in evidence.values():
            meta = (ctx or {}).get("metadata") or {}
            jur = str(meta.get("jurisdiction") or "").strip() or "未知"
            by_jur[jur] = by_jur.get(jur, 0) + 1
            title = str(meta.get("title") or "").strip()
            article = str(meta.get("article_number") or "").strip()
            if title and article:
                articles.append(f"《{title}》{article}")

        expected = list(self._jurisdictions or [])
        gaps = [j for j in expected if j not in by_jur]
        covered_text = "、".join(f"{j}({n})" for j, n in sorted(by_jur.items(), key=lambda kv: -kv[1]))

        parts = [f"已覆盖法域：{covered_text or '（暂无）'}"]
        if gaps:
            parts.append(f"缺口法域：{'、'.join(gaps)}")
        if expected:
            parts.append(f"目标法域：{'、'.join(expected)}")
        if articles:
            parts.append(f"已检索条文 {len(articles)} 条，示例：{'；'.join(articles[:5])}")

        return {
            "covered_jurisdictions": by_jur,
            "expected_jurisdictions": expected,
            "gaps": gaps,
            "article_count": len(evidence),
            "summary": "；".join(parts),
        }
