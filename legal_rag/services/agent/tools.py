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
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from core.config import settings

logger = logging.getLogger(__name__)

# 单工具超时的共享执行池（工具本身无状态，无需请求级隔离）。max_workers=8 与 ES msearch
# 并发上限同量级，可承接 ToolNode 的一轮并发扇出；超时只解除主流程等待，被卡线程靠底层
# HTTP 客户端的 socket 超时（秒级~30s）自行回收，8 是留出余量后的取值。
_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="agent-tool")

# 批量检索的子查询并行池：与 _TOOL_EXECUTOR 分离，避免「工具池内的任务再向同一池提交
# 子任务」造成线程池饥饿（工具调用本身已占用一个 worker）。
_BATCH_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent-batch")

# 对比服务为无请求状态的重型对象（自带 ES 客户端），进程级复用即可。
_COMPARE_SERVICE = None


def _get_compare_service():
    global _COMPARE_SERVICE
    if _COMPARE_SERVICE is None:
        from services.compare_service import CompareService
        _COMPARE_SERVICE = CompareService()
    return _COMPARE_SERVICE


# 法规目录缓存：status/日期等法规级字段只在 law_catalog.json 中，ES 索引不存 status。
# 按文件 mtime 失效，法规管理改动目录后自动重载。
_CATALOG_CACHE: Dict[str, Any] = {"mtime": None, "map": {}}


def _law_catalog_entry(law_id: str) -> Dict:
    """取 law_catalog 中的法规记录（带 mtime 缓存）；失败返回空 dict。"""
    if not law_id:
        return {}
    try:
        from utils.law_catalog import LawCatalog

        catalog = LawCatalog()
        mtime = catalog.catalog_path.stat().st_mtime
        if _CATALOG_CACHE["mtime"] != mtime:
            data = catalog.load()
            _CATALOG_CACHE["map"] = {
                x.get("law_id"): x for x in (data.get("laws") or []) if x.get("law_id")
            }
            _CATALOG_CACHE["mtime"] = mtime
        return _CATALOG_CACHE["map"].get(law_id) or {}
    except Exception:
        return {}


# ============================================================
# 工具参数模型（schema 由 pydantic 自动生成，字段描述即工具选择依据）
# ============================================================
class SearchLawsArgs(BaseModel):
    """检索法规条文的入参。"""

    query: Optional[str] = Field(
        None, description="单条检索查询，建议使用法律术语，可包含法域、主题词")
    queries: Optional[List[str]] = Field(
        None,
        description=(
            "批量检索查询（2-4 条不同子问题/法域），与 query 二选一。"
            "内部并行检索并合并去重，适合首轮一次性覆盖多个法域或方面，"
            "比逐条多次调用更省预算"
        ),
    )
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


class GetArticleTextArgs(BaseModel):
    """读取条款完整正文的入参。"""

    law_id: str = Field(..., description="法规 ID，取自 search_laws 结果的 law_id")
    article_number: str = Field(..., description="条款号，取自 search_laws 结果的 article_number")


class CheckLawStatusArgs(BaseModel):
    """查询法规时效与基本信息的入参。"""

    law_id: str = Field(..., description="法规 ID，取自 search_laws 结果的 law_id")


class CompareJurisdictionsArgs(BaseModel):
    """双法域同主题条款对照的入参。"""

    topic: str = Field(..., description="对比主题/关键词，如 数据保护、网络安全")
    jurisdiction_a: str = Field(..., description="法域 A（如 中国澳门）")
    jurisdiction_b: str = Field(..., description="法域 B（如 中国香港）")
    top_k: Optional[int] = Field(None, description="每个法域检索条款数，默认 5，最大 20")



# 工具观测值统一为 JSON 文本；AgentService 消费 ToolMessage 时再解析回 dict 累积证据。
_SEARCH_LAWS_DESC = (
    "按关键词/语义检索法规条文，返回最相关的条文（含法域、法规名、条号、正文片段）。"
    "适合先定位与问题相关的法条；可指定法域缩小范围。支持 queries 批量检索："
    "首轮一次传入 2-4 条针对不同法域/方面的查询，内部并行并合并去重，比分多次调用更省预算。"
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

_GET_ARTICLE_TEXT_DESC = (
    "读取指定条款的完整正文（超长会截断并标注）。当 search_laws 返回的片段不足以判断"
    "构成要件、例外或程序细节时使用；一次取全文可替代用相近措辞反复检索。"
    "参数取自 search_laws 结果的 law_id 与 article_number。"
)
_CHECK_LAW_STATUS_DESC = (
    "查询法规的时效状态与基本信息：是否现行有效、通过/生效/公布日期、法域与主题。"
    "在把某条款作为依据前确认其未被废止时调用；参数取自 search_laws 结果的 law_id。"
)
_COMPARE_JURISDICTIONS_DESC = (
    "对两个法域在同一主题下做条款级对照（按法规聚合，返回涉及条款号与内容摘要）。"
    "适合「A 与 B 在某方面有何差异」类问题：一次调用即可获得两侧条款分布，"
    "替代分别检索两个法域。"
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
        """把工具执行器包装为 LangChain 工具（各自绑定本请求的 memo 与法域）。"""
        return [
            self._as_tool("search_laws", _SEARCH_LAWS_DESC, SearchLawsArgs, self._search_laws),
            self._as_tool("get_article_text", _GET_ARTICLE_TEXT_DESC,
                          GetArticleTextArgs, self._get_article_text),
            self._as_tool("check_law_status", _CHECK_LAW_STATUS_DESC,
                          CheckLawStatusArgs, self._check_law_status),
            self._as_tool("compare_jurisdictions", _COMPARE_JURISDICTIONS_DESC,
                          CompareJurisdictionsArgs, self._compare_jurisdictions),
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
    def _search_laws(self, query: str = "", queries: Optional[List[str]] = None,
                     jurisdiction: Optional[str] = None,
                     top_k: Optional[int] = None) -> Dict:
        """检索法规条文，支持 queries 批量：子查询并行执行后按条款去重合并。

        固定跳过精排（`_rerank=False`）以换取速度。批量模式让首轮一次调用覆盖多个
        子问题/法域，减少探索轮次；单条模式保留原有的查询分解行为。
        """
        batch = [str(q).strip() for q in (queries or []) if str(q or "").strip()]
        if not batch:
            q = (query or "").strip()
            if not q:
                return {"error": "query 或 queries 至少提供一项", "count": 0, "results": []}
            batch = [q]
        batch = batch[: max(1, int(settings.agent_search_batch_max))]

        try:
            k = min(int(top_k), 10) if top_k else settings.agent_tool_top_k
        except (TypeError, ValueError):
            k = settings.agent_tool_top_k
        k = max(1, k)

        def _run(sub_query: str) -> List[Dict]:
            try:
                return self.retriever.hybrid_search(
                    query=sub_query,
                    top_k=k,
                    jurisdiction=jurisdiction or None,
                    # 批量模式各子查询已聚焦，关闭分解避免二次放大的编码与 ES 开销
                    _decompose=len(batch) == 1,
                    _rerank=False,
                )
            except Exception as e:
                logger.warning("[Agent] 子查询检索失败 %s: %s", sub_query[:30], e)
                return []

        if len(batch) == 1:
            per_query = [_run(batch[0])]
        else:
            per_query = list(_BATCH_EXECUTOR.map(_run, batch))

        # 按 (law_id, article_number) 去重，保留最高分，再按分数降序截断
        merged: Dict[Tuple[str, str], Dict] = {}
        for hits in per_query:
            for r in hits or []:
                key = (r.get("law_id", ""), str(r.get("article_number", "")))
                if key not in merged or float(r.get("score") or 0.0) > float(merged[key].get("score") or 0.0):
                    merged[key] = r

        cap = min(max(k, 1) * len(batch), max(1, int(settings.agent_search_batch_result_max)))
        ordered = sorted(merged.values(), key=lambda x: float(x.get("score") or 0.0), reverse=True)[:cap]

        out = []
        for r in ordered:
            out.append({
                "law_id": r.get("law_id", ""),
                "title": r.get("title", ""),
                "article_number": r.get("article_number", ""),
                "jurisdiction": r.get("jurisdiction", ""),
                "content": (r.get("content", "") or "")[: settings.agent_obs_max_chars],
                "score": round(float(r.get("score") or 0.0), 4),
            })
        return {"count": len(out), "queries_used": batch, "results": out}

    def _get_article_text(self, law_id: str = "", article_number: str = "") -> Dict:
        """读取单条条款的完整正文，替代为补全细节而反复检索。"""
        law_id = (law_id or "").strip()
        article = str(article_number or "").strip()
        if not law_id or not article:
            return {"error": "law_id 与 article_number 均不能为空", "found": False}
        try:
            text = self.retriever.repo.get_article_full_text(law_id, article)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "found": False}
        if not text:
            return {"found": False, "law_id": law_id, "article_number": article,
                    "note": "未取到该条款正文，请核对 law_id 与条号"}

        meta: Dict[str, Any] = {}
        try:
            meta = self.retriever.repo.get_law_meta(law_id) or {}
        except Exception:
            pass
        entry = _law_catalog_entry(law_id)

        cap = max(200, int(settings.agent_article_text_chars))
        return {
            "found": True,
            "law_id": law_id,
            "article_number": article,
            "title": entry.get("law_name_zh") or meta.get("title", ""),
            "jurisdiction": entry.get("jurisdiction") or meta.get("jurisdiction", ""),
            "status": entry.get("status") or meta.get("status", ""),
            "content": text[:cap],
            "total_chars": len(text),
            "truncated": len(text) > cap,
        }

    def _check_law_status(self, law_id: str = "") -> Dict:
        """查询法规时效与基本信息，用于在引用前确认法规未被废止。"""
        law_id = (law_id or "").strip()
        if not law_id:
            return {"error": "law_id 不能为空", "found": False}
        meta: Dict[str, Any] = {}
        try:
            meta = self.retriever.repo.get_law_meta(law_id) or {}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "found": False}
        entry = _law_catalog_entry(law_id)
        if not meta and not entry:
            return {"found": False, "law_id": law_id, "note": "库中未找到该法规"}

        raw_status = str(entry.get("status") or meta.get("status") or "")
        status_map = {"active": "现行有效", "repealed": "已废止", "amended": "已修订"}
        return {
            "found": True,
            "law_id": law_id,
            "title": entry.get("law_name_zh") or meta.get("title", ""),
            "jurisdiction": entry.get("jurisdiction") or meta.get("jurisdiction", ""),
            "status": raw_status,
            "status_text": status_map.get(raw_status, raw_status or "未知"),
            "passing_date": entry.get("passing_date") or meta.get("passing_date", ""),
            "effective_date": entry.get("effective_date") or meta.get("effective_date", ""),
            "publication_date": entry.get("publication_date") or meta.get("publication_date", ""),
        }

    def _compare_jurisdictions(self, topic: str = "", jurisdiction_a: str = "",
                               jurisdiction_b: str = "", top_k: Optional[int] = None) -> Dict:
        """双法域同主题条款对照：一次调用获得两侧条款分布，替代分别检索。"""
        topic = (topic or "").strip()
        jur_a = (jurisdiction_a or "").strip()
        jur_b = (jurisdiction_b or "").strip()
        if not topic or not jur_a or not jur_b:
            return {"error": "topic、jurisdiction_a、jurisdiction_b 均不能为空", "count": 0}

        try:
            k = max(1, min(int(top_k), 20)) if top_k else 5
        except (TypeError, ValueError):
            k = 5

        try:
            result = _get_compare_service().compare(
                topic, jur_a, jur_b, top_k_per_jurisdiction=k,
            )
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "count": 0}

        def _brief(items: List[Dict]) -> List[Dict]:
            brief = []
            for law in (items or [])[:3]:
                previews = law.get("chunks_preview") or []
                brief.append({
                    "law_id": law.get("law_id", ""),
                    "title": law.get("title", ""),
                    "article_numbers": (law.get("article_numbers") or [])[:10],
                    "article_count": law.get("article_count", 0),
                    "keywords": (law.get("keywords") or [])[:5],
                    "preview": (previews[0].get("content", "")[:200] if previews else ""),
                })
            return brief

        side_a, side_b = _brief(result.get("results_a")), _brief(result.get("results_b"))
        return {
            "topic": topic,
            "jurisdiction_a": jur_a,
            "jurisdiction_b": jur_b,
            "a": side_a,
            "b": side_b,
            "count": len(side_a) + len(side_b),
            "cache_hit": bool(result.get("_cache_hit")),
        }

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
