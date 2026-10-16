"""Agent 服务：基于 LangGraph 的受约束多步推理问答。

流程：
    问题 → [探索] LangGraph 编排「模型决策 ↔ 工具执行」循环（检索/引用追溯/图谱查询）
                ↓ 工具结果作为 ToolMessage 回灌，直至模型认为证据足够或触达预算
         → [证据收口] 对累积证据去重 + 补全正文（一次 msearch）+ 精排一次
         → [最终合成] 复用 QAService 的合成 prompt 流式产出带来源标注的答案

分工（框架管编排、自研管策略）：
  - **交给框架**：模型↔工具的循环编排、工具 schema 生成、消息回灌、LLM 客户端；
    LangGraph 是 async-first，但其 `stream()` 提供同步版本，可直接嵌入本项目的
    同步 SSE 生成器，无需 async 桥接；
  - **保留自研**：复杂度路由、三重预算（步数/工具次数/时长）、停滞短路、证据收敛、
    最终合成、三层降级、以及自有 SSE 事件协议（前端契约不变）。这些是实测调优出的
    策略资产，框架不提供，且替换会引入回归风险。

设计取舍（与项目既有约束一致）：
  - **受约束而非放任**：步数、工具调用数、端到端时长三重上限，超限强制收口；
  - **探索便宜、合成讲究**：探索步检索跳过精排（本机精排占检索延迟约 93%），
    只对最终证据集精排一次，兼顾成本与质量；
  - **最终合成复用既有 RAG**：调用 QAService._build_messages，保证回答结构、
    【来源X】标注、图谱证据口径与现有问答完全一致，不另造一套输出格式；
  - **全链路降级**：开关关闭 / 未配置密钥 / 探索未产出证据 / 任何异常，
    一律回退到既有 QAService.ask_stream，端点永远返回可用结果。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from core.config import settings
from services.agent.tools import AgentToolkit
from services.query_decomposer import _CONJUNCTION_RE, _TARGET_ARTICLE_RE
from services.retriever_service import RetrieverService

logger = logging.getLogger(__name__)

# 复杂度路由的语义信号（对比/并列意图）。
# 复用 RetrieverService.COMPARE_KEYWORDS 作为唯一口径：它同时驱动检索侧的跨法域扇出，
# 两者保持一致才能保证「被路由判为复杂的查询」正是检索侧会特殊处理的那些。
_COMPARE_KEYWORDS = tuple(RetrieverService.COMPARE_KEYWORDS)

# route_mode 取值：auto=按复杂度自动决定；agent=强制多步探索；fast=强制既有快速 RAG
ROUTE_MODE_AUTO = "auto"
ROUTE_MODE_AGENT = "agent"
ROUTE_MODE_FAST = "fast"

# decide_route 的三态决策：
#   agent = 强信号，直接多步探索
#   fast  = 强信号，直接快速 RAG
#   probe = 灰区，规则无法定夺 → 由调用方做一次廉价检索质量探测后再定
DECIDE_AGENT = "agent"
DECIDE_FAST = "fast"
DECIDE_PROBE = "probe"


def decide_route(question: str, route_mode: str = ROUTE_MODE_AUTO,
                 routed_jurisdictions: Optional[List[str]] = None) -> tuple:
    """复杂度路由决策，返回 (decision, reason)，decision ∈ {agent, fast, probe}。

    动机：Agent 相对单跳 RAG 稳定多付约 10s 延迟（多轮 LLM 往返 + 收口精排），
    但大量查询本就不需要多步探索，不加区分会拖慢简单查询。

    决策分两层：
      - **强信号**直接定夺（规则）；其中「多法域/对比词/并列连词」判 Agent，
        「极短查询/定向第X条」判 Fast；
      - **灰区**（名义上的单一问题，如「XX机构对YY的监管要求是什么」）返回 probe，
        交由调用方做一次廉价检索探测决定。实测这类查询里有一批（Q051/Q009）单次
        检索捞不到 gold、需要多轮召回，纯规则分流会丢掉它们，故不能一刀切。

    判定优先级（越靠前越强）：
      1. 显式 route_mode 优先（前端「深度模式」= agent；「快速」= fast）；
      2. 关闭复杂度路由开关 → 一律 Agent（保持开关前行为，便于回滚）；
      3. 空 / 短查询 / 定向「第X条」→ Fast。
         定向查询在检索侧有专门的置顶逻辑，而 Agent 的 search_laws 工具不含该逻辑，
         走 Agent 反而更差；
      4. 多法域（检索侧会跨法域扇出）/ 对比词 / 并列连词 → Agent；
      5. 其余 → probe（灰区探测）。

    Args:
        question: 用户问题原文。
        route_mode: auto / agent / fast。
        routed_jurisdictions: 检索侧规则路由出的目标法域（len>1 视为多法域）。
            由调用方注入，避免本函数依赖 RetrieverService 实例，便于单测。

    Returns:
        (decision, reason)，reason 供日志与 SSE 事件观测。
    """
    q = (question or "").strip()

    if route_mode == ROUTE_MODE_FAST:
        return DECIDE_FAST, "manual_fast"
    if route_mode == ROUTE_MODE_AGENT:
        return DECIDE_AGENT, "manual_agent"
    if not settings.agent_complexity_routing_enabled:
        return DECIDE_AGENT, "routing_disabled"
    if not q:
        return DECIDE_FAST, "empty_query"
    if len(q) < max(1, int(settings.agent_min_query_len)):
        return DECIDE_FAST, "short_query"
    if _TARGET_ARTICLE_RE.search(q):
        return DECIDE_FAST, "targeted_article"
    if routed_jurisdictions and len(routed_jurisdictions) > 1:
        return DECIDE_AGENT, "multi_jurisdiction"
    if any(kw in q for kw in _COMPARE_KEYWORDS):
        return DECIDE_AGENT, "comparison"
    if _CONJUNCTION_RE.search(q) and len(q) >= int(settings.agent_conjunction_min_len):
        return DECIDE_AGENT, "multi_intent"
    # 灰区：既无强复杂信号、也不属于可安全走快路径的形态 → 交给探测
    if not settings.agent_probe_enabled:
        return DECIDE_FAST, "simple"
    return DECIDE_PROBE, "ambiguous_single"


class AgentService:
    """LangGraph 编排的多步推理问答服务。"""

    def __init__(self):
        self.retriever = RetrieverService()
        self._qa = None  # 延迟获取 QAService（单例），避免构造期触发预热相关重活
        # 本服务是进程级单例，但**工具运行时与图都不挂在实例上**：工具运行时持有
        # 「单次提问」级的去重 memo，必须按请求新建，否则并发请求会互相清空/串用
        # 对方的缓存（把 A 的检索结果当成 B 的缓存命中）。见 ask_stream 中的构造注释。
        self.api_key = settings.llm_api_key or ""
        self.base_url = settings.llm_api_base_url or "https://api.deepseek.com/v1"
        # LLM 客户端进程级复用：ChatOpenAI 内部持有 httpx 连接池，按请求新建会丢掉
        # keep-alive、每次重新握手；它是无状态且线程安全的，可跨请求共享。
        self._llm = self._make_llm() if self.api_key else None

    def _qa_service(self):
        if self._qa is None:
            from services.qa_service import QAService
            self._qa = QAService()
        return self._qa

    def _make_llm(self) -> ChatOpenAI:
        """构建共享的 ChatOpenAI 客户端（决策步用它产出 tool_calls）。"""
        kwargs: Dict[str, Any] = {
            "model": settings.agent_model or settings.llm_qa_model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": 0.1,
            "max_tokens": 1024,  # 决策步只需产出 tool_calls 或一句结论，无需长文
            # 单次请求超时与端到端预算同量级：超预算的探索本就没有价值
            "timeout": float(settings.agent_timeout_total),
        }
        extra_body = settings.llm_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        return ChatOpenAI(**kwargs)

    # ================================================================
    #  图与提示
    # ================================================================
    @staticmethod
    def _system_prompt(detected_jurisdictions: List[str]) -> str:
        jur_hint = "、".join(detected_jurisdictions) if detected_jurisdictions else "（未识别到，可按需自行判断）"
        # 把工具预算显式告诉模型：实测旧提示词的「证据足够就停」形同虚设，模型几乎
        # 每次都跑满预算（avg≈5.9 / 上限 6）。给出明确的调用次数与停止条件后，
        # 模型才有依据主动收敛。
        max_calls = max(1, int(settings.agent_max_tool_calls))
        return (
            "你是跨境法规检索分析助手，可通过调用工具分步收集证据后再回答问题。\n"
            "【工作方式】\n"
            "1. 先判断问题需要哪些法条/法域的证据。通常 1 次检索即可起步，必要时再补充"
            "检索、追溯引用或查询图谱，不必面面俱到。\n"
            f"2. 本次最多可调用工具 {max_calls} 次；请把调用留给最有价值的信息，"
            "不要用相同或高度相似的查询重复检索同一内容。\n"
            "3. 一旦证据足以回答问题，立即停止本轮（直接输出最终回答文字、不要附带工具调用）；"
            "若某次工具返回的内容没有带来新信息，也应立即停止，避免无效重复。\n"
            "4. 不要凭空编造法条；工具未返回的内容不要臆断。\n"
            "【工具】search_laws=检索条文；follow_references=追溯被引用条款；query_graph=查询实体关系。\n"
            f"【已知法域】{jur_hint}（若非空，检索时优先使用该法域）。"
        )

    def _build_agent(self, system_prompt: str, tools: List[Any]):
        """构建 LangGraph ReAct 图：模型↔工具循环由框架驱动。

        `create_agent` 是当前 LangChain 的推荐入口（`langgraph.prebuilt.create_react_agent`
        已标记弃用）。此处只传模型、工具与系统提示——预算、停滞、证据收敛等策略仍由
        本服务在消费事件流时执行，避免把实测调优的逻辑埋进框架中间件里难以观测。

        图按请求构建（工具绑定请求级 memo），但模型客户端复用进程级实例（见 __init__）。
        """
        return create_agent(self._llm, tools, system_prompt=system_prompt)

    @staticmethod
    def _recursion_limit() -> int:
        """图的超级步上限（防御性第二层预算）。

        LangGraph 以「超级步」计数，一次「模型决策 + 工具执行」约消耗 2 步，故取
        `max_steps * 2 + 1`。主预算由本服务的工具次数与步数计数执行，此处仅防跑飞。
        """
        return max(3, max(1, int(settings.agent_max_steps)) * 2 + 1)

    # ================================================================
    #  证据累积与收口
    # ================================================================
    @staticmethod
    def _parse_tool_result(content: Any) -> Dict:
        """把 ToolMessage 的内容解析回工具结果 dict（工具统一以 JSON 文本回传）。"""
        if isinstance(content, dict):
            return content
        if isinstance(content, list):
            # LangChain 可能以 content blocks 列表承载文本，拼接其中的 text 块
            content = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return {"raw": content}

    @staticmethod
    def _add_ctx(evidence: Dict[str, Dict], r: Dict, similarity: float, source: str = "search") -> None:
        """把一条工具结果并入证据池（按 law_id::article 去重，保留更高相似度）。

        source 标记证据来源（search=检索命中 / reference=引用追溯），收口收敛时据此
        为引用条款保留配额，避免它们被检索命中的高相似度挤出候选池。
        """
        law_id = (r.get("law_id") or "").strip()
        article = str(r.get("article_number") or "").strip()
        if not law_id or not article:
            return
        key = f"{law_id}::{article}"
        if key in evidence:
            if similarity > evidence[key].get("similarity", 0.0):
                evidence[key]["similarity"] = similarity
            return
        evidence[key] = {
            "content": r.get("content", "") or "",
            "metadata": {
                "law_id": law_id,
                "title": r.get("title", "") or "",
                "article_number": article,
                "jurisdiction": r.get("jurisdiction", "") or "",
            },
            "similarity": similarity,
            "_source": source,
        }

    def _absorb_evidence(self, name: str, result: Dict, evidence: Dict[str, Dict],
                         graph_parts: List[str], graph_seen: set) -> bool:
        """吸收工具结果，返回本次调用是否给证据池带来了「新增」（供停滞检测判定）。

        图谱证据按内容去重（graph_seen）：模型可能用不同措辞重复查询图谱，相同文本
        重复追加不仅让「是否新增」判定失真，还会在最终 prompt 里重复堆叠。
        """
        if not isinstance(result, dict) or result.get("error"):
            return False
        before = len(evidence) + len(graph_parts)
        if name == "search_laws":
            for r in result.get("results", []):
                self._add_ctx(evidence, r, float(r.get("score") or 0.0), source="search")
        elif name == "follow_references":
            for r in result.get("referenced", []):
                self._add_ctx(evidence, r, 0.0, source="reference")
        elif name == "query_graph":
            ev = result.get("evidence")
            if ev and ev not in graph_seen:
                graph_seen.add(ev)
                graph_parts.append(ev)
        return (len(evidence) + len(graph_parts)) > before

    @staticmethod
    def _summarize_result(name: str, result: Dict) -> str:
        """为 SSE / trace 生成一行人类可读的观测摘要。"""
        if not isinstance(result, dict):
            return "无返回"
        if result.get("error"):
            return f"失败：{result['error']}"
        if name == "search_laws":
            items = result.get("results", [])
            head = "；".join(f"《{i.get('title','')}》{i.get('article_number','')}" for i in items[:3])
            return f"命中 {len(items)} 条" + (f"：{head}" if head else "")
        if name == "follow_references":
            items = result.get("referenced", [])
            head = "；".join(f"《{i.get('title','')}》{i.get('article_number','')}" for i in items[:3])
            return f"追溯引用 {len(items)} 条" + (f"：{head}" if head else "")
        if name == "query_graph":
            return "命中图谱证据" if result.get("evidence") else "图谱无命中"
        return "完成"

    @staticmethod
    def _converge_candidates(contexts: List[Dict], top_k: int) -> List[Dict]:
        """把证据池收敛到精排候选上限，兼顾成本与召回。

        策略：为「引用追溯」来的条款保留配额（它们是 agent 显式追出的、相似度恒为 0，
        若不保护会被检索命中的高相似度挤出），剩余名额给检索命中按相似度降序填充。
        收敛后条数不超过 agent_finalize_max_candidates（且不少于最终要用的 top_k）。
        """
        cap = max(int(settings.agent_finalize_max_candidates), int(top_k))
        if len(contexts) <= cap:
            return contexts

        refs = [c for c in contexts if c.get("_source") == "reference"]
        hits = [c for c in contexts if c.get("_source") != "reference"]
        hits.sort(key=lambda c: float(c.get("similarity") or 0.0), reverse=True)

        ref_quota = min(len(refs), settings.reference_expand_max, cap)
        picked = hits[: max(0, cap - ref_quota)] + refs[:ref_quota]
        return picked

    def _finalize_evidence(self, question: str, evidence: Dict[str, Dict], top_k: int) -> List[Dict]:
        """证据收口：收敛候选 → 补全正文（一次 msearch）→ 精排一次 → 取 pass_k 条。

        收敛必须排在精排（及正文补全）之前：本机 cross-encoder 精排占检索延迟约 93%，
        30 条候选实测耗时 40s+，先压到 agent_finalize_max_candidates 条再精排是收口
        耗时的决定性因素。任何一步失败均静默降级，不阻断作答。

        返回条数取 max(top_k, agent_finalize_pass_k)：多步探索额外召回的证据若在此处
        被截回 top_k，探索就白做了（实测 Agent 最终证据条数 5.00 < RAG 5.25）。精排成本
        只由收敛上限决定，放宽返回条数不增加精排耗时而让额外召回真正进入 prompt。
        """
        contexts = list(evidence.values())
        if not contexts:
            return []

        raw_n = len(contexts)
        contexts = self._converge_candidates(contexts, top_k)
        if len(contexts) < raw_n:
            logger.info(
                "[Agent] 证据池收敛：%d → %d（精排候选上限 %d）",
                raw_n, len(contexts), settings.agent_finalize_max_candidates,
            )

        pairs = [
            (c["metadata"].get("law_id", ""), c["metadata"].get("article_number", ""))
            for c in contexts
            if c["metadata"].get("law_id") and c["metadata"].get("article_number")
        ]
        try:
            full = self.retriever.repo.get_articles_docs_batch(pairs)
        except Exception as e:
            logger.warning("[Agent] 证据正文补全失败，沿用工具片段: %s", e)
            full = {}
        for c in contexts:
            doc = full.get((c["metadata"].get("law_id", ""), c["metadata"].get("article_number", "")))
            if doc and doc.get("content"):
                c["content"] = doc["content"]

        try:
            contexts = self.retriever._rerank_two_stage(query=question, candidates=contexts)
        except Exception as e:
            logger.warning("[Agent] 最终证据精排失败，沿用原顺序: %s", e)
        return contexts[: max(int(top_k), int(settings.agent_finalize_pass_k))]

    def _probe_retrieval_quality(self, question: str, top_k: int,
                                 jurisdictions: Optional[List[str]] = None) -> bool:
        """灰区查询的廉价探测：判断「一次检索是否已足够」，返回 True 表示足够。

        做法：只做一次 QAService._retrieve_contexts（向量检索 + 去重回取，**不含精排**），
        再复用 QAService._is_retrieval_poor 判定质量。不调用 LLM、不生成答案，因此
        成本远小于一次完整问答。

        为什么探测能预测快速路径的质量：快速路径（QAService.ask_stream）内部走的正是
        同一个 _retrieve_contexts，两条路径的检索结果一致，故探测的判定对快速路径有效。

        为什么用更高的相似度阈值：_is_retrieval_poor 默认阈值 0.6（≈cos 0.33）过于宽松，
        只拦得住「完全没有相关内容」；而灰区查询（如 Q051/Q009）通常能召回多条形似条款、
        只是没覆盖 gold，需要多轮探索才能补全。故这里用 agent_probe_min_similarity
        （默认 0.7）作为「高置信命中」门槛——top1 不够强即判定为证据不足、升级 Agent。

        任何异常一律返回 True（判定为「足够」→ 走快速路径）：探测只是优化项，不应因
        自身故障把所有查询推向 Agent；且快速链路自身已有检索回退，行为可预期。
        """
        try:
            qa = self._qa_service()
            contexts = qa._retrieve_contexts(question, top_k, jurisdictions)
            poor = qa._is_retrieval_poor(
                contexts,
                min_results=max(1, int(settings.agent_probe_min_results)),
                min_similarity=float(settings.agent_probe_min_similarity),
            )
            return not poor
        except Exception as e:
            logger.warning("[Agent] 检索质量探测失败，默认走快速 RAG: %s: %s", type(e).__name__, e)
            return True

    # ================================================================
    #  探索：消费 LangGraph 事件流，执行预算/停滞策略并转发 SSE
    # ================================================================
    def _explore(self, agent, messages_in: List[Dict], total_start: float):
        """同步消费图的事件流，边转发 SSE 边执行预算/停滞策略。

        yield 形状统一为 7 元组 `(event, evidence, graph_parts, trace, tool_calls, turns, stop)`：
        转发事件时仅 `event` 非空，末尾再 yield 一次汇总结果（`event` 为空、其余字段填齐），
        调用方据此区分「转发」与「汇总」，避免两套不同的返回形态。

        为什么不用异步：LangGraph 是 async-first，但 `stream()` 提供同步版本，可直接嵌
        进本项目的同步 SSE 生成器；改 async 会与「重操作不得占用事件循环」的既有铁律冲突，
        还需额外做 async→sync 桥接。

        预算与停滞为何放在这里而非中间件：它们需要在事件流上被观测与说明（trace 要写清
        「为什么这次提前收口」）；放进中间件会把策略埋进框架内部、难以日志化与单测，
        反而更复杂。
        """
        evidence: Dict[str, Dict] = {}
        graph_parts: List[str] = []
        graph_seen: set = set()
        trace: List[Dict] = []
        pending: Dict[str, Dict] = {}  # tool_call_id → 该次调用的 step/name/args
        tool_calls_made = 0
        model_turns = 0  # 模型决策轮次，等价于旧实现里的 step
        stop_reason = "step_budget"
        # 停滞计数：连续多少次工具调用没给证据池带来新增。模型偶尔会用不同措辞重复
        # 检索同一内容，这类调用不产生新证据，留着只会拖长延迟，故达到阈值即收口。
        no_gain_streak = 0
        stagnant_limit = max(1, int(settings.agent_max_stagnant_steps))
        halt = False
        stream = None

        try:
            stream = agent.stream(
                {"messages": messages_in},
                stream_mode="updates",  # 按节点交付增量：model 节点给决策，tools 给观测
                config={"recursion_limit": self._recursion_limit()},
            )
            for chunk in stream:
                if time.time() - total_start > settings.agent_timeout_total:
                    stop_reason = "timeout"
                    logger.info("[Agent] 超出时长上限，收口作答（turn=%d）", model_turns)
                    break

                for _node, payload in (chunk or {}).items():
                    for m in ((payload or {}).get("messages") or []):
                        if isinstance(m, AIMessage):
                            calls = list(getattr(m, "tool_calls", None) or [])
                            if not calls:
                                # 模型主动结束探索（认为证据足够 / 无需工具），最理想的收敛
                                stop_reason = "model_stop"
                                continue
                            model_turns += 1
                            for tc in calls:
                                if tool_calls_made >= settings.agent_max_tool_calls:
                                    stop_reason = "tool_budget"
                                    halt = True
                                    break
                                tool_calls_made += 1
                                args = tc.get("args") or {}
                                pending[tc.get("id")] = {
                                    "step": model_turns, "name": tc.get("name"), "args": args,
                                }
                                yield ({"type": "step", "step": model_turns, "action": "tool_call",
                                        "tool": tc.get("name"), "args": args}, None, None, None, None, None, None)
                        elif isinstance(m, ToolMessage):
                            info = pending.pop(getattr(m, "tool_call_id", ""), None)
                            if info is None:
                                continue
                            result = self._parse_tool_result(m.content)
                            gained = self._absorb_evidence(
                                info["name"], result, evidence, graph_parts, graph_seen)
                            summary = self._summarize_result(info["name"], result)
                            trace.append({"step": info["step"], "tool": info["name"],
                                          "args": info["args"], "summary": summary})
                            yield ({"type": "observation", "step": info["step"],
                                    "tool": info["name"], "summary": summary}, None, None, None, None, None, None)

                            # 停滞短路：连续多次调用都没带来新证据 → 判定为低效重复，立即收口
                            if gained:
                                no_gain_streak = 0
                            else:
                                no_gain_streak += 1
                                if no_gain_streak >= stagnant_limit:
                                    stop_reason = "no_progress"
                                    halt = True
                                    logger.info(
                                        "[Agent] 连续 %d 次工具调用无新增证据，提前收口（turn=%d, tool=%s）",
                                        no_gain_streak, model_turns, info["name"],
                                    )
                if halt:
                    break
        except Exception as e:
            # 图递归上限（步数耗尽）与网络/模型异常都归入预算耗尽，用已得证据作答。
            # 不按类型硬判（避免依赖具体异常类路径），按类名识别递归上限。
            stop_reason = "step_budget" if type(e).__name__ == "GraphRecursionError" else "llm_error"
            logger.warning("[Agent] 探索中断（%s），改用已收集证据作答: %s: %s",
                           stop_reason, type(e).__name__, e)
        finally:
            if stream is not None and halt:
                # 提前收口时显式关闭生成器，避免图在后台继续推进
                close = getattr(stream, "close", None)
                if callable(close):
                    close()

        yield (None, evidence, graph_parts, trace, tool_calls_made, model_turns, stop_reason)

    # ================================================================
    #  主流程（SSE 事件生成器）
    # ================================================================
    def ask_stream(self, question: str, top_k: int = 5,
                   jurisdictions: Optional[List[str]] = None,
                   history: Optional[List[Dict]] = None,
                   route_mode: str = ROUTE_MODE_AUTO) -> Iterator[Dict]:
        """多步推理问答；任何降级路径都回退到既有 RAG，保证有结果。

        route_mode 决定「是否值得走多步探索」：
          - auto（默认）：按复杂度规则决定；
          - agent：强制走 Agent（前端「深度模式」）；
          - fast：强制走既有快速 RAG。
        简单/定向查询走快速 RAG，避免用简单查询的延迟换复杂查询的收益。
        """
        # 开关关闭 / 未配密钥：直接走既有 RAG（端点仍可用）
        if not settings.agent_enabled:
            logger.info("[Agent] agent_enabled=false，降级为既有 RAG 链路")
            yield from self._qa_service().ask_stream(question, top_k, jurisdictions, history)
            return

        if not self.api_key:
            logger.warning("[Agent] 未配置 LLM 密钥，降级为既有 RAG 链路")
            yield from self._qa_service().ask_stream(question, top_k, jurisdictions, history)
            return

        # 复杂度路由：仅 auto 模式需要判定；agent/fast 为显式指定，直接放行/短路。
        # 多法域信号复用检索侧的规则路由（廉价、无副作用），不在此触发语义法域检测。
        if route_mode == ROUTE_MODE_AUTO and settings.agent_complexity_routing_enabled:
            try:
                routed = self.retriever._route_to_jurisdictions(question)
            except Exception:
                routed = None
        else:
            routed = None
        decision, route_reason = decide_route(question, route_mode, routed)

        # 灰区：做一次廉价检索探测（不生成答案）决定走哪条路。
        # 探测与快速路径用的是同一个 QAService._retrieve_contexts，因此探测结果能
        # 较准确地预测「快速路径能否拿到足够证据」；探测失败时保守走快速路径。
        if decision == DECIDE_PROBE:
            if self._probe_retrieval_quality(question, top_k, jurisdictions):
                decision, route_reason = DECIDE_FAST, route_reason + ":probe_ok"
            else:
                decision, route_reason = DECIDE_AGENT, route_reason + ":probe_poor"

        if decision == DECIDE_FAST:
            logger.info("[Agent] 复杂度路由→快速 RAG（reason=%s）", route_reason)
            yield {"type": "route", "mode": "fast", "reason": route_reason}
            # 透传用户原始的 jurisdictions（可能为 None）：与 /qa/stream 完全同路径，
            # 让 QAService 自行做法域检测与降级，保证「快速路径」行为与既有问答一致。
            yield from self._qa_service().ask_stream(question, top_k, jurisdictions, history)
            return

        total_start = time.time()
        yield {"type": "route", "mode": "agent", "reason": route_reason}
        yield {"type": "status", "stage": "plan", "data": "正在规划检索步骤…"}

        # 法域：复用 QA 的检测逻辑，保证与既有链路口径一致
        try:
            if jurisdictions:
                detected = list(jurisdictions)
            else:
                detected = self._qa_service()._detect_jurisdictions(question, history)
        except Exception:
            detected = []

        # 每请求独立的工具运行时与图：工具 memo 是「单次提问」级去重缓存，必须与请求
        # 同生命周期，并发请求各持一份、互不干扰（详见 __init__ 说明）。jurisdictions
        # 供图谱工具做法域内证据过滤，避免跨法域无关证据污染答案。
        toolkit = AgentToolkit(self.retriever, self._qa_service, jurisdictions=detected)
        agent = self._build_agent(self._system_prompt(detected), toolkit.build_tools())

        messages_in: List[Dict] = []
        if history:
            for h in history[-6:]:
                if h.get("role") in ("user", "assistant") and h.get("content"):
                    messages_in.append({"role": h["role"], "content": h["content"]})
        messages_in.append({"role": "user", "content": question})

        # 探索：由 _explore 消费图事件流并执行预算/停滞策略
        evidence: Dict[str, Dict] = {}
        graph_parts: List[str] = []
        trace: List[Dict] = []
        tool_calls_made = 0
        llm_turns = 0
        stop_reason = "step_budget"
        for ev, ev_evidence, ev_parts, ev_trace, ev_calls, ev_turns, ev_stop in self._explore(
                agent, messages_in, total_start):
            if ev is not None:
                yield ev
                continue
            evidence, graph_parts, trace = ev_evidence, ev_parts, ev_trace
            tool_calls_made, llm_turns, stop_reason = ev_calls, ev_turns, ev_stop

        # 探索未产出任何证据 → 降级到既有 RAG（保证有答案，覆盖简单问题/工具全失败）
        if not evidence and not graph_parts:
            logger.info("[Agent] 探索未产出证据，降级为既有 RAG 链路（turns=%d）", llm_turns)
            yield from self._qa_service().ask_stream(question, top_k, jurisdictions, history)
            return

        # ---- 证据收口 + 最终合成（复用 QA 的合成 prompt 与流式调用）----
        t_fin0 = time.time()
        contexts = self._finalize_evidence(question, evidence, top_k)
        kg_text = "\n".join(graph_parts)
        t_finalize = time.time() - t_fin0

        sources = []
        source_jurisdictions = set()
        for ctx in contexts:
            jur = ctx["metadata"].get("jurisdiction", "")
            if jur:
                source_jurisdictions.add(jur)
            content = ctx.get("content", "") or ""
            sources.append({
                "law_id": ctx["metadata"].get("law_id", ""),
                "title": ctx["metadata"].get("title", ""),
                "article_number": ctx["metadata"].get("article_number", ""),
                "jurisdiction": jur,
                "similarity": ctx.get("similarity", 0.0),
                "content": content[:100] + "..." if len(content) > 100 else content,
            })

        yield {
            "type": "sources", "sources": sources,
            "retrieved_count": len(contexts),
            "jurisdictions": list(source_jurisdictions) if not jurisdictions else jurisdictions,
        }
        yield {"type": "status", "stage": "generate", "data": "正在生成回答…"}

        qa = self._qa_service()
        messages_final = qa._build_messages(
            question, contexts, history, kg_text,
            max_primary=max(int(top_k), int(settings.agent_finalize_pass_k)),
        )

        t_gen0 = time.time()
        answer_parts: List[str] = []
        try:
            for chunk in qa._call_llm_stream_with_messages(messages_final):
                answer_parts.append(chunk)
                yield {"type": "content", "data": chunk}
        except Exception as e:
            logger.warning("[Agent] 最终合成失败: %s: %s", type(e).__name__, e)
            if not answer_parts:
                yield {"type": "error", "data": f"生成回答时发生错误：{e}"}
                return
        t_generate = time.time() - t_gen0

        total_time = time.time() - total_start
        logger.info(
            "[Agent] 完成：steps=%d tool_calls=%d llm_turns=%d 证据=%d 图谱=%d stop=%s 耗时=%.2fs",
            len(trace), tool_calls_made, llm_turns + 1, len(contexts), len(graph_parts),
            stop_reason, total_time,
        )
        yield {"type": "end", "cache_hit": False, "stop_reason": stop_reason,
               "tool_calls": tool_calls_made, "timing": {
            "finalize_evidence": f"{t_finalize:.4f}s",
            "generate": f"{t_generate:.4f}s",
            "total": f"{total_time:.4f}s",
        }, "trace": trace}
