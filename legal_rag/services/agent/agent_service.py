"""Agent 服务：LLM 迭代式工具调用问答（ReAct 风格的受约束实现）。

流程：
    问题 → [探索循环] LLM 决定调用哪些工具（检索/引用追溯/图谱查询）
                ↓ 工具结果作为 observation 回灌，直至模型认为证据足够或达到步数上限
         → [证据收口] 对累积证据去重 + 补全正文（一次 msearch）+ 精排一次
         → [最终合成] 复用 QAService 的合成 prompt 流式产出带来源标注的答案

设计取舍（与项目既有约束一致）：
  - **受约束而非放任**：步数、工具调用数、端到端时长三重上限，超限强制收口；
  - **探索便宜、合成讲究**：探索步检索跳过精排（本机精排占检索延迟约 93%），
    只对最终证据集精排一次，兼顾成本与质量；
  - **最终合成复用既有 RAG**：调用 QAService._build_messages，保证回答结构、
    【来源X】标注、图谱证据口径与现有问答完全一致，不另造一套输出格式；
  - **全链路降级**：开关关闭 / 模型不支持工具调用 / 探索未产出证据 / 任何异常，
    一律回退到既有 QAService.ask_stream，端点永远返回可用结果。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Dict, Iterator, List, Optional

from openai import OpenAI

from core.config import settings
from services.agent.tools import AgentTools, TOOL_SCHEMAS
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
    """迭代式工具调用问答服务。"""

    def __init__(self):
        self.retriever = RetrieverService()
        self._qa = None  # 延迟获取 QAService（单例），避免构造期触发预热相关重活
        # 本服务是进程级单例，但**工具执行器不挂在实例上**：其内部 memo 是「单次提问」
        # 级别的去重缓存，必须在 ask_stream 内按请求新建，否则并发请求会互相清空/串用
        # 对方的 memo（把 A 的检索结果当成 B 的缓存命中）。见 ask_stream 中的构造注释。
        self.api_key = settings.llm_api_key or ""
        self.base_url = settings.llm_api_base_url or "https://api.deepseek.com/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.api_key else None

    def _qa_service(self):
        if self._qa is None:
            from services.qa_service import QAService
            self._qa = QAService()
        return self._qa

    # ================================================================
    #  探索循环
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

    def _chat(self, messages: List[Dict]):
        """一次带工具的 LLM 决策调用（非流式：需要拿到 tool_calls）。"""
        kwargs = {
            "model": settings.agent_model or settings.llm_qa_model,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
            "temperature": 0.1,
            "max_tokens": 1024,
        }
        extra_body = settings.llm_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        return self.client.chat.completions.create(**kwargs)

    @staticmethod
    def _assistant_message_dict(msg) -> Dict:
        """把 SDK 的 assistant 消息（含 tool_calls）转成可回传的纯 dict。"""
        tool_calls = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            tool_calls.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            })
        out: Dict = {"role": "assistant", "content": msg.content or ""}
        if tool_calls:
            out["tool_calls"] = tool_calls
        return out

    # ================================================================
    #  证据累积与收口
    # ================================================================
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
    #  主流程（SSE 事件生成器）
    # ================================================================
    def ask_stream(self, question: str, top_k: int = 5,
                   jurisdictions: Optional[List[str]] = None,
                   history: Optional[List[Dict]] = None,
                   route_mode: str = ROUTE_MODE_AUTO) -> Iterator[Dict]:
        """迭代式工具调用问答；任何降级路径都回退到既有 RAG，保证有结果。

        route_mode 决定「是否值得走多步探索」：
          - auto（默认）：按 should_use_agent 的复杂度规则决定；
          - agent：强制走 Agent（前端「深度模式」）；
          - fast：强制走既有快速 RAG。
        简单/定向查询走快速 RAG，避免用简单查询的延迟换复杂查询的收益。
        """
        # 开关关闭：直接走既有 RAG（端点仍可用）
        if not settings.agent_enabled:
            logger.info("[Agent] agent_enabled=false，降级为既有 RAG 链路")
            yield from self._qa_service().ask_stream(question, top_k, jurisdictions, history)
            return

        if not self.client:
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

        # 每请求独立的工具执行器：其 memo 是「单次提问」级去重缓存，必须与请求同生命周期；
        # 并发请求各持一份、互不干扰（详见 __init__ 说明）。jurisdictions 供图谱工具做
        # 法域内证据过滤，避免跨法域无关证据污染答案。
        tools = AgentTools(self.retriever, self._qa_service, jurisdictions=detected)
        messages: List[Dict] = [{"role": "system", "content": self._system_prompt(detected)}]
        if history:
            for h in history[-6:]:
                if h.get("role") in ("user", "assistant") and h.get("content"):
                    messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": question})

        evidence: Dict[str, Dict] = {}
        graph_parts: List[str] = []
        graph_seen: set = set()
        trace: List[Dict] = []
        tool_calls_made = 0
        llm_calls = 0
        # 探索结束原因：用于日志与 end 事件归因（评测时区分「模型自然停」与「被预算截断」）。
        stop_reason = "step_budget"
        # 停滞计数：连续多少次工具调用没给证据池带来新增。模型偶尔会用不同措辞重复
        # 检索同一内容，这类调用不产生新证据，留着只会拖长延迟，故达到阈值即收口。
        no_gain_streak = 0
        stagnant_limit = max(1, int(settings.agent_max_stagnant_steps))

        for step in range(1, max(1, settings.agent_max_steps) + 1):
            if time.time() - total_start > settings.agent_timeout_total:
                stop_reason = "timeout"
                logger.info("[Agent] 超出时长上限，收口作答（step=%d）", step)
                break
            try:
                resp = self._chat(messages)
                llm_calls += 1
            except Exception as e:
                stop_reason = "llm_error"
                logger.warning("[Agent] 探索步 LLM 调用失败（step=%d）: %s: %s", step, type(e).__name__, e)
                break

            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                # 模型主动结束探索（认为证据足够 / 无需工具），这是最理想的收敛方式
                stop_reason = "model_stop"
                break

            messages.append(self._assistant_message_dict(msg))
            tool_budget_hit = False
            for tc in tool_calls:
                if tool_calls_made >= settings.agent_max_tool_calls:
                    stop_reason = "tool_budget"
                    tool_budget_hit = True
                    break
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except Exception:
                    args = {}
                tool_calls_made += 1

                yield {"type": "step", "step": step, "action": "tool_call", "tool": name, "args": args}
                result = tools.execute(name, args)
                gained = self._absorb_evidence(name, result, evidence, graph_parts, graph_seen)
                summary = self._summarize_result(name, result)
                trace.append({"step": step, "tool": name, "args": args, "summary": summary})
                yield {"type": "observation", "step": step, "tool": name, "summary": summary}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

                # 停滞短路：连续多次调用都没带来新证据 → 判定为低效重复，立即收口。
                if gained:
                    no_gain_streak = 0
                else:
                    no_gain_streak += 1
                    if no_gain_streak >= stagnant_limit:
                        stop_reason = "no_progress"
                        logger.info(
                            "[Agent] 连续 %d 次工具调用无新增证据，提前收口（step=%d, tool=%s）",
                            no_gain_streak, step, name,
                        )
                        break
            if tool_budget_hit or stop_reason == "no_progress":
                break

        # 探索未产出任何证据 → 降级到既有 RAG（保证有答案，覆盖简单问题/工具全失败）
        if not evidence and not graph_parts:
            logger.info("[Agent] 探索未产出证据，降级为既有 RAG 链路（steps=%d）", len(trace))
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
            "[Agent] 完成：steps=%d tool_calls=%d llm_calls=%d 证据=%d 图谱=%d stop=%s 耗时=%.2fs",
            len(trace), tool_calls_made, llm_calls + 1, len(contexts), len(graph_parts),
            stop_reason, total_time,
        )
        yield {"type": "end", "cache_hit": False, "stop_reason": stop_reason,
               "tool_calls": tool_calls_made, "timing": {
            "finalize_evidence": f"{t_finalize:.4f}s",
            "generate": f"{t_generate:.4f}s",
            "total": f"{total_time:.4f}s",
        }, "trace": trace}
