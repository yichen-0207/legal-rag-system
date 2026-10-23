"""Agent 服务：基于 LangGraph 的受约束多步推理问答。

流程：问题 → [探索] 模型↔工具循环（检索/引用追溯/图谱）→ [收口] 去重+补全正文+精排一次
→ [合成] 复用 QAService 的证据渲染与流式调用，注入 Agent 专属提示产出分法域对照报告。
框架负责循环编排与工具 schema，自研保留复杂度路由、三重预算（步数/工具次数/时长）、
停滞短路、证据收敛、三层降级与自有 SSE 事件协议；任何开关关闭/无密钥/无证据/异常
均回退 QAService.ask_stream，端点永远返回可用结果。探索步跳过精排（本机精排占检索
延迟约 93%），只对最终证据集精排一次，兼顾成本与质量。
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, Iterator, List, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from core.config import settings
from core.jurisdiction import get_all_jurisdictions
from core.llm import get_chat_model
from services.agent.tools import EVIDENCE_STATUS_TOOL, AgentToolkit
from services.query_decomposer import _CONJUNCTION_RE, _TARGET_ARTICLE_RE
from services.retriever_service import RetrieverService

logger = logging.getLogger(__name__)

# 复杂度路由的语义信号；复用 RetrieverService.COMPARE_KEYWORDS，与检索侧跨法域扇出同口径。
_COMPARE_KEYWORDS = tuple(RetrieverService.COMPARE_KEYWORDS)

# route_mode 取值：auto=按复杂度自动决定；agent=强制多步探索；fast=强制既有快速 RAG
ROUTE_MODE_AUTO = "auto"
ROUTE_MODE_AGENT = "agent"
ROUTE_MODE_FAST = "fast"

# 追问的指代信号：命中才做指代消解（多花一次 LLM 调用）；刻意排除裸「其」「以上」「相关」等高频误命中项。
_ANAPHORA_RE = re.compile(
    r"(它|该|此|上述|前述|前者|后者|这条|那项|这个规定|该项规定|该条|该法规|该法|这|那)"
)
# 省略型追问：去标点后 ≤15 字且以「呢」收尾（如「澳门的呢？」），即便无指代词也依赖上文；
# 门槛不放大，否则会把自足问句误判为追问。
_ELLIPSIS_TAIL = "呢"
_ELLIPSIS_MAX_LEN = 15

# 引用溯源校验：提取答案中的【来源X】编号，与 qa 侧提示词标注格式一致，纯正则零 LLM。
_SOURCE_REF_RE = re.compile(r"【来源(\d+)】")

# decide_route 三态：agent/fast 为强信号直接定夺；probe 为灰区，由调用方做一次廉价检索探测后再定。
DECIDE_AGENT = "agent"
DECIDE_FAST = "fast"
DECIDE_PROBE = "probe"


def decide_route(question: str, route_mode: str = ROUTE_MODE_AUTO,
                 routed_jurisdictions: Optional[List[str]] = None) -> tuple:
    """复杂度路由决策，返回 (decision, reason)，decision ∈ {agent, fast, probe}。

    Agent 比单跳 RAG 稳定多付约 10s 延迟，故强信号（显式 route_mode、定向第X条、
    多法域/对比词/并列连词）直接定夺，灰区返回 probe 交由调用方探测；
    routed_jurisdictions 由调用方注入，避免依赖 RetrieverService 实例以便单测。
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
        self._qa = None  # 延迟获取 QAService（单例），避免构造期触发预热重活
        # 工具运行时持有「单次提问」级的去重 memo，必须按请求新建，否则并发请求会互相
        # 清空/串用缓存（把 A 的检索结果当成 B 的缓存命中）。见 ask_stream。
        self.api_key = settings.llm_api_key or ""
        # LLM 客户端进程级复用：无状态且线程安全，避免每次请求重建 httpx 连接池。
        self._llm = self._make_llm() if self.api_key else None
        # 规划客户端与决策步分开：规划输出短、超时更紧，同样进程级复用（见 _make_plan_llm）。
        self._plan_llm = self._make_plan_llm() if self.api_key else None

    def _qa_service(self):
        if self._qa is None:
            from services.qa_service import QAService
            self._qa = QAService()
        return self._qa

    def _make_llm(self) -> ChatOpenAI:
        """构建共享的 ChatOpenAI 客户端（决策步用它产出 tool_calls）。"""
        # 决策步只需产出 tool_calls 或一句结论；超时与端到端预算同量级，超预算的探索无价值。
        return get_chat_model(
            model=settings.agent_model or settings.llm_qa_model,
            temperature=0.1,
            max_tokens=1024,
            timeout=float(settings.agent_timeout_total),
            max_retries=3,
        )

    def _make_plan_llm(self) -> ChatOpenAI:
        """构建规划专用客户端：短 JSON 输出，超时收紧到 30s；进程级复用以保住连接池。"""
        return get_chat_model(
            model=settings.agent_model or settings.llm_qa_model,
            temperature=0.2,
            max_tokens=int(settings.agent_plan_max_tokens),
            timeout=min(30.0, float(settings.agent_timeout_total)),
            max_retries=1,
        )

    # ================================================================
    #  图与提示
    # ================================================================
    @staticmethod
    def _system_prompt(detected_jurisdictions: List[str]) -> str:
        jur_hint = "、".join(detected_jurisdictions) if detected_jurisdictions else "（未识别到，可按需自行判断）"
        # 把工具预算与「自评通过才停」的停止条件显式告诉模型，否则它会首轮扇出后直接作答，
        # 多步探索退化成单轮检索（深度模式与普通问答无差别）。
        max_calls = max(1, int(settings.agent_max_tool_calls))
        return (
            "你是跨境法规检索分析助手，通过调用工具分步收集证据，最终产出分法域对照的分析报告。\n"
            "【工作方式】\n"
            "1. 先判断问题涉及哪些法域、哪些方面，规划所需证据；起步时可并行发起多条检索"
            "（例如每个法域各一条），无需面面俱到。\n"
            f"2. 本次最多可调用工具 {max_calls} 次；把调用留给最有价值的信息，"
            "不要用相同或高度相似的查询重复检索同一内容。\n"
            "3. 【证据自评，务必执行】每次工具返回后，先自评：问题涉及的每个法域、每个关键"
            "方面是否都已有直接证据？\n"
            "   - 若仍有法域或关键方面缺乏证据，必须继续发起调用补证（换查询措辞、追溯引用"
            "或查询图谱），不得就此作答；\n"
            "   - 仅当各法域/各方面证据齐备，或工具调用预算已用尽时，才输出最终回答。\n"
            "4. 判断可以作答时，直接输出最终回答文字，不要附带工具调用。\n"
            "5. 不要凭空编造法条；工具未返回的内容不要臆断。\n"
            "6. 上下文中若给出了检索计划，请优先按计划推进；判断证据是否充分时，可调用"
            "get_evidence_status 核对已覆盖的法域与缺口，再决定是否补证。\n"
            "【工具】search_laws=检索条文；follow_references=追溯被引用条款；"
            "query_graph=图谱多跳路径（mode=cross 跨法域对齐 / mode=entity 法域内实体关系，"
            "entity 可 deepen=true 纵向深挖；mode=auto 自动判断）；"
            "get_evidence_status=核对证据覆盖（无参数）。\n"
            f"【已知法域】{jur_hint}（若非空，检索时优先使用该法域）。"
        )

    @staticmethod
    def _synthesis_prompt() -> str:
        """Agent 专属合成提示：产出「检索覆盖概览 / 分法域对照 / 差异与共性 / 结论建议」
        的对照式报告，与普通问答的连续解读式回答刻意区分，让深度模式的产出差异可被感知。
        """
        return (
            "你是跨境法规检索分析助手。本次已通过多步工具调用（检索条文 / 追溯引用 / 图谱多跳）"
            "收集证据，请基于这些证据撰写结构化的分法域对照分析报告。\n"
            "\n【报告结构】\n"
            "## 一、检索覆盖概览（80-120字）\n"
            "说明本次证据覆盖了哪些法域、主要涉及哪些法规（《法规名》第X条），并一句话点明问题核心。\n"
            "\n## 二、分法域要点对照\n"
            "按法域分别成节（如「### 澳门」「### 香港」「### 日本」…），每个法域下逐条列出与该问题"
            "相关的关键规定，每条标注出处：法域 + 《法规全称》第 X 条【来源X】；并说明适用范围、"
            "构成要件与实务影响（不要大段照抄原文）。\n"
            "\n## 三、关键差异与共同点\n"
            "用要点或 Markdown 表格，对比各法域在保护力度、适用门槛、责任承担等方面的差异与共性。\n"
            "\n## 四、结论与行动建议\n"
            "给出明确结论与可操作建议，并列 2-3 条延伸提醒。\n"
            "\n【证据边界】\n"
            "- 若某法域或某方面证据不足，必须在对应位置明确写出「现有检索证据未覆盖该方面」，不得略过。\n"
            "- 仅可引用本次【参考资料库】中的【来源X】；历史对话里的旧来源已失效，不得引用。\n"
            "- 所有结论须有【来源X】支撑，严禁编造资料中没有的信息。\n"
            "\n【写作规范】\n"
            "- 使用 Markdown 层级标题，关键术语加粗，法条用行内引用（仅摘录关键短语）。\n"
            "- 总字数 1200-1800 字，以对照分析为主，不堆砌法条原文。"
        )

    def _build_agent(self, system_prompt: str, tools: List[Any]):
        """构建 LangGraph ReAct 图：模型↔工具循环由框架驱动，预算/停滞/证据收敛仍由本服务
        在消费事件流时执行；图按请求构建（工具绑定请求级 memo），模型客户端复用进程级实例。
        """
        return create_agent(self._llm, tools, system_prompt=system_prompt)

    @staticmethod
    def _recursion_limit() -> int:
        """图的超级步上限（防御性第二层预算）：一次「模型决策+工具执行」约 2 步，故取
        `max_steps * 2 + 1`；主预算由服务自身的工具次数与步数计数执行，此处仅防跑飞。
        """
        return max(3, max(1, int(settings.agent_max_steps)) * 2 + 1)

    # ================================================================
    #  显式规划（Plan）：探索前的一次性规划调用
    # ================================================================
    @staticmethod
    def _parse_plan(text: str) -> Optional[Dict]:
        """从规划原文中提取 JSON 计划，解析失败返回 None（调用方静默跳过）；
        容忍 ```json 围栏或夹杂解释文字，取首个 { 到末个 } 之间的片段。
        """
        if not text or not isinstance(text, str):
            return None
        raw = text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw[:4].lower() == "json":
                raw = raw[4:]
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(raw[start:end + 1])
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(parsed, dict):
            return None

        def _clean(key: str, limit: int) -> List[str]:
            vals = parsed.get(key) or []
            if not isinstance(vals, list):
                return []
            return [str(v).strip() for v in vals if str(v).strip()][:limit]

        plan = {
            "sub_questions": _clean("sub_questions", 5),
            "target_jurisdictions": _clean("target_jurisdictions", 5),
            "priority_aspects": _clean("priority_aspects", 5),
        }
        if not any(plan.values()):
            return None
        return plan

    @staticmethod
    def _plan_text(plan: Dict) -> str:
        """把计划渲染为文本并**并入当前用户消息**（作为用户轮前言）：单列 assistant 消息
        会形成错序或让模型直接续写而不发起工具调用，导致零证据降级。
        """
        lines = ["【检索计划（供参考；若工具返回的证据与此不符，一律以证据为准）】"]
        if plan.get("sub_questions"):
            lines.append("拟检索子问题：" + "；".join(plan["sub_questions"]))
        if plan.get("target_jurisdictions"):
            lines.append("拟覆盖法域：" + "、".join(plan["target_jurisdictions"]))
        if plan.get("priority_aspects"):
            lines.append("重点对比方面：" + "、".join(plan["priority_aspects"]))
        return "\n".join(lines)

    @staticmethod
    def _needs_resolution(question: str, history: Optional[List[Dict]]) -> bool:
        """判断追问是否需要指代消解：存在历史且命中指代信号，或去标点后 ≤15 字且以「呢」
        收尾；首轮提问、自足问句返回 False，确保常规问答零额外 LLM 成本。
        """
        if not history:
            return False
        q = (question or "").strip()
        if not q:
            return False
        if _ANAPHORA_RE.search(q):
            return True
        bare = q.rstrip("？?。.！!～~")
        return len(bare) <= _ELLIPSIS_MAX_LEN and bare.endswith(_ELLIPSIS_TAIL)

    def _resolve_query(self, question: str, history: Optional[List[Dict]],
                       detected_jurisdictions: List[str]) -> Optional[str]:
        """把依赖上文的追问改写为自足问题（指代消解），复用规划客户端以免再建连接池。

        **只作检索与规划种子**：合成阶段仍用用户原问题，故改写偏差不改变所回答的问题；
        失败返回 None，调用方回退原问题。
        """
        if not settings.agent_query_rewrite_enabled or self._plan_llm is None:
            return None
        if not self._needs_resolution(question, history):
            return None

        # 只取最近 4 轮作为消解上下文：更早的对话对「它/该规定」指向无帮助，只会稀释提示并推高 token。
        ctx_lines = []
        for h in (history or [])[-4:]:
            role = h.get("role")
            content = (h.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                ctx_lines.append(f"{'用户' if role == 'user' else '助手'}：{content[:200]}")
        if not ctx_lines:
            return None

        jur_hint = "、".join(detected_jurisdictions) if detected_jurisdictions else "未识别"
        sys_prompt = (
            "你是查询改写器。根据对话历史，把用户最新的追问改写成一个可独立检索的自足问题。\n"
            "要求：\n"
            "1. 补全追问中省略的主体（法域、法规、条款、主题），使其脱离上下文也能被理解；\n"
            "2. 不得添加历史中不存在的新事实，不得改变用户意图；\n"
            "3. 仅输出改写后的问题本身，不要解释、不要引号、不要前缀；\n"
            "4. 若追问本身已自足，原样输出。"
        )
        try:
            resp = self._plan_llm.invoke([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": (
                    "【对话历史】\n" + "\n".join(ctx_lines) +
                    f"\n\n【当前法域】{jur_hint}\n【用户追问】{question}"
                )},
            ])
        except Exception as e:
            logger.warning("[Agent] 指代消解失败，回退原问题: %s: %s", type(e).__name__, e)
            return None

        text = getattr(resp, "content", None)
        if isinstance(text, list):
            text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
        resolved = (text or "").strip().strip('"“”「」').strip()
        # 空值、过长、或与原问题完全相同 → 视为无效改写，回退原问题
        if not resolved or len(resolved) > 200 or resolved == question.strip():
            return None
        logger.info("[Agent] 指代消解：%s → %s", question[:40], resolved[:60])
        return resolved

    def _available_jurisdictions(self) -> List[str]:
        """库中实际有语料的法域列表（规划白名单用）；取自 QAService 的 law catalog 映射，
        纯内存读取无 ES/LLM 成本，异常或为空时回退到定义法域。
        """
        try:
            m = self._qa_service()._get_law_jurisdiction_map()
            js = sorted({str(v) for v in (m or {}).values() if v})
            if js:
                return js
        except Exception as e:
            logger.warning("[Agent] 拉取库内法域白名单失败，回退定义法域: %s: %s",
                           type(e).__name__, e)
        return get_all_jurisdictions()

    def _plan(self, question: str, detected_jurisdictions: List[str]) -> Optional[Dict]:
        """一次性规划调用：产出子问题 / 目标法域 / 重点方面，作为探索起点。

        用**独立**的短预算客户端，避免与决策步争用连接池/超时预算；任何异常、超时或
        解析失败都返回 None，由调用方静默跳过——规划失败绝不影响主链路。
        """
        if not settings.agent_plan_enabled or self._plan_llm is None:
            return None
        jur_hint = "、".join(detected_jurisdictions) if detected_jurisdictions else "（未识别）"
        # 法域白名单：只允许规划指向库中实际有语料的法域，避免无数据法域吃掉一次恒零命中
        # 检索；白名单取自 law catalog（零 ES 查询成本），拉取失败回退定义列表，绝不阻断规划。
        allowed = self._available_jurisdictions()
        allowed_text = "、".join(allowed)
        sys_prompt = (
            "你是检索规划器。针对用户的法律问题，规划需要在法规库中检索的证据。"
            "只输出 JSON，不要任何解释文字，格式："
            '{"sub_questions": ["..."], "target_jurisdictions": ["..."], "priority_aspects": ["..."]}。'
            "sub_questions ≤5 条，每条为可直接用于检索的短查询（用常见法律术语，不要生造词）；"
            f"target_jurisdictions 只能从以下白名单中选取，不得出现白名单之外的法域：{allowed_text}，"
            "问题未点名法域时最多选 3 个最相关的；"
            "priority_aspects ≤4 条，为需要重点对比的方面。"
        )
        try:
            resp = self._plan_llm.invoke([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"用户问题：{question}\n已识别法域：{jur_hint}"},
            ])
        except Exception as e:
            logger.warning("[Agent] 规划调用失败，跳过计划注入: %s: %s", type(e).__name__, e)
            return None

        text = getattr(resp, "content", None)
        if text is None:
            text = str(resp)
        elif isinstance(text, list):
            # ChatOpenAI 的 AIMessage.content 可能是 content blocks 列表
            text = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in text
            )
        plan = self._parse_plan(text)
        if plan:
            # 双保险：prompt 约束之外再做交集过滤，白名单外的幻觉法域直接剔除
            before = plan["target_jurisdictions"]
            plan["target_jurisdictions"] = [j for j in before if j in allowed]
            dropped = [j for j in before if j not in allowed]
            if dropped:
                logger.info("[Agent] 规划幻觉法域已剔除：%s", dropped)
            logger.info(
                "[Agent] 规划完成：子问题=%d 法域=%s 方面=%d",
                len(plan["sub_questions"]), plan["target_jurisdictions"],
                len(plan["priority_aspects"]),
            )
        return plan

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
        为引用条款保留配额，避免被检索命中的高相似度挤出候选池。
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

        图谱证据按内容去重（graph_seen）：模型可能换措辞重复查图谱，相同文本重复追加会
        让「是否新增」判定失真，并在最终 prompt 里重复堆叠。
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
            mode = result.get("mode", "")
            count = int(result.get("count") or 0)
            if mode == "flat":
                return f"命中扁平图谱证据（{len(result.get('evidence') or '')} 字）" if count else "图谱无命中"
            label = {"cross": "跨法域对齐", "entity": "实体多跳"}.get(mode, "图谱")
            if not count:
                return "图谱无命中"
            deep = ""
            if (result.get("subgraph") or {}).get("deep_path_count"):
                deep = f"（含纵向深挖 {(result.get('subgraph') or {}).get('deep_path_count')} 条）"
            return f"{label} {count} 条路径{deep}"
        if name == EVIDENCE_STATUS_TOOL:
            return result.get("summary") or "已核对证据覆盖"
        return "完成"

    @staticmethod
    def _converge_candidates(contexts: List[Dict], top_k: int) -> List[Dict]:
        """把证据池收敛到精排候选上限，兼顾成本与召回。

        为「引用追溯」来的条款保留配额（它们相似度恒为 0，不保护会被检索命中挤出），
        剩余名额按相似度降序给检索命中；总条数不超过 agent_finalize_max_candidates
        且不少于最终要用的 top_k。
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

        收敛必须排在精排之前：本机精排占检索延迟约 93%，先压到 agent_finalize_max_candidates
        条再精排是收口耗时的决定性因素；任一步失败均静默降级，不阻断作答。
        返回条数取 max(top_k, agent_finalize_pass_k)，避免多步探索额外召回的证据被截回 top_k。
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

        只做一次 QAService._retrieve_contexts（不含精排）并复用 _is_retrieval_poor 判定，
        不调用 LLM；快速路径内部走的正是同一检索，故探测结果对其有效。
        阈值用 agent_probe_min_similarity（默认 0.7）而非默认 0.6：灰区查询常召回形似
        却未覆盖 gold 的条款，需更高门槛判定证据不足。异常一律返回 True（走快速路径），
        探测只是优化项，不应因自身故障把所有查询推向 Agent。
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
    def _explore(self, agent, messages_in: List[Dict], total_start: float,
                 evidence: Optional[Dict[str, Dict]] = None):
        """同步消费图的事件流，边转发 SSE 边执行预算/停滞策略。

        yield 形状统一为 7 元组 `(event, evidence, graph_parts, trace, tool_calls, turns, stop)`：
        转发事件时仅 `event` 非空，末尾再 yield 一次汇总结果（`event` 为空、其余字段填齐）。
        `evidence` 由调用方传入并**原地累积**，与 get_evidence_status 工具共享同一引用。
        用同步 `stream()` 而非 async：可直接嵌入项目同步 SSE 生成器，避免 async 桥接；
        预算/停滞放在此处而非中间件，便于 trace 说明「为何提前收口」并单测。
        """
        if evidence is None:
            evidence = {}
        graph_parts: List[str] = []
        graph_seen: set = set()
        trace: List[Dict] = []
        pending: Dict[str, Dict] = {}  # tool_call_id → 该次调用的 step/name/args
        tool_calls_made = 0
        model_turns = 0  # 模型决策轮次
        stop_reason = "step_budget"
        # 停滞判定按模型轮次聚合：一轮内任一调用带回新证据即视为有收益（streak 清零）；
        # 按次计数会把并行扇出的部分空手误判为连续低效，导致同轮在途结果被丢弃。
        no_gain_streak = 0
        turn_gained: Dict[int, bool] = {}  # 已出结果的模型轮 → 本轮是否带回新证据
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
                            # 轮末结算：上轮 ToolMessage 已先于本轮 AIMessage 到达，在此判定上轮
                            # 整轮是否有收益；连续零收益轮达阈值才收口（触发即不为新轮发 step）。
                            if model_turns in turn_gained:
                                if turn_gained.pop(model_turns):
                                    no_gain_streak = 0
                                else:
                                    no_gain_streak += 1
                                    if no_gain_streak >= stagnant_limit:
                                        stop_reason = "no_progress"
                                        halt = True
                                        logger.info(
                                            "[Agent] 连续 %d 个模型轮次无新增证据，"
                                            "提前收口（上轮 turn=%d）",
                                            no_gain_streak, model_turns,
                                        )
                            if halt:
                                break
                            calls = list(getattr(m, "tool_calls", None) or [])
                            if not calls:
                                # 模型主动结束探索（认为证据足够 / 无需工具），最理想的收敛
                                stop_reason = "model_stop"
                                continue
                            # 轮边界预算：框架按整轮执行 tool_calls，观察层无法只跑子集；仍有余量
                            # 则整轮受理，触顶则整轮不再受理并在本轮边界收口（不产生半轮在途）。
                            # 代价是计数可超上限至多一轮的调用量（硬上界仍由步数/时长/递归兜底）。
                            if tool_calls_made >= settings.agent_max_tool_calls:
                                stop_reason = "tool_budget"
                                halt = True
                                break
                            model_turns += 1
                            tool_calls_made += len(calls)
                            for tc in calls:
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
                            is_neutral = info["name"] == EVIDENCE_STATUS_TOOL
                            gained = self._absorb_evidence(
                                info["name"], result, evidence, graph_parts, graph_seen)
                            summary = self._summarize_result(info["name"], result)
                            trace.append({"step": info["step"], "tool": info["name"],
                                          "args": info["args"], "summary": summary})
                            yield ({"type": "observation", "step": info["step"],
                                    "tool": info["name"], "summary": summary}, None, None, None, None, None, None)

                            # get_evidence_status 只读、天然不产生新证据，不参与停滞判定，
                            # 否则「模型自查覆盖」这一正向行为会反过来触发提前收口。
                            if is_neutral:
                                continue
                            # 轮粒度聚合：合并进所属模型轮的收益标记（任一调用有证据即该轮有收益）
                            turn_gained[info["step"]] = turn_gained.get(info["step"], False) or gained
                if halt:
                    break
        except Exception as e:
            # 递归上限（步数耗尽）与网络/模型异常都归入预算耗尽，用已得证据作答；
            # 按类名识别递归上限，避免依赖具体异常类路径。
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
        """多步推理问答；开关关闭/无密钥/无证据/异常一律降级到既有 RAG，保证有结果。

        route_mode：auto=按复杂度规则决定；agent=强制探索（前端「深度模式」）；
        fast=强制既有快速 RAG。简单/定向查询走快速路径，避免用其延迟换复杂查询的收益。
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

        # 复杂度路由：仅 auto 模式需要判定；多法域信号复用检索侧的规则路由（廉价、无副作用）。
        if route_mode == ROUTE_MODE_AUTO and settings.agent_complexity_routing_enabled:
            try:
                routed = self.retriever._route_to_jurisdictions(question)
            except Exception:
                routed = None
        else:
            routed = None
        decision, route_reason = decide_route(question, route_mode, routed)

        # 灰区：做一次廉价检索探测（不生成答案）决定走哪条路；探测失败保守走快速路径。
        if decision == DECIDE_PROBE:
            if self._probe_retrieval_quality(question, top_k, jurisdictions):
                decision, route_reason = DECIDE_FAST, route_reason + ":probe_ok"
            else:
                decision, route_reason = DECIDE_AGENT, route_reason + ":probe_poor"

        if decision == DECIDE_FAST:
            logger.info("[Agent] 复杂度路由→快速 RAG（reason=%s）", route_reason)
            yield {"type": "route", "mode": "fast", "reason": route_reason}
            # 透传用户原始 jurisdictions（可能为 None）：与 /qa/stream 完全同路径，保证
            # 「快速路径」行为与既有问答一致。
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

        # 追问指代消解：把「它/该规定/那例外呢？」这类追问改写成自足检索式；只作检索与
        # 规划种子，合成阶段仍用原问题，故改写偏差不会改变答案所回答的问题本身。
        retrieval_query = self._resolve_query(question, history, detected) or question
        if retrieval_query != question:
            yield {"type": "rewrite", "original": question, "rewritten": retrieval_query}

        # 显式规划：用检索式做规划，使子问题不依赖上下文；失败返回 None，仅跳过注入与事件。
        plan = self._plan(retrieval_query, detected)
        if plan:
            yield {"type": "plan", "sub_questions": plan["sub_questions"],
                   "jurisdictions": plan["target_jurisdictions"],
                   "aspects": plan["priority_aspects"]}

        # 证据池在探索前创建，工具运行时（get_evidence_status）与 _explore（原地累积）
        # 共享同一引用，保证跨步工作记忆看到实时台账。
        evidence: Dict[str, Dict] = {}

        # 每请求独立的工具运行时与图：memo 为单次提问级缓存，须与请求同生命周期，
        # 并发请求各持一份、互不干扰（见 __init__）。
        toolkit = AgentToolkit(self.retriever, self._qa_service,
                               jurisdictions=detected, evidence=evidence)
        agent = self._build_agent(self._system_prompt(detected), toolkit.build_tools())

        messages_in: List[Dict] = []
        if history:
            for h in history[-6:]:
                if h.get("role") in ("user", "assistant") and h.get("content"):
                    messages_in.append({"role": h["role"], "content": h["content"]})
        # 消解结果与计划并入当前用户轮前言，避免单列 assistant 消息造成错序；
        # 两者都不存在时 user_content 即原问题。
        prefix_parts: List[str] = []
        if retrieval_query != question:
            prefix_parts.append(
                "【追问指代消解（检索/规划可采用此自足表述，但请回答用户原问题）】" + retrieval_query
            )
        if plan:
            prefix_parts.append(self._plan_text(plan))
        user_content = ("\n\n".join(prefix_parts) + "\n\n【用户问题】" + question) if prefix_parts else question
        messages_in.append({"role": "user", "content": user_content})

        # 探索：由 _explore 消费图事件流并执行预算/停滞策略
        graph_parts: List[str] = []
        trace: List[Dict] = []
        tool_calls_made = 0
        llm_turns = 0
        stop_reason = "step_budget"
        for ev, ev_evidence, ev_parts, ev_trace, ev_calls, ev_turns, ev_stop in self._explore(
                agent, messages_in, total_start, evidence=evidence):
            if ev is not None:
                yield ev
                continue
            evidence, graph_parts, trace = ev_evidence, ev_parts, ev_trace
            tool_calls_made, llm_turns, stop_reason = ev_calls, ev_turns, ev_stop

        # 探索未产出任何证据 → 降级到既有 RAG；并在 end 事件补 stop_reason，让降级可解释。
        if not evidence and not graph_parts:
            logger.info("[Agent] 探索未产出证据，降级为既有 RAG 链路（turns=%d）", llm_turns)
            for ev in self._qa_service().ask_stream(question, top_k, jurisdictions, history):
                if isinstance(ev, dict) and ev.get("type") == "end":
                    ev = dict(ev)
                    ev.setdefault("stop_reason", "no_evidence_degraded")
                yield ev
            return

        # ---- 证据收口 + 最终合成（复用 QA 的合成 prompt 与流式调用）----
        # 收口要跑一次精排（本机占检索延迟约 93%）且尚无正文输出，先推状态事件避免前端
        # 在精排期间长时间无反馈。
        yield {"type": "status", "stage": "finalize",
               "data": f"已收集 {tool_calls_made} 次工具证据，正在收口精排…"}
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
        yield {"type": "status", "stage": "generate", "data": "正在生成分法域对照分析…"}

        qa = self._qa_service()
        messages_final = qa._build_messages(
            question, contexts, history, kg_text,
            max_primary=max(int(top_k), int(settings.agent_finalize_pass_k)),
            system_prompt_override=self._synthesis_prompt(),
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

        # 引用溯源校验：纯正则比对答案中的【来源X】与实际来源数；答案已流式发出无法撤回，
        # 故越界引用只通过 end 事件提示前端，不改正文。
        answer_full = "".join(answer_parts)
        ref_nums = [int(m) for m in _SOURCE_REF_RE.findall(answer_full)]
        invalid_refs = sorted({n for n in ref_nums if n < 1 or n > len(sources)})
        citation_check = {
            "ref_count": len(ref_nums),
            "invalid_count": len(invalid_refs),
            "invalid_refs": invalid_refs,
            "source_count": len(sources),
        }
        if invalid_refs:
            logger.warning("[Agent] 引用越界：来源总数=%d，越界引用=%s",
                           len(sources), invalid_refs)

        total_time = time.time() - total_start
        logger.info(
            "[Agent] 完成：steps=%d tool_calls=%d llm_turns=%d 证据=%d 图谱=%d stop=%s 耗时=%.2fs",
            len(trace), tool_calls_made, llm_turns + 1, len(contexts), len(graph_parts),
            stop_reason, total_time,
        )
        yield {"type": "end", "cache_hit": False, "stop_reason": stop_reason,
               "tool_calls": tool_calls_made, "citation_check": citation_check,
               "timing": {
            "finalize_evidence": f"{t_finalize:.4f}s",
            "generate": f"{t_generate:.4f}s",
            "total": f"{total_time:.4f}s",
        }, "trace": trace}
