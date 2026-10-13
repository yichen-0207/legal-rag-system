"""多跳 B：查询分解（规则前置 → LLM 兜底 → 失败降级单跳）。

把「工伤赔偿和年假规定有什么要求」这类复合查询拆成若干单一意图子查询，交给
RetrieverService 并行检索后合并；规则无法判定时再用 LLM 兜底。

设计约束：
  - 规则前置：能从字面确定性切分就不调 LLM（省时、确定、可复现）；
  - LLM 兜底：仅当查询「明显复合但规则切不干净」时启用，且带超时；
  - 失败降级：任何异常/超时/解析失败一律返回 []，调用方回退单跳，绝不阻断检索；
  - 不处理「多法域」查询：那类查询已由 RetrieverService._route_to_jurisdictions
    的跨法域路由负责，此处主动让路，避免重复检索与语义重叠。
"""
from __future__ import annotations

import json
import logging
import re
from typing import List

from core.config import settings

logger = logging.getLogger(__name__)

# 法域中英文别名：用于识别「多法域」查询并让路给既有路由。
_JUR_ALIASES = {
    "澳门": ["澳门", "澳門", "macau", "macao"],
    "香港": ["香港", "hong kong"],
    "新加坡": ["新加坡", "singapore"],
    "日本": ["日本", "japan"],
    "美国": ["美国", "美國", "united states", "usa"],
    "欧盟": ["欧盟", "歐盟", "european union", "eu"],
}

# 并列 / 对比标志：出现其一才认为查询可能复合。
_CONJUNCTION_RE = re.compile(r"以及|並且|并且|和|與|与|跟|、|對比|对比|比較|比较|分別|分别|各自")

# 定向查询（点名「第 X 条」）不参与分解：置顶逻辑在单跳路径里，分解会绕过它。
_TARGET_ARTICLE_RE = re.compile(r"第\s*\d+\s*[条條]")

# 触发 LLM 兜底的最小查询长度：太短的句子几乎不会是真正的复合问题，
# 用长度门槛挡掉「和平与发展」这类被误判的短语，避免无谓的 LLM 调用与延迟。
_LLM_MIN_LEN = 12
# 规则切分后每段的最小长度（字符）。
_MIN_PART_LEN = 3


def _detect_jurisdictions(text: str) -> List[str]:
    """返回查询中出现过的法域（去重，保持别名表顺序）。"""
    lowered = text.lower()
    return [zh for zh, aliases in _JUR_ALIASES.items()
            if any(a.lower() in lowered for a in aliases)]


# 共享限定语的边界标志：并列结构右半段通常省略了左半段已有的主语/限定语
# （「美国对人工智能技术的监管框架和治理体系」→ 右半段需补回「美国对人工智能技术的」）。
_CONTEXT_BOUNDARY = ("的", "對", "对", "關於", "关于", "就")


def _context_prefix(left: str) -> str:
    """取左半段最后一个限定语边界（的/对/关于/就…）及其之前的部分，作为右半段的共享前缀。"""
    idx = -1
    for marker in _CONTEXT_BOUNDARY:
        pos = left.rfind(marker)
        if pos > idx:
            idx = pos
    return left[: idx + 1] if idx >= 0 else ""


def _split_by_conjunction(query: str) -> List[str]:
    """按第一个并列/对比标志把查询一分为二；任一侧过短则视为切不干净。

    右半段补上左半段的共享限定语，避免产生「治理体系是怎样的」这类丢失主体的
    子查询——那会召回大量无关文档，把噪声带进合并候选池。
    """
    m = _CONJUNCTION_RE.search(query)
    if not m:
        return []
    left = query[:m.start()].strip(" \t，,、。；;：:")
    right = query[m.end():].strip(" \t，,、。；;：:")
    if len(left) < _MIN_PART_LEN or len(right) < _MIN_PART_LEN:
        return []
    prefix = _context_prefix(left)
    right_full = (prefix + right) if prefix else right
    return [left, right_full]


def _parse_llm_subqueries(text: str, original: str) -> List[str]:
    """从 LLM 输出里稳定地取出子查询列表（优先 JSON 数组，退化为逐行）。"""
    parts: List[str] = []
    m = re.search(r"\[.*\]", text or "", re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            parts = [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            parts = []
    if not parts:
        parts = [ln.strip(" \t-•*0123456789.、)") for ln in (text or "").splitlines() if ln.strip()]

    out: List[str] = []
    for p in parts:
        if p and p != original and len(p) >= 2 and p not in out:
            out.append(p)
    return out


def _llm_decompose(query: str) -> List[str]:
    """LLM 兜底：让模型把复合查询拆成 2~3 条自包含子查询。失败返回 []。"""
    if not settings.multihop_decompose_llm_enabled:
        return []
    api_key = settings.llm_api_key or ""
    if not api_key:
        return []
    base_url = settings.llm_api_base_url or "https://api.deepseek.com/v1"
    model = settings.llm_model or "deepseek-flash"
    prompt = (
        "你是法规检索的查询改写助手。把下面这条可能包含多个问题的复合查询，"
        "拆成 2~3 条各自独立、语义完整、可直接用于检索的子查询。\n"
        "要求：\n"
        "1. 每条子查询自包含，不要出现「上述/前者/后者/它」等指代；\n"
        "2. 保留原查询中的法域、时间范围等限定条件；\n"
        "3. 如果它其实只有一个问题、无法拆解，返回空数组 []；\n"
        "4. 只输出 JSON 数组，例如 [\"子查询1\",\"子查询2\"]，不要输出解释。\n"
        f"查询：{query}"
    )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url,
                        timeout=settings.multihop_decompose_llm_timeout)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
            stream=False,
        )
        text = (resp.choices[0].message.content or "").strip()
        return _parse_llm_subqueries(text, query)
    except Exception as e:  # 超时/网络/解析等一律降级
        logger.warning("LLM 查询分解失败，降级单跳: %s", e)
        return []


def decompose_query(query: str) -> List[str]:
    """把复合查询拆成子查询列表；无法/无需分解时返回 []。

    返回的子查询**不包含**原查询——调用方负责把原查询作为锚点一并检索，
    从而保证分解只增加候选、不丢原有召回。
    """
    q = (query or "").strip()
    if len(q) < 4:
        return []

    # 定向查询不分解（会绕过「第 X 条」置顶逻辑）
    if _TARGET_ARTICLE_RE.search(q):
        return []

    # 多法域查询让路给既有跨法域路由，避免重复检索
    if len(_detect_jurisdictions(q)) >= 2:
        return []

    max_sub = max(2, settings.multihop_decompose_max_subqueries - 1)  # 预留 1 个给锚点

    # 规则前置：必须出现并列/对比标志才尝试切分
    if _CONJUNCTION_RE.search(q):
        parts = _split_by_conjunction(q)
        if len(parts) >= 2:
            return parts[:max_sub]
        # 规则切不干净但查询明显复合 → LLM 兜底
        if len(q) >= _LLM_MIN_LEN:
            parts = _llm_decompose(q)
            if len(parts) >= 2:
                return parts[:max_sub]

    return []
