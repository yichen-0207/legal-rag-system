"""检索决策规划器（轻量、纯规则、零 LLM）。

只做决策、不做检索；探测能力由调用方注入，避免与 RetrieverService 的大表重复维护并防循环导入。
优先级：子查询不分解不路由 → 分解优先于路由，但多法域查询让路（否则绕过扇出丢召回）→ 仅在未指定且未检测到单一法域时路由。
plan.reasons 供日志排障，回答「这次查询为什么走了这条路」。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievalPlan:
    """一次检索的执行计划（无副作用）。

    mode 互斥三选一："single" 单跳 / "decompose" 分解并行 / "route" 跨法域扇出；
    decompose 时 jurisdiction 为分解失败回退单跳所用的法域；use_kg 仅供问答链路观测。
    """

    mode: str
    jurisdiction: Optional[str] = None
    jurisdictions: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    expand_refs: bool = False
    use_kg: bool = False
    reasons: List[str] = field(default_factory=list)


def _safe(fn: Callable, arg: str):
    """执行探测函数，异常时降级为 None（不阻断检索决策）。"""
    try:
        return fn(arg)
    except Exception as e:  # 探测失败一律降级，保证决策层不抛异常
        logger.warning("[Planner] 探测函数 %s 执行失败: %s", getattr(fn, "__name__", fn), e)
        return None


def plan_retrieval(
    query: str,
    jurisdiction: Optional[str] = None,
    *,
    allow_decompose: bool = True,
    rerank_active: bool = True,
    detect_jurisdiction: Callable[[str], Optional[str]],
    route_jurisdictions: Callable[[str], Optional[List[str]]],
    decompose_subqueries: Callable[[str], List[str]],
) -> RetrievalPlan:
    """产出一份检索执行计划（无副作用，不触发检索）。

    allow_decompose 子查询传 False（不分解、不路由）；rerank_active 为 False 时不分解——
    无精排「合并后用原查询统一纠偏」时，分解引入的噪声无法纠正，净负收益。
    """
    q = (query or "").strip()
    expand_refs = bool(rerank_active and settings.reference_expand_enabled)
    use_kg = bool(settings.qa_kg_evidence_enabled)

    # ① 子查询：不分解、不路由，仅按传入法域检索
    if not allow_decompose:
        return RetrievalPlan(
            mode="single", jurisdiction=jurisdiction,
            expand_refs=expand_refs, use_kg=use_kg,
            reasons=["子查询：跳过分解与路由"],
        )

    # 多法域信号：分解必须让路（否则绕过扇出路由丢召回）
    routed = _safe(route_jurisdictions, q)
    routed_multi = bool(routed and len(routed) > 1)

    # ② 复合查询分解（优先于路由；让路多法域）
    if (rerank_active and settings.reranker_enabled
            and settings.multihop_decompose_enabled and not routed_multi):
        subs = _safe(decompose_subqueries, q)
        if subs and len(subs) > 1:
            # 分解成功：法域沿用显式传入值，否则自动检测备用
            fallback_jur = jurisdiction if jurisdiction is not None else _safe(detect_jurisdiction, q)
            return RetrievalPlan(
                mode="decompose", sub_queries=list(subs), jurisdiction=fallback_jur,
                expand_refs=expand_refs, use_kg=use_kg,
                reasons=["复合查询：分解为子查询并行检索"],
            )

    # ③ 显式指定法域：单跳限定，不做路由
    if jurisdiction is not None:
        return RetrievalPlan(
            mode="single", jurisdiction=jurisdiction,
            expand_refs=expand_refs, use_kg=use_kg,
            reasons=["已显式指定法域"],
        )

    # ④ 自动法域检测：命中单一法域则限定单跳
    detected = _safe(detect_jurisdiction, q)
    if detected is not None:
        return RetrievalPlan(
            mode="single", jurisdiction=detected,
            expand_refs=expand_refs, use_kg=use_kg,
            reasons=[f"检测到单一法域：{detected}"],
        )

    # ⑤ 跨法域扇出路由（区域词 / 比较词触发）
    if routed_multi:
        return RetrievalPlan(
            mode="route", jurisdictions=list(routed),
            expand_refs=expand_refs, use_kg=use_kg,
            reasons=[f"比较/区域查询：跨法域扇出（{len(routed)} 个法域）"],
        )

    # ⑥ 全域单跳
    return RetrievalPlan(
        mode="single", jurisdiction=None,
        expand_refs=expand_refs, use_kg=use_kg,
        reasons=["全域单跳检索"],
    )
