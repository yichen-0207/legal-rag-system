"""检索决策规划器（轻量、纯规则、零 LLM）。

把原本散落在 RetrieverService 里的检索决策——「是否做复合查询分解」「是否跨法域
扇出路由」「是否做引用链扩展」——收敛到一个可观测、可测试、无副作用的决策层，
对外输出一个 RetrievalPlan。

设计约束（保行为）：
  - 本模块只做「决策」，不做任何检索；探测能力（法域检测 / 跨法域路由 / 查询分解）
    由调用方以可调用对象注入，避免与 RetrieverService 的 REGION_MAP、ZH_EN_MAP 等
    大表重复维护，也避免循环导入；
  - 决策优先级与旧实现**逐条对齐**，切换本模块不改变任何既有查询的行为：
      ① 子查询（allow_decompose=False）不分解、不路由；
      ② 分解优先于路由，且多法域查询「让路」给路由（否则分解会绕过扇出、丢召回，
         这是历史踩坑点）；
      ③ 路由仅在未显式指定法域、且未检测到单一法域时触发；
  - 不引入 LLM 规划：保留「规则前置」的确定性，Agent 阶段只需替换本函数即可。

可观测性：plan 内含 reasons，调用方据此打日志，回答「这次查询为什么走了这条路」。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievalPlan:
    """一次检索的执行计划。

    Attributes:
        mode: 执行模式，三选一且互斥：
            - "single"    : 单跳检索（可限定单一法域）
            - "decompose" : 复合查询分解为多个子查询并行检索后合并
            - "route"     : 跨法域扇出路由（每个法域独立召回后 RRF 合并）
        jurisdiction: single 模式下限定的法域（None 表示不限）；
                      decompose 模式下为「分解失败回退单跳」时应使用的法域。
        jurisdictions: route 模式下的目标法域列表。
        sub_queries: decompose 模式下锚定原查询后的子查询列表。
        expand_refs: 是否在单跳结果上做引用链扩展。
        use_kg: 是否注入知识图谱补充证据（供问答链路观测/对齐，检索侧不使用）。
        reasons: 决策依据，供日志与排障。
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

    Args:
        query: 原始查询串。
        jurisdiction: 调用方显式指定的法域（None 表示未指定）。
        allow_decompose: 是否允许分解。顶层调用为 True；子查询传 False
            （对应旧实现的 _decompose=False，子查询不再分解、不参与路由）。
        rerank_active: 当前是否走精排路径（对应旧实现的 _rerank）。分解依赖
            「合并后用原查询统一精排」来纠偏，无精排时分解引入的噪声无法被纠正，
            故不分解以避免净负收益。
        detect_jurisdiction: 法域自动检测函数，签名 (query) -> Optional[str]。
        route_jurisdictions: 跨法域路由检测函数，签名 (query) -> Optional[List[str]]。
        decompose_subqueries: 复合查询分解函数，返回含锚定原查询的子查询列表；
            无法/无需分解时返回长度 < 2 的列表。

    Returns:
        RetrievalPlan
    """
    q = (query or "").strip()
    expand_refs = bool(rerank_active and settings.reference_expand_enabled)
    use_kg = bool(settings.qa_kg_evidence_enabled)

    # ① 子查询：不分解、不路由，仅按传入法域检索（对应旧实现的 _decompose=False）
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
            # 分解成功。法域沿用显式传入值；若分解失败回退单跳，再按需自动检测。
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
