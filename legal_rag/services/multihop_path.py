"""多跳路径的共享协议：P1 跨法域对齐链与 P2 法域内实体导航链共用同一路径结构，前端只需一套渲染。

可靠性分级是协议的一部分：证据强度不同必须在数据层区分，否则答案会把「对应法规」说成「同一法规」。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

# 来源 → 可靠性等级：L3 确定性 / L2 精确 / L1 相似（不得表述为「同一」）
_SOURCE_LEVELS: Dict[str, str] = {
    "citation_regex": "L3",
    "kg_edge": "L2",
    "concept_matrix": "L2",
    "similar_law": "L1",
    "retrieval": "L1",
}
_SOURCE_LABELS: Dict[str, str] = {
    "citation_regex": "明确援引",
    "kg_edge": "图谱关系",
    "concept_matrix": "同一概念",
    "similar_law": "对应法规（主题相似）",
    "retrieval": "检索命中",
}
_LEVEL_RANK = {"L3": 3, "L2": 2, "L1": 1}


@dataclass
class HopStep:
    """路径上的一步。from_ref → to_ref 为遍历方向，direction 标明与三元组原方向的关系。"""

    hop: int
    kind: str
    from_ref: str
    to_ref: str
    relation: str = ""
    source: str = "kg_edge"
    confidence: float = 0.0
    direction: str = "forward"  # forward=沿三元组方向；reverse=逆向遍历
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def level(self) -> str:
        return _SOURCE_LEVELS.get(self.source, "L1")

    @property
    def source_label(self) -> str:
        return _SOURCE_LABELS.get(self.source, "检索命中")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hop": self.hop,
            "kind": self.kind,
            "from_ref": self.from_ref,
            "to_ref": self.to_ref,
            "relation": self.relation,
            "source": self.source,
            "level": self.level,
            "source_label": self.source_label,
            "confidence": round(float(self.confidence), 3),
            "direction": self.direction,
            "evidence": self.evidence,
        }


@dataclass
class ReasoningPath:
    """一条完整的多跳路径；整体可靠性取各步中最弱的一级（短板原则）。"""

    path_id: str
    steps: List[HopStep] = field(default_factory=list)
    score: float = 0.0
    # 内部字段：节点 ID 序列，供纵向深挖从路径末端继续扩展；不对外序列化
    node_ids: List[str] = field(default_factory=list, repr=False)

    @property
    def level(self) -> str:
        if not self.steps:
            return "L1"
        return min((s.level for s in self.steps), key=lambda lv: _LEVEL_RANK.get(lv, 1))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "hop_count": len(self.steps),
            "level": self.level,
            "score": round(float(self.score), 3),
            "text": path_text(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }


def step_text(step: HopStep) -> str:
    """单步的可读形式，逆向遍历用反向箭头，保证语义不被读反。"""
    if step.direction == "reverse":
        return f"{step.from_ref} ←{step.relation}— {step.to_ref}"
    return f"{step.from_ref} —{step.relation}→ {step.to_ref}"


def path_text(steps: List[HopStep]) -> str:
    return " ⟶ ".join(step_text(s) for s in steps)
