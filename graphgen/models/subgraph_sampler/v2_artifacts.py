from dataclasses import asdict, dataclass, field
from typing import Any

from .artifacts import JudgeScorecard


@dataclass
class GraphEditAction:
    action_type: str
    node_id: str = ""
    anchor_node_id: str = ""
    src_id: str = ""
    tgt_id: str = ""
    intent: str = ""
    technical_focus: str = ""
    approved_question_types: list[str] = field(default_factory=list)
    note: str = ""
    applied: bool = False
    ignored_reason: str = ""
    before_units: int = 0
    after_units: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateSubgraphState:
    candidate_id: str
    seed_node_id: str
    intent: str = ""
    technical_focus: str = ""
    approved_question_types: list[str] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    edge_pairs: list[list[str]] = field(default_factory=list)
    image_grounding_summary: str = ""
    evidence_summary: str = ""
    status: str = "editing"
    degraded: bool = False

    def unit_count(self) -> int:
        return len(self.node_ids) + len(self.edge_pairs)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["unit_count"] = self.unit_count()
        return payload


@dataclass
class JudgeFeedback:
    scorecard: JudgeScorecard
    sufficient: bool = False
    needs_expansion: bool = False
    rejection_reason: str = ""
    suggested_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scorecard": self.scorecard.to_dict(),
            "sufficient": self.sufficient,
            "needs_expansion": self.needs_expansion,
            "rejection_reason": self.rejection_reason,
            "suggested_actions": list(self.suggested_actions),
        }


@dataclass
class AgentSessionTrace:
    session_id: str
    seed_node_id: str
    rounds: int = 0
    degraded: bool = False
    status: str = "editing"
    termination_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
