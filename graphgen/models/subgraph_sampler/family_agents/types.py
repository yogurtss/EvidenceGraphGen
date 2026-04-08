from dataclasses import asdict, dataclass, field
from typing import Any

from graphgen.models.subgraph_sampler.artifacts import JudgeScorecard


@dataclass
class CandidatePoolItem:
    candidate_node_id: str
    bind_from_node_id: str
    bound_edge_pair: list[str]
    hop: int
    theme_signature: str
    frontier_path: list[str] = field(default_factory=list)
    relation_type: str = ""
    entity_type: str = ""
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 4)
        return payload


@dataclass
class FamilySubgraphState:
    candidate_id: str
    qa_family: str
    seed_node_id: str
    selected_node_ids: list[str] = field(default_factory=list)
    selected_edge_pairs: list[list[str]] = field(default_factory=list)
    candidate_pool: list[CandidatePoolItem] = field(default_factory=list)
    frontier_node_id: str = ""
    theme_signature: str = ""
    revision_id: int = 0
    intent: str = ""
    technical_focus: str = ""
    image_grounding_summary: str = ""
    evidence_summary: str = ""
    status: str = "editing"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_pool"] = [
            item.to_dict() if hasattr(item, "to_dict") else item
            for item in self.candidate_pool
        ]
        payload["unit_count"] = len(self.selected_node_ids) + len(self.selected_edge_pairs)
        return payload


@dataclass
class FamilyJudgeFeedback:
    qa_family: str
    scorecard: JudgeScorecard
    sufficient: bool = False
    decision: str = "reject"
    rejection_reason: str = ""
    suggested_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "qa_family": self.qa_family,
            "scorecard": self.scorecard.to_dict(),
            "sufficient": self.sufficient,
            "decision": self.decision,
            "rejection_reason": self.rejection_reason,
            "suggested_action": self.suggested_action,
        }


@dataclass
class FamilySessionTrace:
    session_id: str
    qa_family: str
    seed_node_id: str
    attempted_candidates: int = 0
    accepted_candidates: int = 0
    revisions: int = 0
    status: str = "editing"
    termination_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
