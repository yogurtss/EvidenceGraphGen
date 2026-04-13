import copy
from dataclasses import asdict, dataclass, field
from typing import Any

from graphgen.models.subgraph_sampler.artifacts import JudgeScorecard, to_json_compatible

DEFAULT_FAMILY_QA_TARGETS = {"atomic": 1, "aggregated": 1, "multi_hop": 1}
DEFAULT_FAMILY_MAX_DEPTHS = {"atomic": 0, "aggregated": 2, "multi_hop": 3}
MANDATORY_SCORE_KEYS = (
    "image_indispensability",
    "answer_stability",
    "evidence_closure",
    "technical_relevance",
    "reasoning_depth",
    "hallucination_risk",
    "theme_coherence",
    "overall_score",
)


@dataclass
class FamilyCandidatePoolItem:
    candidate_uid: str
    candidate_node_id: str
    bind_from_node_id: str
    bound_edge_pair: list[str]
    hop: int
    depth: int
    relation_type: str = ""
    entity_type: str = ""
    frontier_path: list[str] = field(default_factory=list)
    bridge_first_hop_id: str = ""
    evidence_summary: str = ""
    edge_direction: str = "outward"
    score: float = 0.0
    analysis_anchor_node_id: str = ""
    virtualized_from_path: list[str] = field(default_factory=list)
    virtualized_from_edge_pair: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 4)
        return payload


@dataclass
class BootstrapPlan:
    qa_family: str
    intent: str = ""
    technical_focus: str = ""
    forbidden_patterns: list[str] = field(default_factory=list)
    image_grounding_summary: str = ""
    bootstrap_rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(asdict(self))


@dataclass
class FamilyTerminationDecision:
    decision: str = "reject"
    sufficient: bool = False
    termination_reason: str = ""
    reason: str = ""
    suggested_action: str = ""
    scorecard: JudgeScorecard = field(default_factory=JudgeScorecard)
    protocol_status: str = "ok"
    protocol_error_type: str = ""
    decision_source: str = "judge"

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "sufficient": bool(self.sufficient),
            "termination_reason": self.termination_reason,
            "reason": self.reason,
            "suggested_action": self.suggested_action,
            "scorecard": self.scorecard.to_dict(),
            "protocol_status": self.protocol_status,
            "protocol_error_type": self.protocol_error_type,
            "decision_source": self.decision_source,
        }


@dataclass
class FamilySessionState:
    qa_family: str
    seed_node_id: str
    image_path: str
    virtual_image_node_id: str = ""
    intent: str = ""
    technical_focus: str = ""
    image_grounding_summary: str = ""
    bootstrap_rationale: str = ""
    forbidden_patterns: list[str] = field(default_factory=list)
    visual_core_node_ids: list[str] = field(default_factory=list)
    analysis_first_hop_node_ids: list[str] = field(default_factory=list)
    analysis_only_node_ids: list[str] = field(default_factory=list)
    selected_node_ids: list[str] = field(default_factory=list)
    selected_edge_pairs: list[list[str]] = field(default_factory=list)
    candidate_pool: list[FamilyCandidatePoolItem] = field(default_factory=list)
    frontier_node_id: str = ""
    direction_mode: str = ""
    direction_anchor_edge: list[str] = field(default_factory=list)
    path_by_node_id: dict[str, list[str]] = field(default_factory=dict)
    edge_direction_by_pair: dict[str, str] = field(default_factory=dict)
    virtual_edge_payload_by_pair: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_outside_depth: int = 0
    step_count: int = 0
    rollback_count: int = 0
    bootstrap_error_count: int = 0
    selector_error_count: int = 0
    judge_error_count: int = 0
    invalid_selection_count: int = 0
    invalid_candidate_repeat_count: int = 0
    last_invalid_candidate_uid: str = ""
    blocked_candidate_uids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qa_family": self.qa_family,
            "seed_node_id": self.seed_node_id,
            "virtual_image_node_id": self.virtual_image_node_id,
            "intent": self.intent,
            "technical_focus": self.technical_focus,
            "image_grounding_summary": self.image_grounding_summary,
            "bootstrap_rationale": self.bootstrap_rationale,
            "forbidden_patterns": list(self.forbidden_patterns),
            "visual_core_node_ids": list(self.visual_core_node_ids),
            "analysis_first_hop_node_ids": list(self.analysis_first_hop_node_ids),
            "analysis_only_node_ids": list(self.analysis_only_node_ids),
            "selected_node_ids": list(self.selected_node_ids),
            "selected_evidence_node_ids": [
                node_id for node_id in self.selected_node_ids if node_id not in set(self.visual_core_node_ids)
            ],
            "selected_edge_pairs": to_json_compatible(self.selected_edge_pairs),
            "candidate_pool": [item.to_dict() for item in self.candidate_pool],
            "frontier_node_id": self.frontier_node_id,
            "direction_mode": self.direction_mode,
            "direction_anchor_edge": list(self.direction_anchor_edge),
            "current_outside_depth": self.current_outside_depth,
            "step_count": self.step_count,
            "rollback_count": self.rollback_count,
            "bootstrap_error_count": self.bootstrap_error_count,
            "selector_error_count": self.selector_error_count,
            "judge_error_count": self.judge_error_count,
            "invalid_selection_count": self.invalid_selection_count,
            "invalid_candidate_repeat_count": self.invalid_candidate_repeat_count,
            "blocked_candidate_uids": list(self.blocked_candidate_uids),
            "unit_count": len(self.selected_node_ids) + len(self.selected_edge_pairs),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "selected_node_ids": list(self.selected_node_ids),
            "selected_edge_pairs": copy.deepcopy(self.selected_edge_pairs),
            "candidate_pool": [item.to_dict() for item in self.candidate_pool],
            "frontier_node_id": self.frontier_node_id,
            "direction_mode": self.direction_mode,
            "direction_anchor_edge": list(self.direction_anchor_edge),
            "path_by_node_id": copy.deepcopy(self.path_by_node_id),
            "edge_direction_by_pair": dict(self.edge_direction_by_pair),
            "virtual_edge_payload_by_pair": copy.deepcopy(self.virtual_edge_payload_by_pair),
            "current_outside_depth": self.current_outside_depth,
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.selected_node_ids = list(snapshot.get("selected_node_ids", []))
        self.selected_edge_pairs = copy.deepcopy(snapshot.get("selected_edge_pairs", []))
        self.candidate_pool = [
            FamilyCandidatePoolItem(**item)
            for item in snapshot.get("candidate_pool", [])
            if isinstance(item, dict)
        ]
        self.frontier_node_id = str(snapshot.get("frontier_node_id", ""))
        self.direction_mode = str(snapshot.get("direction_mode", ""))
        self.direction_anchor_edge = list(snapshot.get("direction_anchor_edge", []))
        self.path_by_node_id = copy.deepcopy(snapshot.get("path_by_node_id", {}))
        self.edge_direction_by_pair = dict(snapshot.get("edge_direction_by_pair", {}))
        self.virtual_edge_payload_by_pair = copy.deepcopy(snapshot.get("virtual_edge_payload_by_pair", {}))
        self.current_outside_depth = int(snapshot.get("current_outside_depth", 0))


@dataclass
class BootstrapStageResult:
    plan: BootstrapPlan | None = None
    protocol_status: str = "ok"
    protocol_error_type: str = ""
    reason: str = ""
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict() if self.plan else None,
            "protocol_status": self.protocol_status,
            "protocol_error_type": self.protocol_error_type,
            "reason": self.reason,
            "raw_payload": to_json_compatible(self.raw_payload),
        }


@dataclass
class SelectorStageResult:
    decision: str = "stop_selection"
    candidate_uid: str = ""
    candidate_node_id: str = ""
    reason: str = ""
    confidence: float = 0.0
    protocol_status: str = "ok"
    protocol_error_type: str = ""
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "candidate_uid": self.candidate_uid,
            "candidate_node_id": self.candidate_node_id,
            "reason": self.reason,
            "confidence": round(float(self.confidence), 4),
            "protocol_status": self.protocol_status,
            "protocol_error_type": self.protocol_error_type,
            "raw_payload": to_json_compatible(self.raw_payload),
        }
