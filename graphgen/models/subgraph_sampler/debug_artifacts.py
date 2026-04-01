from dataclasses import asdict, dataclass, field
from typing import Any

from .artifacts import JudgeScorecard


@dataclass
class DebugTraceStep:
    step_index: int
    phase: str
    step_type: str
    status: str
    degraded: bool = False
    summary: str = ""
    snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DebugTrace:
    sampler_version: str
    seed_node_id: str
    schema_version: str = "1.0"
    final_status: str = "running"
    termination_reason: str = ""
    steps: list[DebugTraceStep] = field(default_factory=list)

    def add_step(
        self,
        *,
        phase: str,
        step_type: str,
        status: str,
        degraded: bool = False,
        summary: str = "",
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.steps.append(
            DebugTraceStep(
                step_index=len(self.steps) + 1,
                phase=phase,
                step_type=step_type,
                status=status,
                degraded=degraded,
                summary=summary,
                snapshot=dict(snapshot or {}),
            )
        )

    def finalize(self, *, final_status: str, termination_reason: str) -> None:
        self.final_status = final_status
        self.termination_reason = termination_reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sampler_version": self.sampler_version,
            "seed_node_id": self.seed_node_id,
            "final_status": self.final_status,
            "termination_reason": self.termination_reason,
            "steps": [step.to_dict() for step in self.steps],
        }


def snapshot_neighborhood(neighborhood: dict[str, Any], *, hop: int | None = None) -> dict[str, Any]:
    payload = {
        "node_ids": list(neighborhood.get("node_ids", [])),
        "edge_count": len(neighborhood.get("edges", [])),
        "node_count": len(neighborhood.get("node_ids", [])),
    }
    if hop is not None:
        payload["hop"] = hop
    return payload


def snapshot_candidate_like(
    *,
    candidate_id: str = "",
    intent: str = "",
    technical_focus: str = "",
    approved_question_types: list[str] | None = None,
    node_ids: list[str] | None = None,
    edge_pairs: list[list[str]] | list[tuple[str, str]] | None = None,
    unit_count: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "candidate_id": candidate_id,
        "intent": intent,
        "technical_focus": technical_focus,
        "approved_question_types": list(approved_question_types or []),
        "node_ids": list(node_ids or []),
        "edge_pairs": [list(pair) for pair in (edge_pairs or [])],
    }
    payload["unit_count"] = (
        int(unit_count)
        if unit_count is not None
        else len(payload["node_ids"]) + len(payload["edge_pairs"])
    )
    if extra:
        payload.update(extra)
    return payload


def snapshot_judge(
    *,
    scorecard: JudgeScorecard,
    rejection_reason: str = "",
    sufficient: bool | None = None,
    needs_expansion: bool | None = None,
    suggested_actions: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "passes": bool(scorecard.passes),
        "judge_scores": scorecard.to_dict(),
        "rejection_reason": rejection_reason,
    }
    if sufficient is not None:
        payload["sufficient"] = bool(sufficient)
    if needs_expansion is not None:
        payload["needs_expansion"] = bool(needs_expansion)
    if suggested_actions is not None:
        payload["suggested_actions"] = list(suggested_actions)
    if extra:
        payload.update(extra)
    return payload
