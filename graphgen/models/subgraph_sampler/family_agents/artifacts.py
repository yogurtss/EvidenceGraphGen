from dataclasses import dataclass
from typing import Any

from graphgen.models.subgraph_sampler.artifacts import JudgeScorecard, to_json_compatible


@dataclass
class FamilySelectedSubgraphArtifact:
    subgraph_id: str
    qa_family: str
    technical_focus: str
    nodes: list[tuple[str, dict]]
    edges: list[tuple[str, str, dict]]
    image_grounding_summary: str
    evidence_summary: str
    judge_scores: JudgeScorecard
    approved_question_types: list[str]
    candidate_pool_snapshot: list[dict[str, Any]]
    frontier_node_id: str
    theme_signature: str
    revision_id: int = 0
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "subgraph_id": self.subgraph_id,
            "qa_family": self.qa_family,
            "technical_focus": self.technical_focus,
            "nodes": to_json_compatible(self.nodes),
            "edges": to_json_compatible(self.edges),
            "image_grounding_summary": self.image_grounding_summary,
            "evidence_summary": self.evidence_summary,
            "judge_scores": self.judge_scores.to_dict(),
            "approved_question_types": list(self.approved_question_types),
            "candidate_pool_snapshot": to_json_compatible(self.candidate_pool_snapshot),
            "frontier_node_id": self.frontier_node_id,
            "theme_signature": self.theme_signature,
            "revision_id": int(self.revision_id),
            "degraded": bool(self.degraded),
        }
