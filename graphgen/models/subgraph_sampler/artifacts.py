import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from graphgen.utils import split_string_by_multi_markers


def normalize_edge_pair(src_id: str, tgt_id: str) -> tuple[str, str]:
    return tuple(sorted((str(src_id), str(tgt_id))))


def split_source_ids(value: Any) -> set[str]:
    if not value:
        return set()
    return {
        item.strip()
        for item in split_string_by_multi_markers(str(value), ["<SEP>"])
        if str(item).strip()
    }


def load_metadata(raw_metadata: Any) -> dict:
    if isinstance(raw_metadata, dict):
        return raw_metadata
    if not raw_metadata:
        return {}
    try:
        parsed = json.loads(raw_metadata)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def compact_text(value: Any, limit: int = 240) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def extract_json_payload(raw_text: str) -> dict:
    if not raw_text:
        return {}
    raw_text = raw_text.strip()
    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_text, re.DOTALL)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    brace_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group(1))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def clip_score(value: Any, *, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return float(default)


@dataclass
class JudgeScorecard:
    image_indispensability: float = 0.0
    answer_stability: float = 0.0
    evidence_closure: float = 0.0
    technical_relevance: float = 0.0
    reasoning_depth: float = 0.0
    hallucination_risk: float = 1.0
    theme_coherence: float = 0.0
    overall_score: float = 0.0
    passes: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            key: round(float(value), 4) if isinstance(value, float) else value
            for key, value in asdict(self).items()
        }


@dataclass
class SubgraphCandidate:
    candidate_id: str
    intent: str
    technical_focus: str
    node_ids: list[str]
    edge_pairs: list[list[str]]
    approved_question_types: list[str] = field(default_factory=list)
    image_grounding_summary: str = ""
    evidence_summary: str = ""
    judge_scores: JudgeScorecard = field(default_factory=JudgeScorecard)
    decision: str = "rejected"
    rejection_reason: str = ""
    degraded: bool = False

    def compact_bundle(self) -> dict[str, Any]:
        bundle = {
            "candidate_id": self.candidate_id,
            "intent": self.intent,
            "node_ids": self.node_ids,
            "edge_pairs": self.edge_pairs,
            "judge_scores": self.judge_scores.to_dict(),
            "decision": self.decision,
        }
        if self.rejection_reason:
            bundle["rejection_reason"] = self.rejection_reason
        return bundle


@dataclass
class SelectedSubgraphArtifact:
    subgraph_id: str
    technical_focus: str
    nodes: list[tuple[str, dict]]
    edges: list[tuple[str, str, dict]]
    image_grounding_summary: str
    evidence_summary: str
    judge_scores: JudgeScorecard
    approved_question_types: list[str]
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "subgraph_id": self.subgraph_id,
            "technical_focus": self.technical_focus,
            "nodes": self.nodes,
            "edges": self.edges,
            "image_grounding_summary": self.image_grounding_summary,
            "evidence_summary": self.evidence_summary,
            "judge_scores": self.judge_scores.to_dict(),
            "approved_question_types": self.approved_question_types,
            "degraded": self.degraded,
        }
