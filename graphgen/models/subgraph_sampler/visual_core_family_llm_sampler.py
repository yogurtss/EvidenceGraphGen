import copy
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from graphgen.models.subgraph_sampler.artifacts import (
    JudgeScorecard,
    clip_score,
    compact_text,
    extract_json_payload,
    load_metadata,
    split_source_ids,
    to_json_compatible,
)


FAMILY_RULES = {
    "atomic": (
        "Keep the visual core minimal. Use the image plus one direct visual fact and avoid"
        " second-layer expansion."
    ),
    "aggregated": (
        "Keep a coherent visual topic. Bootstrap from multiple relevant first-hop nodes,"
        " then allow same-direction breadth and only go deeper when it clearly improves"
        " coherence."
    ),
    "multi_hop": (
        "Bootstrap one first-hop anchor from the visual core, then extend a single"
        " same-direction reasoning chain beyond the core."
    ),
}

DEFAULT_FAMILY_QA_TARGETS = {
    "atomic": 1,
    "aggregated": 1,
    "multi_hop": 1,
}

DEFAULT_FAMILY_MAX_DEPTHS = {
    "atomic": 0,
    "aggregated": 2,
    "multi_hop": 3,
}

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
    keep_first_hop_node_ids: list[str] = field(default_factory=list)
    drop_first_hop_node_ids: list[str] = field(default_factory=list)
    preferred_entity_types: list[str] = field(default_factory=list)
    preferred_relation_types: list[str] = field(default_factory=list)
    forbidden_patterns: list[str] = field(default_factory=list)
    target_reasoning_depth: int = 1
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
    target_reasoning_depth: int = 1
    preferred_entity_types: list[str] = field(default_factory=list)
    preferred_relation_types: list[str] = field(default_factory=list)
    forbidden_patterns: list[str] = field(default_factory=list)
    visual_core_node_ids: list[str] = field(default_factory=list)
    analysis_first_hop_node_ids: list[str] = field(default_factory=list)
    dropped_analysis_first_hop_node_ids: list[str] = field(default_factory=list)
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
            "target_reasoning_depth": self.target_reasoning_depth,
            "preferred_entity_types": list(self.preferred_entity_types),
            "preferred_relation_types": list(self.preferred_relation_types),
            "forbidden_patterns": list(self.forbidden_patterns),
            "visual_core_node_ids": list(self.visual_core_node_ids),
            "analysis_first_hop_node_ids": list(self.analysis_first_hop_node_ids),
            "dropped_analysis_first_hop_node_ids": list(
                self.dropped_analysis_first_hop_node_ids
            ),
            "analysis_only_node_ids": list(self.analysis_only_node_ids),
            "selected_node_ids": list(self.selected_node_ids),
            "selected_evidence_node_ids": [
                node_id
                for node_id in self.selected_node_ids
                if node_id not in set(self.visual_core_node_ids)
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
            "virtual_edge_payload_by_pair": copy.deepcopy(
                self.virtual_edge_payload_by_pair
            ),
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
        self.virtual_edge_payload_by_pair = copy.deepcopy(
            snapshot.get("virtual_edge_payload_by_pair", {})
        )
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
    reason: str = ""
    confidence: float = 0.0
    protocol_status: str = "ok"
    protocol_error_type: str = ""
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "candidate_uid": self.candidate_uid,
            "reason": self.reason,
            "confidence": round(float(self.confidence), 4),
            "protocol_status": self.protocol_status,
            "protocol_error_type": self.protocol_error_type,
            "raw_payload": to_json_compatible(self.raw_payload),
        }


def build_bootstrap_prompt(
    *,
    qa_family: str,
    seed_payload: dict[str, Any],
    visual_core_candidates: list[dict[str, Any]],
    preview_candidates: list[dict[str, Any]],
    runtime_schema: dict[str, Any],
) -> str:
    return (
        "ROLE: VisualCoreBootstrap\n"
        f"QA family: {qa_family}\n"
        f"Family rule: {FAMILY_RULES[qa_family]}\n"
        "Use the image and the seed-local graph neighborhood to bootstrap one high-quality"
        " family-specific subgraph.\n"
        "Important: first-hop visual core candidates are analysis-only image anchors."
        " They help choose the second-hop evidence layer, but they must not become QA"
        " evidence nodes. Use keep_first_hop_node_ids/drop_first_hop_node_ids only to"
        " select analysis anchors.\n"
        "Return strict JSON with keys: intent, technical_focus, keep_first_hop_node_ids,"
        " drop_first_hop_node_ids, preferred_entity_types, preferred_relation_types,"
        " forbidden_patterns, target_reasoning_depth, image_grounding_summary,"
        " bootstrap_rationale.\n"
        f"Seed payload:\n{json.dumps(seed_payload, ensure_ascii=False)}\n"
        f"First-hop visual core candidates:\n{json.dumps(visual_core_candidates, ensure_ascii=False)}\n"
        f"Second-hop preview candidates:\n{json.dumps(preview_candidates, ensure_ascii=False)}\n"
        f"Runtime schema:\n{json.dumps(runtime_schema, ensure_ascii=False)}\n"
    )


def build_selector_prompt(
    *,
    qa_family: str,
    state_payload: dict[str, Any],
    candidate_pool_payload: list[dict[str, Any]],
) -> str:
    return (
        "ROLE: FamilyNodeSelector\n"
        f"QA family: {qa_family}\n"
        f"Family rule: {FAMILY_RULES[qa_family]}\n"
        "Choose at most one next node from the candidate pool.\n"
        "Return strict JSON with keys: decision, candidate_uid, reason, confidence.\n"
        "Allowed decisions: select_candidate, stop_selection.\n"
        f"Current state:\n{json.dumps(state_payload, ensure_ascii=False)}\n"
        f"Candidate pool:\n{json.dumps(candidate_pool_payload, ensure_ascii=False)}\n"
    )


def build_termination_prompt(
    *,
    qa_family: str,
    state_payload: dict[str, Any],
    stage: str,
    last_selected_candidate: dict[str, Any] | None,
) -> str:
    return (
        "ROLE: FamilyTerminationJudge\n"
        f"QA family: {qa_family}\n"
        f"Family rule: {FAMILY_RULES[qa_family]}\n"
        "Evaluate whether the current subgraph is sufficient, needs another step, or should"
        " rollback the last iterative step.\n"
        "Return strict JSON with keys: decision, sufficient, termination_reason, reason,"
        " suggested_action, scores.\n"
        "Allowed decisions: continue, accept, rollback_last_step, reject.\n"
        "scores must include: image_indispensability, answer_stability, evidence_closure,"
        " technical_relevance, reasoning_depth, hallucination_risk, theme_coherence,"
        " overall_score.\n"
        f"Stage: {stage}\n"
        f"Current state:\n{json.dumps(state_payload, ensure_ascii=False)}\n"
        f"Last selected candidate:\n{json.dumps(last_selected_candidate or {}, ensure_ascii=False)}\n"
    )


class VisualCoreFamilyLLMSubgraphSampler:
    FAMILY_ORDER = ("atomic", "aggregated", "multi_hop")
    VISUALIZATION_TRACE_SCHEMA_VERSION = "visual_core_family_timeline_v1"

    def __init__(
        self,
        graph,
        llm_client,
        *,
        family_qa_targets: dict[str, int] | None = None,
        family_max_depths: dict[str, int] | None = None,
        max_steps_per_family: int = 4,
        max_rollbacks_per_family: int = 1,
        judge_pass_threshold: float = 0.68,
        same_source_only: bool = True,
        bootstrap_preview_limit: int = 12,
        max_invalid_selections: int = 2,
        allow_bootstrap_fallback: bool = False,
        max_protocol_retries_per_stage: int = 1,
        max_bootstrap_errors: int = 1,
        max_selector_errors: int = 2,
        max_judge_errors: int = 1,
        min_multi_hop_outside_core_edges: int = 2,
        strict_abstain_on_empty_bootstrap: bool = True,
    ):
        self.graph = graph
        self.llm_client = llm_client
        self.family_qa_targets = self._merge_positive_int_map(
            DEFAULT_FAMILY_QA_TARGETS, family_qa_targets
        )
        self.family_max_depths = self._merge_non_negative_int_map(
            DEFAULT_FAMILY_MAX_DEPTHS, family_max_depths
        )
        self.max_steps_per_family = max(1, int(max_steps_per_family))
        self.max_rollbacks_per_family = max(0, int(max_rollbacks_per_family))
        self.judge_pass_threshold = float(judge_pass_threshold)
        self.same_source_only = bool(same_source_only)
        self.bootstrap_preview_limit = max(1, int(bootstrap_preview_limit))
        self.max_invalid_selections = max(1, int(max_invalid_selections))
        self.allow_bootstrap_fallback = bool(allow_bootstrap_fallback)
        self.max_protocol_retries_per_stage = max(0, int(max_protocol_retries_per_stage))
        self.max_bootstrap_errors = max(1, int(max_bootstrap_errors))
        self.max_selector_errors = max(1, int(max_selector_errors))
        self.max_judge_errors = max(1, int(max_judge_errors))
        self.min_multi_hop_outside_core_edges = max(
            2, int(min_multi_hop_outside_core_edges)
        )
        self.strict_abstain_on_empty_bootstrap = bool(
            strict_abstain_on_empty_bootstrap
        )

    async def sample(self, *, seed_node_id: str) -> dict[str, Any]:
        seed_node = self.graph.get_node(seed_node_id) or {}
        image_path = self._extract_image_path(seed_node)
        if not image_path:
            return {
                "seed_node_id": seed_node_id,
                "seed_image_path": "",
                "selection_mode": "single",
                "selected_subgraphs": [],
                "candidate_bundle": [],
                "abstained": True,
                "sampler_version": "family_llm_v2",
                "termination_reason": "missing_image_asset",
                "family_sessions": [],
                "family_bootstrap_trace": [],
                "family_selection_trace": [],
                "family_termination_trace": [],
                "inferred_schema": {},
                "intent_bundle": [],
                "max_vqas_per_selected_subgraph": max(self.family_qa_targets.values()),
                "visualization_trace": self._empty_visualization_trace(
                    seed_node_id=seed_node_id,
                    image_path="",
                ),
            }

        seed_scope = self._collect_seed_scope(seed_node_id) if self.same_source_only else set()
        runtime_schema = self._infer_runtime_schema(seed_node_id=seed_node_id, seed_scope=seed_scope)
        selected_subgraphs = []
        candidate_bundle = []
        family_sessions = []
        family_bootstrap_trace = []
        family_selection_trace = []
        family_termination_trace = []
        intent_bundle = []

        for qa_family in self.FAMILY_ORDER:
            session_result = await self._run_family_session(
                seed_node_id=seed_node_id,
                image_path=image_path,
                seed_scope=seed_scope,
                runtime_schema=runtime_schema,
                qa_family=qa_family,
            )
            family_sessions.append(session_result["family_session"])
            family_bootstrap_trace.extend(session_result["bootstrap_trace"])
            family_selection_trace.extend(session_result["selection_trace"])
            family_termination_trace.extend(session_result["termination_trace"])
            intent_bundle.append(session_result["bootstrap_plan"])
            candidate_bundle.append(session_result["candidate_bundle"])
            if session_result["selected_subgraph"]:
                selected_subgraphs.append(session_result["selected_subgraph"])

        visualization_trace = self._build_visualization_trace(
            seed_node_id=seed_node_id,
            image_path=image_path,
            selected_subgraphs=selected_subgraphs,
            family_bootstrap_trace=family_bootstrap_trace,
            family_selection_trace=family_selection_trace,
            family_termination_trace=family_termination_trace,
        )
        return {
            "seed_node_id": seed_node_id,
            "seed_image_path": image_path,
            "selection_mode": "multi" if len(selected_subgraphs) > 1 else "single",
            "selected_subgraphs": selected_subgraphs,
            "candidate_bundle": candidate_bundle,
            "abstained": not bool(selected_subgraphs),
            "sampler_version": "family_llm_v2",
            "termination_reason": (
                "family_sessions_completed"
                if selected_subgraphs
                else "no_family_subgraph_selected"
            ),
            "family_sessions": family_sessions,
            "family_bootstrap_trace": family_bootstrap_trace,
            "family_selection_trace": family_selection_trace,
            "family_termination_trace": family_termination_trace,
            "inferred_schema": runtime_schema,
            "intent_bundle": intent_bundle,
            "max_vqas_per_selected_subgraph": max(self.family_qa_targets.values()),
            "visualization_trace": visualization_trace,
        }

    async def _run_family_session(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        seed_scope: set[str],
        runtime_schema: dict[str, Any],
        qa_family: str,
    ) -> dict[str, Any]:
        first_hop = self._collect_visual_core_candidates(seed_node_id=seed_node_id, seed_scope=seed_scope)
        second_hop_preview = self._collect_preview_candidates(
            seed_node_id=seed_node_id,
            first_hop_candidates=first_hop,
            seed_scope=seed_scope,
        )
        seed_payload = {
            "seed_node_id": seed_node_id,
            "seed_node": to_json_compatible(self.graph.get_node(seed_node_id) or {}),
        }
        bootstrap_result = await self._bootstrap_family(
            qa_family=qa_family,
            image_path=image_path,
            seed_payload=seed_payload,
            first_hop=first_hop,
            second_hop_preview=second_hop_preview,
            runtime_schema=runtime_schema,
        )
        bootstrap_plan = bootstrap_result.plan or BootstrapPlan(qa_family=qa_family)
        family_session = {
            "session_id": f"{seed_node_id}-{qa_family}-family-llm",
            "qa_family": qa_family,
            "seed_node_id": seed_node_id,
            "target_qa_count": self.family_qa_targets[qa_family],
            "status": "abstained",
            "abstained": True,
            "termination_reason": "bootstrap_empty",
            "steps": 0,
            "rollbacks": 0,
            "bootstrap_error_count": 0,
            "selector_error_count": 0,
            "judge_error_count": 0,
            "invalid_selection_count": 0,
            "invalid_candidate_repeat_count": 0,
            "protocol_failures": [],
        }
        bootstrap_trace = [
            {
                "qa_family": qa_family,
                "seed_node_id": seed_node_id,
                "visual_core_candidates": [item.to_dict() for item in first_hop],
                "preview_candidates": [item.to_dict() for item in second_hop_preview],
                "bootstrap_plan": bootstrap_plan.to_dict(),
                "protocol_status": bootstrap_result.protocol_status,
                "protocol_error_type": bootstrap_result.protocol_error_type,
                "reason": bootstrap_result.reason,
            }
        ]
        selection_trace = []
        termination_trace = []
        protocol_failures: list[dict[str, Any]] = []
        state: FamilySessionState | None = None

        def _current_node_ids() -> list[str]:
            return list(state.selected_node_ids) if state else [seed_node_id]

        def _current_edge_pairs() -> list[list[str]]:
            return copy.deepcopy(state.selected_edge_pairs) if state else []

        def _finalize(
            *,
            termination_reason: str,
            scorecard: JudgeScorecard | None = None,
            decision: str = "rejected",
            selected_subgraph: dict[str, Any] | None = None,
            stage: str,
            decision_source: str,
            protocol_status: str = "ok",
            reason: str = "",
            protocol_error_type: str = "",
        ) -> dict[str, Any]:
            scorecard = scorecard or JudgeScorecard()
            family_session["status"] = "accepted" if selected_subgraph else "abstained"
            family_session["abstained"] = not bool(selected_subgraph)
            family_session["termination_reason"] = termination_reason
            family_session["steps"] = state.step_count if state else 0
            family_session["rollbacks"] = state.rollback_count if state else 0
            family_session["bootstrap_error_count"] = (
                state.bootstrap_error_count if state else family_session["bootstrap_error_count"]
            )
            family_session["selector_error_count"] = (
                state.selector_error_count if state else family_session["selector_error_count"]
            )
            family_session["judge_error_count"] = (
                state.judge_error_count if state else family_session["judge_error_count"]
            )
            family_session["invalid_selection_count"] = (
                state.invalid_selection_count if state else family_session["invalid_selection_count"]
            )
            family_session["invalid_candidate_repeat_count"] = (
                state.invalid_candidate_repeat_count
                if state
                else family_session["invalid_candidate_repeat_count"]
            )
            family_session["protocol_failures"] = copy.deepcopy(protocol_failures)
            self._append_terminal_trace(
                termination_trace=termination_trace,
                qa_family=qa_family,
                stage=stage,
                state=state,
                termination_reason=termination_reason,
                decision_source=decision_source,
                protocol_status=protocol_status,
                reason=reason,
                protocol_error_type=protocol_error_type,
            )
            candidate_bundle = self._build_candidate_bundle(
                qa_family=qa_family,
                bootstrap_plan=bootstrap_plan,
                node_ids=_current_node_ids(),
                edge_pairs=_current_edge_pairs(),
                decision=decision,
                rejection_reason="" if selected_subgraph else termination_reason,
                scorecard=scorecard,
                abstained=not bool(selected_subgraph),
                protocol_failures=protocol_failures,
            )
            return {
                "family_session": family_session,
                "bootstrap_trace": bootstrap_trace,
                "selection_trace": selection_trace,
                "termination_trace": termination_trace,
                "bootstrap_plan": bootstrap_plan.to_dict(),
                "candidate_bundle": candidate_bundle,
                "selected_subgraph": selected_subgraph,
            }

        if bootstrap_result.protocol_status != "ok":
            family_session["bootstrap_error_count"] = 1
            protocol_failures.append(
                self._protocol_failure_entry(
                    stage="bootstrap",
                    error_type=bootstrap_result.protocol_error_type or "parse_error",
                    reason=bootstrap_result.reason or "bootstrap_protocol_error",
                    raw_payload=bootstrap_result.raw_payload,
                )
            )
            if (
                self.allow_bootstrap_fallback
                and first_hop
                and family_session["bootstrap_error_count"] <= self.max_bootstrap_errors
            ):
                bootstrap_plan = self._build_bootstrap_fallback_plan(
                    qa_family=qa_family,
                    first_hop=first_hop,
                )
                bootstrap_trace[0]["bootstrap_plan"] = bootstrap_plan.to_dict()
                bootstrap_trace[0]["fallback_applied"] = True
            else:
                return _finalize(
                    termination_reason="bootstrap_protocol_error",
                    stage="bootstrap",
                    decision_source="system",
                    protocol_status="error",
                    reason=bootstrap_result.reason or "bootstrap_protocol_error",
                    protocol_error_type=bootstrap_result.protocol_error_type,
                )

        kept_first_hop_ids = self._resolve_first_hop_keeps(
            qa_family=qa_family,
            first_hop=first_hop,
            bootstrap_plan=bootstrap_plan,
        )
        dropped_first_hop_ids = [
            item.candidate_node_id
            for item in first_hop
            if item.candidate_node_id not in kept_first_hop_ids
        ]
        bootstrap_trace[0]["kept_first_hop_node_ids"] = list(kept_first_hop_ids)
        bootstrap_trace[0]["dropped_first_hop_node_ids"] = dropped_first_hop_ids
        bootstrap_trace[0]["analysis_first_hop_node_ids"] = list(kept_first_hop_ids)
        bootstrap_trace[0][
            "dropped_analysis_first_hop_node_ids"
        ] = dropped_first_hop_ids

        if not kept_first_hop_ids:
            return _finalize(
                termination_reason="bootstrap_empty",
                stage="bootstrap",
                decision_source="system",
                reason="No first-hop nodes survived bootstrap keep/drop.",
            )

        state = self._build_bootstrapped_state(
            qa_family=qa_family,
            seed_node_id=seed_node_id,
            image_path=image_path,
            bootstrap_plan=bootstrap_plan,
            first_hop=first_hop,
            kept_first_hop_ids=kept_first_hop_ids,
            dropped_first_hop_ids=dropped_first_hop_ids,
            seed_scope=seed_scope,
        )
        state.bootstrap_error_count = family_session["bootstrap_error_count"]
        state.candidate_pool = [
            item
            for item in state.candidate_pool
            if self._passes_session_guardrails(state, item)
        ]

        bootstrap_decision = await self._judge_state(
            qa_family=qa_family,
            state=state,
            image_path=image_path,
            stage="bootstrap",
            last_selected_candidate=None,
        )
        termination_trace.append(
            {
                "qa_family": qa_family,
                "stage": "bootstrap",
                "state": state.to_dict(),
                **bootstrap_decision.to_dict(),
            }
        )
        if bootstrap_decision.protocol_status != "ok":
            state.judge_error_count += 1
            protocol_failures.append(
                self._protocol_failure_entry(
                    stage="termination",
                    error_type=bootstrap_decision.protocol_error_type or "parse_error",
                    reason=bootstrap_decision.reason or "judge_protocol_error",
                )
            )
            return _finalize(
                termination_reason="judge_protocol_error",
                scorecard=bootstrap_decision.scorecard,
                stage="bootstrap",
                decision_source=bootstrap_decision.decision_source,
                protocol_status="error",
                reason=bootstrap_decision.reason or "judge_protocol_error",
                protocol_error_type=bootstrap_decision.protocol_error_type,
            )

        if bootstrap_decision.decision == "accept":
            if self._passes_family_postcheck(state):
                return _finalize(
                    termination_reason="accepted",
                    scorecard=bootstrap_decision.scorecard,
                    decision="accepted",
                    selected_subgraph=self._materialize_selected_subgraph(
                        state=state,
                        bootstrap_plan=bootstrap_plan,
                        scorecard=bootstrap_decision.scorecard,
                    ),
                    stage="bootstrap",
                    decision_source="judge",
                    reason=bootstrap_decision.reason,
                )

        if bootstrap_decision.decision == "reject":
            return _finalize(
                termination_reason=bootstrap_decision.termination_reason or "judge_rejected",
                scorecard=bootstrap_decision.scorecard,
                stage="bootstrap",
                decision_source="judge",
                reason=bootstrap_decision.reason,
            )

        if not state.candidate_pool:
            return _finalize(
                termination_reason="candidate_pool_exhausted",
                scorecard=bootstrap_decision.scorecard,
                stage="bootstrap",
                decision_source="system",
                reason="Bootstrap completed, but no second-hop candidates remained.",
            )

        while state.step_count < self.max_steps_per_family:
            if not state.candidate_pool:
                return _finalize(
                    termination_reason="candidate_pool_exhausted",
                    stage="selection",
                    decision_source="system",
                    reason="Candidate pool exhausted before another step.",
                )
            selector_result = await self._select_next_candidate(
                qa_family=qa_family,
                state=state,
                image_path=image_path,
            )
            if selector_result.protocol_status != "ok":
                protocol_failures.append(
                    self._protocol_failure_entry(
                        stage="selector",
                        error_type=selector_result.protocol_error_type or "parse_error",
                        reason=selector_result.reason or "selector_protocol_error",
                        raw_payload=selector_result.raw_payload,
                        candidate_uid=selector_result.candidate_uid,
                    )
                )
                if selector_result.protocol_error_type == "semantic_error":
                    state.invalid_selection_count += 1
                    if selector_result.candidate_uid and selector_result.candidate_uid == state.last_invalid_candidate_uid:
                        state.invalid_candidate_repeat_count += 1
                    else:
                        state.invalid_candidate_repeat_count = 1
                    state.last_invalid_candidate_uid = selector_result.candidate_uid
                    selection_trace.append(
                        {
                            "qa_family": qa_family,
                            "step_index": state.step_count + 1,
                            "decision": "invalid_selection",
                            "candidate_uid": selector_result.candidate_uid,
                            "reason": selector_result.reason,
                            "protocol_status": "error",
                            "protocol_error_type": selector_result.protocol_error_type,
                        }
                    )
                    if (
                        state.invalid_selection_count >= self.max_invalid_selections
                        or state.invalid_candidate_repeat_count >= self.max_invalid_selections
                    ):
                        return _finalize(
                            termination_reason="invalid_selection_repeated",
                            stage="selection",
                            decision_source="selector",
                            protocol_status="error",
                            reason=selector_result.reason,
                            protocol_error_type=selector_result.protocol_error_type,
                        )
                    continue

                state.selector_error_count += 1
                selection_trace.append(
                    {
                        "qa_family": qa_family,
                        "step_index": state.step_count + 1,
                        "decision": "selector_protocol_error",
                        "candidate_uid": selector_result.candidate_uid,
                        "reason": selector_result.reason,
                        "protocol_status": "error",
                        "protocol_error_type": selector_result.protocol_error_type,
                    }
                )
                if state.selector_error_count >= self.max_selector_errors:
                    return _finalize(
                        termination_reason="selector_protocol_error",
                        stage="selection",
                        decision_source="selector",
                        protocol_status="error",
                        reason=selector_result.reason,
                        protocol_error_type=selector_result.protocol_error_type,
                    )
                continue

            state.last_invalid_candidate_uid = ""
            state.invalid_candidate_repeat_count = 0
            if selector_result.decision == "stop_selection":
                selection_trace.append(
                    {
                        "qa_family": qa_family,
                        "step_index": state.step_count,
                        "decision": "stop_selection",
                        "candidate_uid": "",
                        "reason": selector_result.reason,
                    }
                )
                return _finalize(
                    termination_reason="selector_stop_requested",
                    stage="selection",
                    decision_source="selector",
                    reason=selector_result.reason,
                )

            previous_snapshot = state.snapshot()
            state.invalid_selection_count = 0
            state.step_count += 1
            candidate = next(
                item
                for item in state.candidate_pool
                if item.candidate_uid == selector_result.candidate_uid
            )
            depth_limit_hit = self._apply_candidate_selection(
                state=state,
                candidate=candidate,
                seed_scope=seed_scope,
                max_depth=self.family_max_depths[qa_family],
            )
            selection_trace.append(
                {
                    "qa_family": qa_family,
                    "step_index": state.step_count,
                    "decision": "select_candidate",
                    "candidate_uid": candidate.candidate_uid,
                    "candidate_node_id": candidate.candidate_node_id,
                    "depth": candidate.depth,
                    "direction_mode": state.direction_mode,
                    "candidate_pool_after_step": [item.to_dict() for item in state.candidate_pool],
                    "reason": selector_result.reason,
                }
            )

            judge_decision = await self._judge_state(
                qa_family=qa_family,
                state=state,
                image_path=image_path,
                stage="selection",
                last_selected_candidate=candidate.to_dict(),
            )
            termination_trace.append(
                {
                    "qa_family": qa_family,
                    "stage": "selection",
                    "step_index": state.step_count,
                    "state": state.to_dict(),
                    "last_selected_candidate": candidate.to_dict(),
                    **judge_decision.to_dict(),
                }
            )
            if judge_decision.protocol_status != "ok":
                state.judge_error_count += 1
                protocol_failures.append(
                    self._protocol_failure_entry(
                        stage="termination",
                        error_type=judge_decision.protocol_error_type or "parse_error",
                        reason=judge_decision.reason or "judge_protocol_error",
                        raw_payload={"decision": judge_decision.decision},
                        candidate_uid=candidate.candidate_uid,
                    )
                )
                if (
                    state.judge_error_count < self.max_judge_errors
                    and state.rollback_count < self.max_rollbacks_per_family
                ):
                    state.restore(previous_snapshot)
                    state.rollback_count += 1
                    if candidate.candidate_uid not in state.blocked_candidate_uids:
                        state.blocked_candidate_uids.append(candidate.candidate_uid)
                    state.candidate_pool = [
                        item
                        for item in state.candidate_pool
                        if item.candidate_uid != candidate.candidate_uid
                    ]
                    selection_trace.append(
                        {
                            "qa_family": qa_family,
                            "step_index": state.step_count,
                            "decision": "rollback_after_judge_protocol_error",
                            "candidate_uid": candidate.candidate_uid,
                            "rollback_count": state.rollback_count,
                            "state_after_rollback": state.to_dict(),
                        }
                    )
                    if not state.candidate_pool:
                        return _finalize(
                            termination_reason="candidate_pool_exhausted",
                            scorecard=judge_decision.scorecard,
                            stage="selection",
                            decision_source="system",
                            reason="Candidate pool exhausted after judge protocol rollback.",
                        )
                    continue
                return _finalize(
                    termination_reason="judge_protocol_error",
                    scorecard=judge_decision.scorecard,
                    stage="selection",
                    decision_source=judge_decision.decision_source,
                    protocol_status="error",
                    reason=judge_decision.reason or "judge_protocol_error",
                    protocol_error_type=judge_decision.protocol_error_type,
                )

            if judge_decision.decision == "accept":
                if self._passes_family_postcheck(state):
                    return _finalize(
                        termination_reason="accepted",
                        scorecard=judge_decision.scorecard,
                        decision="accepted",
                        selected_subgraph=self._materialize_selected_subgraph(
                            state=state,
                            bootstrap_plan=bootstrap_plan,
                            scorecard=judge_decision.scorecard,
                        ),
                        stage="selection",
                        decision_source="judge",
                        reason=judge_decision.reason,
                    )
                if not state.candidate_pool:
                    fallback_reason = (
                        "max_depth_reached" if depth_limit_hit else "candidate_pool_exhausted"
                    )
                    return _finalize(
                        termination_reason=fallback_reason,
                        scorecard=judge_decision.scorecard,
                        stage="selection",
                        decision_source="validator",
                        reason=self._family_postcheck_failure_reason(state),
                    )
                continue

            if judge_decision.decision == "rollback_last_step":
                if state.rollback_count >= self.max_rollbacks_per_family:
                    return _finalize(
                        termination_reason="max_rollbacks_reached",
                        scorecard=judge_decision.scorecard,
                        stage="selection",
                        decision_source="judge",
                        reason=judge_decision.reason,
                    )
                state.restore(previous_snapshot)
                state.rollback_count += 1
                if candidate.candidate_uid not in state.blocked_candidate_uids:
                    state.blocked_candidate_uids.append(candidate.candidate_uid)
                state.candidate_pool = [
                    item
                    for item in state.candidate_pool
                    if item.candidate_uid != candidate.candidate_uid
                ]
                selection_trace.append(
                    {
                        "qa_family": qa_family,
                        "step_index": state.step_count,
                        "decision": "rollback_last_step",
                        "candidate_uid": candidate.candidate_uid,
                        "rollback_count": state.rollback_count,
                        "state_after_rollback": state.to_dict(),
                    }
                )
                if not state.candidate_pool:
                    return _finalize(
                        termination_reason="candidate_pool_exhausted",
                        scorecard=judge_decision.scorecard,
                        stage="selection",
                        decision_source="system",
                        reason="Candidate pool exhausted after rollback.",
                    )
                continue

            if judge_decision.decision == "reject":
                return _finalize(
                    termination_reason=judge_decision.termination_reason or "judge_rejected",
                    scorecard=judge_decision.scorecard,
                    stage="selection",
                    decision_source="judge",
                    reason=judge_decision.reason,
                )

            if not state.candidate_pool:
                return _finalize(
                    termination_reason=(
                        "max_depth_reached" if depth_limit_hit else "candidate_pool_exhausted"
                    ),
                    scorecard=judge_decision.scorecard,
                    stage="selection",
                    decision_source="system",
                    reason="No valid candidates remained after applying the selected node.",
                )

        return _finalize(
            termination_reason="max_steps_reached",
            stage="selection",
            decision_source="system",
            reason="The family session exhausted its step budget.",
        )

    async def _bootstrap_family(
        self,
        *,
        qa_family: str,
        image_path: str,
        seed_payload: dict[str, Any],
        first_hop: list[FamilyCandidatePoolItem],
        second_hop_preview: list[FamilyCandidatePoolItem],
        runtime_schema: dict[str, Any],
    ) -> BootstrapStageResult:
        prompt = build_bootstrap_prompt(
            qa_family=qa_family,
            seed_payload=seed_payload,
            visual_core_candidates=[item.to_dict() for item in first_hop],
            preview_candidates=[item.to_dict() for item in second_hop_preview],
            runtime_schema=runtime_schema,
        )
        last_result = BootstrapStageResult(
            protocol_status="error",
            protocol_error_type="parse_error",
            reason="empty_bootstrap_payload",
        )
        for _ in range(self.max_protocol_retries_per_stage + 1):
            raw = await self.llm_client.generate_answer(
                prompt, image_path=image_path or None
            )
            payload = extract_json_payload(raw)
            validated = self._validate_bootstrap_payload(
                qa_family=qa_family,
                payload=payload,
                first_hop=first_hop,
            )
            if validated.plan is not None:
                return validated
            last_result = validated
        return last_result

    async def _select_next_candidate(
        self,
        *,
        qa_family: str,
        state: FamilySessionState,
        image_path: str,
    ) -> SelectorStageResult:
        prompt = build_selector_prompt(
            qa_family=qa_family,
            state_payload=state.to_dict(),
            candidate_pool_payload=[item.to_dict() for item in state.candidate_pool],
        )
        last_result = SelectorStageResult(
            protocol_status="error",
            protocol_error_type="parse_error",
            reason="empty_selector_payload",
        )
        for _ in range(self.max_protocol_retries_per_stage + 1):
            raw = await self.llm_client.generate_answer(
                prompt, image_path=image_path or None
            )
            payload = extract_json_payload(raw)
            validated = self._validate_selector_payload(
                payload=payload,
                state=state,
            )
            if validated.protocol_status == "ok":
                return validated
            last_result = validated
        return last_result

    async def _judge_state(
        self,
        *,
        qa_family: str,
        state: FamilySessionState,
        image_path: str,
        stage: str,
        last_selected_candidate: dict[str, Any] | None,
    ) -> FamilyTerminationDecision:
        prompt = build_termination_prompt(
            qa_family=qa_family,
            state_payload=state.to_dict(),
            stage=stage,
            last_selected_candidate=last_selected_candidate,
        )
        last_decision = FamilyTerminationDecision(
            decision="reject",
            sufficient=False,
            termination_reason="judge_protocol_error",
            reason="empty_termination_payload",
            scorecard=JudgeScorecard(),
            protocol_status="error",
            protocol_error_type="parse_error",
        )
        for _ in range(self.max_protocol_retries_per_stage + 1):
            raw = await self.llm_client.generate_answer(
                prompt, image_path=image_path or None
            )
            payload = extract_json_payload(raw)
            decision = self._validate_termination_payload(
                payload=payload,
                state=state,
                stage=stage,
            )
            if decision.protocol_status == "ok":
                return decision
            last_decision = decision
        return last_decision

    def _validate_bootstrap_payload(
        self,
        *,
        qa_family: str,
        payload: dict[str, Any],
        first_hop: list[FamilyCandidatePoolItem],
    ) -> BootstrapStageResult:
        if not payload:
            return BootstrapStageResult(
                protocol_status="error",
                protocol_error_type="parse_error",
                reason="empty_bootstrap_payload",
            )

        required_keys = {
            "intent",
            "technical_focus",
            "keep_first_hop_node_ids",
            "drop_first_hop_node_ids",
            "preferred_entity_types",
            "preferred_relation_types",
            "forbidden_patterns",
            "target_reasoning_depth",
            "image_grounding_summary",
            "bootstrap_rationale",
        }
        missing_keys = sorted(required_keys - set(payload.keys()))
        if missing_keys:
            return BootstrapStageResult(
                protocol_status="error",
                protocol_error_type="schema_error",
                reason=f"missing_bootstrap_keys:{','.join(missing_keys)}",
                raw_payload=payload,
            )

        list_keys = (
            "keep_first_hop_node_ids",
            "drop_first_hop_node_ids",
            "preferred_entity_types",
            "preferred_relation_types",
            "forbidden_patterns",
        )
        if any(not isinstance(payload.get(key), list) for key in list_keys):
            return BootstrapStageResult(
                protocol_status="error",
                protocol_error_type="schema_error",
                reason="bootstrap_list_fields_must_be_lists",
                raw_payload=payload,
            )

        valid_first_hop_ids = [item.candidate_node_id for item in first_hop]
        valid_first_hop_set = set(valid_first_hop_ids)
        raw_keep_ids = [str(item) for item in payload.get("keep_first_hop_node_ids", [])]
        raw_drop_ids = [str(item) for item in payload.get("drop_first_hop_node_ids", [])]
        invalid_keep_ids = [
            item for item in raw_keep_ids if item and item not in valid_first_hop_set
        ]
        invalid_drop_ids = [
            item for item in raw_drop_ids if item and item not in valid_first_hop_set
        ]
        if invalid_keep_ids or invalid_drop_ids:
            return BootstrapStageResult(
                protocol_status="error",
                protocol_error_type="semantic_error",
                reason=(
                    "bootstrap_contains_unknown_first_hop_ids:"
                    f"keep={','.join(invalid_keep_ids)};drop={','.join(invalid_drop_ids)}"
                ),
                raw_payload=payload,
            )

        keep_ids = self._stable_filter_ids(raw_keep_ids, valid_first_hop_ids)
        drop_ids = self._stable_filter_ids(raw_drop_ids, valid_first_hop_ids)
        if set(keep_ids) & set(drop_ids):
            return BootstrapStageResult(
                protocol_status="error",
                protocol_error_type="semantic_error",
                reason="bootstrap_keep_drop_overlap",
                raw_payload=payload,
            )
        if len(keep_ids) > self._max_visual_core_keeps(qa_family):
            return BootstrapStageResult(
                protocol_status="error",
                protocol_error_type="semantic_error",
                reason="bootstrap_keep_count_exceeds_family_limit",
                raw_payload=payload,
            )

        try:
            raw_target_depth = int(payload.get("target_reasoning_depth", 1))
        except (TypeError, ValueError):
            return BootstrapStageResult(
                protocol_status="error",
                protocol_error_type="schema_error",
                reason="target_reasoning_depth_must_be_int",
                raw_payload=payload,
            )

        target_depth = max(
            1,
            min(
                self.family_max_depths[qa_family] if qa_family != "atomic" else 1,
                raw_target_depth,
            ),
        )
        return BootstrapStageResult(
            plan=BootstrapPlan(
                qa_family=qa_family,
                intent=compact_text(payload.get("intent", ""), limit=160),
                technical_focus=compact_text(
                    payload.get("technical_focus", qa_family), limit=80
                ),
                keep_first_hop_node_ids=keep_ids,
                drop_first_hop_node_ids=drop_ids,
                preferred_entity_types=self._stable_string_list(
                    payload.get("preferred_entity_types", []), limit=8
                ),
                preferred_relation_types=self._stable_string_list(
                    payload.get("preferred_relation_types", []), limit=8
                ),
                forbidden_patterns=self._stable_string_list(
                    payload.get("forbidden_patterns", []), limit=8
                ),
                target_reasoning_depth=target_depth,
                image_grounding_summary=compact_text(
                    payload.get("image_grounding_summary", ""), limit=240
                ),
                bootstrap_rationale=compact_text(
                    payload.get("bootstrap_rationale", ""), limit=240
                ),
            ),
            raw_payload=payload,
        )

    def _validate_selector_payload(
        self,
        *,
        payload: dict[str, Any],
        state: FamilySessionState,
    ) -> SelectorStageResult:
        if not payload:
            return SelectorStageResult(
                protocol_status="error",
                protocol_error_type="parse_error",
                reason="empty_selector_payload",
            )

        decision = str(payload.get("decision", "")).strip().lower()
        if decision not in {"select_candidate", "stop_selection"}:
            return SelectorStageResult(
                protocol_status="error",
                protocol_error_type="schema_error",
                reason="selector_decision_invalid",
                raw_payload=payload,
            )

        if decision == "stop_selection":
            return SelectorStageResult(
                decision="stop_selection",
                reason=compact_text(payload.get("reason", ""), limit=240),
                confidence=clip_score(payload.get("confidence")),
                raw_payload=payload,
            )

        candidate_uid = str(payload.get("candidate_uid", "")).strip()
        if not candidate_uid:
            return SelectorStageResult(
                protocol_status="error",
                protocol_error_type="schema_error",
                reason="selector_candidate_uid_missing",
                raw_payload=payload,
            )

        if candidate_uid in set(state.blocked_candidate_uids):
            return SelectorStageResult(
                candidate_uid=candidate_uid,
                reason="blocked_candidate_uid",
                protocol_status="error",
                protocol_error_type="semantic_error",
                raw_payload=payload,
            )

        if all(item.candidate_uid != candidate_uid for item in state.candidate_pool):
            return SelectorStageResult(
                candidate_uid=candidate_uid,
                reason="candidate_uid_not_in_pool",
                protocol_status="error",
                protocol_error_type="semantic_error",
                raw_payload=payload,
            )

        return SelectorStageResult(
            decision="select_candidate",
            candidate_uid=candidate_uid,
            reason=compact_text(payload.get("reason", ""), limit=240),
            confidence=clip_score(payload.get("confidence")),
            raw_payload=payload,
        )

    def _validate_termination_payload(
        self,
        *,
        payload: dict[str, Any],
        state: FamilySessionState,
        stage: str,
    ) -> FamilyTerminationDecision:
        if not payload:
            return FamilyTerminationDecision(
                decision="reject",
                sufficient=False,
                termination_reason="judge_protocol_error",
                reason="empty_termination_payload",
                scorecard=JudgeScorecard(),
                protocol_status="error",
                protocol_error_type="parse_error",
            )

        required_keys = {
            "decision",
            "sufficient",
            "termination_reason",
            "reason",
            "suggested_action",
            "scores",
        }
        missing_keys = sorted(required_keys - set(payload.keys()))
        if missing_keys:
            return FamilyTerminationDecision(
                decision="reject",
                sufficient=False,
                termination_reason="judge_protocol_error",
                reason=f"missing_termination_keys:{','.join(missing_keys)}",
                scorecard=JudgeScorecard(),
                protocol_status="error",
                protocol_error_type="schema_error",
            )

        decision = str(payload.get("decision", "")).strip().lower()
        if decision not in {"continue", "accept", "rollback_last_step", "reject"}:
            return FamilyTerminationDecision(
                decision="reject",
                sufficient=False,
                termination_reason="judge_protocol_error",
                reason="termination_decision_invalid",
                scorecard=JudgeScorecard(),
                protocol_status="error",
                protocol_error_type="schema_error",
            )

        scores = payload.get("scores")
        if not isinstance(scores, dict):
            return FamilyTerminationDecision(
                decision="reject",
                sufficient=False,
                termination_reason="judge_protocol_error",
                reason="termination_scores_missing",
                scorecard=JudgeScorecard(),
                protocol_status="error",
                protocol_error_type="schema_error",
            )

        missing_score_keys = [key for key in MANDATORY_SCORE_KEYS if key not in scores]
        if missing_score_keys:
            return FamilyTerminationDecision(
                decision="reject",
                sufficient=False,
                termination_reason="judge_protocol_error",
                reason=f"missing_score_keys:{','.join(missing_score_keys)}",
                scorecard=JudgeScorecard(),
                protocol_status="error",
                protocol_error_type="schema_error",
            )

        scorecard = JudgeScorecard(
            image_indispensability=clip_score(scores.get("image_indispensability")),
            answer_stability=clip_score(scores.get("answer_stability")),
            evidence_closure=clip_score(scores.get("evidence_closure")),
            technical_relevance=clip_score(scores.get("technical_relevance")),
            reasoning_depth=clip_score(scores.get("reasoning_depth")),
            hallucination_risk=clip_score(
                scores.get("hallucination_risk"), default=1.0
            ),
            theme_coherence=clip_score(scores.get("theme_coherence")),
            overall_score=clip_score(scores.get("overall_score")),
            passes=False,
        )
        mandatory_pass = self._passes_mandatory_score_threshold(scorecard)
        sufficient = bool(payload.get("sufficient", False)) and mandatory_pass
        scorecard.passes = sufficient
        termination_reason = compact_text(
            payload.get("termination_reason", ""), limit=120
        ).lower()
        if not termination_reason:
            return FamilyTerminationDecision(
                decision="reject",
                sufficient=False,
                termination_reason="judge_protocol_error",
                reason="termination_reason_missing",
                scorecard=JudgeScorecard(),
                protocol_status="error",
                protocol_error_type="schema_error",
            )

        if decision == "accept" and not sufficient:
            return FamilyTerminationDecision(
                decision="reject",
                sufficient=False,
                termination_reason="judge_protocol_error",
                reason="accept_requires_passing_scores",
                scorecard=scorecard,
                protocol_status="error",
                protocol_error_type="semantic_error",
            )

        if decision == "rollback_last_step" and stage == "bootstrap":
            return FamilyTerminationDecision(
                decision="reject",
                sufficient=False,
                termination_reason="judge_protocol_error",
                reason="rollback_not_allowed_during_bootstrap",
                scorecard=scorecard,
                protocol_status="error",
                protocol_error_type="semantic_error",
            )

        return FamilyTerminationDecision(
            decision=decision,
            sufficient=sufficient,
            termination_reason=termination_reason,
            reason=compact_text(payload.get("reason", ""), limit=240),
            suggested_action=compact_text(
                payload.get("suggested_action", ""), limit=120
            ),
            scorecard=scorecard,
        )

    @staticmethod
    def _protocol_failure_entry(
        *,
        stage: str,
        error_type: str,
        reason: str,
        raw_payload: dict[str, Any] | None = None,
        candidate_uid: str = "",
    ) -> dict[str, Any]:
        return {
            "stage": stage,
            "error_type": error_type,
            "reason": compact_text(reason, limit=240),
            "candidate_uid": candidate_uid,
            "raw_payload": to_json_compatible(raw_payload or {}),
        }

    def _append_terminal_trace(
        self,
        *,
        termination_trace: list[dict[str, Any]],
        qa_family: str,
        stage: str,
        state: FamilySessionState | None,
        termination_reason: str,
        decision_source: str,
        protocol_status: str,
        reason: str,
        protocol_error_type: str = "",
    ) -> None:
        termination_trace.append(
            {
                "qa_family": qa_family,
                "stage": "terminal",
                "terminal_stage": stage,
                "state": state.to_dict() if state else {},
                "decision": "terminal",
                "decision_source": decision_source,
                "protocol_status": protocol_status,
                "protocol_error_type": protocol_error_type,
                "termination_reason": termination_reason,
                "reason": compact_text(reason, limit=240),
            }
        )

    def _build_bootstrap_fallback_plan(
        self,
        *,
        qa_family: str,
        first_hop: list[FamilyCandidatePoolItem],
    ) -> BootstrapPlan:
        fallback_ids = self._fallback_first_hop_keeps(
            qa_family=qa_family,
            first_hop=first_hop,
        )
        return BootstrapPlan(
            qa_family=qa_family,
            intent=f"Fallback visual-core bootstrap for {qa_family}.",
            technical_focus=qa_family,
            keep_first_hop_node_ids=fallback_ids,
            drop_first_hop_node_ids=[
                item.candidate_node_id
                for item in first_hop
                if item.candidate_node_id not in fallback_ids
            ],
            target_reasoning_depth=max(1, self.family_max_depths[qa_family]),
            image_grounding_summary="Fallback bootstrap kept the strongest visual core nodes.",
            bootstrap_rationale=(
                "The model returned no usable bootstrap JSON, so the sampler kept the "
                "strongest visual core nodes."
            ),
        )

    def _passes_mandatory_score_threshold(self, scorecard: JudgeScorecard) -> bool:
        return (
            scorecard.image_indispensability >= 0.65
            and scorecard.answer_stability >= 0.6
            and scorecard.evidence_closure >= 0.6
            and scorecard.technical_relevance >= 0.6
            and scorecard.hallucination_risk <= 0.45
            and scorecard.overall_score >= self.judge_pass_threshold
        )

    def _family_postcheck_failure_reason(self, state: FamilySessionState) -> str:
        if state.qa_family == "atomic":
            return "atomic_requires_one_virtual_image_evidence_node"
        if state.qa_family == "aggregated":
            return "aggregated_direction_or_theme_failed"
        return "multi_hop_requires_deep_chain"

    def _passes_session_guardrails(
        self,
        state: FamilySessionState,
        candidate: FamilyCandidatePoolItem,
    ) -> bool:
        haystack = " ".join(
            [
                candidate.candidate_node_id,
                candidate.entity_type,
                candidate.relation_type,
                candidate.evidence_summary,
            ]
        ).lower()
        for pattern in state.forbidden_patterns:
            token = str(pattern).strip().lower()
            if token and token in haystack:
                return False
        return True

    def _candidate_matches_session_preferences(
        self,
        state: FamilySessionState,
        candidate: FamilyCandidatePoolItem,
    ) -> bool:
        if not self._passes_session_guardrails(state, candidate):
            return False
        entity_ok = (
            not state.preferred_entity_types
            or candidate.entity_type in set(state.preferred_entity_types)
        )
        relation_ok = (
            not state.preferred_relation_types
            or candidate.relation_type in set(state.preferred_relation_types)
        )
        if state.preferred_entity_types and state.preferred_relation_types:
            return entity_ok and relation_ok
        if state.preferred_entity_types or state.preferred_relation_types:
            return entity_ok or relation_ok
        return True

    def _is_aggregated_sibling_compatible(
        self,
        state: FamilySessionState,
        candidate: FamilyCandidatePoolItem,
    ) -> bool:
        return self._is_direction_compatible(
            state, candidate
        ) and self._candidate_matches_session_preferences(state, candidate)

    def _passes_multi_hop_postcheck(
        self,
        *,
        state: FamilySessionState,
        outside_core_nodes: list[str],
    ) -> bool:
        if not self._is_directionally_consistent(state):
            return False
        if not outside_core_nodes:
            return False

        frontier_path = list(state.path_by_node_id.get(state.frontier_node_id, []))
        if len(frontier_path) != len(set(frontier_path)):
            return False

        visual_core = set(state.visual_core_node_ids)
        first_outside_index = next(
            (index for index, node_id in enumerate(frontier_path) if node_id not in visual_core),
            -1,
        )
        if first_outside_index <= 0:
            return False

        outside_edge_count = len(frontier_path) - first_outside_index
        if outside_edge_count < self.min_multi_hop_outside_core_edges:
            return False

        outside_path_nodes = [
            node_id for node_id in frontier_path[first_outside_index:] if node_id not in visual_core
        ]
        if set(outside_core_nodes) != set(outside_path_nodes):
            return False

        degree_counter = Counter()
        outside_path_set = set(outside_path_nodes)
        for src_id, tgt_id in state.selected_edge_pairs:
            if src_id in outside_path_set:
                degree_counter[src_id] += 1
            if tgt_id in outside_path_set:
                degree_counter[tgt_id] += 1

        for index, node_id in enumerate(outside_path_nodes):
            expected_degree = 1 if index == len(outside_path_nodes) - 1 else 2
            if degree_counter.get(node_id, 0) != expected_degree:
                return False
        return True

    def _build_bootstrapped_state(
        self,
        *,
        qa_family: str,
        seed_node_id: str,
        image_path: str,
        bootstrap_plan: BootstrapPlan,
        first_hop: list[FamilyCandidatePoolItem],
        kept_first_hop_ids: list[str],
        dropped_first_hop_ids: list[str],
        seed_scope: set[str],
    ) -> FamilySessionState:
        kept_first_hop = [
            item for item in first_hop if item.candidate_node_id in kept_first_hop_ids
        ]
        virtual_image_node_id = self._virtual_image_node_id(seed_node_id)
        selected_node_ids = [virtual_image_node_id]
        path_by_node_id = {virtual_image_node_id: [virtual_image_node_id]}
        selected_edge_pairs: list[list[str]] = []
        edge_direction_by_pair: dict[str, str] = {}
        visual_core_node_ids = [virtual_image_node_id]
        first_hop_ids = [item.candidate_node_id for item in first_hop]
        analysis_only_node_ids = self._stable_unique_ids(
            [seed_node_id, *first_hop_ids]
        )
        candidate_pool = []
        seen = set()
        selected_for_anchor_expansion = set(selected_node_ids) | set(analysis_only_node_ids)
        initial_max_depth = max(1, self.family_max_depths[qa_family])
        for candidate in kept_first_hop:
            anchor_path_by_node_id = {
                **path_by_node_id,
                candidate.candidate_node_id: [
                    virtual_image_node_id,
                    candidate.candidate_node_id,
                ],
            }
            next_candidates, _ = self._build_candidates_from_bind_node(
                bind_from_node_id=candidate.candidate_node_id,
                selected_node_ids=selected_for_anchor_expansion,
                path_by_node_id=anchor_path_by_node_id,
                visual_core_node_ids={virtual_image_node_id, candidate.candidate_node_id},
                seed_scope=seed_scope,
                max_depth=initial_max_depth,
                blocked_candidate_uids=[],
            )
            for item in next_candidates:
                virtualized = self._virtualize_analysis_candidate(
                    item,
                    virtual_image_node_id=virtual_image_node_id,
                )
                if (
                    virtualized.candidate_uid in seen
                    or virtualized.candidate_node_id in selected_for_anchor_expansion
                ):
                    continue
                seen.add(virtualized.candidate_uid)
                candidate_pool.append(virtualized)
        candidate_pool.sort(key=lambda item: (item.depth, -item.score, item.candidate_uid))
        return FamilySessionState(
            qa_family=qa_family,
            seed_node_id=seed_node_id,
            image_path=image_path,
            virtual_image_node_id=virtual_image_node_id,
            intent=bootstrap_plan.intent or f"{qa_family} visual-core intent",
            technical_focus=bootstrap_plan.technical_focus or qa_family,
            image_grounding_summary=bootstrap_plan.image_grounding_summary,
            bootstrap_rationale=bootstrap_plan.bootstrap_rationale,
            target_reasoning_depth=max(1, bootstrap_plan.target_reasoning_depth),
            preferred_entity_types=list(bootstrap_plan.preferred_entity_types),
            preferred_relation_types=list(bootstrap_plan.preferred_relation_types),
            forbidden_patterns=list(bootstrap_plan.forbidden_patterns),
            visual_core_node_ids=visual_core_node_ids,
            analysis_first_hop_node_ids=list(kept_first_hop_ids),
            dropped_analysis_first_hop_node_ids=list(dropped_first_hop_ids),
            analysis_only_node_ids=analysis_only_node_ids,
            selected_node_ids=selected_node_ids,
            selected_edge_pairs=selected_edge_pairs,
            candidate_pool=candidate_pool,
            frontier_node_id=virtual_image_node_id,
            path_by_node_id=path_by_node_id,
            edge_direction_by_pair=edge_direction_by_pair,
        )

    def _apply_candidate_selection(
        self,
        *,
        state: FamilySessionState,
        candidate: FamilyCandidatePoolItem,
        seed_scope: set[str],
        max_depth: int,
    ) -> bool:
        old_depth = state.current_outside_depth
        if candidate.candidate_node_id not in state.selected_node_ids:
            state.selected_node_ids.append(candidate.candidate_node_id)
        state.selected_edge_pairs.append(list(candidate.bound_edge_pair))
        state.path_by_node_id[candidate.candidate_node_id] = list(candidate.frontier_path)
        state.edge_direction_by_pair[self._pair_key(candidate.bound_edge_pair)] = candidate.edge_direction
        if candidate.virtualized_from_edge_pair:
            state.virtual_edge_payload_by_pair[
                self._pair_key(candidate.bound_edge_pair)
            ] = self._virtual_edge_payload(candidate)
        state.frontier_node_id = candidate.candidate_node_id
        state.current_outside_depth = max(state.current_outside_depth, int(candidate.depth))
        if not state.direction_mode:
            state.direction_mode = candidate.edge_direction
            state.direction_anchor_edge = list(candidate.bound_edge_pair)

        depth_limit_hit = False
        if state.qa_family == "atomic":
            state.candidate_pool = []
            return depth_limit_hit

        base_pool = [
            item
            for item in state.candidate_pool
            if item.candidate_uid != candidate.candidate_uid
            and item.candidate_node_id not in set(state.selected_node_ids)
            and item.candidate_uid not in state.blocked_candidate_uids
        ]
        base_pool = [
            item
            for item in base_pool
            if self._is_direction_compatible(state, item)
            and self._passes_session_guardrails(state, item)
        ]
        if state.qa_family == "aggregated" and candidate.depth > old_depth:
            base_pool = [
                item
                for item in base_pool
                if item.depth >= candidate.depth
                or self._is_aggregated_sibling_compatible(state, item)
            ]
        elif state.qa_family == "multi_hop":
            base_pool = []

        next_candidates, depth_limit_hit = self._build_candidates_from_bind_node(
            bind_from_node_id=candidate.candidate_node_id,
            selected_node_ids=set(state.selected_node_ids)
            | set(state.analysis_only_node_ids),
            path_by_node_id=state.path_by_node_id,
            visual_core_node_ids=set(state.visual_core_node_ids),
            seed_scope=seed_scope,
            max_depth=max_depth,
            blocked_candidate_uids=state.blocked_candidate_uids,
        )
        next_candidates = [
            item
            for item in next_candidates
            if self._is_direction_compatible(state, item)
            and self._passes_session_guardrails(state, item)
        ]
        if state.qa_family == "multi_hop":
            state.candidate_pool = next_candidates
            return depth_limit_hit

        merged = {}
        for item in base_pool + next_candidates:
            merged[item.candidate_uid] = item
        state.candidate_pool = sorted(
            merged.values(),
            key=lambda item: (item.depth, -item.score, item.candidate_uid),
        )
        return depth_limit_hit

    def _passes_family_postcheck(self, state: FamilySessionState) -> bool:
        outside_core_nodes = [
            node_id
            for node_id in state.selected_node_ids
            if node_id not in set(state.visual_core_node_ids)
        ]
        if state.qa_family == "atomic":
            return (
                len(state.visual_core_node_ids) == 1
                and len(outside_core_nodes) == 1
                and len(state.selected_node_ids) == 2
            )
        if state.qa_family == "aggregated":
            return (
                len(state.selected_node_ids) >= 3
                and self._is_directionally_consistent(state)
            )
        return self._passes_multi_hop_postcheck(
            state=state,
            outside_core_nodes=outside_core_nodes,
        )

    def _is_directionally_consistent(self, state: FamilySessionState) -> bool:
        if not state.direction_mode:
            return True
        visual_core = set(state.visual_core_node_ids)
        for edge_pair in state.selected_edge_pairs:
            pair_key = self._pair_key(edge_pair)
            edge_mode = state.edge_direction_by_pair.get(pair_key, "")
            if edge_mode == "core":
                continue
            if edge_pair[0] in visual_core and edge_pair[1] in visual_core:
                continue
            if edge_mode != state.direction_mode:
                return False
        return True

    def _materialize_selected_subgraph(
        self,
        *,
        state: FamilySessionState,
        bootstrap_plan: BootstrapPlan,
        scorecard: JudgeScorecard,
    ) -> dict[str, Any]:
        edge_pairs = [tuple(pair) for pair in state.selected_edge_pairs]
        selected_evidence_node_ids = [
            node_id
            for node_id in state.selected_node_ids
            if node_id not in set(state.visual_core_node_ids)
        ]
        return {
            "subgraph_id": f"{state.seed_node_id}-{state.qa_family}-visual-core",
            "qa_family": state.qa_family,
            "technical_focus": state.technical_focus,
            "nodes": self._node_payloads_for_state(state),
            "edges": self._edge_payloads_for_state(state, edge_pairs),
            "image_grounding_summary": compact_text(state.image_grounding_summary, limit=240),
            "evidence_summary": compact_text(bootstrap_plan.bootstrap_rationale or state.intent, limit=240),
            "judge_scores": scorecard.to_dict(),
            "approved_question_types": [state.qa_family],
            "visual_core_node_ids": list(state.visual_core_node_ids),
            "analysis_first_hop_node_ids": list(state.analysis_first_hop_node_ids),
            "dropped_analysis_first_hop_node_ids": list(
                state.dropped_analysis_first_hop_node_ids
            ),
            "analysis_only_node_ids": list(state.analysis_only_node_ids),
            "selected_evidence_node_ids": selected_evidence_node_ids,
            "original_seed_node_id": state.seed_node_id,
            "virtual_image_node_id": state.virtual_image_node_id,
            "direction_mode": state.direction_mode,
            "direction_anchor_edge": list(state.direction_anchor_edge),
            "intent_signature": compact_text(
                f"{state.qa_family}:{state.intent or state.technical_focus}",
                limit=160,
            ),
            "frontier_node_id": state.frontier_node_id,
            "candidate_pool_snapshot": [item.to_dict() for item in state.candidate_pool],
            "target_qa_count": self.family_qa_targets[state.qa_family],
            "degraded": False,
        }

    def _build_candidate_bundle(
        self,
        *,
        qa_family: str,
        bootstrap_plan: BootstrapPlan,
        node_ids: list[str],
        edge_pairs: list[list[str]],
        decision: str,
        rejection_reason: str,
        scorecard: JudgeScorecard,
        abstained: bool,
        protocol_failures: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "candidate_id": f"{qa_family}-visual-core",
            "qa_family": qa_family,
            "intent": bootstrap_plan.intent,
            "technical_focus": bootstrap_plan.technical_focus or qa_family,
            "node_ids": list(node_ids),
            "edge_pairs": to_json_compatible(edge_pairs),
            "judge_scores": scorecard.to_dict(),
            "decision": decision,
            "rejection_reason": rejection_reason,
            "abstained": abstained,
            "protocol_failures": copy.deepcopy(protocol_failures),
        }

    def _empty_visualization_trace(
        self,
        *,
        seed_node_id: str,
        image_path: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.VISUALIZATION_TRACE_SCHEMA_VERSION,
            "sampler_version": "family_llm_v2",
            "seed_node_id": seed_node_id,
            "seed_image_path": image_path,
            "graph_catalog": {"nodes": {}, "edges": {}},
            "events": [],
        }

    def _build_visualization_trace(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        selected_subgraphs: list[dict[str, Any]],
        family_bootstrap_trace: list[dict[str, Any]],
        family_selection_trace: list[dict[str, Any]],
        family_termination_trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        catalog = {"nodes": {}, "edges": {}}
        events: list[dict[str, Any]] = []

        def _add_event(
            *,
            qa_family: str,
            phase: str,
            event_type: str,
            status: str,
            state: dict[str, Any] | None = None,
            candidate_pool: list[dict[str, Any]] | None = None,
            chosen_candidate: dict[str, Any] | None = None,
            judge: dict[str, Any] | None = None,
            reason: str = "",
            termination_reason: str = "",
            extra: dict[str, Any] | None = None,
        ) -> None:
            state = state if isinstance(state, dict) else {}
            pool = candidate_pool
            if pool is None:
                pool = state.get("candidate_pool", []) if isinstance(state, dict) else []
            self._add_state_to_visualization_catalog(catalog, state)
            self._add_candidates_to_visualization_catalog(catalog, pool)
            if chosen_candidate:
                self._add_candidates_to_visualization_catalog(catalog, [chosen_candidate])

            order = len(events) + 1
            event = {
                "event_id": f"{seed_node_id}:{qa_family}:{order}:{event_type}",
                "order": order,
                "qa_family": qa_family,
                "phase": phase,
                "event_type": event_type,
                "status": status,
                "selected_node_ids": list(state.get("selected_node_ids", [])),
                "selected_edge_pairs": to_json_compatible(
                    state.get("selected_edge_pairs", [])
                ),
                "candidate_pool": to_json_compatible(pool or []),
                "chosen_candidate": to_json_compatible(chosen_candidate or {}),
                "judge": to_json_compatible(judge or {}),
                "reason": compact_text(reason, limit=240),
                "termination_reason": compact_text(termination_reason, limit=160),
            }
            if extra:
                event.update(to_json_compatible(extra))
            events.append(event)

        bootstrap_by_family = {
            item.get("qa_family"): item
            for item in family_bootstrap_trace
            if isinstance(item, dict)
        }
        selection_by_family: dict[str, list[dict[str, Any]]] = {
            family: [] for family in self.FAMILY_ORDER
        }
        termination_by_family: dict[str, list[dict[str, Any]]] = {
            family: [] for family in self.FAMILY_ORDER
        }
        for item in family_selection_trace:
            if isinstance(item, dict):
                selection_by_family.setdefault(str(item.get("qa_family", "")), []).append(
                    item
                )
        for item in family_termination_trace:
            if isinstance(item, dict):
                termination_by_family.setdefault(str(item.get("qa_family", "")), []).append(
                    item
                )
        selected_by_family = {
            item.get("qa_family"): item
            for item in selected_subgraphs
            if isinstance(item, dict)
        }

        for qa_family in self.FAMILY_ORDER:
            bootstrap = bootstrap_by_family.get(qa_family)
            if bootstrap:
                visual_core_candidates = [
                    item
                    for item in bootstrap.get("visual_core_candidates", [])
                    if isinstance(item, dict)
                ]
                preview_candidates = [
                    item
                    for item in bootstrap.get("preview_candidates", [])
                    if isinstance(item, dict)
                ]
                all_bootstrap_candidates = visual_core_candidates + preview_candidates
                self._add_candidates_to_visualization_catalog(
                    catalog, all_bootstrap_candidates
                )
                _add_event(
                    qa_family=qa_family,
                    phase="bootstrap",
                    event_type="bootstrap_candidates_collected",
                    status=str(bootstrap.get("protocol_status", "ok") or "ok"),
                    candidate_pool=all_bootstrap_candidates,
                    reason=str(bootstrap.get("reason", "")),
                    extra={
                        "visual_core_candidate_uids": [
                            item.get("candidate_uid", "")
                            for item in visual_core_candidates
                        ],
                        "preview_candidate_uids": [
                            item.get("candidate_uid", "") for item in preview_candidates
                        ],
                    },
                )
                _add_event(
                    qa_family=qa_family,
                    phase="bootstrap",
                    event_type="bootstrap_plan_created",
                    status=str(bootstrap.get("protocol_status", "ok") or "ok"),
                    candidate_pool=preview_candidates,
                    reason=str(bootstrap.get("bootstrap_plan", {}).get("bootstrap_rationale", "")),
                    extra={
                        "bootstrap_plan": bootstrap.get("bootstrap_plan", {}),
                        "kept_first_hop_node_ids": list(
                            bootstrap.get("kept_first_hop_node_ids", [])
                        ),
                        "dropped_first_hop_node_ids": list(
                            bootstrap.get("dropped_first_hop_node_ids", [])
                        ),
                    },
                )

            family_terminations = termination_by_family.get(qa_family, [])
            bootstrap_judge = next(
                (
                    item
                    for item in family_terminations
                    if item.get("stage") == "bootstrap"
                ),
                None,
            )
            if bootstrap_judge:
                state = bootstrap_judge.get("state", {})
                _add_event(
                    qa_family=qa_family,
                    phase="bootstrap",
                    event_type="bootstrap_state_created",
                    status=str(bootstrap_judge.get("protocol_status", "ok") or "ok"),
                    state=state,
                    reason=str(bootstrap_judge.get("reason", "")),
                    termination_reason=str(
                        bootstrap_judge.get("termination_reason", "")
                    ),
                )
                _add_event(
                    qa_family=qa_family,
                    phase="judge",
                    event_type="judge_decision",
                    status=str(bootstrap_judge.get("decision", "")),
                    state=state,
                    judge=self._visualization_judge_payload(bootstrap_judge),
                    reason=str(bootstrap_judge.get("reason", "")),
                    termination_reason=str(
                        bootstrap_judge.get("termination_reason", "")
                    ),
                )

            selection_judges = [
                item
                for item in family_terminations
                if item.get("stage") == "selection"
            ]
            consumed_judge_indexes: set[int] = set()
            for selection_event in selection_by_family.get(qa_family, []):
                decision = str(selection_event.get("decision", ""))
                if decision == "select_candidate":
                    step_index = selection_event.get("step_index")
                    judge_index, judge_event = self._find_visualization_selection_judge(
                        selection_judges=selection_judges,
                        consumed_indexes=consumed_judge_indexes,
                        step_index=step_index,
                        candidate_uid=str(selection_event.get("candidate_uid", "")),
                    )
                    if judge_index is not None:
                        consumed_judge_indexes.add(judge_index)
                    chosen_candidate = (
                        judge_event.get("last_selected_candidate", {})
                        if judge_event
                        else {
                            "candidate_uid": selection_event.get("candidate_uid", ""),
                            "candidate_node_id": selection_event.get(
                                "candidate_node_id", ""
                            ),
                            "depth": selection_event.get("depth", 0),
                        }
                    )
                    state = judge_event.get("state", {}) if judge_event else {}
                    candidate_pool_after_step = [
                        item
                        for item in selection_event.get(
                            "candidate_pool_after_step", []
                        )
                        if isinstance(item, dict)
                    ]
                    _add_event(
                        qa_family=qa_family,
                        phase="selection",
                        event_type="candidate_selected",
                        status="selected",
                        state=state,
                        candidate_pool=candidate_pool_after_step,
                        chosen_candidate=chosen_candidate,
                        reason=str(selection_event.get("reason", "")),
                    )
                    _add_event(
                        qa_family=qa_family,
                        phase="selection",
                        event_type="candidate_pool_updated",
                        status="updated",
                        state=state,
                        candidate_pool=candidate_pool_after_step,
                        chosen_candidate=chosen_candidate,
                        reason=str(selection_event.get("reason", "")),
                    )
                    if judge_event:
                        _add_event(
                            qa_family=qa_family,
                            phase="judge",
                            event_type="judge_decision",
                            status=str(judge_event.get("decision", "")),
                            state=state,
                            candidate_pool=state.get("candidate_pool", []),
                            chosen_candidate=chosen_candidate,
                            judge=self._visualization_judge_payload(judge_event),
                            reason=str(judge_event.get("reason", "")),
                            termination_reason=str(
                                judge_event.get("termination_reason", "")
                            ),
                        )
                    continue

                if decision in {
                    "rollback_last_step",
                    "rollback_after_judge_protocol_error",
                }:
                    state = selection_event.get("state_after_rollback", {})
                    _add_event(
                        qa_family=qa_family,
                        phase="selection",
                        event_type="rollback",
                        status="rolled_back",
                        state=state,
                        chosen_candidate={
                            "candidate_uid": selection_event.get("candidate_uid", "")
                        },
                        reason=decision,
                        extra={
                            "rollback_count": selection_event.get(
                                "rollback_count", 0
                            )
                        },
                    )
                    continue

                if decision in {
                    "invalid_selection",
                    "selector_protocol_error",
                    "stop_selection",
                }:
                    _add_event(
                        qa_family=qa_family,
                        phase="selection",
                        event_type="candidate_selected",
                        status=decision,
                        chosen_candidate={
                            "candidate_uid": selection_event.get("candidate_uid", "")
                        },
                        reason=str(selection_event.get("reason", "")),
                        extra={
                            "protocol_status": selection_event.get(
                                "protocol_status", ""
                            ),
                            "protocol_error_type": selection_event.get(
                                "protocol_error_type", ""
                            ),
                        },
                    )

            for index, judge_event in enumerate(selection_judges):
                if index in consumed_judge_indexes:
                    continue
                state = judge_event.get("state", {})
                _add_event(
                    qa_family=qa_family,
                    phase="judge",
                    event_type="judge_decision",
                    status=str(judge_event.get("decision", "")),
                    state=state,
                    chosen_candidate=judge_event.get("last_selected_candidate", {}),
                    judge=self._visualization_judge_payload(judge_event),
                    reason=str(judge_event.get("reason", "")),
                    termination_reason=str(judge_event.get("termination_reason", "")),
                )

            terminal_events = [
                item
                for item in family_terminations
                if item.get("stage") == "terminal"
            ]
            for terminal_event in terminal_events:
                state = terminal_event.get("state", {})
                _add_event(
                    qa_family=qa_family,
                    phase=str(terminal_event.get("terminal_stage", "terminal")),
                    event_type="family_terminal",
                    status=str(terminal_event.get("decision_source", "terminal")),
                    state=state,
                    reason=str(terminal_event.get("reason", "")),
                    termination_reason=str(
                        terminal_event.get("termination_reason", "")
                    ),
                    extra={
                        "protocol_status": terminal_event.get("protocol_status", ""),
                        "protocol_error_type": terminal_event.get(
                            "protocol_error_type", ""
                        ),
                    },
                )

            selected_subgraph = selected_by_family.get(qa_family)
            if selected_subgraph:
                self._add_subgraph_to_visualization_catalog(catalog, selected_subgraph)
                _add_event(
                    qa_family=qa_family,
                    phase="materialization",
                    event_type="subgraph_materialized",
                    status="materialized",
                    state={
                        "selected_node_ids": [
                            node[0]
                            for node in selected_subgraph.get("nodes", [])
                            if isinstance(node, (list, tuple)) and node
                        ],
                        "selected_edge_pairs": [
                            [edge[0], edge[1]]
                            for edge in selected_subgraph.get("edges", [])
                            if isinstance(edge, (list, tuple)) and len(edge) >= 2
                        ],
                        "candidate_pool": selected_subgraph.get(
                            "candidate_pool_snapshot", []
                        ),
                    },
                    candidate_pool=selected_subgraph.get("candidate_pool_snapshot", []),
                    reason=str(selected_subgraph.get("evidence_summary", "")),
                    extra={
                        "subgraph_id": selected_subgraph.get("subgraph_id", ""),
                        "target_qa_count": selected_subgraph.get(
                            "target_qa_count", 0
                        ),
                    },
                )

        return {
            "schema_version": self.VISUALIZATION_TRACE_SCHEMA_VERSION,
            "sampler_version": "family_llm_v2",
            "seed_node_id": seed_node_id,
            "seed_image_path": image_path,
            "graph_catalog": to_json_compatible(catalog),
            "events": to_json_compatible(events),
        }

    @staticmethod
    def _visualization_judge_payload(event: dict[str, Any]) -> dict[str, Any]:
        return {
            "decision": event.get("decision", ""),
            "sufficient": bool(event.get("sufficient", False)),
            "termination_reason": event.get("termination_reason", ""),
            "suggested_action": event.get("suggested_action", ""),
            "scorecard": event.get("scorecard", {}),
            "protocol_status": event.get("protocol_status", ""),
            "protocol_error_type": event.get("protocol_error_type", ""),
            "decision_source": event.get("decision_source", "judge"),
        }

    @staticmethod
    def _find_visualization_selection_judge(
        *,
        selection_judges: list[dict[str, Any]],
        consumed_indexes: set[int],
        step_index: Any,
        candidate_uid: str,
    ) -> tuple[int | None, dict[str, Any] | None]:
        for index, judge_event in enumerate(selection_judges):
            if index in consumed_indexes:
                continue
            if judge_event.get("step_index") != step_index:
                continue
            last_selected = judge_event.get("last_selected_candidate", {})
            if not candidate_uid or last_selected.get("candidate_uid") == candidate_uid:
                return index, judge_event
        return None, None

    def _add_state_to_visualization_catalog(
        self,
        catalog: dict[str, dict[str, Any]],
        state: dict[str, Any],
    ) -> None:
        if not isinstance(state, dict):
            return
        for node_id in state.get("selected_node_ids", []):
            self._add_node_id_to_visualization_catalog(
                catalog,
                str(node_id),
                state_payload=state,
            )
        for edge_pair in state.get("selected_edge_pairs", []):
            self._add_edge_pair_to_visualization_catalog(
                catalog,
                edge_pair,
                state_payload=state,
            )
        self._add_candidates_to_visualization_catalog(
            catalog,
            [
                item
                for item in state.get("candidate_pool", [])
                if isinstance(item, dict)
            ],
        )

    def _add_candidates_to_visualization_catalog(
        self,
        catalog: dict[str, dict[str, Any]],
        candidates: list[dict[str, Any]],
    ) -> None:
        for candidate in candidates or []:
            if not isinstance(candidate, dict):
                continue
            node_id = str(candidate.get("candidate_node_id", ""))
            if node_id:
                self._add_node_id_to_visualization_catalog(catalog, node_id)
            bound_edge_pair = candidate.get("bound_edge_pair", [])
            if isinstance(bound_edge_pair, list) and len(bound_edge_pair) >= 2:
                self._add_edge_pair_to_visualization_catalog(
                    catalog,
                    bound_edge_pair,
                    candidate_payload=candidate,
                )

    def _add_subgraph_to_visualization_catalog(
        self,
        catalog: dict[str, dict[str, Any]],
        selected_subgraph: dict[str, Any],
    ) -> None:
        for node in selected_subgraph.get("nodes", []):
            if isinstance(node, (list, tuple)) and len(node) >= 2:
                self._set_visualization_catalog_node(catalog, str(node[0]), node[1])
        for edge in selected_subgraph.get("edges", []):
            if isinstance(edge, (list, tuple)) and len(edge) >= 3:
                self._set_visualization_catalog_edge(
                    catalog,
                    str(edge[0]),
                    str(edge[1]),
                    edge[2],
                )

    def _add_node_id_to_visualization_catalog(
        self,
        catalog: dict[str, dict[str, Any]],
        node_id: str,
        *,
        state_payload: dict[str, Any] | None = None,
    ) -> None:
        if not node_id or node_id in catalog["nodes"]:
            return
        state_payload = state_payload if isinstance(state_payload, dict) else {}
        virtual_image_node_id = str(state_payload.get("virtual_image_node_id", ""))
        if node_id == virtual_image_node_id:
            self._set_visualization_catalog_node(
                catalog,
                node_id,
                self._virtual_image_node_payload_from_state_dict(state_payload),
            )
            return
        node_data = self.graph.get_node(node_id)
        if node_data:
            self._set_visualization_catalog_node(catalog, node_id, node_data)

    def _add_edge_pair_to_visualization_catalog(
        self,
        catalog: dict[str, dict[str, Any]],
        edge_pair: list[str] | tuple[str, str],
        *,
        state_payload: dict[str, Any] | None = None,
        candidate_payload: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(edge_pair, (list, tuple)) or len(edge_pair) < 2:
            return
        src_id = str(edge_pair[0])
        tgt_id = str(edge_pair[1])
        key = self._pair_key((src_id, tgt_id))
        if key in catalog["edges"]:
            return
        candidate_payload = candidate_payload if isinstance(candidate_payload, dict) else {}
        if candidate_payload.get("virtualized_from_edge_pair"):
            edge_data = self._virtual_edge_payload_from_candidate_payload(
                candidate_payload
            )
        else:
            edge_data = self.graph.get_edge(src_id, tgt_id) or self.graph.get_edge(
                tgt_id, src_id
            ) or {}
            state_payload = state_payload if isinstance(state_payload, dict) else {}
            virtual_image_node_id = str(state_payload.get("virtual_image_node_id", ""))
            if not edge_data and virtual_image_node_id in {src_id, tgt_id}:
                edge_data = {
                    "synthetic": True,
                    "description": "Synthetic edge from the virtual image root.",
                }
        self._set_visualization_catalog_edge(catalog, src_id, tgt_id, edge_data)

    @staticmethod
    def _set_visualization_catalog_node(
        catalog: dict[str, dict[str, Any]],
        node_id: str,
        payload: dict[str, Any],
    ) -> None:
        catalog["nodes"][node_id] = {
            "node_id": node_id,
            **to_json_compatible(payload or {}),
        }

    def _set_visualization_catalog_edge(
        self,
        catalog: dict[str, dict[str, Any]],
        src_id: str,
        tgt_id: str,
        payload: dict[str, Any],
    ) -> None:
        catalog["edges"][self._pair_key((src_id, tgt_id))] = {
            "source": src_id,
            "target": tgt_id,
            **to_json_compatible(payload or {}),
        }

    def _virtual_image_node_payload_from_state_dict(
        self,
        state_payload: dict[str, Any],
    ) -> dict[str, Any]:
        seed_node_id = str(state_payload.get("seed_node_id", ""))
        seed_node = copy.deepcopy(self.graph.get_node(seed_node_id) or {})
        metadata = load_metadata(seed_node.get("metadata"))
        if state_payload.get("image_path"):
            metadata.setdefault("image_path", state_payload.get("image_path"))
        metadata.update(
            {
                "synthetic": True,
                "virtualized_from_node_id": seed_node_id,
                "analysis_first_hop_node_ids": list(
                    state_payload.get("analysis_first_hop_node_ids", [])
                ),
            }
        )
        return {
            **seed_node,
            "entity_type": seed_node.get("entity_type", "IMAGE") or "IMAGE",
            "entity_name": seed_node.get("entity_name", seed_node_id),
            "description": compact_text(
                seed_node.get("description", "") or "Virtual image root.",
                limit=240,
            ),
            "metadata": metadata,
        }

    def _virtual_edge_payload_from_candidate_payload(
        self,
        candidate_payload: dict[str, Any],
    ) -> dict[str, Any]:
        source_pair = list(candidate_payload.get("virtualized_from_edge_pair", []))
        edge_data = {}
        if len(source_pair) == 2:
            edge_data = copy.deepcopy(
                self.graph.get_edge(source_pair[0], source_pair[1])
                or self.graph.get_edge(source_pair[1], source_pair[0])
                or {}
            )
        analysis_anchor_node_id = str(
            candidate_payload.get("analysis_anchor_node_id", "")
            or candidate_payload.get("bridge_first_hop_id", "")
        )
        metadata = load_metadata(edge_data.get("metadata"))
        metadata.update(
            {
                "synthetic": True,
                "analysis_anchor_node_id": analysis_anchor_node_id,
                "virtualized_from_path": list(
                    candidate_payload.get("virtualized_from_path", [])
                ),
                "virtualized_from_edge_pair": source_pair,
            }
        )
        edge_data["metadata"] = metadata
        edge_data["synthetic"] = True
        edge_data["analysis_anchor_node_id"] = analysis_anchor_node_id
        edge_data["virtualized_from_path"] = list(
            candidate_payload.get("virtualized_from_path", [])
        )
        edge_data["virtualized_from_edge_pair"] = source_pair
        if not edge_data.get("description"):
            edge_data["description"] = compact_text(
                "Virtual edge from the image to QA evidence through analysis anchor "
                f"{analysis_anchor_node_id}.",
                limit=160,
            )
        return edge_data

    def _resolve_first_hop_keeps(
        self,
        *,
        qa_family: str,
        first_hop: list[FamilyCandidatePoolItem],
        bootstrap_plan: BootstrapPlan,
    ) -> list[str]:
        keep_ids = self._stable_filter_ids(
            bootstrap_plan.keep_first_hop_node_ids,
            [item.candidate_node_id for item in first_hop],
        )
        if (
            not keep_ids
            and self.allow_bootstrap_fallback
            and not self.strict_abstain_on_empty_bootstrap
        ):
            keep_ids = self._fallback_first_hop_keeps(qa_family=qa_family, first_hop=first_hop)
        max_keeps = self._max_visual_core_keeps(qa_family)
        return keep_ids[:max_keeps]

    def _fallback_first_hop_keeps(
        self,
        *,
        qa_family: str,
        first_hop: list[FamilyCandidatePoolItem],
    ) -> list[str]:
        ranked = [item.candidate_node_id for item in first_hop]
        return ranked[: self._max_visual_core_keeps(qa_family)]

    def _max_visual_core_keeps(self, qa_family: str) -> int:
        if qa_family == "atomic":
            return 1
        if qa_family == "aggregated":
            return 3
        return 1

    def _collect_visual_core_candidates(
        self,
        *,
        seed_node_id: str,
        seed_scope: set[str],
    ) -> list[FamilyCandidatePoolItem]:
        path_by_node_id = {seed_node_id: [seed_node_id]}
        candidates, _ = self._build_candidates_from_bind_node(
            bind_from_node_id=seed_node_id,
            selected_node_ids={seed_node_id},
            path_by_node_id=path_by_node_id,
            visual_core_node_ids={seed_node_id},
            seed_scope=seed_scope,
            max_depth=1,
            blocked_candidate_uids=[],
        )
        return candidates

    def _collect_preview_candidates(
        self,
        *,
        seed_node_id: str,
        first_hop_candidates: list[FamilyCandidatePoolItem],
        seed_scope: set[str],
    ) -> list[FamilyCandidatePoolItem]:
        preview = []
        seen = set()
        selected_node_ids = {seed_node_id}
        visual_core_node_ids = {seed_node_id}
        path_by_node_id = {seed_node_id: [seed_node_id]}
        for first_hop in first_hop_candidates:
            selected_node_ids.add(first_hop.candidate_node_id)
            visual_core_node_ids.add(first_hop.candidate_node_id)
            path_by_node_id[first_hop.candidate_node_id] = [seed_node_id, first_hop.candidate_node_id]
        for first_hop in first_hop_candidates:
            candidates, _ = self._build_candidates_from_bind_node(
                bind_from_node_id=first_hop.candidate_node_id,
                selected_node_ids=selected_node_ids,
                path_by_node_id=path_by_node_id,
                visual_core_node_ids=visual_core_node_ids,
                seed_scope=seed_scope,
                max_depth=1,
                blocked_candidate_uids=[],
            )
            for item in candidates:
                if item.candidate_uid in seen:
                    continue
                seen.add(item.candidate_uid)
                preview.append(item)
        preview.sort(key=lambda item: (item.depth, -item.score, item.candidate_uid))
        return preview[: self.bootstrap_preview_limit]

    @staticmethod
    def _virtual_image_node_id(seed_node_id: str) -> str:
        return f"{seed_node_id}::virtual_image"

    @staticmethod
    def _candidate_depth_from_path(
        path: list[str],
        *,
        visual_core_node_ids: set[str],
    ) -> int:
        outside_core_count = sum(
            1 for node_id in path if node_id not in visual_core_node_ids
        )
        return max(1, outside_core_count)

    def _virtualize_analysis_candidate(
        self,
        candidate: FamilyCandidatePoolItem,
        *,
        virtual_image_node_id: str,
    ) -> FamilyCandidatePoolItem:
        analysis_anchor_node_id = (
            candidate.bridge_first_hop_id or candidate.bind_from_node_id
        )
        frontier_path = [virtual_image_node_id, candidate.candidate_node_id]
        return FamilyCandidatePoolItem(
            candidate_uid=(
                f"{virtual_image_node_id}:{candidate.candidate_node_id}:"
                f"1:{analysis_anchor_node_id}"
            ),
            candidate_node_id=candidate.candidate_node_id,
            bind_from_node_id=virtual_image_node_id,
            bound_edge_pair=[virtual_image_node_id, candidate.candidate_node_id],
            hop=1,
            depth=1,
            relation_type=candidate.relation_type,
            entity_type=candidate.entity_type,
            frontier_path=frontier_path,
            bridge_first_hop_id=analysis_anchor_node_id,
            evidence_summary=candidate.evidence_summary,
            edge_direction=candidate.edge_direction,
            score=candidate.score,
            analysis_anchor_node_id=analysis_anchor_node_id,
            virtualized_from_path=list(candidate.frontier_path),
            virtualized_from_edge_pair=list(candidate.bound_edge_pair),
        )

    def _build_candidates_from_bind_node(
        self,
        *,
        bind_from_node_id: str,
        selected_node_ids: set[str],
        path_by_node_id: dict[str, list[str]],
        visual_core_node_ids: set[str],
        seed_scope: set[str],
        max_depth: int,
        blocked_candidate_uids: list[str],
    ) -> tuple[list[FamilyCandidatePoolItem], bool]:
        bind_path = list(path_by_node_id.get(bind_from_node_id, [bind_from_node_id]))
        blocked_set = {str(item) for item in blocked_candidate_uids}
        candidates = []
        seen = set()
        depth_limit_hit = False
        for neighbor_id in self.graph.get_neighbors(bind_from_node_id):
            neighbor_id = str(neighbor_id)
            if neighbor_id in selected_node_ids:
                continue
            edge_data = self.graph.get_edge(bind_from_node_id, neighbor_id) or self.graph.get_edge(
                neighbor_id, bind_from_node_id
            ) or {}
            node_data = self.graph.get_node(neighbor_id) or {}
            if not self._passes_provenance_guardrail(node_data=node_data, edge_data=edge_data, seed_scope=seed_scope):
                continue
            new_path = bind_path + [neighbor_id]
            hop = len(new_path) - 1
            depth = self._candidate_depth_from_path(
                new_path,
                visual_core_node_ids=visual_core_node_ids,
            )
            if depth > max_depth:
                depth_limit_hit = True
                continue
            bridge_first_hop_id = (
                bind_path[1]
                if len(bind_path) >= 2
                else neighbor_id
            )
            edge_direction = self._edge_direction(bind_from_node_id, neighbor_id)
            bound_edge_pair = self._bound_edge_pair(
                bind_from_node_id=bind_from_node_id,
                candidate_node_id=neighbor_id,
                edge_direction=edge_direction,
            )
            candidate_uid = f"{bind_from_node_id}:{neighbor_id}:{hop}:{bridge_first_hop_id}"
            if candidate_uid in blocked_set or candidate_uid in seen:
                continue
            seen.add(candidate_uid)
            candidates.append(
                FamilyCandidatePoolItem(
                    candidate_uid=candidate_uid,
                    candidate_node_id=neighbor_id,
                    bind_from_node_id=bind_from_node_id,
                    bound_edge_pair=bound_edge_pair,
                    hop=hop,
                    depth=depth,
                    relation_type=str(edge_data.get("relation_type", "")),
                    entity_type=str(node_data.get("entity_type", "")),
                    frontier_path=new_path,
                    bridge_first_hop_id=bridge_first_hop_id,
                    evidence_summary=self._candidate_evidence_summary(node_data=node_data, edge_data=edge_data),
                    edge_direction=edge_direction,
                    score=self._candidate_score(
                        bind_from_node_id=bind_from_node_id,
                        candidate_node_id=neighbor_id,
                        edge_data=edge_data,
                    ),
                )
            )
        candidates.sort(key=lambda item: (item.depth, -item.score, item.candidate_uid))
        return candidates, depth_limit_hit

    def _infer_runtime_schema(
        self,
        *,
        seed_node_id: str,
        seed_scope: set[str],
    ) -> dict[str, Any]:
        first_hop = self._collect_visual_core_candidates(seed_node_id=seed_node_id, seed_scope=seed_scope)
        preview = self._collect_preview_candidates(
            seed_node_id=seed_node_id,
            first_hop_candidates=first_hop,
            seed_scope=seed_scope,
        )
        node_ids = {seed_node_id}
        relation_types = []
        node_type_counter = Counter()
        for item in first_hop + preview:
            node_ids.add(item.candidate_node_id)
            if item.entity_type:
                node_type_counter[item.entity_type] += 1
            if item.relation_type:
                relation_types.append(item.relation_type)
        seed_node = self.graph.get_node(seed_node_id) or {}
        if seed_node.get("entity_type"):
            node_type_counter[str(seed_node["entity_type"])] += 1
        modalities = sorted(
            {
                token
                for token in (
                    "image" if "IMAGE" in str(seed_node.get("entity_type", "")).upper() else "",
                    "text" if any("TEXT" in str(item.entity_type).upper() for item in preview) else "",
                )
                if token
            }
        )
        return {
            "node_types": sorted(node_type_counter.keys()),
            "node_type_counts": dict(sorted(node_type_counter.items())),
            "relation_types": sorted(set(relation_types)),
            "modalities": modalities,
            "first_hop_count": len(first_hop),
            "preview_candidate_count": len(preview),
        }

    def _collect_seed_scope(self, seed_node_id: str) -> set[str]:
        seed_scope = set()
        seed_node = self.graph.get_node(seed_node_id) or {}
        metadata = load_metadata(seed_node.get("metadata"))
        seed_scope.update(split_source_ids(seed_node.get("source_id", "")))
        seed_scope.update(split_source_ids(metadata.get("source_trace_id", "")))
        return seed_scope

    def _passes_provenance_guardrail(
        self,
        *,
        node_data: dict[str, Any],
        edge_data: dict[str, Any],
        seed_scope: set[str],
    ) -> bool:
        if not seed_scope:
            return True
        node_scope = split_source_ids(node_data.get("source_id", ""))
        node_scope.update(
            split_source_ids(load_metadata(node_data.get("metadata")).get("source_trace_id", ""))
        )
        edge_scope = split_source_ids(edge_data.get("source_id", ""))
        if node_scope & seed_scope:
            return True
        if edge_scope & seed_scope:
            return True
        return not node_scope and not edge_scope

    def _edge_direction(self, bind_from_node_id: str, candidate_node_id: str) -> str:
        if not bool(getattr(self.graph, "is_directed", lambda: False)()):
            return "outward"
        if self.graph.get_edge(bind_from_node_id, candidate_node_id):
            return "forward"
        if self.graph.get_edge(candidate_node_id, bind_from_node_id):
            return "backward"
        return "outward"

    @staticmethod
    def _bound_edge_pair(
        *,
        bind_from_node_id: str,
        candidate_node_id: str,
        edge_direction: str,
    ) -> list[str]:
        if edge_direction == "backward":
            return [candidate_node_id, bind_from_node_id]
        return [bind_from_node_id, candidate_node_id]

    def _is_direction_compatible(
        self,
        state: FamilySessionState,
        candidate: FamilyCandidatePoolItem,
    ) -> bool:
        if not state.direction_mode:
            return True
        if candidate.edge_direction == state.direction_mode:
            return True
        if state.direction_mode == "outward" and candidate.edge_direction == "outward":
            return True
        return False

    def _candidate_score(
        self,
        *,
        bind_from_node_id: str,
        candidate_node_id: str,
        edge_data: dict[str, Any],
    ) -> float:
        bind_data = self.graph.get_node(bind_from_node_id) or {}
        candidate_data = self.graph.get_node(candidate_node_id) or {}
        shared_keywords = len(
            self._keywords_from_node(bind_data) & self._keywords_from_node(candidate_data)
        )
        relation_bonus = 0.12 if edge_data.get("relation_type") else 0.0
        evidence_bonus = 0.08 if edge_data.get("evidence_span") else 0.0
        return round(shared_keywords * 0.12 + relation_bonus + evidence_bonus, 4)

    def _candidate_evidence_summary(
        self,
        *,
        node_data: dict[str, Any],
        edge_data: dict[str, Any],
    ) -> str:
        snippets = [
            compact_text(edge_data.get("description", ""), limit=120),
            compact_text(edge_data.get("evidence_span", ""), limit=120),
            compact_text(node_data.get("description", ""), limit=120),
            compact_text(node_data.get("evidence_span", ""), limit=120),
        ]
        return " ".join(part for part in snippets if part).strip()

    def _keywords_from_node(self, node_data: dict[str, Any]) -> set[str]:
        raw = " ".join(
            [
                str(node_data.get("entity_name", "")),
                str(node_data.get("description", "")),
                str(node_data.get("evidence_span", "")),
            ]
        )
        return {
            token.lower()
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_\-/]{2,}", raw)
        }

    def _node_payloads(self, node_ids: set[str]) -> list[tuple[str, dict]]:
        payloads = []
        for node_id in sorted(node_ids):
            node_data = self.graph.get_node(node_id)
            if node_data:
                payloads.append((node_id, node_data))
        return payloads

    def _node_payloads_for_state(
        self,
        state: FamilySessionState,
    ) -> list[tuple[str, dict]]:
        payloads = []
        for node_id in state.selected_node_ids:
            if node_id == state.virtual_image_node_id:
                payloads.append((node_id, self._virtual_image_node_payload(state)))
                continue
            node_data = self.graph.get_node(node_id)
            if node_data:
                payloads.append((node_id, node_data))
        return payloads

    def _virtual_image_node_payload(self, state: FamilySessionState) -> dict[str, Any]:
        seed_node = copy.deepcopy(self.graph.get_node(state.seed_node_id) or {})
        metadata = load_metadata(seed_node.get("metadata"))
        if state.image_path:
            metadata.setdefault("image_path", state.image_path)
        metadata.update(
            {
                "synthetic": True,
                "virtualized_from_node_id": state.seed_node_id,
                "analysis_first_hop_node_ids": list(
                    state.analysis_first_hop_node_ids
                ),
            }
        )
        return {
            **seed_node,
            "entity_type": seed_node.get("entity_type", "IMAGE") or "IMAGE",
            "entity_name": seed_node.get("entity_name", state.seed_node_id),
            "description": compact_text(
                seed_node.get("description", "") or "Virtual image root.",
                limit=240,
            ),
            "metadata": metadata,
        }

    def _edge_payloads(
        self,
        edge_pairs: list[tuple[str, str]],
    ) -> list[tuple[str, str, dict]]:
        payloads = []
        seen = set()
        for src_id, tgt_id in edge_pairs:
            key = (str(src_id), str(tgt_id))
            if key in seen:
                continue
            seen.add(key)
            edge_data = self.graph.get_edge(src_id, tgt_id) or self.graph.get_edge(tgt_id, src_id)
            if edge_data:
                payloads.append((src_id, tgt_id, edge_data))
        return payloads

    def _edge_payloads_for_state(
        self,
        state: FamilySessionState,
        edge_pairs: list[tuple[str, str]],
    ) -> list[tuple[str, str, dict]]:
        payloads = []
        seen = set()
        for src_id, tgt_id in edge_pairs:
            key = (str(src_id), str(tgt_id))
            if key in seen:
                continue
            seen.add(key)
            pair_key = self._pair_key(key)
            if pair_key in state.virtual_edge_payload_by_pair:
                payloads.append(
                    (
                        str(src_id),
                        str(tgt_id),
                        copy.deepcopy(state.virtual_edge_payload_by_pair[pair_key]),
                    )
                )
                continue
            edge_data = self.graph.get_edge(src_id, tgt_id) or self.graph.get_edge(
                tgt_id, src_id
            )
            if edge_data:
                payloads.append((src_id, tgt_id, edge_data))
        return payloads

    def _virtual_edge_payload(
        self,
        candidate: FamilyCandidatePoolItem,
    ) -> dict[str, Any]:
        source_pair = list(candidate.virtualized_from_edge_pair)
        edge_data = {}
        if len(source_pair) == 2:
            edge_data = copy.deepcopy(
                self.graph.get_edge(source_pair[0], source_pair[1])
                or self.graph.get_edge(source_pair[1], source_pair[0])
                or {}
            )
        metadata = load_metadata(edge_data.get("metadata"))
        metadata.update(
            {
                "synthetic": True,
                "analysis_anchor_node_id": candidate.analysis_anchor_node_id,
                "virtualized_from_path": list(candidate.virtualized_from_path),
                "virtualized_from_edge_pair": source_pair,
            }
        )
        edge_data["metadata"] = metadata
        edge_data["synthetic"] = True
        edge_data["analysis_anchor_node_id"] = candidate.analysis_anchor_node_id
        edge_data["virtualized_from_path"] = list(candidate.virtualized_from_path)
        edge_data["virtualized_from_edge_pair"] = source_pair
        if not edge_data.get("description"):
            edge_data["description"] = compact_text(
                "Virtual edge from the image to QA evidence through analysis anchor "
                f"{candidate.analysis_anchor_node_id}.",
                limit=160,
            )
        return edge_data

    def _extract_image_path(self, node_data: dict[str, Any]) -> str:
        metadata = load_metadata(node_data.get("metadata"))
        for key in ("image_path", "img_path"):
            if metadata.get(key):
                return str(metadata[key])
        return ""

    @staticmethod
    def _pair_key(edge_pair: list[str] | tuple[str, str]) -> str:
        return f"{edge_pair[0]}->{edge_pair[1]}"

    @staticmethod
    def _stable_filter_ids(values: list[Any], allowed_values: list[str]) -> list[str]:
        allowed = [str(item) for item in allowed_values]
        incoming = {str(item) for item in values if str(item) in set(allowed)}
        return [item for item in allowed if item in incoming]

    @staticmethod
    def _stable_unique_ids(values: list[Any]) -> list[str]:
        seen = set()
        result = []
        for value in values:
            item = str(value)
            if not item or item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    @staticmethod
    def _stable_string_list(values: list[Any], *, limit: int) -> list[str]:
        seen = set()
        result = []
        for item in values or []:
            text = compact_text(item, limit=80)
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _merge_positive_int_map(
        defaults: dict[str, int],
        overrides: dict[str, int] | None,
    ) -> dict[str, int]:
        merged = dict(defaults)
        for key, value in (overrides or {}).items():
            if key in merged:
                try:
                    merged[key] = max(1, int(value))
                except (TypeError, ValueError):
                    continue
        return merged

    @staticmethod
    def _merge_non_negative_int_map(
        defaults: dict[str, int],
        overrides: dict[str, int] | None,
    ) -> dict[str, int]:
        merged = dict(defaults)
        for key, value in (overrides or {}).items():
            if key in merged:
                try:
                    merged[key] = max(0, int(value))
                except (TypeError, ValueError):
                    continue
        return merged
