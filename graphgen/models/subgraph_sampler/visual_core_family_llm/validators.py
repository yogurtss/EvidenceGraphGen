from typing import Any, Callable

from graphgen.models.subgraph_sampler.artifacts import JudgeScorecard, clip_score, compact_text, to_json_compatible
from .models import (
    BootstrapPlan,
    BootstrapStageResult,
    FamilyCandidatePoolItem,
    FamilySessionState,
    FamilyTerminationDecision,
    MANDATORY_SCORE_KEYS,
    SelectorStageResult,
)


def protocol_failure_entry(*, stage: str, error_type: str, reason: str, raw_payload: dict[str, Any] | None = None, candidate_uid: str = "") -> dict[str, Any]:
    return {
        "stage": stage,
        "error_type": error_type,
        "reason": compact_text(reason, limit=240),
        "candidate_uid": candidate_uid,
        "raw_payload": to_json_compatible(raw_payload or {}),
    }


def _validate_bootstrap_payload(*, qa_family: str, payload: dict[str, Any], first_hop: list[FamilyCandidatePoolItem], family_max_depths: dict[str, int], stable_filter_ids: Callable[[list[Any], list[str]], list[str]], max_visual_core_keeps: Callable[[str], int], stable_string_list: Callable[[list[Any], int], list[str]]) -> BootstrapStageResult:
    if not payload:
        return BootstrapStageResult(protocol_status="error", protocol_error_type="parse_error", reason="empty_bootstrap_payload")
    required_keys = {"intent", "technical_focus", "keep_first_hop_node_ids", "drop_first_hop_node_ids", "preferred_entity_types", "preferred_relation_types", "forbidden_patterns", "target_reasoning_depth", "image_grounding_summary", "bootstrap_rationale"}
    missing_keys = sorted(required_keys - set(payload.keys()))
    if missing_keys:
        return BootstrapStageResult(protocol_status="error", protocol_error_type="schema_error", reason=f"missing_bootstrap_keys:{','.join(missing_keys)}", raw_payload=payload)
    list_keys = ("keep_first_hop_node_ids", "drop_first_hop_node_ids", "preferred_entity_types", "preferred_relation_types", "forbidden_patterns")
    if any(not isinstance(payload.get(key), list) for key in list_keys):
        return BootstrapStageResult(protocol_status="error", protocol_error_type="schema_error", reason="bootstrap_list_fields_must_be_lists", raw_payload=payload)
    valid_first_hop_ids = [item.candidate_node_id for item in first_hop]
    valid_first_hop_set = set(valid_first_hop_ids)
    raw_keep_ids = [str(item) for item in payload.get("keep_first_hop_node_ids", [])]
    raw_drop_ids = [str(item) for item in payload.get("drop_first_hop_node_ids", [])]
    invalid_keep_ids = [item for item in raw_keep_ids if item and item not in valid_first_hop_set]
    invalid_drop_ids = [item for item in raw_drop_ids if item and item not in valid_first_hop_set]
    if invalid_keep_ids or invalid_drop_ids:
        return BootstrapStageResult(protocol_status="error", protocol_error_type="semantic_error", reason=("bootstrap_contains_unknown_first_hop_ids:" f"keep={','.join(invalid_keep_ids)};drop={','.join(invalid_drop_ids)}"), raw_payload=payload)
    keep_ids = stable_filter_ids(raw_keep_ids, valid_first_hop_ids)
    drop_ids = stable_filter_ids(raw_drop_ids, valid_first_hop_ids)
    if set(keep_ids) & set(drop_ids):
        return BootstrapStageResult(protocol_status="error", protocol_error_type="semantic_error", reason="bootstrap_keep_drop_overlap", raw_payload=payload)
    if len(keep_ids) > max_visual_core_keeps(qa_family):
        return BootstrapStageResult(protocol_status="error", protocol_error_type="semantic_error", reason="bootstrap_keep_count_exceeds_family_limit", raw_payload=payload)
    try:
        raw_target_depth = int(payload.get("target_reasoning_depth", 1))
    except (TypeError, ValueError):
        return BootstrapStageResult(protocol_status="error", protocol_error_type="schema_error", reason="target_reasoning_depth_must_be_int", raw_payload=payload)
    target_depth = max(1, min(family_max_depths[qa_family] if qa_family != "atomic" else 1, raw_target_depth))
    return BootstrapStageResult(plan=BootstrapPlan(qa_family=qa_family, intent=compact_text(payload.get("intent", ""), limit=160), technical_focus=compact_text(payload.get("technical_focus", qa_family), limit=80), keep_first_hop_node_ids=keep_ids, drop_first_hop_node_ids=drop_ids, preferred_entity_types=stable_string_list(payload.get("preferred_entity_types", []), 8), preferred_relation_types=stable_string_list(payload.get("preferred_relation_types", []), 8), forbidden_patterns=stable_string_list(payload.get("forbidden_patterns", []), 8), target_reasoning_depth=target_depth, image_grounding_summary=compact_text(payload.get("image_grounding_summary", ""), limit=240), bootstrap_rationale=compact_text(payload.get("bootstrap_rationale", ""), limit=240)), raw_payload=payload)


def _validate_selector_payload(*, payload: dict[str, Any], state: FamilySessionState) -> SelectorStageResult:
    if not payload:
        return SelectorStageResult(protocol_status="error", protocol_error_type="parse_error", reason="empty_selector_payload")
    decision = str(payload.get("decision", "")).strip().lower()
    if decision not in {"select_candidate", "stop_selection"}:
        return SelectorStageResult(protocol_status="error", protocol_error_type="schema_error", reason="selector_decision_invalid", raw_payload=payload)
    if decision == "stop_selection":
        return SelectorStageResult(decision="stop_selection", reason=compact_text(payload.get("reason", ""), limit=240), confidence=clip_score(payload.get("confidence")), raw_payload=payload)
    candidate_uid = str(payload.get("candidate_uid", "")).strip()
    if not candidate_uid:
        return SelectorStageResult(protocol_status="error", protocol_error_type="schema_error", reason="selector_candidate_uid_missing", raw_payload=payload)
    if candidate_uid in set(state.blocked_candidate_uids):
        return SelectorStageResult(candidate_uid=candidate_uid, reason="blocked_candidate_uid", protocol_status="error", protocol_error_type="semantic_error", raw_payload=payload)
    if all(item.candidate_uid != candidate_uid for item in state.candidate_pool):
        return SelectorStageResult(candidate_uid=candidate_uid, reason="candidate_uid_not_in_pool", protocol_status="error", protocol_error_type="semantic_error", raw_payload=payload)
    return SelectorStageResult(decision="select_candidate", candidate_uid=candidate_uid, reason=compact_text(payload.get("reason", ""), limit=240), confidence=clip_score(payload.get("confidence")), raw_payload=payload)


def _validate_termination_payload(*, payload: dict[str, Any], stage: str, passes_mandatory_score_threshold: Callable[[JudgeScorecard], bool]) -> FamilyTerminationDecision:
    if not payload:
        return FamilyTerminationDecision(decision="reject", sufficient=False, termination_reason="judge_protocol_error", reason="empty_termination_payload", scorecard=JudgeScorecard(), protocol_status="error", protocol_error_type="parse_error")
    required_keys = {"decision", "sufficient", "termination_reason", "reason", "suggested_action", "scores"}
    missing_keys = sorted(required_keys - set(payload.keys()))
    if missing_keys:
        return FamilyTerminationDecision(decision="reject", sufficient=False, termination_reason="judge_protocol_error", reason=f"missing_termination_keys:{','.join(missing_keys)}", scorecard=JudgeScorecard(), protocol_status="error", protocol_error_type="schema_error")
    decision = str(payload.get("decision", "")).strip().lower()
    if decision not in {"continue", "accept", "rollback_last_step", "reject"}:
        return FamilyTerminationDecision(decision="reject", sufficient=False, termination_reason="judge_protocol_error", reason="termination_decision_invalid", scorecard=JudgeScorecard(), protocol_status="error", protocol_error_type="schema_error")
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        return FamilyTerminationDecision(decision="reject", sufficient=False, termination_reason="judge_protocol_error", reason="termination_scores_missing", scorecard=JudgeScorecard(), protocol_status="error", protocol_error_type="schema_error")
    missing_score_keys = [key for key in MANDATORY_SCORE_KEYS if key not in scores]
    if missing_score_keys:
        return FamilyTerminationDecision(decision="reject", sufficient=False, termination_reason="judge_protocol_error", reason=f"missing_score_keys:{','.join(missing_score_keys)}", scorecard=JudgeScorecard(), protocol_status="error", protocol_error_type="schema_error")
    scorecard = JudgeScorecard(image_indispensability=clip_score(scores.get("image_indispensability")), answer_stability=clip_score(scores.get("answer_stability")), evidence_closure=clip_score(scores.get("evidence_closure")), technical_relevance=clip_score(scores.get("technical_relevance")), reasoning_depth=clip_score(scores.get("reasoning_depth")), hallucination_risk=clip_score(scores.get("hallucination_risk"), default=1.0), theme_coherence=clip_score(scores.get("theme_coherence")), overall_score=clip_score(scores.get("overall_score")), passes=False)
    mandatory_pass = passes_mandatory_score_threshold(scorecard)
    sufficient = bool(payload.get("sufficient", False)) and mandatory_pass
    scorecard.passes = sufficient
    termination_reason = compact_text(payload.get("termination_reason", ""), limit=120).lower()
    if not termination_reason:
        return FamilyTerminationDecision(decision="reject", sufficient=False, termination_reason="judge_protocol_error", reason="termination_reason_missing", scorecard=JudgeScorecard(), protocol_status="error", protocol_error_type="schema_error")
    if decision == "accept" and not sufficient:
        return FamilyTerminationDecision(decision="reject", sufficient=False, termination_reason="judge_protocol_error", reason="accept_requires_passing_scores", scorecard=scorecard, protocol_status="error", protocol_error_type="semantic_error")
    if decision == "rollback_last_step" and stage == "bootstrap":
        return FamilyTerminationDecision(decision="reject", sufficient=False, termination_reason="judge_protocol_error", reason="rollback_not_allowed_during_bootstrap", scorecard=scorecard, protocol_status="error", protocol_error_type="semantic_error")
    return FamilyTerminationDecision(decision=decision, sufficient=sufficient, termination_reason=termination_reason, reason=compact_text(payload.get("reason", ""), limit=240), suggested_action=compact_text(payload.get("suggested_action", ""), limit=120), scorecard=scorecard)
