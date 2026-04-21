from typing import Any

from graphgen.models.subgraph_sampler.artifacts import (
    JudgeScorecard,
    clip_score,
    compact_text,
    to_json_compatible,
)

from .models import (
    BootstrapPlan,
    BootstrapStageResult,
    FamilySessionState,
    FamilyTerminationDecision,
    IntentPlannerStageResult,
    MANDATORY_SCORE_KEYS,
    SelectorStageResult,
)


BOOTSTRAP_REQUIRED_KEYS = {
    "intent",
    "technical_focus",
    "forbidden_patterns",
    "image_grounding_summary",
    "bootstrap_rationale",
}


def protocol_failure_entry(
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


def validate_bootstrap_payload(
    *,
    qa_family: str,
    payload: dict[str, Any],
) -> BootstrapStageResult:
    if not payload:
        return BootstrapStageResult(
            protocol_status="error",
            protocol_error_type="parse_error",
            reason="empty_bootstrap_payload",
        )

    missing_keys = sorted(BOOTSTRAP_REQUIRED_KEYS - set(payload.keys()))
    if missing_keys:
        return BootstrapStageResult(
            protocol_status="error",
            protocol_error_type="schema_error",
            reason=f"missing_bootstrap_keys:{','.join(missing_keys)}",
            raw_payload=payload,
        )

    if not isinstance(payload.get("forbidden_patterns"), list):
        return BootstrapStageResult(
            protocol_status="error",
            protocol_error_type="schema_error",
            reason="bootstrap_forbidden_patterns_must_be_list",
            raw_payload=payload,
        )

    return BootstrapStageResult(
        plan=BootstrapPlan(
            qa_family=qa_family,
            intent=compact_text(payload.get("intent", ""), limit=160),
            technical_focus=compact_text(
                payload.get("technical_focus", qa_family), limit=80
            ),
            forbidden_patterns=_stable_string_list(
                payload.get("forbidden_patterns", []), limit=8
            ),
            image_grounding_summary=compact_text(
                payload.get("image_grounding_summary", ""), limit=240
            ),
            bootstrap_rationale=compact_text(
                payload.get("bootstrap_rationale", ""), limit=240
            ),
        ),
        raw_payload=payload,
    )


def validate_intent_planner_payload(
    *,
    qa_family: str,
    payload: dict[str, Any],
    target_count: int,
) -> IntentPlannerStageResult:
    if not payload:
        return IntentPlannerStageResult(
            protocol_status="error",
            protocol_error_type="parse_error",
            reason="empty_intent_planner_payload",
        )

    intents = payload.get("intents")
    if not isinstance(intents, list):
        return IntentPlannerStageResult(
            protocol_status="error",
            protocol_error_type="schema_error",
            reason="intent_planner_intents_must_be_list",
            raw_payload=payload,
        )
    if len(intents) != int(target_count):
        return IntentPlannerStageResult(
            protocol_status="error",
            protocol_error_type="schema_error",
            reason=f"intent_count_mismatch:expected_{int(target_count)}_got_{len(intents)}",
            raw_payload=payload,
        )

    plans = []
    for index, item in enumerate(intents):
        if not isinstance(item, dict):
            return IntentPlannerStageResult(
                protocol_status="error",
                protocol_error_type="schema_error",
                reason=f"intent_item_{index}_must_be_object",
                raw_payload=payload,
            )
        missing_keys = sorted(BOOTSTRAP_REQUIRED_KEYS - set(item.keys()))
        if missing_keys:
            return IntentPlannerStageResult(
                protocol_status="error",
                protocol_error_type="schema_error",
                reason=f"missing_intent_keys:{','.join(missing_keys)}",
                raw_payload=payload,
            )
        if not isinstance(item.get("forbidden_patterns"), list):
            return IntentPlannerStageResult(
                protocol_status="error",
                protocol_error_type="schema_error",
                reason="intent_forbidden_patterns_must_be_list",
                raw_payload=payload,
            )
        plans.append(
            BootstrapPlan(
                qa_family=qa_family,
                intent=compact_text(item.get("intent", ""), limit=160),
                technical_focus=compact_text(
                    item.get("technical_focus", qa_family), limit=80
                ),
                forbidden_patterns=_stable_string_list(
                    item.get("forbidden_patterns", []), limit=8
                ),
                image_grounding_summary=compact_text(
                    item.get("image_grounding_summary", ""), limit=240
                ),
                bootstrap_rationale=compact_text(
                    item.get("bootstrap_rationale", ""), limit=240
                ),
            )
        )

    return IntentPlannerStageResult(plans=plans, raw_payload=payload)


def validate_selector_payload(
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

    candidate_node_id = str(payload.get("candidate_node_id", "")).strip()
    if not candidate_node_id:
        return SelectorStageResult(
            protocol_status="error",
            protocol_error_type="schema_error",
            reason="selector_candidate_node_id_missing",
            raw_payload=payload,
        )

    matches = [
        item for item in state.candidate_pool
        if item.candidate_node_id == candidate_node_id
    ]
    if not matches:
        return SelectorStageResult(
            candidate_node_id=candidate_node_id,
            reason="candidate_node_id_not_in_pool",
            protocol_status="error",
            protocol_error_type="semantic_error",
            raw_payload=payload,
        )

    unblocked_matches = [
        item for item in matches
        if item.candidate_uid not in set(state.blocked_candidate_uids)
    ]
    if not unblocked_matches:
        return SelectorStageResult(
            candidate_node_id=candidate_node_id,
            reason="blocked_candidate_node_id",
            protocol_status="error",
            protocol_error_type="semantic_error",
            raw_payload=payload,
        )
    if len(unblocked_matches) > 1:
        return SelectorStageResult(
            candidate_node_id=candidate_node_id,
            reason="ambiguous_candidate_node_id",
            protocol_status="error",
            protocol_error_type="semantic_error",
            raw_payload=payload,
        )

    candidate = unblocked_matches[0]

    return SelectorStageResult(
        decision="select_candidate",
        candidate_uid=candidate.candidate_uid,
        candidate_node_id=candidate.candidate_node_id,
        reason=compact_text(payload.get("reason", ""), limit=240),
        confidence=clip_score(payload.get("confidence")),
        raw_payload=payload,
    )


def validate_termination_payload(
    *,
    payload: dict[str, Any],
    stage: str,
    judge_pass_threshold: float,
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
        hallucination_risk=clip_score(scores.get("hallucination_risk"), default=1.0),
        theme_coherence=clip_score(scores.get("theme_coherence")),
        overall_score=clip_score(scores.get("overall_score")),
        passes=False,
    )
    sufficient = bool(payload.get("sufficient", False)) and _passes_score_threshold(
        scorecard=scorecard,
        judge_pass_threshold=judge_pass_threshold,
    )
    scorecard.passes = sufficient
    termination_reason = compact_text(payload.get("termination_reason", ""), limit=120).lower()
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
        suggested_action=compact_text(payload.get("suggested_action", ""), limit=120),
        scorecard=scorecard,
    )


def _passes_score_threshold(
    *,
    scorecard: JudgeScorecard,
    judge_pass_threshold: float,
) -> bool:
    return (
        scorecard.image_indispensability >= 0.65
        and scorecard.answer_stability >= 0.6
        and scorecard.evidence_closure >= 0.6
        and scorecard.technical_relevance >= 0.6
        and scorecard.hallucination_risk <= 0.45
        and scorecard.overall_score >= judge_pass_threshold
    )


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
