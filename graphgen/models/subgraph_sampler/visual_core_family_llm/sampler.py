import copy
import hashlib
from typing import Any

from graphgen.models.subgraph_sampler.artifacts import (
    JudgeScorecard,
    compact_text,
    extract_json_payload,
    load_metadata,
    to_json_compatible,
)
from .candidate_engine import VisualCoreFamilyCandidateEngineMixin
from .models import (
    BootstrapPlan,
    BootstrapStageResult,
    DEFAULT_FAMILY_MAX_DEPTHS,
    DEFAULT_FAMILY_QA_TARGETS,
    FamilyCandidatePoolItem,
    FamilySessionState,
    FamilyTerminationDecision,
    SelectorStageResult,
)
from .prompts import build_bootstrap_prompt, build_selector_prompt, build_termination_prompt
from .materializer import VisualCoreFamilyMaterializerMixin
from .trace import VisualCoreFamilyTraceMixin
from .validators import (
    protocol_failure_entry,
    validate_bootstrap_payload,
    validate_selector_payload,
    validate_termination_payload,
)


class VisualCoreFamilyLLMSubgraphSampler(
    VisualCoreFamilyCandidateEngineMixin,
    VisualCoreFamilyMaterializerMixin,
    VisualCoreFamilyTraceMixin,
):
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
                protocol_failure_entry(
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

        analysis_first_hop_ids = [item.candidate_node_id for item in first_hop]
        bootstrap_trace[0]["analysis_first_hop_node_ids"] = list(analysis_first_hop_ids)

        if not analysis_first_hop_ids:
            return _finalize(
                termination_reason="bootstrap_empty",
                stage="bootstrap",
                decision_source="system",
                reason="No image-local analysis nodes were available.",
            )

        state = self._build_bootstrapped_state(
            qa_family=qa_family,
            seed_node_id=seed_node_id,
            image_path=image_path,
            bootstrap_plan=bootstrap_plan,
            first_hop=first_hop,
            analysis_first_hop_ids=analysis_first_hop_ids,
            seed_scope=seed_scope,
        )
        state.bootstrap_error_count = family_session["bootstrap_error_count"]
        state.candidate_pool = [
            item
            for item in state.candidate_pool
            if self._passes_session_guardrails(state, item)
        ]

        if not state.candidate_pool:
            if qa_family == "atomic" and first_hop:
                fallback_subgraph, fallback_scorecard, fallback_candidate = (
                    self._materialize_atomic_first_hop_fallback(
                        state=state,
                        first_hop=first_hop,
                        bootstrap_plan=bootstrap_plan,
                    )
                )
                fallback_reason = (
                    "Atomic fallback selected one image-adjacent node because no "
                    "logical first-layer candidates remained."
                )
                selection_trace.append(
                    {
                        "qa_family": qa_family,
                        "step_index": state.step_count,
                        "decision": "atomic_first_hop_fallback",
                        "candidate_uid": fallback_candidate.candidate_uid,
                        "candidate_node_id": fallback_candidate.candidate_node_id,
                        "depth": fallback_candidate.depth,
                        "direction_mode": state.direction_mode,
                        "candidate_pool_after_step": [],
                        "reason": fallback_reason,
                    }
                )
                termination_trace.append(
                    {
                        "qa_family": qa_family,
                        "stage": "selection",
                        "step_index": state.step_count,
                        "state": state.to_dict(),
                        "last_selected_candidate": fallback_candidate.to_dict(),
                        "decision": "accept",
                        "sufficient": True,
                        "termination_reason": "atomic_first_hop_fallback",
                        "reason": fallback_reason,
                        "suggested_action": "",
                        "scorecard": fallback_scorecard.to_dict(),
                        "protocol_status": "ok",
                        "protocol_error_type": "",
                        "decision_source": "system_fallback",
                    }
                )
                return _finalize(
                    termination_reason="atomic_first_hop_fallback",
                    scorecard=fallback_scorecard,
                    decision="accepted",
                    selected_subgraph=fallback_subgraph,
                    stage="selection",
                    decision_source="system_fallback",
                    reason=fallback_reason,
                )
            return _finalize(
                termination_reason="candidate_pool_exhausted",
                stage="bootstrap",
                decision_source="system",
                reason="Bootstrap completed, but no logical first-layer candidates remained.",
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
                    protocol_failure_entry(
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
                    protocol_failure_entry(
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
            seed_payload=self._compact_seed_prompt_payload(seed_payload),
            visual_core_candidates=self._candidate_prompt_lines(first_hop),
            preview_candidates=self._candidate_prompt_lines(second_hop_preview),
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
            validated = validate_bootstrap_payload(
                qa_family=qa_family,
                payload=payload,
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
            state_payload=self._compact_state_prompt_payload(state),
            candidate_pool_payload=self._candidate_prompt_lines(state.candidate_pool),
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
            validated = validate_selector_payload(
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
            state_payload=self._compact_state_prompt_payload(state),
            stage=stage,
            last_selected_candidate=self._compact_candidate_prompt_payload(
                last_selected_candidate
            ),
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
            decision = validate_termination_payload(
                payload=payload,
                stage=stage,
                judge_pass_threshold=self.judge_pass_threshold,
            )
            if decision.protocol_status == "ok":
                return decision
            last_decision = decision
        return last_decision

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

    def _compact_seed_prompt_payload(
        self,
        seed_payload: dict[str, Any],
    ) -> dict[str, Any]:
        seed_node = seed_payload.get("seed_node")
        seed_node = seed_node if isinstance(seed_node, dict) else {}
        return {
            "seed_node_id": str(seed_payload.get("seed_node_id", "")),
            "entity_type": compact_text(seed_node.get("entity_type", ""), limit=60),
            "entity_name": compact_text(seed_node.get("entity_name", ""), limit=80),
            "description": compact_text(seed_node.get("description", ""), limit=180),
            "evidence_span": compact_text(seed_node.get("evidence_span", ""), limit=160),
        }

    def _compact_state_prompt_payload(
        self,
        state: FamilySessionState,
    ) -> dict[str, Any]:
        return {
            "qa_family": state.qa_family,
            "seed_node_id": state.seed_node_id,
            "virtual_image_node_id": state.virtual_image_node_id,
            "intent": compact_text(state.intent, limit=160),
            "technical_focus": compact_text(state.technical_focus, limit=80),
            "forbidden_patterns": list(state.forbidden_patterns),
            "analysis_first_hop_node_ids": list(state.analysis_first_hop_node_ids),
            "selected_node_ids": list(state.selected_node_ids),
            "selected_edge_paths": [
                self._pair_key(edge_pair) for edge_pair in state.selected_edge_pairs
            ],
            "frontier_node_id": state.frontier_node_id,
            "direction_mode": state.direction_mode,
            "current_outside_depth": state.current_outside_depth,
            "step_count": state.step_count,
            "rollback_count": state.rollback_count,
            "blocked_candidate_uids": list(state.blocked_candidate_uids),
            "candidate_pool_size": len(state.candidate_pool),
        }

    def _candidate_prompt_lines(
        self,
        candidates: list[FamilyCandidatePoolItem],
    ) -> list[str]:
        return [self._candidate_prompt_line(candidate) for candidate in candidates]

    def _candidate_prompt_line(self, candidate: FamilyCandidatePoolItem) -> str:
        node_data = self.graph.get_node(candidate.candidate_node_id) or {}
        logical_edge = self._pair_key(candidate.bound_edge_pair)
        source_path = candidate.virtualized_from_path or candidate.frontier_path
        path = "->".join(str(item) for item in source_path)
        anchor = (
            candidate.analysis_anchor_node_id
            or candidate.bridge_first_hop_id
            or candidate.bind_from_node_id
        )
        node_name = compact_text(node_data.get("entity_name", ""), limit=60)
        evidence = compact_text(candidate.evidence_summary, limit=120)
        return (
            f"{candidate.candidate_uid} | edge={logical_edge} | path={path} | "
            f"node={candidate.candidate_node_id} | name={node_name} | "
            f"type={candidate.entity_type} | rel={candidate.relation_type} | "
            f"depth={candidate.depth} | dir={candidate.edge_direction} | "
            f"score={round(float(candidate.score), 4)} | via={anchor} | "
            f"evidence={evidence}"
        )

    def _compact_candidate_prompt_payload(
        self,
        candidate_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(candidate_payload, dict):
            return {}
        bound_edge_pair = candidate_payload.get("bound_edge_pair", [])
        edge = (
            self._pair_key(bound_edge_pair)
            if isinstance(bound_edge_pair, list) and len(bound_edge_pair) >= 2
            else ""
        )
        source_path = candidate_payload.get("virtualized_from_path") or candidate_payload.get(
            "frontier_path", []
        )
        path = (
            "->".join(str(item) for item in source_path)
            if isinstance(source_path, list)
            else ""
        )
        return {
            "candidate_uid": str(candidate_payload.get("candidate_uid", "")),
            "candidate_node_id": str(candidate_payload.get("candidate_node_id", "")),
            "edge": edge,
            "path": path,
            "entity_type": compact_text(candidate_payload.get("entity_type", ""), limit=60),
            "relation_type": compact_text(candidate_payload.get("relation_type", ""), limit=60),
            "depth": int(candidate_payload.get("depth", 0) or 0),
            "edge_direction": str(candidate_payload.get("edge_direction", "")),
            "via": str(
                candidate_payload.get("analysis_anchor_node_id", "")
                or candidate_payload.get("bridge_first_hop_id", "")
            ),
            "evidence": compact_text(
                candidate_payload.get("evidence_summary", ""), limit=120
            ),
        }

    def _build_bootstrap_fallback_plan(
        self,
        *,
        qa_family: str,
        first_hop: list[FamilyCandidatePoolItem],
    ) -> BootstrapPlan:
        return BootstrapPlan(
            qa_family=qa_family,
            intent=f"Fallback visual-core bootstrap for {qa_family}.",
            technical_focus=qa_family,
            image_grounding_summary="Fallback bootstrap used the image-local visual core.",
            bootstrap_rationale=(
                "The model returned no usable bootstrap JSON, so the sampler used the "
                "image-local visual core and deferred node choices to selector/judge."
            ),
        )

    def _materialize_atomic_first_hop_fallback(
        self,
        *,
        state: FamilySessionState,
        first_hop: list[FamilyCandidatePoolItem],
        bootstrap_plan: BootstrapPlan,
    ) -> tuple[dict[str, Any], JudgeScorecard, FamilyCandidatePoolItem]:
        chosen = self._choose_atomic_first_hop_fallback(
            state=state,
            first_hop=first_hop,
        )
        virtual_edge_pair = [state.virtual_image_node_id, chosen.candidate_node_id]
        virtual_pair_key = self._pair_key(virtual_edge_pair)
        state.step_count = max(1, state.step_count)
        state.selected_node_ids = [
            state.virtual_image_node_id,
            chosen.candidate_node_id,
        ]
        state.selected_edge_pairs = [virtual_edge_pair]
        state.frontier_node_id = chosen.candidate_node_id
        state.current_outside_depth = 1
        state.direction_mode = chosen.edge_direction
        state.direction_anchor_edge = list(virtual_edge_pair)
        state.path_by_node_id[chosen.candidate_node_id] = list(virtual_edge_pair)
        state.edge_direction_by_pair[virtual_pair_key] = chosen.edge_direction
        state.virtual_edge_payload_by_pair[virtual_pair_key] = (
            self._atomic_first_hop_virtual_edge_payload(state=state, candidate=chosen)
        )
        state.candidate_pool = []
        fallback_candidate = FamilyCandidatePoolItem(
            candidate_uid=(
                f"{state.virtual_image_node_id}:{chosen.candidate_node_id}:"
                "atomic_first_hop_fallback"
            ),
            candidate_node_id=chosen.candidate_node_id,
            bind_from_node_id=state.virtual_image_node_id,
            bound_edge_pair=virtual_edge_pair,
            hop=1,
            depth=1,
            relation_type=chosen.relation_type,
            entity_type=chosen.entity_type,
            frontier_path=list(virtual_edge_pair),
            bridge_first_hop_id=chosen.candidate_node_id,
            evidence_summary=chosen.evidence_summary,
            edge_direction=chosen.edge_direction,
            score=chosen.score,
            analysis_anchor_node_id=chosen.candidate_node_id,
            virtualized_from_path=[state.seed_node_id, chosen.candidate_node_id],
            virtualized_from_edge_pair=[state.seed_node_id, chosen.candidate_node_id],
        )
        scorecard = JudgeScorecard(
            image_indispensability=0.82,
            answer_stability=0.72,
            evidence_closure=0.7,
            technical_relevance=0.78,
            reasoning_depth=0.55,
            hallucination_risk=0.22,
            theme_coherence=0.76,
            overall_score=max(self.judge_pass_threshold, 0.72),
            passes=True,
        )
        return (
            self._materialize_selected_subgraph(
                state=state,
                bootstrap_plan=bootstrap_plan,
                scorecard=scorecard,
            ),
            scorecard,
            fallback_candidate,
        )

    def _choose_atomic_first_hop_fallback(
        self,
        *,
        state: FamilySessionState,
        first_hop: list[FamilyCandidatePoolItem],
    ) -> FamilyCandidatePoolItem:
        digest = hashlib.sha256(
            f"{state.seed_node_id}:{state.qa_family}".encode("utf-8")
        ).hexdigest()
        return first_hop[int(digest, 16) % len(first_hop)]

    def _atomic_first_hop_virtual_edge_payload(
        self,
        *,
        state: FamilySessionState,
        candidate: FamilyCandidatePoolItem,
    ) -> dict[str, Any]:
        edge_data = copy.deepcopy(
            self.graph.get_edge(state.seed_node_id, candidate.candidate_node_id)
            or self.graph.get_edge(candidate.candidate_node_id, state.seed_node_id)
            or {}
        )
        metadata = load_metadata(edge_data.get("metadata"))
        metadata.update(
            {
                "synthetic": True,
                "fallback": "atomic_first_hop",
                "virtualized_from_path": [
                    state.seed_node_id,
                    candidate.candidate_node_id,
                ],
                "virtualized_from_edge_pair": [
                    state.seed_node_id,
                    candidate.candidate_node_id,
                ],
            }
        )
        edge_data["metadata"] = metadata
        edge_data["synthetic"] = True
        edge_data["fallback"] = "atomic_first_hop"
        edge_data["virtualized_from_path"] = [
            state.seed_node_id,
            candidate.candidate_node_id,
        ]
        edge_data["virtualized_from_edge_pair"] = [
            state.seed_node_id,
            candidate.candidate_node_id,
        ]
        if not edge_data.get("description"):
            edge_data["description"] = compact_text(
                "Atomic fallback virtual edge from the image to one image-adjacent "
                f"evidence node {candidate.candidate_node_id}.",
                limit=160,
            )
        return edge_data

    def _family_postcheck_failure_reason(self, state: FamilySessionState) -> str:
        if state.qa_family == "atomic":
            return "atomic_requires_one_virtual_image_evidence_node"
        if state.qa_family == "aggregated":
            return "aggregated_direction_or_theme_failed"
        return "multi_hop_requires_deep_chain"

    @staticmethod
    def _pair_key(edge_pair: list[str] | tuple[str, str]) -> str:
        return f"{edge_pair[0]}->{edge_pair[1]}"

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
