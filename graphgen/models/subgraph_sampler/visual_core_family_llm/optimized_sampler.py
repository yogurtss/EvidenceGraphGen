import copy
import hashlib
from pathlib import Path
from typing import Any

from graphgen.models.subgraph_sampler.artifacts import (
    JudgeScorecard,
    compact_text,
    extract_json_payload,
    load_metadata,
    split_source_ids,
    to_json_compatible,
)

from .models import (
    BootstrapPlan,
    DEFAULT_OPTIMIZED_FAMILY_MAX_DEPTHS,
    DEFAULT_OPTIMIZED_FAMILY_SUBGRAPH_TARGETS,
    FamilyCandidatePoolItem,
    FamilySessionState,
    FamilyTerminationDecision,
    IntentPlannerStageResult,
    SelectorStageResult,
)
from .prompts import (
    build_selector_prompt,
    build_shared_intent_planner_prompt,
    build_termination_prompt,
)
from .sampler import VisualCoreFamilyLLMSubgraphSampler
from .validators import (
    protocol_failure_entry,
    validate_intent_planner_payload,
    validate_selector_payload,
    validate_termination_payload,
)


class OptimizedVisualCoreFamilyLLMSubgraphSampler(VisualCoreFamilyLLMSubgraphSampler):
    """Intent-first visual-core sampler that keeps real image graph structure."""

    SAMPLER_VERSION = "family_llm_optimized_v1"
    FAMILY_ORDER = ("atomic", "aggregated", "multi_hop")
    MIN_ACCEPT_DEPTH = {"atomic": 2, "aggregated": 2, "multi_hop": 3}

    def __init__(
        self,
        graph,
        llm_client,
        *,
        family_subgraph_targets: dict[str, int] | None = None,
        family_qa_targets: dict[str, int] | None = None,
        family_max_depths: dict[str, int] | None = None,
        max_steps_per_family: int = 4,
        max_rollbacks_per_family: int = 1,
        judge_pass_threshold: float = 0.68,
        same_source_only: bool = True,
        bootstrap_preview_limit: int = 12,
        candidate_prompt_limit: int = 30,
        max_invalid_selections: int = 2,
        max_protocol_retries_per_stage: int = 1,
        max_selector_errors: int = 2,
        max_judge_errors: int = 1,
        min_multi_hop_outside_core_edges: int = 2,
    ):
        targets = (
            family_subgraph_targets
            if family_subgraph_targets is not None
            else family_qa_targets
        )
        self.family_subgraph_targets = self._merge_positive_int_map(
            DEFAULT_OPTIMIZED_FAMILY_SUBGRAPH_TARGETS,
            self._normalize_family_map(targets),
        )
        optimized_max_depths = self._merge_non_negative_int_map(
            DEFAULT_OPTIMIZED_FAMILY_MAX_DEPTHS,
            self._normalize_family_map(family_max_depths),
        )
        optimized_max_depths = {
            family: max(optimized_max_depths[family], self.MIN_ACCEPT_DEPTH[family])
            for family in self.FAMILY_ORDER
        }
        super().__init__(
            graph,
            llm_client,
            family_qa_targets={family: 1 for family in self.FAMILY_ORDER},
            family_max_depths=optimized_max_depths,
            max_steps_per_family=max_steps_per_family,
            max_rollbacks_per_family=max_rollbacks_per_family,
            judge_pass_threshold=judge_pass_threshold,
            same_source_only=same_source_only,
            bootstrap_preview_limit=bootstrap_preview_limit,
            max_invalid_selections=max_invalid_selections,
            allow_bootstrap_fallback=False,
            max_protocol_retries_per_stage=max_protocol_retries_per_stage,
            max_bootstrap_errors=1,
            max_selector_errors=max_selector_errors,
            max_judge_errors=max_judge_errors,
            min_multi_hop_outside_core_edges=max(3, int(min_multi_hop_outside_core_edges)),
        )
        self.candidate_prompt_limit = max(1, int(candidate_prompt_limit))

    async def sample(self, *, seed_node_id: str) -> dict[str, Any]:
        seed_node = self.graph.get_node(seed_node_id) or {}
        image_path = self._extract_image_path(seed_node)
        if not image_path:
            return self._missing_image_result(seed_node_id)

        seed_scope = self._collect_seed_scope(seed_node_id) if self.same_source_only else set()
        runtime_schema = self._infer_runtime_schema(
            seed_node_id=seed_node_id,
            seed_scope=seed_scope,
        )
        seed_doc_name = self._doc_name_for_payload(seed_node)
        selected_subgraphs = []
        candidate_bundle = []
        family_sessions = []
        family_bootstrap_trace = []
        family_selection_trace = []
        family_termination_trace = []
        intent_bundle = []
        first_hop = self._collect_visual_core_candidates(
            seed_node_id=seed_node_id,
            seed_scope=seed_scope,
        )
        preview = self._collect_preview_candidates(
            seed_node_id=seed_node_id,
            first_hop_candidates=first_hop,
            seed_scope=seed_scope,
        )
        max_intent_count = max(self.family_subgraph_targets.values())
        planner_result = await self._plan_shared_intents(
            target_count=max_intent_count,
            image_path=image_path,
            seed_payload={
                "seed_node_id": seed_node_id,
                "seed_node": to_json_compatible(seed_node),
            },
            first_hop=first_hop,
            preview=preview,
            runtime_schema=runtime_schema,
            seed_doc_name=seed_doc_name,
        )
        shared_plans = planner_result.plans
        if planner_result.protocol_status != "ok":
            shared_plans = self._fallback_intent_plans(
                qa_family="shared",
                target_count=max_intent_count,
            )

        for qa_family in self.FAMILY_ORDER:
            plans = self._choose_family_intents(
                shared_plans=shared_plans,
                qa_family=qa_family,
                target_count=self.family_subgraph_targets[qa_family],
                seed_node_id=seed_node_id,
            )

            intent_bundle.extend(
                {
                    **plan.to_dict(),
                    "intent_index": index,
                    "protocol_status": planner_result.protocol_status,
                    "protocol_error_type": planner_result.protocol_error_type,
                    "reason": planner_result.reason,
                }
                for index, plan in enumerate(plans, start=1)
            )

            for intent_index, plan in enumerate(plans, start=1):
                session_result = await self._run_intent_session(
                    seed_node_id=seed_node_id,
                    image_path=image_path,
                    seed_scope=seed_scope,
                    qa_family=qa_family,
                    intent_index=intent_index,
                    bootstrap_plan=plan,
                    first_hop=first_hop,
                    preview=preview,
                    planner_result=planner_result,
                    seed_doc_name=seed_doc_name,
                )
                family_sessions.append(session_result["family_session"])
                family_bootstrap_trace.extend(session_result["bootstrap_trace"])
                family_selection_trace.extend(session_result["selection_trace"])
                family_termination_trace.extend(session_result["termination_trace"])
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
        visualization_trace["sampler_version"] = self.SAMPLER_VERSION
        return {
            "seed_node_id": seed_node_id,
            "seed_image_path": image_path,
            "selection_mode": "multi" if len(selected_subgraphs) > 1 else "single",
            "selected_subgraphs": selected_subgraphs,
            "candidate_bundle": candidate_bundle,
            "abstained": not bool(selected_subgraphs),
            "sampler_version": self.SAMPLER_VERSION,
            "termination_reason": (
                "optimized_family_sessions_completed"
                if selected_subgraphs
                else "no_optimized_family_subgraph_selected"
            ),
            "family_sessions": family_sessions,
            "family_bootstrap_trace": family_bootstrap_trace,
            "family_selection_trace": family_selection_trace,
            "family_termination_trace": family_termination_trace,
            "inferred_schema": runtime_schema,
            "intent_bundle": intent_bundle,
            "max_vqas_per_selected_subgraph": 1,
            "visualization_trace": visualization_trace,
        }

    async def _run_intent_session(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        seed_scope: set[str],
        qa_family: str,
        intent_index: int,
        bootstrap_plan: BootstrapPlan,
        first_hop: list[FamilyCandidatePoolItem],
        preview: list[FamilyCandidatePoolItem],
        planner_result: IntentPlannerStageResult,
        seed_doc_name: str,
    ) -> dict[str, Any]:
        state = self._build_optimized_state(
            qa_family=qa_family,
            seed_node_id=seed_node_id,
            image_path=image_path,
            bootstrap_plan=bootstrap_plan,
            first_hop=first_hop,
            seed_scope=seed_scope,
        )
        state.candidate_pool = [
            item
            for item in state.candidate_pool
            if self._passes_session_guardrails(state, item)
        ]
        protocol_failures = []
        if planner_result.protocol_status != "ok":
            protocol_failures.append(
                protocol_failure_entry(
                    stage="intent_planner",
                    error_type=planner_result.protocol_error_type or "parse_error",
                    reason=planner_result.reason or "intent_planner_protocol_error",
                    raw_payload=planner_result.raw_payload,
                )
            )

        family_session = {
            "session_id": f"{seed_node_id}-{qa_family}-{intent_index}-optimized-family-llm",
            "qa_family": qa_family,
            "intent_index": intent_index,
            "seed_node_id": seed_node_id,
            "target_qa_count": 1,
            "target_subgraph_count": self.family_subgraph_targets[qa_family],
            "status": "abstained",
            "abstained": True,
            "termination_reason": "candidate_pool_exhausted",
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
                "intent_index": intent_index,
                "seed_node_id": seed_node_id,
                "visual_core_candidates": [item.to_dict() for item in first_hop],
                "preview_candidates": [item.to_dict() for item in preview],
                "bootstrap_plan": bootstrap_plan.to_dict(),
                "analysis_first_hop_node_ids": [
                    item.candidate_node_id for item in first_hop
                ],
                "protocol_status": planner_result.protocol_status,
                "protocol_error_type": planner_result.protocol_error_type,
                "reason": planner_result.reason,
            }
        ]
        selection_trace = []
        termination_trace = []

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
            family_session["steps"] = state.step_count
            family_session["rollbacks"] = state.rollback_count
            family_session["selector_error_count"] = state.selector_error_count
            family_session["judge_error_count"] = state.judge_error_count
            family_session["invalid_selection_count"] = state.invalid_selection_count
            family_session["invalid_candidate_repeat_count"] = (
                state.invalid_candidate_repeat_count
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
                node_ids=state.selected_node_ids,
                edge_pairs=state.selected_edge_pairs,
                decision=decision,
                rejection_reason="" if selected_subgraph else termination_reason,
                scorecard=scorecard,
                abstained=not bool(selected_subgraph),
                protocol_failures=protocol_failures,
            )
            candidate_bundle["candidate_id"] = (
                f"{qa_family}-{intent_index}-optimized-visual-core"
            )
            candidate_bundle["intent_index"] = intent_index
            if selected_subgraph:
                selected_subgraph["subgraph_id"] = (
                    f"{seed_node_id}-{qa_family}-{intent_index}-optimized-visual-core"
                )
                selected_subgraph["target_qa_count"] = 1
                selected_subgraph["sampler_version"] = self.SAMPLER_VERSION
                selected_subgraph["intent_index"] = intent_index
                selected_subgraph["virtual_image_node_id"] = ""
            return {
                "family_session": family_session,
                "bootstrap_trace": bootstrap_trace,
                "selection_trace": selection_trace,
                "termination_trace": termination_trace,
                "candidate_bundle": candidate_bundle,
                "selected_subgraph": selected_subgraph,
            }

        if not state.candidate_pool:
            return _finalize(
                termination_reason="candidate_pool_exhausted",
                stage="bootstrap",
                decision_source="system",
                reason="No selectable first-layer candidates were available.",
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
                seed_doc_name=seed_doc_name,
                prompt_salt=f"{seed_node_id}:{qa_family}:{intent_index}:selector:{state.step_count}",
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
                    invalid_key = selector_result.candidate_node_id or selector_result.candidate_uid
                    if invalid_key and invalid_key == state.last_invalid_candidate_uid:
                        state.invalid_candidate_repeat_count += 1
                    else:
                        state.invalid_candidate_repeat_count = 1
                    state.last_invalid_candidate_uid = invalid_key
                    selection_trace.append(
                        {
                            "qa_family": qa_family,
                            "intent_index": intent_index,
                            "step_index": state.step_count + 1,
                            "decision": "invalid_selection",
                            "candidate_uid": selector_result.candidate_uid,
                            "candidate_node_id": selector_result.candidate_node_id,
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

            if selector_result.decision == "stop_selection":
                return _finalize(
                    termination_reason="selector_stop_requested",
                    stage="selection",
                    decision_source="selector",
                    reason=selector_result.reason,
                )

            previous_snapshot = state.snapshot()
            state.invalid_selection_count = 0
            state.last_invalid_candidate_uid = ""
            state.invalid_candidate_repeat_count = 0
            state.step_count += 1
            candidate = next(
                item
                for item in state.candidate_pool
                if item.candidate_uid == selector_result.candidate_uid
            )
            self._apply_candidate_selection(state=state, candidate=candidate)
            judge_decision = await self._judge_state(
                qa_family=qa_family,
                state=state,
                image_path=image_path,
                stage="selection",
                last_selected_candidate=candidate.to_dict(),
                seed_doc_name=seed_doc_name,
                prompt_salt=f"{seed_node_id}:{qa_family}:{intent_index}:judge:{state.step_count}",
            )
            termination_trace.append(
                {
                    "qa_family": qa_family,
                    "intent_index": intent_index,
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
                return _finalize(
                    termination_reason="judge_protocol_error",
                    scorecard=judge_decision.scorecard,
                    stage="selection",
                    decision_source=judge_decision.decision_source,
                    protocol_status="error",
                    reason=judge_decision.reason or "judge_protocol_error",
                    protocol_error_type=judge_decision.protocol_error_type,
                )

            depth_limit_hit = False
            if judge_decision.decision in {"accept", "continue"}:
                depth_limit_hit = self._update_candidate_pool_after_judge(
                    state=state,
                    candidate=candidate,
                    seed_scope=seed_scope,
                    max_depth=self.family_max_depths[qa_family],
                )
                selection_trace.append(
                    {
                        "qa_family": qa_family,
                        "intent_index": intent_index,
                        "step_index": state.step_count,
                        "decision": "select_candidate",
                        "candidate_uid": candidate.candidate_uid,
                        "candidate_node_id": candidate.candidate_node_id,
                        "depth": candidate.depth,
                        "direction_mode": state.direction_mode,
                        "candidate_pool_after_step": [
                            item.to_dict() for item in state.candidate_pool
                        ],
                        "reason": selector_result.reason,
                    }
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
                    return _finalize(
                        termination_reason=(
                            "max_depth_reached" if depth_limit_hit else "candidate_pool_exhausted"
                        ),
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
                state.blocked_candidate_uids.append(candidate.candidate_uid)
                state.candidate_pool = [
                    item
                    for item in state.candidate_pool
                    if item.candidate_uid != candidate.candidate_uid
                ]
                selection_trace.append(
                    {
                        "qa_family": qa_family,
                        "intent_index": intent_index,
                        "step_index": state.step_count,
                        "decision": "rollback_last_step",
                        "candidate_uid": candidate.candidate_uid,
                        "candidate_node_id": candidate.candidate_node_id,
                        "rollback_count": state.rollback_count,
                        "state_after_rollback": state.to_dict(),
                        "candidate_pool_after_step": [
                            item.to_dict() for item in state.candidate_pool
                        ],
                    }
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

        return _finalize(
            termination_reason="max_steps_reached",
            stage="selection",
            decision_source="system",
            reason="The optimized family session exhausted its step budget.",
        )

    async def _plan_shared_intents(
        self,
        *,
        target_count: int,
        image_path: str,
        seed_payload: dict[str, Any],
        first_hop: list[FamilyCandidatePoolItem],
        preview: list[FamilyCandidatePoolItem],
        runtime_schema: dict[str, Any],
        seed_doc_name: str,
    ) -> IntentPlannerStageResult:
        prompt = build_shared_intent_planner_prompt(
            target_count=target_count,
            seed_payload=self._compact_seed_prompt_payload(seed_payload),
            first_layer_candidates=self._candidate_prompt_lines(
                self._limit_candidates_for_prompt(
                    first_hop,
                    seed_doc_name=seed_doc_name,
                    salt=f"{seed_payload.get('seed_node_id')}:shared:first",
                )
            ),
            preview_candidates=self._candidate_prompt_lines(
                self._limit_candidates_for_prompt(
                    preview,
                    seed_doc_name=seed_doc_name,
                    salt=f"{seed_payload.get('seed_node_id')}:shared:preview",
                )
            ),
            runtime_schema=runtime_schema,
        )
        last_result = IntentPlannerStageResult(
            protocol_status="error",
            protocol_error_type="parse_error",
            reason="empty_intent_planner_payload",
        )
        for _ in range(self.max_protocol_retries_per_stage + 1):
            raw = await self.llm_client.generate_answer(prompt, image_path=image_path or None)
            payload = extract_json_payload(raw)
            validated = validate_intent_planner_payload(
                qa_family="shared",
                payload=payload,
                target_count=target_count,
            )
            if validated.protocol_status == "ok":
                return validated
            last_result = validated
        return last_result

    async def _select_next_candidate(
        self,
        *,
        qa_family: str,
        state: FamilySessionState,
        image_path: str,
        seed_doc_name: str = "",
        prompt_salt: str = "",
    ) -> SelectorStageResult:
        prompt_candidates = self._limit_candidates_for_prompt(
            state.candidate_pool,
            seed_doc_name=seed_doc_name,
            salt=prompt_salt,
        )
        prompt = build_selector_prompt(
            qa_family=qa_family,
            state_payload=self._compact_state_prompt_payload(state),
            candidate_pool_payload=self._candidate_prompt_lines(prompt_candidates),
        )
        last_result = SelectorStageResult(
            protocol_status="error",
            protocol_error_type="parse_error",
            reason="empty_selector_payload",
        )
        for _ in range(self.max_protocol_retries_per_stage + 1):
            raw = await self.llm_client.generate_answer(prompt, image_path=image_path or None)
            payload = extract_json_payload(raw)
            validation_state = copy.copy(state)
            validation_state.candidate_pool = list(prompt_candidates)
            validated = validate_selector_payload(
                payload=payload,
                state=validation_state,
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
        seed_doc_name: str = "",
        prompt_salt: str = "",
    ) -> FamilyTerminationDecision:
        prompt_candidates = self._limit_candidates_for_prompt(
            state.candidate_pool,
            seed_doc_name=seed_doc_name,
            salt=prompt_salt,
        )
        prompt = build_termination_prompt(
            qa_family=qa_family,
            state_payload=self._compact_state_prompt_payload(state),
            stage=stage,
            last_selected_candidate=self._compact_candidate_prompt_payload(
                last_selected_candidate
            ),
            candidate_pool_payload=self._candidate_prompt_lines(prompt_candidates),
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
            raw = await self.llm_client.generate_answer(prompt, image_path=image_path or None)
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

    def _build_optimized_state(
        self,
        *,
        qa_family: str,
        seed_node_id: str,
        image_path: str,
        bootstrap_plan: BootstrapPlan,
        first_hop: list[FamilyCandidatePoolItem],
        seed_scope: set[str],
    ) -> FamilySessionState:
        path_by_node_id = {seed_node_id: [seed_node_id]}
        first_hop_ids = [item.candidate_node_id for item in first_hop]
        candidate_pool = []
        seen = set()
        for candidate in first_hop:
            if candidate.candidate_uid in seen:
                continue
            seen.add(candidate.candidate_uid)
            candidate_pool.append(candidate)
        candidate_pool.sort(key=lambda item: (item.depth, -item.score, item.candidate_uid))
        return FamilySessionState(
            qa_family=qa_family,
            seed_node_id=seed_node_id,
            image_path=image_path,
            virtual_image_node_id="",
            intent=bootstrap_plan.intent or f"{qa_family} optimized visual-core intent",
            technical_focus=bootstrap_plan.technical_focus or qa_family,
            image_grounding_summary=bootstrap_plan.image_grounding_summary,
            bootstrap_rationale=bootstrap_plan.bootstrap_rationale,
            forbidden_patterns=list(bootstrap_plan.forbidden_patterns),
            visual_core_node_ids=[seed_node_id],
            analysis_first_hop_node_ids=first_hop_ids,
            analysis_only_node_ids=[],
            selected_node_ids=[seed_node_id],
            selected_edge_pairs=[],
            candidate_pool=candidate_pool,
            frontier_node_id=seed_node_id,
            path_by_node_id=path_by_node_id,
            edge_direction_by_pair={},
        )

    def _apply_candidate_selection(
        self,
        *,
        state: FamilySessionState,
        candidate: FamilyCandidatePoolItem,
    ) -> None:
        if candidate.candidate_node_id not in state.selected_node_ids:
            state.selected_node_ids.append(candidate.candidate_node_id)
        state.selected_edge_pairs.append(list(candidate.bound_edge_pair))
        state.path_by_node_id[candidate.candidate_node_id] = list(candidate.frontier_path)
        state.edge_direction_by_pair[self._pair_key(candidate.bound_edge_pair)] = candidate.edge_direction
        state.frontier_node_id = candidate.candidate_node_id
        state.current_outside_depth = max(state.current_outside_depth, int(candidate.depth))
        if not state.direction_mode:
            state.direction_mode = candidate.edge_direction
            state.direction_anchor_edge = list(candidate.bound_edge_pair)
        selected_nodes = set(state.selected_node_ids)
        state.candidate_pool = [
            item
            for item in state.candidate_pool
            if item.candidate_node_id not in selected_nodes
            and item.candidate_uid not in state.blocked_candidate_uids
        ]

    def _update_candidate_pool_after_judge(
        self,
        *,
        state: FamilySessionState,
        candidate: FamilyCandidatePoolItem,
        seed_scope: set[str],
        max_depth: int,
    ) -> bool:
        if state.qa_family == "atomic" and candidate.depth >= self.MIN_ACCEPT_DEPTH["atomic"]:
            state.candidate_pool = []
            return False
        if state.qa_family == "aggregated" and candidate.depth >= 2:
            return super()._update_aggregated_candidate_pool_after_judge(
                state=state,
                candidate=candidate,
                seed_scope=seed_scope,
                max_depth=max_depth,
            )
        if state.qa_family == "multi_hop" or candidate.depth < self.MIN_ACCEPT_DEPTH[state.qa_family]:
            next_candidates, depth_limit_hit = self._build_candidates_from_bind_node(
                bind_from_node_id=candidate.candidate_node_id,
                selected_node_ids=set(state.selected_node_ids),
                path_by_node_id=state.path_by_node_id,
                visual_core_node_ids=set(state.visual_core_node_ids),
                seed_scope=seed_scope,
                max_depth=max_depth,
                blocked_candidate_uids=state.blocked_candidate_uids,
            )
            next_candidates = [
                item
                for item in next_candidates
                if item.candidate_node_id not in set(state.selected_node_ids)
                and self._is_direction_compatible(state, item)
                and self._passes_session_guardrails(state, item)
            ]
            state.candidate_pool = sorted(
                {item.candidate_uid: item for item in next_candidates}.values(),
                key=lambda item: (item.depth, -item.score, item.candidate_uid),
            )
            return depth_limit_hit
        state.candidate_pool = [
            item
            for item in state.candidate_pool
            if item.candidate_node_id not in set(state.selected_node_ids)
            and item.candidate_uid not in state.blocked_candidate_uids
            and self._passes_session_guardrails(state, item)
        ]
        return False

    def _passes_family_postcheck(self, state: FamilySessionState) -> bool:
        if state.current_outside_depth < self.MIN_ACCEPT_DEPTH[state.qa_family]:
            return False
        if not self._is_directionally_consistent(state):
            return False
        if state.qa_family == "atomic":
            return (
                len(state.selected_node_ids) == 3
                and len(state.selected_edge_pairs) == 2
            )
        if state.qa_family == "aggregated":
            return len(state.selected_node_ids) >= 3
        return self._passes_optimized_multi_hop_postcheck(state)

    def _passes_optimized_multi_hop_postcheck(self, state: FamilySessionState) -> bool:
        frontier_path = list(state.path_by_node_id.get(state.frontier_node_id, []))
        if len(frontier_path) != len(set(frontier_path)):
            return False
        if not frontier_path or frontier_path[0] != state.seed_node_id:
            return False
        return state.current_outside_depth >= self.MIN_ACCEPT_DEPTH["multi_hop"]

    def _family_postcheck_failure_reason(self, state: FamilySessionState) -> str:
        if state.qa_family == "atomic":
            return "atomic_requires_depth_at_least_2_with_real_image_path"
        if state.qa_family == "aggregated":
            return "aggregated_requires_second_layer_evidence"
        return "multi_hop_requires_depth_greater_than_2"

    def _candidate_prompt_line(self, candidate: FamilyCandidatePoolItem) -> str:
        node_data = self.graph.get_node(candidate.candidate_node_id) or {}
        edge_data = self.graph.get_edge(*candidate.bound_edge_pair) or self.graph.get_edge(
            candidate.bound_edge_pair[1],
            candidate.bound_edge_pair[0],
        ) or {}
        logical_edge = self._pair_key(candidate.bound_edge_pair)
        node_name = compact_text(node_data.get("entity_name", ""), limit=60)
        description = compact_text(
            " ".join(
                part
                for part in (
                    node_data.get("description", ""),
                    node_data.get("evidence_span", ""),
                    edge_data.get("description", ""),
                    edge_data.get("evidence_span", ""),
                    candidate.evidence_summary,
                )
                if part
            ),
            limit=180,
        )
        return (
            f"node={candidate.candidate_node_id} | edge={logical_edge} | name={node_name} | "
            f"type={candidate.entity_type} | rel={candidate.relation_type} | "
            f"depth={candidate.depth} | dir={candidate.edge_direction} | "
            f"description={description}"
        )

    def _compact_state_prompt_payload(self, state: FamilySessionState) -> dict[str, Any]:
        payload = super()._compact_state_prompt_payload(state)
        payload.pop("virtual_image_node_id", None)
        return payload

    def _limit_candidates_for_prompt(
        self,
        candidates: list[FamilyCandidatePoolItem],
        *,
        seed_doc_name: str,
        salt: str,
    ) -> list[FamilyCandidatePoolItem]:
        if len(candidates) <= self.candidate_prompt_limit:
            return list(candidates)
        same_doc = [
            item for item in candidates if self._candidate_matches_doc(item, seed_doc_name)
        ]
        other = [
            item for item in candidates if not self._candidate_matches_doc(item, seed_doc_name)
        ]
        same_doc = sorted(same_doc, key=lambda item: (item.depth, -item.score, item.candidate_uid))
        if len(same_doc) >= self.candidate_prompt_limit:
            return same_doc[: self.candidate_prompt_limit]
        needed = self.candidate_prompt_limit - len(same_doc)
        other = sorted(
            other,
            key=lambda item: self._stable_digest(f"{salt}:{item.candidate_uid}"),
        )
        return same_doc + other[:needed]

    def _candidate_matches_doc(
        self,
        candidate: FamilyCandidatePoolItem,
        seed_doc_name: str,
    ) -> bool:
        if not seed_doc_name:
            return False
        node_data = self.graph.get_node(candidate.candidate_node_id) or {}
        edge_data = self.graph.get_edge(*candidate.bound_edge_pair) or self.graph.get_edge(
            candidate.bound_edge_pair[1],
            candidate.bound_edge_pair[0],
        ) or {}
        return seed_doc_name in {
            self._doc_name_for_payload(node_data),
            self._doc_name_for_payload(edge_data),
        }

    def _doc_name_for_payload(self, payload: dict[str, Any]) -> str:
        metadata = load_metadata(payload.get("metadata"))
        for key in ("source_name", "source_file", "source_path", "path"):
            if metadata.get(key):
                return self._normalize_doc_name(metadata[key])
        for key in ("source_name", "source_file", "source_path", "path"):
            if payload.get(key):
                return self._normalize_doc_name(payload[key])
        source_ids = sorted(split_source_ids(payload.get("source_id", "")))
        if source_ids:
            return self._normalize_doc_name(source_ids[0])
        return ""

    @staticmethod
    def _normalize_doc_name(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        name = Path(text).name
        if "." in name:
            stem = Path(name).stem
            if stem:
                return stem
        return name

    @staticmethod
    def _stable_digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _fallback_intent_plans(
        self,
        *,
        qa_family: str,
        target_count: int,
    ) -> list[BootstrapPlan]:
        return [
            BootstrapPlan(
                qa_family=qa_family,
                intent=f"Fallback optimized {qa_family} intent {index}.",
                technical_focus=qa_family,
                forbidden_patterns=[],
                image_grounding_summary="Fallback intent derived from the image-local graph.",
                bootstrap_rationale="The intent planner did not return valid JSON, so a deterministic fallback intent was used.",
            )
            for index in range(1, target_count + 1)
        ]

    def _choose_family_intents(
        self,
        *,
        shared_plans: list[BootstrapPlan],
        qa_family: str,
        target_count: int,
        seed_node_id: str,
    ) -> list[BootstrapPlan]:
        ranked = sorted(
            enumerate(shared_plans),
            key=lambda item: self._stable_digest(
                f"{seed_node_id}:{qa_family}:{item[0]}:{item[1].intent}"
            ),
        )
        chosen = [plan for _, plan in ranked[:target_count]]
        return [
            BootstrapPlan(
                qa_family=qa_family,
                intent=plan.intent,
                technical_focus=plan.technical_focus or qa_family,
                forbidden_patterns=list(plan.forbidden_patterns),
                image_grounding_summary=plan.image_grounding_summary,
                bootstrap_rationale=plan.bootstrap_rationale,
            )
            for plan in chosen
        ]

    def _missing_image_result(self, seed_node_id: str) -> dict[str, Any]:
        trace = self._empty_visualization_trace(seed_node_id=seed_node_id, image_path="")
        trace["sampler_version"] = self.SAMPLER_VERSION
        return {
            "seed_node_id": seed_node_id,
            "seed_image_path": "",
            "selection_mode": "single",
            "selected_subgraphs": [],
            "candidate_bundle": [],
            "abstained": True,
            "sampler_version": self.SAMPLER_VERSION,
            "termination_reason": "missing_image_asset",
            "family_sessions": [],
            "family_bootstrap_trace": [],
            "family_selection_trace": [],
            "family_termination_trace": [],
            "inferred_schema": {},
            "intent_bundle": [],
            "max_vqas_per_selected_subgraph": 1,
            "visualization_trace": trace,
        }

    def _empty_visualization_trace(
        self,
        *,
        seed_node_id: str,
        image_path: str,
    ) -> dict[str, Any]:
        trace = super()._empty_visualization_trace(
            seed_node_id=seed_node_id,
            image_path=image_path,
        )
        trace["sampler_version"] = self.SAMPLER_VERSION
        return trace

    @staticmethod
    def _normalize_family_map(values: dict[str, int] | None) -> dict[str, int] | None:
        if values is None:
            return None
        normalized = {}
        for key, value in values.items():
            raw_key = str(key).strip().lower()
            normalized_key = "aggregated" if raw_key == "aggregate" else raw_key
            normalized[normalized_key] = value
        return normalized
