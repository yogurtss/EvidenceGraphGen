from collections import deque
from typing import Any

from graphgen.bases import BaseGraphStorage, BaseLLMWrapper

from .artifacts import (
    JudgeScorecard,
    SelectedSubgraphArtifact,
    clip_score,
    compact_text,
    extract_json_payload,
    load_metadata,
    normalize_edge_pair,
    stabilize_allowed_values,
    split_source_ids,
)
from .constants import ALLOWED_DEGRADED_QUESTION_TYPES, ALLOWED_PRIMARY_QUESTION_TYPES
from .debug_artifacts import (
    DebugTrace,
    snapshot_candidate_like,
    snapshot_judge,
    snapshot_neighborhood,
)
from .v2_artifacts import (
    AgentSessionTrace,
    CandidateSubgraphState,
    GraphEditAction,
    JudgeFeedback,
)
from .v2_prompts import (
    build_v2_candidate_prompt,
    build_v2_editor_prompt,
    build_v2_judge_prompt,
    build_v2_neighborhood_prompt,
)


class GraphEditingVLMSubgraphSampler:
    def __init__(
        self,
        graph: BaseGraphStorage,
        llm_client: BaseLLMWrapper,
        *,
        hard_cap_units: int = 12,
        max_rounds: int = 6,
        max_vqas_per_selected_subgraph: int = 2,
        allow_degraded: bool = True,
        judge_pass_threshold: float = 0.68,
    ):
        self.graph = graph
        self.llm_client = llm_client
        self.hard_cap_units = max(4, int(hard_cap_units))
        self.max_rounds = max(1, int(max_rounds))
        self.max_vqas_per_selected_subgraph = min(
            3, max(1, int(max_vqas_per_selected_subgraph))
        )
        self.allow_degraded = bool(allow_degraded)
        self.judge_pass_threshold = float(judge_pass_threshold)

    async def sample(
        self,
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]],
        *,
        seed_node_id: str,
        debug: bool = False,
    ) -> dict[str, Any]:
        nodes, _edges = batch
        seed_node = self.graph.get_node(seed_node_id) or {}
        image_path = self._extract_image_path(seed_node)
        debug_trace = DebugTrace(sampler_version="v2", seed_node_id=seed_node_id) if debug else None
        if debug_trace is not None:
            debug_trace.add_step(
                phase="setup",
                step_type="seed_validation",
                status="success" if seed_node_id and seed_node and image_path else "failed",
                summary=(
                    "Validated seed node and image asset."
                    if seed_node_id and seed_node and image_path
                    else "Seed node or image asset missing."
                ),
                snapshot={
                    "seed_node_id": seed_node_id,
                    "has_seed_node": bool(seed_node),
                    "image_path": image_path,
                },
            )
        if not seed_node_id or not seed_node or not image_path:
            result = self._build_empty_result(
                seed_node_id=seed_node_id,
                image_path=image_path,
                degraded=False,
                degraded_reason="missing_image_seed_or_asset",
                termination_reason="missing_image_seed_or_asset",
                neighborhood_trace=[],
                edit_trace=[],
                judge_trace=[],
                candidate_states=[],
                candidate_bundle=[],
                agent_session=AgentSessionTrace(
                    session_id=f"{seed_node_id}-v2",
                    seed_node_id=seed_node_id,
                    status="abstained",
                    termination_reason="missing_image_seed_or_asset",
                ).to_dict(),
            )
            return self._attach_debug_trace(
                result,
                debug_trace=debug_trace,
                final_status="abstained",
                termination_reason="missing_image_seed_or_asset",
            )

        primary = await self._run_session(
            nodes=nodes,
            seed_node_id=seed_node_id,
            seed_node=seed_node,
            image_path=image_path,
            degraded=False,
            debug_trace=debug_trace,
        )
        if primary["selected_subgraphs"]:
            return self._attach_debug_trace(
                primary,
                debug_trace=debug_trace,
                final_status="selected",
                termination_reason=primary["termination_reason"],
            )

        if self.allow_degraded:
            if debug_trace is not None:
                debug_trace.add_step(
                    phase="fallback",
                    step_type="degraded_retry",
                    status="running",
                    degraded=True,
                    summary="Retrying graph editing sampler in degraded mode.",
                    snapshot={"trigger": primary["termination_reason"]},
                )
            degraded = await self._run_session(
                nodes=nodes,
                seed_node_id=seed_node_id,
                seed_node=seed_node,
                image_path=image_path,
                degraded=True,
                debug_trace=debug_trace,
            )
            if degraded["selected_subgraphs"]:
                return self._attach_debug_trace(
                    degraded,
                    debug_trace=debug_trace,
                    final_status="selected",
                    termination_reason=degraded["termination_reason"],
                )
            primary["candidate_bundle"].extend(degraded.get("candidate_bundle", []))
            primary["edit_trace"].extend(degraded.get("edit_trace", []))
            primary["judge_trace"].extend(degraded.get("judge_trace", []))
            primary["candidate_states"].extend(degraded.get("candidate_states", []))
            primary["neighborhood_trace"].extend(degraded.get("neighborhood_trace", []))
            primary["termination_reason"] = degraded.get(
                "termination_reason", primary["termination_reason"]
            )
            primary["degraded_reason"] = degraded.get("degraded_reason", "")
            primary["agent_session"] = degraded.get("agent_session", primary["agent_session"])
        return self._attach_debug_trace(
            primary,
            debug_trace=debug_trace,
            final_status="abstained",
            termination_reason=primary["termination_reason"],
        )

    async def _run_session(
        self,
        *,
        nodes: list[tuple[str, dict]],
        seed_node_id: str,
        seed_node: dict,
        image_path: str,
        degraded: bool,
        debug_trace: DebugTrace | None = None,
    ) -> dict[str, Any]:
        seed_scope = self._collect_seed_scope(seed_node_id, nodes)
        if debug_trace is not None:
            debug_trace.add_step(
                phase="setup",
                step_type="seed_scope",
                status="success",
                degraded=degraded,
                summary="Collected provenance scope from the image seed.",
                snapshot={"seed_scope": sorted(seed_scope)},
            )
        neighborhood = self._collect_neighborhood(
            seed_node_id=seed_node_id,
            max_hops=1,
            seed_scope=seed_scope,
        )
        neighborhood_trace = [self._neighborhood_snapshot(neighborhood, hop=1)]
        if debug_trace is not None:
            debug_trace.add_step(
                phase="setup",
                step_type="neighborhood_collection",
                status="success",
                degraded=degraded,
                summary="Collected initial one-hop neighborhood.",
                snapshot=snapshot_neighborhood(neighborhood, hop=1),
            )
        state = CandidateSubgraphState(
            candidate_id="candidate-1",
            seed_node_id=seed_node_id,
            node_ids=[seed_node_id],
            degraded=degraded,
        )
        candidate_states = [state.to_dict()]
        if debug_trace is not None:
            debug_trace.add_step(
                phase="editing",
                step_type="candidate_state",
                status="success",
                degraded=degraded,
                summary="Initialized editable candidate state.",
                snapshot=self._state_snapshot(state),
            )
        edit_trace: list[dict[str, Any]] = []
        judge_trace: list[dict[str, Any]] = []
        candidate_bundle: list[dict[str, Any]] = []
        last_judge_feedback: JudgeFeedback | None = None
        expanded_to_two_hop = False

        for round_index in range(1, self.max_rounds + 1):
            state.status = "editing"
            editor_payload = await self._edit_round(
                seed_node_id=seed_node_id,
                image_path=image_path,
                degraded=degraded,
                round_index=round_index,
                neighborhood=neighborhood,
                current_state=state,
                last_judge_feedback=last_judge_feedback,
            )
            actions = self._parse_actions(editor_payload.get("actions", []))
            self._apply_editor_state_updates(
                state=state,
                payload=editor_payload,
                degraded=degraded,
            )
            commit_requested = False
            round_actions = []
            for action in actions:
                before_units = state.unit_count()
                applied, ignored_reason = self._apply_action(
                    state=state,
                    action=action,
                    neighborhood=neighborhood,
                )
                after_units = state.unit_count()
                action.applied = applied
                action.ignored_reason = ignored_reason
                action.before_units = before_units
                action.after_units = after_units
                round_actions.append(action.to_dict())
                if action.action_type == "commit_for_judgement":
                    commit_requested = True

            if not round_actions:
                round_actions.append(
                    GraphEditAction(
                        action_type="query_nodes",
                        note="editor_returned_no_actions",
                        applied=False,
                        ignored_reason="no_actions",
                        before_units=state.unit_count(),
                        after_units=state.unit_count(),
                    ).to_dict()
                )

            edit_trace.append(
                {
                    "round_index": round_index,
                    "degraded": degraded,
                    "actions": round_actions,
                }
            )
            candidate_states.append(state.to_dict())
            if debug_trace is not None:
                debug_trace.add_step(
                    phase="editing",
                    step_type="edit_round",
                    status="success",
                    degraded=degraded,
                    summary=f"Applied edit round {round_index}.",
                    snapshot={
                        **self._state_snapshot(state),
                        "round_index": round_index,
                        "actions": list(round_actions),
                    },
                )

            if not commit_requested and state.unit_count() >= self.hard_cap_units:
                commit_requested = True

            if not commit_requested and round_index < self.max_rounds:
                continue

            judge_feedback = await self._judge_state(
                seed_node_id=seed_node_id,
                image_path=image_path,
                degraded=degraded,
                state=state,
            )
            last_judge_feedback = judge_feedback
            judge_trace.append(
                {
                    "round_index": round_index,
                    "degraded": degraded,
                    **judge_feedback.to_dict(),
                }
            )
            if debug_trace is not None:
                debug_trace.add_step(
                    phase="judge",
                    step_type="candidate_judgement",
                    status="success" if judge_feedback.sufficient else "failed",
                    degraded=degraded,
                    summary=(
                        f"Judge {'accepted' if judge_feedback.sufficient else 'rejected'} "
                        f"{state.candidate_id} in round {round_index}."
                    ),
                    snapshot={
                        **self._state_snapshot(state),
                        "round_index": round_index,
                        **snapshot_judge(
                            scorecard=judge_feedback.scorecard,
                            rejection_reason=judge_feedback.rejection_reason,
                            sufficient=judge_feedback.sufficient,
                            needs_expansion=judge_feedback.needs_expansion,
                            suggested_actions=judge_feedback.suggested_actions,
                        ),
                    },
                )
            candidate_bundle.append(
                {
                    "candidate_id": state.candidate_id,
                    "intent": state.intent,
                    "node_ids": list(state.node_ids),
                    "edge_pairs": [list(pair) for pair in state.edge_pairs],
                    "judge_scores": judge_feedback.scorecard.to_dict(),
                    "decision": "accepted" if judge_feedback.sufficient else "rejected",
                    "rejection_reason": ""
                    if judge_feedback.sufficient
                    else judge_feedback.rejection_reason or "rejected_by_judge",
                }
            )
            directional_ok = (
                self._is_directionally_consistent(state)
                if self._requires_directional_consistency(state)
                else True
            )
            if judge_feedback.sufficient and directional_ok:
                state.status = "accepted"
                selected = self._materialize_selected_subgraph(state, judge_feedback)
                result = self._build_result(
                    seed_node_id=seed_node_id,
                    image_path=image_path,
                    degraded=degraded,
                    degraded_reason=(
                        "fallback_to_conservative_chart_interpretation" if degraded else ""
                    ),
                    selected_subgraphs=[selected.to_dict()],
                    candidate_bundle=candidate_bundle,
                    neighborhood_trace=neighborhood_trace,
                    edit_trace=edit_trace,
                    judge_trace=judge_trace,
                    candidate_states=candidate_states,
                    termination_reason="judge_marked_sufficient",
                    agent_session=AgentSessionTrace(
                        session_id=f"{seed_node_id}-v2",
                        seed_node_id=seed_node_id,
                        rounds=round_index,
                        degraded=degraded,
                        status="accepted",
                        termination_reason="judge_marked_sufficient",
                    ).to_dict(),
                )
                if debug_trace is not None:
                    debug_trace.add_step(
                        phase="finalize",
                        step_type="selection",
                        status="success",
                        degraded=degraded,
                        summary=f"Selected {state.candidate_id}.",
                        snapshot={
                            "selected_candidate_ids": [state.candidate_id],
                            "round_index": round_index,
                        },
                    )
                return result

            if (
                judge_feedback.needs_expansion
                and not expanded_to_two_hop
                and round_index < self.max_rounds
            ):
                neighborhood = self._collect_neighborhood(
                    seed_node_id=seed_node_id,
                    max_hops=2,
                    seed_scope=seed_scope,
                )
                neighborhood_trace.append(self._neighborhood_snapshot(neighborhood, hop=2))
                expanded_to_two_hop = True
                if debug_trace is not None:
                    debug_trace.add_step(
                        phase="editing",
                        step_type="neighborhood_expansion",
                        status="success",
                        degraded=degraded,
                        summary="Expanded neighborhood from one hop to two hops.",
                        snapshot=snapshot_neighborhood(neighborhood, hop=2),
                    )
                continue

        termination_reason = "degraded_exhausted" if degraded else "no_candidate_passed_judge"
        result = self._build_empty_result(
            seed_node_id=seed_node_id,
            image_path=image_path,
            degraded=degraded,
            degraded_reason=(
                "fallback_to_conservative_chart_interpretation" if degraded else ""
            ),
            termination_reason=termination_reason,
            neighborhood_trace=neighborhood_trace,
            edit_trace=edit_trace,
            judge_trace=judge_trace,
            candidate_states=candidate_states,
            candidate_bundle=candidate_bundle,
            agent_session=AgentSessionTrace(
                session_id=f"{seed_node_id}-v2",
                seed_node_id=seed_node_id,
                rounds=self.max_rounds,
                degraded=degraded,
                status="abstained",
                termination_reason=termination_reason,
            ).to_dict(),
        )
        if debug_trace is not None:
            debug_trace.add_step(
                phase="finalize",
                step_type="selection",
                status="failed",
                degraded=degraded,
                summary="No candidate state was accepted.",
                snapshot={
                    "selected_candidate_ids": [],
                    "candidate_bundle_size": len(candidate_bundle),
                },
            )
        return result

    def _is_directionally_consistent(self, state: CandidateSubgraphState) -> bool:
        """Require edges to progress outward from seed (distance must increase by 1)."""
        if len(state.node_ids) <= 1:
            return True
        adjacency = {node_id: set() for node_id in state.node_ids}
        for src_id, tgt_id in state.edge_pairs:
            if src_id in adjacency and tgt_id in adjacency:
                adjacency[src_id].add(tgt_id)
                adjacency[tgt_id].add(src_id)
        queue = deque([(state.seed_node_id, 0)])
        distances = {state.seed_node_id: 0}
        while queue:
            node_id, distance = queue.popleft()
            for neighbor_id in adjacency.get(node_id, set()):
                if neighbor_id in distances:
                    continue
                distances[neighbor_id] = distance + 1
                queue.append((neighbor_id, distance + 1))
        if any(node_id not in distances for node_id in state.node_ids):
            return False
        for src_id, tgt_id in state.edge_pairs:
            if src_id not in distances or tgt_id not in distances:
                return False
            if abs(distances[src_id] - distances[tgt_id]) != 1:
                return False
        return True

    @staticmethod
    def _requires_directional_consistency(state: CandidateSubgraphState) -> bool:
        return "multi_hop" in set(state.approved_question_types)

    async def _edit_round(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        degraded: bool,
        round_index: int,
        neighborhood: dict[str, Any],
        current_state: CandidateSubgraphState,
        last_judge_feedback: JudgeFeedback | None,
    ) -> dict[str, Any]:
        allowed_question_types = (
            ALLOWED_DEGRADED_QUESTION_TYPES
            if degraded
            else ALLOWED_PRIMARY_QUESTION_TYPES
        )
        prompt = build_v2_editor_prompt(
            seed_node_id=seed_node_id,
            image_path=image_path,
            degraded=degraded,
            hard_cap_units=self.hard_cap_units,
            round_index=round_index,
            allowed_question_types=allowed_question_types,
            current_state=current_state,
            neighborhood_prompt=build_v2_neighborhood_prompt(self.graph, neighborhood),
            last_judge_feedback=last_judge_feedback,
        )
        raw = await self.llm_client.generate_answer(prompt, image_path=image_path or None)
        return extract_json_payload(raw)

    async def _judge_state(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        degraded: bool,
        state: CandidateSubgraphState,
    ) -> JudgeFeedback:
        prompt = build_v2_judge_prompt(
            seed_node_id=seed_node_id,
            image_path=image_path,
            degraded=degraded,
            current_state=state,
            candidate_prompt=build_v2_candidate_prompt(self.graph, state),
            judge_pass_threshold=self.judge_pass_threshold,
        )
        raw = await self.llm_client.generate_answer(prompt, image_path=image_path or None)
        payload = extract_json_payload(raw)
        if not isinstance(payload, dict):
            return self._rejected_feedback("invalid_judge_payload")
        scorecard = JudgeScorecard(
            image_indispensability=clip_score(payload.get("image_indispensability")),
            answer_stability=clip_score(payload.get("answer_stability")),
            evidence_closure=clip_score(payload.get("evidence_closure")),
            technical_relevance=clip_score(payload.get("technical_relevance")),
            reasoning_depth=clip_score(payload.get("reasoning_depth")),
            hallucination_risk=clip_score(payload.get("hallucination_risk"), default=1.0),
            theme_coherence=clip_score(payload.get("theme_coherence")),
            overall_score=clip_score(payload.get("overall_score")),
            passes=bool(payload.get("passes", False)),
        )
        mandatory_pass = (
            scorecard.image_indispensability >= 0.65
            and scorecard.answer_stability >= 0.6
            and scorecard.evidence_closure >= 0.6
            and scorecard.technical_relevance >= 0.6
            and scorecard.hallucination_risk <= 0.45
            and scorecard.overall_score >= self.judge_pass_threshold
        )
        sufficient = bool(payload.get("sufficient", False)) and scorecard.passes and mandatory_pass
        scorecard.passes = sufficient
        return JudgeFeedback(
            scorecard=scorecard,
            sufficient=sufficient,
            needs_expansion=bool(payload.get("needs_expansion", False)),
            rejection_reason=compact_text(payload.get("rejection_reason", ""), limit=160),
            suggested_actions=[
                compact_text(item, limit=40)
                for item in payload.get("suggested_actions", [])
                if str(item).strip()
            ],
        )

    def _parse_actions(self, raw_actions: list[Any]) -> list[GraphEditAction]:
        actions = []
        for item in raw_actions:
            if not isinstance(item, dict):
                continue
            action_type = str(item.get("action_type", "")).strip()
            if not action_type:
                continue
            actions.append(
                GraphEditAction(
                    action_type=action_type,
                    node_id=str(item.get("node_id", "")).strip(),
                    anchor_node_id=str(item.get("anchor_node_id", "")).strip(),
                    src_id=str(item.get("src_id", "")).strip(),
                    tgt_id=str(item.get("tgt_id", "")).strip(),
                    intent=compact_text(item.get("intent", ""), limit=120),
                    technical_focus=compact_text(item.get("technical_focus", ""), limit=80),
                    approved_question_types=[
                        str(q).strip()
                        for q in item.get("approved_question_types", [])
                        if str(q).strip()
                    ],
                    note=compact_text(item.get("note", ""), limit=160),
                )
            )
        # Enforce one action per round for predictable edit trajectories.
        return actions[:1]

    def _apply_editor_state_updates(
        self,
        *,
        state: CandidateSubgraphState,
        payload: dict[str, Any],
        degraded: bool,
    ) -> None:
        allowed_question_types = (
            ALLOWED_DEGRADED_QUESTION_TYPES if degraded else ALLOWED_PRIMARY_QUESTION_TYPES
        )
        if payload.get("intent"):
            state.intent = compact_text(payload["intent"], limit=120)
        if payload.get("technical_focus"):
            state.technical_focus = compact_text(payload["technical_focus"], limit=80)
        question_types = stabilize_allowed_values(
            payload.get("approved_question_types", []),
            allowed_question_types,
        )
        if question_types:
            state.approved_question_types = question_types
        if payload.get("image_grounding_summary"):
            state.image_grounding_summary = compact_text(
                payload["image_grounding_summary"], limit=240
            )
        if payload.get("evidence_summary"):
            state.evidence_summary = compact_text(payload["evidence_summary"], limit=240)

    def _apply_action(
        self,
        *,
        state: CandidateSubgraphState,
        action: GraphEditAction,
        neighborhood: dict[str, Any],
    ) -> tuple[bool, str]:
        available_nodes = set(neighborhood.get("node_ids", []))
        available_edges = {
            normalize_edge_pair(src_id, tgt_id)
            for src_id, tgt_id, _ in neighborhood.get("edges", [])
        }
        action_type = action.action_type
        if action_type in {"query_nodes", "query_edges", "commit_for_judgement"}:
            return True, ""
        if action_type == "add_node":
            node_id = action.node_id
            anchor_node_id = action.anchor_node_id
            if node_id not in available_nodes:
                return False, "node_not_in_neighborhood"
            if node_id in state.node_ids:
                return False, "node_already_present"
            if not anchor_node_id or anchor_node_id not in state.node_ids:
                return False, "anchor_node_missing"
            pair = normalize_edge_pair(action.src_id, action.tgt_id)
            if pair not in available_edges:
                return False, "edge_not_in_neighborhood"
            if node_id not in pair or anchor_node_id not in pair:
                return False, "edge_not_binding_anchor_and_new_node"
            distances = neighborhood.get("distances", {})
            anchor_distance = int(distances.get(anchor_node_id, 10**6))
            node_distance = int(distances.get(node_id, 10**6))
            if (
                self._requires_directional_consistency(state)
                and node_distance <= anchor_distance
            ):
                return False, "direction_not_outward"
            if state.unit_count() + 2 > self.hard_cap_units:
                return False, "hard_cap_exceeded"
            state.node_ids.append(node_id)
            state.edge_pairs.append([pair[0], pair[1]])
            return True, ""
        if action_type == "remove_node":
            node_id = action.node_id
            if node_id == state.seed_node_id:
                return False, "cannot_remove_seed"
            if node_id not in state.node_ids:
                return False, "node_missing"
            state.node_ids = [item for item in state.node_ids if item != node_id]
            state.edge_pairs = [
                pair
                for pair in state.edge_pairs
                if node_id not in pair
            ]
            return True, ""
        if action_type == "add_edge":
            return False, "add_edge_disabled_use_add_node_binding"
        if action_type == "remove_edge":
            pair = normalize_edge_pair(action.src_id, action.tgt_id)
            before = len(state.edge_pairs)
            state.edge_pairs = [
                edge_pair
                for edge_pair in state.edge_pairs
                if normalize_edge_pair(edge_pair[0], edge_pair[1]) != pair
            ]
            if len(state.edge_pairs) == before:
                return False, "edge_missing"
            return True, ""
        if action_type == "revise_intent":
            if action.intent:
                state.intent = action.intent
            if action.technical_focus:
                state.technical_focus = action.technical_focus
            if action.approved_question_types:
                state.approved_question_types = stabilize_allowed_values(
                    action.approved_question_types,
                    ALLOWED_DEGRADED_QUESTION_TYPES
                    if state.degraded
                    else ALLOWED_PRIMARY_QUESTION_TYPES,
                    fallback=state.approved_question_types,
                )
            return True, ""
        return False, "unsupported_action"

    def _collect_seed_scope(
        self, seed_node_id: str, nodes: list[tuple[str, dict]]
    ) -> set[str]:
        seed_scope = set()
        for node_id, node_data in nodes:
            if node_id != seed_node_id:
                continue
            metadata = load_metadata(node_data.get("metadata"))
            seed_scope.update(split_source_ids(node_data.get("source_id", "")))
            seed_scope.update(split_source_ids(metadata.get("source_trace_id", "")))
            break
        return seed_scope

    def _collect_neighborhood(
        self,
        *,
        seed_node_id: str,
        max_hops: int,
        seed_scope: set[str],
    ) -> dict[str, Any]:
        queue = deque([(seed_node_id, 0)])
        visited = {seed_node_id}
        distances = {seed_node_id: 0}
        node_ids = [seed_node_id]
        while queue:
            node_id, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbor_id in self.graph.get_neighbors(node_id):
                if neighbor_id in visited:
                    continue
                edge_data = self.graph.get_edge(node_id, neighbor_id) or self.graph.get_edge(
                    neighbor_id, node_id
                ) or {}
                node_data = self.graph.get_node(neighbor_id) or {}
                if not self._passes_provenance_guardrail(node_data, edge_data, seed_scope):
                    continue
                visited.add(neighbor_id)
                distances[str(neighbor_id)] = depth + 1
                node_ids.append(str(neighbor_id))
                queue.append((neighbor_id, depth + 1))
        edge_payloads = []
        node_set = set(node_ids)
        for node_id in node_ids:
            for neighbor_id in self.graph.get_neighbors(node_id):
                if neighbor_id not in node_set:
                    continue
                pair = normalize_edge_pair(node_id, neighbor_id)
                edge_data = self.graph.get_edge(node_id, neighbor_id) or self.graph.get_edge(
                    neighbor_id, node_id
                )
                if edge_data and (pair[0], pair[1], edge_data) not in edge_payloads:
                    edge_payloads.append((pair[0], pair[1], edge_data))
        ranked = [seed_node_id] + sorted(
            [node_id for node_id in node_ids if node_id != seed_node_id],
            key=lambda item: (distances.get(item, 99), item),
        )
        return {
            "node_ids": ranked,
            "edges": edge_payloads,
            "distances": distances,
            "max_hops": max_hops,
        }

    def _passes_provenance_guardrail(
        self,
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

    def _materialize_selected_subgraph(
        self,
        state: CandidateSubgraphState,
        judge_feedback: JudgeFeedback,
    ) -> SelectedSubgraphArtifact:
        edge_pairs = {
            normalize_edge_pair(src_id, tgt_id)
            for src_id, tgt_id in state.edge_pairs
        }
        return SelectedSubgraphArtifact(
            subgraph_id=state.candidate_id,
            technical_focus=state.technical_focus,
            nodes=self._node_payloads(set(state.node_ids)),
            edges=self._edge_payloads(edge_pairs),
            image_grounding_summary=state.image_grounding_summary,
            evidence_summary=state.evidence_summary,
            judge_scores=judge_feedback.scorecard,
            approved_question_types=state.approved_question_types,
            degraded=state.degraded,
        )

    def _node_payloads(self, node_ids: set[str]) -> list[tuple[str, dict]]:
        payloads = []
        for node_id in sorted(node_ids):
            node_data = self.graph.get_node(node_id)
            if node_data:
                payloads.append((node_id, node_data))
        return payloads

    def _edge_payloads(
        self, edge_pairs: set[tuple[str, str]]
    ) -> list[tuple[str, str, dict]]:
        payloads = []
        for src_id, tgt_id in sorted(edge_pairs):
            edge_data = self.graph.get_edge(src_id, tgt_id) or self.graph.get_edge(
                tgt_id, src_id
            )
            if edge_data:
                payloads.append((src_id, tgt_id, edge_data))
        return payloads

    def _extract_image_path(self, node_data: dict[str, Any]) -> str:
        metadata = load_metadata(node_data.get("metadata"))
        for key in ("image_path", "img_path"):
            if metadata.get(key):
                return str(metadata[key])
        return ""

    def _neighborhood_snapshot(self, neighborhood: dict[str, Any], *, hop: int) -> dict[str, Any]:
        return {
            "hop": hop,
            "node_count": len(neighborhood.get("node_ids", [])),
            "edge_count": len(neighborhood.get("edges", [])),
            "node_ids": list(neighborhood.get("node_ids", [])),
        }

    def _state_snapshot(self, state: CandidateSubgraphState) -> dict[str, Any]:
        return snapshot_candidate_like(
            candidate_id=state.candidate_id,
            intent=state.intent,
            technical_focus=state.technical_focus,
            approved_question_types=state.approved_question_types,
            node_ids=state.node_ids,
            edge_pairs=state.edge_pairs,
            unit_count=state.unit_count(),
            extra={"status": state.status},
        )

    def _rejected_feedback(self, reason: str) -> JudgeFeedback:
        return JudgeFeedback(
            scorecard=JudgeScorecard(
                hallucination_risk=1.0,
                passes=False,
            ),
            sufficient=False,
            needs_expansion=False,
            rejection_reason=reason,
            suggested_actions=[],
        )

    @staticmethod
    def _attach_debug_trace(
        result: dict[str, Any],
        *,
        debug_trace: DebugTrace | None,
        final_status: str,
        termination_reason: str,
    ) -> dict[str, Any]:
        if debug_trace is None:
            return result
        debug_trace.finalize(
            final_status=final_status,
            termination_reason=termination_reason,
        )
        result["debug_trace"] = debug_trace.to_dict()
        return result

    def _build_result(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        degraded: bool,
        degraded_reason: str,
        selected_subgraphs: list[dict[str, Any]],
        candidate_bundle: list[dict[str, Any]],
        neighborhood_trace: list[dict[str, Any]],
        edit_trace: list[dict[str, Any]],
        judge_trace: list[dict[str, Any]],
        candidate_states: list[dict[str, Any]],
        termination_reason: str,
        agent_session: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "seed_node_id": seed_node_id,
            "seed_image_path": image_path,
            "selection_mode": "single",
            "degraded": degraded,
            "degraded_reason": degraded_reason,
            "selected_subgraphs": selected_subgraphs,
            "candidate_bundle": candidate_bundle,
            "abstained": False,
            "max_vqas_per_selected_subgraph": self.max_vqas_per_selected_subgraph,
            "sampler_version": "v2",
            "agent_session": agent_session,
            "candidate_states": candidate_states,
            "edit_trace": edit_trace,
            "judge_trace": judge_trace,
            "neighborhood_trace": neighborhood_trace,
            "termination_reason": termination_reason,
        }

    def _build_empty_result(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        degraded: bool,
        degraded_reason: str,
        termination_reason: str,
        neighborhood_trace: list[dict[str, Any]],
        edit_trace: list[dict[str, Any]],
        judge_trace: list[dict[str, Any]],
        candidate_states: list[dict[str, Any]],
        candidate_bundle: list[dict[str, Any]],
        agent_session: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "seed_node_id": seed_node_id,
            "seed_image_path": image_path,
            "selection_mode": "single",
            "degraded": degraded,
            "degraded_reason": degraded_reason,
            "selected_subgraphs": [],
            "candidate_bundle": candidate_bundle,
            "abstained": True,
            "max_vqas_per_selected_subgraph": self.max_vqas_per_selected_subgraph,
            "sampler_version": "v2",
            "agent_session": agent_session,
            "candidate_states": candidate_states,
            "edit_trace": edit_trace,
            "judge_trace": judge_trace,
            "neighborhood_trace": neighborhood_trace,
            "termination_reason": termination_reason,
        }
