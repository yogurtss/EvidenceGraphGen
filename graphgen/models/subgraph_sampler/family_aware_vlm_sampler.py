from typing import Any

from .artifacts import (
    JudgeScorecard,
    SelectedSubgraphArtifact,
    compact_text,
    normalize_edge_pair,
)
from .graph_editing_vlm_sampler import GraphEditingVLMSubgraphSampler
from .v2_artifacts import AgentSessionTrace, CandidateSubgraphState, GraphEditAction
from .v2_prompts import build_v2_candidate_prompt, build_v2_neighborhood_prompt
from .v3_prompts import build_v3_editor_prompt, build_v3_judge_prompt


class FamilyAwareVLMSubgraphSampler(GraphEditingVLMSubgraphSampler):
    FAMILY_ORDER = ("atomic", "aggregated", "multi_hop")

    def __init__(
        self,
        graph,
        llm_client,
        *,
        hard_cap_units: int = 12,
        max_rounds: int = 6,
        max_vqas_per_selected_subgraph: int = 2,
        judge_pass_threshold: float = 0.68,
        max_multi_hop_hops: int = 3,
        min_subgraphs_per_family: int = 2,
        max_subgraphs_per_family: int = 3,
    ):
        super().__init__(
            graph,
            llm_client,
            hard_cap_units=hard_cap_units,
            max_rounds=max_rounds,
            max_vqas_per_selected_subgraph=max_vqas_per_selected_subgraph,
            allow_degraded=False,
            judge_pass_threshold=judge_pass_threshold,
        )
        self.max_multi_hop_hops = max(2, int(max_multi_hop_hops))
        self.min_subgraphs_per_family = max(1, int(min_subgraphs_per_family))
        self.max_subgraphs_per_family = max(
            self.min_subgraphs_per_family, int(max_subgraphs_per_family)
        )

    async def sample(
        self,
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]],
        *,
        seed_node_id: str,
    ) -> dict[str, Any]:
        nodes, _edges = batch
        seed_node = self.graph.get_node(seed_node_id) or {}
        image_path = self._extract_image_path(seed_node)
        if not seed_node_id or not seed_node or not image_path:
            return self._build_v3_result(
                seed_node_id=seed_node_id,
                image_path=image_path,
                selected_subgraphs=[],
                candidate_bundle=[],
                neighborhood_trace=[],
                edit_trace=[],
                judge_trace=[],
                candidate_states=[],
                agent_sessions=[],
                abstained=True,
                termination_reason="missing_image_seed_or_asset",
            )

        seed_scope = self._collect_seed_scope(seed_node_id, nodes)
        selected_subgraphs = []
        candidate_bundle = []
        neighborhood_trace = []
        edit_trace = []
        judge_trace = []
        candidate_states = []
        agent_sessions = []
        family_results = []

        for qa_family in self.FAMILY_ORDER:
            if not self._family_precheck(seed_node_id=seed_node_id, seed_scope=seed_scope, qa_family=qa_family):
                agent_sessions.append(
                    {
                        **AgentSessionTrace(
                            session_id=f"{seed_node_id}-{qa_family}-v3",
                            seed_node_id=seed_node_id,
                            degraded=False,
                            status="skipped",
                            termination_reason="family_precheck_failed",
                        ).to_dict(),
                        "qa_family": qa_family,
                    }
                )
                continue

            result = await self._run_family_session(
                nodes=nodes,
                seed_node_id=seed_node_id,
                image_path=image_path,
                qa_family=qa_family,
                seed_scope=seed_scope,
            )
            family_results.append(result)
            if result["selected_subgraphs"]:
                selected_subgraphs.extend(result["selected_subgraphs"])
            candidate_bundle.extend(result["candidate_bundle"])
            neighborhood_trace.extend(result["neighborhood_trace"])
            edit_trace.extend(result["edit_trace"])
            judge_trace.extend(result["judge_trace"])
            candidate_states.extend(result["candidate_states"])
            agent_sessions.append(result["agent_session"])

        if selected_subgraphs:
            termination_reason = "family_sessions_completed"
        elif family_results:
            termination_reason = "no_family_candidate_passed_judge"
        else:
            termination_reason = "all_family_prechecks_failed"
        return self._build_v3_result(
            seed_node_id=seed_node_id,
            image_path=image_path,
            selected_subgraphs=selected_subgraphs,
            candidate_bundle=candidate_bundle,
            neighborhood_trace=neighborhood_trace,
            edit_trace=edit_trace,
            judge_trace=judge_trace,
            candidate_states=candidate_states,
            agent_sessions=agent_sessions,
            abstained=not bool(selected_subgraphs),
            termination_reason=termination_reason,
        )

    async def _run_family_session(
        self,
        *,
        nodes: list[tuple[str, dict]],
        seed_node_id: str,
        image_path: str,
        qa_family: str,
        seed_scope: set[str],
    ) -> dict[str, Any]:
        _ = nodes
        settings = self._family_settings(qa_family)
        original_hard_cap_units = self.hard_cap_units
        self.hard_cap_units = settings["hard_cap_units"]
        current_hops = settings["start_hops"]
        neighborhood = self._collect_neighborhood(
            seed_node_id=seed_node_id,
            max_hops=current_hops,
            seed_scope=seed_scope,
        )
        neighborhood_trace = [
            {
                **self._neighborhood_snapshot(neighborhood, hop=current_hops),
                "qa_family": qa_family,
            }
        ]
        candidate_states: list[dict[str, Any]] = []
        edit_trace: list[dict[str, Any]] = []
        judge_trace: list[dict[str, Any]] = []
        candidate_bundle: list[dict[str, Any]] = []
        selected_subgraphs: list[dict[str, Any]] = []
        selected_signatures: set[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = set()
        family_target = self._family_target_count(qa_family)

        try:
            for candidate_index in range(1, family_target + 1):
                state = CandidateSubgraphState(
                    candidate_id=f"{qa_family}-candidate-{candidate_index}",
                    seed_node_id=seed_node_id,
                    approved_question_types=[qa_family],
                    node_ids=[seed_node_id],
                )
                candidate_states.append({**state.to_dict(), "qa_family": qa_family})
                last_judge_feedback = None
                frontier_node_id = seed_node_id

                for round_index in range(1, settings["max_rounds"] + 1):
                    state.status = "editing"
                    editor_payload = await self._edit_family_round(
                        seed_node_id=seed_node_id,
                        image_path=image_path,
                        qa_family=qa_family,
                        round_index=round_index,
                        hard_cap_units=settings["hard_cap_units"],
                        neighborhood=neighborhood,
                        current_state=state,
                        last_judge_feedback=last_judge_feedback,
                    )
                    self._apply_family_state_updates(
                        state=state,
                        payload=editor_payload,
                        qa_family=qa_family,
                    )
                    actions = self._parse_actions(editor_payload.get("actions", []))
                    commit_requested = False
                    round_actions = []
                    frontier_updated = False
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
                        if (
                            qa_family == "multi_hop"
                            and action.action_type == "add_node"
                            and action.applied
                        ):
                            frontier_node_id = action.node_id
                            frontier_updated = True

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
                            "qa_family": qa_family,
                            "candidate_id": state.candidate_id,
                            "round_index": round_index,
                            "actions": round_actions,
                        }
                    )
                    candidate_states.append({**state.to_dict(), "qa_family": qa_family})

                    if qa_family == "multi_hop" and frontier_updated and frontier_node_id:
                        neighborhood, current_hops = self._advance_multihop_frontier(
                            seed_node_id=seed_node_id,
                            seed_scope=seed_scope,
                            state=state,
                            frontier_node_id=frontier_node_id,
                            current_hops=current_hops,
                            max_hops=settings["max_hops"],
                        )
                        neighborhood_trace.append(
                            {
                                **self._neighborhood_snapshot(neighborhood, hop=current_hops),
                                "qa_family": qa_family,
                                "frontier_node_id": frontier_node_id,
                            }
                        )

                    if not commit_requested and state.unit_count() >= settings["hard_cap_units"]:
                        commit_requested = True
                    if not commit_requested and round_index < settings["max_rounds"]:
                        continue

                    judge_feedback = await self._judge_family_state(
                        seed_node_id=seed_node_id,
                        image_path=image_path,
                        qa_family=qa_family,
                        state=state,
                    )
                    last_judge_feedback = judge_feedback
                    judge_trace.append(
                        {
                            "qa_family": qa_family,
                            "candidate_id": state.candidate_id,
                            "round_index": round_index,
                            **judge_feedback.to_dict(),
                        }
                    )
                    candidate_bundle.append(
                        {
                            "candidate_id": state.candidate_id,
                            "qa_family": qa_family,
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

                    if judge_feedback.sufficient and self._passes_family_postcheck(
                        qa_family=qa_family,
                        seed_node_id=seed_node_id,
                        state=state,
                    ):
                        signature = self._candidate_signature(state)
                        if signature not in selected_signatures:
                            selected_signatures.add(signature)
                            state.status = "accepted"
                            selected = self._materialize_family_selected_subgraph(
                                qa_family=qa_family,
                                state=state,
                                judge_scorecard=judge_feedback.scorecard,
                            )
                            selected_subgraphs.append(selected.to_dict())
                        break

                    if (
                        judge_feedback.needs_expansion
                        and current_hops < settings["max_hops"]
                        and round_index < settings["max_rounds"]
                    ):
                        current_hops += 1
                        neighborhood = self._collect_neighborhood(
                            seed_node_id=seed_node_id,
                            max_hops=current_hops,
                            seed_scope=seed_scope,
                        )
                        neighborhood_trace.append(
                            {
                                **self._neighborhood_snapshot(neighborhood, hop=current_hops),
                                "qa_family": qa_family,
                            }
                        )
                        continue

                if len(selected_subgraphs) >= self.max_subgraphs_per_family:
                    break

            return {
                "selected_subgraphs": selected_subgraphs,
                "candidate_bundle": candidate_bundle,
                "neighborhood_trace": neighborhood_trace,
                "edit_trace": edit_trace,
                "judge_trace": judge_trace,
                "candidate_states": candidate_states,
                "agent_session": {
                    **AgentSessionTrace(
                        session_id=f"{seed_node_id}-{qa_family}-v3",
                        seed_node_id=seed_node_id,
                        rounds=settings["max_rounds"] * family_target,
                        degraded=False,
                        status="accepted" if selected_subgraphs else "abstained",
                        termination_reason=(
                            "family_target_satisfied"
                            if len(selected_subgraphs) >= self.min_subgraphs_per_family
                            else "insufficient_unique_family_subgraphs"
                        ),
                    ).to_dict(),
                    "qa_family": qa_family,
                },
            }
        finally:
            self.hard_cap_units = original_hard_cap_units

    async def _edit_family_round(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        qa_family: str,
        round_index: int,
        hard_cap_units: int,
        neighborhood: dict[str, Any],
        current_state: CandidateSubgraphState,
        last_judge_feedback,
    ) -> dict[str, Any]:
        prompt = build_v3_editor_prompt(
            qa_family=qa_family,
            seed_node_id=seed_node_id,
            image_path=image_path,
            hard_cap_units=hard_cap_units,
            round_index=round_index,
            current_state=current_state,
            neighborhood_prompt=build_v2_neighborhood_prompt(self.graph, neighborhood),
            last_judge_feedback=last_judge_feedback,
        )
        raw = await self.llm_client.generate_answer(prompt, image_path=image_path or None)
        return self._extract_json_payload(raw)

    async def _judge_family_state(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        qa_family: str,
        state: CandidateSubgraphState,
    ):
        prompt = build_v3_judge_prompt(
            qa_family=qa_family,
            seed_node_id=seed_node_id,
            image_path=image_path,
            current_state=state,
            candidate_prompt=build_v2_candidate_prompt(self.graph, state),
            judge_pass_threshold=self.judge_pass_threshold,
        )
        raw = await self.llm_client.generate_answer(prompt, image_path=image_path or None)
        return self._judge_feedback_from_payload(self._extract_json_payload(raw))

    def _apply_family_state_updates(
        self,
        *,
        state: CandidateSubgraphState,
        payload: dict[str, Any],
        qa_family: str,
    ) -> None:
        state.approved_question_types = [qa_family]
        if payload.get("intent"):
            state.intent = compact_text(payload["intent"], limit=120)
        if payload.get("technical_focus"):
            state.technical_focus = compact_text(payload["technical_focus"], limit=80)
        if payload.get("image_grounding_summary"):
            state.image_grounding_summary = compact_text(
                payload["image_grounding_summary"], limit=240
            )
        if payload.get("evidence_summary"):
            state.evidence_summary = compact_text(payload["evidence_summary"], limit=240)

    def _family_precheck(
        self,
        *,
        seed_node_id: str,
        seed_scope: set[str],
        qa_family: str,
    ) -> bool:
        settings = self._family_settings(qa_family)
        neighborhood = self._collect_neighborhood(
            seed_node_id=seed_node_id,
            max_hops=settings["max_hops"],
            seed_scope=seed_scope,
        )
        node_count = len(neighborhood.get("node_ids", []))
        edge_count = len(neighborhood.get("edges", []))
        max_distance = max(neighborhood.get("distances", {}).values(), default=0)
        if qa_family == "atomic":
            return node_count >= 2 and edge_count >= 1
        if qa_family == "aggregated":
            return node_count >= 3 and edge_count >= 2
        return node_count >= 3 and edge_count >= 2 and max_distance >= 2

    def _passes_family_postcheck(
        self,
        *,
        qa_family: str,
        seed_node_id: str,
        state: CandidateSubgraphState,
    ) -> bool:
        if qa_family == "atomic":
            return (
                len(state.node_ids) >= 2
                and len(state.edge_pairs) >= 1
                and len(state.node_ids) + len(state.edge_pairs) <= 3
            )
        if qa_family == "aggregated":
            return (
                len(state.node_ids) >= 3
                and len(state.edge_pairs) >= 2
                and self._breadth_score(seed_node_id, state) >= 2
            )
        max_distance = self._max_path_distance_from_seed(seed_node_id, state)
        branch_edges = max(0, len(state.edge_pairs) - max_distance)
        return (
            len(state.node_ids) >= 3
            and len(state.edge_pairs) >= 2
            and max_distance >= 2
            and branch_edges <= 2
            and self._is_directionally_consistent(state)
        )

    def _advance_multihop_frontier(
        self,
        *,
        seed_node_id: str,
        seed_scope: set[str],
        state: CandidateSubgraphState,
        frontier_node_id: str,
        current_hops: int,
        max_hops: int,
    ) -> tuple[dict[str, Any], int]:
        next_hops = min(max_hops, current_hops + 1)
        expanded = self._collect_neighborhood(
            seed_node_id=seed_node_id,
            max_hops=next_hops,
            seed_scope=seed_scope,
        )
        distances = expanded.get("distances", {})
        frontier_distance = distances.get(frontier_node_id)
        if frontier_distance is None:
            return expanded, next_hops

        available_edge_pairs = {
            normalize_edge_pair(src_id, tgt_id)
            for src_id, tgt_id, _ in expanded.get("edges", [])
        }
        allowed_nodes = set(state.node_ids)
        for node_id in expanded.get("node_ids", []):
            if node_id in allowed_nodes:
                continue
            if distances.get(node_id) != frontier_distance + 1:
                continue
            if normalize_edge_pair(frontier_node_id, node_id) in available_edge_pairs:
                allowed_nodes.add(node_id)

        filtered_edges = [
            (src_id, tgt_id, edge_data)
            for src_id, tgt_id, edge_data in expanded.get("edges", [])
            if src_id in allowed_nodes and tgt_id in allowed_nodes
        ]
        ranked_nodes = [seed_node_id] + sorted(
            [node_id for node_id in allowed_nodes if node_id != seed_node_id],
            key=lambda item: (distances.get(item, 99), item),
        )
        return (
            {
                "node_ids": ranked_nodes,
                "edges": filtered_edges,
                "distances": distances,
                "max_hops": next_hops,
            },
            next_hops,
        )

    def _breadth_score(
        self,
        seed_node_id: str,
        state: CandidateSubgraphState,
    ) -> int:
        adjacency = {node_id: set() for node_id in state.node_ids}
        for src_id, tgt_id in state.edge_pairs:
            if src_id in adjacency and tgt_id in adjacency:
                adjacency[src_id].add(tgt_id)
                adjacency[tgt_id].add(src_id)
        return len(adjacency.get(seed_node_id, set()))

    def _max_path_distance_from_seed(
        self,
        seed_node_id: str,
        state: CandidateSubgraphState,
    ) -> int:
        adjacency = {node_id: set() for node_id in state.node_ids}
        for src_id, tgt_id in state.edge_pairs:
            if src_id in adjacency and tgt_id in adjacency:
                adjacency[src_id].add(tgt_id)
                adjacency[tgt_id].add(src_id)
        queue = [(seed_node_id, 0)]
        seen = {seed_node_id}
        max_distance = 0
        while queue:
            node_id, distance = queue.pop(0)
            max_distance = max(max_distance, distance)
            for neighbor_id in adjacency.get(node_id, set()):
                if neighbor_id in seen:
                    continue
                seen.add(neighbor_id)
                queue.append((neighbor_id, distance + 1))
        return max_distance

    def _materialize_family_selected_subgraph(
        self,
        *,
        qa_family: str,
        state: CandidateSubgraphState,
        judge_scorecard: JudgeScorecard,
    ) -> SelectedSubgraphArtifact:
        edge_pairs = {
            tuple(sorted((src_id, tgt_id)))
            for src_id, tgt_id in state.edge_pairs
        }
        return SelectedSubgraphArtifact(
            subgraph_id=state.candidate_id,
            technical_focus=state.technical_focus,
            nodes=self._node_payloads(set(state.node_ids)),
            edges=self._edge_payloads(edge_pairs),
            image_grounding_summary=state.image_grounding_summary,
            evidence_summary=state.evidence_summary,
            judge_scores=judge_scorecard,
            approved_question_types=[qa_family],
            degraded=False,
            qa_family=qa_family,
        )

    def _build_v3_result(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        selected_subgraphs: list[dict[str, Any]],
        candidate_bundle: list[dict[str, Any]],
        neighborhood_trace: list[dict[str, Any]],
        edit_trace: list[dict[str, Any]],
        judge_trace: list[dict[str, Any]],
        candidate_states: list[dict[str, Any]],
        agent_sessions: list[dict[str, Any]],
        abstained: bool,
        termination_reason: str,
    ) -> dict[str, Any]:
        return {
            "seed_node_id": seed_node_id,
            "seed_image_path": image_path,
            "selection_mode": "multi" if len(selected_subgraphs) > 1 else "single",
            "degraded": False,
            "degraded_reason": "",
            "selected_subgraphs": selected_subgraphs,
            "candidate_bundle": candidate_bundle,
            "abstained": abstained,
            "max_vqas_per_selected_subgraph": self.max_vqas_per_selected_subgraph,
            "sampler_version": "v3",
            "agent_session": agent_sessions,
            "candidate_states": candidate_states,
            "edit_trace": edit_trace,
            "judge_trace": judge_trace,
            "neighborhood_trace": neighborhood_trace,
            "termination_reason": termination_reason,
        }

    def _family_target_count(self, qa_family: str) -> int:
        if qa_family == "aggregated":
            return max(self.min_subgraphs_per_family, self.max_subgraphs_per_family - 1)
        return self.max_subgraphs_per_family

    @staticmethod
    def _candidate_signature(
        state: CandidateSubgraphState,
    ) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
        node_sig = tuple(sorted(set(state.node_ids)))
        edge_sig = tuple(
            sorted(tuple(sorted((src_id, tgt_id))) for src_id, tgt_id in state.edge_pairs)
        )
        return (node_sig, edge_sig)

    @staticmethod
    def _extract_json_payload(raw: str) -> dict[str, Any]:
        from .artifacts import extract_json_payload

        return extract_json_payload(raw)

    def _judge_feedback_from_payload(self, payload: dict[str, Any]):
        if not isinstance(payload, dict):
            return self._rejected_feedback("invalid_judge_payload")
        scorecard = JudgeScorecard(
            image_indispensability=self._clip(payload.get("image_indispensability")),
            answer_stability=self._clip(payload.get("answer_stability")),
            evidence_closure=self._clip(payload.get("evidence_closure")),
            technical_relevance=self._clip(payload.get("technical_relevance")),
            reasoning_depth=self._clip(payload.get("reasoning_depth")),
            hallucination_risk=self._clip(payload.get("hallucination_risk"), default=1.0),
            theme_coherence=self._clip(payload.get("theme_coherence")),
            overall_score=self._clip(payload.get("overall_score")),
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
        from .v2_artifacts import JudgeFeedback

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

    @staticmethod
    def _clip(value, *, default: float = 0.0) -> float:
        from .artifacts import clip_score

        return clip_score(value, default=default)

    def _family_settings(self, qa_family: str) -> dict[str, int]:
        if qa_family == "atomic":
            return {
                "hard_cap_units": min(self.hard_cap_units, 4),
                "max_rounds": min(self.max_rounds, 3),
                "start_hops": 1,
                "max_hops": 1,
            }
        if qa_family == "aggregated":
            return {
                "hard_cap_units": min(self.hard_cap_units, 8),
                "max_rounds": self.max_rounds,
                "start_hops": 2,
                "max_hops": 2,
            }
        return {
            "hard_cap_units": self.hard_cap_units,
            "max_rounds": self.max_rounds,
            "start_hops": 2,
            "max_hops": self.max_multi_hop_hops,
        }
