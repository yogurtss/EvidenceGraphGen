"""Visualization trace assembly helpers for visual_core_family_llm sampler."""

import copy
from typing import Any

from graphgen.models.subgraph_sampler.artifacts import (
    compact_text,
    load_metadata,
    to_json_compatible,
)


class VisualCoreFamilyTraceMixin:
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

        bootstrap_by_family: dict[str, list[dict[str, Any]]] = {
            family: [] for family in self.FAMILY_ORDER
        }
        for item in family_bootstrap_trace:
            if isinstance(item, dict):
                bootstrap_by_family.setdefault(str(item.get("qa_family", "")), []).append(
                    item
                )
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
            for bootstrap in bootstrap_by_family.get(qa_family, []):
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
                        "analysis_first_hop_node_ids": list(
                            bootstrap.get("analysis_first_hop_node_ids", [])
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
