"""Candidate generation and family guardrail helpers for visual_core_family_llm sampler."""

import re
from collections import Counter
from typing import Any

from graphgen.models.subgraph_sampler.artifacts import (
    compact_text,
    load_metadata,
    split_source_ids,
)

from .models import BootstrapPlan, FamilyCandidatePoolItem, FamilySessionState


class VisualCoreFamilyCandidateEngineMixin:
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
        analysis_first_hop_ids: list[str],
        seed_scope: set[str],
    ) -> FamilySessionState:
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
        for candidate in first_hop:
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
            forbidden_patterns=list(bootstrap_plan.forbidden_patterns),
            visual_core_node_ids=visual_core_node_ids,
            analysis_first_hop_node_ids=list(analysis_first_hop_ids),
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

        selected_or_reverse_pairs = {
            self._pair_key(candidate.bound_edge_pair),
            self._pair_key(list(reversed(candidate.bound_edge_pair))),
        }
        base_pool = [
            item
            for item in state.candidate_pool
            if item.candidate_uid != candidate.candidate_uid
            and item.candidate_node_id not in set(state.selected_node_ids)
            and item.candidate_uid not in state.blocked_candidate_uids
            and self._pair_key(item.bound_edge_pair) not in selected_or_reverse_pairs
        ]
        base_pool = [
            item
            for item in base_pool
            if self._is_direction_compatible(state, item)
            and self._passes_session_guardrails(state, item)
        ]
        if old_depth > 0 and candidate.depth > old_depth:
            base_pool = [
                item
                for item in base_pool
                if item.depth >= candidate.depth
            ]

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
            depth = self._candidate_depth_from_path(
                new_path,
                visual_core_node_ids=visual_core_node_ids,
            )
            effective_hop = depth
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
            candidate_uid = (
                f"{bind_from_node_id}:{neighbor_id}:{effective_hop}:{bridge_first_hop_id}"
            )
            if candidate_uid in blocked_set or candidate_uid in seen:
                continue
            seen.add(candidate_uid)
            candidates.append(
                FamilyCandidatePoolItem(
                    candidate_uid=candidate_uid,
                    candidate_node_id=neighbor_id,
                    bind_from_node_id=bind_from_node_id,
                    bound_edge_pair=bound_edge_pair,
                    hop=effective_hop,
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
        relation_type_counter = Counter()
        node_type_counter = Counter()
        for item in first_hop + preview:
            node_ids.add(item.candidate_node_id)
            if item.entity_type:
                node_type_counter[item.entity_type] += 1
            if item.relation_type:
                relation_type_counter[item.relation_type] += 1
        seed_node = self.graph.get_node(seed_node_id) or {}
        if seed_node.get("entity_type"):
            node_type_counter[str(seed_node["entity_type"])] += 1
        schema_item_limit = min(max(4, self.bootstrap_preview_limit), 12)
        top_node_type_counts = dict(node_type_counter.most_common(schema_item_limit))
        top_relation_type_counts = dict(
            relation_type_counter.most_common(schema_item_limit)
        )
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
            "node_types": list(top_node_type_counts.keys()),
            "node_type_counts": top_node_type_counts,
            "relation_types": list(top_relation_type_counts.keys()),
            "relation_type_counts": top_relation_type_counts,
            "modalities": modalities,
            "first_hop_count": len(first_hop),
            "preview_candidate_count": len(preview),
            "schema_scope": "seed_image_neighbors_and_logical_first_layer_compact",
            "truncated": (
                len(node_type_counter) > schema_item_limit
                or len(relation_type_counter) > schema_item_limit
            ),
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
