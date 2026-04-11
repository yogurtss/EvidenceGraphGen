"""Subgraph materialization helpers for visual_core_family_llm sampler."""

import copy
from typing import Any

from graphgen.models.subgraph_sampler.artifacts import (
    JudgeScorecard,
    compact_text,
    load_metadata,
    to_json_compatible,
)

from .models import BootstrapPlan, FamilyCandidatePoolItem, FamilySessionState


class VisualCoreFamilyMaterializerMixin:
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
