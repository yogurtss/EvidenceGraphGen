from typing import Any

from graphgen.models.subgraph_sampler.artifacts import load_metadata

from .aggregated_agent import AggregatedFamilyAgent
from .atomic_agent import AtomicFamilyAgent
from .multihop_agent import MultiHopFamilyAgent


class FamilySubgraphOrchestrator:
    FAMILY_ORDER = ("atomic", "aggregated", "multi_hop")

    def __init__(
        self,
        graph,
        llm_client=None,
        *,
        max_selected_subgraphs_per_family: int = 3,
        judge_pass_threshold: float = 0.68,
        max_multi_hop_hops: int = 3,
    ):
        self.graph = graph
        self.llm_client = llm_client
        self.agents = {
            "atomic": AtomicFamilyAgent(
                graph,
                llm_client,
                max_selected_subgraphs=max_selected_subgraphs_per_family,
                judge_pass_threshold=judge_pass_threshold,
                max_hops=1,
            ),
            "aggregated": AggregatedFamilyAgent(
                graph,
                llm_client,
                max_selected_subgraphs=max_selected_subgraphs_per_family,
                judge_pass_threshold=judge_pass_threshold,
                max_hops=2,
            ),
            "multi_hop": MultiHopFamilyAgent(
                graph,
                llm_client,
                max_selected_subgraphs=max_selected_subgraphs_per_family,
                judge_pass_threshold=judge_pass_threshold,
                max_hops=max_multi_hop_hops,
            ),
        }

    def sample(self, *, seed_node_id: str) -> dict[str, Any]:
        seed_node = self.graph.get_node(seed_node_id) or {}
        image_path = self._extract_image_path(seed_node)
        if not image_path:
            return {
                "seed_node_id": seed_node_id,
                "seed_image_path": "",
                "selection_mode": "single",
                "degraded": False,
                "degraded_reason": "",
                "selected_subgraphs": [],
                "candidate_bundle": [],
                "family_sessions": [],
                "family_candidates": [],
                "family_edit_trace": [],
                "family_judge_trace": [],
                "abstained": True,
                "sampler_version": "family_agents_v1",
                "termination_reason": "missing_image_seed_or_asset",
            }

        seed_scope = self._collect_seed_scope(seed_node_id)
        selected_subgraphs = []
        family_candidates = []
        family_edit_trace = []
        family_judge_trace = []
        family_sessions = []

        for qa_family in self.FAMILY_ORDER:
            result = self.agents[qa_family].sample(
                seed_node_id=seed_node_id,
                image_path=image_path,
                seed_scope=seed_scope,
            )
            family_sessions.append(result["family_session"])
            family_candidates.extend(result["family_candidates"])
            family_edit_trace.extend(result["family_edit_trace"])
            family_judge_trace.extend(result["family_judge_trace"])
            for selected in result["selected_subgraphs"]:
                selected["seed_node_id"] = seed_node_id
                selected_subgraphs.append(selected)

        selected_subgraphs.sort(
            key=lambda item: (
                item.get("qa_family", ""),
                -float(item.get("judge_scores", {}).get("overall_score", 0.0)),
                item.get("subgraph_id", ""),
            )
        )
        return {
            "seed_node_id": seed_node_id,
            "seed_image_path": image_path,
            "selection_mode": "multi" if len(selected_subgraphs) > 1 else "single",
            "degraded": False,
            "degraded_reason": "",
            "selected_subgraphs": selected_subgraphs,
            "candidate_bundle": family_candidates,
            "family_sessions": family_sessions,
            "family_candidates": family_candidates,
            "family_edit_trace": family_edit_trace,
            "family_judge_trace": family_judge_trace,
            "abstained": not bool(selected_subgraphs),
            "sampler_version": "family_agents_v1",
            "termination_reason": (
                "family_candidates_selected"
                if selected_subgraphs
                else "no_family_candidate_passed"
            ),
        }

    def continue_subgraph(
        self,
        *,
        selected_subgraph: dict[str, Any],
        revision_reason: str,
    ) -> dict[str, Any] | None:
        qa_family = str(selected_subgraph.get("qa_family", "")).strip().lower()
        if qa_family not in self.agents:
            return None
        seed_node_id = str(selected_subgraph.get("seed_node_id", ""))
        if not seed_node_id:
            return None
        seed_scope = self._collect_seed_scope(seed_node_id)
        revised = self.agents[qa_family].continue_subgraph(
            selected_subgraph=selected_subgraph,
            revision_reason=revision_reason,
            seed_scope=seed_scope,
        )
        if revised:
            revised["seed_node_id"] = seed_node_id
        return revised

    def _collect_seed_scope(self, seed_node_id: str) -> set[str]:
        return self.agents["atomic"]._collect_seed_scope(seed_node_id)

    @staticmethod
    def _extract_image_path(node_data: dict[str, Any]) -> str:
        metadata = load_metadata(node_data.get("metadata"))
        for key in ("image_path", "img_path"):
            if metadata.get(key):
                return str(metadata[key])
        return ""
