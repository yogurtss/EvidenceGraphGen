from graphgen.models.subgraph_sampler.artifacts import JudgeScorecard

from .base import BaseFamilyAgent
from .types import CandidatePoolItem, FamilyJudgeFeedback, FamilySubgraphState


class AggregatedFamilyAgent(BaseFamilyAgent):
    qa_family = "aggregated"

    def _build_candidate_states(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        seed_scope: set[str],
    ) -> list[FamilySubgraphState]:
        _ = image_path
        distances = self._build_distance_map(seed_node_id, 2)
        seed_pool = self._build_direct_candidate_pool(
            bind_from_node_ids=[seed_node_id],
            seed_scope=seed_scope,
            selected_node_ids={seed_node_id},
            distances=distances,
            frontier_paths={seed_node_id: [seed_node_id]},
        )
        states = []
        for index, anchor in enumerate(seed_pool[: self.max_selected_subgraphs], start=1):
            state = FamilySubgraphState(
                candidate_id=f"{self.qa_family}-candidate-{index}",
                qa_family=self.qa_family,
                seed_node_id=seed_node_id,
                selected_node_ids=[seed_node_id],
                candidate_pool=[],
                frontier_node_id=seed_node_id,
                theme_signature=anchor.theme_signature,
                revision_id=0,
                intent="Aggregate same-theme evidence around one topic.",
                technical_focus="theme_aggregation",
                image_grounding_summary="The image anchors a coherent local topic.",
                evidence_summary="The subgraph integrates same-theme evidence with breadth-first selection.",
            )
            self._apply_candidate(state, anchor)
            breadth_pool = self._build_breadth_pool(
                seed_node_id=seed_node_id,
                anchor=anchor,
                seed_scope=seed_scope,
                selected_node_ids=set(state.selected_node_ids),
                distances=distances,
            )
            additions = 0
            remaining_pool = []
            for candidate in breadth_pool:
                if self._is_theme_compatible(state.theme_signature, candidate.theme_signature) and additions < 2:
                    self._apply_candidate(state, candidate)
                    additions += 1
                else:
                    remaining_pool.append(candidate)
            state.candidate_pool = remaining_pool
            state.frontier_node_id = anchor.candidate_node_id
            state.image_grounding_summary = (
                f"The image anchors {anchor.candidate_node_id} and its same-theme neighbors."
            )
            state.evidence_summary = (
                f"Width-first aggregation keeps the topic around {anchor.candidate_node_id} coherent."
            )
            states.append(state)
        return states

    def _build_breadth_pool(
        self,
        *,
        seed_node_id: str,
        anchor: CandidatePoolItem,
        seed_scope: set[str],
        selected_node_ids: set[str],
        distances: dict[str, int],
    ) -> list[CandidatePoolItem]:
        bind_nodes = [seed_node_id, anchor.candidate_node_id]
        frontier_paths = {
            seed_node_id: [seed_node_id],
            anchor.candidate_node_id: list(anchor.frontier_path),
        }
        return self._build_direct_candidate_pool(
            bind_from_node_ids=bind_nodes,
            seed_scope=seed_scope,
            selected_node_ids=selected_node_ids,
            distances=distances,
            frontier_paths=frontier_paths,
        )

    def _is_theme_compatible(self, anchor_signature: str, candidate_signature: str) -> bool:
        anchor_parts = {part for part in anchor_signature.split("|") if part}
        candidate_parts = {part for part in candidate_signature.split("|") if part}
        return bool(anchor_parts & candidate_parts)

    def _refine_state(
        self,
        *,
        state: FamilySubgraphState,
        revision_reason: str,
        seed_scope: set[str],
    ) -> FamilySubgraphState | None:
        _ = revision_reason
        remaining = [
            item
            for item in state.candidate_pool
            if self._is_theme_compatible(state.theme_signature, item.theme_signature)
        ]
        if not remaining:
            return None
        revised = FamilySubgraphState(
            candidate_id=state.candidate_id,
            qa_family=self.qa_family,
            seed_node_id=state.seed_node_id,
            selected_node_ids=list(state.selected_node_ids),
            selected_edge_pairs=list(state.selected_edge_pairs),
            candidate_pool=[
                item for item in state.candidate_pool if item.candidate_node_id != remaining[0].candidate_node_id
            ],
            frontier_node_id=state.frontier_node_id,
            theme_signature=state.theme_signature,
            revision_id=state.revision_id + 1,
            intent=state.intent,
            technical_focus=state.technical_focus,
            image_grounding_summary=state.image_grounding_summary,
            evidence_summary=state.evidence_summary,
        )
        self._apply_candidate(revised, remaining[0])
        revised.image_grounding_summary = (
            f"{state.image_grounding_summary} Revision broadens the same-theme neighborhood."
        )
        revised.evidence_summary = (
            f"{state.evidence_summary} Revision adds width without changing the topic."
        )
        return revised

    def _judge_state(self, state: FamilySubgraphState) -> FamilyJudgeFeedback:
        node_count = len(state.selected_node_ids)
        edge_count = len(state.selected_edge_pairs)
        sufficient = node_count >= 3 and edge_count >= 2
        breadth_bonus = min(0.18, max(0, node_count - 2) * 0.06)
        scorecard = JudgeScorecard(
            image_indispensability=0.83 if sufficient else 0.58,
            answer_stability=0.8 if sufficient else 0.55,
            evidence_closure=0.76 + breadth_bonus if sufficient else 0.5,
            technical_relevance=0.84 if sufficient else 0.6,
            reasoning_depth=0.36,
            hallucination_risk=0.12 if sufficient else 0.3,
            theme_coherence=0.85 if sufficient else 0.5,
            overall_score=0.8 + breadth_bonus if sufficient else 0.5,
            passes=sufficient,
        )
        return FamilyJudgeFeedback(
            qa_family=self.qa_family,
            scorecard=scorecard,
            sufficient=sufficient and scorecard.overall_score >= self.judge_pass_threshold,
            decision="accept" if sufficient else "reject",
            rejection_reason="" if sufficient else "aggregated_requires_same_theme_breadth",
            suggested_action="" if sufficient else "add_same_theme_sibling",
        )
