from graphgen.models.subgraph_sampler.artifacts import JudgeScorecard

from .base import BaseFamilyAgent
from .types import FamilyJudgeFeedback, FamilySubgraphState


class AtomicFamilyAgent(BaseFamilyAgent):
    qa_family = "atomic"

    def _build_candidate_states(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        seed_scope: set[str],
    ) -> list[FamilySubgraphState]:
        distances = self._build_distance_map(seed_node_id, 1)
        direct_pool = self._build_direct_candidate_pool(
            bind_from_node_ids=[seed_node_id],
            seed_scope=seed_scope,
            selected_node_ids={seed_node_id},
            distances=distances,
            frontier_paths={seed_node_id: [seed_node_id]},
        )
        states = []
        for index, candidate in enumerate(direct_pool[: self.max_selected_subgraphs], start=1):
            remaining_pool = [
                item
                for item in direct_pool
                if item.candidate_node_id != candidate.candidate_node_id
            ]
            state = FamilySubgraphState(
                candidate_id=f"{self.qa_family}-candidate-{index}",
                qa_family=self.qa_family,
                seed_node_id=seed_node_id,
                selected_node_ids=[seed_node_id],
                candidate_pool=remaining_pool,
                intent="Answer one direct image-grounded fact.",
                technical_focus="atomic_fact",
                image_grounding_summary=(
                    f"The image grounds the direct supporting relation to {candidate.candidate_node_id}."
                ),
                evidence_summary=(
                    f"A single bound edge is sufficient to support {candidate.candidate_node_id}."
                ),
            )
            self._apply_candidate(state, candidate)
            states.append(state)
        return states

    def _refine_state(
        self,
        *,
        state: FamilySubgraphState,
        revision_reason: str,
        seed_scope: set[str],
    ) -> FamilySubgraphState | None:
        _ = revision_reason
        if not state.candidate_pool:
            return None
        replacement = state.candidate_pool[0]
        revised = FamilySubgraphState(
            candidate_id=state.candidate_id,
            qa_family=self.qa_family,
            seed_node_id=state.seed_node_id,
            selected_node_ids=[state.seed_node_id],
            candidate_pool=[item for item in state.candidate_pool if item.candidate_node_id != replacement.candidate_node_id],
            frontier_node_id="",
            theme_signature=replacement.theme_signature,
            revision_id=state.revision_id + 1,
            intent=state.intent,
            technical_focus=state.technical_focus,
            image_grounding_summary=state.image_grounding_summary,
            evidence_summary=state.evidence_summary,
        )
        self._apply_candidate(revised, replacement)
        revised.image_grounding_summary = (
            f"{state.image_grounding_summary} Revised to use {replacement.candidate_node_id}."
        )
        revised.evidence_summary = (
            f"{state.evidence_summary} Revision keeps the subgraph minimal."
        )
        return revised

    def _judge_state(self, state: FamilySubgraphState) -> FamilyJudgeFeedback:
        llm_feedback = self._llm_judge_state(state)
        if llm_feedback is not None:
            return llm_feedback
        node_count = len(state.selected_node_ids)
        edge_count = len(state.selected_edge_pairs)
        sufficient = node_count == 2 and edge_count == 1
        scorecard = JudgeScorecard(
            image_indispensability=0.88 if sufficient else 0.55,
            answer_stability=0.82 if sufficient else 0.52,
            evidence_closure=0.8 if sufficient else 0.5,
            technical_relevance=0.78 if sufficient else 0.55,
            reasoning_depth=0.18,
            hallucination_risk=0.1 if sufficient else 0.35,
            theme_coherence=0.84 if sufficient else 0.56,
            overall_score=0.82 if sufficient else 0.48,
            passes=sufficient,
        )
        return FamilyJudgeFeedback(
            qa_family=self.qa_family,
            scorecard=scorecard,
            sufficient=sufficient and scorecard.overall_score >= self.judge_pass_threshold,
            decision="accept" if sufficient else "reject",
            rejection_reason="" if sufficient else "atomic_requires_minimal_single_support",
            suggested_action="" if sufficient else "replace_support_node",
        )
