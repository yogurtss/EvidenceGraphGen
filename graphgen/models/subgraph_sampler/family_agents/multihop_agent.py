from graphgen.models.subgraph_sampler.artifacts import JudgeScorecard

from .base import BaseFamilyAgent
from .types import FamilyJudgeFeedback, FamilySubgraphState


class MultiHopFamilyAgent(BaseFamilyAgent):
    qa_family = "multi_hop"

    def _build_candidate_states(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        seed_scope: set[str],
    ) -> list[FamilySubgraphState]:
        _ = image_path
        distances = self._build_distance_map(seed_node_id, self.max_hops)
        first_hop_pool = self._build_direct_candidate_pool(
            bind_from_node_ids=[seed_node_id],
            seed_scope=seed_scope,
            selected_node_ids={seed_node_id},
            distances=distances,
            frontier_paths={seed_node_id: [seed_node_id]},
        )
        states = []
        for index, first_hop in enumerate(first_hop_pool[: self.max_selected_subgraphs], start=1):
            state = FamilySubgraphState(
                candidate_id=f"{self.qa_family}-candidate-{index}",
                qa_family=self.qa_family,
                seed_node_id=seed_node_id,
                selected_node_ids=[seed_node_id],
                selected_edge_pairs=[],
                candidate_pool=[],
                frontier_node_id=seed_node_id,
                theme_signature=first_hop.theme_signature,
                revision_id=0,
                intent="Follow one explicit reasoning chain.",
                technical_focus="reasoning_chain",
                image_grounding_summary="The image anchors the first reasoning step.",
                evidence_summary="The chain advances one frontier at a time.",
            )
            self._apply_candidate(state, first_hop)
            frontier = first_hop
            siblings_pruned = [item.candidate_node_id for item in first_hop_pool if item.candidate_node_id != first_hop.candidate_node_id]

            depth = 1
            while depth < self.max_hops:
                next_pool = self._build_direct_candidate_pool(
                    bind_from_node_ids=[frontier.candidate_node_id],
                    seed_scope=seed_scope,
                    selected_node_ids=set(state.selected_node_ids),
                    distances=distances,
                    frontier_paths={frontier.candidate_node_id: list(frontier.frontier_path)},
                )
                if not next_pool:
                    state.candidate_pool = []
                    break
                next_candidate = self._choose_next_candidate(
                    next_pool=next_pool,
                    seed_scope=seed_scope,
                    selected_node_ids=set(state.selected_node_ids),
                    distances=distances,
                )
                self._apply_candidate(state, next_candidate)
                frontier = next_candidate
                depth += 1
                if depth >= 2:
                    state.candidate_pool = self._build_direct_candidate_pool(
                        bind_from_node_ids=[frontier.candidate_node_id],
                        seed_scope=seed_scope,
                        selected_node_ids=set(state.selected_node_ids),
                        distances=distances,
                        frontier_paths={
                            frontier.candidate_node_id: list(frontier.frontier_path)
                        },
                    )
                    state.image_grounding_summary = (
                        f"The image grounds the first step, and the chain extends through {frontier.candidate_node_id}."
                    )
                    state.evidence_summary = (
                        f"Depth-first expansion pruned same-layer siblings: {siblings_pruned}."
                    )
                    break
            state.frontier_node_id = frontier.candidate_node_id
            states.append(state)
        return states

    def _choose_next_candidate(
        self,
        *,
        next_pool,
        seed_scope: set[str],
        selected_node_ids: set[str],
        distances: dict[str, int],
    ):
        ranked = []
        for candidate in next_pool:
            forward_pool = self._build_direct_candidate_pool(
                bind_from_node_ids=[candidate.candidate_node_id],
                seed_scope=seed_scope,
                selected_node_ids=selected_node_ids | {candidate.candidate_node_id},
                distances=distances,
                frontier_paths={
                    candidate.candidate_node_id: list(candidate.frontier_path)
                },
            )
            ranked.append((len(forward_pool), candidate.score, candidate))
        ranked.sort(
            key=lambda item: (-item[0], -item[1], item[2].candidate_node_id)
        )
        return ranked[0][2]

    def _refine_state(
        self,
        *,
        state: FamilySubgraphState,
        revision_reason: str,
        seed_scope: set[str],
    ) -> FamilySubgraphState | None:
        _ = revision_reason
        frontier_node_id = state.frontier_node_id or state.selected_node_ids[-1]
        distances = self._build_distance_map(state.seed_node_id, self.max_hops + state.revision_id + 1)
        next_pool = self._build_direct_candidate_pool(
            bind_from_node_ids=[frontier_node_id],
            seed_scope=seed_scope,
            selected_node_ids=set(state.selected_node_ids),
            distances=distances,
            frontier_paths={frontier_node_id: [state.seed_node_id, frontier_node_id]},
        )
        if not next_pool:
            return None
        next_candidate = next_pool[0]
        revised = FamilySubgraphState(
            candidate_id=state.candidate_id,
            qa_family=self.qa_family,
            seed_node_id=state.seed_node_id,
            selected_node_ids=list(state.selected_node_ids),
            selected_edge_pairs=list(state.selected_edge_pairs),
            candidate_pool=[],
            frontier_node_id=state.frontier_node_id,
            theme_signature=state.theme_signature,
            revision_id=state.revision_id + 1,
            intent=state.intent,
            technical_focus=state.technical_focus,
            image_grounding_summary=state.image_grounding_summary,
            evidence_summary=state.evidence_summary,
        )
        self._apply_candidate(revised, next_candidate)
        revised.candidate_pool = []
        revised.image_grounding_summary = (
            f"{state.image_grounding_summary} Revision pushes the frontier to {next_candidate.candidate_node_id}."
        )
        revised.evidence_summary = (
            f"{state.evidence_summary} Revision prunes same-layer siblings and keeps one chain."
        )
        return revised

    def _judge_state(self, state: FamilySubgraphState) -> FamilyJudgeFeedback:
        edge_count = len(state.selected_edge_pairs)
        max_depth = max(0, len(state.selected_node_ids) - 1)
        sufficient = edge_count >= 2 and max_depth >= 2
        depth_bonus = min(0.12, max(0, edge_count - 2) * 0.04)
        scorecard = JudgeScorecard(
            image_indispensability=0.84 if sufficient else 0.58,
            answer_stability=0.78 if sufficient else 0.56,
            evidence_closure=0.78 + depth_bonus if sufficient else 0.52,
            technical_relevance=0.86 if sufficient else 0.62,
            reasoning_depth=0.82 if sufficient else 0.42,
            hallucination_risk=0.14 if sufficient else 0.28,
            theme_coherence=0.8 if sufficient else 0.52,
            overall_score=0.82 + depth_bonus if sufficient else 0.54,
            passes=sufficient,
        )
        return FamilyJudgeFeedback(
            qa_family=self.qa_family,
            scorecard=scorecard,
            sufficient=sufficient and scorecard.overall_score >= self.judge_pass_threshold,
            decision="accept" if sufficient else "reject",
            rejection_reason="" if sufficient else "multi_hop_requires_deep_chain",
            suggested_action="" if sufficient else "advance_frontier",
        )
