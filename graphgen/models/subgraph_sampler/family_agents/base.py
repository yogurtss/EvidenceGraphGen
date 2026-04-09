import json
import re
from abc import ABC, abstractmethod
from collections import deque
from typing import Any

from graphgen.models.subgraph_sampler.artifacts import (
    JudgeScorecard,
    compact_text,
    load_metadata,
    normalize_edge_pair,
    split_source_ids,
)

from .artifacts import FamilySelectedSubgraphArtifact
from .prompts import (
    build_family_subgraph_edit_prompt,
    build_family_subgraph_judge_prompt,
)
from .types import (
    CandidatePoolItem,
    FamilyJudgeFeedback,
    FamilySessionTrace,
    FamilySubgraphState,
)


class BaseFamilyAgent(ABC):
    qa_family = ""

    def __init__(
        self,
        graph,
        llm_client=None,
        *,
        max_selected_subgraphs: int = 3,
        judge_pass_threshold: float = 0.68,
        max_hops: int = 3,
    ):
        self.graph = graph
        self.llm_client = llm_client
        self.max_selected_subgraphs = max(1, int(max_selected_subgraphs))
        self.judge_pass_threshold = float(judge_pass_threshold)
        self.max_hops = max(1, int(max_hops))

    def sample(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        seed_scope: set[str],
    ) -> dict[str, Any]:
        states = self._build_candidate_states(
            seed_node_id=seed_node_id,
            image_path=image_path,
            seed_scope=seed_scope,
        )
        selected_subgraphs = []
        family_candidates = []
        family_edit_trace = []
        family_judge_trace = []
        accepted = 0

        for state in states[: self.max_selected_subgraphs]:
            state = self._llm_edit_state(
                state=state,
                revision_reason="sample_initial_round",
            )
            family_edit_trace.append(
                {
                    "qa_family": self.qa_family,
                    "candidate_id": state.candidate_id,
                    "revision_id": state.revision_id,
                    "actions": self._describe_state_actions(state),
                }
            )
            judge_feedback = self._judge_state(state)
            family_judge_trace.append(
                {
                    "qa_family": self.qa_family,
                    "candidate_id": state.candidate_id,
                    "revision_id": state.revision_id,
                    **judge_feedback.to_dict(),
                }
            )

            decision = "accepted" if judge_feedback.sufficient else "rejected"
            family_candidates.append(
                {
                    "candidate_id": state.candidate_id,
                    "qa_family": self.qa_family,
                    "revision_id": state.revision_id,
                    "intent": state.intent,
                    "technical_focus": state.technical_focus,
                    "node_ids": list(state.selected_node_ids),
                    "edge_pairs": list(state.selected_edge_pairs),
                    "theme_signature": state.theme_signature,
                    "frontier_node_id": state.frontier_node_id,
                    "judge_scores": judge_feedback.scorecard.to_dict(),
                    "decision": decision,
                    "rejection_reason": judge_feedback.rejection_reason,
                    "candidate_pool_snapshot": [
                        item.to_dict() for item in state.candidate_pool
                    ],
                }
            )

            if not judge_feedback.sufficient:
                continue

            selected_subgraphs.append(
                self._materialize_selected_subgraph(state, judge_feedback).to_dict()
            )
            accepted += 1

        session = FamilySessionTrace(
            session_id=f"{seed_node_id}-{self.qa_family}-family",
            qa_family=self.qa_family,
            seed_node_id=seed_node_id,
            attempted_candidates=len(states),
            accepted_candidates=accepted,
            status="accepted" if accepted else "abstained",
            termination_reason=(
                "family_candidates_selected" if accepted else "no_candidate_passed"
            ),
        )
        return {
            "selected_subgraphs": selected_subgraphs,
            "family_candidates": family_candidates,
            "family_edit_trace": family_edit_trace,
            "family_judge_trace": family_judge_trace,
            "family_session": session.to_dict(),
        }

    def continue_subgraph(
        self,
        *,
        selected_subgraph: dict[str, Any],
        revision_reason: str,
        seed_scope: set[str],
    ) -> dict[str, Any] | None:
        state = self._state_from_selected_subgraph(selected_subgraph)
        revised = self._refine_state(
            state=state,
            revision_reason=revision_reason,
            seed_scope=seed_scope,
        )
        if revised is None:
            return None
        judge_feedback = self._judge_state(revised)
        if not judge_feedback.sufficient:
            return None
        return self._materialize_selected_subgraph(revised, judge_feedback).to_dict()

    def _llm_edit_state(
        self,
        *,
        state: FamilySubgraphState,
        revision_reason: str,
    ) -> FamilySubgraphState:
        if not self.llm_client or not state.candidate_pool:
            return state
        prompt = build_family_subgraph_edit_prompt(
            qa_family=self.qa_family,
            state_payload=state.to_dict(),
            candidate_pool_payload=[item.to_dict() for item in state.candidate_pool],
            revision_reason=revision_reason,
        )
        payload = self._call_llm_json(prompt)
        if not payload:
            return state
        decision = str(payload.get("decision", "")).strip().lower()
        if decision != "select_candidate":
            return state
        candidate_node_id = str(payload.get("candidate_node_id", "")).strip()
        if not candidate_node_id:
            return state
        candidate = next(
            (item for item in state.candidate_pool if item.candidate_node_id == candidate_node_id),
            None,
        )
        if candidate is None:
            return state
        self._apply_candidate(state, candidate)
        return state

    def _llm_judge_state(self, state: FamilySubgraphState) -> FamilyJudgeFeedback | None:
        if not self.llm_client:
            return None
        prompt = build_family_subgraph_judge_prompt(
            qa_family=self.qa_family,
            state_payload=state.to_dict(),
        )
        payload = self._call_llm_json(prompt)
        if not payload:
            return None
        scores = payload.get("scores") if isinstance(payload.get("scores"), dict) else {}
        required = {
            "image_indispensability",
            "answer_stability",
            "evidence_closure",
            "technical_relevance",
            "reasoning_depth",
            "hallucination_risk",
            "theme_coherence",
            "overall_score",
        }
        if not required.issubset(set(scores.keys())):
            return None
        try:
            scorecard = JudgeScorecard(
                image_indispensability=float(scores["image_indispensability"]),
                answer_stability=float(scores["answer_stability"]),
                evidence_closure=float(scores["evidence_closure"]),
                technical_relevance=float(scores["technical_relevance"]),
                reasoning_depth=float(scores["reasoning_depth"]),
                hallucination_risk=float(scores["hallucination_risk"]),
                theme_coherence=float(scores["theme_coherence"]),
                overall_score=float(scores["overall_score"]),
                passes=bool(payload.get("sufficient", False)),
            )
        except (TypeError, ValueError):
            return None
        decision = str(payload.get("decision", "")).strip().lower()
        return FamilyJudgeFeedback(
            qa_family=self.qa_family,
            scorecard=scorecard,
            sufficient=bool(payload.get("sufficient", False))
            and scorecard.overall_score >= self.judge_pass_threshold,
            decision="accept" if decision == "accept" else "reject",
            rejection_reason=str(payload.get("rejection_reason", "")).strip(),
            suggested_action=str(payload.get("suggested_action", "")).strip(),
        )

    @abstractmethod
    def _build_candidate_states(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        seed_scope: set[str],
    ) -> list[FamilySubgraphState]:
        raise NotImplementedError

    @abstractmethod
    def _refine_state(
        self,
        *,
        state: FamilySubgraphState,
        revision_reason: str,
        seed_scope: set[str],
    ) -> FamilySubgraphState | None:
        raise NotImplementedError

    @abstractmethod
    def _judge_state(self, state: FamilySubgraphState) -> FamilyJudgeFeedback:
        raise NotImplementedError

    def _state_from_selected_subgraph(
        self, selected_subgraph: dict[str, Any]
    ) -> FamilySubgraphState:
        edge_pairs = []
        for edge in selected_subgraph.get("edges", []):
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                edge_pairs.append([str(edge[0]), str(edge[1])])
        candidate_pool = [
            CandidatePoolItem(**item)
            for item in selected_subgraph.get("candidate_pool_snapshot", [])
            if isinstance(item, dict)
        ]
        seed_node_id = str(selected_subgraph.get("seed_node_id") or "")
        if not seed_node_id:
            nodes = selected_subgraph.get("nodes", [])
            if nodes:
                seed_node_id = str(nodes[0][0])
        return FamilySubgraphState(
            candidate_id=str(selected_subgraph.get("subgraph_id", "")),
            qa_family=self.qa_family,
            seed_node_id=seed_node_id,
            selected_node_ids=[str(node[0]) for node in selected_subgraph.get("nodes", [])],
            selected_edge_pairs=edge_pairs,
            candidate_pool=candidate_pool,
            frontier_node_id=str(selected_subgraph.get("frontier_node_id", "")),
            theme_signature=str(selected_subgraph.get("theme_signature", "")),
            revision_id=int(selected_subgraph.get("revision_id", 0)),
            technical_focus=str(selected_subgraph.get("technical_focus", "")),
            image_grounding_summary=str(
                selected_subgraph.get("image_grounding_summary", "")
            ),
            evidence_summary=str(selected_subgraph.get("evidence_summary", "")),
            intent=str(selected_subgraph.get("technical_focus", "")),
        )

    def _build_direct_candidate_pool(
        self,
        *,
        bind_from_node_ids: list[str],
        seed_scope: set[str],
        selected_node_ids: set[str],
        distances: dict[str, int] | None = None,
        frontier_paths: dict[str, list[str]] | None = None,
    ) -> list[CandidatePoolItem]:
        distances = distances or {}
        frontier_paths = frontier_paths or {node_id: [node_id] for node_id in bind_from_node_ids}
        pool = []
        seen = set()
        for bind_from_node_id in bind_from_node_ids:
            bind_path = list(frontier_paths.get(bind_from_node_id, [bind_from_node_id]))
            for neighbor_id in self.graph.get_neighbors(bind_from_node_id):
                neighbor_id = str(neighbor_id)
                if neighbor_id in selected_node_ids:
                    continue
                edge_data = self.graph.get_edge(bind_from_node_id, neighbor_id) or self.graph.get_edge(
                    neighbor_id, bind_from_node_id
                ) or {}
                node_data = self.graph.get_node(neighbor_id) or {}
                if not self._passes_provenance_guardrail(node_data, edge_data, seed_scope):
                    continue
                edge_pair = list(normalize_edge_pair(bind_from_node_id, neighbor_id))
                signature = (neighbor_id, bind_from_node_id, tuple(edge_pair))
                if signature in seen:
                    continue
                seen.add(signature)
                hop = max(1, int(distances.get(bind_from_node_id, len(bind_path) - 1)) + 1)
                theme_signature = self._build_theme_signature(
                    bind_from_node_id=bind_from_node_id,
                    candidate_node_id=neighbor_id,
                    edge_data=edge_data,
                )
                pool.append(
                    CandidatePoolItem(
                        candidate_node_id=neighbor_id,
                        bind_from_node_id=str(bind_from_node_id),
                        bound_edge_pair=edge_pair,
                        hop=hop,
                        theme_signature=theme_signature,
                        frontier_path=bind_path + [neighbor_id],
                        relation_type=str(edge_data.get("relation_type", "")),
                        entity_type=str(node_data.get("entity_type", "")),
                        score=self._candidate_score(
                            bind_from_node_id=bind_from_node_id,
                            candidate_node_id=neighbor_id,
                            edge_data=edge_data,
                            hop=hop,
                        ),
                    )
                )
        return sorted(
            pool,
            key=lambda item: (-item.score, item.hop, item.candidate_node_id),
        )

    def _build_theme_signature(
        self,
        *,
        bind_from_node_id: str,
        candidate_node_id: str,
        edge_data: dict[str, Any],
    ) -> str:
        bind_data = self.graph.get_node(bind_from_node_id) or {}
        candidate_data = self.graph.get_node(candidate_node_id) or {}
        tokens = sorted(
            self._keywords_from_node(bind_data)
            & self._keywords_from_node(candidate_data)
        )[:3]
        return "|".join(
            [
                str(edge_data.get("relation_type", "")).strip().lower(),
                str(candidate_data.get("entity_type", "")).strip().lower(),
                ",".join(tokens),
            ]
        )

    def _candidate_score(
        self,
        *,
        bind_from_node_id: str,
        candidate_node_id: str,
        edge_data: dict[str, Any],
        hop: int,
    ) -> float:
        bind_data = self.graph.get_node(bind_from_node_id) or {}
        candidate_data = self.graph.get_node(candidate_node_id) or {}
        shared_keywords = len(
            self._keywords_from_node(bind_data) & self._keywords_from_node(candidate_data)
        )
        relation_bonus = 0.12 if edge_data.get("relation_type") else 0.0
        evidence_bonus = 0.08 if edge_data.get("evidence_span") else 0.0
        return round(shared_keywords * 0.12 + relation_bonus + evidence_bonus - hop * 0.02, 4)

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

    def _collect_seed_scope(self, seed_node_id: str) -> set[str]:
        seed_scope = set()
        for node_id, node_data in self.graph.get_all_nodes() or []:
            if str(node_id) != str(seed_node_id):
                continue
            metadata = load_metadata(node_data.get("metadata"))
            seed_scope.update(split_source_ids(node_data.get("source_id", "")))
            seed_scope.update(split_source_ids(metadata.get("source_trace_id", "")))
            break
        return seed_scope

    def _build_distance_map(self, seed_node_id: str, max_hops: int) -> dict[str, int]:
        queue = deque([(str(seed_node_id), 0)])
        distances = {str(seed_node_id): 0}
        while queue:
            node_id, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbor_id in self.graph.get_neighbors(node_id):
                neighbor_id = str(neighbor_id)
                if neighbor_id in distances:
                    continue
                distances[neighbor_id] = depth + 1
                queue.append((neighbor_id, depth + 1))
        return distances

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

    def _apply_candidate(
        self,
        state: FamilySubgraphState,
        candidate: CandidatePoolItem,
    ) -> None:
        if candidate.candidate_node_id not in state.selected_node_ids:
            state.selected_node_ids.append(candidate.candidate_node_id)
        normalized_pair = list(normalize_edge_pair(*candidate.bound_edge_pair))
        if normalized_pair not in state.selected_edge_pairs:
            state.selected_edge_pairs.append(normalized_pair)
        state.frontier_node_id = candidate.candidate_node_id
        state.theme_signature = state.theme_signature or candidate.theme_signature
        state.candidate_pool = [
            item
            for item in state.candidate_pool
            if item.candidate_node_id != candidate.candidate_node_id
        ]

    def _describe_state_actions(self, state: FamilySubgraphState) -> list[dict[str, Any]]:
        actions = []
        if len(state.selected_node_ids) > 1:
            for node_id in state.selected_node_ids[1:]:
                actions.append(
                    {
                        "action_type": "add_node",
                        "node_id": node_id,
                    }
                )
        if state.candidate_pool:
            actions.append(
                {
                    "action_type": "prune_candidate_pool",
                    "remaining_candidate_ids": [
                        item.candidate_node_id for item in state.candidate_pool
                    ],
                }
            )
        return actions

    def _materialize_selected_subgraph(
        self,
        state: FamilySubgraphState,
        judge_feedback: FamilyJudgeFeedback,
    ) -> FamilySelectedSubgraphArtifact:
        edge_pairs = {
            tuple(normalize_edge_pair(src_id, tgt_id))
            for src_id, tgt_id in state.selected_edge_pairs
        }
        return FamilySelectedSubgraphArtifact(
            subgraph_id=state.candidate_id,
            qa_family=self.qa_family,
            technical_focus=state.technical_focus or self.qa_family,
            nodes=self._node_payloads(set(state.selected_node_ids)),
            edges=self._edge_payloads(edge_pairs),
            image_grounding_summary=compact_text(state.image_grounding_summary, limit=240),
            evidence_summary=compact_text(state.evidence_summary, limit=240),
            judge_scores=judge_feedback.scorecard,
            approved_question_types=[self.qa_family],
            candidate_pool_snapshot=[item.to_dict() for item in state.candidate_pool],
            frontier_node_id=state.frontier_node_id,
            theme_signature=state.theme_signature,
            revision_id=state.revision_id,
        )

    def _safe_json_dumps(self, payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False)

    def _call_llm_json(self, prompt: str) -> dict[str, Any]:
        if not self.llm_client:
            return {}
        raw = self.llm_client.generate_answer(prompt)
        if hasattr(raw, "__await__"):
            raw = __import__("asyncio").run(raw)
        if not isinstance(raw, str):
            return {}
        try:
            payload = json.loads(raw)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}
