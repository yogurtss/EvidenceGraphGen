from collections import deque
from typing import Any

from graphgen.bases import BaseGraphStorage, BaseLLMWrapper
from .artifacts import (
    JudgeScorecard,
    SelectedSubgraphArtifact,
    SubgraphCandidate,
    clip_score,
    compact_text,
    extract_json_payload,
    load_metadata,
    normalize_edge_pair,
    split_source_ids,
)
from .constants import (
    ALLOWED_DEGRADED_QUESTION_TYPES,
    ALLOWED_PRIMARY_QUESTION_TYPES,
    TECHNICAL_KEYWORDS,
)
from .prompts import (
    build_assembler_prompt,
    build_candidate_prompt,
    build_judge_prompt,
    build_neighborhood_prompt,
    build_planner_prompt,
)


class VLMSubgraphSampler:
    def __init__(
        self,
        graph: BaseGraphStorage,
        llm_client: BaseLLMWrapper,
        *,
        max_units: int = 10,
        max_hops_from_seed: int = 4,
        candidate_pool_size: int = 3,
        max_selected_subgraphs: int = 1,
        max_vqas_per_selected_subgraph: int = 2,
        allow_degraded: bool = True,
        judge_pass_threshold: float = 0.68,
        theme_split_threshold: float = 0.18,
    ):
        self.graph = graph
        self.llm_client = llm_client
        self.max_units = max(3, int(max_units))
        self.max_hops_from_seed = max(1, int(max_hops_from_seed))
        self.candidate_pool_size = max(1, int(candidate_pool_size))
        self.max_selected_subgraphs = max(1, int(max_selected_subgraphs))
        self.max_vqas_per_selected_subgraph = min(
            3, max(1, int(max_vqas_per_selected_subgraph))
        )
        self.allow_degraded = bool(allow_degraded)
        self.judge_pass_threshold = float(judge_pass_threshold)
        self.theme_split_threshold = float(theme_split_threshold)

    async def sample(
        self,
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]],
        *,
        seed_node_id: str,
    ) -> dict:
        nodes, edges = batch
        seed_node = self.graph.get_node(seed_node_id) or {}
        image_path = self._extract_image_path(seed_node)
        if not seed_node_id or not seed_node or not image_path:
            return self._build_empty_result(
                seed_node_id=seed_node_id,
                image_path=image_path,
                candidate_bundle=[],
                degraded=False,
                degraded_reason="missing_image_seed_or_asset",
            )

        seed_chunk_ids, seed_source_ids = self._collect_seed_scope(seed_node_id, nodes)
        neighborhood = self._collect_neighborhood(
            seed_node_id=seed_node_id,
            seed_source_ids=seed_source_ids,
            seed_chunk_ids=seed_chunk_ids,
        )
        if len(neighborhood["node_ids"]) <= 1:
            return self._build_empty_result(
                seed_node_id=seed_node_id,
                image_path=image_path,
                candidate_bundle=[],
                degraded=False,
                degraded_reason="insufficient_neighborhood",
            )

        primary_candidates = await self._build_candidates(
            seed_node_id=seed_node_id,
            seed_node=seed_node,
            image_path=image_path,
            neighborhood=neighborhood,
            degraded=False,
        )
        accepted_primary = [c for c in primary_candidates if c.decision == "accepted"]

        degraded_reason = ""
        final_candidates = primary_candidates
        degraded = False
        if not accepted_primary and self.allow_degraded:
            degraded_candidates = await self._build_candidates(
                seed_node_id=seed_node_id,
                seed_node=seed_node,
                image_path=image_path,
                neighborhood=neighborhood,
                degraded=True,
            )
            final_candidates = primary_candidates + degraded_candidates
            accepted_primary = [
                c for c in degraded_candidates if c.decision == "accepted"
            ]
            degraded = bool(accepted_primary)
            degraded_reason = "fallback_to_conservative_chart_interpretation"

        if not accepted_primary:
            return self._build_empty_result(
                seed_node_id=seed_node_id,
                image_path=image_path,
                candidate_bundle=[c.compact_bundle() for c in final_candidates],
                degraded=False,
                degraded_reason=degraded_reason or "no_candidate_passed_judge",
            )

        selected = self._select_candidates(accepted_primary)
        selection_mode = "multi" if len(selected) > 1 else "single"
        selected_subgraphs = [
            self._materialize_selected_subgraph(candidate).to_dict()
            for candidate in selected
        ]
        return {
            "seed_node_id": seed_node_id,
            "seed_image_path": image_path,
            "selection_mode": selection_mode,
            "degraded": degraded,
            "degraded_reason": degraded_reason if degraded else "",
            "selected_subgraphs": selected_subgraphs,
            "candidate_bundle": [c.compact_bundle() for c in final_candidates],
            "abstained": False,
            "max_vqas_per_selected_subgraph": self.max_vqas_per_selected_subgraph,
        }

    async def _build_candidates(
        self,
        *,
        seed_node_id: str,
        seed_node: dict,
        image_path: str,
        neighborhood: dict[str, Any],
        degraded: bool,
    ) -> list[SubgraphCandidate]:
        intents = await self._propose_intents(
            seed_node_id=seed_node_id,
            seed_node=seed_node,
            image_path=image_path,
            neighborhood=neighborhood,
            degraded=degraded,
        )
        candidates: list[SubgraphCandidate] = []
        for index, intent in enumerate(intents[: self.candidate_pool_size], start=1):
            candidate = await self._assemble_candidate(
                seed_node_id=seed_node_id,
                image_path=image_path,
                neighborhood=neighborhood,
                intent=intent,
                degraded=degraded,
                candidate_index=index,
            )
            if candidate is None:
                candidates.append(
                    self._build_rejected_candidate(
                        candidate_index=index,
                        intent=intent,
                        degraded=degraded,
                        reason="invalid_candidate_payload",
                    )
                )
                continue
            scorecard, rejection_reason = await self._judge_candidate(
                seed_node_id=seed_node_id,
                image_path=image_path,
                neighborhood=neighborhood,
                candidate=candidate,
                degraded=degraded,
            )
            candidate.judge_scores = scorecard
            candidate.decision = "accepted" if scorecard.passes else "rejected"
            candidate.rejection_reason = rejection_reason if not scorecard.passes else ""
            candidates.append(candidate)
        candidates.sort(
            key=lambda item: item.judge_scores.overall_score,
            reverse=True,
        )
        return candidates

    async def _propose_intents(
        self,
        *,
        seed_node_id: str,
        seed_node: dict,
        image_path: str,
        neighborhood: dict[str, Any],
        degraded: bool,
    ) -> list[dict[str, Any]]:
        allowed_question_types = (
            ALLOWED_DEGRADED_QUESTION_TYPES if degraded else ALLOWED_PRIMARY_QUESTION_TYPES
        )
        prompt = build_planner_prompt(
            seed_node_id=seed_node_id,
            seed_description=seed_node.get("description", ""),
            image_path=image_path,
            allowed_question_types=allowed_question_types,
            degraded=degraded,
            neighborhood_prompt=build_neighborhood_prompt(self.graph, neighborhood),
            candidate_pool_size=self.candidate_pool_size,
        )
        raw = await self.llm_client.generate_answer(prompt, image_path=image_path or None)
        payload = extract_json_payload(raw)
        intents = payload.get("intents") if isinstance(payload, dict) else None
        normalized: list[dict[str, Any]] = []
        if isinstance(intents, list):
            for item in intents:
                if not isinstance(item, dict):
                    continue
                intent_text = compact_text(item.get("intent", "technical interpretation"))
                technical_focus = compact_text(
                    item.get("technical_focus", intent_text), limit=80
                )
                selected_question_types = [
                    q
                    for q in item.get("question_types", [])
                    if q in allowed_question_types
                ]
                keywords = [
                    compact_text(keyword, limit=40)
                    for keyword in item.get("priority_keywords", [])
                    if str(keyword).strip()
                ]
                if intent_text:
                    normalized.append(
                        {
                            "intent": intent_text,
                            "technical_focus": technical_focus or intent_text,
                            "question_types": selected_question_types
                            or allowed_question_types[:1],
                            "priority_keywords": keywords,
                        }
                    )
        return normalized

    async def _assemble_candidate(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        neighborhood: dict[str, Any],
        intent: dict[str, Any],
        degraded: bool,
        candidate_index: int,
    ) -> SubgraphCandidate | None:
        prompt = build_assembler_prompt(
            seed_node_id=seed_node_id,
            intent=intent,
            degraded=degraded,
            max_units=self.max_units,
            neighborhood_prompt=build_neighborhood_prompt(self.graph, neighborhood),
        )
        raw = await self.llm_client.generate_answer(prompt, image_path=image_path or None)
        payload = extract_json_payload(raw)
        if not isinstance(payload, dict):
            return None

        node_ids = self._sanitize_node_ids(
            payload.get("node_ids", []),
            neighborhood=neighborhood,
            seed_node_id=seed_node_id,
        )
        edge_pairs = self._sanitize_edge_pairs(
            payload.get("edge_pairs", []),
            neighborhood=neighborhood,
            node_ids=node_ids,
        )
        if len(node_ids) + len(edge_pairs) > self.max_units or len(node_ids) < 2:
            return None

        approved_question_types = [
            item
            for item in payload.get("approved_question_types", [])
            if item in (ALLOWED_DEGRADED_QUESTION_TYPES if degraded else ALLOWED_PRIMARY_QUESTION_TYPES)
        ]
        return SubgraphCandidate(
            candidate_id=f"candidate-{candidate_index}",
            intent=intent.get("intent", "technical interpretation"),
            technical_focus=compact_text(
                payload.get("technical_focus", intent.get("technical_focus", "technical interpretation")),
                limit=80,
            ),
            node_ids=node_ids,
            edge_pairs=[list(pair) for pair in edge_pairs],
            approved_question_types=approved_question_types
            or list(intent.get("question_types", []))
            or (
                ALLOWED_DEGRADED_QUESTION_TYPES[:1]
                if degraded
                else ALLOWED_PRIMARY_QUESTION_TYPES[:1]
            ),
            image_grounding_summary=compact_text(
                payload.get("image_grounding_summary", ""),
                limit=240,
            ),
            evidence_summary=compact_text(
                payload.get("evidence_summary", ""),
                limit=240,
            ),
            degraded=degraded,
        )

    async def _judge_candidate(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        neighborhood: dict[str, Any],
        candidate: SubgraphCandidate,
        degraded: bool,
    ) -> tuple[JudgeScorecard, str]:
        prompt = build_judge_prompt(
            seed_node_id=seed_node_id,
            degraded=degraded,
            judge_pass_threshold=self.judge_pass_threshold,
            candidate=candidate,
            candidate_prompt=build_candidate_prompt(self.graph, candidate),
        )
        raw = await self.llm_client.generate_answer(prompt, image_path=image_path or None)
        payload = extract_json_payload(raw)
        if not isinstance(payload, dict):
            return self._rejected_scorecard(), "invalid_judge_payload"

        try:
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
        except (TypeError, ValueError):
            return self._rejected_scorecard(), "invalid_judge_payload"

        mandatory_pass = (
            scorecard.image_indispensability >= 0.65
            and scorecard.answer_stability >= 0.6
            and scorecard.evidence_closure >= 0.6
            and scorecard.technical_relevance >= 0.6
            and scorecard.hallucination_risk <= 0.45
            and scorecard.overall_score >= self.judge_pass_threshold
        )
        scorecard.passes = scorecard.passes and mandatory_pass
        if not scorecard.passes:
            return (
                scorecard,
                compact_text(
                    payload.get("rejection_reason", ""),
                    limit=160,
                )
                or "rejected_by_judge",
            )
        return scorecard, ""

    def _select_candidates(
        self, candidates: list[SubgraphCandidate]
    ) -> list[SubgraphCandidate]:
        if not candidates:
            return []
        selected = [candidates[0]]
        if self.max_selected_subgraphs <= 1:
            return selected

        for candidate in candidates[1:]:
            if len(selected) >= self.max_selected_subgraphs:
                break
            if not self._themes_are_distinct(selected, candidate):
                continue
            selected.append(candidate)
        return selected

    def _themes_are_distinct(
        self, selected: list[SubgraphCandidate], candidate: SubgraphCandidate
    ) -> bool:
        for item in selected:
            overlap = len(set(item.node_ids) & set(candidate.node_ids)) / max(
                1, len(set(item.node_ids) | set(candidate.node_ids))
            )
            score_gap = abs(item.judge_scores.overall_score - candidate.judge_scores.overall_score)
            if overlap > 0.55:
                return False
            if item.technical_focus == candidate.technical_focus and score_gap < self.theme_split_threshold:
                return False
        return True

    def _materialize_selected_subgraph(
        self, candidate: SubgraphCandidate
    ) -> SelectedSubgraphArtifact:
        node_ids = list(dict.fromkeys(candidate.node_ids))
        edge_pairs = {
            normalize_edge_pair(src_id, tgt_id) for src_id, tgt_id in candidate.edge_pairs
        }
        return SelectedSubgraphArtifact(
            subgraph_id=candidate.candidate_id,
            technical_focus=candidate.technical_focus,
            nodes=self._node_payloads(set(node_ids)),
            edges=self._edge_payloads(edge_pairs),
            image_grounding_summary=candidate.image_grounding_summary,
            evidence_summary=candidate.evidence_summary,
            judge_scores=candidate.judge_scores,
            approved_question_types=candidate.approved_question_types,
            degraded=candidate.degraded,
        )

    def _build_empty_result(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        candidate_bundle: list[dict[str, Any]],
        degraded: bool,
        degraded_reason: str,
    ) -> dict:
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
        }

    def _build_rejected_candidate(
        self,
        *,
        candidate_index: int,
        intent: dict[str, Any],
        degraded: bool,
        reason: str,
    ) -> SubgraphCandidate:
        candidate = SubgraphCandidate(
            candidate_id=f"candidate-{candidate_index}",
            intent=intent.get("intent", "technical interpretation"),
            technical_focus=compact_text(
                intent.get("technical_focus", "technical interpretation"), limit=80
            ),
            node_ids=[],
            edge_pairs=[],
            approved_question_types=list(intent.get("question_types", [])),
            degraded=degraded,
        )
        candidate.judge_scores = self._rejected_scorecard()
        candidate.decision = "rejected"
        candidate.rejection_reason = reason
        return candidate

    @staticmethod
    def _rejected_scorecard() -> JudgeScorecard:
        return JudgeScorecard(
            image_indispensability=0.0,
            answer_stability=0.0,
            evidence_closure=0.0,
            technical_relevance=0.0,
            reasoning_depth=0.0,
            hallucination_risk=1.0,
            theme_coherence=0.0,
            overall_score=0.0,
            passes=False,
        )

    def _collect_seed_scope(
        self, seed_node_id: str, nodes: list[tuple[str, dict]]
    ) -> tuple[set[str], set[str]]:
        seed_chunk_ids = set()
        seed_source_ids = set()
        for node_id, node_data in nodes:
            if node_id != seed_node_id:
                continue
            metadata = load_metadata(node_data.get("metadata"))
            direct_source_ids = split_source_ids(node_data.get("source_id", ""))
            trace_source_ids = split_source_ids(metadata.get("source_trace_id", ""))
            seed_chunk_ids.update(direct_source_ids)
            if not seed_chunk_ids:
                seed_chunk_ids.update(trace_source_ids)
            seed_source_ids.update(direct_source_ids)
            seed_source_ids.update(trace_source_ids)
            break
        return seed_chunk_ids, seed_source_ids

    def _collect_neighborhood(
        self,
        *,
        seed_node_id: str,
        seed_source_ids: set[str],
        seed_chunk_ids: set[str],
    ) -> dict[str, Any]:
        queue = deque([(seed_node_id, 0)])
        visited = {seed_node_id}
        distances = {seed_node_id: 0}
        node_ids = [seed_node_id]
        while queue:
            node_id, depth = queue.popleft()
            if depth >= self.max_hops_from_seed:
                continue
            for neighbor_id in self.graph.get_neighbors(node_id):
                if neighbor_id in visited:
                    continue
                node_data = self.graph.get_node(neighbor_id) or {}
                if seed_source_ids or seed_chunk_ids:
                    if not self._belongs_to_seed_scope(
                        node_data, seed_source_ids, seed_chunk_ids
                    ) and depth > 0:
                        continue
                visited.add(neighbor_id)
                distances[str(neighbor_id)] = depth + 1
                node_ids.append(str(neighbor_id))
                queue.append((neighbor_id, depth + 1))

        edge_payloads = []
        node_set = set(node_ids)
        for node_id in node_ids:
            for neighbor_id in self.graph.get_neighbors(node_id):
                pair = normalize_edge_pair(node_id, neighbor_id)
                if neighbor_id not in node_set:
                    continue
                edge_data = self.graph.get_edge(node_id, neighbor_id) or self.graph.get_edge(
                    neighbor_id, node_id
                )
                if edge_data and (str(pair[0]), str(pair[1]), edge_data) not in edge_payloads:
                    edge_payloads.append((str(pair[0]), str(pair[1]), edge_data))

        ranked_node_ids = [seed_node_id]
        remaining = []
        for node_id in node_ids:
            if node_id == seed_node_id:
                continue
            node_data = self.graph.get_node(node_id) or {}
            node_score = 0.0
            if str(node_data.get("evidence_span", "")).strip():
                node_score += 1.0
            node_score += max(0.0, 0.6 - 0.1 * distances.get(node_id, 0))
            node_score += 0.15 * self._count_textual_technical_hits(
                "\n".join(
                    [
                        str(node_data.get("description", "")),
                        str(node_data.get("entity_name", "")),
                    ]
                )
            )
            remaining.append((node_score, node_id))
        remaining.sort(reverse=True)
        ranked_node_ids.extend([node_id for _, node_id in remaining])
        return {
            "node_ids": ranked_node_ids,
            "edges": edge_payloads,
            "distances": distances,
        }

    def _belongs_to_seed_scope(
        self, payload: dict, seed_source_ids: set[str], seed_chunk_ids: set[str]
    ) -> bool:
        source_ids = split_source_ids(payload.get("source_id", ""))
        if source_ids & seed_chunk_ids:
            return True
        if source_ids & seed_source_ids:
            return True
        metadata = load_metadata(payload.get("metadata"))
        nested_source_ids = split_source_ids(metadata.get("source_trace_id", ""))
        return bool(nested_source_ids & seed_source_ids)

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

    def _sanitize_node_ids(
        self,
        raw_node_ids: list[Any],
        *,
        neighborhood: dict[str, Any],
        seed_node_id: str,
    ) -> list[str]:
        available = set(neighborhood["node_ids"])
        node_ids = [seed_node_id]
        for item in raw_node_ids:
            node_id = str(item)
            if node_id == seed_node_id or node_id not in available:
                continue
            if node_id not in node_ids:
                node_ids.append(node_id)
        return node_ids[: self.max_units]

    def _sanitize_edge_pairs(
        self,
        raw_edge_pairs: list[Any],
        *,
        neighborhood: dict[str, Any],
        node_ids: list[str],
    ) -> list[tuple[str, str]]:
        available_pairs = {
            normalize_edge_pair(src_id, tgt_id)
            for src_id, tgt_id, _ in neighborhood["edges"]
        }
        node_set = set(node_ids)
        cleaned = []
        for item in raw_edge_pairs:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            pair = normalize_edge_pair(str(item[0]), str(item[1]))
            if pair not in available_pairs:
                continue
            if pair[0] not in node_set or pair[1] not in node_set:
                continue
            if pair not in cleaned:
                cleaned.append(pair)
        return cleaned[: self.max_units]

    def _extract_image_path(self, node_data: dict) -> str:
        metadata = load_metadata(node_data.get("metadata"))
        for key in ("image_path", "img_path"):
            if metadata.get(key):
                return str(metadata[key])
        return ""

    def _count_textual_technical_hits(self, text_blob: str) -> int:
        text_blob = text_blob.lower()
        return sum(
            1
            for keywords in TECHNICAL_KEYWORDS.values()
            for keyword in keywords
            if keyword in text_blob
        )
