from __future__ import annotations

from collections import Counter, deque
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
from .constants import ALLOWED_DEGRADED_QUESTION_TYPES, ALLOWED_PRIMARY_QUESTION_TYPES
from .schema_guided_prompts import (
    build_schema_guided_candidate_prompt,
    build_schema_guided_judge_prompt,
    build_schema_guided_neighborhood_prompt,
    build_schema_guided_planner_prompt,
)


class SchemaGuidedVLMSubgraphSampler:
    def __init__(
        self,
        graph: BaseGraphStorage,
        llm_client: BaseLLMWrapper,
        *,
        candidate_pool_size: int = 3,
        max_selected_subgraphs: int = 1,
        max_vqas_per_selected_subgraph: int = 2,
        initial_hops: int = 1,
        max_hops_after_reflection: int = 2,
        hard_cap_units: int = 12,
        section_scoped: bool = True,
        same_source_only: bool = True,
        allow_type_relaxation: bool = True,
        allow_degraded: bool = True,
        judge_pass_threshold: float = 0.68,
    ):
        self.graph = graph
        self.llm_client = llm_client
        self.candidate_pool_size = max(1, int(candidate_pool_size))
        self.max_selected_subgraphs = max(1, int(max_selected_subgraphs))
        self.max_vqas_per_selected_subgraph = min(
            3, max(1, int(max_vqas_per_selected_subgraph))
        )
        self.initial_hops = max(1, int(initial_hops))
        self.max_hops_after_reflection = max(
            self.initial_hops, int(max_hops_after_reflection)
        )
        self.hard_cap_units = max(4, int(hard_cap_units))
        self.section_scoped = bool(section_scoped)
        self.same_source_only = bool(same_source_only)
        self.allow_type_relaxation = bool(allow_type_relaxation)
        self.allow_degraded = bool(allow_degraded)
        self.judge_pass_threshold = float(judge_pass_threshold)

    async def sample(
        self,
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]],
        *,
        seed_node_id: str,
        debug: bool = False,
    ) -> dict[str, Any]:
        del debug
        nodes, _edges = batch
        seed_node = self.graph.get_node(seed_node_id) or {}
        image_path = self._extract_image_path(seed_node)
        if not seed_node_id or not seed_node or not image_path:
            return self._build_empty_result(
                seed_node_id=seed_node_id,
                image_path=image_path,
                candidate_bundle=[],
                degraded=False,
                degraded_reason="missing_image_seed_or_asset",
                inferred_schema={},
                intent_bundle=[],
                retrieval_trace=[],
                termination_reason="missing_image_seed_or_asset",
            )

        all_nodes = {str(node_id): node_data for node_id, node_data in nodes}
        seed_scope = self._collect_seed_scope(seed_node)
        seed_section = self._extract_section_path(seed_node)
        inferred_schema = self._infer_runtime_schema(
            seed_node_id=seed_node_id,
            seed_scope=seed_scope,
            seed_section=seed_section,
        )

        primary = await self._run_sampling_session(
            seed_node_id=seed_node_id,
            seed_node=seed_node,
            image_path=image_path,
            all_nodes=all_nodes,
            seed_scope=seed_scope,
            seed_section=seed_section,
            inferred_schema=inferred_schema,
            degraded=False,
        )
        if primary["selected_subgraphs"]:
            return primary

        if self.allow_degraded:
            degraded = await self._run_sampling_session(
                seed_node_id=seed_node_id,
                seed_node=seed_node,
                image_path=image_path,
                all_nodes=all_nodes,
                seed_scope=seed_scope,
                seed_section=seed_section,
                inferred_schema=inferred_schema,
                degraded=True,
            )
            if degraded["selected_subgraphs"]:
                return degraded
            primary["candidate_bundle"].extend(degraded.get("candidate_bundle", []))
            primary["retrieval_trace"].extend(degraded.get("retrieval_trace", []))
            primary["termination_reason"] = degraded.get(
                "termination_reason", primary["termination_reason"]
            )
            primary["degraded_reason"] = degraded.get("degraded_reason", "")
            primary["degraded"] = degraded.get("degraded", primary["degraded"])
        return primary

    async def _run_sampling_session(
        self,
        *,
        seed_node_id: str,
        seed_node: dict[str, Any],
        image_path: str,
        all_nodes: dict[str, dict[str, Any]],
        seed_scope: set[str],
        seed_section: str,
        inferred_schema: dict[str, Any],
        degraded: bool,
    ) -> dict[str, Any]:
        planning_neighborhood = self._collect_neighborhood(
            seed_node_id=seed_node_id,
            seed_scope=seed_scope,
            seed_section=seed_section,
            max_hops=self.initial_hops,
        )
        intents = await self._propose_intents(
            seed_node_id=seed_node_id,
            seed_node=seed_node,
            image_path=image_path,
            inferred_schema=inferred_schema,
            planning_neighborhood=planning_neighborhood,
            degraded=degraded,
        )

        accepted_candidates: list[SubgraphCandidate] = []
        candidate_bundle: list[dict[str, Any]] = []
        retrieval_trace: list[dict[str, Any]] = []
        for index, intent in enumerate(intents[: self.candidate_pool_size], start=1):
            candidate, trace_entries = await self._retrieve_intent_with_reflection(
                seed_node_id=seed_node_id,
                image_path=image_path,
                all_nodes=all_nodes,
                seed_scope=seed_scope,
                seed_section=seed_section,
                inferred_schema=inferred_schema,
                intent=intent,
                degraded=degraded,
                candidate_index=index,
            )
            retrieval_trace.extend(trace_entries)
            for entry in trace_entries:
                bundle_item = dict(entry.get("candidate_bundle_item", {}))
                if bundle_item:
                    candidate_bundle.append(bundle_item)
            if candidate is not None and candidate.decision == "accepted":
                accepted_candidates.append(candidate)

        accepted_candidates.sort(
            key=lambda item: item.judge_scores.overall_score,
            reverse=True,
        )
        selected_candidates = self._select_candidates(accepted_candidates)
        selected_subgraphs = [
            self._materialize_selected_subgraph(candidate).to_dict()
            for candidate in selected_candidates
        ]

        return {
            "seed_node_id": seed_node_id,
            "seed_image_path": image_path,
            "selection_mode": "single",
            "degraded": degraded and bool(selected_subgraphs),
            "degraded_reason": (
                "fallback_to_schema_relaxed_retrieval" if degraded and selected_subgraphs else ""
            ),
            "selected_subgraphs": selected_subgraphs,
            "candidate_bundle": candidate_bundle,
            "abstained": not bool(selected_subgraphs),
            "max_vqas_per_selected_subgraph": self.max_vqas_per_selected_subgraph,
            "inferred_schema": inferred_schema,
            "intent_bundle": intents,
            "retrieval_trace": retrieval_trace,
            "termination_reason": (
                "selected_schema_guided_candidate"
                if selected_subgraphs
                else ("degraded_exhausted" if degraded else "no_candidate_passed_judge")
            ),
        }

    async def _propose_intents(
        self,
        *,
        seed_node_id: str,
        seed_node: dict[str, Any],
        image_path: str,
        inferred_schema: dict[str, Any],
        planning_neighborhood: dict[str, Any],
        degraded: bool,
    ) -> list[dict[str, Any]]:
        allowed_question_types = (
            ALLOWED_DEGRADED_QUESTION_TYPES
            if degraded
            else ALLOWED_PRIMARY_QUESTION_TYPES
        )
        prompt = build_schema_guided_planner_prompt(
            seed_node_id=seed_node_id,
            seed_description=seed_node.get("description", ""),
            image_path=image_path,
            inferred_schema=inferred_schema,
            neighborhood_prompt=build_schema_guided_neighborhood_prompt(
                self.graph, planning_neighborhood
            ),
            candidate_pool_size=self.candidate_pool_size,
            allowed_question_types=allowed_question_types,
            degraded=degraded,
        )
        raw = await self.llm_client.generate_answer(prompt, image_path=image_path or None)
        payload = extract_json_payload(raw)
        normalized: list[dict[str, Any]] = []
        intents = payload.get("intents") if isinstance(payload, dict) else None
        if isinstance(intents, list):
            for item in intents:
                if not isinstance(item, dict):
                    continue
                intent_text = compact_text(item.get("intent", ""), limit=120)
                if not intent_text:
                    continue
                question_types = [
                    str(q).strip()
                    for q in item.get("question_types", [])
                    if str(q).strip() in allowed_question_types
                ]
                normalized.append(
                    {
                        "intent": intent_text,
                        "technical_focus": compact_text(
                            item.get("technical_focus", intent_text), limit=80
                        ),
                        "question_types": question_types
                        or self._fallback_question_types(
                            technical_focus=str(item.get("technical_focus", intent_text)),
                            degraded=degraded,
                        ),
                        "priority_keywords": self._normalize_short_list(
                            item.get("priority_keywords", []),
                            limit=40,
                        ),
                        "target_node_types": self._normalize_short_list(
                            item.get("target_node_types", []),
                            limit=40,
                        ),
                        "target_relation_types": self._normalize_short_list(
                            item.get("target_relation_types", []),
                            limit=40,
                        ),
                        "required_modalities": self._normalize_short_list(
                            item.get("required_modalities", []),
                            limit=24,
                        ),
                        "evidence_requirements": self._normalize_short_list(
                            item.get("evidence_requirements", []),
                            limit=32,
                        ),
                    }
                )
        if normalized:
            return normalized[: self.candidate_pool_size]

        return [
            {
                "intent": compact_text(
                    seed_node.get("description", "technical interpretation"),
                    limit=120,
                )
                or "technical interpretation",
                "technical_focus": self._guess_technical_focus(seed_node),
                "question_types": self._fallback_question_types(
                    technical_focus=self._guess_technical_focus(seed_node),
                    degraded=degraded,
                ),
                "priority_keywords": self._extract_keywords_from_text(
                    seed_node.get("description", "")
                )[:4],
                "target_node_types": [],
                "target_relation_types": [],
                "required_modalities": ["image", "text"],
                "evidence_requirements": ["same_section", "same_source"],
            }
        ]

    async def _retrieve_intent_with_reflection(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        all_nodes: dict[str, dict[str, Any]],
        seed_scope: set[str],
        seed_section: str,
        inferred_schema: dict[str, Any],
        intent: dict[str, Any],
        degraded: bool,
        candidate_index: int,
    ) -> tuple[SubgraphCandidate | None, list[dict[str, Any]]]:
        trace_entries: list[dict[str, Any]] = []
        retrieval_stages = self._build_retrieval_stages()
        best_candidate: SubgraphCandidate | None = None
        for stage_index, stage in enumerate(retrieval_stages, start=1):
            neighborhood = self._collect_neighborhood(
                seed_node_id=seed_node_id,
                seed_scope=seed_scope,
                seed_section=seed_section,
                max_hops=stage["max_hops"],
                include_sibling_text=stage["include_sibling_text"],
                all_nodes=all_nodes,
            )
            candidate = self._retrieve_candidate(
                seed_node_id=seed_node_id,
                neighborhood=neighborhood,
                intent=intent,
                degraded=degraded,
                candidate_id=f"candidate-{candidate_index}",
                retrieval_stage=stage["name"],
                relax_types=bool(stage["relax_types"]),
            )
            if candidate is None:
                trace_entries.append(
                    {
                        "intent": intent.get("intent", ""),
                        "retrieval_stage": stage["name"],
                        "accepted": False,
                        "needs_expansion": stage_index < len(retrieval_stages),
                        "candidate_bundle_item": {
                            "candidate_id": f"candidate-{candidate_index}",
                            "intent": intent.get("intent", ""),
                            "technical_focus": intent.get("technical_focus", ""),
                            "decision": "rejected",
                            "rejection_reason": "retrieval_returned_empty_candidate",
                            "retrieval_stage": stage["name"],
                            "judge_scores": self._rejected_scorecard().to_dict(),
                            "approved_question_types": intent.get("question_types", []),
                        },
                    }
                )
                continue

            scorecard, rejection_reason, needs_expansion, suggested_actions = (
                await self._judge_candidate(
                    seed_node_id=seed_node_id,
                    image_path=image_path,
                    inferred_schema=inferred_schema,
                    candidate=candidate,
                    degraded=degraded,
                    retrieval_stage=stage["name"],
                )
            )
            candidate.judge_scores = scorecard
            candidate.decision = "accepted" if scorecard.passes else "rejected"
            candidate.rejection_reason = rejection_reason if not scorecard.passes else ""
            bundle_item = {
                "candidate_id": candidate.candidate_id,
                "intent": candidate.intent,
                "technical_focus": candidate.technical_focus,
                "node_ids": candidate.node_ids,
                "edge_pairs": candidate.edge_pairs,
                "approved_question_types": candidate.approved_question_types,
                "judge_scores": scorecard.to_dict(),
                "decision": candidate.decision,
                "rejection_reason": candidate.rejection_reason,
                "retrieval_stage": stage["name"],
            }
            trace_entries.append(
                {
                    "intent": intent.get("intent", ""),
                    "retrieval_stage": stage["name"],
                    "accepted": scorecard.passes,
                    "needs_expansion": needs_expansion,
                    "suggested_actions": suggested_actions,
                    "candidate_bundle_item": bundle_item,
                }
            )
            if best_candidate is None or (
                candidate.judge_scores.overall_score
                > best_candidate.judge_scores.overall_score
            ):
                best_candidate = candidate
            if scorecard.passes:
                return candidate, trace_entries
            if not needs_expansion:
                break
        return best_candidate, trace_entries

    def _retrieve_candidate(
        self,
        *,
        seed_node_id: str,
        neighborhood: dict[str, Any],
        intent: dict[str, Any],
        degraded: bool,
        candidate_id: str,
        retrieval_stage: str,
        relax_types: bool,
    ) -> SubgraphCandidate | None:
        selected_nodes = self._select_candidate_nodes(
            seed_node_id=seed_node_id,
            neighborhood=neighborhood,
            intent=intent,
            relax_types=relax_types or degraded,
        )
        if len(selected_nodes) < 2:
            return None

        selected_edges = self._select_candidate_edges(
            selected_nodes=selected_nodes,
            neighborhood=neighborhood,
            intent=intent,
            relax_types=relax_types or degraded,
        )
        if not selected_edges and len(selected_nodes) < 3:
            return None

        technical_focus = compact_text(
            intent.get("technical_focus", intent.get("intent", "technical interpretation")),
            limit=80,
        )
        return SubgraphCandidate(
            candidate_id=candidate_id,
            intent=compact_text(intent.get("intent", "technical interpretation"), limit=120),
            technical_focus=technical_focus,
            node_ids=selected_nodes,
            edge_pairs=[list(pair) for pair in selected_edges],
            approved_question_types=list(intent.get("question_types", []))
            or self._fallback_question_types(
                technical_focus=technical_focus, degraded=degraded
            ),
            image_grounding_summary=compact_text(
                (
                    f"The seed image is required because `{technical_focus}` must be read "
                    f"from the image and linked with {retrieval_stage} evidence."
                ),
                limit=240,
            ),
            evidence_summary=compact_text(
                self._build_evidence_summary(
                    selected_nodes=selected_nodes,
                    selected_edges=selected_edges,
                    retrieval_stage=retrieval_stage,
                ),
                limit=240,
            ),
            degraded=degraded,
        )

    async def _judge_candidate(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        inferred_schema: dict[str, Any],
        candidate: SubgraphCandidate,
        degraded: bool,
        retrieval_stage: str,
    ) -> tuple[JudgeScorecard, str, bool, list[str]]:
        prompt = build_schema_guided_judge_prompt(
            seed_node_id=seed_node_id,
            degraded=degraded,
            judge_pass_threshold=self.judge_pass_threshold,
            retrieval_stage=retrieval_stage,
            candidate=candidate,
            candidate_prompt=build_schema_guided_candidate_prompt(
                graph=self.graph,
                candidate=candidate,
                retrieval_stage=retrieval_stage,
                inferred_schema=inferred_schema,
            ),
        )
        raw = await self.llm_client.generate_answer(prompt, image_path=image_path or None)
        payload = extract_json_payload(raw)
        if not isinstance(payload, dict):
            return self._rejected_scorecard(), "invalid_judge_payload", True, []

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
        mandatory_pass = (
            scorecard.image_indispensability >= 0.65
            and scorecard.answer_stability >= 0.6
            and scorecard.evidence_closure >= 0.6
            and scorecard.technical_relevance >= 0.6
            and scorecard.hallucination_risk <= 0.45
            and scorecard.overall_score >= self.judge_pass_threshold
        )
        sufficient = bool(payload.get("sufficient", scorecard.passes)) and mandatory_pass
        scorecard.passes = bool(scorecard.passes and sufficient)
        rejection_reason = compact_text(
            payload.get("rejection_reason", ""), limit=160
        ) or ("rejected_by_judge" if not scorecard.passes else "")
        suggested_actions = self._normalize_short_list(
            payload.get("suggested_actions", []), limit=48
        )
        return (
            scorecard,
            rejection_reason,
            bool(payload.get("needs_expansion", False)),
            suggested_actions,
        )

    def _select_candidate_nodes(
        self,
        *,
        seed_node_id: str,
        neighborhood: dict[str, Any],
        intent: dict[str, Any],
        relax_types: bool,
    ) -> list[str]:
        node_scores: list[tuple[float, str]] = []
        distances = neighborhood.get("distances", {})
        target_node_types = {
            self._canonical_token(item) for item in intent.get("target_node_types", [])
        }
        required_modalities = {
            self._canonical_token(item) for item in intent.get("required_modalities", [])
        }
        keywords = {
            self._canonical_token(item)
            for item in (
                list(intent.get("priority_keywords", []))
                + self._extract_keywords_from_text(intent.get("intent", ""))
                + self._extract_keywords_from_text(intent.get("technical_focus", ""))
            )
        }

        for node_id in neighborhood.get("node_ids", []):
            if node_id == seed_node_id:
                continue
            node_data = self.graph.get_node(node_id) or {}
            node_type = self._canonical_token(node_data.get("entity_type", ""))
            modality = self._normalize_modality(node_data)
            node_text = self._node_text_blob(node_id, node_data)
            type_match = bool(target_node_types) and any(
                target in node_type or node_type in target
                for target in target_node_types
                if target and node_type
            )
            if target_node_types and not type_match and not relax_types:
                continue

            score = 0.0
            score += max(0.0, 4.0 - float(distances.get(node_id, 4)))
            score += 3.0 if type_match else 0.0
            if required_modalities and modality in required_modalities:
                score += 2.0
            keyword_hits = sum(1 for keyword in keywords if keyword and keyword in node_text)
            score += float(keyword_hits) * 1.5
            if node_data.get("evidence_span"):
                score += 0.5
            if node_data.get("description"):
                score += 0.5
            node_scores.append((score, node_id))

        node_scores.sort(key=lambda item: (-item[0], item[1]))
        max_nodes = max(2, (self.hard_cap_units + 1) // 2)
        selected = [seed_node_id]
        for _score, node_id in node_scores:
            if node_id not in selected:
                selected.append(node_id)
            if len(selected) >= max_nodes:
                break
        return selected

    def _select_candidate_edges(
        self,
        *,
        selected_nodes: list[str],
        neighborhood: dict[str, Any],
        intent: dict[str, Any],
        relax_types: bool,
    ) -> list[tuple[str, str]]:
        del relax_types
        selected_set = set(selected_nodes)
        target_relation_types = {
            self._canonical_token(item)
            for item in intent.get("target_relation_types", [])
        }
        keywords = {
            self._canonical_token(item)
            for item in intent.get("priority_keywords", [])
        }
        ranked_edges: list[tuple[float, tuple[str, str]]] = []
        for src_id, tgt_id, edge_data in neighborhood.get("edges", []):
            if src_id not in selected_set or tgt_id not in selected_set:
                continue
            pair = normalize_edge_pair(src_id, tgt_id)
            relation_text = self._edge_text_blob(edge_data)
            relation_type = self._canonical_token(
                edge_data.get("relation_type", edge_data.get("description", ""))
            )
            score = 1.0
            if target_relation_types and any(
                token in relation_type or relation_type in token
                for token in target_relation_types
                if token and relation_type
            ):
                score += 3.0
            keyword_hits = sum(
                1 for keyword in keywords if keyword and keyword in relation_text
            )
            score += float(keyword_hits) * 1.5
            if edge_data.get("evidence_span"):
                score += 0.5
            ranked_edges.append((score, pair))

        ranked_edges.sort(key=lambda item: (-item[0], item[1]))
        max_edges = max(1, self.hard_cap_units - len(selected_nodes))
        chosen: list[tuple[str, str]] = []
        for _score, pair in ranked_edges:
            if pair not in chosen:
                chosen.append(pair)
            if len(chosen) >= max_edges:
                break
        return chosen

    def _build_retrieval_stages(self) -> list[dict[str, Any]]:
        stages = [
            {
                "name": "same_section_one_hop",
                "max_hops": self.initial_hops,
                "include_sibling_text": False,
                "relax_types": False,
            }
        ]
        if self.max_hops_after_reflection > self.initial_hops:
            stages.append(
                {
                    "name": "same_section_two_hop",
                    "max_hops": self.max_hops_after_reflection,
                    "include_sibling_text": False,
                    "relax_types": False,
                }
            )
        stages.append(
            {
                "name": "same_source_sibling_text",
                "max_hops": self.max_hops_after_reflection,
                "include_sibling_text": True,
                "relax_types": self.allow_type_relaxation,
            }
        )
        return stages

    def _collect_seed_scope(self, seed_node: dict[str, Any]) -> set[str]:
        metadata = load_metadata(seed_node.get("metadata"))
        seed_scope = set()
        seed_scope.update(split_source_ids(seed_node.get("source_id", "")))
        seed_scope.update(split_source_ids(metadata.get("source_trace_id", "")))
        return seed_scope

    def _collect_neighborhood(
        self,
        *,
        seed_node_id: str,
        seed_scope: set[str],
        seed_section: str,
        max_hops: int,
        include_sibling_text: bool = False,
        all_nodes: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        queue = deque([(seed_node_id, 0)])
        visited = {seed_node_id}
        distances = {seed_node_id: 0}
        node_ids = [seed_node_id]
        while queue:
            node_id, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbor_id in self.graph.get_neighbors(node_id):
                if neighbor_id in visited:
                    continue
                edge_data = self.graph.get_edge(node_id, neighbor_id) or self.graph.get_edge(
                    neighbor_id, node_id
                ) or {}
                node_data = self.graph.get_node(neighbor_id) or {}
                if not self._passes_scope_filters(
                    node_data=node_data,
                    edge_data=edge_data,
                    seed_scope=seed_scope,
                    seed_section=seed_section,
                ):
                    continue
                visited.add(neighbor_id)
                distances[str(neighbor_id)] = depth + 1
                node_ids.append(str(neighbor_id))
                queue.append((neighbor_id, depth + 1))

        if include_sibling_text and all_nodes:
            for node_id, node_data in all_nodes.items():
                if node_id in visited or node_id == seed_node_id:
                    continue
                if self._normalize_modality(node_data) != "text":
                    continue
                if not self._passes_node_scope(
                    node_data=node_data,
                    seed_scope=seed_scope,
                    seed_section=seed_section,
                ):
                    continue
                visited.add(node_id)
                distances[node_id] = max_hops + 1
                node_ids.append(node_id)

        edge_payloads = []
        node_set = set(node_ids)
        for node_id in node_ids:
            for neighbor_id in self.graph.get_neighbors(node_id):
                if neighbor_id not in node_set:
                    continue
                pair = normalize_edge_pair(node_id, neighbor_id)
                edge_data = self.graph.get_edge(node_id, neighbor_id) or self.graph.get_edge(
                    neighbor_id, node_id
                )
                if edge_data and (pair[0], pair[1], edge_data) not in edge_payloads:
                    edge_payloads.append((pair[0], pair[1], edge_data))
        ranked_nodes = [seed_node_id] + sorted(
            [node_id for node_id in node_ids if node_id != seed_node_id],
            key=lambda item: (distances.get(item, 99), item),
        )
        return {
            "node_ids": ranked_nodes,
            "edges": edge_payloads,
            "distances": distances,
            "max_hops": max_hops,
        }

    def _passes_scope_filters(
        self,
        *,
        node_data: dict[str, Any],
        edge_data: dict[str, Any],
        seed_scope: set[str],
        seed_section: str,
    ) -> bool:
        if self.section_scoped and seed_section:
            node_section = self._extract_section_path(node_data)
            if node_section and node_section != seed_section:
                return False
        if not self.same_source_only or not seed_scope:
            return True

        metadata = load_metadata(node_data.get("metadata"))
        node_scope = split_source_ids(node_data.get("source_id", ""))
        node_scope.update(split_source_ids(metadata.get("source_trace_id", "")))
        edge_scope = split_source_ids(edge_data.get("source_id", ""))
        if node_scope & seed_scope:
            return True
        if edge_scope & seed_scope:
            return True
        return not node_scope and not edge_scope

    def _passes_node_scope(
        self,
        *,
        node_data: dict[str, Any],
        seed_scope: set[str],
        seed_section: str,
    ) -> bool:
        if self.section_scoped and seed_section:
            node_section = self._extract_section_path(node_data)
            if node_section and node_section != seed_section:
                return False
        if self.same_source_only and seed_scope:
            node_scope = split_source_ids(node_data.get("source_id", ""))
            metadata = load_metadata(node_data.get("metadata"))
            node_scope.update(split_source_ids(metadata.get("source_trace_id", "")))
            if node_scope and not (node_scope & seed_scope):
                return False
        return True

    def _passes_edge_scope(self, *, edge_data: dict[str, Any], seed_scope: set[str]) -> bool:
        if self.same_source_only and seed_scope:
            edge_scope = split_source_ids(edge_data.get("source_id", ""))
            if edge_scope and not (edge_scope & seed_scope):
                return False
        return True

    def _infer_runtime_schema(
        self,
        *,
        seed_node_id: str,
        seed_scope: set[str],
        seed_section: str,
    ) -> dict[str, Any]:
        schema_neighborhood = self._collect_neighborhood(
            seed_node_id=seed_node_id,
            seed_scope=seed_scope,
            seed_section=seed_section,
            max_hops=self.max_hops_after_reflection,
        )
        node_type_counter: Counter[str] = Counter()
        relation_counter: Counter[str] = Counter()
        modality_counter: Counter[str] = Counter()
        path_counter: Counter[str] = Counter()
        for node_id in schema_neighborhood.get("node_ids", []):
            node_data = self.graph.get_node(node_id) or {}
            node_type = self._canonical_token(node_data.get("entity_type", "")) or "unknown"
            node_type_counter[node_type] += 1
            modality_counter[self._normalize_modality(node_data)] += 1
            path = self._extract_section_path(node_data)
            if path:
                path_counter[path] += 1
        for _src_id, _tgt_id, edge_data in schema_neighborhood.get("edges", []):
            relation = self._canonical_token(
                edge_data.get("relation_type", edge_data.get("description", ""))
            )
            if relation:
                relation_counter[relation] += 1
        return {
            "seed_section": seed_section,
            "seed_scope": sorted(seed_scope),
            "node_types": [
                {"type": key, "count": value}
                for key, value in node_type_counter.most_common(10)
            ],
            "relation_types": [
                {"type": key, "count": value}
                for key, value in relation_counter.most_common(10)
            ],
            "modalities": [
                {"modality": key, "count": value}
                for key, value in modality_counter.most_common(6)
            ],
            "section_paths": [
                {"path": key, "count": value}
                for key, value in path_counter.most_common(6)
            ],
        }

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
            overlap = len(set(candidate.node_ids) & set(selected[0].node_ids)) / max(
                1, len(set(candidate.node_ids) | set(selected[0].node_ids))
            )
            if overlap <= 0.55:
                selected.append(candidate)
        return selected

    def _materialize_selected_subgraph(
        self, candidate: SubgraphCandidate
    ) -> SelectedSubgraphArtifact:
        edge_pairs = {
            normalize_edge_pair(src_id, tgt_id) for src_id, tgt_id in candidate.edge_pairs
        }
        return SelectedSubgraphArtifact(
            subgraph_id=candidate.candidate_id,
            technical_focus=candidate.technical_focus,
            nodes=self._node_payloads(set(candidate.node_ids)),
            edges=self._edge_payloads(edge_pairs),
            image_grounding_summary=candidate.image_grounding_summary,
            evidence_summary=candidate.evidence_summary,
            judge_scores=candidate.judge_scores,
            approved_question_types=candidate.approved_question_types,
            degraded=candidate.degraded,
        )

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

    def _build_empty_result(
        self,
        *,
        seed_node_id: str,
        image_path: str,
        candidate_bundle: list[dict[str, Any]],
        degraded: bool,
        degraded_reason: str,
        inferred_schema: dict[str, Any],
        intent_bundle: list[dict[str, Any]],
        retrieval_trace: list[dict[str, Any]],
        termination_reason: str,
    ) -> dict[str, Any]:
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
            "inferred_schema": inferred_schema,
            "intent_bundle": intent_bundle,
            "retrieval_trace": retrieval_trace,
            "termination_reason": termination_reason,
        }

    def _build_evidence_summary(
        self,
        *,
        selected_nodes: list[str],
        selected_edges: list[tuple[str, str]],
        retrieval_stage: str,
    ) -> str:
        return (
            f"Selected {len(selected_nodes)} nodes and {len(selected_edges)} edges "
            f"from `{retrieval_stage}` to align image-local structure with nearby text evidence."
        )

    def _guess_technical_focus(self, node_data: dict[str, Any]) -> str:
        text = self._node_text_blob("seed", node_data)
        if any(token in text for token in ("timing", "delay", "latency", "cycle")):
            return "timing constraint interpretation"
        if any(token in text for token in ("compare", "versus", "difference")):
            return "parameter comparison"
        if any(token in text for token in ("architecture", "block", "bank", "path")):
            return "architecture relation"
        return "image-grounded technical interpretation"

    def _fallback_question_types(self, *, technical_focus: str, degraded: bool) -> list[str]:
        focus = self._canonical_token(technical_focus)
        if degraded:
            return list(ALLOWED_DEGRADED_QUESTION_TYPES[:1])
        if any(token in focus for token in ("constraint", "causal", "timing")):
            return ["local_constraint_or_causal_interpretation"]
        if any(token in focus for token in ("compare", "comparison", "parameter")):
            return ["parameter_relation_understanding"]
        if any(token in focus for token in ("multi", "reasoning", "dependency")):
            return ["light_multi_hop_technical_reasoning"]
        return ["chart_diagram_interpretation"]

    def _extract_image_path(self, node_data: dict[str, Any]) -> str:
        metadata = load_metadata(node_data.get("metadata"))
        for key in ("image_path", "img_path"):
            if metadata.get(key):
                return str(metadata[key])
        return ""

    def _extract_section_path(self, node_data: dict[str, Any]) -> str:
        metadata = load_metadata(node_data.get("metadata"))
        path = metadata.get("path") or node_data.get("path")
        return compact_text(path, limit=200) if path else ""

    def _normalize_modality(self, node_data: dict[str, Any]) -> str:
        entity_type = self._canonical_token(node_data.get("entity_type", ""))
        if "image" in entity_type:
            return "image"
        if "table" in entity_type:
            return "table"
        if "video" in entity_type:
            return "video"
        metadata = load_metadata(node_data.get("metadata"))
        modality = self._canonical_token(metadata.get("modality", ""))
        return modality or "text"

    def _node_text_blob(self, node_id: str, node_data: dict[str, Any]) -> str:
        metadata = load_metadata(node_data.get("metadata"))
        values = [
            node_id,
            node_data.get("entity_type", ""),
            node_data.get("description", ""),
            node_data.get("evidence_span", ""),
            metadata.get("path", ""),
        ]
        return " ".join(str(value or "").lower() for value in values)

    def _edge_text_blob(self, edge_data: dict[str, Any]) -> str:
        values = [
            edge_data.get("relation_type", ""),
            edge_data.get("description", ""),
            edge_data.get("evidence_span", ""),
        ]
        return " ".join(str(value or "").lower() for value in values)

    def _normalize_short_list(self, values: Any, *, limit: int) -> list[str]:
        if not isinstance(values, list):
            return []
        normalized = []
        for value in values:
            text = compact_text(value, limit=limit)
            if text:
                normalized.append(text)
        return normalized

    def _extract_keywords_from_text(self, text: Any) -> list[str]:
        raw_tokens = str(text or "").replace("/", " ").replace("_", " ").split()
        tokens = []
        for token in raw_tokens:
            token = self._canonical_token(token)
            if len(token) >= 3 and token not in tokens:
                tokens.append(token)
        return tokens

    @staticmethod
    def _canonical_token(value: Any) -> str:
        text = str(value or "").strip().lower()
        normalized = []
        for char in text:
            normalized.append(char if char.isalnum() else "_")
        return "_".join(filter(None, "".join(normalized).split("_")))

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
