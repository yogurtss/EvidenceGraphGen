import json
import math
import re
from collections import deque
from typing import Any

from graphgen.bases import BaseGraphStorage
from graphgen.utils import split_string_by_multi_markers

ANCHOR_ROLE = "anchor"
LOCAL_CORE_ROLE = "local_core"
BRIDGE_ROLE = "bridge"
SUPPORT_ROLE = "support"
COMPARISON_ROLE = "comparison"
CONCLUSION_ROLE = "conclusion"


def _split_source_ids(value: Any) -> set[str]:
    if not value:
        return set()
    return {
        item.strip()
        for item in split_string_by_multi_markers(str(value), ["<SEP>"])
        if str(item).strip()
    }


def _load_metadata(raw_metadata: Any) -> dict:
    if isinstance(raw_metadata, dict):
        return raw_metadata
    if not raw_metadata:
        return {}
    try:
        parsed = json.loads(raw_metadata)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    nested = parsed.get("metadata")
    if isinstance(nested, dict):
        merged = dict(parsed)
        merged.update(nested)
        return merged
    return parsed


def _normalize_edge_pair(src_id: str, tgt_id: str) -> tuple[str, str]:
    return tuple(sorted((str(src_id), str(tgt_id))))


class ValueAwareSubgraphSampler:
    def __init__(
        self,
        graph: BaseGraphStorage,
        *,
        max_units: int = 10,
        max_steps: int = 6,
        max_hops_from_seed: int = 4,
        min_score_improvement: float = 0.2,
    ):
        self.graph = graph
        self.max_units = max(2, int(max_units))
        self.max_steps = max(1, int(max_steps))
        self.max_hops_from_seed = max(1, int(max_hops_from_seed))
        self.min_score_improvement = float(min_score_improvement)

    def sample(
        self,
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]],
    ) -> dict:
        nodes, edges = batch
        seed_node_id = self._select_seed_node(nodes)
        if not seed_node_id:
            return self._fallback_result(nodes, edges, reason="no_vision_seed")

        seed_chunk_ids, seed_source_ids = self._collect_seed_scope(seed_node_id, nodes)
        if not seed_chunk_ids:
            return self._fallback_result(
                nodes, edges, reason="missing_seed_chunk_ids", seed_node_id=seed_node_id
            )

        local_core_node_ids, local_core_edge_pairs = self._build_local_core(
            seed_node_id=seed_node_id,
            nodes=nodes,
            edges=edges,
            seed_chunk_ids=seed_chunk_ids,
        )
        if len(local_core_node_ids) <= 1:
            return self._fallback_result(
                nodes,
                edges,
                reason="missing_local_core",
                seed_node_id=seed_node_id,
                seed_chunk_ids=seed_chunk_ids,
            )

        initial_candidate = self._evaluate_candidate(
            seed_node_id=seed_node_id,
            seed_chunk_ids=seed_chunk_ids,
            seed_source_ids=seed_source_ids,
            local_core_node_ids=local_core_node_ids,
            local_core_edge_pairs=local_core_edge_pairs,
            extension_node_ids=set(),
            extension_edge_pairs=set(),
            node_roles={seed_node_id: ANCHOR_ROLE, **{nid: LOCAL_CORE_ROLE for nid in local_core_node_ids if nid != seed_node_id}},
            extension_depths={},
            parent_extensions={},
        )

        best = initial_candidate
        current = initial_candidate
        steps = 0

        while steps < self.max_steps:
            proposals = self._propose_extensions(
                seed_node_id=seed_node_id,
                seed_chunk_ids=seed_chunk_ids,
                seed_source_ids=seed_source_ids,
                current=current,
            )
            if not proposals:
                break

            candidate_evaluations = []
            for proposal in proposals:
                candidate = self._evaluate_candidate(
                    seed_node_id=seed_node_id,
                    seed_chunk_ids=seed_chunk_ids,
                    seed_source_ids=seed_source_ids,
                    local_core_node_ids=set(current["local_core_node_ids"]),
                    local_core_edge_pairs=set(current["local_core_edge_pairs"]),
                    extension_node_ids=set(current["extension_node_ids"]) | {proposal["node_id"]},
                    extension_edge_pairs=set(current["extension_edge_pairs"]) | set(proposal["edge_pairs"]),
                    node_roles={**current["node_roles"], proposal["node_id"]: proposal["role"]},
                    extension_depths={
                        **current["extension_depths"],
                        proposal["node_id"]: proposal["depth"],
                    },
                    parent_extensions={
                        **current["parent_extensions"],
                        proposal["node_id"]: proposal["parent_id"],
                    },
                )
                if candidate["passes_hard_constraints"]:
                    candidate_evaluations.append(candidate)

            if not candidate_evaluations:
                break

            candidate_evaluations.sort(
                key=lambda item: (
                    item["score"],
                    -item["size_units"],
                ),
                reverse=True,
            )
            best_candidate = candidate_evaluations[0]
            if best_candidate["score"] <= current["score"] + self.min_score_improvement:
                break

            current = best_candidate
            if current["score"] > best["score"]:
                best = current
            steps += 1

        task_type, task_type_reason = self._select_task_type(best)
        rationale = self._build_rationale(
            best,
            task_type=task_type,
            task_type_reason=task_type_reason,
            steps=steps,
        )

        selected_nodes = self._node_payloads(best["node_ids"])
        selected_edges = self._edge_payloads(best["edge_pairs"])
        local_core_subgraph = self._serialize_subgraph(
            best["local_core_node_ids"], best["local_core_edge_pairs"]
        )
        extension_subgraph = self._serialize_subgraph(
            best["extension_node_ids"], best["extension_edge_pairs"]
        )

        return {
            "seed_node_id": seed_node_id,
            "seed_chunk_ids": sorted(seed_chunk_ids),
            "nodes": selected_nodes,
            "edges": selected_edges,
            "task_type": task_type,
            "task_type_reason": task_type_reason,
            "subgraph_score": round(best["score"], 4),
            "selection_rationale": rationale,
            "value_breakdown": best["value_breakdown"],
            "candidate_subgraph": {
                "node_ids": [node_id for node_id, _ in selected_nodes],
                "edge_pairs": [list(edge[:2]) for edge in selected_edges],
            },
            "local_core_subgraph": local_core_subgraph,
            "extension_subgraph": extension_subgraph,
            "node_roles": dict(sorted(best["node_roles"].items())),
        }

    def _fallback_result(
        self,
        nodes: list[tuple[str, dict]],
        edges: list[tuple[Any, Any, dict]],
        *,
        reason: str,
        seed_node_id: str = "",
        seed_chunk_ids: set[str] | None = None,
    ) -> dict:
        node_roles = {}
        if seed_node_id:
            node_roles[seed_node_id] = ANCHOR_ROLE
        return {
            "seed_node_id": seed_node_id or (nodes[0][0] if nodes else ""),
            "seed_chunk_ids": sorted(seed_chunk_ids or set()),
            "nodes": nodes,
            "edges": edges,
            "task_type": "aggregated",
            "task_type_reason": "fallback_default",
            "subgraph_score": 0.0,
            "selection_rationale": [f"fallback:{reason}"],
            "value_breakdown": {
                "answerability": 0.0,
                "visual_dependence": 0.0,
                "training_value": 0.0,
                "reasoning_richness": 0.0,
                "mismatch_penalty": 1.0,
            },
            "candidate_subgraph": {
                "node_ids": [node_id for node_id, _ in nodes],
                "edge_pairs": [list(edge[:2]) for edge in edges],
            },
            "local_core_subgraph": {"node_ids": [], "edge_pairs": []},
            "extension_subgraph": {"node_ids": [], "edge_pairs": []},
            "node_roles": node_roles,
        }

    @staticmethod
    def _select_seed_node(nodes: list[tuple[str, dict]]) -> str:
        for node_id, node_data in nodes:
            entity_type = str(node_data.get("entity_type", "")).upper()
            if entity_type in {"IMAGE", "TABLE"}:
                return node_id
            metadata = _load_metadata(node_data.get("metadata"))
            if metadata.get("img_path") or metadata.get("table_caption"):
                return node_id
        return nodes[0][0] if nodes else ""

    def _collect_seed_scope(
        self, seed_node_id: str, nodes: list[tuple[str, dict]]
    ) -> tuple[set[str], set[str]]:
        seed_chunk_ids = set()
        seed_source_ids = set()
        for node_id, node_data in nodes:
            if node_id != seed_node_id:
                continue
            metadata = _load_metadata(node_data.get("metadata"))
            seed_chunk_ids.update(_split_source_ids(metadata.get("source_trace_id", "")))
            seed_source_ids.update(_split_source_ids(node_data.get("source_id", "")))
            break
        return seed_chunk_ids, seed_source_ids

    def _build_local_core(
        self,
        *,
        seed_node_id: str,
        nodes: list[tuple[str, dict]],
        edges: list[tuple[Any, Any, dict]],
        seed_chunk_ids: set[str],
    ) -> tuple[set[str], set[tuple[str, str]]]:
        current_nodes = {seed_node_id}
        current_edges: set[tuple[str, str]] = set()

        for node_id, node_data in nodes:
            if node_id == seed_node_id:
                continue
            if self._belongs_to_seed_chunk(node_data, seed_chunk_ids):
                current_nodes.add(node_id)

        for src_id, tgt_id, edge_data in edges:
            edge_pair = _normalize_edge_pair(src_id, tgt_id)
            if self._belongs_to_seed_chunk(edge_data, seed_chunk_ids):
                current_edges.add(edge_pair)
                current_nodes.add(str(src_id))
                current_nodes.add(str(tgt_id))
                continue
            if seed_node_id in {src_id, tgt_id}:
                other_id = str(tgt_id if str(src_id) == seed_node_id else src_id)
                if other_id in current_nodes:
                    current_edges.add(edge_pair)

        for node_id in list(current_nodes):
            for neighbor_id in self.graph.get_neighbors(node_id):
                if neighbor_id not in current_nodes:
                    continue
                edge_data = self.graph.get_edge(node_id, neighbor_id) or self.graph.get_edge(
                    neighbor_id, node_id
                )
                if edge_data and self._belongs_to_seed_chunk(edge_data, seed_chunk_ids):
                    current_edges.add(_normalize_edge_pair(node_id, neighbor_id))

        return current_nodes, current_edges

    def _belongs_to_seed_chunk(self, payload: dict, seed_chunk_ids: set[str]) -> bool:
        if not seed_chunk_ids:
            return False
        source_ids = _split_source_ids(payload.get("source_id", ""))
        return bool(source_ids & seed_chunk_ids)

    def _belongs_to_seed_scope(
        self, payload: dict, seed_source_ids: set[str], seed_chunk_ids: set[str]
    ) -> bool:
        source_ids = _split_source_ids(payload.get("source_id", ""))
        if source_ids & seed_chunk_ids:
            return True
        if source_ids & seed_source_ids:
            return True
        metadata = _load_metadata(payload.get("metadata"))
        nested_source_ids = _split_source_ids(metadata.get("source_trace_id", ""))
        return bool(nested_source_ids & seed_chunk_ids)

    def _propose_extensions(
        self,
        *,
        seed_node_id: str,
        seed_chunk_ids: set[str],
        seed_source_ids: set[str],
        current: dict,
    ) -> list[dict]:
        proposals = []
        seen_nodes = set(current["node_ids"])

        if current["extension_node_ids"]:
            frontier = set(current["extension_node_ids"]) | set(current["local_core_node_ids"])
        else:
            frontier = set(current["local_core_node_ids"])

        for parent_id in frontier:
            parent_role = current["node_roles"].get(parent_id, LOCAL_CORE_ROLE)
            for neighbor_id in self.graph.get_neighbors(parent_id):
                if neighbor_id in seen_nodes:
                    continue
                node_data = self.graph.get_node(neighbor_id) or {}
                if not self._belongs_to_seed_scope(node_data, seed_source_ids, seed_chunk_ids):
                    continue
                hop_distance = self._shortest_distance_within_scope(
                    seed_node_id, neighbor_id, seed_source_ids, seed_chunk_ids
                )
                if hop_distance is None or hop_distance > self.max_hops_from_seed:
                    continue
                role = self._infer_extension_role(parent_id, neighbor_id, parent_role)
                depth = 1 if parent_role in {ANCHOR_ROLE, LOCAL_CORE_ROLE} else current["extension_depths"].get(parent_id, 1) + 1
                if depth > self.max_hops_from_seed:
                    continue
                edge_pairs = {
                    _normalize_edge_pair(parent_id, neighbor_id),
                }
                proposals.append(
                    {
                        "node_id": neighbor_id,
                        "parent_id": parent_id,
                        "role": role,
                        "depth": depth,
                        "edge_pairs": edge_pairs,
                    }
                )

        deduped = {}
        for proposal in proposals:
            node_id = proposal["node_id"]
            existing = deduped.get(node_id)
            if existing is None or proposal["depth"] < existing["depth"]:
                deduped[node_id] = proposal
        return list(deduped.values())

    def _infer_extension_role(
        self,
        parent_id: str,
        neighbor_id: str,
        parent_role: str,
    ) -> str:
        edge_data = self.graph.get_edge(parent_id, neighbor_id) or self.graph.get_edge(
            neighbor_id, parent_id
        ) or {}
        relation_type = str(edge_data.get("relation_type", "")).lower()
        description = str(edge_data.get("description", "")).lower()
        node_data = self.graph.get_node(neighbor_id) or {}
        node_description = str(node_data.get("description", "")).lower()
        text_blob = "\n".join([relation_type, description, node_description])

        if any(token in text_blob for token in ["compare", "difference", "higher", "lower", "versus"]):
            return COMPARISON_ROLE
        if any(token in text_blob for token in ["impact", "improve", "performance", "result", "lead", "therefore"]):
            return CONCLUSION_ROLE
        if any(token in text_blob for token in ["constraint", "depends", "requires", "timing", "latency", "activation"]):
            return BRIDGE_ROLE
        if parent_role in {BRIDGE_ROLE, COMPARISON_ROLE}:
            return CONCLUSION_ROLE
        return SUPPORT_ROLE

    def _shortest_distance_within_scope(
        self,
        start_id: str,
        target_id: str,
        seed_source_ids: set[str],
        seed_chunk_ids: set[str],
    ) -> int | None:
        queue = deque([(start_id, 0)])
        visited = {start_id}
        while queue:
            node_id, distance = queue.popleft()
            if node_id == target_id:
                return distance
            if distance >= self.max_hops_from_seed:
                continue
            for neighbor_id in self.graph.get_neighbors(node_id):
                if neighbor_id in visited:
                    continue
                node_data = self.graph.get_node(neighbor_id) or {}
                if not self._belongs_to_seed_scope(node_data, seed_source_ids, seed_chunk_ids):
                    continue
                visited.add(neighbor_id)
                queue.append((neighbor_id, distance + 1))
        return None

    def _collect_edges_within(self, node_ids: set[str]) -> set[tuple[str, str]]:
        edge_pairs: set[tuple[str, str]] = set()
        for node_id in node_ids:
            for neighbor_id in self.graph.get_neighbors(node_id):
                if neighbor_id in node_ids:
                    edge_pairs.add(_normalize_edge_pair(node_id, neighbor_id))
        return edge_pairs

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

    def _evaluate_candidate(
        self,
        *,
        seed_node_id: str,
        seed_chunk_ids: set[str],
        seed_source_ids: set[str],
        local_core_node_ids: set[str],
        local_core_edge_pairs: set[tuple[str, str]],
        extension_node_ids: set[str],
        extension_edge_pairs: set[tuple[str, str]],
        node_roles: dict[str, str],
        extension_depths: dict[str, int],
        parent_extensions: dict[str, str],
    ) -> dict:
        node_ids = set(local_core_node_ids) | set(extension_node_ids)
        edge_pairs = set(local_core_edge_pairs) | set(extension_edge_pairs)
        node_payloads = self._node_payloads(node_ids)
        edge_payloads = self._edge_payloads(edge_pairs)

        hard_checks = {
            "budget_valid": self._budget_valid(node_ids, edge_pairs),
            "vision_centered": self._vision_centered(
                seed_node_id, local_core_node_ids, local_core_edge_pairs, node_roles
            ),
            "answerable": self._answerable(
                seed_node_id,
                local_core_node_ids,
                extension_node_ids,
                node_roles,
                edge_pairs,
            ),
            "evidence_sufficient": self._evidence_sufficient(
                local_core_node_ids, extension_node_ids, node_payloads, edge_payloads
            ),
            "coherent": self._coherent(
                seed_node_id,
                seed_chunk_ids,
                seed_source_ids,
                local_core_node_ids,
                extension_node_ids,
                edge_payloads,
            ),
        }
        passes_hard_constraints = all(hard_checks.values())

        value_breakdown = self._score_candidate(
            seed_node_id=seed_node_id,
            local_core_node_ids=local_core_node_ids,
            extension_node_ids=extension_node_ids,
            node_roles=node_roles,
            node_payloads=node_payloads,
            edge_payloads=edge_payloads,
            hard_checks=hard_checks,
        )
        score = (
            value_breakdown["training_value"]
            + value_breakdown["visual_dependence"]
            + value_breakdown["reasoning_richness"]
            - value_breakdown["mismatch_penalty"]
        )
        if not passes_hard_constraints:
            score = -math.inf

        return {
            "node_ids": node_ids,
            "edge_pairs": edge_pairs,
            "local_core_node_ids": set(local_core_node_ids),
            "local_core_edge_pairs": set(local_core_edge_pairs),
            "extension_node_ids": set(extension_node_ids),
            "extension_edge_pairs": set(extension_edge_pairs),
            "node_roles": dict(node_roles),
            "extension_depths": dict(extension_depths),
            "parent_extensions": dict(parent_extensions),
            "size_units": len(node_ids) + len(edge_pairs),
            "passes_hard_constraints": passes_hard_constraints,
            "hard_checks": hard_checks,
            "value_breakdown": value_breakdown,
            "score": float(score),
        }

    def _budget_valid(
        self, node_ids: set[str], edge_pairs: set[tuple[str, str]]
    ) -> bool:
        return len(node_ids) + len(edge_pairs) <= self.max_units

    @staticmethod
    def _vision_centered(
        seed_node_id: str,
        local_core_node_ids: set[str],
        local_core_edge_pairs: set[tuple[str, str]],
        node_roles: dict[str, str],
    ) -> bool:
        if seed_node_id not in local_core_node_ids:
            return False
        local_nodes = {
            node_id
            for node_id, role in node_roles.items()
            if role in {ANCHOR_ROLE, LOCAL_CORE_ROLE}
        }
        if len(local_nodes) < 2:
            return False
        return any(seed_node_id in edge for edge in local_core_edge_pairs)

    @staticmethod
    def _answerable(
        seed_node_id: str,
        local_core_node_ids: set[str],
        extension_node_ids: set[str],
        node_roles: dict[str, str],
        edge_pairs: set[tuple[str, str]],
    ) -> bool:
        if len(local_core_node_ids) < 2:
            return False
        if not extension_node_ids:
            return any(seed_node_id in edge for edge in edge_pairs)

        bridge_nodes = {
            node_id
            for node_id in extension_node_ids
            if node_roles.get(node_id) == BRIDGE_ROLE
        }
        conclusion_like_nodes = {
            node_id
            for node_id in extension_node_ids
            if node_roles.get(node_id) in {CONCLUSION_ROLE, SUPPORT_ROLE, COMPARISON_ROLE}
        }
        if bridge_nodes and conclusion_like_nodes:
            return True
        if len(conclusion_like_nodes) >= 2:
            return True
        return False

    @staticmethod
    def _evidence_sufficient(
        local_core_node_ids: set[str],
        extension_node_ids: set[str],
        node_payloads: list[tuple[str, dict]],
        edge_payloads: list[tuple[str, str, dict]],
    ) -> bool:
        local_node_evidence = 0
        extension_evidence = 0
        edge_evidence = 0
        for node_id, node_data in node_payloads:
            if str(node_data.get("evidence_span", "")).strip():
                if node_id in local_core_node_ids:
                    local_node_evidence += 1
                elif node_id in extension_node_ids:
                    extension_evidence += 1
        for _, _, edge_data in edge_payloads:
            if str(edge_data.get("evidence_span", "")).strip():
                edge_evidence += 1
        return local_node_evidence >= 1 and edge_evidence >= 1 and (
            extension_evidence >= 1 or len(extension_node_ids) == 0
        )

    def _coherent(
        self,
        seed_node_id: str,
        seed_chunk_ids: set[str],
        seed_source_ids: set[str],
        local_core_node_ids: set[str],
        extension_node_ids: set[str],
        edge_payloads: list[tuple[str, str, dict]],
    ) -> bool:
        if not local_core_node_ids:
            return False
        for node_id in local_core_node_ids:
            node_data = self.graph.get_node(node_id) or {}
            if node_id == seed_node_id:
                continue
            if not self._belongs_to_seed_chunk(node_data, seed_chunk_ids):
                return False

        if not extension_node_ids:
            return True

        extension_supported = False
        for src_id, tgt_id, edge_data in edge_payloads:
            edge_pair_nodes = {src_id, tgt_id}
            if edge_pair_nodes & extension_node_ids and edge_pair_nodes & local_core_node_ids:
                if self._belongs_to_seed_scope(edge_data, seed_source_ids, seed_chunk_ids):
                    extension_supported = True
        return extension_supported

    def _score_candidate(
        self,
        *,
        seed_node_id: str,
        local_core_node_ids: set[str],
        extension_node_ids: set[str],
        node_roles: dict[str, str],
        node_payloads: list[tuple[str, dict]],
        edge_payloads: list[tuple[str, str, dict]],
        hard_checks: dict[str, bool],
    ) -> dict[str, float]:
        text_blob = "\n".join(
            [str(node_data.get("description", "")) for _, node_data in node_payloads]
            + [str(edge_data.get("description", "")) for _, _, edge_data in edge_payloads]
            + [str(edge_data.get("relation_type", "")) for _, _, edge_data in edge_payloads]
        ).lower()

        training_value = 0.0
        if any(token in text_blob for token in ["latency", "bandwidth", "capacity", "frequency", "voltage", "power", "timing", "trcd", "trp", "tras", "cas", "channel", "bank", "rank"]):
            training_value += 1.0
        if any(token in text_blob for token in ["compare", "difference", "higher", "lower", "faster", "slower", "trade"]):
            training_value += 1.0
        if any(token in text_blob for token in ["constraint", "depends", "requires", "before", "after", "limit"]):
            training_value += 1.0
        if any(token in text_blob for token in ["topology", "architecture", "organization", "maps", "layout"]):
            training_value += 1.0
        if re.search(r"\b\d+(?:\.\d+)?\b", text_blob):
            training_value += 1.0

        role_bonus = {
            BRIDGE_ROLE: 0.8,
            SUPPORT_ROLE: 0.4,
            COMPARISON_ROLE: 0.8,
            CONCLUSION_ROLE: 0.9,
        }
        for node_id in extension_node_ids:
            training_value += role_bonus.get(node_roles.get(node_id, ""), 0.0)

        visual_edges = sum(
            1 for src_id, tgt_id, _ in edge_payloads if seed_node_id in {src_id, tgt_id}
        )
        local_core_size = max(1, len(local_core_node_ids))
        visual_dependence = 1.0 + min(2.0, visual_edges * 0.4) + min(1.0, local_core_size * 0.25)

        reasoning_richness = 0.0
        bridge_count = sum(1 for node_id in extension_node_ids if node_roles.get(node_id) == BRIDGE_ROLE)
        conclusion_count = sum(1 for node_id in extension_node_ids if node_roles.get(node_id) == CONCLUSION_ROLE)
        comparison_count = sum(1 for node_id in extension_node_ids if node_roles.get(node_id) == COMPARISON_ROLE)
        support_count = sum(1 for node_id in extension_node_ids if node_roles.get(node_id) == SUPPORT_ROLE)
        reasoning_richness += min(1.5, bridge_count * 0.8)
        reasoning_richness += min(1.5, conclusion_count * 0.7)
        reasoning_richness += min(1.0, comparison_count * 0.6)
        reasoning_richness += min(0.8, support_count * 0.3)
        if bridge_count >= 1 and conclusion_count >= 1:
            reasoning_richness += 0.8
        if comparison_count >= 2:
            reasoning_richness += 0.6

        mismatch_penalty = 0.0
        missing_evidence = sum(
            1 for _, _, edge_data in edge_payloads if not str(edge_data.get("evidence_span", "")).strip()
        )
        mismatch_penalty += min(1.5, missing_evidence * 0.2)
        generic_relations = sum(
            1
            for _, _, edge_data in edge_payloads
            if str(edge_data.get("relation_type", "")).strip().lower() in {"", "related_to"}
        )
        mismatch_penalty += min(1.0, generic_relations * 0.2)

        extension_size = len(extension_node_ids)
        local_size = max(1, len(local_core_node_ids))
        if extension_size > local_size:
            mismatch_penalty += (extension_size - local_size) * 0.35
        if len(node_payloads) + len(edge_payloads) > max(4, self.max_units // 2):
            mismatch_penalty += 0.05 * (
                len(node_payloads) + len(edge_payloads) - max(4, self.max_units // 2)
            )
        if not hard_checks["answerable"]:
            mismatch_penalty += 2.0

        answerability = 1.0 if hard_checks["answerable"] else 0.0
        if hard_checks["evidence_sufficient"]:
            answerability += 0.5
        if hard_checks["coherent"]:
            answerability += 0.5

        return {
            "answerability": round(answerability, 4),
            "visual_dependence": round(visual_dependence, 4),
            "training_value": round(training_value, 4),
            "reasoning_richness": round(reasoning_richness, 4),
            "mismatch_penalty": round(mismatch_penalty, 4),
        }

    def _select_task_type(self, candidate: dict) -> tuple[str, str]:
        bridge_nodes = [
            node_id
            for node_id in candidate["extension_node_ids"]
            if candidate["node_roles"].get(node_id) == BRIDGE_ROLE
        ]
        conclusion_nodes = [
            node_id
            for node_id in candidate["extension_node_ids"]
            if candidate["node_roles"].get(node_id) in {CONCLUSION_ROLE, SUPPORT_ROLE}
        ]
        comparison_nodes = [
            node_id
            for node_id in candidate["extension_node_ids"]
            if candidate["node_roles"].get(node_id) == COMPARISON_ROLE
        ]

        if bridge_nodes and conclusion_nodes:
            return "multi_hop", "local_core_bridge_conclusion_chain"
        if len(comparison_nodes) >= 2 or len(candidate["extension_node_ids"]) >= 2:
            return "aggregated", "multiple_theme_extensions_around_local_core"
        return "aggregated", "default_theme_aggregation"

    def _build_rationale(
        self,
        candidate: dict,
        *,
        task_type: str,
        task_type_reason: str,
        steps: int,
    ) -> list[str]:
        rationale = [
            f"task_type={task_type}",
            f"task_type_reason={task_type_reason}",
            f"search_steps={steps}",
            f"size_units={candidate['size_units']}",
            f"local_core_nodes={len(candidate['local_core_node_ids'])}",
            f"extension_nodes={len(candidate['extension_node_ids'])}",
            f"training_value={candidate['value_breakdown']['training_value']}",
            f"visual_dependence={candidate['value_breakdown']['visual_dependence']}",
            f"reasoning_richness={candidate['value_breakdown']['reasoning_richness']}",
        ]
        for check_name, passed in candidate["hard_checks"].items():
            rationale.append(f"{check_name}={str(passed).lower()}")
        return rationale

    def _serialize_subgraph(
        self, node_ids: set[str], edge_pairs: set[tuple[str, str]]
    ) -> dict:
        return {
            "node_ids": sorted(node_ids),
            "edge_pairs": [list(edge_pair) for edge_pair in sorted(edge_pairs)],
        }
