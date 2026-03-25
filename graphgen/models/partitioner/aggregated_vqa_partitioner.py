from __future__ import annotations

import json
import random
from collections import deque
from typing import Any, Iterable, List, Set, Tuple

from graphgen.bases import BaseGraphStorage
from graphgen.bases.datatypes import Community

from .anchor_bfs_partitioner import AnchorBFSPartitioner

NODE_UNIT: str = "n"
EDGE_UNIT: str = "e"


class AggregatedVQAPartitioner(AnchorBFSPartitioner):
    """Image-centric anchor partitioner for aggregated VQA communities.

    Compared with ``AnchorBFSPartitioner`` this implementation adds:
    - optional section/path scope around each anchor image
    - required modality checks (e.g. image + text)
    - a minimum unit constraint so sampled communities are more aggregated
    """

    def __init__(
        self,
        *,
        anchor_type: str | List[str] = "image",
        anchor_ids: Set[str] | None = None,
    ) -> None:
        super().__init__(anchor_type=anchor_type, anchor_ids=anchor_ids)

    def partition(
        self,
        g: BaseGraphStorage,
        max_units_per_community: int = 10,
        min_units_per_community: int = 5,
        section_scoped: bool = True,
        required_modalities: List[str] | None = None,
        **kwargs: Any,
    ) -> Iterable[Community]:
        del kwargs
        required_modalities = [
            str(item).strip().lower() for item in (required_modalities or ["image", "text"])
        ]

        nodes = g.get_all_nodes()
        anchors: Set[str] = self._pick_anchor_ids(nodes)
        if not anchors:
            return

        used_n: set[str] = set()
        used_e: set[frozenset[str]] = set()

        seeds = list(anchors)
        random.shuffle(seeds)

        for seed_node in seeds:
            if seed_node in used_n:
                continue

            comm_n, comm_e, touched_n, touched_e = self._grow_scoped_community(
                seed=seed_node,
                g=g,
                max_units=max_units_per_community,
                section_scoped=section_scoped,
            )
            if not comm_n and not comm_e:
                continue

            if len(comm_n) + len(comm_e) < int(min_units_per_community):
                continue

            if not self._satisfy_required_modalities(
                comm_n, g=g, required_modalities=required_modalities
            ):
                continue

            used_n.update(touched_n)
            used_e.update(touched_e)
            yield Community(id=seed_node, nodes=comm_n, edges=comm_e)

    @staticmethod
    def _extract_metadata(node_meta: dict) -> dict:
        raw_metadata = node_meta.get("metadata")
        if isinstance(raw_metadata, dict):
            nested = raw_metadata.get("metadata")
            if isinstance(nested, dict):
                return {**raw_metadata, **nested}
            return raw_metadata
        if isinstance(raw_metadata, str):
            try:
                parsed = json.loads(raw_metadata)
            except (TypeError, json.JSONDecodeError):
                return {}
            if isinstance(parsed, dict):
                nested = parsed.get("metadata")
                if isinstance(nested, dict):
                    return {**parsed, **nested}
                return parsed
        return {}

    @classmethod
    def _get_section_path(cls, node_meta: dict) -> str:
        metadata = cls._extract_metadata(node_meta)
        path = metadata.get("path") or node_meta.get("path")
        if not path:
            return ""
        return str(path).strip()

    @staticmethod
    def _normalize_modality(node_meta: dict) -> str:
        entity_type = str(node_meta.get("entity_type", "")).lower()
        if "image" in entity_type:
            return "image"
        if "table" in entity_type:
            return "table"
        if "video" in entity_type:
            return "video"
        if entity_type:
            return "text"

        metadata = node_meta.get("metadata")
        if isinstance(metadata, dict):
            modality = str(metadata.get("modality", "")).lower()
            if modality:
                return modality
        return "text"

    @classmethod
    def _satisfy_required_modalities(
        cls,
        node_ids: list[str],
        g: BaseGraphStorage,
        required_modalities: list[str],
    ) -> bool:
        modalities = {cls._normalize_modality(g.get_node(node_id) or {}) for node_id in node_ids}
        return all(modality in modalities for modality in required_modalities)

    @classmethod
    def _grow_scoped_community(
        cls,
        seed: str,
        g: BaseGraphStorage,
        max_units: int,
        section_scoped: bool,
    ) -> Tuple[List[str], List[Tuple[str, str]], Set[str], Set[frozenset[str]]]:
        comm_n: List[str] = []
        comm_e: List[Tuple[str, str]] = []
        queue: deque[tuple[str, Any]] = deque([(NODE_UNIT, seed)])
        cnt = 0

        touched_n: set[str] = set()
        touched_e: set[frozenset[str]] = set()

        seed_meta = g.get_node(seed) or {}
        seed_section = cls._get_section_path(seed_meta)

        while queue and cnt < max_units:
            k, it = queue.popleft()

            if k == NODE_UNIT:
                if it in touched_n:
                    continue
                node_meta = g.get_node(it) or {}
                if section_scoped and seed_section:
                    node_section = cls._get_section_path(node_meta)
                    if node_section and node_section != seed_section and it != seed:
                        continue

                touched_n.add(it)
                comm_n.append(it)
                cnt += 1
                for nei in g.get_neighbors(it):
                    e_key = frozenset((it, nei))
                    if e_key not in touched_e:
                        queue.append((EDGE_UNIT, e_key))
            else:
                if it in touched_e:
                    continue
                u, v = tuple(it)
                if section_scoped and seed_section:
                    u_meta = g.get_node(u) or {}
                    v_meta = g.get_node(v) or {}
                    u_section = cls._get_section_path(u_meta)
                    v_section = cls._get_section_path(v_meta)
                    valid_u = (not u_section) or (u_section == seed_section)
                    valid_v = (not v_section) or (v_section == seed_section)
                    if not (valid_u and valid_v):
                        continue

                touched_e.add(it)
                comm_e.append((u, v))
                cnt += 1
                for n in it:
                    if n not in touched_n:
                        queue.append((NODE_UNIT, n))

        return comm_n, comm_e, touched_n, touched_e
