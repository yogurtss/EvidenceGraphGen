import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from graphgen.utils import logger, split_string_by_multi_markers


def split_source_ids(value: Any) -> list[str]:
    if not value:
        return []
    return split_string_by_multi_markers(str(value), ["<SEP>"])


def load_metadata(raw_metadata: Any) -> dict:
    if isinstance(raw_metadata, dict):
        return raw_metadata
    if not raw_metadata:
        return {}
    try:
        parsed = json.loads(raw_metadata)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _is_visual_node(node_data: dict) -> bool:
    entity_type = str(node_data.get("entity_type", "")).upper()
    if any(tag in entity_type for tag in ("IMAGE", "TABLE", "FORMULA")):
        return True
    metadata = load_metadata(node_data.get("metadata"))
    return any(
        metadata.get(key)
        for key in ("image_path", "img_path", "table_img_path", "equation_img_path")
    )


def _coerce_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value).strip()


def _metadata_marker(metadata: dict, *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value:
            return str(value)
    return ""


def _source_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    name = Path(text).name
    if "." in name:
        stem = Path(name).stem
        if stem:
            return stem
    return name


@dataclass(frozen=True)
class SourceChunkRecord:
    chunk_id: str
    content: str
    source_trace_id: str = ""
    source_path: str = ""
    source_name: str = ""


@dataclass(frozen=True)
class _CandidateSourceId:
    chunk_id: str
    origin_rank: int
    order: int


class SourceChunkContextBuilder:
    def __init__(
        self,
        chunk_storages: Sequence[Any] | None,
        *,
        chunks_per_entity: int = 3,
    ) -> None:
        self.chunk_storages = list(chunk_storages or [])
        self.chunks_per_entity = max(1, int(chunks_per_entity or 3))

    def build(
        self,
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]],
    ) -> str:
        nodes, edges = batch
        if not self.chunk_storages:
            return ""

        visual_trace_ids, visual_paths, visual_names = self._collect_visual_markers(
            nodes
        )
        formatted_blocks = []
        for entity_index, (node_id, node_data) in enumerate(nodes, start=1):
            candidates = self._candidate_source_ids_for_entity(
                node_id, node_data, edges
            )
            records = self._select_records(
                candidates,
                visual_trace_ids=visual_trace_ids,
                visual_paths=visual_paths,
                visual_names=visual_names,
            )
            if not records:
                continue
            formatted_blocks.append(
                self._format_entity_records(entity_index, node_id, records)
            )
        return "\n\n".join(formatted_blocks)

    def _collect_visual_markers(
        self,
        nodes: list[tuple[str, dict]],
    ) -> tuple[set[str], set[str], set[str]]:
        trace_ids: set[str] = set()
        paths: set[str] = set()
        names: set[str] = set()

        for _, node_data in nodes:
            if not _is_visual_node(node_data):
                continue
            metadata = load_metadata(node_data.get("metadata"))
            self._add_markers_from_metadata(metadata, trace_ids, paths, names)
            for chunk_id in split_source_ids(node_data.get("source_id", "")):
                record = self._fetch_record(chunk_id)
                if record:
                    if record.source_trace_id:
                        trace_ids.add(record.source_trace_id)
                    if record.source_path:
                        paths.add(record.source_path)
                    if record.source_name:
                        names.add(record.source_name)
        return trace_ids, paths, names

    @staticmethod
    def _add_markers_from_metadata(
        metadata: dict,
        trace_ids: set[str],
        paths: set[str],
        names: set[str],
    ) -> None:
        trace_id = _metadata_marker(metadata, "source_trace_id")
        source_path = _metadata_marker(metadata, "source_path", "path")
        source_name = _source_name(
            _metadata_marker(
                metadata,
                "source_name",
                "source_file",
                "source_path",
                "path",
            )
        )
        if trace_id:
            trace_ids.add(trace_id)
        if source_path:
            paths.add(source_path)
        if source_name:
            names.add(source_name)

    @staticmethod
    def _candidate_source_ids_for_entity(
        node_id: str,
        node_data: dict,
        edges: list[tuple[Any, Any, dict]],
    ) -> list[_CandidateSourceId]:
        raw_candidates: list[tuple[str, int]] = [
            (chunk_id, 0)
            for chunk_id in split_source_ids(node_data.get("source_id", ""))
        ]
        for src_id, tgt_id, edge_data in edges:
            if str(src_id) != str(node_id) and str(tgt_id) != str(node_id):
                continue
            raw_candidates.extend(
                (chunk_id, 1)
                for chunk_id in split_source_ids(edge_data.get("source_id", ""))
            )

        candidates = []
        seen = set()
        for order, (chunk_id, origin_rank) in enumerate(raw_candidates):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            candidates.append(
                _CandidateSourceId(
                    chunk_id=chunk_id,
                    origin_rank=origin_rank,
                    order=order,
                )
            )
        return candidates

    def _select_records(
        self,
        candidates: Iterable[_CandidateSourceId],
        *,
        visual_trace_ids: set[str],
        visual_paths: set[str],
        visual_names: set[str],
    ) -> list[SourceChunkRecord]:
        candidate_records: list[tuple[tuple[int, int, int], SourceChunkRecord]] = []
        for candidate in candidates:
            record = self._fetch_record(candidate.chunk_id)
            if not record:
                continue
            priority = 2
            if record.source_trace_id and record.source_trace_id in visual_trace_ids:
                priority = 0
            elif (
                (record.source_path and record.source_path in visual_paths)
                or (record.source_name and record.source_name in visual_names)
            ):
                priority = 1
            candidate_records.append(
                ((priority, candidate.origin_rank, candidate.order), record)
            )

        candidate_records.sort(key=lambda item: item[0])
        return [record for _, record in candidate_records[: self.chunks_per_entity]]

    def _fetch_record(self, chunk_id: str) -> SourceChunkRecord | None:
        for storage in self.chunk_storages:
            try:
                item = storage.get_by_id(chunk_id)
            except Exception as exc:  # pragma: no cover - defensive storage adapter guard
                logger.debug("Failed to fetch source chunk %s: %s", chunk_id, exc)
                continue
            if not isinstance(item, dict):
                continue
            content = _coerce_content(item.get("content"))
            if not content:
                continue
            metadata = load_metadata(item.get("metadata"))
            source_path = _metadata_marker(metadata, "source_path", "path")
            source_name = _source_name(
                _metadata_marker(
                    metadata,
                    "source_name",
                    "source_file",
                    "source_path",
                    "path",
                )
            )
            return SourceChunkRecord(
                chunk_id=chunk_id,
                content=content,
                source_trace_id=_metadata_marker(metadata, "source_trace_id"),
                source_path=source_path,
                source_name=source_name,
            )
        logger.debug("Source chunk %s was not found in configured namespaces", chunk_id)
        return None

    @staticmethod
    def _format_entity_records(
        entity_index: int,
        node_id: str,
        records: list[SourceChunkRecord],
    ) -> str:
        lines = [f"Entity {entity_index}: {node_id}"]
        for record_index, record in enumerate(records, start=1):
            if record.source_name:
                lines.append(f"[{record_index}] Source: {record.source_name}")
            else:
                lines.append(f"[{record_index}] Source")
            lines.append(record.content)
        return "\n".join(lines)
