import json
from typing import Any


def _compact_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _load_metadata(raw_metadata: Any) -> dict:
    if isinstance(raw_metadata, dict):
        return raw_metadata
    if not raw_metadata:
        return {}
    try:
        parsed = json.loads(raw_metadata)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _compact_metadata_text(value: Any) -> str:
    if isinstance(value, list):
        return _compact_field("\n".join(str(item) for item in value if item))
    return _compact_field(value)


def _is_image_node(node_data: dict) -> bool:
    entity_type = str(node_data.get("entity_type", "")).upper()
    if "IMAGE" in entity_type:
        return True
    metadata = _load_metadata(node_data.get("metadata"))
    return any(
        metadata.get(key)
        for key in ("image_path", "img_path", "image_caption", "note_text")
    )


def _visual_metadata_lines(node_data: dict) -> list[str]:
    if not _is_image_node(node_data):
        return []
    metadata = _load_metadata(node_data.get("metadata"))
    caption = _compact_metadata_text(
        metadata.get("image_caption") or metadata.get("caption")
    )
    notes = _compact_metadata_text(metadata.get("note_text") or metadata.get("notes"))
    lines = []
    if caption:
        lines.append(f"   Image Caption: {caption}")
    if notes:
        lines.append(f"   Notes: {notes}")
    return lines


def format_node_context(
    index: int,
    node: tuple[str, dict],
    *,
    include_visual_metadata: bool = False,
) -> str:
    node_id, node_data = node
    description = _compact_field(node_data.get("description", ""))
    evidence = _compact_field(node_data.get("evidence_span", ""))

    parts = [f"{index}. {node_id}: {description}"]
    if evidence:
        parts.append(f"   Evidence: {evidence}")
    if include_visual_metadata:
        parts.extend(_visual_metadata_lines(node_data))
    return "\n".join(parts)


def format_edge_context(index: int, edge: tuple[Any, Any, dict]) -> str:
    src_id, tgt_id, edge_data = edge
    description = _compact_field(edge_data.get("description", ""))
    relation_type = _compact_field(edge_data.get("relation_type", ""))
    evidence = _compact_field(edge_data.get("evidence_span", ""))

    relation_label = f" [{relation_type}]" if relation_type else ""
    parts = [f"{index}. {src_id} -- {tgt_id}{relation_label}: {description}"]
    if evidence:
        parts.append(f"   Evidence: {evidence}")
    return "\n".join(parts)


def build_grounded_context(
    batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]],
    *,
    include_visual_metadata: bool = False,
) -> tuple[str, str]:
    nodes, edges = batch
    entities_str = "\n".join(
        format_node_context(
            index + 1,
            node,
            include_visual_metadata=include_visual_metadata,
        )
        for index, node in enumerate(nodes)
    )
    relationships_str = "\n".join(
        format_edge_context(index + 1, edge) for index, edge in enumerate(edges)
    )
    return entities_str, relationships_str
