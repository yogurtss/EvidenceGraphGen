from typing import Tuple

from graphgen.bases import BaseOperator
from graphgen.utils import evidence_supported_by_text


class FilterEntitiesService(BaseOperator):
    """Filter unsupported entity nodes and dangling edges from KG results."""

    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        source_sep: str = "<SEP>",
    ):
        super().__init__(
            working_dir=working_dir,
            kv_backend=kv_backend,
            op_name="filter_entities",
        )
        self.source_sep = source_sep

    def process(self, batch: list) -> Tuple[list, dict]:
        text_chunk_content_by_id = {
            item.get("_trace_id"): item.get("content", "")
            for item in batch
            if item.get("type") == "text" and item.get("_trace_id")
        }

        kg_records = [
            item
            for item in batch
            if isinstance(item.get("node"), dict) or isinstance(item.get("edge"), dict)
        ]

        nodes = [item for item in kg_records if item.get("node")]
        edges = [item for item in kg_records if item.get("edge")]

        kept_nodes = []
        removed_entity_names = set()

        for item in nodes:
            node = item["node"]
            if self._node_supported_by_source(node, text_chunk_content_by_id):
                kept_nodes.append(item)
                continue
            removed_entity_names.add(node.get("entity_name"))

        kept_edges = [
            item
            for item in edges
            if item["edge"].get("src_id") not in removed_entity_names
            and item["edge"].get("tgt_id") not in removed_entity_names
        ]

        return kept_nodes + kept_edges, {}

    def _node_supported_by_source(
        self, node: dict, text_chunk_content_by_id: dict
    ) -> bool:
        source_ids = self._split_field(node.get("source_id"))
        evidence_spans = self._split_field(node.get("evidence_span"))

        if not source_ids or not evidence_spans:
            return True

        source_texts = [
            text_chunk_content_by_id[source_id]
            for source_id in source_ids
            if source_id in text_chunk_content_by_id
        ]
        if not source_texts:
            return True

        return any(
            evidence_supported_by_text(evidence_span, source_text)
            for evidence_span in evidence_spans
            for source_text in source_texts
        )

    def _split_field(self, value) -> list:
        return [
            part.strip()
            for part in str(value or "").split(self.source_sep)
            if part and part.strip()
        ]
