from typing import Tuple

from graphgen.bases import BaseGraphStorage, BaseKVStorage, BaseOperator
from graphgen.common.init_storage import init_storage
from graphgen.utils import evidence_supported_by_text


class FilterEntitiesService(BaseOperator):
    """Filter unsupported entity nodes and dangling edges from KG results."""

    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        graph_backend: str = "kuzu",
        source_sep: str = "<SEP>",
        source_namespace: str = "tree_chunk",
    ):
        super().__init__(
            working_dir=working_dir,
            kv_backend=kv_backend,
            op_name="filter_entities",
        )
        self.graph_storage: BaseGraphStorage = init_storage(
            backend=graph_backend,
            working_dir=working_dir,
            namespace="graph",
        )
        self.source_storage: BaseKVStorage = init_storage(
            backend=kv_backend,
            working_dir=working_dir,
            namespace=source_namespace,
        )
        self.source_sep = source_sep

    def process(self, batch: list) -> Tuple[list, dict]:
        # aggregate operator: ignore per-batch rows and always load full storages
        self.graph_storage.reload()
        self.source_storage.reload()

        text_chunk_content_by_id = {}
        for trace_id, item in (self.source_storage.get_all() or {}).items():
            if not isinstance(item, dict):
                continue
            if item.get("type") != "text":
                continue
            chunk_id = item.get("_trace_id") or trace_id
            if chunk_id:
                text_chunk_content_by_id[chunk_id] = item.get("content", "")

        nodes = [
            {"_trace_id": node_id, "node": node, "edge": {}}
            for node_id, node in (self.graph_storage.get_all_nodes() or [])
        ]
        edges = [
            {
                "_trace_id": str(frozenset((src_id, tgt_id))),
                "node": {},
                "edge": edge,
            }
            for src_id, tgt_id, edge in (self.graph_storage.get_all_edges() or [])
        ]

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

    def split(self, batch):
        """
        Aggregate operators must evaluate the full upstream dataset in one pass.
        Bypass BaseOperator cache-based splitting so process() always executes.
        """
        import pandas as pd

        return batch, pd.DataFrame()

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
