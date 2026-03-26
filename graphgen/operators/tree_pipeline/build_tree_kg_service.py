from typing import Tuple

from graphgen.bases import BaseGraphStorage, BaseLLMWrapper, BaseOperator
from graphgen.bases.datatypes import Chunk
from graphgen.common.init_llm import init_llm
from graphgen.common.init_storage import init_storage
from graphgen.utils import evidence_supported_by_text, logger

from graphgen.operators.build_kg.build_mm_kg import build_mm_kg
from graphgen.operators.build_kg.build_text_kg import build_text_kg


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class BuildTreeKGService(BaseOperator):
    """Build KG from tree-aware chunks while keeping output format compatible."""

    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        graph_backend: str = "kuzu",
        **build_kwargs,
    ):
        super().__init__(
            working_dir=working_dir,
            kv_backend=kv_backend,
            op_name="build_tree_kg",
        )
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.graph_storage: BaseGraphStorage = init_storage(
            backend=graph_backend,
            working_dir=working_dir,
            namespace="graph",
        )
        self.build_kwargs = build_kwargs
        self.max_loop: int = int(self.build_kwargs.get("max_loop", 3))
        self.relation_confidence_threshold: float = float(
            self.build_kwargs.get("relation_confidence_threshold", 0.5)
        )
        self.require_entity_evidence: bool = _to_bool(
            self.build_kwargs.get("require_entity_evidence", False)
        )
        self.require_relation_evidence: bool = _to_bool(
            self.build_kwargs.get("require_relation_evidence", True)
        )
        self.validate_evidence_in_source: bool = _to_bool(
            self.build_kwargs.get("validate_evidence_in_source", False)
        )
        self.text_require_entity_evidence: bool = _to_bool(
            self.build_kwargs.get(
                "text_require_entity_evidence", self.require_entity_evidence
            )
        )
        self.mm_require_entity_evidence: bool = _to_bool(
            self.build_kwargs.get("mm_require_entity_evidence", False)
        )
        self.text_require_relation_evidence: bool = _to_bool(
            self.build_kwargs.get(
                "text_require_relation_evidence", self.require_relation_evidence
            )
        )
        self.mm_require_relation_evidence: bool = _to_bool(
            self.build_kwargs.get(
                "mm_require_relation_evidence", self.require_relation_evidence
            )
        )
        self.text_validate_evidence_in_source: bool = _to_bool(
            self.build_kwargs.get(
                "text_validate_evidence_in_source", self.validate_evidence_in_source
            )
        )
        self.mm_validate_evidence_in_source: bool = _to_bool(
            self.build_kwargs.get(
                "mm_validate_evidence_in_source", self.validate_evidence_in_source
            )
        )
        self.filter_text_entities_without_source_evidence: bool = _to_bool(
            self.build_kwargs.get("filter_text_entities_without_source_evidence", False)
        )

    @staticmethod
    def _inject_tree_context(chunk: Chunk) -> Chunk:
        metadata = dict(chunk.metadata or {})
        path = metadata.get("path")
        if not path:
            return chunk

        contextual_content = f"[Document Path]\n{path}\n\n[Chunk]\n{chunk.content}"
        return Chunk(
            id=chunk.id,
            content=contextual_content,
            type=chunk.type,
            metadata=metadata,
        )

    def process(self, batch: list) -> Tuple[list, dict]:
        chunks = [Chunk.from_dict(doc["_trace_id"], doc) for doc in batch]
        text_chunk_content_by_id = {
            chunk.id: chunk.content for chunk in chunks if chunk.type == "text"
        }
        text_chunks = [self._inject_tree_context(chunk) for chunk in chunks if chunk.type == "text"]
        mm_chunks = [
            chunk
            for chunk in chunks
            if chunk.type in ("image", "video", "table", "formula")
        ]

        nodes = []
        edges = []

        if text_chunks:
            logger.info("[Tree Text Entity and Relation Extraction] processing ...")
            text_nodes, text_edges = build_text_kg(
                llm_client=self.llm_client,
                kg_instance=self.graph_storage,
                chunks=text_chunks,
                max_loop=self.max_loop,
                relation_confidence_threshold=self.relation_confidence_threshold,
                require_entity_evidence=self.text_require_entity_evidence,
                require_relation_evidence=self.text_require_relation_evidence,
                validate_evidence_in_source=self.text_validate_evidence_in_source,
            )
            if self.filter_text_entities_without_source_evidence:
                text_nodes, text_edges = self._filter_text_entities_by_source_evidence(
                    text_nodes=text_nodes,
                    text_edges=text_edges,
                    text_chunk_content_by_id=text_chunk_content_by_id,
                )
            nodes += text_nodes
            edges += text_edges
        else:
            logger.info("All tree text chunks are already in the storage")

        if mm_chunks:
            logger.info("[Tree Multi-modal Entity and Relation Extraction] processing ...")
            mm_nodes, mm_edges = build_mm_kg(
                llm_client=self.llm_client,
                kg_instance=self.graph_storage,
                chunks=mm_chunks,
                relation_confidence_threshold=self.relation_confidence_threshold,
                require_entity_evidence=self.mm_require_entity_evidence,
                require_relation_evidence=self.mm_require_relation_evidence,
                validate_evidence_in_source=self.mm_validate_evidence_in_source,
            )
            nodes += mm_nodes
            edges += mm_edges
        else:
            logger.info("All tree multi-modal chunks are already in the storage")

        self.graph_storage.index_done_callback()
        meta_updates = {}
        results = []

        for node in nodes:
            if not node:
                continue
            trace_id = node["entity_name"]
            results.append({"_trace_id": trace_id, "node": node, "edge": {}})
            for source_id in node.get("source_id", "").split("<SEP>"):
                meta_updates.setdefault(source_id, []).append(trace_id)

        for edge in edges:
            if not edge:
                continue
            trace_id = str(frozenset((edge["src_id"], edge["tgt_id"])))
            results.append({"_trace_id": trace_id, "node": {}, "edge": edge})
            for source_id in edge.get("source_id", "").split("<SEP>"):
                meta_updates.setdefault(source_id, []).append(trace_id)

        return results, meta_updates

    @staticmethod
    def _filter_text_entities_by_source_evidence(
        text_nodes: list,
        text_edges: list,
        text_chunk_content_by_id: dict,
    ) -> Tuple[list, list]:
        kept_text_nodes = []
        removed_entity_names = set()

        for node in text_nodes:
            source_ids = [
                source_id.strip()
                for source_id in str(node.get("source_id", "")).split("<SEP>")
                if source_id.strip()
            ]
            source_texts = [
                text_chunk_content_by_id[source_id]
                for source_id in source_ids
                if source_id in text_chunk_content_by_id
            ]
            if not source_texts:
                kept_text_nodes.append(node)
                continue

            evidence_spans = [
                evidence_span.strip()
                for evidence_span in str(node.get("evidence_span", "")).split("<SEP>")
                if evidence_span.strip()
            ]
            if not evidence_spans:
                kept_text_nodes.append(node)
                continue

            has_supported_evidence = any(
                evidence_supported_by_text(evidence_span, source_text)
                for evidence_span in evidence_spans
                for source_text in source_texts
            )
            if has_supported_evidence:
                kept_text_nodes.append(node)
                continue

            removed_entity_names.add(node.get("entity_name"))

        kept_text_edges = [
            edge
            for edge in text_edges
            if edge.get("src_id") not in removed_entity_names
            and edge.get("tgt_id") not in removed_entity_names
        ]

        return kept_text_nodes, kept_text_edges
