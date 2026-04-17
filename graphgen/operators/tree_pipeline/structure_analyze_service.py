from pathlib import Path
from typing import Tuple

from graphgen.bases import BaseOperator

from .tree_utils import merge_metadata, normalize_components


class StructureAnalyzeService(BaseOperator):
    """Convert flat document records into a lightweight component-pack representation."""

    def __init__(self, working_dir: str = "cache", kv_backend: str = "rocksdb"):
        super().__init__(
            working_dir=working_dir,
            kv_backend=kv_backend,
            op_name="structure_analyze",
        )

    def process(self, batch: list) -> Tuple[list, dict]:
        results = []
        meta_updates = {}

        for doc in batch:
            source_trace_id = doc["_trace_id"]
            source_path = doc.get("source_path") or doc.get("path") or ""
            source_file = doc.get("source_file") or (
                Path(str(source_path)).name if source_path else ""
            )
            components = normalize_components(doc)

            result = {
                "type": "component_pack",
                "source_trace_id": source_trace_id,
                "components": components,
                "metadata": merge_metadata(
                    doc,
                    {
                        "source_type": doc.get("type", "text"),
                        "source_path": str(source_path) if source_path else "",
                        "source_file": source_file,
                    },
                ),
            }
            result["_trace_id"] = self.get_trace_id(result)
            results.append(result)
            meta_updates.setdefault(source_trace_id, []).append(result["_trace_id"])

        return results, meta_updates
