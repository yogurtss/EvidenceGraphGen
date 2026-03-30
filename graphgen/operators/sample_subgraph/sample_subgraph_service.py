from typing import Tuple

from graphgen.bases import BaseGraphStorage, BaseOperator
from graphgen.common.init_storage import init_storage
from graphgen.models import ValueAwareSubgraphSampler


class SampleSubgraphService(BaseOperator):
    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        graph_backend: str = "kuzu",
        max_units: int = 10,
        max_steps: int = 6,
        max_hops_from_seed: int = 4,
        min_score_improvement: float = 0.2,
    ):
        super().__init__(
            working_dir=working_dir,
            kv_backend=kv_backend,
            op_name="sample_subgraph",
        )
        self.graph_storage: BaseGraphStorage = init_storage(
            backend=graph_backend,
            working_dir=working_dir,
            namespace="graph",
        )
        self.sampler = ValueAwareSubgraphSampler(
            self.graph_storage,
            max_units=max_units,
            max_steps=max_steps,
            max_hops_from_seed=max_hops_from_seed,
            min_score_improvement=min_score_improvement,
        )

    def process(self, batch: list) -> Tuple[list, dict]:
        # aggregate operator: ignore per-batch rows and always load full graph
        self.graph_storage.reload()
        nodes = self.graph_storage.get_all_nodes() or []
        edges = self.graph_storage.get_all_edges() or []
        seed_node_ids = [
            node_id for node_id, node_data in nodes if self._is_image_node(node_data)
        ]

        results = []
        for seed_node_id in seed_node_ids:
            sampled = self.sampler.sample((nodes, edges), seed_node_id=seed_node_id)
            sampled["_trace_id"] = self.get_trace_id(sampled)
            results.append(sampled)
        return results, {}

    def split(self, batch):
        """
        Aggregate operators must evaluate the full upstream dataset in one pass.
        Bypass BaseOperator cache-based splitting so process() always executes.
        """
        import pandas as pd

        return batch, pd.DataFrame()

    @staticmethod
    def _is_image_node(node_data: dict) -> bool:
        entity_type = str((node_data or {}).get("entity_type", "")).upper()
        if "IMAGE" in entity_type:
            return True
        metadata = node_data.get("metadata") if isinstance(node_data, dict) else {}
        if isinstance(metadata, str):
            import json

            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        if not isinstance(metadata, dict):
            return False
        return bool(
            metadata.get("image_path")
            or metadata.get("img_path")
            or metadata.get("image_caption")
        )
