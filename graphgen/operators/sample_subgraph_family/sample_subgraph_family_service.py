import json
from typing import Tuple

from graphgen.bases import BaseGraphStorage, BaseLLMWrapper, BaseOperator
from graphgen.common.init_llm import init_llm
from graphgen.common.init_storage import init_storage
from graphgen.models import FamilySubgraphOrchestrator


class SampleSubgraphFamilyService(BaseOperator):
    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        graph_backend: str = "kuzu",
        max_selected_subgraphs_per_family: int = 3,
        judge_pass_threshold: float = 0.68,
        max_multi_hop_hops: int = 3,
    ):
        super().__init__(
            working_dir=working_dir,
            kv_backend=kv_backend,
            op_name="sample_subgraph_family",
        )
        self.graph_storage: BaseGraphStorage = init_storage(
            backend=graph_backend,
            working_dir=working_dir,
            namespace="graph",
        )
        self.llm_client: BaseLLMWrapper | None = init_llm("synthesizer")
        self.orchestrator = FamilySubgraphOrchestrator(
            self.graph_storage,
            self.llm_client,
            max_selected_subgraphs_per_family=max_selected_subgraphs_per_family,
            judge_pass_threshold=judge_pass_threshold,
            max_multi_hop_hops=max_multi_hop_hops,
        )

    def process(self, batch: list) -> Tuple[list, dict]:
        self.graph_storage.reload()
        nodes = self.graph_storage.get_all_nodes() or []
        seed_node_ids = [
            node_id
            for node_id, node_data in nodes
            if self._is_image_seed_node(node_data)
        ]
        rows = []
        for seed_node_id in seed_node_ids:
            sampled = self.orchestrator.sample(seed_node_id=str(seed_node_id))
            sampled["_trace_id"] = self.get_trace_id(sampled)
            rows.append(sampled)
        return rows, {}

    def split(self, batch):
        import pandas as pd

        return batch, pd.DataFrame()

    @staticmethod
    def _is_image_seed_node(node_data: dict) -> bool:
        entity_type = str((node_data or {}).get("entity_type", "")).upper()
        if "IMAGE" in entity_type:
            return True
        metadata = node_data.get("metadata") if isinstance(node_data, dict) else {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        if not isinstance(metadata, dict):
            return False
        return bool(metadata.get("image_path") or metadata.get("img_path"))
