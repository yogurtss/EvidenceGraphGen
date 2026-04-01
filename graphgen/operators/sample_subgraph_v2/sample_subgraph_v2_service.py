import asyncio
import json
from typing import Tuple

from graphgen.bases import BaseGraphStorage, BaseLLMWrapper, BaseOperator
from graphgen.common.init_llm import init_llm
from graphgen.common.init_storage import init_storage
from graphgen.models import GraphEditingVLMSubgraphSampler


class SampleSubgraphV2Service(BaseOperator):
    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        graph_backend: str = "kuzu",
        hard_cap_units: int = 12,
        max_rounds: int = 6,
        max_vqas_per_selected_subgraph: int = 2,
        allow_degraded: bool = True,
        judge_pass_threshold: float = 0.68,
        debug: bool = False,
    ):
        super().__init__(
            working_dir=working_dir,
            kv_backend=kv_backend,
            op_name="sample_subgraph_v2",
        )
        self.graph_storage: BaseGraphStorage = init_storage(
            backend=graph_backend,
            working_dir=working_dir,
            namespace="graph",
        )
        self.llm_client: BaseLLMWrapper | None = init_llm("synthesizer")
        if self.llm_client is None:
            raise ValueError(
                "sample_subgraph_v2 requires a configured synthesizer VLM."
            )
        self.sampler = GraphEditingVLMSubgraphSampler(
            self.graph_storage,
            self.llm_client,
            hard_cap_units=hard_cap_units,
            max_rounds=max_rounds,
            max_vqas_per_selected_subgraph=max_vqas_per_selected_subgraph,
            allow_degraded=allow_degraded,
            judge_pass_threshold=judge_pass_threshold,
        )
        self.debug = bool(debug)

    def process(self, batch: list) -> Tuple[list, dict]:
        self.graph_storage.reload()
        nodes = self.graph_storage.get_all_nodes() or []
        edges = self.graph_storage.get_all_edges() or []
        seed_node_ids = [
            node_id for node_id, node_data in nodes if self._is_image_seed_node(node_data)
        ]

        async def _sample_all() -> list[dict]:
            results = []
            for seed_node_id in seed_node_ids:
                sampled = await self.sampler.sample(
                    (nodes, edges),
                    seed_node_id=seed_node_id,
                    debug=self.debug,
                )
                sampled["_trace_id"] = self.get_trace_id(sampled)
                results.append(sampled)
            return results

        return asyncio.run(_sample_all()), {}

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
