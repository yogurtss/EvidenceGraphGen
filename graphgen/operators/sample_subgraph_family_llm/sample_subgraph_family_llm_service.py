import asyncio
import json
from typing import Tuple

from graphgen.bases import BaseGraphStorage, BaseLLMWrapper, BaseOperator
from graphgen.common.init_llm import init_llm
from graphgen.common.init_storage import init_storage
from graphgen.models import VisualCoreFamilyLLMSubgraphSampler


class SampleSubgraphFamilyLLMService(BaseOperator):
    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        graph_backend: str = "kuzu",
        family_qa_targets: dict[str, int] | None = None,
        family_max_depths: dict[str, int] | None = None,
        max_steps_per_family: int = 4,
        max_rollbacks_per_family: int = 1,
        judge_pass_threshold: float = 0.68,
        same_source_only: bool = True,
        allow_bootstrap_fallback: bool = False,
        max_protocol_retries_per_stage: int = 1,
        max_bootstrap_errors: int = 1,
        max_selector_errors: int = 2,
        max_judge_errors: int = 1,
        min_multi_hop_outside_core_edges: int = 2,
    ):
        super().__init__(
            working_dir=working_dir,
            kv_backend=kv_backend,
            op_name="sample_subgraph_family_llm",
        )
        self.graph_storage: BaseGraphStorage = init_storage(
            backend=graph_backend,
            working_dir=working_dir,
            namespace="graph",
        )
        self.llm_client: BaseLLMWrapper | None = init_llm("synthesizer")
        if self.llm_client is None:
            raise ValueError(
                "sample_subgraph_family_llm requires a configured synthesizer VLM."
            )
        self.sampler = VisualCoreFamilyLLMSubgraphSampler(
            self.graph_storage,
            self.llm_client,
            family_qa_targets=family_qa_targets,
            family_max_depths=family_max_depths,
            max_steps_per_family=max_steps_per_family,
            max_rollbacks_per_family=max_rollbacks_per_family,
            judge_pass_threshold=judge_pass_threshold,
            same_source_only=same_source_only,
            allow_bootstrap_fallback=allow_bootstrap_fallback,
            max_protocol_retries_per_stage=max_protocol_retries_per_stage,
            max_bootstrap_errors=max_bootstrap_errors,
            max_selector_errors=max_selector_errors,
            max_judge_errors=max_judge_errors,
            min_multi_hop_outside_core_edges=min_multi_hop_outside_core_edges,
        )

    def process(self, batch: list) -> Tuple[list, dict]:
        self.graph_storage.reload()
        nodes = self.graph_storage.get_all_nodes() or []
        seed_node_ids = [
            node_id
            for node_id, node_data in nodes
            if self._is_image_seed_node(node_data)
        ]

        async def _sample_all() -> list[dict]:
            results = []
            for seed_node_id in seed_node_ids:
                sampled = await self.sampler.sample(seed_node_id=str(seed_node_id))
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
