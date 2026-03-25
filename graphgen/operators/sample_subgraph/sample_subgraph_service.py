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
        self.graph_storage.reload()
        results = []
        meta_updates = {}
        for item in batch:
            sampled = self.sampler.sample((item.get("nodes", []), item.get("edges", [])))
            sampled["_trace_id"] = self.get_trace_id(sampled)
            results.append(sampled)
            meta_updates.setdefault(item["_trace_id"], []).append(sampled["_trace_id"])
        return results, meta_updates
