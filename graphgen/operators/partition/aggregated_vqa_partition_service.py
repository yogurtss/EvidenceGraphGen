from .partition_service import PartitionService


class AggregatedVQAPartitionService(PartitionService):
    """Dedicated operator for image-centric aggregated VQA partitioning."""

    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        graph_backend: str = "kuzu",
        **partition_kwargs,
    ):
        params = {
            "method": "aggregated_vqa_anchor_bfs",
            "method_params": {
                "anchor_type": ["image"],
                "max_units_per_community": 12,
                "min_units_per_community": 6,
                "section_scoped": True,
                "required_modalities": ["image", "text"],
            },
        }
        if partition_kwargs:
            params["method_params"].update(partition_kwargs)
        super().__init__(
            working_dir=working_dir,
            kv_backend=kv_backend,
            graph_backend=graph_backend,
            **params,
        )
