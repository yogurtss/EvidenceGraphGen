from pathlib import Path
from unittest.mock import patch

from graphgen.models.subgraph_sampler.value_aware_sampler import ValueAwareSubgraphSampler
from graphgen.operators.generate.generate_service import GenerateService
from graphgen.operators.sample_subgraph import SampleSubgraphService
from graphgen.storage import NetworkXStorage


class _DummyKV:
    def get_by_id(self, key):
        return None

    def get_by_ids(self, ids):
        return []

    def upsert(self, batch):
        return None

    def update(self, batch):
        return None

    def reload(self):
        return None

    def index_done_callback(self):
        return None


class _DummyLLM:
    tokenizer = None


class _DummyGenerator:
    def __init__(self, label: str):
        self.label = label

    async def generate(self, batch):
        return [{"question": f"{self.label} question", "answer": f"{self.label} answer"}]

    @staticmethod
    def format_generation_results(result: dict, output_data_format: str) -> dict:
        return {
            "messages": [
                {"role": "user", "content": result["question"]},
                {"role": "assistant", "content": result["answer"]},
            ]
        }


def _build_graph(storage: NetworkXStorage):
    nodes = {
        "table_seed": {
            "entity_type": "TABLE",
            "entity_name": "table_seed",
            "description": "DDR timing comparison table",
            "meta_data": '{"table_caption":["Table 1. DDR timing comparison."],"source_trace_id":"doc1-table"}',
            "source_id": "doc1-table<SEP>doc1-text",
            "evidence_span": "Table 1. DDR timing comparison.",
        },
        "latency_metric": {
            "entity_type": "METRIC",
            "entity_name": "latency_metric",
            "description": "tRCD latency metric for row activation",
            "source_id": "doc1-table",
            "evidence_span": "tRCD latency metric",
        },
        "precharge_metric": {
            "entity_type": "METRIC",
            "entity_name": "precharge_metric",
            "description": "tRP precharge timing metric",
            "source_id": "doc1-table",
            "evidence_span": "tRP precharge timing metric",
        },
        "row_activation": {
            "entity_type": "CONCEPT",
            "entity_name": "row_activation",
            "description": "row activation affects access timing constraints",
            "source_id": "doc1-text",
            "evidence_span": "row activation affects access timing",
        },
        "random_access_perf": {
            "entity_type": "PERFORMANCE",
            "entity_name": "random_access_perf",
            "description": "lower activation latency improves random access performance",
            "source_id": "doc1-text",
            "evidence_span": "improves random access performance",
        },
        "off_topic_bandwidth": {
            "entity_type": "METRIC",
            "entity_name": "off_topic_bandwidth",
            "description": "HBM bandwidth metric from another document",
            "source_id": "doc2-text",
            "evidence_span": "HBM bandwidth metric",
        },
    }
    edges = [
        (
            "table_seed",
            "latency_metric",
            {
                "relation_type": "shows_metric",
                "description": "table highlights tRCD latency metric",
                "evidence_span": "Table 1 highlights tRCD latency metric",
                "source_id": "doc1-table",
            },
        ),
        (
            "table_seed",
            "precharge_metric",
            {
                "relation_type": "shows_metric",
                "description": "table highlights tRP precharge timing metric",
                "evidence_span": "Table 1 highlights tRP precharge timing metric",
                "source_id": "doc1-table",
            },
        ),
        (
            "latency_metric",
            "row_activation",
            {
                "relation_type": "constrains",
                "description": "tRCD constrains row activation timing",
                "evidence_span": "tRCD constrains row activation timing",
                "source_id": "doc1-text",
            },
        ),
        (
            "row_activation",
            "random_access_perf",
            {
                "relation_type": "impacts",
                "description": "row activation latency impacts random access performance",
                "evidence_span": "impacts random access performance",
                "source_id": "doc1-text",
            },
        ),
        (
            "table_seed",
            "off_topic_bandwidth",
            {
                "relation_type": "related_to",
                "description": "merged graph includes another document metric",
                "evidence_span": "",
                "source_id": "doc2-text",
            },
        ),
    ]
    for node_id, node_data in nodes.items():
        storage.upsert_node(node_id, node_data)
    for src_id, tgt_id, edge_data in edges:
        storage.upsert_edge(src_id, tgt_id, edge_data)


def test_value_aware_sampler_prefers_same_source_multihop(tmp_path: Path):
    storage = NetworkXStorage(
        working_dir=str(tmp_path / "cache"),
        namespace="graph_sampler",
    )
    _build_graph(storage)

    sampler = ValueAwareSubgraphSampler(storage, max_units=7, max_steps=4)
    batch = (
        [
            ("table_seed", storage.get_node("table_seed")),
            ("latency_metric", storage.get_node("latency_metric")),
            ("off_topic_bandwidth", storage.get_node("off_topic_bandwidth")),
        ],
        [
            (
                "table_seed",
                "latency_metric",
                storage.get_edge("table_seed", "latency_metric"),
            ),
            (
                "table_seed",
                "off_topic_bandwidth",
                storage.get_edge("table_seed", "off_topic_bandwidth"),
            ),
        ],
    )

    result = sampler.sample(batch)
    selected_node_ids = {node_id for node_id, _ in result["nodes"]}

    assert result["task_type"] == "multi_hop"
    assert {"table_seed", "latency_metric", "row_activation"}.issubset(selected_node_ids)
    assert "off_topic_bandwidth" not in selected_node_ids
    assert result["local_core_subgraph"]["node_ids"] == [
        "latency_metric",
        "precharge_metric",
        "table_seed",
    ]
    assert result["node_roles"]["latency_metric"] == "local_core"
    assert result["node_roles"]["row_activation"] == "bridge"
    assert any("vision_centered=true" == item for item in result["selection_rationale"])


def test_sample_subgraph_service_uses_global_graph(tmp_path: Path):
    graph_storage = NetworkXStorage(
        working_dir=str(tmp_path / "cache"),
        namespace="graph",
    )
    _build_graph(graph_storage)

    def _init_storage(backend: str, working_dir: str, namespace: str):
        if namespace == "graph":
            return graph_storage
        return _DummyKV()

    with patch("graphgen.common.init_storage.init_storage", side_effect=_init_storage):
        service = SampleSubgraphService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            graph_backend="networkx",
            max_units=7,
            max_steps=4,
        )

    batch = [
        {
            "_trace_id": "partition-1",
            "nodes": [
                ("table_seed", graph_storage.get_node("table_seed")),
                ("latency_metric", graph_storage.get_node("latency_metric")),
            ],
            "edges": [
                (
                    "table_seed",
                    "latency_metric",
                    graph_storage.get_edge("table_seed", "latency_metric"),
                )
            ],
        }
    ]
    rows, _ = service.process(batch)

    assert len(rows) == 1
    assert rows[0]["task_type"] == "multi_hop"
    assert rows[0]["seed_node_id"] == "table_seed"
    assert any(node_id == "random_access_perf" for node_id, _ in rows[0]["nodes"])
    assert rows[0]["task_type_reason"] == "local_core_bridge_conclusion_chain"


def test_generate_service_auto_dispatches_by_task_type(tmp_path: Path):
    with patch(
        "graphgen.operators.generate.generate_service.init_storage",
        return_value=_DummyKV(),
    ), patch(
        "graphgen.operators.generate.generate_service.init_llm",
        return_value=_DummyLLM(),
    ):
        service = GenerateService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            method="auto",
            data_format="ChatML",
        )

    service.generator_map = {
        "aggregated": _DummyGenerator("aggregated"),
        "multi_hop": _DummyGenerator("multi_hop"),
        "vqa": _DummyGenerator("vqa"),
    }

    batch = [
        {
            "_trace_id": "sample-1",
            "task_type": "multi_hop",
            "seed_node_id": "table_seed",
            "selection_rationale": ["task_type=multi_hop"],
            "value_breakdown": {"training_value": 3.0},
            "subgraph_score": 5.0,
            "task_type_reason": "local_core_bridge_conclusion_chain",
            "local_core_subgraph": {"node_ids": ["table_seed"], "edge_pairs": []},
            "extension_subgraph": {"node_ids": ["row_activation"], "edge_pairs": []},
            "node_roles": {"table_seed": "anchor", "row_activation": "bridge"},
            "seed_chunk_ids": ["doc1-table"],
            "nodes": [("table_seed", {"entity_type": "TABLE", "description": "desc"})],
            "edges": [],
        }
    ]

    rows, _ = service.process(batch)

    assert len(rows) == 1
    assert rows[0]["task_type"] == "multi_hop"
    assert rows[0]["messages"][0]["content"] == "multi_hop question"
    assert rows[0]["seed_node_id"] == "table_seed"
    assert rows[0]["task_type_reason"] == "local_core_bridge_conclusion_chain"
