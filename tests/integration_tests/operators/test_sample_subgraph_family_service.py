import json
from pathlib import Path
from unittest.mock import patch

from graphgen.models.subgraph_sampler.family_agents.orchestrator import (
    FamilySubgraphOrchestrator,
)
from graphgen.operators.generate_agentic_vqa import GenerateAgenticVQAService
from graphgen.operators.sample_subgraph_family import SampleSubgraphFamilyService
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


class _DummyGenerator:
    def __init__(self, label: str):
        self.label = label

    async def generate(self, batch):
        _nodes, edges = batch
        edge_count = len(edges)
        return [
            {
                "question": f"{self.label} question with {edge_count} edges",
                "answer": f"{self.label} answer with {edge_count} edges",
                "img_path": "figures/fig1.png",
            }
        ]

    @staticmethod
    def format_generation_results(result: dict, output_data_format: str) -> dict:
        return {
            "messages": [
                {"role": "user", "content": [{"text": result["question"], "image": result.get("img_path", "")}]},
                {"role": "assistant", "content": [{"type": "text", "text": result["answer"]}]},
            ]
        }


class _DummyFamilyLLM:
    tokenizer = None

    def __init__(self):
        self.aggregated_judge_calls = 0

    async def generate_answer(self, prompt: str, history=None, **extra):
        if "ROLE: FamilyQAJudge" in prompt:
            return self._judge_response(prompt)
        if "ROLE: FamilyQARevision" in prompt:
            return json.dumps(
                {
                    "question": "aggregated revised question",
                    "answer": "aggregated revised answer",
                }
            )
        return "{}"

    def _judge_response(self, prompt: str) -> str:
        family = self._extract_family(prompt)
        revision_id = self._extract_revision_id(prompt)
        if family == "atomic":
            return json.dumps(
                {
                    "decision": "accept",
                    "confidence": 0.91,
                    "reason": "direct evidence is sufficient",
                }
            )
        if family == "aggregated":
            self.aggregated_judge_calls += 1
            if self.aggregated_judge_calls == 1:
                return json.dumps(
                    {
                        "decision": "revise_qa_only",
                        "confidence": 0.72,
                        "reason": "needs a wider summary style",
                        "qa_revision_instruction": "Rewrite the QA as a broader thematic explanation.",
                    }
                )
            return json.dumps(
                {
                    "decision": "accept",
                    "confidence": 0.88,
                    "reason": "revised QA now matches aggregated style",
                }
            )
        if family == "multi_hop" and revision_id == 0:
            return json.dumps(
                {
                    "decision": "refine_subgraph_then_regenerate",
                    "confidence": 0.79,
                    "reason": "need one more reasoning step",
                    "subgraph_revision_instruction": "Advance the active frontier by one step.",
                }
            )
        return json.dumps(
            {
                "decision": "accept",
                "confidence": 0.9,
                "reason": "chain is now deep enough",
            }
        )

    @staticmethod
    def _extract_family(prompt: str) -> str:
        marker = "QA family: "
        if marker not in prompt:
            return "atomic"
        return prompt.split(marker, 1)[1].splitlines()[0].strip()

    @staticmethod
    def _extract_revision_id(prompt: str) -> int:
        marker = '"revision_id": '
        if marker not in prompt:
            return 0
        try:
            return int(prompt.split(marker, 1)[1].split(",", 1)[0].strip())
        except (TypeError, ValueError):
            return 0


def _build_family_graph(storage: NetworkXStorage):
    Path(storage.working_dir).mkdir(parents=True, exist_ok=True)
    nodes = {
        "image_seed": {
            "entity_type": "IMAGE",
            "entity_name": "image_seed",
            "description": "Timing figure with highlighted latency and voltage metrics.",
            "metadata": '{"image_path":"figures/fig1.png","source_trace_id":"doc1-image"}',
            "source_id": "doc1-image<SEP>doc1-text",
            "evidence_span": "Figure 1 highlights timing metrics and signal behavior.",
        },
        "latency_metric": {
            "entity_type": "METRIC",
            "entity_name": "latency_metric",
            "description": "Timing latency metric for activation.",
            "source_id": "doc1-text",
            "evidence_span": "timing latency metric for activation",
        },
        "voltage_metric": {
            "entity_type": "METRIC",
            "entity_name": "voltage_metric",
            "description": "Timing voltage metric shown next to latency.",
            "source_id": "doc1-text",
            "evidence_span": "timing voltage metric",
        },
        "decoder_logic": {
            "entity_type": "ARCHITECTURE",
            "entity_name": "decoder_logic",
            "description": "Decoder logic layout unrelated to timing aggregation.",
            "source_id": "doc1-text",
            "evidence_span": "decoder logic layout",
        },
        "row_activation": {
            "entity_type": "CONCEPT",
            "entity_name": "row_activation",
            "description": "Timing activation step required before reads.",
            "source_id": "doc1-text",
            "evidence_span": "timing activation step required before reads",
        },
        "timing_window": {
            "entity_type": "METRIC",
            "entity_name": "timing_window",
            "description": "Timing window constraint adjacent to the latency metric.",
            "source_id": "doc1-text",
            "evidence_span": "timing window constraint",
        },
        "bank_conflict": {
            "entity_type": "CONCEPT",
            "entity_name": "bank_conflict",
            "description": "Bank conflict side effect for misaligned timing.",
            "source_id": "doc1-text",
            "evidence_span": "bank conflict side effect",
        },
        "random_access_perf": {
            "entity_type": "PERFORMANCE",
            "entity_name": "random_access_perf",
            "description": "Timing changes affect random access performance downstream.",
            "source_id": "doc1-text",
            "evidence_span": "timing changes affect random access performance downstream",
        },
    }
    edges = [
        (
            "image_seed",
            "latency_metric",
            {
                "relation_type": "shows_metric",
                "description": "The figure highlights the latency metric.",
                "evidence_span": "Figure 1 highlights the latency metric.",
                "source_id": "doc1-image",
            },
        ),
        (
            "image_seed",
            "voltage_metric",
            {
                "relation_type": "shows_metric",
                "description": "The figure also highlights a voltage metric.",
                "evidence_span": "Figure 1 highlights the voltage metric.",
                "source_id": "doc1-image",
            },
        ),
        (
            "image_seed",
            "decoder_logic",
            {
                "relation_type": "shows_structure",
                "description": "The figure outlines decoder logic elsewhere.",
                "evidence_span": "Figure 1 outlines decoder logic.",
                "source_id": "doc1-image",
            },
        ),
        (
            "latency_metric",
            "row_activation",
            {
                "relation_type": "constrains",
                "description": "Latency constrains row activation timing.",
                "evidence_span": "latency constrains row activation timing",
                "source_id": "doc1-text",
            },
        ),
        (
            "latency_metric",
            "timing_window",
            {
                "relation_type": "constrains",
                "description": "Latency and timing window are jointly interpreted.",
                "evidence_span": "latency and timing window are jointly interpreted",
                "source_id": "doc1-text",
            },
        ),
        (
            "latency_metric",
            "bank_conflict",
            {
                "relation_type": "impacts",
                "description": "Latency misalignment can increase bank conflicts.",
                "evidence_span": "latency misalignment can increase bank conflicts",
                "source_id": "doc1-text",
            },
        ),
        (
            "row_activation",
            "random_access_perf",
            {
                "relation_type": "impacts",
                "description": "Activation timing affects random access performance.",
                "evidence_span": "activation timing affects random access performance",
                "source_id": "doc1-text",
            },
        ),
    ]
    for node_id, node_data in nodes.items():
        storage.upsert_node(node_id, node_data)
    for src_id, tgt_id, edge_data in edges:
        storage.upsert_edge(src_id, tgt_id, edge_data)
    storage.index_done_callback()


def test_family_orchestrator_enforces_family_specific_rules(tmp_path: Path):
    storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_family_graph(storage)
    orchestrator = FamilySubgraphOrchestrator(storage, _DummyFamilyLLM())

    result = orchestrator.sample(seed_node_id="image_seed")

    atomic = next(
        item for item in result["selected_subgraphs"] if item["qa_family"] == "atomic"
    )
    assert len(atomic["nodes"]) == 2
    assert len(atomic["edges"]) == 1

    aggregated = next(
        item
        for item in result["selected_subgraphs"]
        if item["qa_family"] == "aggregated"
    )
    aggregated_node_ids = {node[0] for node in aggregated["nodes"]}
    assert len(aggregated_node_ids) >= 3
    assert "decoder_logic" not in aggregated_node_ids
    assert "candidate_pool_snapshot" in aggregated
    assert "theme_signature" in aggregated

    multi_hop = next(
        item for item in result["selected_subgraphs"] if item["qa_family"] == "multi_hop"
    )
    assert len(multi_hop["edges"]) >= 2
    assert multi_hop["frontier_node_id"] in {node[0] for node in multi_hop["nodes"]}
    assert all(
        candidate["bind_from_node_id"] == multi_hop["frontier_node_id"]
        for candidate in multi_hop["candidate_pool_snapshot"]
    )

    revised = orchestrator.continue_subgraph(
        selected_subgraph=multi_hop,
        revision_reason="Advance the active frontier by one step.",
    )
    assert revised is not None
    revised_node_ids = {node[0] for node in revised["nodes"]}
    assert "random_access_perf" in revised_node_ids
    assert "bank_conflict" not in revised_node_ids
    assert revised["revision_id"] == 1


def test_sample_subgraph_family_service_and_generate_agentic_vqa_integrate(
    tmp_path: Path,
):
    graph_storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_family_graph(graph_storage)

    def _init_storage(backend: str, working_dir: str, namespace: str):
        if namespace == "graph":
            return graph_storage
        return _DummyKV()

    with patch(
        "graphgen.operators.sample_subgraph_family.sample_subgraph_family_service.init_storage",
        side_effect=_init_storage,
    ), patch(
        "graphgen.operators.sample_subgraph_family.sample_subgraph_family_service.init_llm",
        return_value=_DummyFamilyLLM(),
    ):
        sampler_service = SampleSubgraphFamilyService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            graph_backend="networkx",
        )

    sampled_rows, _ = sampler_service.process([{"_trace_id": "build-grounded-tree-kg-1"}])
    assert len(sampled_rows) == 1
    sampled = sampled_rows[0]
    assert sampled["sampler_version"] == "family_agents_v1"
    assert {"atomic", "aggregated", "multi_hop"} <= {
        item["qa_family"] for item in sampled["selected_subgraphs"]
    }

    with patch(
        "graphgen.operators.generate_agentic_vqa.generate_agentic_vqa_service.init_storage",
        side_effect=_init_storage,
    ), patch(
        "graphgen.operators.generate_agentic_vqa.generate_agentic_vqa_service.init_llm",
        return_value=_DummyFamilyLLM(),
    ):
        generate_service = GenerateAgenticVQAService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            graph_backend="networkx",
            data_format="ChatML",
            max_selected_per_family_for_generation=1,
        )

    generate_service.generator_map = {
        "atomic": _DummyGenerator("atomic"),
        "aggregated": _DummyGenerator("aggregated"),
        "multi_hop": _DummyGenerator("multi_hop"),
    }

    qa_rows, _ = generate_service.process(
        [
            {
                **sampled,
                "_trace_id": "sample-subgraph-family-1",
            }
        ]
    )

    assert {row["qa_family"] for row in qa_rows} == {
        "atomic",
        "aggregated",
        "multi_hop",
    }
    rows_by_family = {row["qa_family"]: row for row in qa_rows}
    assert rows_by_family["aggregated"]["termination_reason"] == "accepted_after_qa_revision"
    assert rows_by_family["multi_hop"]["subgraph_revision_id"] == 1
    assert "refine_subgraph" in rows_by_family["multi_hop"]["generation_trace"]
    assert "revise_qa_only" in rows_by_family["aggregated"]["qa_judge_trace"]
