import json
from pathlib import Path
from unittest.mock import patch

from graphgen.models.subgraph_sampler.family_aware_vlm_sampler import (
    FamilyAwareVLMSubgraphSampler,
)
from graphgen.operators.generate.generate_service import GenerateService
from graphgen.operators.sample_subgraph_v3 import SampleSubgraphV3Service
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
    def __init__(self, label: str, count: int = 1):
        self.label = label
        self.count = count

    async def generate(self, batch):
        return [
            {
                "question": f"{self.label} question {index + 1}",
                "answer": f"{self.label} answer {index + 1}",
            }
            for index in range(self.count)
        ]

    @staticmethod
    def format_generation_results(result: dict, output_data_format: str) -> dict:
        return {
            "messages": [
                {"role": "user", "content": result["question"]},
                {"role": "assistant", "content": result["answer"]},
            ]
        }


class _DummyV3LLM:
    tokenizer = None

    def __init__(self):
        self.editor_rounds = {"atomic": 0, "aggregated": 0, "multi_hop": 0}
        self.judge_rounds = {"atomic": 0, "aggregated": 0, "multi_hop": 0}

    async def generate_answer(self, prompt: str, history=None, **extra):
        family = self._extract_family(prompt)
        if "ROLE: EditorV3" in prompt:
            self.editor_rounds[family] += 1
            return self._editor_response(family, self.editor_rounds[family])
        if "ROLE: JudgeV3" in prompt:
            self.judge_rounds[family] += 1
            return self._judge_response(family, self.judge_rounds[family])
        return "{}"

    @staticmethod
    def _extract_family(prompt: str) -> str:
        marker = "QA family: "
        if marker not in prompt:
            return "atomic"
        return prompt.split(marker, 1)[1].splitlines()[0].strip()

    @staticmethod
    def _editor_response(family: str, round_index: int) -> str:
        if family == "atomic":
            return json.dumps(
                {
                    "intent": "Read the highlighted timing parameter",
                    "technical_focus": "timing",
                    "image_grounding_summary": "The image is needed to identify the timing label.",
                    "evidence_summary": "A direct visual edge connects the figure to the metric.",
                    "actions": [
                        {"action_type": "add_node", "node_id": "latency_metric"},
                        {
                            "action_type": "add_edge",
                            "src_id": "image_seed",
                            "tgt_id": "latency_metric",
                        },
                        {"action_type": "commit_for_judgement"},
                    ],
                }
            )
        if family == "aggregated":
            return json.dumps(
                {
                    "intent": "Explain the timing dependency",
                    "technical_focus": "timing",
                    "image_grounding_summary": "The image anchors the metric and its nearby labels.",
                    "evidence_summary": "The metric and row activation combine into one coherent explanation.",
                    "actions": [
                        {"action_type": "add_node", "node_id": "latency_metric"},
                        {"action_type": "add_node", "node_id": "row_activation"},
                        {
                            "action_type": "add_edge",
                            "src_id": "image_seed",
                            "tgt_id": "latency_metric",
                        },
                        {
                            "action_type": "add_edge",
                            "src_id": "latency_metric",
                            "tgt_id": "row_activation",
                        },
                        {"action_type": "commit_for_judgement"},
                    ],
                }
            )
        if round_index == 1:
            return json.dumps(
                {
                    "intent": "Trace the timing consequence",
                    "technical_focus": "timing",
                    "image_grounding_summary": "The image starts the reasoning chain at the timing metric.",
                    "evidence_summary": "The first pass reaches row activation but not the downstream effect.",
                    "actions": [
                        {"action_type": "add_node", "node_id": "latency_metric"},
                        {"action_type": "add_node", "node_id": "row_activation"},
                        {
                            "action_type": "add_edge",
                            "src_id": "image_seed",
                            "tgt_id": "latency_metric",
                        },
                        {
                            "action_type": "add_edge",
                            "src_id": "latency_metric",
                            "tgt_id": "row_activation",
                        },
                        {"action_type": "commit_for_judgement"},
                    ],
                }
            )
        return json.dumps(
            {
                "intent": "Trace the timing consequence",
                "technical_focus": "timing",
                "image_grounding_summary": "The image grounds the full timing chain.",
                "evidence_summary": "The chain now reaches the downstream performance effect.",
                "actions": [
                    {"action_type": "add_node", "node_id": "random_access_perf"},
                    {
                        "action_type": "add_edge",
                        "src_id": "row_activation",
                        "tgt_id": "random_access_perf",
                    },
                    {"action_type": "commit_for_judgement"},
                ],
            }
        )

    @staticmethod
    def _judge_response(family: str, round_index: int) -> str:
        if family == "multi_hop" and round_index == 1:
            return json.dumps(
                {
                    "image_indispensability": 0.86,
                    "answer_stability": 0.72,
                    "evidence_closure": 0.6,
                    "technical_relevance": 0.84,
                    "reasoning_depth": 0.55,
                    "hallucination_risk": 0.12,
                    "theme_coherence": 0.83,
                    "overall_score": 0.7,
                    "passes": False,
                    "sufficient": False,
                    "needs_expansion": True,
                    "rejection_reason": "need_downstream_effect",
                    "suggested_actions": ["add_node", "add_edge"],
                }
            )
        return json.dumps(
            {
                "image_indispensability": 0.9,
                "answer_stability": 0.82,
                "evidence_closure": 0.8,
                "technical_relevance": 0.88,
                "reasoning_depth": 0.72 if family == "multi_hop" else 0.4,
                "hallucination_risk": 0.08,
                "theme_coherence": 0.86,
                "overall_score": 0.85,
                "passes": True,
                "sufficient": True,
                "needs_expansion": False,
                "rejection_reason": "",
                "suggested_actions": [],
            }
        )


def _build_graph(storage: NetworkXStorage):
    Path(storage.working_dir).mkdir(parents=True, exist_ok=True)
    nodes = {
        "image_seed": {
            "entity_type": "IMAGE",
            "entity_name": "image_seed",
            "description": "Timing diagram showing activation latency across bank groups.",
            "metadata": '{"image_path":"figures/fig1.png","source_trace_id":"doc1-image"}',
            "source_id": "doc1-image<SEP>doc1-text",
            "evidence_span": "Figure 1 shows activation latency across bank groups.",
        },
        "latency_metric": {
            "entity_type": "METRIC",
            "entity_name": "latency_metric",
            "description": "tRCD activation latency metric for opening a row.",
            "source_id": "doc1-text",
            "evidence_span": "tRCD activation latency metric",
        },
        "row_activation": {
            "entity_type": "CONCEPT",
            "entity_name": "row_activation",
            "description": "Row activation must satisfy timing constraints before a read.",
            "source_id": "doc1-text",
            "evidence_span": "row activation must satisfy timing constraints",
        },
        "random_access_perf": {
            "entity_type": "PERFORMANCE",
            "entity_name": "random_access_perf",
            "description": "Lower activation latency improves random access performance.",
            "source_id": "doc1-text",
            "evidence_span": "lower activation latency improves random access performance",
        },
    }
    edges = [
        (
            "image_seed",
            "latency_metric",
            {
                "relation_type": "shows_metric",
                "description": "The figure highlights the tRCD latency metric.",
                "evidence_span": "Figure 1 highlights the tRCD metric.",
                "source_id": "doc1-image",
            },
        ),
        (
            "latency_metric",
            "row_activation",
            {
                "relation_type": "constrains",
                "description": "tRCD constrains row activation timing.",
                "evidence_span": "tRCD constrains row activation timing",
                "source_id": "doc1-text",
            },
        ),
        (
            "row_activation",
            "random_access_perf",
            {
                "relation_type": "impacts",
                "description": "Activation latency impacts random access performance.",
                "evidence_span": "activation latency impacts random access performance",
                "source_id": "doc1-text",
            },
        ),
    ]
    for node_id, node_data in nodes.items():
        storage.upsert_node(node_id, node_data)
    for src_id, tgt_id, edge_data in edges:
        storage.upsert_edge(src_id, tgt_id, edge_data)
    storage.index_done_callback()


def _build_atomic_only_graph(storage: NetworkXStorage):
    Path(storage.working_dir).mkdir(parents=True, exist_ok=True)
    storage.upsert_node(
        "image_seed",
        {
            "entity_type": "IMAGE",
            "entity_name": "image_seed",
            "description": "Timing figure with one highlighted parameter.",
            "metadata": '{"image_path":"figures/fig1.png","source_trace_id":"doc1-image"}',
            "source_id": "doc1-image<SEP>doc1-text",
        },
    )
    storage.upsert_node(
        "latency_metric",
        {
            "entity_type": "METRIC",
            "entity_name": "latency_metric",
            "description": "tRCD activation latency metric.",
            "source_id": "doc1-text",
        },
    )
    storage.upsert_edge(
        "image_seed",
        "latency_metric",
        {
            "relation_type": "shows_metric",
            "description": "The figure highlights the tRCD latency metric.",
            "source_id": "doc1-image",
        },
    )
    storage.index_done_callback()


def test_v3_sampler_builds_family_specific_selected_subgraphs(tmp_path: Path):
    storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_graph(storage)
    sampler = FamilyAwareVLMSubgraphSampler(storage, _DummyV3LLM(), max_rounds=3)

    result = __import__("asyncio").run(
        sampler.sample((storage.get_all_nodes(), storage.get_all_edges()), seed_node_id="image_seed")
    )

    assert result["sampler_version"] == "v3"
    assert result["abstained"] is False
    assert {item["qa_family"] for item in result["selected_subgraphs"]} == {
        "atomic",
        "aggregated",
        "multi_hop",
    }
    assert any(
        item["qa_family"] == "multi_hop" and item["hop"] == 3
        for item in result["neighborhood_trace"]
    )
    assert all("qa_family" in item for item in result["agent_session"])


def test_v3_sampler_skips_families_that_fail_precheck(tmp_path: Path):
    storage = NetworkXStorage(
        working_dir=str(tmp_path / "cache"),
        namespace="graph_atomic_only",
    )
    _build_atomic_only_graph(storage)
    sampler = FamilyAwareVLMSubgraphSampler(storage, _DummyV3LLM(), max_rounds=2)

    result = __import__("asyncio").run(
        sampler.sample((storage.get_all_nodes(), storage.get_all_edges()), seed_node_id="image_seed")
    )

    assert {item["qa_family"] for item in result["selected_subgraphs"]} == {"atomic"}
    skipped = {
        item["qa_family"]
        for item in result["agent_session"]
        if item["status"] == "skipped"
    }
    assert skipped == {"aggregated", "multi_hop"}


def test_v3_service_and_generate_auto_integrate(tmp_path: Path):
    graph_storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_graph(graph_storage)

    def _init_storage(backend: str, working_dir: str, namespace: str):
        if namespace == "graph":
            return graph_storage
        return _DummyKV()

    with patch(
        "graphgen.operators.sample_subgraph_v3.sample_subgraph_v3_service.init_storage",
        side_effect=_init_storage,
    ), patch(
        "graphgen.operators.sample_subgraph_v3.sample_subgraph_v3_service.init_llm",
        return_value=_DummyV3LLM(),
    ):
        service = SampleSubgraphV3Service(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            graph_backend="networkx",
            max_rounds=3,
        )

    rows, _ = service.process([{"_trace_id": "build-grounded-tree-kg-1"}])
    assert len(rows) == 1
    assert rows[0]["sampler_version"] == "v3"

    with patch(
        "graphgen.operators.generate.generate_service.init_storage",
        return_value=_DummyKV(),
    ), patch(
        "graphgen.operators.generate.generate_service.init_llm",
        return_value=_DummyV3LLM(),
    ):
        generate_service = GenerateService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            method="auto",
            data_format="ChatML",
        )

    generate_service.generator_map = {
        "atomic": _DummyGenerator("atomic"),
        "aggregated": _DummyGenerator("aggregated"),
        "multi_hop": _DummyGenerator("multi_hop"),
        "vqa": _DummyGenerator("vqa"),
    }
    qa_rows, _ = generate_service.process(rows)

    assert {row["qa_family"] for row in qa_rows} == {"atomic", "aggregated", "multi_hop"}
    assert {row["generator_key"] for row in qa_rows} == {"atomic", "aggregated", "multi_hop"}
    assert all(json.loads(row["sub_graph"])["nodes"] for row in qa_rows)
