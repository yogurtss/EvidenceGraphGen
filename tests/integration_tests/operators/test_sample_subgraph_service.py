import json
from pathlib import Path
from unittest.mock import patch

from graphgen.models.subgraph_sampler.agentic_vlm_sampler import VLMSubgraphSampler
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

    def __init__(self, *, planner_mode: str = "single", degraded_success: bool = True):
        self.planner_mode = planner_mode
        self.degraded_success = degraded_success

    async def generate_answer(self, prompt: str, history=None, **extra):
        if "ROLE: Planner" in prompt:
            return self._planner_response(prompt)
        if "ROLE: RetrieverAssembler" in prompt:
            return self._assembler_response(prompt)
        if "ROLE: Judge" in prompt:
            return self._judge_response(prompt)
        return "{}"

    def _planner_response(self, prompt: str) -> str:
        if "Need degraded mode: true" in prompt:
            return json.dumps(
                {
                    "intents": [
                        {
                            "intent": "Use the image seed for conservative timing interpretation",
                            "technical_focus": "timing",
                            "question_types": ["conservative_chart_interpretation"],
                            "priority_keywords": ["timing", "latency", "trcd"],
                        }
                    ]
                }
            )
        if self.planner_mode == "multi":
            return json.dumps(
                {
                    "intents": [
                        {
                            "intent": "Explain timing constraints around activation latency",
                            "technical_focus": "timing",
                            "question_types": [
                                "parameter_relation_understanding",
                                "light_multi_hop_technical_reasoning",
                            ],
                            "priority_keywords": ["timing", "latency", "activation"],
                        },
                        {
                            "intent": "Explain architecture organization around bank grouping",
                            "technical_focus": "architecture",
                            "question_types": ["chart_diagram_interpretation"],
                            "priority_keywords": ["architecture", "bank", "group"],
                        },
                    ]
                }
            )
        return json.dumps(
            {
                "intents": [
                    {
                        "intent": "Explain timing constraints around activation latency",
                        "technical_focus": "timing",
                        "question_types": [
                            "parameter_relation_understanding",
                            "light_multi_hop_technical_reasoning",
                        ],
                        "priority_keywords": ["timing", "latency", "activation"],
                    }
                ]
            }
        )

    @staticmethod
    def _assembler_response(prompt: str) -> str:
        degraded = "Need degraded mode: true" in prompt
        if '"technical_focus": "architecture"' in prompt or "Technical focus: architecture" in prompt:
            return json.dumps(
                {
                    "technical_focus": "architecture",
                    "node_ids": ["image_seed", "bank_group", "channel_layout"],
                    "edge_pairs": [
                        ["image_seed", "bank_group"],
                        ["bank_group", "channel_layout"],
                    ],
                    "approved_question_types": ["chart_diagram_interpretation"],
                    "image_grounding_summary": "The figure layout shows the bank grouping and channel placement.",
                    "evidence_summary": "The selected nodes describe the architectural grouping shown in the image.",
                }
            )
        if degraded:
            return json.dumps(
                {
                    "technical_focus": "timing",
                    "node_ids": ["image_seed", "latency_metric"],
                    "edge_pairs": [["image_seed", "latency_metric"]],
                    "approved_question_types": ["conservative_chart_interpretation"],
                    "image_grounding_summary": "The image is needed to read the highlighted timing metric.",
                    "evidence_summary": "A direct image-to-metric link supports a conservative technical interpretation question.",
                }
            )
        return json.dumps(
            {
                "technical_focus": "timing",
                "node_ids": [
                    "image_seed",
                    "latency_metric",
                    "row_activation",
                    "random_access_perf",
                ],
                "edge_pairs": [
                    ["image_seed", "latency_metric"],
                    ["latency_metric", "row_activation"],
                    ["row_activation", "random_access_perf"],
                ],
                "approved_question_types": [
                    "parameter_relation_understanding",
                    "light_multi_hop_technical_reasoning",
                ],
                "image_grounding_summary": "The image is required to identify the timing metric under discussion.",
                "evidence_summary": "The selected chain connects the visual timing metric to activation behavior and performance.",
            }
        )

    def _judge_response(self, prompt: str) -> str:
        degraded = "Degraded mode: true" in prompt
        if degraded and not self.degraded_success:
            return json.dumps(
                {
                    "image_indispensability": 0.55,
                    "answer_stability": 0.52,
                    "evidence_closure": 0.54,
                    "technical_relevance": 0.7,
                    "reasoning_depth": 0.2,
                    "hallucination_risk": 0.22,
                    "theme_coherence": 0.75,
                    "overall_score": 0.5,
                    "passes": False,
                    "rejection_reason": "insufficient_evidence",
                }
            )
        if degraded:
            return json.dumps(
                {
                    "image_indispensability": 0.86,
                    "answer_stability": 0.81,
                    "evidence_closure": 0.78,
                    "technical_relevance": 0.84,
                    "reasoning_depth": 0.42,
                    "hallucination_risk": 0.12,
                    "theme_coherence": 0.83,
                    "overall_score": 0.8,
                    "passes": True,
                    "rejection_reason": "",
                }
            )
        if self.planner_mode == "reject":
            return json.dumps(
                {
                    "image_indispensability": 0.25,
                    "answer_stability": 0.82,
                    "evidence_closure": 0.8,
                    "technical_relevance": 0.85,
                    "reasoning_depth": 0.3,
                    "hallucination_risk": 0.1,
                    "theme_coherence": 0.8,
                    "overall_score": 0.42,
                    "passes": False,
                    "rejection_reason": "weak_image_necessity",
                }
            )
        if self.planner_mode == "multi" and "Candidate technical focus: architecture" in prompt:
            return json.dumps(
                {
                    "image_indispensability": 0.82,
                    "answer_stability": 0.78,
                    "evidence_closure": 0.75,
                    "technical_relevance": 0.86,
                    "reasoning_depth": 0.4,
                    "hallucination_risk": 0.15,
                    "theme_coherence": 0.87,
                    "overall_score": 0.78,
                    "passes": True,
                    "rejection_reason": "",
                }
            )
        return json.dumps(
            {
                "image_indispensability": 0.9,
                "answer_stability": 0.84,
                "evidence_closure": 0.82,
                "technical_relevance": 0.91,
                "reasoning_depth": 0.7,
                "hallucination_risk": 0.08,
                "theme_coherence": 0.88,
                "overall_score": 0.86,
                "passes": True,
                "rejection_reason": "",
            }
        )


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
        "bank_group": {
            "entity_type": "COMPONENT",
            "entity_name": "bank_group",
            "description": "Bank groups organize rows to structure timing behavior in the architecture.",
            "source_id": "doc1-text",
            "evidence_span": "bank groups organize rows",
        },
        "channel_layout": {
            "entity_type": "ARCHITECTURE",
            "entity_name": "channel_layout",
            "description": "Channel layout groups bank resources across the architecture.",
            "source_id": "doc1-text",
            "evidence_span": "channel layout groups bank resources",
        },
        "off_topic_bandwidth": {
            "entity_type": "METRIC",
            "entity_name": "off_topic_bandwidth",
            "description": "HBM bandwidth metric from another document.",
            "source_id": "doc2-text",
            "evidence_span": "",
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
        (
            "image_seed",
            "bank_group",
            {
                "relation_type": "shows_architecture",
                "description": "The figure labels bank groups in the architecture.",
                "evidence_span": "Figure 1 labels bank groups.",
                "source_id": "doc1-image",
            },
        ),
        (
            "bank_group",
            "channel_layout",
            {
                "relation_type": "organized_in",
                "description": "Bank groups are organized within the channel layout.",
                "evidence_span": "bank groups are organized within the channel layout",
                "source_id": "doc1-text",
            },
        ),
        (
            "image_seed",
            "off_topic_bandwidth",
            {
                "relation_type": "related_to",
                "description": "Merged graph includes unrelated bandwidth information.",
                "evidence_span": "",
                "source_id": "doc2-text",
            },
        ),
    ]
    for node_id, node_data in nodes.items():
        storage.upsert_node(node_id, node_data)
    for src_id, tgt_id, edge_data in edges:
        storage.upsert_edge(src_id, tgt_id, edge_data)
    storage.index_done_callback()


def test_agentic_sampler_builds_single_selected_subgraph(tmp_path: Path):
    storage = NetworkXStorage(
        working_dir=str(tmp_path / "cache"),
        namespace="graph_sampler",
    )
    _build_graph(storage)
    sampler = VLMSubgraphSampler(
        storage,
        _DummyLLM(planner_mode="single"),
        candidate_pool_size=3,
    )
    batch = (storage.get_all_nodes(), storage.get_all_edges())

    result = __import__("asyncio").run(
        sampler.sample(batch, seed_node_id="image_seed")
    )

    assert result["abstained"] is False
    assert result["selection_mode"] == "single"
    assert result["seed_image_path"] == "figures/fig1.png"
    assert len(result["selected_subgraphs"]) == 1
    selected = result["selected_subgraphs"][0]
    assert selected["technical_focus"] == "timing"
    assert "light_multi_hop_technical_reasoning" in selected["approved_question_types"]
    assert selected["judge_scores"]["passes"] is True
    assert any(node_id == "random_access_perf" for node_id, _ in selected["nodes"])


def test_agentic_sampler_emits_debug_trace_when_enabled(tmp_path: Path):
    storage = NetworkXStorage(
        working_dir=str(tmp_path / "cache"),
        namespace="graph_sampler_debug",
    )
    _build_graph(storage)
    sampler = VLMSubgraphSampler(
        storage,
        _DummyLLM(planner_mode="single"),
    )

    result = __import__("asyncio").run(
        sampler.sample(
            (storage.get_all_nodes(), storage.get_all_edges()),
            seed_node_id="image_seed",
            debug=True,
        )
    )

    trace = result["debug_trace"]
    assert trace["sampler_version"] == "v1"
    assert trace["final_status"] == "selected"
    assert [step["step_index"] for step in trace["steps"]] == list(
        range(1, len(trace["steps"]) + 1)
    )
    step_types = [step["step_type"] for step in trace["steps"]]
    assert "neighborhood_collection" in step_types
    assert "planner_intents" in step_types
    assert "candidate_judgement" in step_types
    assert step_types[-1] == "selection"


def test_sample_subgraph_service_can_keep_multiple_themes(tmp_path: Path):
    graph_storage = NetworkXStorage(
        working_dir=str(tmp_path / "cache"),
        namespace="graph",
    )
    _build_graph(graph_storage)

    def _init_storage(backend: str, working_dir: str, namespace: str):
        if namespace == "graph":
            return graph_storage
        return _DummyKV()

    with patch(
        "graphgen.operators.sample_subgraph.sample_subgraph_service.init_storage",
        side_effect=_init_storage,
    ), patch(
        "graphgen.operators.sample_subgraph.sample_subgraph_service.init_llm",
        return_value=_DummyLLM(planner_mode="multi"),
    ):
        service = SampleSubgraphService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            graph_backend="networkx",
            max_selected_subgraphs=2,
            candidate_pool_size=3,
        )

    rows, _ = service.process([{"_trace_id": "build-grounded-tree-kg-1"}])

    assert len(rows) == 1
    assert rows[0]["selection_mode"] == "multi"
    assert len(rows[0]["selected_subgraphs"]) == 2
    focuses = {item["technical_focus"] for item in rows[0]["selected_subgraphs"]}
    assert {"timing", "architecture"} == focuses


def test_agentic_sampler_rejects_candidates_with_weak_image_necessity(tmp_path: Path):
    storage = NetworkXStorage(
        working_dir=str(tmp_path / "cache"),
        namespace="graph_sampler_reject",
    )
    _build_graph(storage)
    sampler = VLMSubgraphSampler(
        storage,
        _DummyLLM(planner_mode="reject"),
        allow_degraded=False,
    )

    result = __import__("asyncio").run(
        sampler.sample((storage.get_all_nodes(), storage.get_all_edges()), seed_node_id="image_seed")
    )

    assert result["abstained"] is True
    assert result["selected_subgraphs"] == []
    assert result["candidate_bundle"][0]["decision"] == "rejected"
    assert "weak_image_necessity" in result["candidate_bundle"][0]["rejection_reason"]


def test_agentic_sampler_uses_degraded_path_when_primary_candidates_fail(tmp_path: Path):
    storage = NetworkXStorage(
        working_dir=str(tmp_path / "cache"),
        namespace="graph_sampler_degraded",
    )
    _build_graph(storage)
    sampler = VLMSubgraphSampler(
        storage,
        _DummyLLM(planner_mode="reject", degraded_success=True),
        allow_degraded=True,
    )

    result = __import__("asyncio").run(
        sampler.sample((storage.get_all_nodes(), storage.get_all_edges()), seed_node_id="image_seed")
    )

    assert result["abstained"] is False
    assert result["degraded"] is True
    assert result["degraded_reason"] == "fallback_to_conservative_chart_interpretation"
    assert result["selected_subgraphs"][0]["degraded"] is True
    assert result["selected_subgraphs"][0]["approved_question_types"] == [
        "conservative_chart_interpretation"
    ]


def test_agentic_sampler_abstains_if_degraded_path_also_fails(tmp_path: Path):
    storage = NetworkXStorage(
        working_dir=str(tmp_path / "cache"),
        namespace="graph_sampler_fail",
    )
    _build_graph(storage)
    sampler = VLMSubgraphSampler(
        storage,
        _DummyLLM(planner_mode="reject", degraded_success=False),
        allow_degraded=True,
    )

    result = __import__("asyncio").run(
        sampler.sample((storage.get_all_nodes(), storage.get_all_edges()), seed_node_id="image_seed")
    )

    assert result["abstained"] is True
    assert result["selected_subgraphs"] == []
    assert len(result["candidate_bundle"]) >= 2


def test_agentic_sampler_debug_trace_records_degraded_failure(tmp_path: Path):
    storage = NetworkXStorage(
        working_dir=str(tmp_path / "cache"),
        namespace="graph_sampler_debug_fail",
    )
    _build_graph(storage)
    sampler = VLMSubgraphSampler(
        storage,
        _DummyLLM(planner_mode="reject", degraded_success=False),
        allow_degraded=True,
    )

    result = __import__("asyncio").run(
        sampler.sample(
            (storage.get_all_nodes(), storage.get_all_edges()),
            seed_node_id="image_seed",
            debug=True,
        )
    )

    trace = result["debug_trace"]
    assert trace["final_status"] == "abstained"
    assert trace["steps"][-1]["status"] == "failed"
    assert any(step["phase"] == "fallback" for step in trace["steps"])


def test_agentic_sampler_omits_debug_trace_by_default(tmp_path: Path):
    storage = NetworkXStorage(
        working_dir=str(tmp_path / "cache"),
        namespace="graph_sampler_no_debug",
    )
    _build_graph(storage)
    sampler = VLMSubgraphSampler(storage, _DummyLLM())

    result = __import__("asyncio").run(
        sampler.sample(
            (storage.get_all_nodes(), storage.get_all_edges()),
            seed_node_id="image_seed",
        )
    )

    assert "debug_trace" not in result


def test_generate_service_consumes_selected_subgraphs_and_preserves_metadata(tmp_path: Path):
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
            "seed_node_id": "image_seed",
            "seed_image_path": "figures/fig1.png",
            "selection_mode": "single",
            "degraded": False,
            "candidate_bundle": [{"candidate_id": "candidate-1", "decision": "accepted"}],
            "max_vqas_per_selected_subgraph": 2,
            "selected_subgraphs": [
                {
                    "subgraph_id": "candidate-1",
                    "technical_focus": "timing",
                    "nodes": [
                        ("image_seed", {"entity_type": "IMAGE", "metadata": '{"image_path":"figures/fig1.png"}'}),
                        ("latency_metric", {"entity_type": "METRIC", "description": "tRCD latency"}),
                    ],
                    "edges": [
                        (
                            "image_seed",
                            "latency_metric",
                            {"relation_type": "shows_metric", "description": "figure shows tRCD"},
                        )
                    ],
                    "image_grounding_summary": "The figure is needed to identify the timing metric.",
                    "evidence_summary": "The edge ties the figure to the metric.",
                    "judge_scores": {"overall_score": 0.88, "passes": True},
                    "approved_question_types": [
                        "parameter_relation_understanding",
                        "light_multi_hop_technical_reasoning",
                    ],
                    "degraded": False,
                }
            ],
        }
    ]

    rows, _ = service.process(batch)

    assert len(rows) == 2
    assert {row["task_type"] for row in rows} == {"aggregated", "multi_hop"}
    assert all(row["seed_node_id"] == "image_seed" for row in rows)
    assert all(row["subgraph_id"] == "candidate-1" for row in rows)
    assert all("candidate_bundle" in row for row in rows)


def test_generate_service_respects_per_subgraph_budget_with_single_generator(tmp_path: Path):
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
        "aggregated": _DummyGenerator("aggregated", count=3),
        "multi_hop": _DummyGenerator("multi_hop"),
        "vqa": _DummyGenerator("vqa"),
    }

    batch = [
        {
            "_trace_id": "sample-2",
            "seed_node_id": "image_seed",
            "seed_image_path": "figures/fig1.png",
            "selection_mode": "single",
            "degraded": False,
            "candidate_bundle": [{"candidate_id": "candidate-2", "decision": "accepted"}],
            "max_vqas_per_selected_subgraph": 2,
            "selected_subgraphs": [
                {
                    "subgraph_id": "candidate-2",
                    "technical_focus": "comparison",
                    "nodes": [
                        ("image_seed", {"entity_type": "IMAGE", "metadata": '{"image_path":"figures/fig1.png"}'}),
                        ("latency_metric", {"entity_type": "METRIC", "description": "tRCD latency"}),
                    ],
                    "edges": [
                        (
                            "image_seed",
                            "latency_metric",
                            {"relation_type": "shows_metric", "description": "figure shows tRCD"},
                        )
                    ],
                    "image_grounding_summary": "The figure is needed to read the metric.",
                    "evidence_summary": "The figure and metric node support direct interpretation.",
                    "judge_scores": {"overall_score": 0.9, "passes": True},
                    "approved_question_types": ["chart_diagram_interpretation"],
                    "degraded": False,
                }
            ],
        }
    ]

    rows, _ = service.process(batch)

    assert len(rows) == 2
    assert {row["generator_key"] for row in rows} == {"aggregated"}


def test_sample_subgraph_service_requires_synthesizer_vlm(tmp_path: Path):
    graph_storage = NetworkXStorage(
        working_dir=str(tmp_path / "cache"),
        namespace="graph",
    )
    _build_graph(graph_storage)

    def _init_storage(backend: str, working_dir: str, namespace: str):
        if namespace == "graph":
            return graph_storage
        return _DummyKV()

    with patch(
        "graphgen.operators.sample_subgraph.sample_subgraph_service.init_storage",
        side_effect=_init_storage,
    ), patch(
        "graphgen.operators.sample_subgraph.sample_subgraph_service.init_llm",
        return_value=None,
    ):
        try:
            SampleSubgraphService(
                working_dir=str(tmp_path / "cache"),
                kv_backend="json_kv",
                graph_backend="networkx",
            )
        except ValueError as exc:
            assert "requires a configured synthesizer VLM" in str(exc)
        else:
            raise AssertionError("SampleSubgraphService should fail without a synthesizer VLM")


def test_sample_subgraph_service_can_enable_debug_trace(tmp_path: Path):
    graph_storage = NetworkXStorage(
        working_dir=str(tmp_path / "cache"),
        namespace="graph",
    )
    _build_graph(graph_storage)

    def _init_storage(backend: str, working_dir: str, namespace: str):
        if namespace == "graph":
            return graph_storage
        return _DummyKV()

    with patch(
        "graphgen.operators.sample_subgraph.sample_subgraph_service.init_storage",
        side_effect=_init_storage,
    ), patch(
        "graphgen.operators.sample_subgraph.sample_subgraph_service.init_llm",
        return_value=_DummyLLM(),
    ):
        service = SampleSubgraphService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            graph_backend="networkx",
            debug=True,
        )

    rows, _ = service.process([{"_trace_id": "build-grounded-tree-kg-1"}])

    assert rows
    assert "debug_trace" in rows[0]
