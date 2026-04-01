import json
from pathlib import Path
from unittest.mock import patch

from graphgen.models.subgraph_sampler.graph_editing_vlm_sampler import (
    GraphEditingVLMSubgraphSampler,
)
from graphgen.operators.generate.generate_service import GenerateService
from graphgen.operators.sample_subgraph_v2 import SampleSubgraphV2Service
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


class _DummyV2LLM:
    tokenizer = None

    def __init__(self, mode: str = "single"):
        self.mode = mode
        self.editor_round = 0
        self.judge_round = 0

    async def generate_answer(self, prompt: str, history=None, **extra):
        if "ROLE: EditorV2" in prompt:
            self.editor_round += 1
            return self._editor_response(prompt)
        if "ROLE: JudgeV2" in prompt:
            self.judge_round += 1
            return self._judge_response(prompt)
        return "{}"

    def _editor_response(self, prompt: str) -> str:
        degraded = "Degraded mode: true" in prompt
        if degraded:
            return json.dumps(
                {
                    "intent": "Use the image for conservative timing interpretation",
                    "technical_focus": "timing",
                    "approved_question_types": ["conservative_chart_interpretation"],
                    "image_grounding_summary": "The image is needed to read the timing highlight.",
                    "evidence_summary": "The subgraph ties the figure to a conservative timing metric.",
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

        if self.mode == "remove":
            if self.editor_round == 1:
                return json.dumps(
                    {
                        "intent": "Explain timing behavior",
                        "technical_focus": "timing",
                        "approved_question_types": ["parameter_relation_understanding"],
                        "image_grounding_summary": "The image identifies the timing metric and nearby labels.",
                        "evidence_summary": "The candidate currently mixes timing and architecture nodes.",
                        "actions": [
                            {"action_type": "add_node", "node_id": "latency_metric"},
                            {"action_type": "add_node", "node_id": "bank_group"},
                            {
                                "action_type": "add_edge",
                                "src_id": "image_seed",
                                "tgt_id": "latency_metric",
                            },
                            {
                                "action_type": "add_edge",
                                "src_id": "image_seed",
                                "tgt_id": "bank_group",
                            },
                            {"action_type": "commit_for_judgement"},
                        ],
                    }
                )
            return json.dumps(
                {
                    "intent": "Explain timing behavior",
                    "technical_focus": "timing",
                    "approved_question_types": ["parameter_relation_understanding"],
                    "image_grounding_summary": "The image is needed to read the timing metric.",
                    "evidence_summary": "The timing-only subgraph is now focused.",
                    "actions": [
                        {"action_type": "remove_node", "node_id": "bank_group"},
                        {"action_type": "commit_for_judgement"},
                    ],
                }
            )

        if self.mode == "revise":
            if self.editor_round == 1:
                return json.dumps(
                    {
                        "intent": "Describe the chart",
                        "technical_focus": "timing",
                        "approved_question_types": ["chart_diagram_interpretation"],
                        "image_grounding_summary": "The image presents the timing metric.",
                        "evidence_summary": "The figure and metric are connected.",
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
            return json.dumps(
                {
                    "intent": "Explain the timing consequence",
                    "technical_focus": "timing",
                    "approved_question_types": ["light_multi_hop_technical_reasoning"],
                    "image_grounding_summary": "The image anchors the timing consequence discussion.",
                    "evidence_summary": "The revised intent now matches the technical reasoning target.",
                    "actions": [
                        {
                            "action_type": "revise_intent",
                            "intent": "Explain the timing consequence",
                            "technical_focus": "timing",
                            "approved_question_types": [
                                "light_multi_hop_technical_reasoning"
                            ],
                        },
                        {"action_type": "commit_for_judgement"},
                    ],
                }
            )

        if self.mode == "expand":
            if self.editor_round == 1:
                return json.dumps(
                    {
                        "intent": "Explain timing dependencies",
                        "technical_focus": "timing",
                        "approved_question_types": ["parameter_relation_understanding"],
                        "image_grounding_summary": "The image surfaces the timing metric.",
                        "evidence_summary": "The first pass anchors the metric but lacks closure.",
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
            return json.dumps(
                {
                    "intent": "Explain timing dependencies",
                    "technical_focus": "timing",
                    "approved_question_types": [
                        "parameter_relation_understanding",
                        "light_multi_hop_technical_reasoning",
                    ],
                    "image_grounding_summary": "The image anchors the timing chain.",
                    "evidence_summary": "The expanded graph closes the chain through row activation.",
                    "actions": [
                        {"action_type": "add_node", "node_id": "row_activation"},
                        {
                            "action_type": "add_edge",
                            "src_id": "latency_metric",
                            "tgt_id": "row_activation",
                        },
                        {"action_type": "commit_for_judgement"},
                    ],
                }
            )

        if self.mode == "hard_cap":
            return json.dumps(
                {
                    "intent": "Try to overgrow the candidate",
                    "technical_focus": "timing",
                    "approved_question_types": ["parameter_relation_understanding"],
                    "image_grounding_summary": "The image is the anchor.",
                    "evidence_summary": "The editor tries to exceed the cap.",
                    "actions": [
                        {"action_type": "add_node", "node_id": "latency_metric"},
                        {"action_type": "add_node", "node_id": "bank_group"},
                        {"action_type": "add_node", "node_id": "row_activation"},
                        {
                            "action_type": "add_edge",
                            "src_id": "image_seed",
                            "tgt_id": "latency_metric",
                        },
                        {
                            "action_type": "add_edge",
                            "src_id": "image_seed",
                            "tgt_id": "bank_group",
                        },
                        {"action_type": "commit_for_judgement"},
                    ],
                }
            )

        return json.dumps(
            {
                "intent": "Explain timing metric",
                "technical_focus": "timing",
                "approved_question_types": ["parameter_relation_understanding"],
                "image_grounding_summary": "The image is needed to identify the timing metric.",
                "evidence_summary": "The subgraph ties the figure to the metric.",
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

    def _judge_response(self, prompt: str) -> str:
        degraded = "Degraded mode: true" in prompt
        if degraded:
            if self.mode == "fail":
                return json.dumps(
                    {
                        "image_indispensability": 0.4,
                        "answer_stability": 0.45,
                        "evidence_closure": 0.42,
                        "technical_relevance": 0.5,
                        "reasoning_depth": 0.2,
                        "hallucination_risk": 0.3,
                        "theme_coherence": 0.5,
                        "overall_score": 0.4,
                        "passes": False,
                        "sufficient": False,
                        "needs_expansion": False,
                        "rejection_reason": "degraded_insufficient",
                        "suggested_actions": [],
                    }
                )
            return json.dumps(
                {
                    "image_indispensability": 0.85,
                    "answer_stability": 0.8,
                    "evidence_closure": 0.78,
                    "technical_relevance": 0.84,
                    "reasoning_depth": 0.3,
                    "hallucination_risk": 0.1,
                    "theme_coherence": 0.8,
                    "overall_score": 0.8,
                    "passes": True,
                    "sufficient": True,
                    "needs_expansion": False,
                    "rejection_reason": "",
                    "suggested_actions": [],
                }
            )

        if self.mode == "expand" and self.judge_round == 1:
            return json.dumps(
                {
                    "image_indispensability": 0.8,
                    "answer_stability": 0.56,
                    "evidence_closure": 0.52,
                    "technical_relevance": 0.82,
                    "reasoning_depth": 0.35,
                    "hallucination_risk": 0.18,
                    "theme_coherence": 0.8,
                    "overall_score": 0.6,
                    "passes": False,
                    "sufficient": False,
                    "needs_expansion": True,
                    "rejection_reason": "needs_bridge_node",
                    "suggested_actions": ["add_node", "add_edge"],
                }
            )

        if self.mode == "remove" and self.judge_round == 1:
            return json.dumps(
                {
                    "image_indispensability": 0.72,
                    "answer_stability": 0.58,
                    "evidence_closure": 0.55,
                    "technical_relevance": 0.7,
                    "reasoning_depth": 0.3,
                    "hallucination_risk": 0.22,
                    "theme_coherence": 0.45,
                    "overall_score": 0.57,
                    "passes": False,
                    "sufficient": False,
                    "needs_expansion": False,
                    "rejection_reason": "remove_architecture_noise",
                    "suggested_actions": ["remove_node"],
                }
            )

        if self.mode == "revise" and self.judge_round == 1:
            return json.dumps(
                {
                    "image_indispensability": 0.75,
                    "answer_stability": 0.6,
                    "evidence_closure": 0.62,
                    "technical_relevance": 0.58,
                    "reasoning_depth": 0.25,
                    "hallucination_risk": 0.2,
                    "theme_coherence": 0.72,
                    "overall_score": 0.59,
                    "passes": False,
                    "sufficient": False,
                    "needs_expansion": False,
                    "rejection_reason": "revise_question_family",
                    "suggested_actions": ["revise_intent"],
                }
            )

        if self.mode == "fail":
            return json.dumps(
                {
                    "image_indispensability": 0.45,
                    "answer_stability": 0.5,
                    "evidence_closure": 0.48,
                    "technical_relevance": 0.55,
                    "reasoning_depth": 0.2,
                    "hallucination_risk": 0.28,
                    "theme_coherence": 0.5,
                    "overall_score": 0.45,
                    "passes": False,
                    "sufficient": False,
                    "needs_expansion": False,
                    "rejection_reason": "never_sufficient",
                    "suggested_actions": [],
                }
            )

        return json.dumps(
            {
                "image_indispensability": 0.9,
                "answer_stability": 0.84,
                "evidence_closure": 0.82,
                "technical_relevance": 0.9,
                "reasoning_depth": 0.55,
                "hallucination_risk": 0.08,
                "theme_coherence": 0.88,
                "overall_score": 0.86,
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
        "bank_group": {
            "entity_type": "COMPONENT",
            "entity_name": "bank_group",
            "description": "Bank groups organize rows to structure timing behavior in the architecture.",
            "source_id": "doc1-text",
            "evidence_span": "bank groups organize rows",
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


def test_v2_sampler_builds_single_selected_subgraph_and_emits_v2_fields(tmp_path: Path):
    storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_graph(storage)
    sampler = GraphEditingVLMSubgraphSampler(storage, _DummyV2LLM("single"))

    result = __import__("asyncio").run(
        sampler.sample((storage.get_all_nodes(), storage.get_all_edges()), seed_node_id="image_seed")
    )

    assert result["sampler_version"] == "v2"
    assert result["abstained"] is False
    assert result["selected_subgraphs"]
    assert result["edit_trace"]
    assert result["judge_trace"]
    assert result["candidate_states"]
    assert result["termination_reason"] == "judge_marked_sufficient"


def test_v2_sampler_emits_debug_trace_when_enabled(tmp_path: Path):
    storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_graph(storage)
    sampler = GraphEditingVLMSubgraphSampler(storage, _DummyV2LLM("expand"))

    result = __import__("asyncio").run(
        sampler.sample(
            (storage.get_all_nodes(), storage.get_all_edges()),
            seed_node_id="image_seed",
            debug=True,
        )
    )

    trace = result["debug_trace"]
    assert trace["sampler_version"] == "v2"
    assert trace["final_status"] == "selected"
    assert [step["step_index"] for step in trace["steps"]] == list(
        range(1, len(trace["steps"]) + 1)
    )
    step_types = [step["step_type"] for step in trace["steps"]]
    assert "edit_round" in step_types
    assert "candidate_judgement" in step_types
    assert "neighborhood_expansion" in step_types
    assert step_types[-1] == "selection"


def test_v2_sampler_uses_one_hop_without_expansion_when_sufficient(tmp_path: Path):
    storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_graph(storage)
    sampler = GraphEditingVLMSubgraphSampler(storage, _DummyV2LLM("single"))

    result = __import__("asyncio").run(
        sampler.sample((storage.get_all_nodes(), storage.get_all_edges()), seed_node_id="image_seed")
    )

    assert [item["hop"] for item in result["neighborhood_trace"]] == [1]
    assert "off_topic_bandwidth" not in result["neighborhood_trace"][0]["node_ids"]


def test_v2_sampler_expands_to_two_hop_when_judge_requests_more_context(tmp_path: Path):
    storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_graph(storage)
    sampler = GraphEditingVLMSubgraphSampler(storage, _DummyV2LLM("expand"))

    result = __import__("asyncio").run(
        sampler.sample((storage.get_all_nodes(), storage.get_all_edges()), seed_node_id="image_seed")
    )

    assert result["abstained"] is False
    assert [item["hop"] for item in result["neighborhood_trace"]] == [1, 2]
    selected_node_ids = {node_id for node_id, _ in result["selected_subgraphs"][0]["nodes"]}
    assert "row_activation" in selected_node_ids


def test_v2_sampler_removes_noisy_nodes_after_judge_feedback(tmp_path: Path):
    storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_graph(storage)
    sampler = GraphEditingVLMSubgraphSampler(storage, _DummyV2LLM("remove"))

    result = __import__("asyncio").run(
        sampler.sample((storage.get_all_nodes(), storage.get_all_edges()), seed_node_id="image_seed")
    )

    selected_node_ids = {node_id for node_id, _ in result["selected_subgraphs"][0]["nodes"]}
    assert "bank_group" not in selected_node_ids
    assert any(
        action["action_type"] == "remove_node"
        for round_item in result["edit_trace"]
        for action in round_item["actions"]
    )


def test_v2_sampler_can_revise_intent_without_mutating_graph_facts(tmp_path: Path):
    storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_graph(storage)
    original_node = dict(storage.get_node("latency_metric"))
    sampler = GraphEditingVLMSubgraphSampler(storage, _DummyV2LLM("revise"))

    result = __import__("asyncio").run(
        sampler.sample((storage.get_all_nodes(), storage.get_all_edges()), seed_node_id="image_seed")
    )

    assert any(
        action["action_type"] == "revise_intent"
        for round_item in result["edit_trace"]
        for action in round_item["actions"]
    )
    assert "light_multi_hop_technical_reasoning" in result["selected_subgraphs"][0]["approved_question_types"]
    assert storage.get_node("latency_metric") == original_node


def test_v2_sampler_uses_degraded_mode_and_can_abstain(tmp_path: Path):
    storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_graph(storage)

    degraded_sampler = GraphEditingVLMSubgraphSampler(
        storage,
        _DummyV2LLM("fail"),
        max_rounds=2,
        allow_degraded=True,
    )
    degraded_result = __import__("asyncio").run(
        degraded_sampler.sample((storage.get_all_nodes(), storage.get_all_edges()), seed_node_id="image_seed")
    )
    assert degraded_result["abstained"] is True
    assert degraded_result["termination_reason"] == "degraded_exhausted"

    pass_sampler = GraphEditingVLMSubgraphSampler(
        storage,
        _DummyV2LLM("single"),
        max_rounds=2,
        allow_degraded=True,
    )
    pass_result = __import__("asyncio").run(
        pass_sampler.sample((storage.get_all_nodes(), storage.get_all_edges()), seed_node_id="image_seed")
    )
    assert pass_result["abstained"] is False


def test_v2_sampler_respects_hard_cap_and_records_ignored_actions(tmp_path: Path):
    storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_graph(storage)
    sampler = GraphEditingVLMSubgraphSampler(
        storage,
        _DummyV2LLM("hard_cap"),
        hard_cap_units=4,
        max_rounds=1,
        allow_degraded=False,
    )

    result = __import__("asyncio").run(
        sampler.sample((storage.get_all_nodes(), storage.get_all_edges()), seed_node_id="image_seed")
    )

    last_state = result["candidate_states"][-1]
    assert last_state["unit_count"] <= 4
    assert any(
        action["ignored_reason"] == "hard_cap_exceeded"
        for round_item in result["edit_trace"]
        for action in round_item["actions"]
    )


def test_v2_sampler_debug_trace_includes_action_failures(tmp_path: Path):
    storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_graph(storage)
    sampler = GraphEditingVLMSubgraphSampler(
        storage,
        _DummyV2LLM("hard_cap"),
        hard_cap_units=4,
        max_rounds=1,
        allow_degraded=False,
    )

    result = __import__("asyncio").run(
        sampler.sample(
            (storage.get_all_nodes(), storage.get_all_edges()),
            seed_node_id="image_seed",
            debug=True,
        )
    )

    edit_steps = [
        step for step in result["debug_trace"]["steps"] if step["step_type"] == "edit_round"
    ]
    assert edit_steps
    assert any(
        action["ignored_reason"] == "hard_cap_exceeded"
        for action in edit_steps[-1]["snapshot"]["actions"]
    )


def test_v2_sampler_omits_debug_trace_by_default(tmp_path: Path):
    storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_graph(storage)
    sampler = GraphEditingVLMSubgraphSampler(storage, _DummyV2LLM("single"))

    result = __import__("asyncio").run(
        sampler.sample((storage.get_all_nodes(), storage.get_all_edges()), seed_node_id="image_seed")
    )

    assert "debug_trace" not in result


def test_v2_service_and_generate_auto_integrate(tmp_path: Path):
    graph_storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_graph(graph_storage)

    def _init_storage(backend: str, working_dir: str, namespace: str):
        if namespace == "graph":
            return graph_storage
        return _DummyKV()

    with patch(
        "graphgen.operators.sample_subgraph_v2.sample_subgraph_v2_service.init_storage",
        side_effect=_init_storage,
    ), patch(
        "graphgen.operators.sample_subgraph_v2.sample_subgraph_v2_service.init_llm",
        return_value=_DummyV2LLM("single"),
    ):
        service = SampleSubgraphV2Service(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            graph_backend="networkx",
        )
    rows, _ = service.process([{"_trace_id": "build-grounded-tree-kg-1"}])
    assert len(rows) == 1
    assert rows[0]["sampler_version"] == "v2"

    with patch(
        "graphgen.operators.generate.generate_service.init_storage",
        return_value=_DummyKV(),
    ), patch(
        "graphgen.operators.generate.generate_service.init_llm",
        return_value=_DummyV2LLM("single"),
    ):
        generate_service = GenerateService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            method="auto",
            data_format="ChatML",
        )

    generate_service.generator_map = {
        "aggregated": _DummyGenerator("aggregated"),
        "multi_hop": _DummyGenerator("multi_hop"),
        "vqa": _DummyGenerator("vqa"),
    }
    qa_rows, _ = generate_service.process(rows)
    assert qa_rows
    assert all(row["subgraph_id"] == "candidate-1" for row in qa_rows)


def test_v2_service_can_enable_debug_trace(tmp_path: Path):
    graph_storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_graph(graph_storage)

    def _init_storage(backend: str, working_dir: str, namespace: str):
        if namespace == "graph":
            return graph_storage
        return _DummyKV()

    with patch(
        "graphgen.operators.sample_subgraph_v2.sample_subgraph_v2_service.init_storage",
        side_effect=_init_storage,
    ), patch(
        "graphgen.operators.sample_subgraph_v2.sample_subgraph_v2_service.init_llm",
        return_value=_DummyV2LLM("single"),
    ):
        service = SampleSubgraphV2Service(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            graph_backend="networkx",
            debug=True,
        )

    rows, _ = service.process([{"_trace_id": "build-grounded-tree-kg-1"}])

    assert rows
    assert "debug_trace" in rows[0]
