import json
from pathlib import Path
from unittest.mock import patch

from graphgen.operators.generate.generate_service import GenerateService
from graphgen.operators.sample_subgraph_family_llm import (
    SampleSubgraphFamilyLLMService,
)
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


class _DummyVisualCoreLLM:
    tokenizer = None

    def __init__(self):
        self.selector_calls = {"atomic": 0, "aggregated": 0, "multi_hop": 0}

    async def generate_answer(self, prompt: str, history=None, **extra):
        if "ROLE: VisualCoreBootstrap" in prompt:
            return self._bootstrap_response(self._extract_family(prompt))
        if "ROLE: FamilyNodeSelector" in prompt:
            return self._selector_response(prompt)
        if "ROLE: FamilyTerminationJudge" in prompt:
            return self._termination_response(prompt)
        return "{}"

    def _bootstrap_response(self, family: str) -> str:
        if family == "atomic":
            return json.dumps(
                {
                    "intent": "Read the highlighted timing metric",
                    "technical_focus": "timing",
                    "forbidden_patterns": ["decoder"],
                    "image_grounding_summary": "The figure grounds the direct timing metric.",
                    "bootstrap_rationale": "Keep one direct visual metric and stop after the visual core.",
                }
            )
        if family == "aggregated":
            return json.dumps(
                {
                    "intent": "Explain the timing topic around the highlighted figure",
                    "technical_focus": "timing",
                    "forbidden_patterns": ["decoder"],
                    "image_grounding_summary": "The image anchors the timing metric cluster.",
                    "bootstrap_rationale": "Keep the timing-related first-hop nodes and expand toward coherent timing evidence.",
                }
            )
        return json.dumps(
            {
                "intent": "Trace the timing consequence beyond the visual core",
                "technical_focus": "timing",
                "forbidden_patterns": ["decoder"],
                "image_grounding_summary": "The image anchors the timing chain at latency_metric.",
                "bootstrap_rationale": "Start from latency_metric and follow one causal chain outward.",
            }
        )

    def _selector_response(self, prompt: str) -> str:
        family = self._extract_family(prompt)
        state = self._extract_json_between(prompt, "Current state:\n", "\nCandidate lines:\n")
        candidate_lines = self._extract_json_after(prompt, "Candidate lines:\n")
        self.selector_calls[family] = self.selector_calls.get(family, 0) + 1
        if family == "aggregated":
            target = "row_activation" if "row_activation" not in state.get("selected_node_ids", []) else "random_access_perf"
            candidate_uid = self._candidate_uid_for_node(candidate_lines, target)
            return json.dumps(
                {
                    "decision": "select_candidate",
                    "candidate_uid": candidate_uid,
                    "reason": f"Select {target} for the timing aggregation.",
                    "confidence": 0.9,
                }
            )
        if family == "multi_hop":
            if self.selector_calls[family] == 1:
                target = "bank_conflict"
            elif self.selector_calls[family] == 2:
                target = "row_activation"
            else:
                target = "random_access_perf"
            candidate_uid = self._candidate_uid_for_node(candidate_lines, target)
            return json.dumps(
                {
                    "decision": "select_candidate",
                    "candidate_uid": candidate_uid,
                    "reason": f"Select {target} for the active chain.",
                    "confidence": 0.92,
                }
            )
        if family == "atomic":
            candidate_uid = self._candidate_uid_for_node(candidate_lines, "row_activation")
            return json.dumps(
                {
                    "decision": "select_candidate",
                    "candidate_uid": candidate_uid,
                    "reason": "Select one second-hop timing fact for atomic QA.",
                    "confidence": 0.9,
                }
            )
        return json.dumps({"decision": "stop_selection", "reason": "unknown_family"})

    def _termination_response(self, prompt: str) -> str:
        family = self._extract_family(prompt)
        stage = self._extract_stage(prompt)
        state = self._extract_json_between(prompt, "Current state:\n", "\nLast selected candidate:\n")
        selected_nodes = state.get("selected_node_ids", [])
        if family == "atomic":
            return self._decision_json("accept", True, "accepted", 0.88)
        if family == "aggregated":
            if stage == "bootstrap":
                return self._decision_json("continue", False, "continue", 0.62)
            if "random_access_perf" in selected_nodes:
                return self._decision_json("accept", True, "accepted", 0.9)
            return self._decision_json("continue", False, "continue", 0.7)
        if stage == "bootstrap":
            return self._decision_json("continue", False, "continue", 0.6)
        if "bank_conflict" in selected_nodes:
            return self._decision_json("rollback_last_step", False, "continue", 0.45)
        if "random_access_perf" in selected_nodes:
            return self._decision_json("accept", True, "accepted", 0.91)
        return self._decision_json("continue", False, "continue", 0.73)

    @staticmethod
    def _decision_json(
        decision: str,
        sufficient: bool,
        termination_reason: str,
        overall_score: float,
    ) -> str:
        return json.dumps(
            {
                "decision": decision,
                "sufficient": sufficient,
                "termination_reason": termination_reason,
                "reason": decision,
                "suggested_action": "",
                "scores": {
                    "image_indispensability": 0.88,
                    "answer_stability": 0.8 if sufficient else 0.66,
                    "evidence_closure": 0.82 if sufficient else 0.61,
                    "technical_relevance": 0.9,
                    "reasoning_depth": 0.84 if decision == "accept" else 0.55,
                    "hallucination_risk": 0.1,
                    "theme_coherence": 0.87,
                    "overall_score": overall_score,
                },
            }
        )

    @staticmethod
    def _extract_family(prompt: str) -> str:
        marker = "QA family: "
        return prompt.split(marker, 1)[1].splitlines()[0].strip()

    @staticmethod
    def _extract_stage(prompt: str) -> str:
        marker = "Stage: "
        return prompt.split(marker, 1)[1].splitlines()[0].strip()

    @staticmethod
    def _extract_json_between(prompt: str, start_marker: str, end_marker: str):
        raw = prompt.split(start_marker, 1)[1].split(end_marker, 1)[0]
        return json.loads(raw)

    @staticmethod
    def _extract_json_after(prompt: str, start_marker: str):
        return json.loads(prompt.split(start_marker, 1)[1])

    @staticmethod
    def _candidate_uid_for_node(candidate_lines: list[str], node_id: str) -> str:
        marker = f"node={node_id} |"
        return next(line.split(" | ", 1)[0] for line in candidate_lines if marker in line)


class _EmptyBootstrapLLM(_DummyVisualCoreLLM):
    async def generate_answer(self, prompt: str, history=None, **extra):
        if "ROLE: VisualCoreBootstrap" in prompt:
            return "{}"
        return await super().generate_answer(prompt, history=history, **extra)


class _EmptyKeepBootstrapLLM(_DummyVisualCoreLLM):
    def _bootstrap_response(self, family: str) -> str:
        return json.dumps(
            {
                "intent": f"abstain-{family}",
                "technical_focus": "none",
                "keep_first_hop_node_ids": [],
                "drop_first_hop_node_ids": [
                    "latency_metric",
                    "voltage_metric",
                    "decoder_logic",
                ],
                "preferred_entity_types": [],
                "preferred_relation_types": [],
                "forbidden_patterns": [],
                "target_reasoning_depth": 1,
                "image_grounding_summary": "No grounded visual-core path should be kept.",
                "bootstrap_rationale": "The family should abstain for this seed.",
            }
        )


class _InvalidSelectorLLM(_DummyVisualCoreLLM):
    def _selector_response(self, prompt: str) -> str:
        family = self._extract_family(prompt)
        if family in {"aggregated", "multi_hop"}:
            return json.dumps(
                {
                    "decision": "select_candidate",
                    "candidate_uid": "missing:candidate:uid",
                    "reason": "Keep selecting an invalid candidate to test termination.",
                    "confidence": 0.2,
                }
            )
        return super()._selector_response(prompt)


class _MissingJudgeScoresLLM(_DummyVisualCoreLLM):
    def _termination_response(self, prompt: str) -> str:
        family = self._extract_family(prompt)
        stage = self._extract_stage(prompt)
        if family == "aggregated" and stage == "selection":
            payload = json.loads(
                self._decision_json("accept", True, "accepted", 0.9)
            )
            payload["scores"].pop("overall_score", None)
            return json.dumps(payload)
        return super()._termination_response(prompt)


class _ShallowMultiHopAcceptLLM(_DummyVisualCoreLLM):
    def _selector_response(self, prompt: str) -> str:
        family = self._extract_family(prompt)
        if family != "multi_hop":
            return super()._selector_response(prompt)
        candidate_lines = self._extract_json_after(prompt, "Candidate lines:\n")
        candidate_uid = self._candidate_uid_for_node(candidate_lines, "row_activation")
        return json.dumps(
            {
                "decision": "select_candidate",
                "candidate_uid": candidate_uid,
                "reason": "Try to accept after a shallow single outside-core hop.",
                "confidence": 0.93,
            }
        )

    def _termination_response(self, prompt: str) -> str:
        family = self._extract_family(prompt)
        stage = self._extract_stage(prompt)
        if family == "multi_hop" and stage == "selection":
            return self._decision_json("accept", True, "accepted", 0.92)
        return super()._termination_response(prompt)


class _ChaoticVisualCoreLLM(_DummyVisualCoreLLM):
    def _bootstrap_response(self, family: str) -> str:
        if family == "atomic":
            return json.dumps(
                {
                    "intent": "skip atomic",
                    "technical_focus": "none",
                    "keep_first_hop_node_ids": [],
                    "drop_first_hop_node_ids": [
                        "latency_metric",
                        "voltage_metric",
                        "decoder_logic",
                    ],
                    "preferred_entity_types": [],
                    "preferred_relation_types": [],
                    "forbidden_patterns": [],
                    "target_reasoning_depth": 1,
                    "image_grounding_summary": "Atomic should abstain.",
                    "bootstrap_rationale": "No clean atomic visual fact should be kept.",
                }
            )
        return super()._bootstrap_response(family)

    def _selector_response(self, prompt: str) -> str:
        family = self._extract_family(prompt)
        if family == "aggregated":
            return json.dumps(
                {
                    "decision": "select_candidate",
                    "candidate_uid": "still:missing:uid",
                    "reason": "Repeated invalid selector output.",
                    "confidence": 0.1,
                }
            )
        if family == "multi_hop":
            candidate_lines = self._extract_json_after(prompt, "Candidate lines:\n")
            candidate_uid = self._candidate_uid_for_node(candidate_lines, "row_activation")
            return json.dumps(
                {
                    "decision": "select_candidate",
                    "candidate_uid": candidate_uid,
                    "reason": "Shallow multi-hop accept attempt.",
                    "confidence": 0.9,
                }
            )
        return super()._selector_response(prompt)

    def _termination_response(self, prompt: str) -> str:
        family = self._extract_family(prompt)
        stage = self._extract_stage(prompt)
        if family == "multi_hop" and stage == "selection":
            return self._decision_json("accept", True, "accepted", 0.9)
        return super()._termination_response(prompt)


def _build_visual_core_graph(storage: NetworkXStorage):
    Path(storage.working_dir).mkdir(parents=True, exist_ok=True)
    nodes = {
        "image_seed": {
            "entity_type": "IMAGE",
            "entity_name": "image_seed",
            "description": "Timing figure with highlighted latency and voltage labels.",
            "metadata": '{"image_path":"figures/fig1.png","source_trace_id":"doc1-image"}',
            "source_id": "doc1-image<SEP>doc1-text",
            "evidence_span": "Figure 1 highlights timing labels and local effects.",
        },
        "latency_metric": {
            "entity_type": "METRIC",
            "entity_name": "latency_metric",
            "description": "Latency metric highlighted next to the timing trace.",
            "source_id": "doc1-text",
            "evidence_span": "latency metric highlighted next to the timing trace",
        },
        "voltage_metric": {
            "entity_type": "METRIC",
            "entity_name": "voltage_metric",
            "description": "Voltage metric shown adjacent to the same timing topic.",
            "source_id": "doc1-text",
            "evidence_span": "voltage metric shown adjacent to the same timing topic",
        },
        "decoder_logic": {
            "entity_type": "ARCHITECTURE",
            "entity_name": "decoder_logic",
            "description": "Decoder logic branch that should be pruned during bootstrap.",
            "source_id": "doc1-text",
            "evidence_span": "decoder logic branch",
        },
        "row_activation": {
            "entity_type": "CONCEPT",
            "entity_name": "row_activation",
            "description": "Latency constrains row activation timing.",
            "source_id": "doc1-text",
            "evidence_span": "latency constrains row activation timing",
        },
        "timing_window": {
            "entity_type": "METRIC",
            "entity_name": "timing_window",
            "description": "Timing window sibling of the latency metric.",
            "source_id": "doc1-text",
            "evidence_span": "timing window sibling of the latency metric",
        },
        "bank_conflict": {
            "entity_type": "CONCEPT",
            "entity_name": "bank_conflict",
            "description": "Latency misalignment can increase bank conflicts.",
            "source_id": "doc1-text",
            "evidence_span": "latency misalignment can increase bank conflicts",
        },
        "voltage_guard": {
            "entity_type": "CONCEPT",
            "entity_name": "voltage_guard",
            "description": "Voltage guard rule linked to the same visual topic.",
            "source_id": "doc1-text",
            "evidence_span": "voltage guard rule linked to the same visual topic",
        },
        "decoder_lane": {
            "entity_type": "ARCHITECTURE",
            "entity_name": "decoder_lane",
            "description": "Decoder lane detail that should disappear with decoder_logic.",
            "source_id": "doc1-text",
            "evidence_span": "decoder lane detail",
        },
        "random_access_perf": {
            "entity_type": "PERFORMANCE",
            "entity_name": "random_access_perf",
            "description": "Activation timing affects random access performance.",
            "source_id": "doc1-text",
            "evidence_span": "activation timing affects random access performance",
        },
    }
    edges = [
        ("image_seed", "latency_metric", {"relation_type": "shows_metric", "source_id": "doc1-image"}),
        ("image_seed", "voltage_metric", {"relation_type": "shows_metric", "source_id": "doc1-image"}),
        ("image_seed", "decoder_logic", {"relation_type": "shows_structure", "source_id": "doc1-image"}),
        ("latency_metric", "row_activation", {"relation_type": "constrains", "source_id": "doc1-text"}),
        ("latency_metric", "timing_window", {"relation_type": "constrains", "source_id": "doc1-text"}),
        ("latency_metric", "bank_conflict", {"relation_type": "impacts", "source_id": "doc1-text"}),
        ("voltage_metric", "voltage_guard", {"relation_type": "explains", "source_id": "doc1-text"}),
        ("decoder_logic", "decoder_lane", {"relation_type": "contains", "source_id": "doc1-text"}),
        ("row_activation", "random_access_perf", {"relation_type": "impacts", "source_id": "doc1-text"}),
    ]
    for node_id, node_data in nodes.items():
        storage.upsert_node(node_id, node_data)
    for src_id, tgt_id, edge_data in edges:
        storage.upsert_edge(src_id, tgt_id, edge_data)
    storage.index_done_callback()


def _build_first_hop_only_graph(storage: NetworkXStorage):
    Path(storage.working_dir).mkdir(parents=True, exist_ok=True)
    nodes = {
        "image_seed": {
            "entity_type": "IMAGE",
            "entity_name": "image_seed",
            "description": "Timing figure with only image-adjacent metric labels.",
            "metadata": '{"image_path":"figures/first-hop-only.png","source_trace_id":"doc2-image"}',
            "source_id": "doc2-image<SEP>doc2-text",
            "evidence_span": "Figure 2 highlights local timing labels.",
        },
        "latency_metric": {
            "entity_type": "METRIC",
            "entity_name": "latency_metric",
            "description": "Latency metric shown directly in the image.",
            "source_id": "doc2-text",
            "evidence_span": "latency metric shown directly in the image",
        },
        "voltage_metric": {
            "entity_type": "METRIC",
            "entity_name": "voltage_metric",
            "description": "Voltage metric shown directly in the image.",
            "source_id": "doc2-text",
            "evidence_span": "voltage metric shown directly in the image",
        },
    }
    edges = [
        (
            "image_seed",
            "latency_metric",
            {
                "relation_type": "shows_metric",
                "source_id": "doc2-image",
                "metadata": '{"source_trace_id":"doc2-image"}',
            },
        ),
        (
            "image_seed",
            "voltage_metric",
            {
                "relation_type": "shows_metric",
                "source_id": "doc2-image",
                "metadata": '{"source_trace_id":"doc2-image"}',
            },
        ),
    ]
    for node_id, node_data in nodes.items():
        storage.upsert_node(node_id, node_data)
    for src_id, tgt_id, edge_data in edges:
        storage.upsert_edge(src_id, tgt_id, edge_data)
    storage.index_done_callback()


def test_sample_subgraph_family_llm_bootstraps_visual_core_and_updates_candidates(
    tmp_path: Path,
):
    graph_storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_visual_core_graph(graph_storage)

    def _init_storage(backend: str, working_dir: str, namespace: str):
        if namespace == "graph":
            return graph_storage
        return _DummyKV()

    with patch(
        "graphgen.operators.sample_subgraph_family_llm.sample_subgraph_family_llm_service.init_storage",
        side_effect=_init_storage,
    ), patch(
        "graphgen.operators.sample_subgraph_family_llm.sample_subgraph_family_llm_service.init_llm",
        return_value=_DummyVisualCoreLLM(),
    ):
        service = SampleSubgraphFamilyLLMService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            graph_backend="networkx",
            family_qa_targets={"atomic": 1, "aggregated": 2, "multi_hop": 1},
            family_max_depths={"atomic": 0, "aggregated": 2, "multi_hop": 3},
            max_steps_per_family=3,
            max_rollbacks_per_family=1,
        )

    rows, _ = service.process([{"_trace_id": "build-grounded-tree-kg-1"}])
    assert len(rows) == 1
    result = rows[0]
    assert result["sampler_version"] == "family_llm_v2"
    assert {item["qa_family"] for item in result["selected_subgraphs"]} == {
        "atomic",
        "aggregated",
        "multi_hop",
    }

    bootstrap_by_family = {
        item["qa_family"]: item for item in result["family_bootstrap_trace"]
    }
    aggregated_bootstrap = bootstrap_by_family["aggregated"]
    assert aggregated_bootstrap["analysis_first_hop_node_ids"] == [
        "latency_metric",
        "voltage_metric",
        "decoder_logic",
    ]
    assert {
        candidate["candidate_node_id"]
        for candidate in aggregated_bootstrap["preview_candidates"]
    } >= {"row_activation", "timing_window", "bank_conflict", "voltage_guard", "decoder_lane"}

    aggregated_first_step = next(
        item
        for item in result["family_selection_trace"]
        if item["qa_family"] == "aggregated"
        and item.get("candidate_node_id") == "row_activation"
    )
    aggregated_first_step_candidates = {
        candidate["candidate_node_id"]
        for candidate in aggregated_first_step["candidate_pool_after_step"]
    }
    assert "decoder_lane" not in aggregated_first_step_candidates
    assert {
        "timing_window",
        "bank_conflict",
        "voltage_guard",
        "random_access_perf",
    } <= aggregated_first_step_candidates

    selected_by_family = {
        item["qa_family"]: item for item in result["selected_subgraphs"]
    }

    atomic = selected_by_family["atomic"]
    assert {node[0] for node in atomic["nodes"]} == {
        "image_seed::virtual_image",
        "row_activation",
    }
    assert atomic["visual_core_node_ids"] == ["image_seed::virtual_image"]
    assert atomic["analysis_first_hop_node_ids"] == [
        "latency_metric",
        "voltage_metric",
        "decoder_logic",
    ]
    assert atomic["analysis_only_node_ids"] == [
        "image_seed",
        "latency_metric",
        "voltage_metric",
        "decoder_logic",
    ]
    assert "latency_metric" not in {node[0] for node in atomic["nodes"]}
    atomic_virtual_edges = [
        edge for edge in atomic["edges"] if edge[0] == "image_seed::virtual_image"
    ]
    assert len(atomic_virtual_edges) == 1
    assert atomic_virtual_edges[0][1] == "row_activation"
    assert atomic_virtual_edges[0][2]["synthetic"] is True
    assert atomic_virtual_edges[0][2]["analysis_anchor_node_id"] == "latency_metric"
    assert atomic_virtual_edges[0][2]["virtualized_from_path"] == [
        "image_seed::virtual_image",
        "latency_metric",
        "row_activation",
    ]
    assert atomic["target_qa_count"] == 1

    aggregated = selected_by_family["aggregated"]
    aggregated_nodes = {node[0] for node in aggregated["nodes"]}
    assert {"image_seed::virtual_image", "row_activation", "random_access_perf"} <= aggregated_nodes
    assert not {"image_seed", "latency_metric", "voltage_metric"} & aggregated_nodes
    assert aggregated["visual_core_node_ids"] == ["image_seed::virtual_image"]
    assert aggregated["analysis_first_hop_node_ids"] == [
        "latency_metric",
        "voltage_metric",
        "decoder_logic",
    ]
    assert aggregated["direction_mode"] == "outward"
    assert aggregated["direction_anchor_edge"] == ["image_seed::virtual_image", "row_activation"]
    assert {
        candidate["candidate_node_id"] for candidate in aggregated["candidate_pool_snapshot"]
    } == set()
    assert aggregated["target_qa_count"] == 2

    aggregated_deeper_step = next(
        item
        for item in result["family_selection_trace"]
        if item["qa_family"] == "aggregated"
        and item["candidate_node_id"] == "random_access_perf"
    )
    assert {
        candidate["candidate_node_id"]
        for candidate in aggregated_deeper_step["candidate_pool_after_step"]
    } == set()

    multi_hop = selected_by_family["multi_hop"]
    multi_hop_nodes = {node[0] for node in multi_hop["nodes"]}
    assert "bank_conflict" not in multi_hop_nodes
    assert {"image_seed::virtual_image", "row_activation", "random_access_perf"} <= multi_hop_nodes
    assert "latency_metric" not in multi_hop_nodes
    multi_hop_rollback = next(
        item
        for item in result["family_selection_trace"]
        if item["qa_family"] == "multi_hop" and item["decision"] == "rollback_last_step"
    )
    assert multi_hop_rollback["candidate_uid"].endswith("bank_conflict:1:latency_metric")
    multi_hop_chain_step = next(
        item
        for item in result["family_selection_trace"]
        if item["qa_family"] == "multi_hop"
        and item.get("candidate_node_id") == "row_activation"
    )
    assert {
        candidate["candidate_node_id"]
        for candidate in multi_hop_chain_step["candidate_pool_after_step"]
    } == {"timing_window", "voltage_guard", "random_access_perf"}

    visualization_trace = result["visualization_trace"]
    assert visualization_trace["schema_version"] == "visual_core_family_timeline_v1"
    assert visualization_trace["sampler_version"] == "family_llm_v2"
    assert [event["order"] for event in visualization_trace["events"]] == list(
        range(1, len(visualization_trace["events"]) + 1)
    )
    event_types = [event["event_type"] for event in visualization_trace["events"]]
    assert "bootstrap_candidates_collected" in event_types
    assert "candidate_selected" in event_types
    assert "candidate_pool_updated" in event_types
    assert "judge_decision" in event_types
    assert "rollback" in event_types
    assert "subgraph_materialized" in event_types

    graph_catalog = visualization_trace["graph_catalog"]
    assert "image_seed::virtual_image" in graph_catalog["nodes"]
    assert "row_activation" in graph_catalog["nodes"]
    virtual_edge = graph_catalog["edges"]["image_seed::virtual_image->row_activation"]
    assert virtual_edge["synthetic"] is True
    assert virtual_edge["analysis_anchor_node_id"] == "latency_metric"
    assert virtual_edge["virtualized_from_path"] == [
        "image_seed::virtual_image",
        "latency_metric",
        "row_activation",
    ]

    aggregated_pool_event = next(
        event
        for event in visualization_trace["events"]
        if event["qa_family"] == "aggregated"
        and event["event_type"] == "candidate_pool_updated"
        and event["chosen_candidate"].get("candidate_node_id") == "random_access_perf"
    )
    assert {
        candidate["candidate_node_id"]
        for candidate in aggregated_pool_event["candidate_pool"]
    } == set()

    rollback_event = next(
        event
        for event in visualization_trace["events"]
        if event["qa_family"] == "multi_hop"
        and event["event_type"] == "rollback"
    )
    assert rollback_event["selected_node_ids"] == ["image_seed::virtual_image"]
    assert rollback_event["chosen_candidate"]["candidate_uid"].endswith(
        "bank_conflict:1:latency_metric"
    )


def test_generate_service_prefers_target_qa_count_for_family_llm_subgraphs(
    tmp_path: Path,
):
    with patch(
        "graphgen.operators.generate.generate_service.init_storage",
        return_value=_DummyKV(),
    ), patch(
        "graphgen.operators.generate.generate_service.init_llm",
        return_value=_DummyVisualCoreLLM(),
    ):
        service = GenerateService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            method="auto",
            data_format="ChatML",
        )

    service.generator_map = {
        "atomic": _DummyGenerator("atomic", count=3),
        "aggregated": _DummyGenerator("aggregated", count=3),
        "multi_hop": _DummyGenerator("multi_hop", count=3),
        "vqa": _DummyGenerator("vqa", count=3),
    }

    rows, _ = service.process(
        [
            {
                "_trace_id": "family-llm-generate",
                "seed_node_id": "image_seed",
                "seed_image_path": "figures/fig1.png",
                "selection_mode": "multi",
                "degraded": False,
                "max_vqas_per_selected_subgraph": 1,
                "selected_subgraphs": [
                    {
                        "subgraph_id": "atomic-1",
                        "qa_family": "atomic",
                        "technical_focus": "timing",
                        "nodes": [
                            ("image_seed", {"entity_type": "IMAGE"}),
                            ("latency_metric", {"entity_type": "METRIC"}),
                        ],
                        "edges": [("image_seed", "latency_metric", {"relation_type": "shows_metric"})],
                        "approved_question_types": ["atomic"],
                        "target_qa_count": 1,
                    },
                    {
                        "subgraph_id": "aggregated-1",
                        "qa_family": "aggregated",
                        "technical_focus": "timing",
                        "nodes": [
                            ("image_seed", {"entity_type": "IMAGE"}),
                            ("latency_metric", {"entity_type": "METRIC"}),
                            ("row_activation", {"entity_type": "CONCEPT"}),
                        ],
                        "edges": [
                            ("image_seed", "latency_metric", {"relation_type": "shows_metric"}),
                            ("latency_metric", "row_activation", {"relation_type": "constrains"}),
                        ],
                        "approved_question_types": ["aggregated"],
                        "target_qa_count": 2,
                    },
                    {
                        "subgraph_id": "multi-hop-1",
                        "qa_family": "multi_hop",
                        "technical_focus": "timing",
                        "nodes": [
                            ("image_seed", {"entity_type": "IMAGE"}),
                            ("latency_metric", {"entity_type": "METRIC"}),
                            ("row_activation", {"entity_type": "CONCEPT"}),
                        ],
                        "edges": [
                            ("image_seed", "latency_metric", {"relation_type": "shows_metric"}),
                            ("latency_metric", "row_activation", {"relation_type": "constrains"}),
                        ],
                        "approved_question_types": ["multi_hop"],
                        "target_qa_count": 1,
                    },
                ],
            }
        ]
    )

    assert len(rows) == 4
    qa_family_counts = {}
    for row in rows:
        qa_family_counts[row["qa_family"]] = qa_family_counts.get(row["qa_family"], 0) + 1
        visualization_trace = json.loads(row["visualization_trace"])
        qa_event = visualization_trace["events"][-1]
        assert qa_event["event_type"] == "qa_generated"
        assert qa_event["qa_trace_id"] == row["_trace_id"]
        assert qa_event["qa_family"] == row["qa_family"]
        assert qa_event["generator_key"] == row["generator_key"]
        assert qa_event["question"] in row["messages"][0]["content"]
        assert qa_event["answer"] in row["messages"][1]["content"]
        assert visualization_trace["graph_catalog"]["nodes"]
        assert visualization_trace["graph_catalog"]["edges"]
    assert qa_family_counts == {"atomic": 1, "aggregated": 2, "multi_hop": 1}


def test_family_llm_sampler_atomic_falls_back_to_first_hop_without_second_hop(
    tmp_path: Path,
):
    graph_storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_first_hop_only_graph(graph_storage)
    llm = _DummyVisualCoreLLM()

    def _init_storage(backend: str, working_dir: str, namespace: str):
        if namespace == "graph":
            return graph_storage
        return _DummyKV()

    with patch(
        "graphgen.operators.sample_subgraph_family_llm.sample_subgraph_family_llm_service.init_storage",
        side_effect=_init_storage,
    ), patch(
        "graphgen.operators.sample_subgraph_family_llm.sample_subgraph_family_llm_service.init_llm",
        return_value=llm,
    ):
        service = SampleSubgraphFamilyLLMService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            graph_backend="networkx",
            max_steps_per_family=3,
        )

    rows, _ = service.process([{"_trace_id": "first-hop-only"}])
    result = rows[0]
    selected_by_family = {
        item["qa_family"]: item for item in result["selected_subgraphs"]
    }
    assert set(selected_by_family) == {"atomic"}

    atomic = selected_by_family["atomic"]
    selected_node_ids = {node[0] for node in atomic["nodes"]}
    assert "image_seed::virtual_image" in selected_node_ids
    selected_first_hop = next(
        node_id for node_id in selected_node_ids if node_id != "image_seed::virtual_image"
    )
    assert selected_first_hop in {"latency_metric", "voltage_metric"}

    assert atomic["selected_evidence_node_ids"] == [selected_first_hop]
    assert atomic["visual_core_node_ids"] == ["image_seed::virtual_image"]
    assert atomic["analysis_first_hop_node_ids"] == [
        "latency_metric",
        "voltage_metric",
    ]
    assert atomic["target_qa_count"] == 1

    fallback_edges = [
        edge
        for edge in atomic["edges"]
        if edge[0] == "image_seed::virtual_image" and edge[1] == selected_first_hop
    ]
    assert len(fallback_edges) == 1
    assert fallback_edges[0][2]["synthetic"] is True
    assert fallback_edges[0][2]["fallback"] == "atomic_first_hop"
    assert fallback_edges[0][2]["virtualized_from_edge_pair"] == [
        "image_seed",
        selected_first_hop,
    ]
    assert fallback_edges[0][2]["metadata"]["source_trace_id"] == "doc2-image"

    sessions = {item["qa_family"]: item for item in result["family_sessions"]}
    assert sessions["atomic"]["status"] == "accepted"
    assert sessions["atomic"]["termination_reason"] == "atomic_first_hop_fallback"
    assert sessions["aggregated"]["termination_reason"] == "candidate_pool_exhausted"
    assert sessions["multi_hop"]["termination_reason"] == "candidate_pool_exhausted"
    assert llm.selector_calls == {"atomic": 0, "aggregated": 0, "multi_hop": 0}

    selection_events = [
        item for item in result["family_selection_trace"] if item["qa_family"] == "atomic"
    ]
    assert selection_events == [
        {
            "qa_family": "atomic",
            "step_index": 1,
            "decision": "atomic_first_hop_fallback",
            "candidate_uid": (
                f"image_seed::virtual_image:{selected_first_hop}:"
                "atomic_first_hop_fallback"
            ),
            "candidate_node_id": selected_first_hop,
            "depth": 1,
            "direction_mode": "outward",
            "candidate_pool_after_step": [],
            "reason": (
                "Atomic fallback selected one image-adjacent node because no "
                "logical first-layer candidates remained."
            ),
        }
    ]
    fallback_termination = next(
        item
        for item in result["family_termination_trace"]
        if item["qa_family"] == "atomic"
        and item["stage"] == "selection"
        and item["termination_reason"] == "atomic_first_hop_fallback"
    )
    assert fallback_termination["decision"] == "accept"
    assert fallback_termination["decision_source"] == "system_fallback"


def test_family_llm_sampler_abstains_on_bootstrap_protocol_error(tmp_path: Path):
    graph_storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_visual_core_graph(graph_storage)

    def _init_storage(backend: str, working_dir: str, namespace: str):
        if namespace == "graph":
            return graph_storage
        return _DummyKV()

    with patch(
        "graphgen.operators.sample_subgraph_family_llm.sample_subgraph_family_llm_service.init_storage",
        side_effect=_init_storage,
    ), patch(
        "graphgen.operators.sample_subgraph_family_llm.sample_subgraph_family_llm_service.init_llm",
        return_value=_EmptyBootstrapLLM(),
    ):
        service = SampleSubgraphFamilyLLMService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            graph_backend="networkx",
            allow_bootstrap_fallback=False,
        )

    rows, _ = service.process([{"_trace_id": "bootstrap-protocol-error"}])
    result = rows[0]
    assert result["abstained"] is True
    assert result["selected_subgraphs"] == []
    assert all(
        session["termination_reason"] == "bootstrap_protocol_error"
        for session in result["family_sessions"]
    )
    assert all(session["abstained"] is True for session in result["family_sessions"])
    assert all(
        session["bootstrap_error_count"] == 1 for session in result["family_sessions"]
    )
    assert all(
        bundle["protocol_failures"][0]["error_type"] == "parse_error"
        for bundle in result["candidate_bundle"]
    )
    terminal_reasons = [
        item["termination_reason"]
        for item in result["family_termination_trace"]
        if item["stage"] == "terminal"
    ]
    assert terminal_reasons == [
        "bootstrap_protocol_error",
        "bootstrap_protocol_error",
        "bootstrap_protocol_error",
    ]


def test_family_llm_sampler_ignores_legacy_empty_bootstrap_keep_list(tmp_path: Path):
    graph_storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_visual_core_graph(graph_storage)

    def _init_storage(backend: str, working_dir: str, namespace: str):
        if namespace == "graph":
            return graph_storage
        return _DummyKV()

    with patch(
        "graphgen.operators.sample_subgraph_family_llm.sample_subgraph_family_llm_service.init_storage",
        side_effect=_init_storage,
    ), patch(
        "graphgen.operators.sample_subgraph_family_llm.sample_subgraph_family_llm_service.init_llm",
        return_value=_EmptyKeepBootstrapLLM(),
    ):
        service = SampleSubgraphFamilyLLMService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            graph_backend="networkx",
            allow_bootstrap_fallback=False,
        )

    rows, _ = service.process([{"_trace_id": "bootstrap-empty"}])
    result = rows[0]
    assert result["abstained"] is False
    assert {item["qa_family"] for item in result["selected_subgraphs"]} == {
        "atomic",
        "aggregated",
        "multi_hop",
    }
    assert all(session["termination_reason"] == "accepted" for session in result["family_sessions"])
    assert all(not bundle["protocol_failures"] for bundle in result["candidate_bundle"])


def test_family_llm_sampler_stops_on_repeated_invalid_selector_choice(tmp_path: Path):
    graph_storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_visual_core_graph(graph_storage)

    def _init_storage(backend: str, working_dir: str, namespace: str):
        if namespace == "graph":
            return graph_storage
        return _DummyKV()

    with patch(
        "graphgen.operators.sample_subgraph_family_llm.sample_subgraph_family_llm_service.init_storage",
        side_effect=_init_storage,
    ), patch(
        "graphgen.operators.sample_subgraph_family_llm.sample_subgraph_family_llm_service.init_llm",
        return_value=_InvalidSelectorLLM(),
    ):
        service = SampleSubgraphFamilyLLMService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            graph_backend="networkx",
            max_steps_per_family=3,
            max_selector_errors=2,
        )

    rows, _ = service.process([{"_trace_id": "invalid-selector"}])
    result = rows[0]
    sessions = {item["qa_family"]: item for item in result["family_sessions"]}
    assert sessions["atomic"]["status"] == "accepted"
    assert sessions["aggregated"]["termination_reason"] == "invalid_selection_repeated"
    assert sessions["multi_hop"]["termination_reason"] == "invalid_selection_repeated"
    assert sessions["aggregated"]["invalid_candidate_repeat_count"] >= 2
    invalid_events = [
        item
        for item in result["family_selection_trace"]
        if item["qa_family"] == "aggregated" and item["decision"] == "invalid_selection"
    ]
    assert len(invalid_events) == 2


def test_family_llm_sampler_terminates_on_judge_protocol_error(tmp_path: Path):
    graph_storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_visual_core_graph(graph_storage)

    def _init_storage(backend: str, working_dir: str, namespace: str):
        if namespace == "graph":
            return graph_storage
        return _DummyKV()

    with patch(
        "graphgen.operators.sample_subgraph_family_llm.sample_subgraph_family_llm_service.init_storage",
        side_effect=_init_storage,
    ), patch(
        "graphgen.operators.sample_subgraph_family_llm.sample_subgraph_family_llm_service.init_llm",
        return_value=_MissingJudgeScoresLLM(),
    ):
        service = SampleSubgraphFamilyLLMService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            graph_backend="networkx",
        )

    rows, _ = service.process([{"_trace_id": "judge-protocol-error"}])
    result = rows[0]
    sessions = {item["qa_family"]: item for item in result["family_sessions"]}
    assert sessions["aggregated"]["termination_reason"] == "judge_protocol_error"
    assert sessions["aggregated"]["judge_error_count"] == 1
    aggregated_bundle = next(
        bundle for bundle in result["candidate_bundle"] if bundle["qa_family"] == "aggregated"
    )
    assert aggregated_bundle["protocol_failures"][0]["error_type"] == "schema_error"


def test_family_llm_sampler_rejects_shallow_multi_hop_accept(tmp_path: Path):
    graph_storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_visual_core_graph(graph_storage)

    def _init_storage(backend: str, working_dir: str, namespace: str):
        if namespace == "graph":
            return graph_storage
        return _DummyKV()

    with patch(
        "graphgen.operators.sample_subgraph_family_llm.sample_subgraph_family_llm_service.init_storage",
        side_effect=_init_storage,
    ), patch(
        "graphgen.operators.sample_subgraph_family_llm.sample_subgraph_family_llm_service.init_llm",
        return_value=_ShallowMultiHopAcceptLLM(),
    ):
        service = SampleSubgraphFamilyLLMService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            graph_backend="networkx",
            max_steps_per_family=1,
            min_multi_hop_outside_core_edges=2,
        )

    rows, _ = service.process([{"_trace_id": "shallow-multi-hop"}])
    result = rows[0]
    selected_families = {item["qa_family"] for item in result["selected_subgraphs"]}
    assert "multi_hop" not in selected_families
    multi_hop_session = next(
        item for item in result["family_sessions"] if item["qa_family"] == "multi_hop"
    )
    assert multi_hop_session["termination_reason"] == "max_steps_reached"
    accept_event = next(
        item
        for item in result["family_termination_trace"]
        if item["qa_family"] == "multi_hop"
        and item["stage"] == "selection"
        and item["decision"] == "accept"
    )
    assert accept_event["state"]["current_outside_depth"] == 1


def test_family_llm_sampler_remains_stable_under_chaotic_llm_outputs(tmp_path: Path):
    graph_storage = NetworkXStorage(working_dir=str(tmp_path / "cache"), namespace="graph")
    _build_visual_core_graph(graph_storage)

    def _init_storage(backend: str, working_dir: str, namespace: str):
        if namespace == "graph":
            return graph_storage
        return _DummyKV()

    with patch(
        "graphgen.operators.sample_subgraph_family_llm.sample_subgraph_family_llm_service.init_storage",
        side_effect=_init_storage,
    ), patch(
        "graphgen.operators.sample_subgraph_family_llm.sample_subgraph_family_llm_service.init_llm",
        return_value=_ChaoticVisualCoreLLM(),
    ):
        service = SampleSubgraphFamilyLLMService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            graph_backend="networkx",
            max_steps_per_family=1,
            allow_bootstrap_fallback=False,
            min_multi_hop_outside_core_edges=2,
        )

    rows, _ = service.process([{"_trace_id": "chaotic-llm"}])
    result = rows[0]
    assert result["abstained"] is False
    sessions = {item["qa_family"]: item for item in result["family_sessions"]}
    assert sessions["atomic"]["termination_reason"] == "accepted"
    assert sessions["aggregated"]["termination_reason"] == "invalid_selection_repeated"
    assert sessions["multi_hop"]["termination_reason"] == "max_steps_reached"
    terminal_events = [
        item for item in result["family_termination_trace"] if item["stage"] == "terminal"
    ]
    assert len(terminal_events) == 3
