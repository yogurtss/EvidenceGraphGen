import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Tuple

from graphgen.bases import BaseGenerator, BaseGraphStorage, BaseLLMWrapper, BaseOperator
from graphgen.common.init_llm import init_llm
from graphgen.common.init_storage import init_storage
from graphgen.models import (
    AggregatedVQAGenerator,
    AtomicVQAGenerator,
    FamilySubgraphOrchestrator,
    MultiHopVQAGenerator,
)
from graphgen.models.subgraph_sampler.artifacts import extract_json_payload
from graphgen.models.subgraph_sampler.family_agents.prompts import (
    build_family_generation_judge_prompt,
    build_family_qa_revision_prompt,
)


@dataclass
class QAJudgeDecision:
    decision: str = "accept"
    confidence: float = 0.0
    reason: str = ""
    qa_revision_instruction: str = ""
    subgraph_revision_instruction: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": round(float(self.confidence), 4),
            "reason": self.reason,
            "qa_revision_instruction": self.qa_revision_instruction,
            "subgraph_revision_instruction": self.subgraph_revision_instruction,
        }


class GenerateAgenticVQAService(BaseOperator):
    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        graph_backend: str = "kuzu",
        data_format: str = "ChatML",
        max_graph_revision_rounds: int = 2,
        max_qa_revision_rounds: int = 2,
        max_total_rounds: int = 4,
        max_selected_per_family_for_generation: int = 2,
        max_selected_subgraphs_per_family: int = 3,
        judge_pass_threshold: float = 0.68,
        max_multi_hop_hops: int = 3,
    ):
        super().__init__(
            working_dir=working_dir,
            kv_backend=kv_backend,
            op_name="generate_agentic_vqa",
        )
        self.graph_storage: BaseGraphStorage = init_storage(
            backend=graph_backend,
            working_dir=working_dir,
            namespace="graph",
        )
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.data_format = data_format
        self.max_graph_revision_rounds = max(0, int(max_graph_revision_rounds))
        self.max_qa_revision_rounds = max(0, int(max_qa_revision_rounds))
        self.max_total_rounds = max(1, int(max_total_rounds))
        self.max_selected_per_family_for_generation = max(
            1, int(max_selected_per_family_for_generation)
        )
        self.orchestrator = FamilySubgraphOrchestrator(
            self.graph_storage,
            self.llm_client,
            max_selected_subgraphs_per_family=max_selected_subgraphs_per_family,
            judge_pass_threshold=judge_pass_threshold,
            max_multi_hop_hops=max_multi_hop_hops,
        )
        self.generator_map: dict[str, BaseGenerator] = {
            "atomic": AtomicVQAGenerator(self.llm_client),
            "aggregated": AggregatedVQAGenerator(self.llm_client),
            "multi_hop": MultiHopVQAGenerator(self.llm_client),
        }

    def process(self, batch: list) -> Tuple[list, dict]:
        self.graph_storage.reload()
        final_results = []
        meta_updates = {}
        for item in batch:
            selected_subgraphs = self._select_top_subgraphs(item)
            for selected_subgraph in selected_subgraphs:
                accepted = self._run_generation_session(item, selected_subgraph)
                if not accepted:
                    continue
                accepted["_trace_id"] = self.get_trace_id(accepted)
                final_results.append(accepted)
                meta_updates.setdefault(item["_trace_id"], []).append(accepted["_trace_id"])
        return final_results, meta_updates

    def _select_top_subgraphs(self, item: dict) -> list[dict]:
        selected_subgraphs = item.get("selected_subgraphs", [])
        grouped = {}
        for selected in selected_subgraphs:
            if not isinstance(selected, dict):
                continue
            qa_family = str(selected.get("qa_family", "")).strip().lower()
            if qa_family not in {"atomic", "aggregated", "multi_hop"}:
                continue
            grouped.setdefault(qa_family, []).append(copy.deepcopy(selected))
        picked = []
        for qa_family, family_subgraphs in grouped.items():
            family_subgraphs.sort(
                key=lambda row: (
                    -float(row.get("judge_scores", {}).get("overall_score", 0.0)),
                    row.get("subgraph_id", ""),
                )
            )
            for selected in family_subgraphs[: self.max_selected_per_family_for_generation]:
                selected["seed_node_id"] = item.get("seed_node_id", selected.get("seed_node_id"))
                picked.append(selected)
        return picked

    def _run_generation_session(
        self, item: dict, selected_subgraph: dict[str, Any]
    ) -> dict[str, Any] | None:
        qa_family = str(selected_subgraph.get("qa_family", "")).strip().lower()
        generator = self.generator_map.get(qa_family)
        if generator is None:
            return None

        current_subgraph = copy.deepcopy(selected_subgraph)
        graph_rounds = 0
        qa_rounds = 0
        total_rounds = 0
        generation_trace = []
        qa_judge_trace = []
        generation_session_id = (
            f"{item.get('seed_node_id', 'seed')}-{qa_family}-{selected_subgraph.get('subgraph_id', 'candidate')}"
        )
        termination_reason = "max_total_rounds_exhausted"

        while total_rounds < self.max_total_rounds:
            qa_pairs = self._generate_for_subgraph(generator, current_subgraph)
            if not qa_pairs:
                termination_reason = "generator_empty"
                break
            qa_pair = qa_pairs[0]
            generation_trace.append(
                {
                    "round_index": total_rounds + 1,
                    "stage": "generate",
                    "qa_family": qa_family,
                    "subgraph_revision_id": int(current_subgraph.get("revision_id", 0)),
                    "question": qa_pair.get("question", ""),
                    "answer": qa_pair.get("answer", ""),
                }
            )
            decision = self._judge_qa(current_subgraph, qa_pair)
            qa_judge_trace.append(
                {
                    "round_index": total_rounds + 1,
                    "qa_family": qa_family,
                    "subgraph_revision_id": int(current_subgraph.get("revision_id", 0)),
                    **decision.to_dict(),
                }
            )

            if decision.decision == "accept":
                termination_reason = "accepted"
                return self._build_output_row(
                    item=item,
                    subgraph_item=current_subgraph,
                    qa_pair=qa_pair,
                    generator_key=qa_family,
                    generation_session_id=generation_session_id,
                    generation_trace=generation_trace,
                    qa_judge_trace=qa_judge_trace,
                    termination_reason=termination_reason,
                    formatter=generator,
                )

            if (
                decision.decision == "revise_qa_only"
                and qa_rounds < self.max_qa_revision_rounds
            ):
                qa_rounds += 1
                total_rounds += 1
                revised_qa = self._revise_qa(current_subgraph, qa_pair, decision)
                if revised_qa:
                    generation_trace.append(
                        {
                            "round_index": total_rounds,
                            "stage": "revise_qa",
                            "qa_family": qa_family,
                            "subgraph_revision_id": int(current_subgraph.get("revision_id", 0)),
                            "question": revised_qa.get("question", ""),
                            "answer": revised_qa.get("answer", ""),
                        }
                    )
                    revised_decision = self._judge_qa(current_subgraph, revised_qa)
                    qa_judge_trace.append(
                        {
                            "round_index": total_rounds,
                            "qa_family": qa_family,
                            "subgraph_revision_id": int(current_subgraph.get("revision_id", 0)),
                            **revised_decision.to_dict(),
                        }
                    )
                    if revised_decision.decision == "accept":
                        termination_reason = "accepted_after_qa_revision"
                        return self._build_output_row(
                            item=item,
                            subgraph_item=current_subgraph,
                            qa_pair=revised_qa,
                            generator_key=qa_family,
                            generation_session_id=generation_session_id,
                            generation_trace=generation_trace,
                            qa_judge_trace=qa_judge_trace,
                            termination_reason=termination_reason,
                            formatter=generator,
                        )
                    decision = revised_decision
                else:
                    termination_reason = "qa_revision_failed"
                    break

            if (
                decision.decision == "refine_subgraph_then_regenerate"
                and graph_rounds < self.max_graph_revision_rounds
            ):
                graph_rounds += 1
                total_rounds += 1
                revised_subgraph = self.orchestrator.continue_subgraph(
                    selected_subgraph=current_subgraph,
                    revision_reason=decision.subgraph_revision_instruction or decision.reason,
                )
                if not revised_subgraph:
                    termination_reason = "no_refinement_available"
                    break
                current_subgraph = revised_subgraph
                generation_trace.append(
                    {
                        "round_index": total_rounds,
                        "stage": "refine_subgraph",
                        "qa_family": qa_family,
                        "subgraph_revision_id": int(current_subgraph.get("revision_id", 0)),
                        "reason": decision.subgraph_revision_instruction or decision.reason,
                    }
                )
                continue

            termination_reason = decision.decision or "rejected"
            break

        return None

    def _generate_for_subgraph(
        self,
        generator: BaseGenerator,
        selected_subgraph: dict[str, Any],
    ) -> list[dict]:
        nodes = selected_subgraph.get("nodes", [])
        edges = selected_subgraph.get("edges", [])
        if not nodes:
            return []
        return __import__("asyncio").run(generator.generate((nodes, edges)))

    def _judge_qa(
        self,
        selected_subgraph: dict[str, Any],
        qa_pair: dict[str, Any],
    ) -> QAJudgeDecision:
        qa_family = str(selected_subgraph.get("qa_family", "")).strip().lower()
        prompt = build_family_generation_judge_prompt(
            qa_family=qa_family,
            subgraph_payload=self._subgraph_prompt_payload(selected_subgraph),
            qa_pair=qa_pair,
        )
        raw = self.llm_client.generate_answer(prompt)
        if hasattr(raw, "__await__"):
            raw = __import__("asyncio").run(raw)
        payload = extract_json_payload(raw) if isinstance(raw, str) else {}
        decision = str(payload.get("decision", "")).strip().lower()
        if decision in {
            "accept",
            "revise_qa_only",
            "refine_subgraph_then_regenerate",
            "reject",
        }:
            return QAJudgeDecision(
                decision=decision,
                confidence=float(payload.get("confidence", 0.0) or 0.0),
                reason=str(payload.get("reason", "")).strip(),
                qa_revision_instruction=str(
                    payload.get("qa_revision_instruction", "")
                ).strip(),
                subgraph_revision_instruction=str(
                    payload.get("subgraph_revision_instruction", "")
                ).strip(),
            )
        return self._fallback_qa_decision(selected_subgraph, qa_pair)

    def _fallback_qa_decision(
        self, selected_subgraph: dict[str, Any], qa_pair: dict[str, Any]
    ) -> QAJudgeDecision:
        question = str(qa_pair.get("question", "")).strip()
        answer = str(qa_pair.get("answer", "")).strip()
        if not question or not answer:
            return QAJudgeDecision(
                decision="reject",
                confidence=0.1,
                reason="missing_question_or_answer",
            )
        qa_family = str(selected_subgraph.get("qa_family", "")).strip().lower()
        edge_count = len(selected_subgraph.get("edges", []))
        if qa_family == "multi_hop" and edge_count < 2:
            return QAJudgeDecision(
                decision="refine_subgraph_then_regenerate",
                confidence=0.55,
                reason="need_deeper_reasoning_chain",
                subgraph_revision_instruction="Extend the active frontier to add one more reasoning step.",
            )
        return QAJudgeDecision(decision="accept", confidence=0.75, reason="heuristic_accept")

    def _revise_qa(
        self,
        selected_subgraph: dict[str, Any],
        qa_pair: dict[str, Any],
        decision: QAJudgeDecision,
    ) -> dict[str, Any] | None:
        qa_family = str(selected_subgraph.get("qa_family", "")).strip().lower()
        prompt = build_family_qa_revision_prompt(
            qa_family=qa_family,
            subgraph_payload=self._subgraph_prompt_payload(selected_subgraph),
            qa_pair=qa_pair,
            qa_revision_instruction=decision.qa_revision_instruction or decision.reason,
        )
        raw = self.llm_client.generate_answer(prompt)
        if hasattr(raw, "__await__"):
            raw = __import__("asyncio").run(raw)
        payload = extract_json_payload(raw) if isinstance(raw, str) else {}
        question = str(payload.get("question", "")).strip()
        answer = str(payload.get("answer", "")).strip()
        if not question or not answer:
            question = self._extract_tagged_content(raw, "question")
            answer = self._extract_tagged_content(raw, "answer")
        if not question or not answer:
            return None
        revised = dict(qa_pair)
        revised["question"] = question
        revised["answer"] = answer
        return revised

    @staticmethod
    def _extract_tagged_content(raw: str, tag: str) -> str:
        if not isinstance(raw, str):
            return ""
        match = re.search(rf"<{tag}>(.*?)</{tag}>", raw, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _build_output_row(
        self,
        *,
        item: dict[str, Any],
        subgraph_item: dict[str, Any],
        qa_pair: dict[str, Any],
        generator_key: str,
        generation_session_id: str,
        generation_trace: list[dict[str, Any]],
        qa_judge_trace: list[dict[str, Any]],
        termination_reason: str,
        formatter: BaseGenerator,
    ) -> dict[str, Any]:
        res = formatter.format_generation_results(
            qa_pair, output_data_format=self.data_format
        )
        sub_graph = self._serialize_sub_graph_payload(subgraph_item)
        sub_graph_summary = self._build_sub_graph_summary(
            sub_graph.get("nodes", []), sub_graph.get("edges", [])
        )
        res["sub_graph"] = json.dumps(sub_graph, ensure_ascii=False)
        res["sub_graph_summary"] = json.dumps(sub_graph_summary, ensure_ascii=False)
        res["qa_family"] = generator_key
        res["generator_key"] = generator_key
        res["task_type"] = generator_key
        res["subgraph_revision_id"] = int(subgraph_item.get("revision_id", 0))
        res["generation_session_id"] = generation_session_id
        res["generation_trace"] = json.dumps(generation_trace, ensure_ascii=False)
        res["qa_judge_trace"] = json.dumps(qa_judge_trace, ensure_ascii=False)
        res["termination_reason"] = termination_reason
        if item.get("seed_node_id"):
            res["seed_node_id"] = item["seed_node_id"]
        if item.get("seed_image_path"):
            res["seed_image_path"] = item["seed_image_path"]
        if subgraph_item.get("subgraph_id"):
            res["subgraph_id"] = subgraph_item["subgraph_id"]
        if subgraph_item.get("technical_focus"):
            res["technical_focus"] = subgraph_item["technical_focus"]
        if subgraph_item.get("theme_signature"):
            res["theme_signature"] = subgraph_item["theme_signature"]
        if subgraph_item.get("frontier_node_id"):
            res["frontier_node_id"] = subgraph_item["frontier_node_id"]
        if subgraph_item.get("judge_scores"):
            res["judge_scores"] = json.dumps(
                subgraph_item["judge_scores"], ensure_ascii=False
            )
        return res

    @staticmethod
    def _subgraph_prompt_payload(subgraph_item: dict[str, Any]) -> dict[str, Any]:
        return {
            "qa_family": subgraph_item.get("qa_family"),
            "subgraph_id": subgraph_item.get("subgraph_id"),
            "revision_id": subgraph_item.get("revision_id", 0),
            "node_ids": [str(node[0]) for node in subgraph_item.get("nodes", [])],
            "edge_pairs": [
                [str(edge[0]), str(edge[1])]
                for edge in subgraph_item.get("edges", [])
                if isinstance(edge, (list, tuple)) and len(edge) >= 2
            ],
            "frontier_node_id": subgraph_item.get("frontier_node_id", ""),
            "theme_signature": subgraph_item.get("theme_signature", ""),
            "candidate_pool_snapshot": subgraph_item.get("candidate_pool_snapshot", []),
        }

    @staticmethod
    def _normalize_sub_graph_value(value):
        try:
            return json.loads(json.dumps(value, ensure_ascii=False))
        except (TypeError, ValueError):
            return copy.deepcopy(value)

    @classmethod
    def _serialize_sub_graph_payload(cls, item: dict) -> dict:
        return {
            "nodes": cls._normalize_sub_graph_value(item.get("nodes", [])),
            "edges": cls._normalize_sub_graph_value(item.get("edges", [])),
        }

    @staticmethod
    def _build_sub_graph_summary(nodes: list, edges: list) -> dict:
        def _node_label(node) -> str:
            if not isinstance(node, (list, tuple)) or not node:
                return str(node)
            return str(node[0])

        def _edge_label(edge) -> str:
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                return str(edge)
            return f"{edge[0]} -> {edge[1]}"

        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_ids": [_node_label(node) for node in nodes[:10]],
            "edge_pairs": [_edge_label(edge) for edge in edges[:10]],
        }
