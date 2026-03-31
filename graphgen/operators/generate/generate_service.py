import copy
import json
import re
from typing import Tuple

from graphgen.bases import BaseKVStorage, BaseLLMWrapper, BaseOperator
from graphgen.common.init_llm import init_llm
from graphgen.common.init_storage import init_storage
from graphgen.utils import logger, run_concurrent


class GenerateService(BaseOperator):
    """
    Generate question-answer pairs from selected subgraphs or direct graph slices.
    """

    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        method: str = "aggregated",
        data_format: str = "ChatML",
        **generate_kwargs,
    ):
        super().__init__(
            working_dir=working_dir, kv_backend=kv_backend, op_name="generate"
        )
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.generate_storage: BaseKVStorage = init_storage(
            backend=kv_backend, working_dir=working_dir, namespace="generate"
        )

        self.method = method
        self.data_format = data_format
        self.generator_map = {}

        if self.method == "atomic":
            from graphgen.models import AtomicGenerator

            self.generator = AtomicGenerator(self.llm_client)
        elif self.method == "aggregated":
            from graphgen.models import AggregatedGenerator

            self.generator = AggregatedGenerator(self.llm_client)
        elif self.method == "multi_hop":
            from graphgen.models import MultiHopGenerator

            self.generator = MultiHopGenerator(self.llm_client)
        elif self.method == "multi_hop_vqa":
            from graphgen.models import MultiHopVQAGenerator

            self.generator = MultiHopVQAGenerator(self.llm_client)
        elif self.method == "cot":
            from graphgen.models import CoTGenerator

            self.generator = CoTGenerator(self.llm_client)
        elif self.method == "vqa":
            from graphgen.models import VQAGenerator

            self.generator = VQAGenerator(self.llm_client)
        elif self.method == "aggregated_vqa":
            from graphgen.models import AggregatedVQAGenerator

            self.generator = AggregatedVQAGenerator(self.llm_client)
        elif self.method == "auto":
            from graphgen.models import (
                AggregatedVQAGenerator,
                MultiHopVQAGenerator,
                VQAGenerator,
            )

            self.generator = None
            self.generator_map = {
                "aggregated": AggregatedVQAGenerator(self.llm_client),
                "multi_hop": MultiHopVQAGenerator(self.llm_client),
                "vqa": VQAGenerator(self.llm_client),
            }
        elif self.method == "multi_choice":
            from graphgen.models import MultiChoiceGenerator

            self.generator = MultiChoiceGenerator(
                self.llm_client,
                num_of_questions=generate_kwargs.get("num_of_questions", 5),
            )
        elif self.method == "multi_answer":
            from graphgen.models import MultiAnswerGenerator

            self.generator = MultiAnswerGenerator(
                self.llm_client,
                num_of_questions=generate_kwargs.get("num_of_questions", 3),
            )
        elif self.method == "fill_in_blank":
            from graphgen.models import FillInBlankGenerator

            self.generator = FillInBlankGenerator(
                self.llm_client,
                num_of_questions=generate_kwargs.get("num_of_questions", 5),
            )
        elif self.method == "true_false":
            from graphgen.models import TrueFalseGenerator

            self.generator = TrueFalseGenerator(
                self.llm_client,
                num_of_questions=generate_kwargs.get("num_of_questions", 5),
            )
        else:
            raise ValueError(f"Unsupported generation mode: {method}")

    def process(self, batch: list) -> Tuple[list, dict]:
        """
        Generate question-answer pairs from selected subgraphs or direct graph slices.
        """
        logger.info("[Generation] mode: %s, batches: %d", self.method, len(batch))
        if self.method == "auto":
            tasks = self._build_auto_tasks(batch)

            async def _generate_task(task):
                return await self._generate_auto_qa_pairs(task)

            results = run_concurrent(
                _generate_task,
                tasks,
                desc="Generating QAs",
                unit="batch",
            )
        else:
            triples = [(item["nodes"], item["edges"]) for item in batch]
            results = run_concurrent(
                self.generator.generate,
                triples,
                desc="Generating QAs",
                unit="batch",
            )

        meta_updates = {}
        final_results = []
        if self.method == "auto":
            for task, generated_pairs in zip(tasks, results):
                if not generated_pairs:
                    continue
                item = task["item"]
                input_trace_id = item["_trace_id"]
                subgraph_item = task["subgraph_item"]
                sub_graph = {
                    "nodes": copy.deepcopy(subgraph_item.get("nodes", [])),
                    "edges": copy.deepcopy(subgraph_item.get("edges", [])),
                }
                sub_graph_summary = self._build_sub_graph_summary(
                    subgraph_item.get("nodes", []), subgraph_item.get("edges", [])
                )
                for generated in generated_pairs:
                    generator_key = generated["generator_key"]
                    qa_pair = generated["qa_pair"]
                    formatter = self.generator_map.get(
                        generator_key, self.generator_map["aggregated"]
                    )
                    res = formatter.format_generation_results(
                        qa_pair, output_data_format=self.data_format
                    )
                    res["sub_graph"] = json.dumps(sub_graph, ensure_ascii=False)
                    res["sub_graph_summary"] = json.dumps(
                        sub_graph_summary, ensure_ascii=False
                    )
                    self._attach_common_metadata(
                        res=res,
                        item=item,
                        subgraph_item=subgraph_item,
                        generator_key=generator_key,
                    )
                    res["_trace_id"] = self.get_trace_id(res)
                    final_results.append(res)
                    meta_updates.setdefault(input_trace_id, []).append(res["_trace_id"])
        else:
            for item, qa_pairs in zip(batch, results):
                if not qa_pairs:
                    continue
                input_trace_id = item["_trace_id"]
                sub_graph = {
                    "nodes": copy.deepcopy(item.get("nodes", [])),
                    "edges": copy.deepcopy(item.get("edges", [])),
                }
                sub_graph_summary = self._build_sub_graph_summary(
                    item.get("nodes", []), item.get("edges", [])
                )
                for qa_pair in qa_pairs:
                    res = self.generator.format_generation_results(
                        qa_pair, output_data_format=self.data_format
                    )
                    res["sub_graph"] = json.dumps(sub_graph, ensure_ascii=False)
                    res["sub_graph_summary"] = json.dumps(
                        sub_graph_summary, ensure_ascii=False
                    )
                    self._attach_common_metadata(
                        res=res,
                        item=item,
                        subgraph_item=item,
                        generator_key=item.get(
                            "generator_key", item.get("task_type", self.method)
                        ),
                    )
                    res["_trace_id"] = self.get_trace_id(res)
                    final_results.append(res)
                    meta_updates.setdefault(input_trace_id, []).append(res["_trace_id"])
        return final_results, meta_updates

    def split(self, batch):
        to_process, recovered = super().split(batch)
        if not recovered.empty and "sub_graph" in recovered.columns:
            recovered = recovered.copy()
            recovered["sub_graph"] = recovered["sub_graph"].apply(
                lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
            )
        if not recovered.empty and "sub_graph_summary" in recovered.columns:
            recovered = recovered.copy()
            recovered["sub_graph_summary"] = recovered["sub_graph_summary"].apply(
                lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
            )
        if not recovered.empty and "sub_graph" in recovered.columns:
            recovered = recovered.copy()
            recovered["sub_graph_summary"] = recovered.apply(
                lambda row: row.get("sub_graph_summary")
                if row.get("sub_graph_summary")
                else self._build_summary_from_serialized_sub_graph(row.get("sub_graph")),
                axis=1,
            )
        return to_process, recovered

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

    @classmethod
    def _build_summary_from_serialized_sub_graph(cls, sub_graph) -> str | None:
        if not sub_graph:
            return None
        try:
            parsed = json.loads(sub_graph) if isinstance(sub_graph, str) else sub_graph
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(parsed, dict):
            return None
        summary = cls._build_sub_graph_summary(
            parsed.get("nodes", []), parsed.get("edges", [])
        )
        return json.dumps(summary, ensure_ascii=False)

    def _build_auto_tasks(self, batch: list) -> list[dict]:
        tasks = []
        for item in batch:
            selected_subgraphs = item.get("selected_subgraphs")
            if not isinstance(selected_subgraphs, list) or not selected_subgraphs:
                continue

            max_qas = min(
                3,
                max(1, int(item.get("max_vqas_per_selected_subgraph", 2))),
            )
            for selected in selected_subgraphs:
                if not isinstance(selected, dict):
                    continue
                nodes = selected.get("nodes")
                edges = selected.get("edges")
                if nodes is None or edges is None or not nodes:
                    continue
                generator_keys = self._resolve_generator_keys(
                    selected.get("approved_question_types", []),
                    degraded=bool(selected.get("degraded")),
                    max_qas=max_qas,
                )
                tasks.append(
                    {
                        "item": item,
                        "subgraph_item": selected,
                        "generator_keys": generator_keys,
                        "triple": (nodes, edges),
                        "max_qas": max_qas,
                    }
                )
        return tasks

    async def _generate_auto_qa_pairs(self, task: dict) -> list[dict]:
        triple = task["triple"]
        max_qas = max(1, int(task.get("max_qas", 1)))
        generator_keys = task.get("generator_keys", [])
        generated = []
        seen = set()

        for generator_key in generator_keys:
            if len(generated) >= max_qas:
                break
            generator = self.generator_map.get(generator_key) or self.generator_map[
                "aggregated"
            ]
            qa_pairs = await generator.generate(triple)
            accepted_pairs = self._dedupe_qa_pairs(
                qa_pairs,
                max_qas=max_qas - len(generated),
                seen=seen,
            )
            for qa_pair in accepted_pairs:
                generated.append(
                    {
                        "generator_key": generator_key,
                        "qa_pair": qa_pair,
                    }
                )
                if len(generated) >= max_qas:
                    break
        return generated

    def _attach_common_metadata(
        self,
        *,
        res: dict,
        item: dict,
        subgraph_item: dict,
        generator_key: str,
    ) -> None:
        if item.get("seed_node_id"):
            res["seed_node_id"] = item["seed_node_id"]
        if item.get("seed_image_path"):
            res["seed_image_path"] = item["seed_image_path"]
        if item.get("selection_mode"):
            res["selection_mode"] = item["selection_mode"]
        if item.get("degraded") is not None:
            res["degraded"] = bool(
                subgraph_item.get("degraded", item.get("degraded", False))
            )
        if item.get("degraded_reason"):
            res["degraded_reason"] = item["degraded_reason"]
        if item.get("candidate_bundle") is not None:
            res["candidate_bundle"] = json.dumps(
                item.get("candidate_bundle", []), ensure_ascii=False
            )
        if subgraph_item.get("subgraph_id"):
            res["subgraph_id"] = subgraph_item["subgraph_id"]
        if subgraph_item.get("technical_focus"):
            res["technical_focus"] = subgraph_item["technical_focus"]
        if subgraph_item.get("image_grounding_summary"):
            res["image_grounding_summary"] = subgraph_item["image_grounding_summary"]
        if subgraph_item.get("evidence_summary"):
            res["evidence_summary"] = subgraph_item["evidence_summary"]
        if subgraph_item.get("judge_scores"):
            res["judge_scores"] = json.dumps(
                subgraph_item["judge_scores"], ensure_ascii=False
            )
        if subgraph_item.get("approved_question_types"):
            res["approved_question_types"] = json.dumps(
                subgraph_item["approved_question_types"], ensure_ascii=False
            )

        if generator_key:
            res["generator_key"] = generator_key
            res["task_type"] = generator_key

    def _resolve_generator_keys(
        self,
        approved_question_types: list,
        *,
        degraded: bool,
        max_qas: int,
    ) -> list[str]:
        question_types = [
            str(item).strip().lower()
            for item in approved_question_types
            if str(item).strip()
        ]
        generator_keys = []
        for question_type in question_types:
            if any(token in question_type for token in ("multi_hop", "causal", "constraint")):
                generator_key = "multi_hop"
            elif any(token in question_type for token in ("chart", "diagram", "parameter", "comparison", "interpretation")):
                generator_key = "aggregated"
            else:
                generator_key = "vqa"
            if generator_key not in generator_keys:
                generator_keys.append(generator_key)
        if degraded and "aggregated" not in generator_keys:
            generator_keys.insert(0, "aggregated")
        if not generator_keys:
            generator_keys = ["aggregated"]
        return generator_keys[:max_qas]

    @staticmethod
    def _normalize_signature(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip().lower())

    def _dedupe_qa_pairs(
        self,
        qa_pairs: list[dict],
        *,
        max_qas: int,
        seen: set[str] | None = None,
    ) -> list[dict]:
        deduped = []
        seen = seen if seen is not None else set()
        for qa_pair in qa_pairs or []:
            if not isinstance(qa_pair, dict):
                continue
            question = qa_pair.get("question", "")
            answer = qa_pair.get("answer", "")
            signature = (
                f"{self._normalize_signature(question)}|"
                f"{self._normalize_signature(answer)}"
            )
            if not question or not answer or signature in seen:
                continue
            seen.add(signature)
            deduped.append(qa_pair)
            if len(deduped) >= max_qas:
                break
        return deduped
