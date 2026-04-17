import copy
import json
import random
import re
from collections import Counter
from typing import Any, Tuple

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
        self.include_source_chunk_context = bool(
            generate_kwargs.get("include_source_chunk_context", False)
        )
        self.source_chunk_context_count = max(
            1, int(generate_kwargs.get("source_chunk_context_count", 3))
        )
        self.source_chunk_max_chars = max(
            100, int(generate_kwargs.get("source_chunk_max_chars", 600))
        )
        self.chunk_storage: BaseKVStorage = init_storage(
            backend=kv_backend,
            working_dir=working_dir,
            namespace="chunk",
        )
        self.tree_chunk_storage: BaseKVStorage = init_storage(
            backend=kv_backend,
            working_dir=working_dir,
            namespace="tree_chunk",
        )

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
        elif self.method == "atomic_vqa":
            from graphgen.models import AtomicVQAGenerator

            self.generator = AtomicVQAGenerator(self.llm_client)
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
                AtomicVQAGenerator,
                AggregatedVQAGenerator,
                MultiHopVQAGenerator,
                VQAGenerator,
            )

            self.generator = None
            self.generator_map = {
                "atomic": AtomicVQAGenerator(self.llm_client),
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
            triples = [
                self._build_generation_triple(item["nodes"], item["edges"])
                for item in batch
            ]
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
                sub_graph = self._serialize_sub_graph_payload(subgraph_item)
                sub_graph_summary = self._build_sub_graph_summary(
                    sub_graph.get("nodes", []), sub_graph.get("edges", [])
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
                    self._attach_qa_visualization_trace(
                        res=res,
                        item=item,
                        subgraph_item=subgraph_item,
                        generator_key=generator_key,
                        qa_pair=qa_pair,
                        sub_graph_summary=sub_graph_summary,
                    )
                    final_results.append(res)
                    meta_updates.setdefault(input_trace_id, []).append(res["_trace_id"])
        else:
            for item, qa_pairs in zip(batch, results):
                if not qa_pairs:
                    continue
                input_trace_id = item["_trace_id"]
                sub_graph = self._serialize_sub_graph_payload(item)
                sub_graph_summary = self._build_sub_graph_summary(
                    sub_graph.get("nodes", []), sub_graph.get("edges", [])
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

            for selected in selected_subgraphs:
                if not isinstance(selected, dict):
                    continue
                max_qas = min(
                    3,
                    max(
                        1,
                        int(
                            selected.get(
                                "target_qa_count",
                                item.get("max_vqas_per_selected_subgraph", 2),
                            )
                        ),
                    ),
                )
                normalized_subgraph = self._serialize_sub_graph_payload(selected)
                nodes = normalized_subgraph.get("nodes")
                edges = normalized_subgraph.get("edges")
                if nodes is None or edges is None or not nodes:
                    continue
                generator_keys = self._resolve_generator_keys(
                    selected.get("approved_question_types", []),
                    qa_family=selected.get("qa_family"),
                    degraded=bool(selected.get("degraded")),
                    max_qas=max_qas,
                )
                tasks.append(
                    {
                        "item": item,
                        "subgraph_item": {
                            **selected,
                            "nodes": nodes,
                            "edges": edges,
                        },
                        "generator_keys": generator_keys,
                        "triple": self._build_generation_triple(nodes, edges),
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
        if subgraph_item.get("qa_family"):
            res["qa_family"] = str(subgraph_item["qa_family"])

        if generator_key:
            res["generator_key"] = generator_key
            res["task_type"] = generator_key

    def _attach_qa_visualization_trace(
        self,
        *,
        res: dict,
        item: dict,
        subgraph_item: dict,
        generator_key: str,
        qa_pair: dict,
        sub_graph_summary: dict,
    ) -> None:
        trace = self._normalize_visualization_trace(
            item.get("visualization_trace"),
            item=item,
        )
        if not trace:
            return
        trace = copy.deepcopy(trace)
        graph_catalog = trace.setdefault("graph_catalog", {"nodes": {}, "edges": {}})
        self._add_subgraph_to_visualization_catalog(graph_catalog, subgraph_item)
        events = trace.setdefault("events", [])
        order = len(events) + 1
        subgraph_id = str(subgraph_item.get("subgraph_id", ""))
        qa_family = str(subgraph_item.get("qa_family", ""))
        event = {
            "event_id": f"{subgraph_id or item.get('seed_node_id', 'subgraph')}:{order}:qa_generated",
            "order": order,
            "qa_family": qa_family,
            "phase": "generation",
            "event_type": "qa_generated",
            "status": "generated",
            "selected_node_ids": [
                node[0]
                for node in subgraph_item.get("nodes", [])
                if isinstance(node, (list, tuple)) and node
            ],
            "selected_edge_pairs": [
                [edge[0], edge[1]]
                for edge in subgraph_item.get("edges", [])
                if isinstance(edge, (list, tuple)) and len(edge) >= 2
            ],
            "candidate_pool": [],
            "chosen_candidate": {},
            "judge": {},
            "reason": "",
            "termination_reason": "",
            "subgraph_id": subgraph_id,
            "generator_key": generator_key,
            "question": qa_pair.get("question", ""),
            "answer": qa_pair.get("answer", ""),
            "qa_trace_id": res.get("_trace_id", ""),
            "sub_graph_summary": sub_graph_summary,
        }
        events.append(event)
        res["visualization_trace"] = json.dumps(trace, ensure_ascii=False)

    @staticmethod
    def _normalize_visualization_trace(value: Any, *, item: dict) -> dict | None:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = None
        if isinstance(value, dict):
            trace = copy.deepcopy(value)
        elif item.get("sampler_version") == "family_llm_v2" or any(
            isinstance(selected, dict) and selected.get("qa_family")
            for selected in (item.get("selected_subgraphs") or [])
        ):
            trace = {
                "schema_version": "visual_core_family_timeline_v1",
                "sampler_version": item.get("sampler_version", "family_llm_v2"),
                "seed_node_id": item.get("seed_node_id", ""),
                "seed_image_path": item.get("seed_image_path", ""),
                "graph_catalog": {"nodes": {}, "edges": {}},
                "events": [],
            }
        else:
            return None
        trace.setdefault("schema_version", "visual_core_family_timeline_v1")
        trace.setdefault("sampler_version", item.get("sampler_version", "family_llm_v2"))
        trace.setdefault("seed_node_id", item.get("seed_node_id", ""))
        trace.setdefault("seed_image_path", item.get("seed_image_path", ""))
        trace.setdefault("graph_catalog", {"nodes": {}, "edges": {}})
        trace.setdefault("events", [])
        return trace

    @classmethod
    def _add_subgraph_to_visualization_catalog(
        cls,
        graph_catalog: dict,
        subgraph_item: dict,
    ) -> None:
        nodes = graph_catalog.setdefault("nodes", {})
        edges = graph_catalog.setdefault("edges", {})
        for node in subgraph_item.get("nodes", []):
            if isinstance(node, (list, tuple)) and len(node) >= 2:
                node_id = str(node[0])
                nodes[node_id] = {
                    "node_id": node_id,
                    **cls._normalize_sub_graph_value(node[1] or {}),
                }
        for edge in subgraph_item.get("edges", []):
            if isinstance(edge, (list, tuple)) and len(edge) >= 3:
                src_id = str(edge[0])
                tgt_id = str(edge[1])
                edges[f"{src_id}->{tgt_id}"] = {
                    "source": src_id,
                    "target": tgt_id,
                    **cls._normalize_sub_graph_value(edge[2] or {}),
                }

    def _resolve_generator_keys(
        self,
        approved_question_types: list,
        *,
        qa_family: str | None = None,
        degraded: bool,
        max_qas: int,
    ) -> list[str]:
        normalized_family = str(qa_family or "").strip().lower()
        if normalized_family in {"atomic", "aggregated", "multi_hop"}:
            return [normalized_family]

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
    def _normalize_signature(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip().lower())

    @staticmethod
    def _split_source_ids(value: Any) -> list[str]:
        if not value:
            return []
        return [part.strip() for part in str(value).split("<SEP>") if part.strip()]

    def _build_generation_triple(
        self,
        nodes: list[tuple[str, dict]],
        edges: list[tuple[Any, Any, dict]],
    ) -> tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]:
        if not self.include_source_chunk_context:
            return nodes, edges
        source_nodes = self._build_source_chunk_prompt_nodes(nodes, edges)
        if not source_nodes:
            return nodes, edges
        return [*nodes, *source_nodes], edges

    def _build_source_chunk_prompt_nodes(
        self,
        nodes: list[tuple[str, dict]],
        edges: list[tuple[Any, Any, dict]],
    ) -> list[tuple[str, dict]]:
        source_ids = self._collect_ranked_source_ids(nodes, edges)
        if not source_ids:
            return []
        selected_ids = source_ids[: self.source_chunk_context_count]
        prompt_nodes = []
        for index, source_id in enumerate(selected_ids, start=1):
            content = self._get_source_chunk_content(source_id)
            if not content:
                continue
            prompt_nodes.append(
                (
                    f"__SOURCE_CHUNK_{index}",
                    {
                        "entity_type": "SOURCE_CHUNK",
                        "entity_name": source_id,
                        "description": content[: self.source_chunk_max_chars],
                        "source_id": source_id,
                    },
                )
            )
        return prompt_nodes

    def _collect_ranked_source_ids(
        self,
        nodes: list[tuple[str, dict]],
        edges: list[tuple[Any, Any, dict]],
    ) -> list[str]:
        counter: Counter[str] = Counter()
        for _, node_data in nodes:
            for source_id in self._split_source_ids(node_data.get("source_id")):
                if self._get_source_chunk_content(source_id):
                    counter[source_id] += 1
        for edge in edges:
            if len(edge) < 3:
                continue
            edge_data = edge[2] if isinstance(edge[2], dict) else {}
            for source_id in self._split_source_ids(edge_data.get("source_id")):
                if self._get_source_chunk_content(source_id):
                    counter[source_id] += 1

        if not counter:
            return []

        same_source_ids = [source_id for source_id, freq in counter.items() if freq > 1]
        other_source_ids = [source_id for source_id, freq in counter.items() if freq <= 1]
        random.shuffle(same_source_ids)
        random.shuffle(other_source_ids)
        ranked = sorted(
            same_source_ids, key=lambda source_id: counter[source_id], reverse=True
        )
        if len(ranked) >= self.source_chunk_context_count:
            return ranked
        return [*ranked, *other_source_ids]

    def _get_source_chunk_content(self, source_id: str) -> str:
        for storage in (self.tree_chunk_storage, self.chunk_storage):
            record = storage.get_by_id(source_id)
            if isinstance(record, dict):
                content = str(record.get("content", "")).strip()
                if content:
                    return content
            elif isinstance(record, str) and record.strip():
                return record.strip()
        return ""

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
