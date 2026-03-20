from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from .models import EvidenceItem, RunRecord, RunStats, SampleListItem, SamplePage, SampleRecord


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
        return ""
    if isinstance(value, list):
        texts = [_coerce_text(item) for item in value]
        return "\n".join([item for item in texts if item]).strip()
    return ""


def _extract_question(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return _coerce_text(message.get("content"))
    return ""


def _extract_answer(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            return _coerce_text(message.get("content"))
    return ""


def _extract_image_path(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("image"), str):
                    image_path = item["image"].strip()
                    if image_path:
                        return image_path
        elif isinstance(content, dict) and isinstance(content.get("image"), str):
            image_path = content["image"].strip()
            if image_path:
                return image_path
    return None


def _parse_json_blob(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, dict):
        return value, None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            return None, str(exc)
        if isinstance(parsed, dict):
            return parsed, None
    return None, "Unsupported sub_graph format"


def _parse_json_summary(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _summary_from_graph(sub_graph: dict[str, Any] | None) -> dict[str, Any] | None:
    if not sub_graph:
        return None
    nodes = sub_graph.get("nodes", [])
    edges = sub_graph.get("edges", [])
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_ids": [str(node[0]) for node in nodes[:10] if isinstance(node, list) and node],
        "edge_pairs": [
            f"{edge[0]} -> {edge[1]}"
            for edge in edges[:10]
            if isinstance(edge, list) and len(edge) >= 2
        ],
    }


def _extract_evidence_items(sub_graph: dict[str, Any] | None) -> list[EvidenceItem]:
    if not sub_graph:
        return []

    evidence_items: list[EvidenceItem] = []

    for node in sub_graph.get("nodes", []):
        if not isinstance(node, list) or len(node) < 2 or not isinstance(node[1], dict):
            continue
        label = str(node[0])
        metadata = node[1]
        evidence_span = str(metadata.get("evidence_span", "")).strip()
        if not evidence_span:
            continue
        evidence_items.append(
            EvidenceItem(
                kind="node",
                label=label,
                evidence_span=evidence_span,
                source_id=metadata.get("source_id"),
                description=metadata.get("description"),
            )
        )

    for edge in sub_graph.get("edges", []):
        if not isinstance(edge, list) or len(edge) < 3 or not isinstance(edge[2], dict):
            continue
        src_id, tgt_id, metadata = edge[0], edge[1], edge[2]
        label = str(metadata.get("relation_type") or f"{src_id} -> {tgt_id}")
        evidence_span = str(metadata.get("evidence_span", "")).strip()
        if not evidence_span:
            continue
        evidence_items.append(
            EvidenceItem(
                kind="edge",
                label=label,
                evidence_span=evidence_span,
                source_id=metadata.get("source_id"),
                description=metadata.get("description"),
            )
        )

    return evidence_items


def _resolve_asset_path(raw_path: str | None, record_file: Path, cwd: Path) -> str | None:
    if not raw_path:
        return None

    candidate = Path(raw_path)
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.extend([cwd / candidate, record_file.parent / candidate])

    for item in candidates:
        try:
            resolved = item.expanduser().resolve()
        except FileNotFoundError:
            continue
        if resolved.exists():
            return str(resolved)
    return None


class DataPlatformStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = (base_dir or Path.cwd()).resolve()
        self.runs: dict[str, RunRecord] = {}
        self.samples: dict[str, SampleRecord] = {}
        self.samples_by_run: dict[str, list[str]] = {}
        self.allowed_asset_paths: set[str] = set()

    def scan(self, root_path: str) -> tuple[list[RunRecord], int]:
        resolved_root = Path(root_path)
        if not resolved_root.is_absolute():
            resolved_root = (self.base_dir / resolved_root).resolve()

        if not resolved_root.exists() or not resolved_root.is_dir():
            raise FileNotFoundError(f"Directory not found: {resolved_root}")

        discovered_runs: dict[str, RunRecord] = {}
        discovered_samples: dict[str, SampleRecord] = {}
        discovered_samples_by_run: dict[str, list[str]] = {}
        discovered_assets: set[str] = set()

        for run_dir in sorted((resolved_root / "output").glob("*")):
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            generate_dir = run_dir / "generate"
            jsonl_files = sorted(generate_dir.glob("*.jsonl"))
            if not jsonl_files:
                continue

            config_path = run_dir / "config.yaml"
            task_type = "unknown"
            if config_path.exists():
                with config_path.open("r", encoding="utf-8") as handle:
                    config = yaml.safe_load(handle) or {}
                task_type = self._infer_task_type(config)
            else:
                config = {}

            run_samples: list[SampleRecord] = []
            entity_counter: Counter[str] = Counter()
            relation_counter: Counter[str] = Counter()
            evidence_total = 0
            evidence_with_span = 0

            for jsonl_file in jsonl_files:
                with jsonl_file.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        payload = json.loads(line)
                        sample = self._normalize_sample(
                            payload=payload,
                            run_id=run_id,
                            source_file=jsonl_file,
                            line_number=line_number,
                        )
                        if sample.image_path:
                            discovered_assets.add(sample.image_path)
                        if sample.sub_graph:
                            for node in sample.sub_graph.get("nodes", []):
                                if (
                                    isinstance(node, list)
                                    and len(node) >= 2
                                    and isinstance(node[1], dict)
                                ):
                                    entity_type = str(node[1].get("entity_type", "unknown"))
                                    entity_counter[entity_type] += 1
                                    evidence_total += 1
                                    if str(node[1].get("evidence_span", "")).strip():
                                        evidence_with_span += 1
                            for edge in sample.sub_graph.get("edges", []):
                                if (
                                    isinstance(edge, list)
                                    and len(edge) >= 3
                                    and isinstance(edge[2], dict)
                                ):
                                    relation_type = str(
                                        edge[2].get("relation_type", "unknown")
                                    )
                                    relation_counter[relation_type] += 1
                                    evidence_total += 1
                                    if str(edge[2].get("evidence_span", "")).strip():
                                        evidence_with_span += 1

                        discovered_samples[sample.sample_id] = sample
                        run_samples.append(sample)

            if not run_samples:
                continue

            stats = RunStats(
                question_texts=[sample.question for sample in run_samples if sample.question],
                answer_texts=[sample.answer for sample in run_samples if sample.answer],
                entity_type_counts=dict(entity_counter),
                relation_type_counts=dict(relation_counter),
                evidence_coverage=(
                    evidence_with_span / evidence_total if evidence_total else 0.0
                ),
            )

            run_record = RunRecord(
                run_id=run_id,
                root_path=str(resolved_root),
                config_path=str(config_path.resolve()) if config_path.exists() else None,
                generated_at=int(run_id) if run_id.isdigit() else None,
                sample_count=len(run_samples),
                task_type=task_type,
                has_image=any(sample.image_path for sample in run_samples),
                has_sub_graph=any(sample.sub_graph for sample in run_samples),
                stats=stats,
            )
            discovered_runs[run_id] = run_record
            discovered_samples_by_run[run_id] = [
                sample.sample_id for sample in sorted(run_samples, key=lambda item: item.sample_id)
            ]

        self.runs = discovered_runs
        self.samples = discovered_samples
        self.samples_by_run = discovered_samples_by_run
        self.allowed_asset_paths = discovered_assets
        return list(sorted(self.runs.values(), key=lambda item: item.run_id, reverse=True)), len(
            self.samples
        )

    def list_runs(self) -> list[RunRecord]:
        return list(sorted(self.runs.values(), key=lambda item: item.run_id, reverse=True))

    def list_samples(
        self,
        run_id: str,
        *,
        page: int = 1,
        page_size: int = 20,
        search: str | None = None,
        has_image: bool | None = None,
        has_graph: bool | None = None,
    ) -> SamplePage:
        if run_id not in self.samples_by_run:
            raise KeyError(run_id)

        sample_ids = self.samples_by_run[run_id]
        items = [self.samples[sample_id] for sample_id in sample_ids]

        if search:
            query = search.strip().lower()
            items = [
                item
                for item in items
                if query in item.question.lower() or query in item.answer.lower()
            ]
        if has_image is not None:
            items = [item for item in items if bool(item.image_path) is has_image]
        if has_graph is not None:
            items = [item for item in items if bool(item.sub_graph) is has_graph]

        total = len(items)
        start = max(0, (page - 1) * page_size)
        end = start + page_size
        paged_items = items[start:end]

        return SamplePage(
            items=[
                SampleListItem(
                    sample_id=item.sample_id,
                    run_id=item.run_id,
                    question=item.question,
                    answer_preview=item.answer[:140],
                    image_path=item.image_path,
                    node_count=(item.sub_graph_summary or {}).get("node_count", 0),
                    edge_count=(item.sub_graph_summary or {}).get("edge_count", 0),
                    has_graph=bool(item.sub_graph),
                )
                for item in paged_items
            ],
            total=total,
            page=page,
            page_size=page_size,
        )

    def get_sample(self, sample_id: str) -> SampleRecord:
        if sample_id not in self.samples:
            raise KeyError(sample_id)
        return self.samples[sample_id]

    def is_asset_allowed(self, asset_path: str) -> bool:
        return asset_path in self.allowed_asset_paths

    @staticmethod
    def _infer_task_type(config: dict[str, Any]) -> str:
        for node in config.get("nodes", []):
            if node.get("id") == "generate":
                params = node.get("params", {})
                if isinstance(params, dict):
                    return str(params.get("method", "unknown"))
        return "unknown"

    def _normalize_sample(
        self,
        *,
        payload: dict[str, Any],
        run_id: str,
        source_file: Path,
        line_number: int,
    ) -> SampleRecord:
        messages = payload.get("messages")
        question = _extract_question(messages)
        answer = _extract_answer(messages)
        raw_image_path = _extract_image_path(messages)
        image_path = _resolve_asset_path(raw_image_path, source_file, self.base_dir)
        sub_graph, graph_parse_error = _parse_json_blob(payload.get("sub_graph"))
        sub_graph_summary = _parse_json_summary(payload.get("sub_graph_summary"))
        if sub_graph and not sub_graph_summary:
            sub_graph_summary = _summary_from_graph(sub_graph)

        evidence_items = _extract_evidence_items(sub_graph)
        sample_key = payload.get("_trace_id") or f"{source_file}:{line_number}"
        sample_id = hashlib.sha1(f"{run_id}:{sample_key}".encode("utf-8")).hexdigest()[:16]

        return SampleRecord(
            sample_id=sample_id,
            run_id=run_id,
            source_file=str(source_file.resolve()),
            trace_id=payload.get("_trace_id"),
            question=question,
            answer=answer,
            image_path=image_path,
            sub_graph=sub_graph,
            sub_graph_summary=sub_graph_summary,
            evidence_items=evidence_items,
            raw_record=payload,
            graph_parse_error=graph_parse_error,
        )
