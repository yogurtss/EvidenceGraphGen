import json
from typing import Any

from graphgen.bases import BaseGraphStorage

from .artifacts import SubgraphCandidate, compact_text, load_metadata


def build_schema_guided_neighborhood_prompt(
    graph: BaseGraphStorage, neighborhood: dict[str, Any]
) -> str:
    lines = ["Nodes:"]
    for index, node_id in enumerate(neighborhood.get("node_ids", [])[:24], start=1):
        node_data = graph.get_node(node_id) or {}
        metadata = load_metadata(node_data.get("metadata"))
        path = metadata.get("path", "")
        lines.append(
            f"{index}. {node_id} | type={node_data.get('entity_type', '')} "
            f"| distance={neighborhood.get('distances', {}).get(node_id, 0)} "
            f"| path={compact_text(path, limit=80)} "
            f"| desc={compact_text(node_data.get('description', ''), limit=120)} "
            f"| evidence={compact_text(node_data.get('evidence_span', ''), limit=80)}"
        )
    lines.append("Edges:")
    for index, (src_id, tgt_id, edge_data) in enumerate(
        neighborhood.get("edges", [])[:28], start=1
    ):
        lines.append(
            f"{index}. {src_id} -- {tgt_id} | rel={edge_data.get('relation_type', '')} "
            f"| desc={compact_text(edge_data.get('description', ''), limit=120)} "
            f"| evidence={compact_text(edge_data.get('evidence_span', ''), limit=80)}"
        )
    return "\n".join(lines)


def build_schema_guided_candidate_prompt(
    *,
    graph: BaseGraphStorage,
    candidate: SubgraphCandidate,
    retrieval_stage: str,
    inferred_schema: dict[str, Any],
) -> str:
    lines = [
        f"Retrieval stage: {retrieval_stage}",
        "Inferred schema summary:",
        json.dumps(inferred_schema, ensure_ascii=False),
        "Nodes:",
    ]
    for node_id in candidate.node_ids:
        node_data = graph.get_node(node_id) or {}
        lines.append(
            f"- {node_id} | type={node_data.get('entity_type', '')} "
            f"| desc={compact_text(node_data.get('description', ''), limit=120)} "
            f"| evidence={compact_text(node_data.get('evidence_span', ''), limit=80)}"
        )
    lines.append("Edges:")
    for src_id, tgt_id in candidate.edge_pairs:
        edge_data = graph.get_edge(src_id, tgt_id) or graph.get_edge(tgt_id, src_id) or {}
        lines.append(
            f"- {src_id} -- {tgt_id} | rel={edge_data.get('relation_type', '')} "
            f"| desc={compact_text(edge_data.get('description', ''), limit=120)} "
            f"| evidence={compact_text(edge_data.get('evidence_span', ''), limit=80)}"
        )
    return "\n".join(lines)


def build_schema_guided_planner_prompt(
    *,
    seed_node_id: str,
    seed_description: str,
    image_path: str,
    inferred_schema: dict[str, Any],
    neighborhood_prompt: str,
    candidate_pool_size: int,
    allowed_question_types: list[str],
    degraded: bool,
) -> str:
    return (
        "ROLE: SchemaAwarePlanner\n"
        "You are planning high-quality image-grounded technical VQA intents.\n"
        "Do not answer questions. Decompose this image seed into 2-3 answerable intent families.\n"
        "Each intent should specify likely schema targets so retrieval can filter by type.\n"
        "Return strict JSON with key `intents`.\n"
        f"Seed node id: {seed_node_id}\n"
        f"Seed description: {compact_text(seed_description, limit=240)}\n"
        f"Image path: {image_path}\n"
        f"Need degraded mode: {str(degraded).lower()}\n"
        f"Allowed question types: {json.dumps(allowed_question_types, ensure_ascii=False)}\n"
        "Runtime inferred schema:\n"
        f"{json.dumps(inferred_schema, ensure_ascii=False)}\n"
        "Local neighborhood summary:\n"
        f"{neighborhood_prompt}\n"
        "JSON schema:\n"
        '{\n'
        '  "intents": [\n'
        '    {\n'
        '      "intent": "short sentence",\n'
        '      "technical_focus": "timing / architecture / comparison / constraint",\n'
        '      "question_types": ["..."],\n'
        '      "priority_keywords": ["..."],\n'
        '      "target_node_types": ["..."],\n'
        '      "target_relation_types": ["..."],\n'
        '      "required_modalities": ["image", "text"],\n'
        '      "evidence_requirements": ["same_section", "same_source"]\n'
        "    }\n"
        "  ]\n"
        "}\n"
        f"Return at most {candidate_pool_size} intents."
    )


def build_schema_guided_judge_prompt(
    *,
    seed_node_id: str,
    degraded: bool,
    judge_pass_threshold: float,
    retrieval_stage: str,
    candidate: SubgraphCandidate,
    candidate_prompt: str,
) -> str:
    return (
        "ROLE: SchemaGuidedJudge\n"
        "Evaluate whether a schema-guided technical VQA subgraph is strong enough for image-grounded QA.\n"
        "If the current candidate lacks enough closure but should be expanded to a broader retrieval stage, set `needs_expansion` to true.\n"
        "Use normalized scores from 0 to 1. High hallucination_risk is bad.\n"
        f"Seed node id: {seed_node_id}\n"
        f"Retrieval stage: {retrieval_stage}\n"
        f"Degraded mode: {str(degraded).lower()}\n"
        f"Judge pass threshold: {judge_pass_threshold}\n"
        f"Candidate intent: {candidate.intent}\n"
        f"Candidate technical focus: {candidate.technical_focus}\n"
        f"Candidate question types: {json.dumps(candidate.approved_question_types, ensure_ascii=False)}\n"
        "Candidate nodes and edges:\n"
        f"{candidate_prompt}\n"
        "Return strict JSON:\n"
        '{\n'
        '  "image_indispensability": 0.0,\n'
        '  "answer_stability": 0.0,\n'
        '  "evidence_closure": 0.0,\n'
        '  "technical_relevance": 0.0,\n'
        '  "reasoning_depth": 0.0,\n'
        '  "hallucination_risk": 0.0,\n'
        '  "theme_coherence": 0.0,\n'
        '  "overall_score": 0.0,\n'
        '  "passes": true,\n'
        '  "sufficient": true,\n'
        '  "needs_expansion": false,\n'
        '  "rejection_reason": "",\n'
        '  "suggested_actions": ["broaden to 2-hop", "add same-source text evidence"]\n'
        "}\n"
    )
