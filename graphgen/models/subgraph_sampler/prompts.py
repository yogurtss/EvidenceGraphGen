import json
from typing import Any

from graphgen.bases import BaseGraphStorage

from .artifacts import SubgraphCandidate, compact_text


def build_neighborhood_prompt(
    graph: BaseGraphStorage, neighborhood: dict[str, Any]
) -> str:
    lines = ["Nodes:"]
    for index, node_id in enumerate(neighborhood["node_ids"][:18], start=1):
        node_data = graph.get_node(node_id) or {}
        lines.append(
            f"{index}. {node_id} | type={node_data.get('entity_type', '')} "
            f"| distance={neighborhood['distances'].get(node_id, 0)} "
            f"| desc={compact_text(node_data.get('description', ''), limit=120)} "
            f"| evidence={compact_text(node_data.get('evidence_span', ''), limit=80)}"
        )
    lines.append("Edges:")
    for index, (src_id, tgt_id, edge_data) in enumerate(
        neighborhood["edges"][:20], start=1
    ):
        lines.append(
            f"{index}. {src_id} -- {tgt_id} | rel={edge_data.get('relation_type', '')} "
            f"| desc={compact_text(edge_data.get('description', ''), limit=120)} "
            f"| evidence={compact_text(edge_data.get('evidence_span', ''), limit=80)}"
        )
    return "\n".join(lines)


def build_candidate_prompt(
    graph: BaseGraphStorage, candidate: SubgraphCandidate
) -> str:
    lines = ["Nodes:"]
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


def build_planner_prompt(
    *,
    seed_node_id: str,
    seed_description: str,
    image_path: str,
    allowed_question_types: list[str],
    degraded: bool,
    neighborhood_prompt: str,
    candidate_pool_size: int,
) -> str:
    return (
        "ROLE: Planner\n"
        "You are planning high-quality technical VQA subgraphs for a technical document image.\n"
        "Return strict JSON with key `intents`.\n"
        f"Seed node id: {seed_node_id}\n"
        f"Seed description: {compact_text(seed_description)}\n"
        f"Image path: {image_path}\n"
        f"Allowed question types: {json.dumps(allowed_question_types, ensure_ascii=False)}\n"
        f"Need degraded mode: {str(degraded).lower()}\n"
        "Available neighborhood summary:\n"
        f"{neighborhood_prompt}\n"
        "JSON schema:\n"
        '{\n'
        '  "intents": [\n'
        '    {\n'
        '      "intent": "short sentence",\n'
        '      "technical_focus": "timing constraint / architecture relation / parameter comparison",\n'
        '      "question_types": ["..."],\n'
        '      "priority_keywords": ["..."]\n'
        "    }\n"
        "  ]\n"
        "}\n"
        f"Return at most {candidate_pool_size} intents."
    )


def build_assembler_prompt(
    *,
    seed_node_id: str,
    intent: dict[str, Any],
    degraded: bool,
    max_units: int,
    neighborhood_prompt: str,
) -> str:
    return (
        "ROLE: RetrieverAssembler\n"
        "Select a compact KG-only subgraph for one technical VQA intent.\n"
        "You must only use node ids and edge pairs from the provided neighborhood.\n"
        f"Seed node id: {seed_node_id}\n"
        f"Intent: {intent.get('intent', '')}\n"
        f"Technical focus: {intent.get('technical_focus', '')}\n"
        f"Preferred question types: {json.dumps(intent.get('question_types', []), ensure_ascii=False)}\n"
        f"Priority keywords: {json.dumps(intent.get('priority_keywords', []), ensure_ascii=False)}\n"
        f"Max units (nodes + edges): {max_units}\n"
        f"Need degraded mode: {str(degraded).lower()}\n"
        "Neighborhood:\n"
        f"{neighborhood_prompt}\n"
        "Return strict JSON:\n"
        '{\n'
        '  "technical_focus": "string",\n'
        '  "node_ids": ["seed", "..."],\n'
        '  "edge_pairs": [["a", "b"]],\n'
        '  "approved_question_types": ["..."],\n'
        '  "image_grounding_summary": "why the image is necessary",\n'
        '  "evidence_summary": "how the selected graph supports the question family"\n'
        "}\n"
    )


def build_judge_prompt(
    *,
    seed_node_id: str,
    degraded: bool,
    judge_pass_threshold: float,
    candidate: SubgraphCandidate,
    candidate_prompt: str,
) -> str:
    return (
        "ROLE: Judge\n"
        "Evaluate whether a technical VQA subgraph is strong enough for image-grounded QA.\n"
        "Use normalized scores from 0 to 1. High hallucination_risk is bad.\n"
        f"Seed node id: {seed_node_id}\n"
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
        '  "rejection_reason": ""\n'
        "}\n"
    )
