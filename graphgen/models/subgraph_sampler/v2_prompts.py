import json
from typing import Any

from graphgen.bases import BaseGraphStorage

from .artifacts import compact_text
from .v2_artifacts import CandidateSubgraphState, JudgeFeedback


def build_v2_neighborhood_prompt(
    graph: BaseGraphStorage, neighborhood: dict[str, Any]
) -> str:
    lines = ["Nodes:"]
    for index, node_id in enumerate(neighborhood.get("node_ids", [])[:24], start=1):
        node_data = graph.get_node(node_id) or {}
        lines.append(
            f"{index}. {node_id} | type={node_data.get('entity_type', '')} "
            f"| distance={neighborhood.get('distances', {}).get(node_id, 0)} "
            f"| desc={compact_text(node_data.get('description', ''), limit=120)} "
            f"| evidence={compact_text(node_data.get('evidence_span', ''), limit=80)}"
        )
    lines.append("Edges:")
    for index, (src_id, tgt_id, edge_data) in enumerate(
        neighborhood.get("edges", [])[:30], start=1
    ):
        lines.append(
            f"{index}. {src_id} -- {tgt_id} | rel={edge_data.get('relation_type', '')} "
            f"| desc={compact_text(edge_data.get('description', ''), limit=120)} "
            f"| evidence={compact_text(edge_data.get('evidence_span', ''), limit=80)}"
        )
    return "\n".join(lines)


def build_v2_candidate_prompt(
    graph: BaseGraphStorage, state: CandidateSubgraphState
) -> str:
    lines = [
        f"Intent: {state.intent}",
        f"Technical focus: {state.technical_focus}",
        f"Approved question types: {json.dumps(state.approved_question_types, ensure_ascii=False)}",
        f"Current units: {state.unit_count()}",
        "Nodes:",
    ]
    for node_id in state.node_ids:
        node_data = graph.get_node(node_id) or {}
        lines.append(
            f"- {node_id} | type={node_data.get('entity_type', '')} "
            f"| desc={compact_text(node_data.get('description', ''), limit=120)}"
        )
    lines.append("Edges:")
    for src_id, tgt_id in state.edge_pairs:
        edge_data = graph.get_edge(src_id, tgt_id) or graph.get_edge(tgt_id, src_id) or {}
        lines.append(
            f"- {src_id} -- {tgt_id} | rel={edge_data.get('relation_type', '')} "
            f"| desc={compact_text(edge_data.get('description', ''), limit=120)}"
        )
    return "\n".join(lines)


def build_v2_editor_prompt(
    *,
    seed_node_id: str,
    image_path: str,
    degraded: bool,
    hard_cap_units: int,
    round_index: int,
    allowed_question_types: list[str],
    current_state: CandidateSubgraphState,
    neighborhood_prompt: str,
    last_judge_feedback: JudgeFeedback | None,
) -> str:
    judge_payload = (
        json.dumps(last_judge_feedback.to_dict(), ensure_ascii=False)
        if last_judge_feedback
        else "{}"
    )
    return (
        "ROLE: EditorV2\n"
        "You are a graph-editing agent for image-grounded technical VQA.\n"
        "Edit the current candidate subgraph using only the provided neighborhood.\n"
        f"Seed node id: {seed_node_id}\n"
        f"Image path: {image_path}\n"
        f"Degraded mode: {str(degraded).lower()}\n"
        f"Round index: {round_index}\n"
        f"Allowed question types: {json.dumps(allowed_question_types, ensure_ascii=False)}\n"
        f"Hard cap units: {hard_cap_units}\n"
        "Valid actions: query_nodes, query_edges, add_node, add_edge, remove_node, "
        "remove_edge, revise_intent, commit_for_judgement.\n"
        "Current candidate state:\n"
        f"{json.dumps(current_state.to_dict(), ensure_ascii=False)}\n"
        "Last judge feedback:\n"
        f"{judge_payload}\n"
        "Available neighborhood:\n"
        f"{neighborhood_prompt}\n"
        "Return strict JSON:\n"
        '{\n'
        '  "intent": "string",\n'
        '  "technical_focus": "string",\n'
        '  "approved_question_types": ["..."],\n'
        '  "image_grounding_summary": "string",\n'
        '  "evidence_summary": "string",\n'
        '  "actions": [\n'
        '    {"action_type": "add_node", "node_id": "..."},\n'
        '    {"action_type": "add_edge", "src_id": "...", "tgt_id": "..."},\n'
        '    {"action_type": "remove_node", "node_id": "..."},\n'
        '    {"action_type": "remove_edge", "src_id": "...", "tgt_id": "..."},\n'
        '    {"action_type": "revise_intent", "intent": "...", "technical_focus": "...", "approved_question_types": ["..."]},\n'
        '    {"action_type": "query_nodes", "note": "optional"},\n'
        '    {"action_type": "query_edges", "note": "optional"},\n'
        '    {"action_type": "commit_for_judgement"}\n'
        "  ]\n"
        "}\n"
    )


def build_v2_judge_prompt(
    *,
    seed_node_id: str,
    image_path: str,
    degraded: bool,
    current_state: CandidateSubgraphState,
    candidate_prompt: str,
    judge_pass_threshold: float,
) -> str:
    return (
        "ROLE: JudgeV2\n"
        "Evaluate whether the current edited subgraph is sufficient for high-quality image-grounded technical VQA.\n"
        "Use normalized scores from 0 to 1. High hallucination_risk is bad.\n"
        f"Seed node id: {seed_node_id}\n"
        f"Image path: {image_path}\n"
        f"Degraded mode: {str(degraded).lower()}\n"
        f"Judge pass threshold: {judge_pass_threshold}\n"
        "Current candidate:\n"
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
        '  "passes": false,\n'
        '  "sufficient": false,\n'
        '  "needs_expansion": false,\n'
        '  "rejection_reason": "",\n'
        '  "suggested_actions": ["remove_node", "revise_intent"]\n'
        "}\n"
    )
