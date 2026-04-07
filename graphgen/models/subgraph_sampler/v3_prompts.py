import json

from .v2_artifacts import CandidateSubgraphState, JudgeFeedback


FAMILY_EDITOR_GUIDANCE = {
    "atomic": (
        "Target a compact subgraph for one atomic image-grounded QA. "
        "Prefer a single directly supported fact, parameter readout, or one-hop relation from the visual seed. "
        "Do not over-expand."
    ),
    "aggregated": (
        "Target an aggregated image-grounded QA. "
        "Collect same-theme neighbors that jointly support a coherent technical explanation around one intent."
    ),
    "multi_hop": (
        "Target a multi-hop image-grounded QA. "
        "Expand toward a verifiable reasoning chain with at least two edges, and prefer deeper evidence closure over shallow readout."
    ),
}


FAMILY_JUDGE_GUIDANCE = {
    "atomic": (
        "Accept only if the candidate supports a single precise question with direct evidence and low structural complexity."
    ),
    "aggregated": (
        "Accept only if the candidate supports integrated explanation across multiple related neighbors without drifting off-theme."
    ),
    "multi_hop": (
        "Accept only if the candidate contains a clear multi-step reasoning chain and the answer depends on combining multiple edges."
    ),
}


def build_v3_editor_prompt(
    *,
    qa_family: str,
    seed_node_id: str,
    image_path: str,
    hard_cap_units: int,
    round_index: int,
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
        "ROLE: EditorV3\n"
        "You are a family-aware graph-editing agent for image-grounded technical VQA.\n"
        f"QA family: {qa_family}\n"
        f"Family guidance: {FAMILY_EDITOR_GUIDANCE[qa_family]}\n"
        f"Seed node id: {seed_node_id}\n"
        f"Image path: {image_path}\n"
        f"Round index: {round_index}\n"
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
        '  "image_grounding_summary": "string",\n'
        '  "evidence_summary": "string",\n'
        '  "actions": [\n'
        '    {"action_type": "add_node", "node_id": "..."},\n'
        '    {"action_type": "add_edge", "src_id": "...", "tgt_id": "..."},\n'
        '    {"action_type": "remove_node", "node_id": "..."},\n'
        '    {"action_type": "remove_edge", "src_id": "...", "tgt_id": "..."},\n'
        '    {"action_type": "revise_intent", "intent": "...", "technical_focus": "..."},\n'
        '    {"action_type": "query_nodes", "note": "optional"},\n'
        '    {"action_type": "query_edges", "note": "optional"},\n'
        '    {"action_type": "commit_for_judgement"}\n'
        "  ]\n"
        "}\n"
    )


def build_v3_judge_prompt(
    *,
    qa_family: str,
    seed_node_id: str,
    image_path: str,
    current_state: CandidateSubgraphState,
    candidate_prompt: str,
    judge_pass_threshold: float,
) -> str:
    return (
        "ROLE: JudgeV3\n"
        "Evaluate whether the current edited subgraph is sufficient for the requested image-grounded technical QA family.\n"
        f"QA family: {qa_family}\n"
        f"Family guidance: {FAMILY_JUDGE_GUIDANCE[qa_family]}\n"
        f"Seed node id: {seed_node_id}\n"
        f"Image path: {image_path}\n"
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
        '  "suggested_actions": ["add_node"]\n'
        "}\n"
    )
