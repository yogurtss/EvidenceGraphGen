import json

from .v2_artifacts import CandidateSubgraphState, JudgeFeedback


FAMILY_EDITOR_GUIDANCE = {
    "atomic": (
        "Target a compact subgraph for one atomic image-grounded QA. "
        "Prefer a single-edge minimal closure first (seed + one supporting relation), "
        "then expand only if judge feedback explicitly requires it. "
        "Do not over-expand."
    ),
    "aggregated": (
        "Target an aggregated image-grounded QA. "
        "Prioritize breadth: collect same-theme sibling neighbors around one intent, "
        "and avoid collapsing into only one narrow chain."
    ),
    "multi_hop": (
        "Target a multi-hop image-grounded QA. "
        "Prioritize depth: follow one verifiable chain step-by-step with at least two edges, "
        "and avoid adding many same-layer side nodes. "
        "Keep edge expansion in the same outward direction from the seed node."
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
    selectable_node_prompt: str,
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
        "Valid actions: query_nodes, query_edges, add_node, remove_node, "
        "remove_edge, revise_intent, commit_for_judgement.\n"
        "Action policy: exactly ONE action per round. "
        "When action_type is add_node, choose node_id ONLY from `Selectable add_node options` below. "
        "Each selectable node is already bound to one or more valid (anchor_node_id, edge) pairs. "
        "Prefer providing anchor_node_id that matches one listed option. "
        "If your provided src_id/tgt_id is inconsistent, executor will ignore it and use the bound edge.\n"
        "Current candidate state:\n"
        f"{json.dumps(current_state.to_dict(), ensure_ascii=False)}\n"
        "Last judge feedback:\n"
        f"{judge_payload}\n"
        "Selectable add_node options (derived from current candidate + neighborhood):\n"
        f"{selectable_node_prompt}\n"
        "Available neighborhood:\n"
        f"{neighborhood_prompt}\n"
        "Return strict JSON:\n"
        '{\n'
        '  "intent": "string",\n'
        '  "technical_focus": "string",\n'
        '  "image_grounding_summary": "string",\n'
        '  "evidence_summary": "string",\n'
        '  "action": {\n'
        '    "action_type": "add_node | remove_node | remove_edge | revise_intent | query_nodes | query_edges | commit_for_judgement",\n'
        '    "node_id": "...",\n'
        '    "anchor_node_id": "...",\n'
        '    "src_id": "...",\n'
        '    "tgt_id": "...",\n'
        '    "intent": "...",\n'
        '    "technical_focus": "...",\n'
        '    "note": "optional"\n'
        "  }\n"
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
