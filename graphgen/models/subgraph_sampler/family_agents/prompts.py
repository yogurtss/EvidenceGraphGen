import json


FAMILY_RULES = {
    "atomic": (
        "Prefer a minimal closure around one direct fact. "
        "Stay at one hop, keep only one supporting node, and avoid density-driven expansion."
    ),
    "aggregated": (
        "Prefer breadth around one coherent theme. "
        "Only merge same-theme sibling or near-sibling nodes and avoid long chains."
    ),
    "multi_hop": (
        "Prefer depth and an explicit reasoning chain. "
        "When moving deeper, prune same-layer siblings and expand only from the active frontier."
    ),
}


def build_family_agent_prompt(
    *,
    qa_family: str,
    state_payload: dict,
    candidate_pool_payload: list[dict],
) -> str:
    return (
        "ROLE: FamilySubgraphAgent\n"
        f"QA family: {qa_family}\n"
        f"Rule: {FAMILY_RULES[qa_family]}\n"
        "Current state:\n"
        f"{json.dumps(state_payload, ensure_ascii=False)}\n"
        "Candidate pool:\n"
        f"{json.dumps(candidate_pool_payload, ensure_ascii=False)}\n"
    )


def build_family_generation_judge_prompt(
    *,
    qa_family: str,
    subgraph_payload: dict,
    qa_pair: dict,
) -> str:
    return (
        "ROLE: FamilyQAJudge\n"
        f"QA family: {qa_family}\n"
        f"Rule: {FAMILY_RULES[qa_family]}\n"
        "Evaluate the QA against the family-specific subgraph.\n"
        "Return strict JSON with keys: decision, confidence, reason, "
        "qa_revision_instruction, subgraph_revision_instruction.\n"
        "Allowed decisions: accept, revise_qa_only, refine_subgraph_then_regenerate, reject.\n"
        f"Subgraph:\n{json.dumps(subgraph_payload, ensure_ascii=False)}\n"
        f"QA:\n{json.dumps(qa_pair, ensure_ascii=False)}\n"
    )


def build_family_qa_revision_prompt(
    *,
    qa_family: str,
    subgraph_payload: dict,
    qa_pair: dict,
    qa_revision_instruction: str,
) -> str:
    return (
        "ROLE: FamilyQARevision\n"
        f"QA family: {qa_family}\n"
        f"Rule: {FAMILY_RULES[qa_family]}\n"
        "Rewrite the QA so it aligns with the subgraph and the revision instruction.\n"
        "Return strict JSON with question and answer.\n"
        f"Revision instruction: {qa_revision_instruction}\n"
        f"Subgraph:\n{json.dumps(subgraph_payload, ensure_ascii=False)}\n"
        f"Current QA:\n{json.dumps(qa_pair, ensure_ascii=False)}\n"
    )


def build_family_subgraph_edit_prompt(
    *,
    qa_family: str,
    state_payload: dict,
    candidate_pool_payload: list[dict],
    revision_reason: str = "",
) -> str:
    return (
        "ROLE: FamilySubgraphEdit\n"
        f"QA family: {qa_family}\n"
        f"Rule: {FAMILY_RULES[qa_family]}\n"
        "Choose at most one action for this round.\n"
        "Return strict JSON with keys: decision, candidate_node_id, reason, confidence.\n"
        "Allowed decisions: select_candidate, stop.\n"
        f"Revision reason: {revision_reason}\n"
        f"Current state:\n{json.dumps(state_payload, ensure_ascii=False)}\n"
        f"Candidate pool:\n{json.dumps(candidate_pool_payload, ensure_ascii=False)}\n"
    )


def build_family_subgraph_judge_prompt(
    *,
    qa_family: str,
    state_payload: dict,
) -> str:
    return (
        "ROLE: FamilySubgraphJudge\n"
        f"QA family: {qa_family}\n"
        f"Rule: {FAMILY_RULES[qa_family]}\n"
        "Evaluate whether the state is sufficient for this family.\n"
        "Return strict JSON with keys: decision, sufficient, rejection_reason, suggested_action, scores.\n"
        "scores must include: image_indispensability, answer_stability, evidence_closure, technical_relevance, reasoning_depth, hallucination_risk, theme_coherence, overall_score.\n"
        "decision should be accept or reject.\n"
        f"State:\n{json.dumps(state_payload, ensure_ascii=False)}\n"
    )
