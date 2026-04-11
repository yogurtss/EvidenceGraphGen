import json
from typing import Any

FAMILY_RULES = {
    "atomic": (
        "Keep the visual core minimal. Use the image plus one direct visual fact and avoid"
        " second-layer expansion."
    ),
    "aggregated": (
        "Keep a coherent visual topic. Bootstrap from one relevant first-hop analysis anchor,"
        " then allow same-direction breadth and only go deeper when it clearly improves"
        " coherence."
    ),
    "multi_hop": (
        "Bootstrap one first-hop anchor from the visual core, then extend a single"
        " same-direction reasoning chain beyond the core."
    ),
}


def build_bootstrap_prompt(*, qa_family: str, seed_payload: dict[str, Any], visual_core_candidates: list[dict[str, Any]], preview_candidates: list[dict[str, Any]], runtime_schema: dict[str, Any]) -> str:
    return (
        "ROLE: VisualCoreBootstrap\n"
        f"QA family: {qa_family}\n"
        f"Family rule: {FAMILY_RULES[qa_family]}\n"
        "Use the image and the seed-local graph neighborhood to bootstrap one high-quality"
        " family-specific subgraph.\n"
        "Important: first-hop visual core candidates are analysis-only image anchors."
        " They are used only for schema/theme analysis and must never become QA evidence"
        " nodes or be selected as operation targets.\n"
        "Pick exactly ONE first-hop anchor for analysis; all concrete graph operations must"
        " start from second-hop (or deeper) candidates.\n"
        "Use keep_first_hop_node_ids/drop_first_hop_node_ids only to select analysis anchors.\n"
        "Return strict JSON with keys: intent, technical_focus, keep_first_hop_node_ids,"
        " drop_first_hop_node_ids, preferred_entity_types, preferred_relation_types,"
        " forbidden_patterns, target_reasoning_depth, image_grounding_summary,"
        " bootstrap_rationale.\n"
        f"Seed payload:\n{json.dumps(seed_payload, ensure_ascii=False)}\n"
        f"First-hop visual core candidates:\n{json.dumps(visual_core_candidates, ensure_ascii=False)}\n"
        f"Second-hop preview candidates:\n{json.dumps(preview_candidates, ensure_ascii=False)}\n"
        f"Runtime schema:\n{json.dumps(runtime_schema, ensure_ascii=False)}\n"
    )


def build_selector_prompt(*, qa_family: str, state_payload: dict[str, Any], candidate_pool_payload: list[dict[str, Any]]) -> str:
    return (
        "ROLE: FamilyNodeSelector\n"
        f"QA family: {qa_family}\n"
        f"Family rule: {FAMILY_RULES[qa_family]}\n"
        "Choose at most one next node from the candidate pool.\n"
        "Return strict JSON with keys: decision, candidate_uid, reason, confidence.\n"
        "Allowed decisions: select_candidate, stop_selection.\n"
        f"Current state:\n{json.dumps(state_payload, ensure_ascii=False)}\n"
        f"Candidate pool:\n{json.dumps(candidate_pool_payload, ensure_ascii=False)}\n"
    )


def build_termination_prompt(*, qa_family: str, state_payload: dict[str, Any], stage: str, last_selected_candidate: dict[str, Any] | None) -> str:
    return (
        "ROLE: FamilyTerminationJudge\n"
        f"QA family: {qa_family}\n"
        f"Family rule: {FAMILY_RULES[qa_family]}\n"
        "Evaluate whether the current subgraph is sufficient, needs another step, or should"
        " rollback the last iterative step.\n"
        "Return strict JSON with keys: decision, sufficient, termination_reason, reason,"
        " suggested_action, scores.\n"
        "Allowed decisions: continue, accept, rollback_last_step, reject.\n"
        "scores must include: image_indispensability, answer_stability, evidence_closure,"
        " technical_relevance, reasoning_depth, hallucination_risk, theme_coherence,"
        " overall_score.\n"
        f"Stage: {stage}\n"
        f"Current state:\n{json.dumps(state_payload, ensure_ascii=False)}\n"
        f"Last selected candidate:\n{json.dumps(last_selected_candidate or {}, ensure_ascii=False)}\n"
    )
