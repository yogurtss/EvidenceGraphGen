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

JUDGE_RULES = {
    "atomic": (
        "Accept only when the graph has the image plus one grounded evidence node."
        " Do not ask for further expansion."
    ),
    "aggregated": (
        "Judge topic coherence and whether the remaining candidate pool still has"
        " useful same-topic evidence. Do not focus on a fixed node count."
    ),
    "multi_hop": (
        "Judge whether the selected graph forms a single same-direction chain deep"
        " enough beyond the visual core, or should continue along that chain."
    ),
}


def build_bootstrap_prompt(*, qa_family: str, seed_payload: dict[str, Any], visual_core_candidates: list[str], preview_candidates: list[str], runtime_schema: dict[str, Any]) -> str:
    return (
        "ROLE: VisualCoreBootstrap\n"
        f"QA family: {qa_family}\n"
        f"Family rule: {FAMILY_RULES[qa_family]}\n"
        "Use the image and the compact seed-local graph neighborhood to define the"
        " family intent and constraints.\n"
        "Important: image-adjacent candidates are analysis-only visual core context."
        " Treat them as part of the image node itself; concrete selector operations"
        " start from the logical first layer shown in preview candidate lines.\n"
        "Do not choose nodes here. Return strict JSON with keys: intent,"
        " technical_focus, forbidden_patterns, image_grounding_summary,"
        " bootstrap_rationale.\n"
        f"Seed payload:\n{json.dumps(seed_payload, ensure_ascii=False)}\n"
        f"Image-adjacent visual core lines:\n{json.dumps(visual_core_candidates, ensure_ascii=False)}\n"
        f"Logical first-layer preview lines:\n{json.dumps(preview_candidates, ensure_ascii=False)}\n"
        f"Runtime schema:\n{json.dumps(runtime_schema, ensure_ascii=False)}\n"
    )


def build_shared_intent_planner_prompt(
    *,
    target_count: int,
    seed_payload: dict[str, Any],
    first_layer_candidates: list[str],
    preview_candidates: list[str],
    runtime_schema: dict[str, Any],
) -> str:
    return (
        "ROLE: VisualCoreSharedIntentPlanner\n"
        f"Target intent count: {int(target_count)}\n"
        "Create diverse visual-core sampling intents that can be reused by atomic,"
        " aggregated, and multi_hop samplers. First-layer image-derived entities are"
        " selectable anchors; selector steps will later adapt each intent to the QA"
        " family rule.\n"
        "Family rules:\n"
        f"{json.dumps(FAMILY_RULES, ensure_ascii=False)}\n"
        "Return strict JSON with one key: intents. intents must be an array with"
        " exactly the target count. Each intent object must contain keys: intent,"
        " technical_focus, forbidden_patterns, image_grounding_summary,"
        " bootstrap_rationale.\n"
        f"Seed payload:\n{json.dumps(seed_payload, ensure_ascii=False)}\n"
        f"Selectable first-layer candidate lines:\n{json.dumps(first_layer_candidates, ensure_ascii=False)}\n"
        f"Second-layer preview lines:\n{json.dumps(preview_candidates, ensure_ascii=False)}\n"
        f"Runtime schema:\n{json.dumps(runtime_schema, ensure_ascii=False)}\n"
    )


def build_selector_prompt(*, qa_family: str, state_payload: dict[str, Any], candidate_pool_payload: list[str]) -> str:
    return (
        "ROLE: FamilyNodeSelector\n"
        f"QA family: {qa_family}\n"
        f"Family rule: {FAMILY_RULES[qa_family]}\n"
        "Choose at most one next node from the compact candidate lines.\n"
        "Return strict JSON with keys: decision, candidate_node_id, reason, confidence.\n"
        "Allowed decisions: select_candidate, stop_selection.\n"
        f"Current state:\n{json.dumps(state_payload, ensure_ascii=False)}\n"
        f"Candidate lines:\n{json.dumps(candidate_pool_payload, ensure_ascii=False)}\n"
    )


def build_termination_prompt(*, qa_family: str, state_payload: dict[str, Any], stage: str, last_selected_candidate: dict[str, Any] | None, candidate_pool_payload: list[str]) -> str:
    return (
        "ROLE: FamilyTerminationJudge\n"
        f"QA family: {qa_family}\n"
        f"Family rule: {FAMILY_RULES[qa_family]}\n"
        f"Judge rule: {JUDGE_RULES[qa_family]}\n"
        "Evaluate whether the current subgraph is sufficient, needs another step, or should"
        " rollback the last step.\n"
        "Return strict JSON with keys: decision, sufficient, termination_reason, reason,"
        " suggested_action, scores.\n"
        "Allowed decisions: continue, accept, rollback_last_step, reject.\n"
        "scores must include: image_indispensability, answer_stability, evidence_closure,"
        " technical_relevance, reasoning_depth, hallucination_risk, theme_coherence,"
        " overall_score.\n"
        f"Stage: {stage}\n"
        f"Current state:\n{json.dumps(state_payload, ensure_ascii=False)}\n"
        f"Last selected candidate:\n{json.dumps(last_selected_candidate or {}, ensure_ascii=False)}\n"
        f"Current candidate pool:\n{json.dumps(candidate_pool_payload, ensure_ascii=False)}\n"
    )
