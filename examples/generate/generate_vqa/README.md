# Generate VQAs

## DRAM-oriented high-quality VQA pipeline

This workflow is suitable for generating VQA training data from memory-system materials (e.g., DRAM timing diagrams, architecture figures, specs).

### 1) Prepare input
- Put your multimodal samples in JSON format (text + image path).
- Ensure each sample has enough textual context and image metadata so the graph builder can connect entities and relations.

### 2) Run generation
```bash
bash examples/generate/generate_vqa/generate_vqa.sh
```

### 2.1) Tree-pipeline VQA
If your source is structured markdown / MoDora-style content, use `tree_vqa_config.yaml`.
This variant runs `structure_analyze -> hierarchy_generate -> tree_construct -> tree_chunk -> build_grounded_tree_kg`
before partitioning, so image/table VQA samples are grounded by tree-local evidence spans.

### 2.2) Agentic subgraph reasoning
If you want the system to keep a vision-centered seed and then select a higher-value subgraph for
`aggregated` or `multi_hop`, use `agentic_subgraph_reasoning_config.yaml`.
This variant inserts:

```text
partition -> sample_subgraph -> generate(method=auto)
```

`sample_subgraph` now runs a VLM-driven planner / retriever-assembler / judge loop around each
image seed, constructs one or more explicit `selected_subgraphs`, and passes them to
`generate(method=auto)` for downstream question generation.

If you want the graph-editing `v2` workflow, use `agentic_subgraph_reasoning_v2_config.yaml`.
This variant inserts:

```text
partition -> sample_subgraph_v2 -> generate(method=auto)
```

If you want the family-aware `v3` workflow, use `agentic_subgraph_reasoning_v3_config.yaml`.
This variant inserts:

```text
build_grounded_tree_kg -> sample_subgraph_v3 -> generate(method=auto)
```

`sample_subgraph_v3` will independently try to produce:

- one `atomic` subgraph
- one `aggregated` subgraph
- one `multi_hop` subgraph

and then let `generate(method=auto)` route them by `qa_family` instead of guessing from free-form question types.
For the full design notes, see `docs/vlm_vqa/agentic_subgraph_v3.md`.

If you want the new visual-core LLM family workflow, use `agentic_family_llm_vqa_config.yaml`.
This variant inserts:

```text
build_grounded_tree_kg -> sample_subgraph_family_llm -> generate(method=auto)
```

`sample_subgraph_family_llm` first bootstraps family-specific first-hop analysis
anchors, then builds final QA subgraphs from a virtual image node connected to
second-layer evidence while a separate termination judge decides whether to
continue, accept, rollback the last step, or reject.

The strict-mode robustness knobs live under `sample_subgraph_family_llm.params`:
`allow_bootstrap_fallback`, `max_protocol_retries_per_stage`,
`max_bootstrap_errors`, `max_selector_errors`, `max_judge_errors`,
`min_multi_hop_outside_core_edges`, and `strict_abstain_on_empty_bootstrap`.

### 3) Quality controls already enabled
- Prompt-level constraints for DRAM/VQA reasoning (structure, timing, performance, comparison, grounding).
- Post-generation filtering in `VQAGenerator`:
  - drop empty QA pairs
  - drop uncertain answers (e.g., unknown)
  - deduplicate near-identical QA pairs
  - enforce context keyword grounding
- Evidence-aware context injection:
  - entities and relations can carry `evidence_span`
  - VQA prompts now include those evidence snippets explicitly
  - `build_grounded_tree_kg` can reject unsupported entity/relation evidence

### 4) Recommended config tuning
In `vqa_config.yaml` under `generate.params`, tune the general generation settings such as `data_format`.
For `agentic_subgraph_reasoning_config.yaml`, the main knobs are `sample_subgraph.params.max_units`,
`candidate_pool_size`, and `max_hops_from_seed`.
For `agentic_subgraph_reasoning_v2_config.yaml`, the main knobs are
`sample_subgraph_v2.params.hard_cap_units`, `max_rounds`, and `judge_pass_threshold`.
For `agentic_subgraph_reasoning_v3_config.yaml`, the main knobs are
`sample_subgraph_v3.params.hard_cap_units`, `max_rounds`, `judge_pass_threshold`, and `max_multi_hop_hops`.
