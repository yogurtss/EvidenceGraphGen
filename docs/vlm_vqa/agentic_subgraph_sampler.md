# Agentic VLM Subgraph Sampler

这篇文档只覆盖当前保留的 `sample_subgraph` 路线：

```text
build_grounded_tree_kg -> sample_subgraph -> generate(method=auto)
```

它的模型实现位于：

- `graphgen/models/subgraph_sampler/agentic_vlm_sampler.py`
- `graphgen/models/subgraph_sampler/artifacts.py`
- `graphgen/models/subgraph_sampler/prompts.py`
- `graphgen/models/subgraph_sampler/constants.py`
- `graphgen/models/subgraph_sampler/debug_artifacts.py`

operator 入口位于：

- `graphgen/operators/sample_subgraph/sample_subgraph_service.py`

示例配置位于：

- `examples/generate/generate_vqa/agentic_subgraph_reasoning_config.yaml`

## 目标

`sample_subgraph` 不直接生成 QA。它先以图中的 image seed 为中心，收集同源的多模态 evidence，再让 VLM 规划、组装和评审一个适合后续生成的目标子图。

输出会继续交给统一的：

```text
generate(method=auto)
```

`GenerateService` 会根据 sampler 输出里的 intent、question type 和 selected subgraph 信息路由到下游 generator。

## 流程

`VLMSubgraphSampler.sample()` 的主流程是：

1. 校验 `seed_node_id` 和 seed image asset。
2. 从 image seed 收集 `source_id` / chunk scope。
3. 构造 seed-local neighborhood。
4. 调用 VLM planner 生成意图和候选方向。
5. 调用 assembler 选择具体节点/边，形成 candidate subgraph。
6. 调用 judge 评估视觉必要性、grounding、question value 和图结构质量。
7. 选择通过 judge 的 candidate，输出 `selected_subgraphs`。
8. 如果主路径失败且允许 degraded mode，则走更保守的 fallback。

核心输出字段包括：

- `seed_node_id`
- `seed_image_path`
- `selection_mode`
- `selected_subgraphs`
- `candidate_bundle`
- `intent_bundle`
- `abstained`
- `sampler_version = v1`
- `termination_reason`
- `max_vqas_per_selected_subgraph`

## Debug Trace

`sample_subgraph.params.debug = true` 时，sampler 会额外输出 `debug_trace`，记录 seed 校验、scope 收集、neighborhood、candidate、judge 和最终选择等阶段。

这条 trace 面向工程调试，不是下游 QA 生成的必需字段。

## Visual-Core Family LLM

如果要按 `atomic` / `aggregated` / `multi_hop` family 分别采样，请使用当前保留的 visual-core family LLM 路线：

```text
build_grounded_tree_kg -> sample_subgraph_family_llm -> generate(method=auto)
```

主实现位于：

- `graphgen/models/subgraph_sampler/visual_core_family_llm/sampler.py`
- `graphgen/models/subgraph_sampler/visual_core_family_llm_sampler.py`

配置示例位于：

- `examples/generate/generate_vqa/agentic_family_llm_vqa_config.yaml`
