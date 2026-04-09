# Family-Decoupled Subgraph Agents

现在有两条 family 路径：

```text
build_grounded_tree_kg
-> sample_subgraph_family
-> generate_agentic_vqa
```

```text
build_grounded_tree_kg
-> sample_subgraph_family_llm
-> generate(method=auto)
```

其中新的 `sample_subgraph_family_llm` 采用 visual-core bootstrap：

```text
seed image
-> keep/drop first-hop visual core
-> second-hop candidate pool
-> llm node selector
-> termination judge
```

核心约束：

- `seed + 保留下来的 first-hop` 视为 `visual core`
- `atomic` 默认只停留在 visual core
- `aggregated` 和 `multi_hop` 从 second-hop 开始正式扩张
- “同向”从第一条离开 visual core 的边开始冻结
- `generate(method=auto)` 优先读取每个 selected subgraph 的 `target_qa_count`
- 协议错误不会再静默 fallback；默认直接进入明确的 `abstain / protocol_error` 终止路径
- `multi_hop` 需要通过代码侧链式 postcheck，visual core 外至少两条连续边
- `family_termination_trace` 会为每个 terminal path 记录 `decision_source / protocol_status / termination_reason`

## 1. 三个独立 agent

- `atomic`
  - 单事实、最小闭包
  - 只从 seed 的直接连接候选中选一个 supporting node
- `aggregated`
  - 同主题宽度优先
  - 先选主题锚点，再并入 same-theme sibling / near-sibling
- `multi_hop`
  - 深度优先
  - 维护 active frontier，深推时剪掉同层 sibling

## 2. 代码位置

- `graphgen/models/subgraph_sampler/family_agents/`
- `graphgen/operators/sample_subgraph_family/`
- `graphgen/operators/generate_agentic_vqa/`

## 3. Candidate Pool

统一 schema：

- `candidate_node_id`
- `bind_from_node_id`
- `bound_edge_pair`
- `hop`
- `theme_signature`
- `frontier_path`

公共约束：

- 候选节点必须已和当前待扩展节点相连
- `add_node` 时自动加入绑定 edge
- 节点并入后会从当前 pool 中移除重复候选
- `multi_hop` 继续向深度推进时，只保留新 frontier 的下一层候选

## 4. 输出与联动生成

`sample_subgraph_family` 除了兼容 `selected_subgraphs`，还会输出：

- `family_sessions`
- `family_candidates`
- `family_edit_trace`
- `family_judge_trace`

每个 `selected_subgraphs[*]` 额外带：

- `qa_family`
- `candidate_pool_snapshot`
- `frontier_node_id`
- `theme_signature`
- `revision_id`

`generate_agentic_vqa` 会执行：

```text
family subgraph
-> family generator
-> family QA judge
-> revise_qa_only / refine_subgraph_then_regenerate / accept / reject
```

其中 `refine_subgraph_then_regenerate` 会回调 family agent continuation，而不是直接改图。

## 5. 配置

- `examples/generate/generate_vqa/agentic_family_vqa_config.yaml`
- `examples/generate/generate_vqa/agentic_family_llm_vqa_config.yaml`

`sample_subgraph_family_llm` 额外的鲁棒性参数：

- `allow_bootstrap_fallback`
- `max_protocol_retries_per_stage`
- `max_bootstrap_errors`
- `max_selector_errors`
- `max_judge_errors`
- `min_multi_hop_outside_core_edges`
- `strict_abstain_on_empty_bootstrap`
