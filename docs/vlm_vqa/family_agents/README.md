# Family-Decoupled Subgraph Agents

这条新链路并行于现有 `sample_subgraph_v3 -> generate(method=auto)`：

```text
build_grounded_tree_kg
-> sample_subgraph_family
-> generate_agentic_vqa
```

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
