# Visual-Core Family LLM Subgraph Sampler

当前保留的 family VQA 路径是：

```text
build_grounded_tree_kg
-> sample_subgraph_family_llm
-> generate(method=auto)
```

主实现位于 `graphgen/models/subgraph_sampler/visual_core_family_llm/sampler.py`（兼容导出保留在 `graphgen/models/subgraph_sampler/visual_core_family_llm_sampler.py`）。它把子图构造拆成 bootstrap、selector、termination judge 和代码侧 postcheck，输出 `family_llm_v2` 子图，再交给 `generate(method=auto)` 按 `qa_family` 生成 QA。

## 当前流程

`sample_subgraph_family_llm` 的状态机如下：

```text
image seed
-> collect seed scope / runtime schema
-> collect first-hop image-local candidates
-> collect second-hop preview candidates
-> bootstrap keep/drop first-hop analysis anchors
-> create <seed>::virtual_image
-> build virtualized second-hop candidate pool
-> selector chooses one legal candidate per step
-> code updates selected nodes / synthetic edge metadata / frontier / direction
-> termination judge accepts, continues, rolls back, or rejects
-> materialize family_llm_v2 selected_subgraph
-> generate(method=auto) routes to the matching family generator
```

关键语义：

- first-hop image-local nodes 只作为 `analysis_first_hop_node_ids`，不进入最终 QA 证据子图。
- 最终 QA 子图以虚拟 image node 为根，例如 `<seed>::virtual_image`。
- 第一条 QA 证据边是 synthetic edge：`virtual_image -> second-hop evidence node`，原 KG 不会被修改。
- synthetic edge 会记录 `analysis_anchor_node_id`、`virtualized_from_path`、`virtualized_from_edge_pair`，用于回溯原始 KG 路径。
- `analysis_only_node_ids` 会阻止原始 image seed 和 first-hop 节点被后续 selector 选回最终 QA 子图。
- `generate(method=auto)` 会读取每个 `selected_subgraphs[*].qa_family` 和 `target_qa_count`，路由到对应 family generator。
- VQA、aggregated、multi-hop 生成器的 prompt 会注入 image caption / note metadata，提升图文 grounding。

## 创新点

- Analysis-only first-hop anchors：把 image 直接产出的第一层节点视为 image 的分析层，而不是 QA 证据层，减少 QA 被 image 原生信息困住的问题。
- Virtual image root：最终子图使用虚拟 image node 直连第二层或更深证据节点，保留图像不可替代性，同时不污染原始 KG。
- One-node-per-step selector：LLM selector 每轮只能从合法 candidate pool 里选择一个节点，避免一次性保留多个节点导致 atomic QA 答案不唯一。
- Judge-gated state machine：selector 负责策略选择，termination judge 负责是否继续或回滚，代码负责协议校验、方向一致性、深度预算和 family postcheck。
- Family-specific constraints：`atomic` 要求虚拟 image + 一个证据节点；`aggregated` 允许同主题宽扩；`multi_hop` 维护一条 active chain，并要求 visual core 外有足够深度。
- Evidence-aware generation：`VQAGenerator`、`AggregatedGenerator`、`MultiHopGenerator` 会通过公共 context formatter 把 image caption / note 写入 prompt 上下文。

## Family 行为

`atomic`：

- bootstrap 只选择 first-hop analysis anchor。
- selector 从 second-hop candidate pool 中选择一个 QA 证据节点。
- postcheck 要求最终子图只有虚拟 image node 和一个 evidence node。

`aggregated`：

- bootstrap 可以保留多个 first-hop analysis anchors。
- selector 先选一个二层主题证据，再允许同方向、同主题的 breadth expansion。
- 深化后仍保留兼容 sibling，避免退化成纯 DFS。

`multi_hop`：

- bootstrap 只保留一个 first-hop analysis anchor。
- selector 维护 active frontier。
- 每次向深层推进时剪掉同层 sibling，只保留新 frontier 的下一层候选。
- postcheck 要求方向一致，并且 visual core 外至少有 `min_multi_hop_outside_core_edges` 条连续边。

## 输出与调试

顶层输出保留：

- `selected_subgraphs`
- `candidate_bundle`
- `family_sessions`
- `family_bootstrap_trace`
- `family_selection_trace`
- `family_termination_trace`
- `intent_bundle`
- `sampler_version = family_llm_v2`

每个 `selected_subgraphs[*]` 重点字段：

- `qa_family`
- `nodes`
- `edges`
- `visual_core_node_ids`
- `analysis_first_hop_node_ids`
- `analysis_only_node_ids`
- `selected_evidence_node_ids`
- `original_seed_node_id`
- `virtual_image_node_id`
- `direction_mode`
- `direction_anchor_edge`
- `frontier_node_id`
- `candidate_pool_snapshot`
- `target_qa_count`

鲁棒性字段：

- `protocol_status`
- `protocol_error_type`
- `protocol_failures`
- `bootstrap_error_count`
- `selector_error_count`
- `judge_error_count`
- `invalid_selection_count`
- `invalid_candidate_repeat_count`
- `blocked_candidate_uids`

## 配置

主要配置文件：

- `examples/generate/generate_vqa/agentic_family_llm_vqa_config.yaml`

`sample_subgraph_family_llm` 常用参数：

- `family_qa_targets`
- `family_max_depths`
- `max_steps_per_family`
- `max_rollbacks_per_family`
- `judge_pass_threshold`
- `same_source_only`
- `allow_bootstrap_fallback`
- `max_protocol_retries_per_stage`
- `max_bootstrap_errors`
- `max_selector_errors`
- `max_judge_errors`
- `min_multi_hop_outside_core_edges`

## 当前已知风险

- Bootstrap 阶段 judge 仍然会在只包含虚拟 image node、尚未选择二层证据节点时运行。如果 judge 直接返回 `reject`，该 family session 会提前终止。更稳的做法是 bootstrap 只做协议和候选池检查，真正的 accept / reject 从第一步 selection 后开始。
- Dropped first-hop 当前语义是“不作为初始 analysis anchor 使用，并且 first-hop 本身不会进入最终 QA 子图”。如果某个 dropped 分支的后代通过其他已选证据节点可达，代码没有全局 blocked-descendant set 来强制排除它。

## 后续改进方向

- 调整 bootstrap early-reject 行为：当存在合法 second-hop candidate 且当前只有虚拟 image root 时，将 bootstrap judge 的 `accept/reject` 视为 `continue`，或直接跳过 bootstrap judge。
- 增加针对 bootstrap judge 早退的测试，覆盖 atomic / aggregated / multi-hop 都必须先进入至少一次 selector。
- 明确 dropped first-hop 的强弱语义：如果需要强隔离，就在 bootstrap 后构建 dropped descendant blocklist；如果只是初始 anchor drop，就把文档和 trace 名称说清楚。
- 在 trace 中增加简洁的 anchor rationale，例如“为什么这个 second-hop 证据是通过哪个 first-hop analysis anchor 进入的”。
- 给 caption / note prompt 注入增加可配置开关，避免超长 caption 或 nearby note 在大图子图上带来 prompt 膨胀。
- 引入更细的 family target：例如 atomic 可区分 visual fact / numeric fact，aggregated 可区分 breadth topic / comparison，multi-hop 可区分 causal / constraint chain。

## 心智模型

这个 agent 的角色边界可以概括为：

- LLM bootstrap 是意图和分析锚点发现层。
- LLM selector 是逐步选择策略层。
- LLM termination judge 是局部质量判断层。
- Code validator 是硬约束层。
- Materializer 是不改 KG 的虚拟证据投影层。

因此 LLM 不再是直接决定最终子图的唯一裁判，而是被状态机、协议校验和 family-specific postcheck 包住的决策组件。
