# Agentic Subgraph Sampler

## 导读

本文专门解释当前新增的 `agentic subgraph sampler`，也就是：

```text
partition -> sample_subgraph -> generate(method=auto)
```

它的目标不是简单做“远端信息融合”，而是在预算受限、仍然保持 vision-centered 的前提下，从融合后的全局 KG 里主动挑选一个**更值得生成 QA 的单个子图**。

这里最重要的前提是：

- `image/table` 周围第一圈 entity/relation 不是 agent 后续随便搜索出来的邻居
- 它们本来就来自 `MMKGBuilder` 对该 image/table chunk 的直接抽取
- 所以 agent 真正操作的是：
  - 固定 `vision anchor + modality-local core`
  - 再从全局融合图中挑选 `bridge/support/comparison/conclusion` 扩展

当前这条链路主要服务：

- `aggregated`
- `multi_hop`

而不是 `atomic`。

## 1. 它在整个 pipeline 里的位置

对应配置见：

- `examples/generate/generate_vqa/agentic_subgraph_reasoning_config.yaml`

完整链路是：

```text
read
-> structure_analyze
-> hierarchy_generate
-> tree_construct
-> tree_chunk
-> build_grounded_tree_kg
-> partition(anchor_bfs, image/table)
-> sample_subgraph
-> generate(method=auto)
```

这里各阶段的职责分工是：

- `partition`
  - 只负责给出一个 vision-centered 初始 seed/community
- `sample_subgraph`
  - 在融合图上围绕这个 seed 做受控扩展和价值选择
- `generate(method=auto)`
  - 根据 sampler 输出的 `task_type`，自动走 `AggregatedVQAGenerator` 或 `MultiHopVQAGenerator`

换句话说，`partition` 不再承担“最终上下文选择器”的职责，真正决定最终子图的是 `sample_subgraph`。

## 2. 当前实现的核心目标

当前实现对应代码：

- `graphgen/models/subgraph_sampler/value_aware_sampler.py`
- `graphgen/operators/sample_subgraph/sample_subgraph_service.py`

它优化的是一个平衡目标：

1. 先满足安全阈值
2. 再在安全阈值内最大化训练价值

这里的“安全阈值”主要指：

- 子图仍然以 image/table seed 为中心
- 子图内部仍然闭合、可答
- 关键证据不太稀薄
- 不要把融合图里别的文档内容错误拼进来

而“训练价值”主要指：

- 能否支持更高价值的问题形态
- 是否更适合 `aggregated` 或 `multi_hop`
- 是否真的依赖视觉 seed，而不是退化成普通文本 QA

## 3. 为什么它需要存在

当前 GraphGen 的固定 partition 有两个典型问题：

### 3.1 只靠局部扩张，容易过窄

`anchor_bfs` 很适合保证 image/table 在中心，但它拿到的往往只是 vision node 周边的一小团局部事实。

这会导致：

- 上下文很稳
- 但训练价值可能偏低
- 对 `aggregated` 和真正的 `multi_hop` 支撑不够

### 3.2 一旦盲目做大社区，又容易错配

如果继续扩大 BFS 半径，或者直接做更大的社区，很容易把：

- 不同 section 的主题
- 融合图里同名但不同文档的事实
- 与当前图表无关的高频实体

一起拉进来。

所以 sampler 的任务不是“拿更多信息”，而是：

**在受限预算内，挑一个更高价值但仍然可靠的子图。**

## 4. 它为什么叫 agentic

当前第一版不是一个自由漫游的大 agent，而是一个**受控的、策略驱动的子图搜索器**。

它的 agentic 性主要体现在：

- 不是固定一次性给定最终社区
- 而是从 seed 出发，迭代地产生候选扩展
- 每次扩展后重新评估是否值得继续
- 当增益不再明显时主动停止

所以它更像：

```text
budgeted search + value-based stopping
```

而不是：

```text
static partition
```

当前实现还没有引入独立 LLM 工具调用器来做多轮规划；第一版的“agentic”主要由启发式搜索策略体现。

## 5. 三层结构

当前更准确的理解应当是：

```text
Layer 0: vision anchor
Layer 1: modality-local extracted core
Layer 2: merged global extensions
```

### 5.1 Layer 0: vision anchor

就是 image/table 节点本身。

### 5.2 Layer 1: modality-local extracted core

这是由 `MMKGBuilder` 对该 image/table chunk 直接抽出的 mini-KG：

- seed 本体
- 从该 chunk 抽出的 entity
- 以及这些 entity 与 seed 的局部关系

这层不是 agent 搜索出来的，而是 agent 的固定起点。

### 5.3 Layer 2: merged global extensions

这是 merge 进全局图后，和 local core 相连的更远事实：

- bridge node
- support fact
- comparison branch
- conclusion node

agent 真正要做的是在这一层里做受控选择。

## 6. 当前实现如何工作

### 6.1 输入

`sample_subgraph` 接收的是 `partition` 输出的 batch：

- `nodes`
- `edges`

其中应该已经包含：

- image/table seed
- vision 周边的一些局部邻居

### 6.2 选 seed

`ValueAwareSubgraphSampler._select_seed_node()` 会优先找：

- `entity_type == IMAGE`
- `entity_type == TABLE`

如果没有，再退化到 metadata 里带：

- `img_path`
- `table_caption`

的节点。

### 6.3 恢复 seed 的文档归属

这里有一个非常重要的实现细节：

当前 KG 是**融合图**，不是按文档隔离的图。

所以 sampler 不能假设“邻近节点一定属于同一文档”，而是需要显式恢复 seed 的归属范围。

当前做法是：

1. 优先取 seed node 自身的：
   - `source_id`
   - `metadata.source_trace_id`
2. 只有 seed 本身拿不到归属信息时，才退化到 seed 相邻边的 `source_id`

这一步的作用是建立 `seed_source_ids`，后续所有候选节点都要尽量和它重合。

### 6.4 恢复 local core

当前实现不再把“同源近邻”直接当作 core，而是显式恢复：

- `seed_chunk_ids`
- `local_core_node_ids`
- `local_core_edge_pairs`

规则是：

- `seed_chunk_ids` 优先取 seed metadata 里的 `source_trace_id`
- 再补 seed node 的 `source_id`
- `local core` 只吸收那些 `node.source_id` 或 `edge.source_id` 明确包含 `seed_chunk_ids` 的节点/边

也就是说，local core 对应的是：

**由该 image/table chunk 自身抽出的原生 mini-KG**

### 6.5 候选扩展

agent 的搜索空间不再是“任意当前邻居”，而是“基于 local core 的 extension 选择”。

第一轮候选只能由 local core 触发，后续才允许从已接受的 extension 继续走。

当前 extension 会被标成这些角色之一：

- `bridge`
- `support`
- `comparison`
- `conclusion`

### 6.6 动作空间

从概念上看，当前 agent 的动作空间可以理解成：

- 保留 local core
- 添加 bridge node
- 添加 support fact
- 添加 comparison branch
- 添加 conclusion node
- reject candidate
- stop and commit

虽然实现还不是自由 LLM agent，但这个动作空间已经比固定 partition 更接近真正的 agent search。
当前关键控制参数是：

- `max_units`
- `max_steps`
- `max_hops_from_seed`
- `min_score_improvement`

## 7. 打分逻辑

当前实现采用“先过硬约束，再做软排序”。

### 7.1 硬约束

在 `_evaluate_candidate()` 中，候选子图必须同时满足：

- `budget_valid`
  - 节点数 + 边数不能超过预算
- `vision_centered`
  - 不只是 seed 在图里
  - 还必须保留足够的 local core
- `answerable`
  - 要能形成 local-core 驱动的聚合或桥接结构
- `evidence_sufficient`
  - local core 和 extension 都要有足够 evidence
- `coherent`
  - local core 必须由 `seed_chunk_ids` 支撑
  - extension 必须通过 local core 接入，而不是替代 local core 成为新中心

只要有一项不通过，这个候选子图就直接被淘汰。

### 7.2 软目标

通过硬约束后，再计算：

- `training_value`
- `visual_dependence`
- `reasoning_richness`
- `mismatch_penalty`

最终分数是：

```text
training_value
+ visual_dependence
+ reasoning_richness
- mismatch_penalty
```

#### `training_value`

当前仍是启发式实现，但已经不只看关键词，也会按 extension 角色加分：

- `bridge`
- `support`
- `comparison`
- `conclusion`

#### `visual_dependence`

这部分主要看：

- 子图中有多少边仍直接连着 seed
- seed 是否真的是 image/table 节点
- seed metadata 是否带 `img_path` / `table_caption`

作用是避免 sampler 退化成“普通文本 QA 子图选择器”。

#### `reasoning_richness`

这部分现在更强调结构角色：

- 是否存在 `bridge -> conclusion` 链
- 是否存在 comparison 分支
- 是否形成围绕 local core 的多事实簇

#### `mismatch_penalty`

惩罚项除了旧的缺证据/泛化关系外，还会惩罚：

- extension 节点数量反过来压过 local core
- extension 在结构上比 local core 更中心

## 8. 停止规则

这条链路最关键的不是“能扩多远”，而是“什么时候应该停”。

当前 stopping rule 是：

- 每轮都选当前得分最高的候选扩展
- 如果它对当前最优结果的增益不超过 `min_score_improvement`
- 就停止搜索

也就是说，它不追求：

- 最大社区
- 最远信息
- 最多节点

它追求的是：

**只在价值增益明显时才继续扩展。**

## 9. `task_type` 如何判定

当前实现会自动决定最终子图更适合：

- `aggregated`
- `multi_hop`

逻辑在 `_select_task_type()` 中。

### 判成 `multi_hop`

需要出现：

- `local core -> bridge -> conclusion/support`

也就是 bridge 不能只是存在，而要对最终结构不可省略。

### 判成 `aggregated`

更像围绕 local core 的多事实主题簇：

- comparison 分支
- 多个 support/conclusion 分支
- 价值来自聚合，而不是单链推理

这意味着当前第一版偏向保守：

- 只有在结构上已经比较像多跳链时才会给 `multi_hop`
- 否则更愿意把它当成主题聚合子图

## 10. 输出数据结构

`sample_subgraph` 的输出至少包含：

- `seed_node_id`
- `nodes`
- `edges`
- `task_type`
- `subgraph_score`
- `selection_rationale`
- `value_breakdown`
- `candidate_subgraph`
- `task_type_reason`
- `local_core_subgraph`
- `extension_subgraph`
- `node_roles`

其中：

- `selection_rationale`
  - 记录了任务类型、搜索步数、得分、各个硬约束是否通过
- `value_breakdown`
  - 记录了关键打分项

这些字段会继续被 `generate(method=auto)` 带到最终结果里，方便审计。

## 11. `generate(method=auto)` 做了什么

当前 `GenerateService` 新增了 `method: auto`。

行为是：

1. 看输入 batch 里的 `task_type`
2. 若是 `aggregated`，调用 `AggregatedVQAGenerator`
3. 若是 `multi_hop`，调用 `MultiHopVQAGenerator`
4. 若缺省，则默认回退到 `aggregated`

同时会把这些 sampler 元信息一起写进输出：

- `task_type`
- `seed_node_id`
- `selection_rationale`
- `value_breakdown`
- `subgraph_score`
- `task_type_reason`
- `local_core_subgraph`
- `extension_subgraph`
- `node_roles`
- `seed_chunk_ids`

所以这条链路不仅改变了子图选择，还把“为什么选这个子图”也留下来了。

## 12. YAML 入口

当前配套 YAML：

- `examples/generate/generate_vqa/agentic_subgraph_reasoning_config.yaml`

关键参数：

```yaml
sample_subgraph:
  params:
    max_units: 8
    max_steps: 4
    max_hops_from_seed: 4
```

以及：

```yaml
generate:
  params:
    method: auto
```

如果你要调这条链路，优先关注：

- `partition.method_params.max_units_per_community`
- `sample_subgraph.params.max_units`
- `sample_subgraph.params.max_steps`
- `sample_subgraph.params.max_hops_from_seed`

## 13. 当前实现的边界与限制

这部分很重要。

### 12.1 当前不是自由 LLM agent

虽然名字叫 agentic sampler，但当前第一版实现仍然是**启发式受控搜索**，不是：

- 自由工具调用 LLM agent
- 多轮 planner/executor/judge 体系
- 可回退的显式动作轨迹系统

它的 agentic 性主要体现在：

- 逐步扩展
- 逐步评估
- 动态停止

### 13.2 当前 graph 约束仍较弱

目前主要依赖：

- `source_id`
- seed 对齐
- hop 限制
- evidence 数量
- relation type 启发式惩罚

它还没有：

- 真正的路径语义校验
- 更强的 cross-document disambiguation
- learned reranker / reward model
- 更细粒度的 table/image 结构化选择

### 13.3 当前 training value 仍是启发式

`training_value` 还不是 learned score，也没有领域 schema。

它目前更多是在做：

- 关键词驱动的高价值信号近似

所以这是一个可用的第一版，但还不是最终形态。

## 14. 当前代码和目标模型的偏差

即使这次已经引入了 `local core / extension / node_roles`，当前实现仍然还有一些偏启发式的地方：

- extension role 仍是规则推断，不是 learned planner
- coherent 仍主要依赖 `source_id` 和局部连接，不是真正的路径语义校验
- `training_value` 仍是规则分，不是 learned score

所以这条实现已经更接近正确心智模型，但还不是最终形态。

## 15. 这条路线适合解决什么问题

当前这条路线最适合：

- 不满足于固定 partition 的一刀切局部社区
- 想在 budget 内让子图更“值钱”
- 又不想直接把整个系统升级成完全自由的 agent

它尤其适合：

- `aggregated`
- `multi_hop`

因为这两类任务都比 `atomic` 更依赖：

- 子图选择质量
- 结构闭合程度
- 训练价值与复杂度之间的平衡

## 16. 后续可演进方向

这条链路往后自然可以继续增强：

1. 把当前启发式候选扩展改成真正的 tool-using planner
2. 引入路径级语义约束，而不只看 hop 和 source overlap
3. 把 `training_value` 变成 learned scorer 或 judge model
4. 引入 table 细粒度结构，以支持更强的参数/比较采样
5. 引入显式 negative mining，让 mismatch penalty 更稳定

但在当前仓库阶段，这个第一版已经完成了很关键的一步：

**把“固定 partition”升级成了“价值驱动的受控子图选择”。**
