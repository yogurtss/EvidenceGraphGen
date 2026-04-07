# Agentic Subgraph V3

## 导读

`sample_subgraph_v3` 是当前 agentic subgraph 系列里最偏向“按题型分别选图”的版本。

它不再先挑一个通用子图，再让下游 generator 自己适配，而是围绕同一个 `image seed` 分别尝试：

- `atomic`
- `aggregated`
- `multi_hop`

每个 family 都有自己独立的 graph-editing session、独立的子图预算、独立的停止条件。

完整链路是：

```text
build_grounded_tree_kg
-> sample_subgraph_v3
-> generate(method=auto)
```

对应配置：

- `examples/generate/generate_vqa/agentic_subgraph_reasoning_v3_config.yaml`

## 1. 设计目标

`v3` 主要解决两个问题：

1. `v1/v2` 虽然能选到好子图，但更像“先选图、后猜题型”，三类 QA 的覆盖不稳定。
2. `generate(method=auto)` 在旧路径里主要依赖 `approved_question_types` 模糊映射，难以稳定保证 `atomic / aggregated / multi_hop` 都有机会出现。

所以 `v3` 的核心思路是：

- family-aware sampling
- family-aware routing
- family-aware trace

也就是：

- 选图阶段先按 family 拆开
- 生成阶段优先按 `qa_family` 严格路由
- 调试阶段也按 family 单独看 session

## 2. 三类 family 怎么分工

### 2.1 `atomic`

目标是单一核心事实、直接图像读数、或一个短的一跳关系。

典型特征：

- 从 seed 出发尽量少扩图
- 子图预算最小
- 倾向只保留必要节点和边
- 如果没有清晰直接证据，就直接跳过

这类子图通常适合：

- 参数读取
- 图中标注解释
- 单关系理解

### 2.2 `aggregated`

目标是围绕一个 intent，收集同主题 neighbors，支持一段可整合、可重述的技术解释。

典型特征：

- 比 `atomic` 更宽，但不要求长链推理
- 强调 theme coherence
- 更在意 explanation-quality，而不是 hop 深度

这类子图通常适合：

- 局部结构解释
- 约束整合
- 参数与行为的局部综合说明

### 2.3 `multi_hop`

目标是显式追求可验证的 reasoning chain，而不是只靠 prompt 假装多跳。

典型特征：

- 默认允许比 `v2` 更深的 hop 扩张
- 需要至少一条真实多步链
- 如果 judge 认为链不闭合，会继续扩图

这类子图通常适合：

- “图中参数 -> 中间机制 -> 下游影响” 这种链式问题
- 需要组合多条边才能稳定回答的问题

## 3. 输出结构

`v3` 顶层仍然兼容 `selected_subgraphs`，但每个子图会新增：

- `qa_family`

取值固定为：

- `atomic`
- `aggregated`
- `multi_hop`

同时 trace 也带 family：

- `agent_session[*].qa_family`
- `candidate_states[*].qa_family`
- `edit_trace[*].qa_family`
- `judge_trace[*].qa_family`
- `neighborhood_trace[*].qa_family`

所以 `v3` 的一个 seed 最多会产出 3 个 family-specific subgraph，而不是 1 个通用 subgraph。

## 4. 生成阶段如何衔接

`GenerateService(method=auto)` 在 `v3` 下会优先读取：

- `selected_subgraphs[*].qa_family`

然后严格路由到：

- `atomic -> atomic_vqa`
- `aggregated -> aggregated_vqa`
- `multi_hop -> multi_hop_vqa`

这里和 `v1/v2` 最大的区别是：

- `v1/v2` 仍然兼容旧的 question-type 模糊映射
- `v3` 不再优先猜题型，而是直接按 canonical family 路由

## 5. 关键配置项

`agentic_subgraph_reasoning_v3_config.yaml` 当前最重要的参数是：

- `hard_cap_units`
  - family session 的最大图单元预算上限
- `max_rounds`
  - 单个 family 最多允许多少轮编辑
- `max_vqas_per_selected_subgraph`
  - 每个 selected subgraph 最多保留多少条 QA
- `judge_pass_threshold`
  - family judge 的总分阈值
- `max_multi_hop_hops`
  - `multi_hop` family 允许扩到的最远 hop

## 6. 推荐理解方式

把 `v3` 理解成：

```text
one image seed
-> three family-specific editing sessions
-> zero to three selected subgraphs
-> strict family-based auto generation
```

这样它更像一个“面向训练数据配比的选图器”，而不只是一个更聪明的单子图搜索器。
