# GraphGen VLM / VQA 架构与代码现状

## 导读

本文基于当前仓库代码，重新梳理 GraphGen 在 `atomic`、`aggregated`、`multi_hop`、`vqa` 四类样本生成上的整体架构与技术细节，重点回答三个问题：

1. 从配置到输出，整条链路是如何组织的。
2. tree pipeline、KG、partition、generator 分别在 VQA 生成里承担什么职责。
3. 当前代码里哪些能力是真正的“VQA 专有能力”，哪些仍然是通用 QA 能力。

最重要的结论先说在前面：

- GraphGen 的统一外壳是“YAML 配置驱动的 DAG 执行器 + operator/service + generator/template + storage”。
- `atomic`、`aggregated`、`multi_hop`、`vqa` 共用同一个 `generate` operator 入口，但它们依赖的 generator 行为并不相同。
- 当前代码里，真正显式处理图片路径、多题生成、VQA 质量过滤的是 `VQAGenerator`。
- `atomic`、`aggregated`、`multi_hop` 仍然是通用 QA 生成器；它们是否呈现出“多模态 / VQA 风格”，主要由前面的 tree-aware chunk、KG 和 anchor partition 决定，而不是 generator 自身有图片字段或视觉样本格式支持。

## 1. 总体执行架构

统一入口是 `graphgen/run.py`。运行时读取 YAML 配置，把 `nodes` 定义成一张 DAG，交给 `Engine` 调度执行。

从代码结构看，这个框架可以拆成五层：

```text
YAML config
  -> Engine / DAG execution
  -> operators (read / chunk / build_kg / partition / generate ...)
  -> models (partitioner / generator / KG builder)
  -> storage (KV + graph storage)
```

其中和 VLM/VQA 最相关的是四段链路：

```text
输入解析 -> chunk / tree_chunk -> build_kg / build_tree_kg -> partition -> generate
```

当前仓库还新增了三条可选增强链路，用于在全局融合 KG 上直接扫描 `image seed`，做更高价值的子图选择：

```text
build_grounded_tree_kg -> sample_subgraph -> generate(method=auto)
build_grounded_tree_kg -> sample_subgraph_v2 -> generate(method=auto)
build_grounded_tree_kg -> schema_guided_subgraph -> generate(method=auto)
```

这里的 `sample_subgraph` / `sample_subgraph_v2` / `schema_guided_subgraph` 都不直接生成 QA，而是先固定：

- `image seed`
- `seed-local multimodal evidence`

再在融合后的全局 KG 中按 `source_id`、seed chunk 和预算约束挑选一个更适合 `aggregated` 或 `multi_hop` 的单个目标子图。

三者的差异主要在“如何选图”：

- `sample_subgraph`
  - LLM 做单轮 intent planning + candidate assembly
- `sample_subgraph_v2`
  - LLM 做 stateful graph editing
- `schema_guided_subgraph`
  - LLM 做 schema-aware intent planning，具体 retrieval broadening 由代码按固定阶段控制

`GenerateService` 是所有 QA/VQA 任务的统一出口。它根据 `method` 动态选择：

- `atomic -> AtomicGenerator`
- `aggregated -> AggregatedGenerator`
- `multi_hop -> MultiHopGenerator`
- `vqa -> VQAGenerator`

也就是说，`atomic`、`aggregated`、`multi_hop`、`vqa` 在调度层是平级模式，而不是一套 VQA 内部子类型。

## 2. 四类样本的真实分工

### 2.1 `atomic`

`AtomicGenerator` 接收一个局部子图，把节点和边渲染成 grounded context，然后要求模型只输出一组 `<question> + <answer>`。

它的特点是：

- 单问单答
- 没有图片字段注入
- 没有专门的质量过滤
- 输出格式走 `BaseGenerator.format_generation_results()`

所以 `atomic` 的“原子性”主要不来自 generator，而来自 partition 是否把社区压到了足够小。典型配置是：

```text
partition(method=dfs, max_units_per_community=1) -> generate(method=atomic)
```

### 2.2 `aggregated`

`AggregatedGenerator` 是两阶段生成：

```text
子图上下文 -> 重述答案(rephrased text) -> 基于答案反推问题
```

也就是说，`aggregated` 并不是直接让模型同时写题和答案，而是先把局部图重组成一段更连贯的综合答案，再围绕这段答案生成问题。

它天然适合：

- 多事实整合
- 因果/时序/结构化叙述
- 比单点 atomic 更宽的局部上下文

但它同样没有图片字段处理，也没有 VQA 专用后处理。

### 2.3 `multi_hop`

`MultiHopGenerator` 与 `AtomicGenerator` 的实现形态很接近，也是单次 prompt、单组 QA 输出。它和 atomic 的主要差异几乎全部来自 prompt：

- 明确要求“问题必须通过多步推理才能回答”
- 依赖 partition 给到至少包含多跳路径的局部子图

因此，`multi_hop` 当前更像“prompt 驱动的多跳 QA”，还不是“路径约束驱动的多跳 QA”。

### 2.4 `vqa`

`VQAGenerator` 是当前唯一明确面向多模态 VQA 的生成器，和前三者相比多了三层专有逻辑：

1. prompt 明确要求一次输出 6 到 10 组 QA，并强调事实性、可验证性、跨模态 grounding、数值读取、关系推理和 DRAM 场景。
2. 生成后执行质量过滤，去掉空结果、占位词、未知答案、重复问答、完全不命中文本上下文关键词的样本。
3. 从节点 `metadata` 中提取 `img_path`，再把图片字段写入 Alpaca / ShareGPT / ChatML 输出结构。

所以从代码现状看，真正意义上的“VQA 输出”目前只由 `method: vqa` 显式支持。

## 3. 从输入到局部图上下文

### 3.1 普通管线

普通 VQA / QA 配置通常是：

```text
read -> chunk -> build_kg -> partition -> generate
```

这里的 `chunk` 是通用文本切分。输入即使带图片，也更偏向“文本 chunk + 多模态节点混合建图”的模式。

### 3.2 tree 管线

tree VQA 和 tree atomic 走的是：

```text
read
-> structure_analyze
-> hierarchy_generate
-> tree_construct
-> tree_chunk
-> build_grounded_tree_kg 或 build_tree_kg
-> partition
-> generate
```

这是当前 `docs/vlm_vqa` 最值得关注的链路，因为它决定了多模态样本是不是“结构保真”的。

#### `structure_analyze`

`StructureAnalyzeService` 会把 markdown 文档拆成 component pack。当前 `tree_utils.py` 已经支持识别：

- `section`
- `text`
- `image`
- `table`

其中：

- image 会提取 `img_path`、caption、notes
- table 会提取 caption 和 table body

这一步把原始 markdown 从“平文本”提升成“可感知组件类型的输入”。

#### `hierarchy_generate`

这一层不会改变内容，只会为各个 component 推导 `title_level`，为后续树构建准备层级信息。

#### `tree_construct`

`TreeConstructService` 把 component 组织成显式树结构，并给每个节点补充：

- `path`
- `level`
- `parent_id`
- `node_id`

因此后续所有 chunk 都不再只是“某段文本”，而是“文档树中某个位置上的内容片段”。

#### `tree_chunk`

`TreeChunkService` 会把树节点变成下游 KG builder 可消费的 chunk。关键点有两个：

- `split_text_nodes: false` 时，预分段 paragraph/image/table 会尽量保持原状，不再被二次打散。
- 输出 metadata 会保留 `path`、`level`、`node_id`、`parent_id`、`source_trace_id` 等字段。

这一步决定了 tree VQA 能不能保住局部结构边界。

## 4. KG 构建与 evidence grounding

### 4.1 普通 KG

`BuildKGService` 会把 chunk 分成：

- `text` -> `build_text_kg`
- `image/table/video/formula` -> `build_mm_kg`

然后把抽取出的节点、边写入图存储。

### 4.2 tree KG

`BuildTreeKGService` 的关键增强不是换了另一套 builder，而是先把 tree path 注入文本内容：

```text
[Document Path]
root/...

[Chunk]
...
```

也就是说，文本抽取时模型看到的不只是 chunk 正文，还能看到 chunk 在文档树中的位置。

### 4.3 grounded tree KG

`BuildGroundedTreeKGService` 在 `BuildTreeKGService` 之上，默认开启更严格的 evidence 约束：

- `require_entity_evidence=True`
- `require_relation_evidence=True`
- `validate_evidence_in_source=True`
- text relation evidence 更严格
- multimodal relation evidence 也要求校验

这使得 `evidence_span` 不只是调试字段，而是进入了实际保留/丢弃逻辑。

### 4.4 `evidence_span` 为什么关键

当前所有 generator 都通过 `context_utils.build_grounded_context()` 渲染子图上下文。节点和边如果有 `evidence_span`，会被写成：

```text
Evidence: ...
```

所以 evidence 的作用贯穿三层：

1. KG 构建阶段决定节点/边是否保留。
2. partition 后进入 community。
3. generate 时直接进入 prompt，变成模型必须遵守的 grounding 证据。

对于 tree VQA，这是一条真正的端到端证据链。

## 5. partition 如何决定样本形态

generator 只能利用拿到的局部图；真正决定样本像不像 atomic / aggregated / multihop / VQA 的，首先是 partition。

### 5.1 `dfs`

`DFSPartitioner` 适合做最小局部社区。典型 atomic 配置：

```yaml
method: dfs
method_params:
  max_units_per_community: 1
```

这会把社区压得很小，因此更接近“单事实单问题”。

### 5.2 `anchor_bfs`

`AnchorBFSPartitioner` 会先选定锚点类型，再围绕锚点 BFS 扩张。它特别适合多模态场景：

- 普通 VQA 常用 `anchor_type: image`
- tree VQA / tree atomic 常用 `anchor_type: [image, table]`

这意味着社区不是随机切出来的，而是围绕图像或表格向外吸附相邻实体和关系，因此更容易形成“图表 + 邻近文字证据”的局部上下文。

### 5.3 `ece`

`aggregated` 和默认 `multi_hop` 示例配置更偏向 `ece` 分区。它依赖 `quiz -> judge -> partition(ece)` 的前置链路，用 comprehension loss 或相关评分挑选更适合生成的问题上下文。

因此：

- `aggregated` 更像“挑出信息量更大、可重述性更强的社区”
- `multi_hop` 更像“挑出至少能支持若干关系串联的社区”

但这仍然不是严格的路径验证。

## 6. prompt 与输出格式的差异

### 6.1 共性

四类 generator 都先把 partition 得到的 community 渲染成：

- 实体列表
- 关系列表
- 可选 `Evidence:` 行

### 6.2 差异

- `atomic`：直接单问单答。
- `multi_hop`：单问单答，但 prompt 强调多步推理。
- `aggregated`：先重写答案，再反推问题。
- `vqa`：一次输出多组 QA，并强调多模态 grounding、训练价值和难度配比。

### 6.3 输出数据结构

`GenerateService` 在 generator 产出 QA 后，会统一附加：

- `sub_graph`
- `sub_graph_summary`
- `_trace_id`

普通 generator 默认走 `BaseGenerator.format_generation_results()`，支持：

- `Alpaca`
- `Sharegpt`
- `ChatML`

`VQAGenerator` 重写了这层格式化逻辑，使用户消息里可以带图片字段。现在 `AggregatedVQAGenerator` 和 `MultiHopVQAGenerator` 也复用了这类多模态输出能力，用于 image/table 驱动的聚合与多跳样本。

## 7. 当前代码里“multihop / aggregated / atomic VQA”到底意味着什么

从命名上看，容易把 `atomic`、`aggregated`、`multi_hop` 理解成 VQA 的三个子类。当前实现里，这件事已经部分成立，但不是完全对称。

更准确的说法是：

- `atomic`、`aggregated`、`multi_hop` 是三种 QA 生成方式。
- `vqa` 是单独的一种多模态 QA 生成方式。
- 同时，`aggregated` 和 `multi_hop` 现在各自有了显式的 VQA 版本：`AggregatedVQAGenerator`、`MultiHopVQAGenerator`。
- tree pipeline 和 anchor partition 可以让前三者拿到更像 VQA 的多模态局部上下文。
- 但只要 generator 仍然是 `AtomicGenerator` / `AggregatedGenerator` / `MultiHopGenerator`，输出里就不会自动拥有 `img_path` 注入和多模态输出格式；这部分需要对应的 VQA 版本来承担。

所以当前仓库里严格说：

- 有 `atomic QA`
- 有 `aggregated QA`
- 有 `multi_hop QA`
- 有 `VQA`
- 有显式独立实现的 `aggregated VQA` / `multi_hop VQA`
- 仍然还没有显式独立实现的 `atomic VQA` generator

现有的 tree atomic 更接近：

```text
tree-aware multimodal local graph + atomic QA generator
```

而不是：

```text
atomic VQA generator
```

这也是后续文档、实验和命名时需要特别注意的地方。

## 8. 推荐理解方式

如果从工程角度理解当前 GraphGen，可以把整套系统视为两层组合：

### 8.1 上层：上下文构建层

负责回答“给生成器什么材料”：

- 普通 chunk 还是 tree-aware chunk
- 普通 KG 还是 grounded tree KG
- 随机社区、最小社区，还是 image/table anchor 社区

### 8.2 下层：样本成形层

负责回答“把材料变成什么样的监督样本”：

- `atomic`
- `aggregated`
- `multi_hop`
- `vqa`

对 VLM/VQA 来说，上层实际上决定 groundedness 和 multimodalness，下层决定问题形态与训练格式。

## 9. 现阶段最重要的工程判断

基于当前代码，可以得出几个比较稳的判断：

1. tree VQA 是现有最完整、最 grounded 的多模态生成链路，因为它同时具备结构保真、evidence 约束、image/table anchor partition、VQA prompt 和图片字段输出。
2. tree atomic 是“树结构保真 + 多模态局部上下文 + atomic QA”的混合方案，适合做单轮监督数据，但不能直接等同于严格 VQA。
3. `aggregated` 和 `multi_hop` 当前的深度主要仍靠 prompt 与 partition 形状驱动，还没有显式的路径验证、证据组合打分或跨模态依赖判别。
4. 如果后续要真正支持 `atomic VQA / aggregated VQA / multihop VQA`，最自然的演进方向不是只改 prompt，而是补三类能力：
   - 图片字段和多模态输出格式注入
   - 基于 evidence/path 的结构化过滤
   - 对“是否真的需要图像/表格、是否真的需要多跳”的后验校验

## 10. 相关配置入口

当前最关键的几个配置文件是：

- `examples/generate/generate_vqa/vqa_config.yaml`
- `examples/generate/generate_vqa/tree_vqa_config.yaml`
- `examples/generate/generate_atomic_qa/atomic_config.yaml`
- `examples/generate/generate_atomic_qa/tree_atomic_config.yaml`
- `examples/generate/generate_aggregated_qa/aggregated_config.yaml`
- `examples/generate/generate_multi_hop_qa/multi_hop_config.yaml`

建议配合本文一起阅读 `docs/vlm_vqa/research.md` 和 `docs/vlm_vqa/roadmap.md`：

- `architecture.md` 侧重“当前代码怎么工作”
- `research.md` 侧重“VQA/tree atomic 细节拆解”
- `roadmap.md` 侧重“后续该往哪里演进”
