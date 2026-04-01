# Docs 导航

当前 `docs/` 按主题拆成了两个主目录，避免所有文档平铺在一层。

## 1. `docs/vlm_vqa/`

这组文档聚焦 GraphGen 面向专业文档的 VLM / VQA 数据生成路线。

推荐阅读顺序：

1. `docs/vlm_vqa/architecture.md`
2. `docs/vlm_vqa/agentic_subgraph_sampler.md`
3. `docs/vlm_vqa/schema_guided_subgraph.md`
4. `docs/vlm_vqa/research.md`
5. `docs/vlm_vqa/roadmap.md`
6. `docs/vlm_vqa/plans/kg_grounding.md`
7. `docs/vlm_vqa/plans/multimodal_alignment.md`
8. `docs/vlm_vqa/plans/question_depth.md`
9. `docs/vlm_vqa/plans/eval_benchmark.md`
10. `docs/vlm_vqa/execution/p0_checklist.md`

目录含义：

- `architecture.md`
  - 基于当前代码的整体架构、生成链路与任务分工说明
- `agentic_subgraph_sampler.md`
  - value-aware agentic subgraph sampler 的目标、实现细节与配置入口
- `schema_guided_subgraph.md`
  - schema-guided subgraph sampler 的设计定位、执行逻辑与配置入口
- `research.md`
  - 当前 GraphGen VQA / Atomic QA 原理研究
- `roadmap.md`
  - 顶层路线图
- `plans/`
  - 各专项规划文档
- `execution/`
  - 更贴近工程执行的阶段清单

## 2. `docs/tree_pipeline/`

这组文档聚焦 tree pipeline 的局部专题说明。

- `docs/tree_pipeline/structure_analyze_vqa_changes.md`
  - markdown 结构分析、image/table 组件拆分与 tree VQA 相关变更说明

## 3. 后续建议

如果后面继续补文档，建议沿用这个结构：

- VLM / VQA 总体路线与计划，继续放 `docs/vlm_vqa/`
- tree pipeline、chunk、parser、fixture 之类的专题说明，放 `docs/tree_pipeline/`
- 如果后续有 benchmark 结果、实验记录，可以再单独增加 `docs/experiments/`
