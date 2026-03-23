# GraphGen Data Platform V1

## 概述

这次在仓库里新增了一套独立的数据平台，用来面向 GraphGen 生成结果做本地可视化浏览，重点支持：

- 导入 `cache` 这类 GraphGen 输出目录
- 浏览 Question / Answer
- 预览 VQA 图片
- 可视化 `sub_graph`
- 展示节点和边上的 `evidence_span`
- 为后续词频分析、run 对比、evidence 完整度分析预留统一数据层

这套平台没有复用现有 `webui` 的 Gradio 页面，而是采用了独立的前后端分离结构。

## 实现内容

### 1. 后端

新增目录：

- `data_platform/backend`

核心文件：

- `data_platform/backend/main.py`
- `data_platform/backend/store.py`
- `data_platform/backend/models.py`

后端技术栈：

- FastAPI
- Pydantic
- 本地文件扫描 + JSONL 解析

后端能力：

- 扫描 `cache/output/<run_id>/generate/*.jsonl`
- 读取同级 `config.yaml`
- 标准化提取：
  - `question`
  - `answer`
  - `image_path`
  - `sub_graph`
  - `sub_graph_summary`
  - `evidence_items`
- 维护 run 级统计缓存：
  - `question_texts`
  - `answer_texts`
  - `entity_type_counts`
  - `relation_type_counts`
  - `evidence_coverage`
- 提供受控图片访问，避免任意文件读取

### 2. 前端

新增目录：

- `data_platform/frontend`

前端技术栈：

- React
- TypeScript
- Vite
- Cytoscape.js

前端页面结构：

- 左栏：run 列表和目录导入
- 次左栏：样本列表、分页、搜索、过滤
- 主内容区：Question / Answer / 图片预览 / 交互图谱
- 右侧栏：节点或边的实际内容、evidence 列表、原文高亮

图谱交互能力：

- 节点 / 边点击选中
- metadata 检视
- `evidence_span` 展示与 evidence 联动
- 缩放、拖拽、fit 视图
- 按 `entity_type` 做节点颜色区分
- 显示 `relation_type` 边标签
- 点击 evidence 时高亮对应原文并联动图元素

## API 设计

当前实现的后端接口如下：

- `POST /api/imports/scan`
  - 输入：`{ "root_path": "cache" }`
  - 扫描 GraphGen 输出目录并建立内存索引

- `GET /api/runs`
  - 返回 run 列表

- `GET /api/runs/{run_id}/samples`
  - 支持分页
  - 支持 `search`
  - 支持 `has_image`
  - 支持 `has_graph`

- `GET /api/samples/{sample_id}`
  - 返回样本详情
  - 包含 `source_contexts`，用于 evidence 原文高亮

- `GET /api/assets?path=...`
  - 返回已索引图片文件
  - 非索引图片路径会被拒绝

- `GET /api/health`
  - 健康检查接口

## 数据标准化规则

当前实现默认面向 GraphGen 的 ChatML 输出。

标准化规则如下：

- `question`
  - 从 `messages` 中提取首个 `role=user` 的文本

- `answer`
  - 从 `messages` 中提取首个 `role=assistant` 的文本

- `image_path`
  - 优先从 user message content 中的 `image` 字段提取
  - 再解析为绝对路径

- `sub_graph`
  - 如果是字符串，则尝试反序列化 JSON
  - 如果解析失败，则保留浏览能力，但图谱不展示

- `evidence_items`
  - 直接从 `sub_graph.nodes[*].evidence_span`
  - 以及 `sub_graph.edges[*].evidence_span`
  - 聚合为统一列表
- `source_contexts`
  - 从 `messages` 中提取原始文本、图片 caption、表格文本
  - 作为前端 evidence 高亮的统一原文来源

## 测试与验证

新增测试文件：

- `tests/data_platform/test_backend_api.py`

覆盖内容包括：

- 扫描输出目录
- ChatML 解析
- 图片样本访问
- 非法 `sub_graph` 回退
- 搜索与分页
- `/api/assets` 安全限制

另外还做过的本地验证：

- `DataPlatformStore().scan("cache")` 已能扫描当前仓库里的真实 `cache`
- `python -m compileall data_platform/backend` 通过
- 前后端都已经在本地成功启动并验证过健康检查
- 附带 `data_platform/mock_cache` 作为前端演示数据

## 运行方式

### 后端

```bash
conda activate graphgen
uvicorn data_platform.backend.main:app --reload
```

默认地址：

- `http://127.0.0.1:8000`

### 前端

```bash
conda activate graphgen
cd data_platform/frontend
npm install
npm run dev
```

默认地址：

- `http://127.0.0.1:5173`

前端会通过 Vite 代理把 `/api/*` 转发到后端。

## 当前已知限制

- 当前主路径只针对 ChatML 做了兼容，Alpaca / ShareGPT 还没有作为主展示格式处理
- 统计缓存已经预留，但词频分析页面还没有实现
- `sub_graph` 仍然是基于当前 GraphGen 输出结构做展示，没有额外做图谱 schema 演化层
- run 与 sample 索引目前保存在内存中，适合本地单用户使用

## 后续建议

下一步比较适合继续做的方向：

1. 新增词频分析和 run 对比页面
2. 增强 evidence 展示，比如按 node / edge / source 聚合
3. 支持更多 GraphGen 输出格式
4. 引入持久化索引，减少重复扫描大目录的成本
