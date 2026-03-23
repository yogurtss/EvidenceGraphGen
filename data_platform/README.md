# GraphGen Data Platform

独立的数据平台用于浏览 GraphGen 的生成结果，重点支持：

- 导入 `cache` 这类 GraphGen 输出目录
- 浏览 Question / Answer
- 预览 VQA 图片
- 可视化 `sub_graph`
- 展示节点和边上的 `evidence_span`

## 目录结构

- `data_platform/backend`
  Python + FastAPI 后端，负责扫描 `cache/output/<run_id>/generate/*.jsonl`
- `data_platform/frontend`
  React + Vite 前端，负责三栏工作台和交互图谱

## 启动后端

在项目根目录执行：

```bash
uvicorn data_platform.backend.main:app --reload
```

默认监听 `http://127.0.0.1:8000`。

## 启动前端

在另一个终端执行：

```bash
cd data_platform/frontend
npm install
npm run dev
```

默认监听 `http://127.0.0.1:5173`，并通过 Vite 代理把 `/api/*` 请求转发到后端。

## 使用方式

1. 启动后端和前端。
2. 打开前端页面。
3. 在左上角导入框输入 GraphGen 输出目录，例如 `cache`。
4. 导入后选择某个 run。
5. 在页面中间查看 Question / Answer / Image / sub_graph。
6. 在右侧查看 node/edge 实际内容、全部 evidence，以及 evidence 对应的原文高亮。

## Mock 数据

仓库内已附带一组可直接预览的测试数据：

- 目录：`data_platform/mock_cache`
- 默认前端导入路径也会指向这份 mock 数据

如果你还没有自己的 `cache` 输出，可以直接启动前后端后立即浏览这组示例。
