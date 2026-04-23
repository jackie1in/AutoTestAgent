# Graph Agent

基于 Browser-Use 的 UI 自动化：**测绘 → 图谱 → 寻径 → 回放**。先通过浏览器探索流程并生成有向图，再按业务意图寻径，最后用测试数据回放。

## Overview

- **测绘**：用自定义 CartographyAgent（手写 ReAct）探索登录/业务流程，记录每一步的 selector、动作与语义意图。
- **图谱**：图持久化为 Neo4j（节点为 State/Zone/Menu，边为 Transition，含 selector / action / intent / param_name / action_value），供前端展示与寻径使用。
- **寻径**：根据用户选择的意图在 Neo4j 中查询 transitions，用 GraphEdge 列表做 DFS 匹配找到从起点到目标状态的路径。
- **回放**：按路径顺序执行 click/fill 等操作，用用户填写的测试数据填充表单并验证结果。

## Install

在项目根目录（含 `pyproject.toml`）执行：

```bash
uv sync --group graph_agent
playwright install chromium
```

## Env

运行测绘或回放前需配置 LLM：

- **OpenAI**：`OPENAI_API_KEY`（可选 `OPENAI_MODEL`，默认 `gpt-4o-mini`）
- **Anthropic**：`ANTHROPIC_API_KEY`（可选 `ANTHROPIC_MODEL`）

任选其一即可；测绘与回放会从环境变量读取。

## Run mapping

测绘流程为**先 Scout 再 Mapping**：先扫描页面得到可交互元素清单，再探索流程并仅保留清单内元素对应的边。在项目根目录执行：

```bash
uv run python -m graph_agent.cartography.runner
uv run python graph_agent/run_mapping.py
```

可选参数：`--url`（起始 URL）、`--inventory`（清单 JSON 路径，默认 `graph_agent/data/element_inventory.json`）。图谱直接写入 Neo4j，无需 `--output`。

若要录制登录流程，可在 `.env` 中设置 `MAPPING_USERNAME` 与 `MAPPING_PASSWORD`。仅当这两个变量至少一个存在时，Mapping 才会追加登录相关提示词；未配置时不会注入登录提示。

示例：指定 URL 与清单路径

```bash
uv run python -m graph_agent.cartography.runner --url https://example.com/login --inventory graph_agent/data/element_inventory.json
```

## Start web

启动 FastAPI + 前端（开发时热重载）：

```bash
uv run uvicorn graph_agent.web.app:app --reload
```

浏览器打开 [http://localhost:8000/](http://localhost:8000/) 即可使用。

## Usage

1. **图谱**：页面顶部展示从 Neo4j 查询的图，节点为 State/Zone/Menu，边为 Transition（selector、动作、语义意图）。
2. **意图**：从下拉框选择一条语义意图（对应边的 `semantic_label`），用于寻径。
3. **测试数据**：可按表单真实参数名填写测试数据（如 username、password）；回放优先使用录制值，缺失时再按图中的 `param_name` 从 `test_data` 取值。
4. **回放**：点击「回放」后，后端按所选意图寻径并执行回放，日志以 SSE 流式输出在「回放日志」区域。
