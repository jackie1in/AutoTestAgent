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

按需查知识（可选，默认关闭）：

- `CARTOGRAPHY_KNOWLEDGE_ON_DEMAND_ENABLED=false`
- `CARTOGRAPHY_KNOWLEDGE_MIN_INTERVAL_SEC=15`
- `CARTOGRAPHY_KNOWLEDGE_TRIGGER_SCORE_THRESHOLD=2.0`
- `CARTOGRAPHY_KNOWLEDGE_TRIGGER_PROFILE=balanced`（可选：`conservative` / `balanced` / `aggressive`）
- `CARTOGRAPHY_KNOWLEDGE_QUERY_TIMEOUT_MS=1200`
- `CARTOGRAPHY_KNOWLEDGE_TOPK=5`
- `MAPPING_RELEASE_ID=`（可选，设置后优先读取该 release 的 active revisions）

### 策略切换（UI / API）

当前支持三种触发策略模板：

- `conservative`：更少触发查询，偏保守
- `balanced`：默认策略，均衡触发频率与命中收益
- `aggressive`：更容易触发查询，偏激进

#### UI 操作

1. 启动 Web 后，在页面顶部的「策略」下拉框选择 `conservative` / `balanced` / `aggressive`。
2. 切换成功后，会在页面消息区看到“已切换按需查知识策略”提示。
3. 顶部 dashboard 摘要中的“策略”会同步刷新为当前值。

#### API 操作

查询当前策略：

```bash
curl -s http://localhost:8000/api/settings/knowledge
```

返回示例：

```json
{
  "trigger_profile": "balanced",
  "allowed_profiles": ["aggressive", "balanced", "conservative"]
}
```

切换策略：

```bash
curl -s -X POST http://localhost:8000/api/settings/knowledge \
  -H "Content-Type: application/json" \
  -d '{"trigger_profile":"aggressive"}'
```

返回示例：

```json
{
  "ok": true,
  "trigger_profile": "aggressive"
}
```

无效值会返回 `400`，并附带 `allowed_profiles`。

> 说明：该切换是**进程内实时生效**（当前运行中的 Web/Worker 进程）。重启后会按环境变量 `CARTOGRAPHY_KNOWLEDGE_TRIGGER_PROFILE` 重新加载。

#### 推荐策略选择指南

- **稳定性优先（生产巡检 / 成本敏感）**：优先 `conservative`
  - 适用：页面结构稳定、历史知识质量较高、希望减少查询开销与扰动。
  - 观察指标：`knowledge_query_count` 低、`knowledge_timeout_count` 低、流程成功率稳定。
  - 何时升级到 `balanced`：出现连续“卡住/失败/低置信”且命中率提升空间明显。

- **覆盖率优先（新系统接入 / 页面频繁变化）**：优先 `aggressive`
  - 适用：站点改版频繁、初次测绘、希望尽快利用历史知识补全路径。
  - 观察指标：`knowledge_hit_count` 与命中率上升，失败步数下降。
  - 何时降级到 `balanced`/`conservative`：超时率持续偏高（如 >20%）或收益不明显。

- **默认均衡（大多数场景）**：使用 `balanced`
  - 适用：日常回归与持续测绘，兼顾稳定性与覆盖率。
  - 调整建议：先用 `balanced` 观察 1~2 个会话，再根据“命中率/超时率/失败步数”微调。

一个简单决策顺序：

1. 先用 `balanced`。
2. 若“失败多、卡住多、命中有收益” → 切 `aggressive`。
3. 若“超时高、收益低、流程已稳” → 切 `conservative`。

## Run mapping

测绘流程为**先 Scout 再 Mapping**：先扫描页面得到可交互元素清单，再探索流程并仅保留清单内元素对应的边。在项目根目录执行：

```bash
uv run python -m graph_agent.cartography.runner
uv run python graph_agent/run_mapping.py
```

可选参数：`--url`（起始 URL）、`--inventory`（清单 JSON 路径，默认 `graph_agent/data/element_inventory.json`）。图谱直接写入 Neo4j，无需 `--output`。

若要录制登录流程，可在 `.env` 中设置 `MAPPING_USERNAME` 与 `MAPPING_PASSWORD`。仅当这两个变量至少一个存在时，Mapping 才会追加登录相关提示词；未配置时不会注入登录提示。

启用按需查知识后，pipeline 仅在“低置信/失败/冲突/卡住”等触发条件达到阈值时发起查询，并在超时或查询异常时自动降级，不会中断测绘流程。相关命中与延迟指标会写入会话统计（`layout_metrics` 扩展字段）。

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
