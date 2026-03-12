# Graph Mapping 与意图分析 PRD

## 1. 背景与目标

当前 `graph_agent/data` 的产物与预期不一致，主要表现为：

- `element_inventory.json` 为空，无法支撑后续高质量测绘。
- `graph.json` 中有效边过少（仅 1 条），且 `intent` 为 `null`。
- `intent_failure_reason` 显示 LLM 消息格式兼容问题，导致意图推断失败。
- 状态节点仍混入大量流程外状态（如 `write file`、`done`、`unknown`）。

本 PRD 目标是定义一套可落地的改造要求，使系统达到：

1) 可稳定生成可执行路径；  
2) 意图分析可观测、可重试、可覆盖；  
3) 图数据结构清晰且支持后续自动化回放。

## 2. 范围

### 2.1 In Scope

- Mapping 流程（Scout + Mapping）产物质量提升。
- 状态（state）生成与边过滤策略优化。
- AI-only 意图推断链路稳定化（无规则兜底）。
- 失败原因记录与缺失意图重推机制。
- API 与前端对空意图场景的兼容展示。

### 2.2 Out of Scope

- 新增复杂前端交互页面。
- 新增多浏览器并发调度能力。
- 引入新的向量检索或知识库系统。

### 2.3 兼容性策略

- 本次改造允许 **Breaking Changes**，可不兼容旧数据结构与旧接口行为。
- 以“可执行回放链路正确性”优先，不以向后兼容为约束。
- 若确需兼容，仅保留最小读取能力（读旧图失败时给明确错误与迁移提示）。

## 3. 术语定义

- `state`：图节点，代表一次可识别页面状态（优先 URL，缺失时使用可解释 pseudo-state）。
- `intent`：业务意图对象，可为空。
- `intent_failure_reason`：AI 推断失败原因，必须可追踪。
- `re-infer-missing`：对 `intent=null` 边执行重推覆盖的 CLI 能力。

## 4. 核心问题陈述

1. **可执行路径不足**：图中有效边数量不足，无法形成稳定回放链路。  
2. **意图缺失不可恢复**：虽然记录了失败原因，但缺少标准化重推覆盖流程。  
3. **测绘噪声偏高**：流程外动作影响状态/边质量，降低图可用性。  
4. **观测与验收标准不明确**：缺少统一质量门槛与统计口径。

## 5. 产品需求

## 5.0 跨模块联动要求（必须实现）

为确保“测绘 -> 意图 -> 寻径 -> 回放”可执行，本需求必须联动改造 `@graph_agent` 以下模块：

- `graph_agent/mapping/parser.py`：意图推断与失败原因写入。
- `graph_agent/mapping/run.py`：构图、过滤、重推入口与统计。
- `graph_agent/graph/io.py`：图结构读写与字段一致性。
- `graph_agent/graph/pathfinding.py`：空意图容错与路径选择稳定性。
- `graph_agent/playback/engine.py`：回放日志与空意图兼容执行。
- `graph_agent/web/app.py`：意图列表输出与诊断数据暴露。
- （如涉及）`graph_agent/web/static/*`：前端展示结构同步。

禁止只改单模块导致链路断裂；任一模块字段变化必须同步更新上下游。

## 5.1 数据质量要求

- `graph.json` 中每条边必须满足：
  - `source`/`target`/`selector`/`action` 字段完整。
  - `intent` 可为空，但为空时必须有 `intent_failure_reason`。
- `element_inventory.json` 不允许长期为空；为空需标记失败并终止流程。
- `metadata` 需保留：
  - `filtered_non_ui_edges`
  - `mapping_stopped`
  - `stop_reason`
  - 新增建议：`intent_missing_count`

## 5.2 状态生成要求

- URL 可用时：使用清洗后的 URL 作为节点 ID。
- URL 缺失时：生成可解释 pseudo-state（包含 step/action/brief hash）。
- 不允许仅使用不可解释的裸序号状态（如纯 `State 12`）。

## 5.3 边过滤策略

- 保留 agent 执行自由度，但落盘时过滤非业务动作边（如 `read_file`/`write_file`/`done`/`unknown`）。
- 过滤动作计数必须写入 `metadata.filtered_non_ui_edges`。

## 5.4 AI-only 意图推断

- 推断必须由 AI 生成（`key/confidence/summary/verb/object`）。
- 禁止规则兜底生成伪意图。
- 失败处理：
  - `intent = null`
  - 记录 `intent_failure_reason`
  - 打印失败日志（action/selector/source/target/reason）。

## 5.5 缺失意图重推

- 提供 CLI 参数：`--re-infer-missing`。
- 行为：
  1) 读取现有图；
  2) 找出 `intent=null` 边；
  3) 逐条重推；
  4) 成功覆盖 intent 并清空 failure reason；
  5) 输出统计（total/succeeded/failed）。

## 5.6 API 与展示

- `/api/intents` 必须忽略 `intent=null` 边。
- `/api/graph` 应返回 `intent_failure_reason` 便于排障。
- 前端可显示缺失意图数量或诊断提示（建议项）。

## 5.7 回放可执行性保障

- 即使图中存在 `intent=null` 边，回放核心链路也必须可执行，不允许因空意图直接中断。
- 回放模块不得强依赖 `intent.summary` 非空，日志与执行流程需容错。
- 寻径模块输出的边序列必须满足回放最小可执行条件：
  - 每条边包含有效 `selector` 与可执行 `action`；
  - 路径首步可从起始 URL 触达；
  - 终点状态与预期 URL 校验策略一致。

## 6. 验收标准（必须满足）

## 6.1 功能验收

- 可通过一次完整 mapping 生成包含多条可执行业务边的图。
- 可通过 `--re-infer-missing` 对缺失意图进行覆盖更新。
- 回放接口在空意图边存在时不崩溃。

## 6.2 质量验收

- 关键测试全通过（路径、I/O、意图、重推）。
- lints / type checks 无新增错误。
- 目标站点一次测绘后满足：
  - `intent_missing_count / edge_count <= 30%`（初始阶段门槛）
  - 至少 1 条长度 >= 2 的可执行业务路径
- 新增联动验收：
  - `pathfinding -> playback` 端到端测试至少 1 条通过；
  - 任一模块字段变更后，全链路测试仍通过。

## 7. 指标与观测

- `edge_count`
- `intent_missing_count`
- `intent_success_rate = 1 - intent_missing_count / edge_count`
- `filtered_non_ui_edges`
- `re_infer_success_rate`

## 8. 里程碑建议

- M1：数据模型与 I/O 兼容完成（intent 可空 + failure reason）。
- M2：AI-only 推断与日志完成。
- M3：`--re-infer-missing` 完成并可覆盖。
- M4：验证口径落地（测试 + 指标输出）。

## 9. 风险与缓解

- **风险**：LLM 输出不稳定或格式漂移。  
  **缓解**：严格 JSON 提示词 + 失败记录 + 重推能力。

- **风险**：站点本身存在不可自动化节点（如认证弹窗）。  
  **缓解**：停止原因标准化写入 `stop_reason`，并在统计中剔除不可达步骤。

- **风险**：测绘范围过大导致噪声状态增多。  
  **缓解**：任务模板限制探索边界与步数，按域分批测绘。

## 10. 交付物

- `prd.md`（本文档）
- 可运行的 mapping/re-infer CLI
- 通过的自动化测试报告
- 更新后的 `graph.json` 样例与质量指标输出
- 跨模块变更清单（模块、字段、影响面、迁移说明）

## 11. Tasks（for Ralph）

以下任务按执行顺序编排，Ralph 可逐项领取并提交。

### 11.0 任务状态看板（Checklist）

> 用于展示当前完成状态，执行中请直接勾选更新。

- [x] T1 GraphEdge 模型改造（Breaking）
- [x] T2 Parser 改为 AI-only 推断（无规则兜底）
- [x] T3 Graph I/O 兼容 `intent=null`
- [x] T4 Mapping 构图与统计联动
- [x] T5 新增 `--re-infer-missing` 重推能力
- [x] T6 Pathfinding 空意图容错
- [x] T7 Playback 回放兼容
- [x] T8 API 与前端展示联动
- [x] T9 测试与验收
- [x] T10 端到端验收（目标站点）

示例：

```markdown
## Tasks
- [x] create auth
- [x] add dashboard
- [x] done task (skipped)
```

### T1. GraphEdge 模型改造（Breaking）

- **目标**：允许 `intent` 为空，并记录 `intent_failure_reason`。
- **涉及模块**：`graph_agent/models.py`
- **任务**：
  - 将 `GraphEdge.intent` 改为 `Intent | None`。
  - 新增 `GraphEdge.intent_failure_reason: str | None`。
  - 确保 Pydantic 校验可接受 `intent=null`。
- **DoD**：
  - 新旧样例对象均可构造通过。
  - 相关类型检查无新增错误。

### T2. Parser 改为 AI-only 推断（无规则兜底）

- **目标**：完全移除规则兜底，仅保留 AI 推断与失败记录。
- **涉及模块**：`graph_agent/mapping/parser.py`
- **任务**：
  - 保留结构化 AI 输出（`key/confidence/summary/verb/object`）。
  - 失败时设置 `intent=None`，并写入 `intent_failure_reason`。
  - 打印失败日志（至少包含 `action/selector/source/target/reason`）。
  - 删除或停用规则 fallback 路径。
- **DoD**：
  - AI失败不再生成伪意图。
  - 失败原因可在日志和图边中追踪。

### T3. Graph I/O 兼容 `intent=null`

- **目标**：图读写完整支持空意图和失败原因。
- **涉及模块**：`graph_agent/graph/io.py`
- **任务**：
  - `save_graph` 正确落盘 `intent=null` 与 `intent_failure_reason`。
  - `load_graph` 正确恢复上述字段，不抛异常。
  - 保持 metadata 字段读写一致。
- **DoD**：
  - round-trip（save -> load）后字段一致。
  - 旧图读取行为可预期（失败时报清晰错误）。

### T4. Mapping 构图与统计联动

- **目标**：构图可保留空意图边，并保留现有动作过滤策略。
- **涉及模块**：`graph_agent/mapping/run.py`
- **任务**：
  - 写边时保留 `intent` 及 `intent_failure_reason`。
  - 继续过滤非业务动作边（`read_file/write_file/done/unknown`）。
  - metadata 保留 `filtered_non_ui_edges`、`mapping_stopped`、`stop_reason`。
  - 建议新增 `intent_missing_count` 统计。
- **DoD**：
  - 新图能看到空意图边与失败原因。
  - 统计字段可用于质量评估。

### T5. 新增 `--re-infer-missing` 重推能力

- **目标**：支持对 `intent=null` 边进行手动重推覆盖。
- **涉及模块**：`graph_agent/mapping/run.py`（必要时拆 helper）
- **任务**：
  - CLI 新增 `--re-infer-missing`。
  - 读取图 -> 筛选空意图边 -> 调用 AI 重推 -> 成功覆盖。
  - 覆盖成功后清空 `intent_failure_reason`。
  - 输出统计：`total/succeeded/failed`。
- **DoD**：
  - 可对现有图执行重推，且结果可落盘复现。

### T6. Pathfinding 空意图容错

- **目标**：寻径稳定跳过空意图边，避免崩溃。
- **涉及模块**：`graph_agent/graph/pathfinding.py`
- **任务**：
  - 匹配前判断 `intent is None`，直接跳过。
  - 保证返回路径对象对回放模块可用。
- **DoD**：
  - 存在空意图边时，寻径不报错且可返回有效路径。

### T7. Playback 回放兼容

- **目标**：回放不依赖 `intent.summary` 必然存在。
- **涉及模块**：`graph_agent/playback/engine.py`
- **任务**：
  - 日志构造时兼容 `edge.intent is None`。
  - 执行逻辑仅依赖 `selector/action/data_key/constraints`。
- **DoD**：
  - 含空意图边的路径回放不因日志字段报错。

### T8. API 与前端展示联动

- **目标**：前端不显示空意图，诊断信息可见。
- **涉及模块**：`graph_agent/web/app.py`、`graph_agent/web/static/*`
- **任务**：
  - `/api/intents` 忽略 `intent=null` 边。
  - `/api/graph` 返回 `intent_failure_reason`。
  - 前端可选展示 missing 计数或失败原因摘要。
- **DoD**：
  - 意图下拉不出现空值选项。
  - 图接口可用于排障。

### T9. 测试与验收

- **目标**：形成稳定可回归测试集。
- **涉及模块**：`tests/graph_agent/*`
- **任务**：
  - 新增/更新用例覆盖：
    - AI失败 -> `intent=None + failure_reason`
    - pathfinding 跳过空意图
    - `--re-infer-missing` 覆盖成功与失败分支
    - I/O round-trip 包含 `intent=null`
    - 回放模块空意图兼容
  - 执行全量相关测试与 lints。
- **DoD**：
  - 关键测试全部通过。
  - 无新增 lints/type errors。

### T10. 端到端验收（目标站点）

- **目标**：验证“测绘 -> 意图 -> 寻径 -> 回放”链路可执行。
- **建议站点**：`https://the-internet.herokuapp.com`
- **任务**：
  - 运行 mapping 生成图。
  - 统计 `edge_count`、`intent_missing_count`、`intent_success_rate`。
  - 执行一次 `--re-infer-missing` 并记录改进。
  - 选择至少 1 条长度 >= 2 的路径完成回放。
- **DoD**：
  - 达到 PRD 第 6 节验收标准。

## 12. Ralph 执行规范

- 每完成一个 Task，提交：
  - 变更文件清单
  - 关键 diff 说明
  - 测试命令与结果
  - 风险与后续动作
- 若发现与 PRD 冲突，先提变更建议，不直接偏离。
- 禁止只改单模块后宣称完成；必须提供链路级验证证据。
