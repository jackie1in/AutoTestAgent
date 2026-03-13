# Task 8: 回归与收尾

## 执行摘要

- **日期**: 2026-03-13
- **状态**: 完成

## 验收结果

### 1. Focused Suites

以下测试全部通过（107 个用例）：

- `tests/graph_agent/test_models.py`
- `tests/graph_agent/test_io.py`
- `tests/graph_agent/test_parser.py`
- `tests/graph_agent/test_run.py`
- `tests/graph_agent/test_playback.py`

### 2. Lint

- `ruff check graph_agent tests` 通过
- `ruff format --check graph_agent tests` 通过
- 涉及文件 ReadLints 无新增错误

### 3. 本轮修复

为通过 lint，进行了最小修改：

- `graph_agent/mapping/run.py`: 移除未使用的 `descriptor` 变量及 `re` 导入
- `graph_agent/mapping/scout.py`: 将歧义变量名 `l` 重命名为 `label_lower`
- `graph_agent/run_mapping.py`: 为脚本内 import 添加 `# noqa: E402`

## Residual Risks（剩余风险）

1. **Selector 漂移**
   - 真实站点页面结构变动快，绝对 `xpath` 易失效
   - 建议：定期重录图谱；对关键业务链增加多 selector 兜底

2. **iframe 异步加载**
   - 业务页可能在 iframe 内异步加载，录制与回放时序可能不一致
   - 建议：对 iframe 场景增加显式 wait 策略；在验收中单独标记 iframe 链路

3. **多标签页元数据**
   - 若录制侧未稳定产出 tab 元数据，回放会继续失败
   - 建议：在 mapping 阶段增加 tab 产出校验；对 OPEN/SWITCH/CLOSE 场景做专项回归

4. **同 URL 多状态**
   - 同 URL 多状态页面若状态建模不足，可能继续出现路径错误折叠
   - 建议：继续依赖 opaque state id + URL 元数据；对关键业务链增加状态区分用例

5. **真实站点验收依赖**
   - 真实验收依赖 `mapping.run` 产出的 `graph.json`，站点变更会影响结果
   - 建议：保留 fixture 图用于单元测试；真实验收作为独立 CI 或手动触发

## 后续建议

1. **持续回归**
   - 每次改动后运行 focused suites
   - 在 CI 中集成 `ruff check` 和 `ruff format`

2. **真实验收**
   - 定期执行 `uv run python -m graph_agent.mapping.run` 更新图谱
   - 使用 `test_playback_acceptance.py` 做真实回放验收

3. **文档与样本**
   - 维护目标业务链说明文档
   - 对失败样本保留日志与图谱快照便于调试
