# 最大化测绘指南

## 核心特性

1. 无步数限制探索 - 基于完成度判断终止
2. 智能任务分解 - 根据页面类型自动选择策略
3. 并行探索 - 同时探索多个 Zone
4. 细粒度断点续传 - 元素级别进度跟踪
5. 递归菜单探索 - 自动展开所有菜单层级

## 快速开始

```python
from graph_agent.cartography import (
    MaximalOrchestrator,
    ExplorationConfig,
)

config = ExplorationConfig(
    min_completion_ratio=0.95,
    max_steps_per_zone=1000,
    max_parallel_zones=3,
)

orchestrator = MaximalOrchestrator(driver, config)
result = await orchestrator.run_maximal_exploration(
    start_url="https://example.com",
    app_id="my_app",
    time_budget_hours=24,
)
```

## 配置选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| min_completion_ratio | 0.95 | 目标完成度 |
| max_steps_per_zone | 1000 | 单 Zone 最大步数 |
| max_time_per_zone_ms | 600000 | 单 Zone 最大时间 |
| max_parallel_zones | 3 | 并行 Zone 数 |

## 完成度计算

完成度 = 0.4 * 元素点击率 + 0.2 * 输入框填充率 + 0.1 * 下拉框测试率 + 0.2 * 表单提交率 + 0.1 * 模态框测试率

## 推荐配置

小型网站(<20页面): min_completion_ratio=0.98, max_parallel_zones=2
中型网站(20-100): min_completion_ratio=0.95, max_parallel_zones=4
大型网站(>100): min_completion_ratio=0.90, max_parallel_zones=6
