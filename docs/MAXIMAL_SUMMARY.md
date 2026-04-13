# 最大化测绘方案总结

## 目标
实现网站的"一次性最大化测绘"，达到 95%+ 的功能覆盖率。

## 核心改进

### 1. 移除步数限制
- 原：每个 Zone 25 步硬限制
- 新：基于完成度判断终止（目标 95%）
- 软限制：1000 步/10 分钟（防止无限循环）

### 2. 智能策略检测
根据页面特征自动选择最佳探索策略：
- FORM：测试所有字段组合
- TABLE：测试分页/排序/筛选
- MENU：递归展开所有层级
- MODAL：测试所有弹窗

### 3. 细粒度断点续传
```
Session (10分钟) 
  → Zone (60秒)
    → Element (每次交互)
```
中断后可从任意元素恢复。

### 4. 并行探索
- 同时探索多个 Zone
- 独立浏览器实例
- 可配置并发数（推荐 3-6）

### 5. 递归菜单探索
- 自动展开所有菜单层级
- 记录完整菜单路径
- 支持返回后重新探索

## 新增文件

| 文件 | 功能 | 代码量 |
|------|------|--------|
| maximal_explorer.py | 最大化探索核心 | ~500 行 |
| recursive_menu_explorer.py | 递归菜单探索 | ~350 行 |
| form_combination_tester.py | 表单组合测试 | ~150 行 |

## 预期效果

### 覆盖提升
| 场景 | 传统 | 最大化 |
|------|------|--------|
| 简单表单 | 60% | 98% |
| 复杂表格 | 40% | 95% |
| 多级菜单 | 30% | 90% |
| 模态框密集 | 20% | 85% |

### 时间估算
| 网站规模 | 预计时间 | 配置 |
|---------|---------|------|
| 小型(<20页) | 2-4h | 2并行 |
| 中型(20-100页) | 8-16h | 4并行 |
| 大型(>100页) | 24-48h | 6并行 |

## 关键配置

```python
ExplorationConfig(
    min_completion_ratio=0.95,  # 目标完成度
    max_steps_per_zone=1000,    # 单 Zone 最大步数
    max_time_per_zone_ms=600_000,  # 单 Zone 最大时间
    max_parallel_zones=3,       # 并行 Zone 数
    enable_smart_strategy=True, # 启用智能策略
    enable_recursive_menu=True, # 递归菜单
    enable_form_combination=True,  # 表单组合
)
```

## 使用示例

```python
# 基础用法
orchestrator = MaximalOrchestrator(driver, config)
result = await orchestrator.run_maximal_exploration(
    start_url="https://example.com",
    app_id="my_app",
    time_budget_hours=24,
)

print(f"完成度: {result['total_completion_ratio']:.1%}")
print(f"States: {result['states_discovered']}")
print(f"Zones: {result['zones_explored']}")
```

## 注意事项

1. **资源消耗**：需要更多浏览器实例和内存
2. **时间成本**：完整测绘可能需要 24-48 小时
3. **存储增长**：更多 transitions/checkpoints 需要存储
4. **网络流量**：更多交互产生更多网络请求

## 下一步优化

1. AI 预测：预测哪些交互最可能发现新状态
2. 自适应步数：根据页面复杂度动态调整
3. 结果去重：避免重复记录相同 transition
4. 可视化报告：生成探索覆盖热力图
