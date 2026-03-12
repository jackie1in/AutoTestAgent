# 嵌套 iframe 录制与回放设计

**目标**

为 `graph_agent` 增加对嵌套 iframe 的完整支持：录制阶段能够识别交互元素所在的多层 frame 路径，持久化到图数据中，回放阶段能够按该路径逐层进入正确的 frame 后执行 `click` / `fill`。

**背景**

当前系统只记录元素自身的 `selector` 与快照信息，但没有记录元素所属的 frame 上下文。对于 iframe 内元素，尤其是多层嵌套 iframe，单独保存元素 selector 不足以在回放时唯一定位目标，因此会导致录制成功、回放失败，或者错误地在顶层文档查找元素。

**设计原则**

1. 明确建模 frame 上下文，而不是把 frame 信息塞进 `selector` 字符串。
2. 一次支持嵌套 iframe，避免先做单层后再重构。
3. 保持顶层文档元素的行为不变，非 iframe 场景应继续使用现有路径。
4. 错误信息应区分“frame 找不到”和“frame 内元素找不到”。

## 方案选择

### 方案 A：把 frame 信息编码进 `selector`

把 iframe 链路拼成自定义 selector 前缀，例如 `frame(...) >> frame(...) >> #target`。

**优点**

- 改动面最小。

**缺点**

- 污染 selector 语义。
- 容易与现有 xpath/css 规则冲突。
- 不利于测试、调试和后续扩展。

### 方案 B：只增加单层 frame 字段

为边或元素增加 `frame_selector`，仅记录目标元素所在的直接 iframe。

**优点**

- 实现简单。

**缺点**

- 无法满足本次“嵌套 iframe”目标。
- 后续仍需重构为链式结构。

### 方案 C：显式 `frame_path` 建模

为元素和边增加有序 `frame_path` 数组，保存从顶层页面到目标元素所在 frame 的完整定位链。

**优点**

- 与嵌套 iframe 目标直接匹配。
- 数据结构清晰，便于序列化、调试和测试。
- 回放逻辑可以逐层下钻，错误定位更准确。

**缺点**

- 模型、录制、存储、回放、测试都需要联动修改。

**结论**

采用方案 C。

## 数据模型设计

### 新增模型

建议新增 `FrameLocatorSnapshot`：

- `selector: str`
- `xpath: str | None = None`
- `x_path: str | None = None`
- `css_selector: str | None = None`
- `name: str | None = None`
- `id: str | None = None`
- `attributes: dict[str, Any] = Field(default_factory=dict)`

该结构用于描述单层 iframe 元素本身的可回放定位信息。

### 扩展现有模型

在 `ElementSnapshot` 增加：

- `frame_path: list[FrameLocatorSnapshot] = Field(default_factory=list)`

在 `GraphEdge` 增加：

- `frame_path: list[FrameLocatorSnapshot] = Field(default_factory=list)`

说明：

- `ElementSnapshot.frame_path` 用于保留原始交互元素上下文。
- `GraphEdge.frame_path` 用于回放时直接消费，避免每次再从 `element` 中推断。
- 顶层页面元素使用空数组 `[]`。

## 录制设计

修改 `graph_agent/mapping/parser.py`。

### 提取策略

新增辅助函数：

- `_frame_snapshot_from_interacted(...)`
- `_extract_frame_path_from_interacted(...)`

目标是从 `interacted_element` 中递归提取 frame 父链。由于 browser-use 返回结构可能是对象、字典、列表混合，提取逻辑应复用现有 normalize 风格，兼容：

- 字典对象
- 带 `to_dict()` / `dict()` 的对象
- 可能存在的父 frame 字段或 owner frame 字段

### 结果落点

- `ElementSnapshot.frame_path` 保存完整路径。
- `parse_browser_use_step()` 返回 `GraphEdge` 时，把同样的路径写入 `edge.frame_path`。

### 回退策略

若当前 browser-use 提供的数据中无法提取 frame 父链：

- 不伪造路径。
- 记录为空数组。
- 顶层场景不受影响。

这意味着“录制端是否真的拿得到 frame 链”是交付风险之一，需要用真实页面验证。

## 存储设计

修改：

- `graph_agent/models.py`
- `graph_agent/graph/io.py`

要求：

- `save_graph()` 正确序列化 `GraphEdge.frame_path` 和 `ElementSnapshot.frame_path`
- `load_graph()` 正确反序列化

本项目当前对旧格式没有强兼容要求，因此不为历史 graph 数据做复杂迁移；旧图缺少 `frame_path` 时按默认空数组处理即可。

## 回放设计

修改 `graph_agent/playback/engine.py`。

### 新增辅助函数

- `_resolve_playback_context(page, frame_path)`
- `_locator_in_context(page, selector, frame_path)`

### 执行逻辑

1. 初始上下文为顶层 `page`
2. 遍历 `frame_path`
3. 每一层通过 `frame_locator(frame_selector)` 下钻
4. 在最终上下文中执行 `locator(selector)`
5. 保持现有 `fill` / `click` / `wait_for_network` 流程不变

### 错误语义

如果第 N 层 frame 无法定位，返回类似：

- `Failed to locate iframe level 1: <selector>`
- `Failed to locate iframe level 2: <selector>`

如果 frame 已定位但目标元素失败，继续沿用现有元素操作错误。

### 非目标范围

- 不处理跨域 iframe 的安全限制规避。
- 不增加 frame 自动恢复或模糊匹配。
- 不改造意图推理逻辑，frame 只影响定位与回放。

## 测试设计

### `tests/graph_agent/test_models.py`

- 验证 `GraphEdge` / `ElementSnapshot` / 新增 frame 模型的序列化与默认值。

### `tests/graph_agent/test_parser.py`

- 新增顶层元素 `frame_path == []`
- 新增单层 iframe 提取
- 新增双层嵌套 iframe 提取

### `tests/graph_agent/test_io.py`

- 保存再加载 graph 后，`frame_path` 不丢失。

### `tests/graph_agent/test_playback.py`

使用 fake page / fake locator 扩展测试桩，新增：

- 顶层元素仍走 `page.locator(selector)`
- 单层 iframe 走一次 frame 下钻
- 双层 iframe 走两次 frame 下钻
- 缺失中间 frame 时返回明确错误

## 风险与验证

### 主要风险

1. browser-use 当前 `interacted_element` 结构里可能不包含完整父 frame 链。
2. fake Playwright 测试桩需要补充 `frame_locator()` 才能覆盖回放逻辑。
3. 部分 iframe selector 可能不稳定，导致真实页面回放依赖 xpath/css 的优先级选择。

### 验证标准

满足以下条件即可认为改造完成：

1. 顶层页面已有测试全部保持通过。
2. parser 能从嵌套 iframe 元素中产出正确 `frame_path`。
3. graph 保存与加载后 `frame_path` 保持不变。
4. playback 能在双层 iframe 路径下把 `click` / `fill` 发到正确上下文。
5. 当某层 iframe 丢失时，报错明确指出失败层级。

## 实施顺序

1. 先补模型与 IO 的失败测试。
2. 再补 parser 的 `frame_path` 提取失败测试。
3. 再补 playback 的嵌套 iframe 回放失败测试。
4. 逐步实现最小代码直至全部转绿。
5. 最后用真实页面做一次手工验证，确认 browser-use 侧确实提供了可提取的 frame 链。
