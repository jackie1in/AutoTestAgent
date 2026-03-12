# 多标签页上下文支持设计

**目标**

为 `graph_agent` 增加完整的多标签页上下文支持，覆盖以下生命周期：

1. 当前标签页触发新标签页或 popup
2. 后续操作继续在新标签页执行
3. 在多个已打开标签页之间显式切换
4. 关闭指定标签页
5. 关闭后切回其他仍存活标签页

该能力需要同时作用于录制、图存储和回放，不再假设系统始终只操作单个 `page`。

## 背景

当前实现存在以下单页假设：

- `mapping/scout.py` 与 `mapping/run.py` 初始化导航都固定 `new_tab: false`
- `playback/engine.py` 只创建并维护一个 `page`
- `GraphEdge` 目前只表达页面内交互，不表达标签页生命周期事件
- `mapping` 记录的 `source_url` / `target_url` 无法区分“同 URL 不同 tab”的上下文

这会导致如下问题：

- 点击打开新页后，系统无法知道后续动作属于哪个标签页
- 关闭或切换标签页的行为无法被建模和回放
- 两个标签页 URL 相同或标题相似时，无法可靠区分当前活动页

## 方案比较

### 方案 A：依赖 URL 或标题推断活动标签页

通过 URL / title 推断当前动作属于哪个 tab。

**优点**

- 改动小
- 无需引入新的生命周期事件类型

**缺点**

- 相同 URL 或相同标题页面下会歧义
- popup 打开后若先经历跳转链，推断会很脆弱
- 不适合关闭后再切回的场景

**结论**

不采用。

### 方案 B：只支持“打开新页后继续”

只记录 popup 的来源页和结果页，不支持后续显式切换/关闭建模。

**优点**

- 第一版实现最简单

**缺点**

- 不满足本次“完整生命周期”目标
- 未来仍需重构

**结论**

不采用。

### 方案 C：显式标签页上下文建模

把标签页生命周期动作作为一等事件建模，每条边记录明确的 `tab_id`。

**优点**

- 能准确表达打开、切换、关闭与后续页面内操作
- 与 iframe 支持兼容，形成“tab + frame + selector”完整定位链
- 回放错误信息可以精确定位到 tab 层

**缺点**

- 需要修改模型、录制、图构建、回放和测试

**结论**

采用方案 C。

## 数据模型设计

### 新增枚举

新增 `TabActionType`：

- `OPEN`
- `SWITCH`
- `CLOSE`

用于描述标签页生命周期动作。

### 新增模型

新增 `TabSnapshot`，用于表示已知标签页上下文：

- `tab_id: str`
- `opener_tab_id: str | None = None`
- `url: str | None = None`
- `title: str | None = None`

### 扩展 `GraphEdge`

新增字段：

- `tab_id: str = "tab-0"`
- `target_tab_id: str | None = None`
- `tab_action: TabActionType | None = None`
- `tab: TabSnapshot | None = None`

约定：

- 普通页面内动作：`tab_action is None`，`tab_id` 必填
- 打开新页动作：`tab_action == OPEN`，`tab_id` 是发起页，`target_tab_id` 是新页
- 切换动作：`tab_action == SWITCH`，`target_tab_id` 是切换目标
- 关闭动作：`tab_action == CLOSE`，`target_tab_id` 表示被关闭页；若省略则默认关闭当前 `tab_id`

### 与 iframe 的关系

页面内元素定位使用三层上下文：

1. `tab_id`
2. `frame_path`
3. `selector`

即：先选中正确 tab，再进入正确 iframe，再定位元素。

## 录制设计

### 录制目标

录制阶段需要显式拿到：

- 当前动作发生在哪个 tab
- 当前 tab 是否由某个 opener 打开
- 本步是否打开了新 tab
- 本步是否发生切换或关闭

### 建模要求

历史记录不能只保留 click/fill/navigate 动作，需要把标签页事件也纳入可序列化数据。

建议录制输出按时间顺序产生两类边：

1. `tab lifecycle edge`
2. `page interaction edge`

例如：

1. `OPEN`：从 `tab-0` 打开 `tab-1`
2. `SWITCH`：活动标签切到 `tab-1`
3. `CLICK`：在 `tab-1` 内点击元素

### `tab_id` 生成策略

建议以录制顺序稳定生成：

- 初始页固定为 `tab-0`
- 第一个 popup / 新页为 `tab-1`
- 之后按出现顺序递增

这样便于测试和序列化。

## 图构建设计

### 边的表达

图中的边应允许两类动作：

1. 页面内动作
2. 标签页生命周期动作

标签页生命周期动作允许：

- `selector=""`
- `frame_path=[]`
- `intent=None`

但必须有：

- `tab_id`
- `tab_action`
- 合法的 `target_tab_id`

### 节点与状态

现有节点主要表达页面 URL 状态，这一层可以暂不大改。

第一版建议：

- 节点继续保留现有 `state/url` 语义
- 通过边上的 `tab_id` 区分同 URL 不同标签上下文

这样能在不重写整套 pathfinding 的情况下引入多标签页能力。

## 回放设计

### 运行时上下文

回放引擎从“单个 `page`”升级为：

- `browser`
- `context`
- `pages_by_tab_id: dict[str, Any]`
- `active_tab_id: str`
- `opener_by_tab_id: dict[str, str | None]`

### 回放规则

#### 普通页面内动作

对于 `CLICK` / `FILL` / `NAVIGATE`：

1. 先根据 `edge.tab_id` 找到对应 `page`
2. 若 `tab_id` 不存在，报错
3. 在该 page 上继续执行已有 `frame_path + selector` 定位逻辑

#### `OPEN`

对于 `tab_action == OPEN`：

1. 在发起页监听 popup/new page
2. 执行触发打开新页的动作
3. 获取新 `page`
4. 将新页绑定到 `target_tab_id`
5. 记录 opener 关系

#### `SWITCH`

对于 `tab_action == SWITCH`：

1. 验证 `target_tab_id` 已存在且未关闭
2. 更新 `active_tab_id`
3. 不要求一定触发 UI 动作，可作为纯上下文切换边

#### `CLOSE`

对于 `tab_action == CLOSE`：

1. 找到目标标签页
2. 调用 `page.close()`
3. 从运行时映射移除
4. 若关闭的是当前活动页，则切回：
   - opener 页，如果仍存在
   - 否则切回任一仍存活页
   - 若无存活页则报错

## 错误语义

需要新增更明确的多标签页错误：

- `Failed to open target tab: tab-1`
- `Target tab does not exist: tab-2`
- `Target tab already closed: tab-2`
- `Cannot switch tab because no active page exists`
- `Cannot close the last remaining tab`

这些错误要与现有 iframe / selector 错误区分开。

## 测试设计

### `tests/graph_agent/test_models.py`

- `GraphEdge` 支持 `tab_id` / `target_tab_id` / `tab_action`
- `TabSnapshot` round-trip

### `tests/graph_agent/test_io.py`

- graph 保存/加载后 tab 上下文字段不丢失
- `OPEN/SWITCH/CLOSE` 边能正确 round-trip

### `tests/graph_agent/test_parser.py`

- 若录制输入包含 tab 元数据，`parse_browser_use_step()` 能生成正确 `tab_id`
- popup/new tab 元事件可映射为 `tab_action`

### `tests/graph_agent/test_playback.py`

至少覆盖：

1. 打开新标签页并继续操作
2. 从 `tab-0` 切到 `tab-1`
3. 关闭 `tab-1` 后回到 `tab-0`
4. 切换到不存在 tab 时给出明确错误
5. `tab + iframe` 叠加定位

## 非目标范围

第一版明确不做：

- 跨浏览器窗口支持
- 多个 browser context 支持
- 通过 URL/title 模糊恢复 tab
- 浏览器重启后的会话恢复

## 实施顺序

1. 先补模型与 IO 的 failing tests
2. 再补回放端 `OPEN/SWITCH/CLOSE` failing tests
3. 再补 parser/录制映射的 failing tests
4. 再串联 `tab + iframe` 的集成测试
5. 最后用真实站点验证“Quality Inspection”类新标签场景

## 成功标准

满足以下条件即可认为完成：

1. 可录制并持久化多标签页生命周期
2. 可回放打开、切换、关闭标签页
3. 可在指定 tab 内继续执行普通页面操作
4. 可与已有 iframe 支持叠加
5. 遇到 tab 缺失/关闭等问题时，错误信息足够明确
