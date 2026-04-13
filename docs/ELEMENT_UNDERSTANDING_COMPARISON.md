# 页面元素理解对比：自动测绘系统 vs page-agent

## 1. 架构定位对比

### 自动测绘系统 (Cartography)
- **架构**: 后端服务架构，基于 Python + Neo4j
- **定位**: 专注于"测绘"——生成完整的应用地图
- **持久化**: 数据持久化，支持多会话合并
- **浏览器控制**: 使用 browser-use 作为底层浏览器控制

### page-agent
- **架构**: 浏览器扩展架构，基于 TypeScript
- **定位**: 专注于"执行"——完成特定任务
- **持久化**: 状态保存在内存，每次重新索引
- **浏览器控制**: 自研 PageController 控制浏览器

---

## 2. 元素表示方式对比

### 自动测绘系统：ElementSnapshot

```python
class ElementSnapshot(BaseModel):
    index: int              # 元素索引 (来自 browser-use)
    selector: str           # CSS selector
    tag_name: str           # HTML 标签
    attributes: dict        # 所有 HTML 属性
    text_content: str       # 文本内容
    frame_path: list        # iframe 层级路径
    zone_id: str            # 所属 Zone
    bounding_box: dict      # 位置和尺寸
    is_visible: bool        # 可见性
    element_type: str       # 元素类型 (button/input/select)
```

**存储方式**:
- JSON 序列化存储在 Neo4j
- 支持跨会话查询和匹配
- 支持基于 selector 的重放

### page-agent：InteractiveElementDomNode

```typescript
interface InteractiveElementDomNode {
  index: number           // 元素索引 (动态生成)
  tagName: string         // HTML 标签
  attributes: Record      // 关键属性 (id, class, name)
  text: string            // 文本内容
  rect: DOMRect           // 位置和尺寸
  isVisible: boolean      // 可见性
  isTopLayer: boolean     // 是否在顶层
}
```

**存储方式**:
- 内存中的 Map: `selectorMap` (index -> element)
- 每次 `updateTree()` 重新生成索引
- 不持久化，页面刷新后重置

---

## 3. 元素索引机制对比

### 自动测绘系统

```
browser-use DOM 树
       ↓
  提取交互元素
       ↓
  分配 index (基于可见顺序)
       ↓
  生成 selector (CSS selector)
       ↓
  创建 ElementSnapshot
       ↓
  存储到 Neo4j (持久化)
```

**特点**:
- index 由 browser-use 分配
- 同时保留 index 和 selector 两种定位方式
- selector 用于回放时的元素定位
- 支持 iframe 内的元素 (frame_path)

### page-agent

```
当前页面 DOM
     ↓
getFlatTree() - 提取扁平树
     ↓
过滤交互元素
     ↓
高亮并分配 index (从 0 开始)
     ↓
生成 simplifiedHTML (给 LLM)
     ↓
存储在 selectorMap (内存)
```

**特点**:
- index 动态分配，每次可能不同
- 使用 simplifiedHTML 表示页面状态
- 通过 index 映射到实际 DOM 元素
- 支持 viewport 过滤 (只索引可见元素)

---

## 4. 元素交互方式对比

### 自动测绘系统

```python
# 基于 index 的交互 (探索阶段)
await controller.click_element(index=5)

# 基于 selector 的交互 (回放阶段)
await page.locator('[data-testid="submit"]').click()

# 基于 ElementSnapshot 的交互
await playback_engine.execute_action(
    edge=GraphEdge,
    element=ElementSnapshot  # 包含 selector + frame_path
)
```

**优势**:
- 回放时使用 CSS selector，更稳定
- 支持多种定位策略 (data-testid, aria-label, etc.)
- 支持 iframe 内元素操作

### page-agent

```typescript
// 基于 index 的交互
await pageController.clickElement(index=5)
await pageController.inputText(index=3, text="hello")
await pageController.selectOption(index=2, text="Option 1")

// 内部通过 index 查找元素
const element = getElementByIndex(selectorMap, index)
await clickElement(element)
```

**优势**:
- 简化 LLM 的决策空间 (只需 index)
- 动态索引，适应页面变化
- 实时更新，避免 stale element

---

## 5. 元素理解深度对比

### 自动测绘系统：多层次理解

```
┌─────────────────────────────────────────────────────────────┐
│ Zone 级别                                                    │
│ • 将页面划分为 Zone (导航/内容/表单/表格等)                     │
│ • ZoneDiscoverer 识别不同类型的内容区域                        │
├─────────────────────────────────────────────────────────────┤
│ 意图级别                                                     │
│ • 解析元素操作的意图 (auth.fill.username, search.submit)      │
│ • IntentParser 从 action 推理语义                             │
├─────────────────────────────────────────────────────────────┤
│ 图关系级别                                                   │
│ • State (节点) - 页面状态                                     │
│ • Transition (边) - 元素触发的状态转换                         │
│ • 构建完整的应用图                                             │
└─────────────────────────────────────────────────────────────┘
```

### page-agent：单层次理解

```
┌─────────────────────────────────────────────────────────────┐
│ 交互元素识别                                                  │
│ • 识别 clickable, input, select 等交互元素                    │
│ • 生成 index 映射                                             │
├─────────────────────────────────────────────────────────────┤
│ 页面状态表示                                                  │
│ • BrowserState: header + content + footer                     │
│ • simplifiedHTML 给 LLM 消费                                   │
├─────────────────────────────────────────────────────────────┤
│ 工具调用                                                     │
│ • LLM 输出 action (click_element_by_index)                    │
│ • 执行后立即观察新状态                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 优缺点对比

### 自动测绘系统

| 优点 | 缺点 |
|------|------|
| ✅ 数据持久化，支持跨会话分析 | ❌ 架构复杂，组件多 |
| ✅ 完整的应用图，支持路径规划 | ❌ 初次部署需要 Neo4j |
| ✅ 回放稳定性高 (selector 定位) | ❌ selector 可能因页面变化失效 |
| ✅ 支持覆盖率分析 | |
| ✅ 支持 Zone 级别的并行探索 | |

### page-agent

| 优点 | 缺点 |
|------|------|
| ✅ 架构轻量，易于部署 | ❌ 无数据持久化 |
| ✅ 实时索引，适应动态页面 | ❌ 每次重新索引，无法积累知识 |
| ✅ 扩展形式，用户体验好 | ❌ 不支持复杂的路径规划 |
| ✅ 开发体验好 (TypeScript) | ❌ 回放稳定性依赖 index 稳定性 |

---

## 7. 适用场景

### 自动测绘系统
- 自动化测试平台
- 回归测试
- 应用地图生成
- 覆盖率分析
- 智能探索 (95%+ 覆盖率)

### page-agent
- 个人助手扩展
- 单次任务执行
- 快速原型验证
- 实时页面操作

---

## 8. 技术融合建议

### 从 page-agent 借鉴
1. **实时 DOM 索引机制** - 避免 stale element 问题
2. **simplifiedHTML 表示** - 减少 LLM token 消耗
3. **视觉遮罩** - 防止用户干扰自动化
4. **React 补丁** - 更好地处理 React 应用

### page-agent 可以借鉴
1. **图存储架构** - 持久化状态和知识积累
2. **意图解析** - 语义化操作，提高可维护性
3. **Zone 概念** - 分区处理复杂页面
4. **覆盖率追踪** - 量化探索完整度

---

## 9. 核心差异总结

| 维度 | 自动测绘系统 | page-agent |
|------|-------------|-----------|
| **目标** | 生成完整应用地图 | 完成特定任务 |
| **持久化** | Neo4j 图数据库 | 内存 (无持久化) |
| **元素定位** | selector (稳定) | index (动态) |
| **架构** | 后端服务 | 浏览器扩展 |
| **状态管理** | State + Transition | BrowserState |
| **知识积累** | 跨会话 | 单次会话 |
| **适用** | 企业级测试 | 个人助手 |
