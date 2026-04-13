# AutoTestAgent v2 — 自动测绘与智能测试平台 PRD

## 1. 背景与目标

### 1.1 现状

当前 `graph_agent` 实现了基础的"测绘 → 图谱 → 寻径 → 回放"链路，但存在以下局限：

- **单次测绘**：无法增量补全图谱，每次测绘覆盖面有限。
- **扁平结构**：所有交互元素被拍平到同一层级，LLM 在复杂 SPA 中容易越界操作（如在填表单时乱点菜单）。
- **无 iframe 支持**：虽有 `frame_path` 字段，但缺乏系统性的多级 iframe 探索策略。
- **JSON 文件存储**：无法支撑图的关系查询、影响分析、GraphRAG 等高级能力。
- **无断言体系**：回放只验证"操作是否报错"，不验证"操作是否达成了业务目的"。
- **意图依赖缺失**：不知道"登录需要先有账号"这类因果关系。

### 1.2 目标

构建一个支持 **多次自主测绘、Scope 受限探索、智能检查点、意图依赖分析** 的自动化测试平台。核心产出物存储于 Neo4j，支持 GraphRAG 高质量召回。

### 1.3 术语统一

| 旧术语 | 新术语 | 定义 |
|--------|--------|------|
| edge | **transition** | 状态之间的操作序列，代表一次有意义的状态转移 |
| assertion / check | **checkpoint** | 操作执行后的验证检查点，验证操作是否生效或业务目标是否达成 |
| state / node | **state** | 页面状态节点，由 URL + DOM 指纹唯一标识 |
| zone | **zone** | 页面内的独立功能区域（表单、表格、操作栏等） |
| scope | **scope** | 限制 agent 操作范围的约束定义 |
| session | **session** | 一次完整的探索运行，产出增量图数据 |
| menu (概念) | **menu** | 导航菜单项节点，与 **state** 分离；菜单树用 `(:Menu)` 表达，**state** 只表示页面/界面状态 |

---

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          AutoTestAgent v2                                │
│                                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐          │
│  │ Cartography│→ │ Graph Store │→ │  Planner   │→ │ Executor │          │
│  │  Engine    │  │  (Neo4j)   │  │            │  │          │          │
│  └────────────┘  └─────┬──────┘  └────────────┘  └──────────┘          │
│       ↑                │↕              ↓              ↓                 │
│  ┌────────────┐  ┌─────┴──────┐  ┌────────────┐  ┌──────────┐          │
│  │   Scope    │  │  Coverage  │  │ Checkpoint  │  │ Reporter │          │
│  │  Manager   │  │  Analyzer  │  │  Engine     │  │          │          │
│  └────────────┘  └────────────┘  └────────────┘  └──────────┘          │
│                                                                          │
│  ┌─────────────────────┐  ┌─────────────────────────────────────┐       │
│  │ lib/PageController  │  │ retriever/ (VectorRetriever 抽象层)  │       │
│  │ (页面理解引擎)       │  │ ┌─────────┐ ┌───────┐ ┌──────────┐ │       │
│  │                     │  │ │InMemory │ │Neo4j  │ │Chroma/...│ │       │
│  │                     │  │ │Retriever│ │Vector │ │(可扩展)   │ │       │
│  │                     │  │ └─────────┘ └───────┘ └──────────┘ │       │
│  └─────────────────────┘  └─────────────────────────────────────┘       │
│                                                                          │
│  Runtime: browser-use (CDP) + LLM + EmbeddingProvider                    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Neo4j 部署

使用 Docker 部署 Neo4j Community Edition：

```yaml
# docker-compose.yml
services:
  neo4j:
    image: neo4j:5.26.24-community
    container_name: autotestagent-neo4j
    ports:
      - "7474:7474"        # HTTP (Browser)
      - "7687:7687"        # Bolt (Driver)
    environment:
      - NEO4J_AUTH=neo4j/autotestagent
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_server_memory_heap_initial__size=512m
      - NEO4J_server_memory_heap_max__size=1G
      - NEO4J_server_memory_pagecache_size=256m
    volumes:
      - ./neo4j_data/data:/data
      - ./neo4j_data/logs:/logs
      - ./neo4j_data/plugins:/plugins
    restart: unless-stopped
```

**启动 / 停止：**

```bash
docker compose up -d          # 启动
docker compose down            # 停止 (保留数据)
docker compose down -v         # 停止并清除数据
```

**连接配置（`.env`）：**

```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=autotestagent
```

**数据目录：**

```
neo4j_data/                    # ⚠️ 已加入 .gitignore, 不纳入版本管理
├── data/                      # 数据库文件
├── logs/                      # 日志
└── plugins/                   # APOC 等插件 (容器自动下载)
```

> `neo4j_data/` 是本地持久化目录，通过 volume 挂载到容器内。数据仅存在于开发者本地，不提交到 git。首次 `docker compose up` 会自动创建。

---

## 3. Neo4j Schema 设计

### 3.1 节点类型 (Node Labels)

> 共 **13 种**节点类型。**Menu** 与 **State** 分离：菜单是导航结构实体，页面状态由 **State** 表示；二者通过 `LEADS_TO` 等关系连接。Transition 被具象化（reify）为节点而非关系，因为 Neo4j 的关系不能再拥有指向其他节点的关系——而 Transition 需要挂载 Checkpoint、关联 Frame、关联 Intent 等。

```cypher
// ===== 应用 (顶层组织节点) =====
(:App {
  id: STRING,               // 唯一标识, e.g. "app:my-crm"
  name: STRING,              // 应用名称
  entry_url: STRING,         // 入口 URL
  description: STRING,       // 应用描述
  created_at: DATETIME,
  last_session_at: DATETIME, // 最近一次 session 时间
  total_sessions: INTEGER,   // 统计: 总 session 数
  total_states: INTEGER,     // 统计: 总 state 数
  total_transitions: INTEGER // 统计: 总 transition 数
})

// ===== 导航菜单项（独立节点，与 State 分离）=====
// 用于：菜单层级查询、权限/范围与菜单对齐、GraphRAG「某菜单下有哪些页面」、菜单变更检测
(:Menu {
  id: STRING,                // e.g. "menu:app1:sys:user-mgmt"
  label: STRING,             // 界面显示文本, e.g. "用户管理"
  level: INTEGER,            // 0=顶层, 1=二级, …（相对 App 内菜单树）
  order_index: INTEGER,      // 同级排序（可选，用于稳定序列化）
  selector: STRING,          // 可点击菜单项的 CSS selector（回放/识别用，可为空若仅能从文本定位）
  menu_key: STRING,          // 可选：产品侧稳定 key（与 label 解耦）
  app_id: STRING,            // 所属 App.id
  stable_path: STRING,       // 可选：规范化路径键, e.g. "system/user"（便于合并多 session）
  first_discovered: DATETIME,
  last_seen: DATETIME
})

// ===== 页面状态 =====
(:State {
  id: STRING,               // 唯一标识, e.g. "state:system-users"
  url: STRING,               // 清洗后的页面 URL
  title: STRING,             // document.title
  fingerprint: STRING,       // DOM 结构指纹，检测页面变化
  menu_path: [STRING],       // 冗余：人类可读路径 ["系统管理","用户管理"]，与 Menu 树一致时可由图遍历生成；离线/兼容场景可只写此字段
  first_discovered: DATETIME,
  last_visited: DATETIME,
  visit_count: INTEGER,
  
  // SPA 相关
  spa_route: STRING,         // SPA 路由路径 (可能与 url 相同)
  is_modal: BOOLEAN,         // 是否为弹窗产生的状态
  parent_state_id: STRING,   // 弹窗场景下的父状态
  app_id: STRING             // 所属 App 的 id (冗余索引)
})

// ===== 状态转移 (具象化节点) =====
// 原 "edge"，统一为 Transition 节点，通过 FROM/TO 关系连接两端 State
(:Transition {
  id: STRING,                // "transition:step-3"
  
  // 操作信息
  selector: STRING,          // 交互元素的 CSS selector
  action: STRING,            // click | fill | select | rich_text | navigate
  action_value: STRING,      // fill/select 时的输入值
  param_name: STRING,        // 参数化名称, e.g. "username"
  
  // 元素快照 (JSON)
  element_snapshot: STRING,  // JSON: ElementSnapshot (xpath, css_selector, attributes...)
  
  // frame 上下文 (JSON)
  frame_path: STRING,        // JSON: [FrameLocatorSnapshot], 用于回放时定位 iframe
  
  // tab 上下文
  tab_id: STRING DEFAULT "tab-0",
  target_tab_id: STRING,
  tab_action: STRING,        // open | switch | close
  
  // agent 思考
  thought: STRING,           // agent 执行时的原始思考/决策描述
  
  // 质量元数据
  confidence: FLOAT DEFAULT 0.5,
  first_discovered: DATETIME,
  last_validated: DATETIME,
  session_id: STRING,        // 冗余索引: 首次发现的 session (图关系: Session-[:DISCOVERED]->Transition)
  validation_count: INTEGER DEFAULT 0,
  
  // 步骤索引
  step_index: INTEGER        // 在 session 中的原始位置
})

// ===== 功能区域 =====
(:Zone {
  id: STRING,                // "zone:system-users:search-form-0"
  zone_type: STRING,         // search_form | data_table | detail_form |
                             // action_bar | tab_panel | tree_panel | modal
  root_selector: STRING,     // 区域根元素的 CSS selector
  summary: STRING,           // 人类可读描述, e.g. "用户名, 手机号, 状态"
  interactive_count: INTEGER,
  
  // 探索状态
  exploration_status: STRING,  // undiscovered | discovered | partial |
                               // explored | validated | stale
  last_explored: DATETIME
})

// ===== iframe 上下文 =====
(:Frame {
  id: STRING,                // "frame:main-content-iframe"
  selector: STRING,          // iframe 元素的 CSS selector
  xpath: STRING,
  name: STRING,              // iframe name 属性
  src: STRING,               // iframe src URL
  depth: INTEGER,            // 嵌套深度 (0 = 主文档)
  parent_frame_id: STRING    // 父 frame ID, null = 主文档
})

// ===== 业务实体 =====
(:Entity {
  id: STRING,                // "entity:user"
  name: STRING,              // "用户账号"
  key_fields: [STRING],      // ["username", "password"]
  description: STRING
})

// ===== 实体实例（测试数据池）=====
(:EntityInstance {
  id: STRING,                // "instance:user:admin"
  data: STRING,              // JSON: {"username": "admin", "password": "Admin123"}
  status: STRING,            // available | consumed | expired
  session_id: STRING,        // 冗余索引: 创建此实例的 session (图关系: Session-[:CREATED]->EntityInstance)
  created_at: DATETIME
})

// ===== 业务意图 =====
(:Intent {
  id: STRING,                // "intent:login"
  name: STRING,              // "用户登录"
  raw: STRING,               // agent 原始思考文本
  summary: STRING,           // "使用用户名密码完成系统登录"
  verb: STRING,              // "Login"
  object: STRING,            // "System"
  key: STRING,               // "user.login"
  confidence: FLOAT
})

// ===== 检查点 =====
(:Checkpoint {
  id: STRING,
  
  // 检查点类型
  layer: STRING,             // structural | data | behavioral | semantic |
                             // entity | lifecycle
  timing: STRING,            // before | immediate | after_blur | after_submit | after
  
  // 测试期望
  expect: STRING,            // should_pass | should_fail
  severity: STRING,          // critical | major | minor | info
  
  // 规则定义 (JSON, 按 layer 不同结构不同)
  rule_type: STRING,         // url_match | element_exists | validation_error |
                             // field_value | toast | entity_exists |
                             // entity_created | page_matches_intent | ...
  rule: STRING,              // JSON: 具体的规则参数
  description: STRING,       // 人类可读描述
  
  // 来源
  origin_type: STRING,       // cartography | strategy | manual | inferred
  session_id: STRING,        // 冗余索引: 生成此检查点的 session (图关系: Session-[:GENERATED]->Checkpoint)
  
  // 执行统计
  total_runs: INTEGER DEFAULT 0,
  pass_count: INTEGER DEFAULT 0,
  fail_count: INTEGER DEFAULT 0,
  flaky: BOOLEAN DEFAULT false,
  last_result: STRING        // pass | fail | skip | null
})

// ===== 字段约束 =====
(:FieldConstraint {
  id: STRING,
  selector: STRING,          // 字段的 CSS selector
  field_name: STRING,        // 字段名 (来自 label)
  input_type: STRING,        // text | number | email | password | date | select
  
  // HTML 约束
  required: BOOLEAN,
  min_length: INTEGER,
  max_length: INTEGER,
  min_value: FLOAT,
  max_value: FLOAT,
  pattern: STRING,
  step: FLOAT,
  
  // 学习到的约束 (JSON array)
  learned_constraints: STRING,  // JSON: [{error_message, trigger_value, constraint_type}]
  valid_examples: [STRING]
})

// ===== 派生测试用例 =====
(:TestCase {
  id: STRING,
  name: STRING,              // "用户名-超最大长度"
  category: STRING,          // boundary | equivalence | negative | format |
                             // required | positive | chaos
  description: STRING,
  
  // 值覆盖 (JSON)
  field_overrides: STRING,   // JSON: {"#username": "abc123456789"}
  
  // 执行统计
  total_runs: INTEGER DEFAULT 0,
  pass_count: INTEGER DEFAULT 0,
  fail_count: INTEGER DEFAULT 0,
  last_run: DATETIME,
  last_result: STRING
})

// ===== 探索会话 =====
(:Session {
  id: STRING,                // "session:2026-03-26T10:30:00Z"
  app_id: STRING,            // 所属 App 的 id
  timestamp: DATETIME,
  duration_ms: INTEGER,
  focus: STRING,             // breadth | depth | validation
  
  // 成果统计
  states_discovered: INTEGER,
  states_updated: INTEGER,
  transitions_discovered: INTEGER,
  transitions_validated: INTEGER,
  transitions_invalidated: INTEGER,
  checkpoints_generated: INTEGER
})
```

### 3.2 关系类型 (Relationship Types)

> Transition 已具象化为节点（见 3.1），因此状态转移通过 `FROM/TO` 关系连接。
> 这使得 Transition 可以自由挂载 Checkpoint、关联 Frame、关联 Intent 等。

```
状态转移的基本结构:

  (:State)<-[:FROM]-(:Transition)-[:TO]->(:State)

读法: "Transition t 从 State A 出发, 到达 State B"
```

```cypher
// ===== App 包含关系 =====
(:App)-[:HAS_STATE]->(:State)     // App 拥有的所有页面状态
(:App)-[:HAS_SESSION]->(:Session) // App 下的所有探索会话
(:App)-[:HAS_MENU]->(:Menu)      // App 下的菜单项（整棵树上的节点均可挂此关系，或仅用 Menu.app_id 属性查询）

// ===== 菜单树结构 =====
// 子节点指向父节点，根节点的 parent 为空（不连 CHILD_OF 或连向虚拟根）
(:Menu)-[:CHILD_OF {
  order_index: INTEGER         // 可选：在父下的顺序
}]->(:Menu)

// 点击某菜单项后到达的页面状态（常见 1:1；SPA 同项多次可达不同 State 时可多条）
(:Menu)-[:LEADS_TO {
  first_seen: DATETIME,        // 可选
  session_id: STRING           // 首次确认该关系的 session
}]->(:State)

// 某次转移由点击菜单项触发（已知 selector/文本与 Menu 对齐时写入）
(:Transition)-[:NAVIGATED_VIA]->(:Menu)

// State 上仍可有 menu_path[] 冗余；规范路径也可由 (State)<-[:LEADS_TO]-(Menu) 沿 CHILD_OF 回溯拼出

// ===== 状态转移 (核心) =====
// Transition 是节点, 通过 FROM/TO 连接两端 State
(:Transition)-[:FROM]->(:State)   // 转移的起点状态
(:Transition)-[:TO]->(:State)     // 转移的终点状态

// ===== 页面包含区域 =====
(:State)-[:HAS_ZONE]->(:Zone)

// ===== Transition 所属区域 =====
// 标记此 Transition 发生在页面的哪个功能区域内
// 用于 Zone 级覆盖率统计和 Scope 精细化控制
(:Transition)-[:IN_ZONE]->(:Zone)

// ===== 页面包含 iframe =====
(:State)-[:HAS_FRAME {
  depth: INTEGER             // 嵌套深度
}]->(:Frame)

// ===== iframe 嵌套关系 =====
(:Frame)-[:CONTAINS_FRAME]->(:Frame)

// ===== Transition 发生在某个 iframe 内 =====
(:Transition)-[:IN_FRAME]->(:Frame)

// ===== Transition 关联意图 =====
// 多个 Transition 可以共享同一个 Intent (如多步登录共享 "用户登录" 意图)
(:Transition)-[:REALIZES]->(:Intent)

// ===== Transition 操作受约束字段 =====
// fill/select 操作的目标字段可能有约束定义
// 用于回放前校验输入值、派生边界测试
(:Transition)-[:OPERATES_ON]->(:FieldConstraint)

// ===== 意图的前置/后置检查点 =====
// REQUIRES_ENTITY 和 PRODUCES_ENTITY 统一为 entity 层的 Checkpoint
(:Intent)-[:HAS_PRECONDITION]->(:Checkpoint {
  layer: "entity",
  rule_type: "entity_exists"   // 需要某实体存在
})

(:Intent)-[:HAS_POSTCONDITION]->(:Checkpoint {
  layer: "entity",
  rule_type: "entity_created"  // 会创建某实体
})

// ===== 检查点挂载 =====

// Transition 级别的检查点（操作前/后）
(:Transition)-[:CHECK_BEFORE]->(:Checkpoint)
(:Transition)-[:CHECK_AFTER]->(:Checkpoint)

// Action 级别的检查点（单步操作验证）
// 通过 checkpoint 的 timing 字段区分: immediate | after_blur | after_submit
// 挂载在 Transition 上, timing 决定检查时机

// State 级别的检查点（页面固有属性，如无障碍）
(:State)-[:CHECK_STATE]->(:Checkpoint)

// ===== Checkpoint 验证的实体 =====
// entity 层 Checkpoint 引用的目标 Entity (显式图关系, 不依赖 rule JSON 解析)
(:Checkpoint)-[:CHECKS_ENTITY]->(:Entity)

// ===== 实体关系 =====

// 意图消费实体（前置条件）
(:Intent)-[:REQUIRES_ENTITY {
  field_mapping: STRING      // JSON: {"username": "username", "password": "password"}
}]->(:Entity)

// 意图产出实体（后置效果）
(:Intent)-[:PRODUCES_ENTITY {
  field_mapping: STRING
}]->(:Entity)

// 实体实例归属
(:Entity)-[:HAS_INSTANCE]->(:EntityInstance)

// 实体实例由某意图创建
(:EntityInstance)-[:CREATED_BY_INTENT]->(:Intent)

// ===== 意图间依赖 =====

// 意图 A 的输出是意图 B 的输入
(:Intent)-[:DATA_FLOW {
  from_slot: STRING,         // 输出槽名
  to_slot: STRING            // 输入槽名
}]->(:Intent)

// 意图 A 和 B 是达成同一目标的替代方案
(:Intent)-[:ALTERNATIVE_OF]->(:Intent)

// 意图 A 必须在 B 之前执行 (非数据依赖的顺序约束)
(:Intent)-[:MUST_PRECEDE]->(:Intent)

// ===== 字段约束关联 =====
(:Zone)-[:HAS_FIELD]->(:FieldConstraint)

// ===== 测试用例关联 =====

// 测试用例基于某个 Transition 派生
(:TestCase)-[:DERIVED_FROM]->(:Transition)

// 测试用例包含的检查点
(:TestCase)-[:EXPECTS]->(:Checkpoint)

// 测试用例覆盖某个字段约束
(:TestCase)-[:COVERS]->(:FieldConstraint)

// ===== 探索会话关联 =====
// 关系用于图遍历; 节点属性上保留 session_id 做冗余索引
(:Session)-[:DISCOVERED]->(:State)
(:Session)-[:DISCOVERED]->(:Transition)
(:Session)-[:GENERATED]->(:Checkpoint)
(:Session)-[:VALIDATED]->(:Transition)
(:Session)-[:INVALIDATED]->(:Transition)
(:Session)-[:CREATED]->(:EntityInstance)  // 替代 EntityInstance.created_by_session 属性
(:Session)-[:DISCOVERED]->(:Menu)         // 本次 session 发现或更新的菜单节点
```

### 3.3 多级 iframe 建模示例

```cypher
// 主页面
CREATE (s1:State {id: "state:main-page", url: "http://app/dashboard"})
CREATE (s2:State {id: "state:saved", url: "http://app/dashboard"})

// 一级 iframe
CREATE (f1:Frame {id: "frame:content-iframe", selector: "#content-frame",
                  depth: 1, src: "http://app/content"})
CREATE (s1)-[:HAS_FRAME {depth: 1}]->(f1)

// 二级 iframe (嵌套在一级 iframe 中)
CREATE (f2:Frame {id: "frame:editor-iframe", selector: "#editor-frame",
                  depth: 2, src: "http://app/editor",
                  parent_frame_id: "frame:content-iframe"})
CREATE (f1)-[:CONTAINS_FRAME]->(f2)

// 在二级 iframe 中点击保存按钮: Transition 是节点
CREATE (t:Transition {
  id: "transition:save-click",
  selector: "#save-btn",
  action: "click",
  frame_path: '[{"selector":"#content-frame"},{"selector":"#editor-frame"}]'
})
CREATE (t)-[:FROM]->(s1)
CREATE (t)-[:TO]->(s2)

// 标记此 transition 发生在二级 iframe 中
CREATE (t)-[:IN_FRAME]->(f2)

// 为此 transition 挂载检查点
CREATE (cp:Checkpoint {
  id: "cp:save-success",
  layer: "behavioral",
  timing: "after",
  expect: "should_pass",
  rule_type: "toast",
  rule: '{"message_contains": "保存成功"}',
  description: "点击保存后应出现成功提示"
})
CREATE (t)-[:CHECK_AFTER]->(cp)
```

### 3.4 SPA 状态识别策略

```cypher
// SPA 中 URL 可能不变但 DOM 变化的情况
// 用 fingerprint 区分不同状态

// 同一 URL 下的不同 tab 切换
CREATE (s1:State {
  id: "state:user-detail-info",
  url: "http://app/user/1",
  fingerprint: "fp-info-tab",
  title: "用户详情 - 基本信息"
})

CREATE (s2:State {
  id: "state:user-detail-perm",
  url: "http://app/user/1",
  fingerprint: "fp-perm-tab",
  title: "用户详情 - 权限配置"
})

// 同一 URL, 不同状态 — Transition 是节点
CREATE (t:Transition {
  id: "transition:switch-to-perm-tab",
  selector: ".tab-permissions",
  action: "click"
})
CREATE (t)-[:FROM]->(s1)
CREATE (t)-[:TO]->(s2)

// 关联意图
CREATE (i:Intent {
  id: "intent:view-permissions",
  name: "查看权限配置",
  key: "user.view_permissions",
  verb: "View", object: "Permissions",
  summary: "切换到权限配置标签页"
})
CREATE (t)-[:REALIZES]->(i)

// 弹窗产生的临时状态
CREATE (modal_state:State {
  id: "state:create-user-modal",
  url: "http://app/users",
  fingerprint: "fp-create-modal",
  is_modal: true,
  parent_state_id: "state:user-list"
})
```

### 3.5 Menu 树与导航建模

菜单与页面状态分离：**Menu** 表达产品导航结构；**State** 表达 URL + 指纹下的界面状态。同一 State 可被多个 Menu 指向（例如侧栏与快捷入口），也可不经菜单直达（深链），此时仅有 State 无 `LEADS_TO` 入边。

```cypher
// App 与顶层菜单
CREATE (app:App {id: "app:crm", name: "CRM", entry_url: "http://app/"})
CREATE (m_sys:Menu {
  id: "menu:crm:system",
  label: "系统管理",
  level: 0,
  order_index: 0,
  selector: "nav .menu-system",
  app_id: "app:crm"
})
CREATE (m_user:Menu {
  id: "menu:crm:system:users",
  label: "用户管理",
  level: 1,
  order_index: 0,
  selector: "nav .submenu-users",
  app_id: "app:crm"
})
CREATE (app)-[:HAS_MENU]->(m_sys)
CREATE (app)-[:HAS_MENU]->(m_user)
CREATE (m_user)-[:CHILD_OF]->(m_sys)

// 点击「用户管理」后进入的列表页 State
CREATE (s_list:State {
  id: "state:users-list",
  url: "http://app/users",
  title: "用户列表",
  fingerprint: "fp-users-list",
  menu_path: ["系统管理", "用户管理"],
  app_id: "app:crm"
})
CREATE (m_user)-[:LEADS_TO]->(s_list)
CREATE (app)-[:HAS_STATE]->(s_list)

// 经菜单点击进入的 Transition 可挂 Menu（便于回放与归因）
CREATE (s_dash:State {
  id: "state:dashboard",
  url: "http://app/dashboard",
  title: "首页",
  fingerprint: "fp-dash",
  app_id: "app:crm"
})
CREATE (app)-[:HAS_STATE]->(s_dash)
CREATE (t:Transition {
  id: "transition:open-users-from-menu",
  selector: "nav .submenu-users",
  action: "click"
})
CREATE (t)-[:FROM]->(s_dash)
CREATE (t)-[:TO]->(s_list)
CREATE (t)-[:NAVIGATED_VIA]->(m_user)
```

**与旧方案的区别**：不再使用 `(:State)-[:MENU_PARENT|MENU_NEXT]->(:State)` 表达菜单层级（会把导航壳与真实页面状态混在同一类型上）。菜单层级仅由 **Menu** + `CHILD_OF` 表达。

---

## 4. Scope 机制

### 4.1 Scope 定义

```python
class ExplorationScope(BaseModel):
    """限制 agent 操作范围的约束"""
    id: str
    name: str
    
    # 允许交互的区域 (CSS selectors)
    include_selectors: list[str] = []
    # 禁止交互的区域 (CSS selectors)
    exclude_selectors: list[str] = []
    # 是否允许导致 URL 变化
    allow_navigation: bool = False
    # 是否允许 iframe 内操作
    allow_iframe: bool = True
    # 允许的最大 iframe 嵌套深度
    max_iframe_depth: int = 3


# 预定义 scope 模板
SCOPE_MENU_DISCOVERY = ExplorationScope(
    id="scope:menu-discovery",
    name="菜单发现",
    include_selectors=["nav", ".sidebar", ".menu", "[role='navigation']", ".ant-menu"],
    exclude_selectors=["main", ".content", "form", ".modal"],
    allow_navigation=True,
    allow_iframe=False,
)

SCOPE_PAGE_EXPLORATION = ExplorationScope(
    id="scope:page-exploration",
    name="页面内探索",
    include_selectors=["main", ".content", "[role='main']", ".ant-layout-content"],
    exclude_selectors=["nav", ".sidebar", ".menu", "[role='navigation']"],
    allow_navigation=False,
    allow_iframe=True,
)
```

### 4.2 Scope 应用方式

利用 page-agent 的 `data-page-agent-not-interactive` 属性机制，在 DOM 提取前标记排除元素。**不修改元素样式，对页面渲染零影响。**

```python
async def apply_scope(page: Page, scope: ExplorationScope) -> None:
    """在 Playwright page 上应用 scope 过滤"""
    await page.evaluate("""(scope) => {
        // 清除上一轮标记
        document.querySelectorAll('[data-scope-excluded]').forEach(el => {
            el.removeAttribute('data-scope-excluded')
        })
        
        // 标记排除区域内的交互元素
        const interactiveSelector = 'a, button, input, select, textarea, ' +
            '[role="button"], [role="tab"], [tabindex], [onclick]'
        
        for (const sel of scope.exclude_selectors) {
            document.querySelectorAll(sel).forEach(container => {
                container.querySelectorAll(interactiveSelector).forEach(el => {
                    el.setAttribute('data-scope-excluded', '')
                })
            })
        }
        
        // 特殊: 弹窗中的元素始终可交互
        document.querySelectorAll(
            '.ant-modal, .el-dialog, [role="dialog"]'
        ).forEach(modal => {
            modal.querySelectorAll('[data-scope-excluded]').forEach(el => {
                el.removeAttribute('data-scope-excluded')
            })
        })
    }""", {"exclude_selectors": scope.exclude_selectors})
```

---

## 5. 多次自主测绘

### 5.1 探索调度器

```python
class ExplorationScheduler:
    """分析图的覆盖率, 生成下次探索的优先级任务列表"""
    
    async def schedule(
        self,
        neo4j_driver: AsyncDriver,
        focus: Literal["breadth", "depth", "validation"] = "breadth",
        max_tasks: int = 20,
    ) -> list[ScheduledTask]:
        """
        优先级策略:
        P100: 未发现的叶子菜单页面
        P80:  已发现页面中未探索的 Zone
        P60:  部分完成的 Zone (有未尝试的交互)
        P50:  页面指纹变化, 需要重新扫描 Zone
        P40:  失败的任务重试
        P20:  低置信度或过期的 Transition 需要验证
        """
```

### 5.2 图合并策略

```python
class GraphMerger:
    """多次探索结果的合并引擎"""
    
    async def merge(
        self,
        neo4j_driver: AsyncDriver,
        new_findings: CartographyResult,
        session_id: str,
    ) -> MergeReport:
        """
        合并规则:
        - 新 State (url+fingerprint 不存在)     → 直接创建
        - 已有 State (fingerprint 变化)         → 更新, 相关 Transition 降低置信度
        - 新 Transition (source+target+intent)  → 创建, confidence=0.5
        - 已有 Transition (action_sequence 一致) → confidence += 0.2
        - 已有 Transition (action_sequence 冲突) → 保留更短路径
        - Transition 验证失败                    → confidence -= 0.3, 不删除
        """
```

### 5.3 覆盖率模型

```python
class CoverageReport(BaseModel):
    menu_coverage: float         # 已探索叶子页面 / 总叶子页面
    zone_coverage: float         # 已探索 zone / 总发现 zone
    interaction_coverage: float  # 已尝试交互 / 总可交互任务
    transition_confidence: TransitionConfidenceDistribution
    overall_completeness: float  # 加权综合分
    recommendation: Literal["complete", "needs_more", "needs_validation"]
```

---

## 6. Checkpoint 体系

### 6.1 统一的 Checkpoint 分层

所有验证检查统一为 Checkpoint，按 `layer` 区分类型：

| Layer | 确定性 | 说明 | 示例 |
|-------|--------|------|------|
| `structural` | 100% | DOM 结构/URL | url_match, element_exists, title_match |
| `data` | 95% | 元素值/文本 | field_value, element_text, input_value |
| `behavioral` | 85% | 需要交互验证 | toast_appears, modal_state, form_submittable |
| `entity` | 90% | 实体存在性 | entity_exists, entity_created, entity_modified |
| `lifecycle` | 90% | 状态生命周期 | state_entered, state_exited, permission_check |
| `semantic` | 70% | 需 LLM 判断 | page_matches_intent, no_error_state |

### 6.2 Entity Checkpoint（REQUIRES / PRODUCES 统一为 Checkpoint）

```python
# REQUIRES_ENTITY 表达为 Checkpoint
Checkpoint(
    layer="entity",
    timing="before",
    expect="should_pass",
    rule_type="entity_exists",
    rule=json.dumps({
        "entity_id": "entity:user",
        "match_fields": {"username": "{{username}}"},
    }),
    severity="critical",
    description="登录前用户账号必须存在",
)

# PRODUCES_ENTITY 表达为 Checkpoint
Checkpoint(
    layer="entity",
    timing="after",
    expect="should_pass",
    rule_type="entity_created",
    rule=json.dumps({
        "entity_id": "entity:user",
        "created_fields": ["username", "password"],
    }),
    severity="critical",
    description="注册成功后应创建新用户实体",
)
```

### 6.3 Checkpoint 挂载层级

```
一条 Transition 的检查点执行顺序:

  ┌─ CHECK_BEFORE ───────────────────────────────┐
  │  entity checkpoint: 用户账号存在?              │
  │  structural checkpoint: 在正确的页面?          │
  └──────────────────────────────────────────────┘
         │
         ▼
  ┌─ Action[0]: fill #username "liaoling" ───────┐
  │  timing=immediate: field_value 匹配?          │
  └──────────────────────────────────────────────┘
         │
         ▼
  ┌─ Action[1]: fill #password "Aa111111" ───────┐
  │  timing=immediate: field_value 非空?           │
  └──────────────────────────────────────────────┘
         │
         ▼
  ┌─ Action[2]: click #sbbtn ────────────────────┐
  │  timing=immediate: no_page_error?             │
  │  timing=immediate: loading_completed?          │
  └──────────────────────────────────────────────┘
         │
         ▼
  ┌─ CHECK_AFTER ────────────────────────────────┐
  │  structural: URL 跳转到首页?                   │
  │  behavioral: 无错误 toast?                     │
  │  lifecycle:  state_entered("logged_in")?       │
  │  semantic:   页面看起来像主控制台?              │
  └──────────────────────────────────────────────┘
```

### 6.4 反例测试的 Checkpoint（expect=should_fail）

```python
# 正例: 合法输入应该通过
Checkpoint(expect="should_pass", rule_type="validation_error",
           rule='{"selector":"#username","visible":false}',
           description="合法用户名不应触发校验错误")

# 反例: 非法输入应该被拦截
Checkpoint(expect="should_fail", rule_type="validation_error",
           rule='{"selector":"#username","visible":true,"message_contains":"10"}',
           description="超长用户名应触发最大长度校验错误")

# 执行逻辑:
# expect=should_pass + 检查通过 → checkpoint PASS
# expect=should_pass + 检查失败 → checkpoint FAIL (功能坏了)
# expect=should_fail + 检查通过 → checkpoint PASS (正确拦截了非法输入)
# expect=should_fail + 检查失败 → checkpoint FAIL (非法输入被放行了!)
```

---

## 7. iframe 探索策略

> **实现状态**：iframe 发现和处理完全由 browser-use 的 `DomService` 原生支持，无需自建 JS 注入或 `FrameManager`。

### 7.1 browser-use 原生 iframe 支持

browser-use 的 `DomService.get_dom_tree()` 通过 CDP 协议自动处理多层级 iframe：

- **同源 iframe**：通过 `contentDocument` 递归构建 DOM 子树，自动合并到主文档的 DOM 树中
- **跨域 iframe**：通过 CDP `Target` 获取独立 session，递归调用 `get_dom_tree()`
- **嵌套深度限制**：`max_iframe_depth=5`（默认），防止无限递归
- **最大 iframe 数**：`max_iframes=100`（默认）
- **坐标系转换**：自动累计 `total_frame_offset`，保证元素坐标正确
- **iframe 内滚动修正**：自动修正 `scrollRects` 偏移
- **可见性判定**：结合所有父级 iframe 判断元素是否可见
- **尺寸过滤**：跳过小于 50×50px 的 iframe（避免隐藏 iframe）

### 7.2 跨 frame 操作路由

`PageController` 在调用 `DomService.get_serialized_dom_tree()` 时，iframe 内的元素自动编入 `selector_map`。每个元素携带 `session_id` 和 `backend_node_id`，确保 browser-use 通过正确的 CDP session 执行操作：

```python
# PageController 内部 — 透明跨 iframe 操作
element = Element(self._session, node.backend_node_id, node.session_id)
await element.click()  # 自动路由到正确的 iframe CDP session
```

### 7.3 跨域 iframe 配置

`DomService` 默认关闭跨域 iframe 处理（`cross_origin_iframes=False`）。如需开启：

```python
dom_service = DomService(browser_session, cross_origin_iframes=True)
```

---

## 8. 完整探索流程

```python
async def run_cartography_session(
    start_url: str,
    neo4j_driver: AsyncDriver,
    focus: str = "breadth",
    time_budget_ms: int = 600_000,  # 10 分钟
    app_id: str | None = None,      # 关联的 App 节点 id
    app_name: str = "",              # App 名称（首次自动创建）
) -> SessionReport:
    """一次完整的测绘 session"""
    
    # 0. 确保 App 节点存在（首次自动创建）
    if app_id:
        await ensure_app_node(neo4j_driver, app_id, app_name, start_url)
    
    session = Session(id=f"session:{datetime.utcnow().isoformat()}", app_id=app_id)
    
    # 1. 加载现有图状态
    coverage = await CoverageAnalyzer(neo4j_driver).compute()
    
    # 2. 首次运行: 程序化提取菜单树 (不用 LLM)
    if coverage.menu_coverage == 0:
        menu_tree = await extract_menu_tree(page)
        await store_menu_tree(neo4j_driver, menu_tree)
    
    # 3. 生成任务列表
    tasks = await ExplorationScheduler().schedule(neo4j_driver, focus=focus)
    
    # 4. 逐任务执行
    for task in tasks:
        if time_exceeded(time_budget_ms): break
        
        match task.type:
            case "discover_page":
                findings = await discover_page(task.menu_path, page, scope=SCOPE_PAGE_EXPLORATION)
                await GraphMerger().merge(neo4j_driver, findings, session.id, app_id=app_id)
                
            case "explore_zone":
                findings = await explore_zone(task.zone, page, scope=zone_scope(task.zone))
                await GraphMerger().merge(neo4j_driver, findings, session.id, app_id=app_id)
                # 自动生成 checkpoint (DOM diff)
                checkpoints = generate_checkpoints_from_diff(before_snapshot, after_snapshot)
                await store_checkpoints(neo4j_driver, checkpoints)
                
            case "validate_transition":
                result = await validate_transition(task.transition, page)
                await update_transition_confidence(neo4j_driver, task.transition.id, result)
                
            case "probe_constraints":
                constraints = await probe_field_constraints(task.zone, page)
                test_cases = derive_boundary_tests(constraints)
                await store_test_cases(neo4j_driver, test_cases)
        
        # 每个任务后保存进度
        await session.save_progress(neo4j_driver)
    
    # 5. 更新覆盖率
    final_coverage = await CoverageAnalyzer(neo4j_driver).compute()
    session.complete(final_coverage)
    await session.save(neo4j_driver)
    
    return session.report()
```

---

## 9. GraphRAG 查询能力

存储在 Neo4j 中的数据支持以下查询模式：

```cypher
-- ========================================
-- 图遍历说明:
-- Transition 是节点, FROM/TO 都从 Transition 指向 State
-- 前进方向: State <-[:FROM]- Transition -[:TO]-> State
-- 变长路径必须用无方向遍历: -[:FROM|TO*]-
-- ========================================

-- "怎么登录系统"
MATCH (s1:State)<-[:FROM]-(t:Transition)-[:TO]->(s2:State),
      (t)-[:REALIZES]->(i:Intent)
WHERE i.key CONTAINS 'login'
RETURN s1, t, s2, i

-- "登录需要什么前提条件"
MATCH (i:Intent {key: "user.login"})-[:HAS_PRECONDITION]->(cp:Checkpoint)
RETURN cp.description, cp.rule_type

-- "用户账号有哪些创建方式"
MATCH (i:Intent)-[:PRODUCES_ENTITY]->(e:Entity {name: "用户账号"})
RETURN i.name, i.summary

-- "从登录到创建订单的完整路径"
-- 关键: 无方向变长路径, 因为 FROM/TO 都从 Transition 指向 State
-- 路径交替经过: State - Transition - State - Transition - State
MATCH path = shortestPath(
  (s1:State {url: "/login"})-[:FROM|TO*]-(s2:State {url: "/order/create"})
)
RETURN path

-- "系统管理下面有哪些子菜单"
MATCH (parent:Menu {id: "menu:crm:system"})<-[:CHILD_OF]-(child:Menu)
RETURN child.label, child.selector, child.id
ORDER BY child.order_index

-- "用户管理菜单对应哪个页面状态"
MATCH (m:Menu {id: "menu:crm:system:users"})-[:LEADS_TO]->(s:State)
RETURN m.label, s.url, s.title, s.fingerprint

-- "从 App 经菜单可达的所有 State（一层 LEADS_TO；多层需遍历 Menu 树后再 LEADS_TO）"
MATCH (app:App {id: "app:crm"})-[:HAS_MENU]->(m:Menu)
MATCH (m)-[:LEADS_TO]->(s:State)
RETURN DISTINCT m.label, s.url, s.title

-- "哪些页面的测试覆盖率最低"
MATCH (s:State)-[:HAS_ZONE]->(z:Zone)
WHERE z.exploration_status IN ["undiscovered", "discovered"]
RETURN s.url, s.title, count(z) AS unexplored_zones
ORDER BY unexplored_zones DESC

-- "哪些 transition 置信度低于阈值"
MATCH (s1:State)<-[:FROM]-(t:Transition)-[:TO]->(s2:State)
WHERE t.confidence < 0.4
RETURN s1.url, t.action, t.selector, t.confidence
ORDER BY t.confidence ASC

-- "某个 transition 的所有检查点"
MATCH (t:Transition {id: "transition:step-3"})-[r:CHECK_BEFORE|CHECK_AFTER]->(cp:Checkpoint)
RETURN type(r) AS timing, cp.layer, cp.rule_type, cp.description, cp.expect

-- "某个 transition 发生在哪个 iframe 中"
MATCH (t:Transition {id: "transition:step-5"})-[:IN_FRAME]->(f:Frame)
RETURN f.selector, f.depth, f.src

-- "某个 Zone 内的所有 Transition 及其意图"
MATCH (z:Zone {id: "zone:login-form"})<-[:IN_ZONE]-(t:Transition)
OPTIONAL MATCH (t)-[:REALIZES]->(i:Intent)
RETURN t.action, t.selector, i.summary

-- "哪些字段约束尚未被测试用例覆盖"
MATCH (z:Zone)-[:HAS_FIELD]->(fc:FieldConstraint)
WHERE NOT exists { (tc:TestCase)-[:COVERS]->(fc) }
RETURN z.id, fc.field_name, fc.input_type, fc.required

-- "某个实体的全生命周期: 谁创建, 谁消费, 检查点验证情况"
MATCH (e:Entity {name: "用户账号"})
OPTIONAL MATCH (creator:Intent)-[:PRODUCES_ENTITY]->(e)
OPTIONAL MATCH (consumer:Intent)-[:REQUIRES_ENTITY]->(e)
OPTIONAL MATCH (cp:Checkpoint)-[:CHECKS_ENTITY]->(e)
RETURN e.name,
       collect(DISTINCT creator.name) AS created_by,
       collect(DISTINCT consumer.name) AS consumed_by,
       collect(DISTINCT cp.description) AS verified_by
```

---

## 10. VectorRetriever 抽象层

### 10.1 设计动机

当前 `CognitiveProcessor` 直接调用 embedding API 并在内存中做余弦相似度计算。这种方式存在明显局限：

1. **不可持久化**：每次启动都要重新计算所有 embedding，无法跨 session 复用。
2. **不可扩展**：内存中全量比对，节点数超过万级后搜索耗时线性增长。
3. **无法利用 ANN 索引**：无法使用 HNSW、IVF 等近似最近邻算法加速检索。
4. **存储绑定**：向量和图数据耦合在同一存储引擎中，无法独立扩缩容或选型。
5. **无 hybrid 查询**：不支持 graph traversal + vector similarity 的混合检索（GraphRAG 的核心能力）。

VectorRetriever 提供**存储无关的向量检索抽象**，允许后期按需接入不同的 vector DB（Neo4j Vector Index、Chroma、Qdrant、Milvus、Pinecone 等），同时解耦 embedding 生成（EmbeddingProvider）与向量存储/检索（VectorRetriever）。

### 10.2 抽象协议

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ===== Embedding 生成 =====

class EmbeddingProvider(ABC):
    """将文本转换为向量。解耦 embedding 模型与向量存储。"""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量文本 → 向量。"""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """返回向量维度 (e.g. 1536 for text-embedding-3-small)。"""
        ...


# ===== 检索结果 =====

@dataclass
class VectorResult:
    """单条向量检索结果。"""
    id: str                                  # 对应 Neo4j 节点的 id 字段
    score: float                             # 相似度得分 (0~1, 1=完全相同)
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: list[float] | None = None        # 可选返回原始向量


@dataclass
class VectorSearchRequest:
    """检索请求。"""
    collection: str                          # 集合名 (对应 node label)
    query_vector: list[float]                # 查询向量
    top_k: int = 10                          # 返回条数
    min_score: float = 0.0                   # 最低相似度阈值
    filter: dict[str, Any] | None = None     # 元数据过滤条件
    include_vectors: bool = False            # 是否返回原始向量


# ===== 向量检索器 =====

class VectorRetriever(ABC):
    """向量检索抽象层。后端可以是 Neo4j Vector Index、Chroma、Qdrant 等。"""

    @abstractmethod
    async def upsert(
        self,
        collection: str,
        entries: list[VectorEntry],
    ) -> int:
        """写入或更新向量。返回受影响条数。
        
        collection 对应一个向量集合 (可映射为 vector DB 的 collection/index/namespace)。
        如果 id 已存在则覆盖。
        """
        ...

    @abstractmethod
    async def search(
        self,
        request: VectorSearchRequest,
    ) -> list[VectorResult]:
        """向量相似度检索。返回按 score 降序排列的结果。"""
        ...

    @abstractmethod
    async def delete(
        self, collection: str, ids: list[str]
    ) -> int:
        """按 id 删除。返回实际删除条数。"""
        ...

    @abstractmethod
    async def count(self, collection: str) -> int:
        """返回集合中的向量总数。"""
        ...

    async def ensure_collection(
        self, collection: str, dimension: int, **kwargs
    ) -> None:
        """确保集合/索引存在。幂等操作。
        
        默认空实现, 不需要预创建集合的后端可不覆写。
        """
        pass

    async def drop_collection(self, collection: str) -> None:
        """删除整个集合。默认逐条删除, 后端可覆写为原子操作。"""
        ...


@dataclass
class VectorEntry:
    """待写入的向量条目。"""
    id: str                                  # 唯一标识, 与 Neo4j 节点 id 对齐
    vector: list[float]                      # 向量
    metadata: dict[str, Any] = field(default_factory=dict)  # 元数据 (用于过滤)
```

### 10.3 向量化集合 (Collections)

系统中需要向量化的数据按 Neo4j 节点类型组织为以下集合：

| Collection | 源节点 | 向量化内容 | 用途 |
|-----------|--------|-----------|------|
| `state_fingerprints` | `State` | DOM 简化 HTML 的 embedding | 状态去重、相似页面发现 |
| `menu_embeddings` | `Menu` | `label + stable_path + menu_key` 的 embedding | 菜单项语义匹配、跨 session 合并、与意图/State 联合召回 |
| `intent_embeddings` | `Intent` | `name + summary` 的 embedding | 意图匹配、意图合并、依赖发现 |
| `zone_embeddings` | `Zone` | `zone_type + summary` 的 embedding | 相似区域发现、探索模板匹配 |
| `transition_embeddings` | `Transition` | `thought + action + selector` 的 embedding | 相似操作检索、操作去重 |
| `checkpoint_embeddings` | `Checkpoint` | `description + rule_type` 的 embedding | 检查点模板复用、相似断言推荐 |
| `entity_embeddings` | `Entity` | `name + description + key_fields` 的 embedding | 实体匹配、跨意图实体关联 |

**embedding 文本组装规则：**

```python
EMBEDDING_TEMPLATES: dict[str, Callable] = {
    "state_fingerprints": lambda s: f"url:{s.url} title:{s.title}\n{s.simplified_html[:2000]}",
    "menu_embeddings": lambda m: f"{m.label} path:{m.stable_path or ''} key:{m.menu_key or m.id}",
    "intent_embeddings": lambda i: f"{i.verb} {i.object}: {i.summary}",
    "zone_embeddings": lambda z: f"[{z.zone_type}] {z.summary}",
    "transition_embeddings": lambda t: f"{t.action} {t.selector}: {t.thought or ''}",
    "checkpoint_embeddings": lambda c: f"[{c.layer}/{c.rule_type}] {c.description}",
    "entity_embeddings": lambda e: f"{e.name}: {e.description} fields={','.join(e.key_fields)}",
}
```

### 10.4 预置后端实现

#### 10.4.1 InMemoryVectorRetriever（开发/测试）

```python
class InMemoryVectorRetriever(VectorRetriever):
    """
    纯内存实现。暴力余弦相似度搜索。
    适用于: 单元测试、开发调试、小规模数据 (<1000 条)。
    
    特点:
    - 零依赖 (仅 numpy)
    - 进程退出即丢失
    - O(N) 搜索复杂度
    """
```

#### 10.4.2 Neo4jVectorRetriever（推荐默认）

```python
class Neo4jVectorRetriever(VectorRetriever):
    """
    使用 Neo4j 5.x+ 内置 Vector Index。
    适用于: 图+向量一体化方案，中等规模 (<100K 向量)。
    
    优势:
    - 无需额外基础设施, 复用 Neo4j 连接
    - 支持 Cypher + Vector 混合查询 (真正的 GraphRAG)
    - embedding 与节点同事务写入, 一致性有保障
    
    映射规则:
    - collection → Neo4j Vector Index (一个 label 一个 index)
    - VectorEntry.id → 节点 id 属性
    - VectorEntry.vector → 节点 embedding 属性 (db.index.vector.createNodeIndex)
    - VectorEntry.metadata → 节点其他属性 (用于 WHERE 过滤)
    
    Cypher 示例:
    ```cypher
    -- 创建向量索引
    CALL db.index.vector.createNodeIndex(
      'state_fingerprints',   -- index name
      'State',                -- label
      'embedding',            -- vector property
      1536,                   -- dimension
      'cosine'                -- similarity function
    )
    
    -- 向量搜索 + 图遍历 (GraphRAG hybrid)
    CALL db.index.vector.queryNodes('state_fingerprints', 5, $query_vector)
    YIELD node AS state, score
    WHERE score >= 0.8
    MATCH (state)-[:HAS_ZONE]->(z:Zone)
    OPTIONAL MATCH (z)<-[:IN_ZONE]-(t:Transition)-[:REALIZES]->(i:Intent)
    RETURN state.url, state.title, score,
           collect(DISTINCT z.summary) AS zones,
           collect(DISTINCT i.name) AS intents
    ORDER BY score DESC
    ```
    """
```

#### 10.4.3 外部 Vector DB 适配器（可扩展）

```python
class ChromaVectorRetriever(VectorRetriever):
    """Chroma 适配器。适用于: 本地部署、快速原型。"""

class QdrantVectorRetriever(VectorRetriever):
    """Qdrant 适配器。适用于: 高性能、支持过滤的大规模检索。"""

class MilvusVectorRetriever(VectorRetriever):
    """Milvus 适配器。适用于: 超大规模 (百万级+) 向量检索。"""
```

外部适配器为**可选依赖**，通过 extras 安装：

```
pip install autotestagent[chroma]
pip install autotestagent[qdrant]
pip install autotestagent[milvus]
```

### 10.5 EmbeddingProvider 预置实现

```python
class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI text-embedding-3-small / text-embedding-3-large。
    支持自定义 base_url 以兼容本地部署的兼容 API。
    """
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 100,               # 单次 API 调用最大文本数
    ): ...


class CachedEmbeddingProvider(EmbeddingProvider):
    """
    带缓存的 Provider 装饰器。
    - 用 text 的 SHA256 作为缓存 key
    - 支持内存缓存 + 可选磁盘缓存 (SQLite)
    - 避免相同文本重复调用 API
    """
    def __init__(
        self,
        inner: EmbeddingProvider,
        cache_path: str | None = None,       # None=仅内存, str=SQLite 路径
    ): ...
```

### 10.6 核心使用场景

#### 场景 1: 状态去重（cartography）

```python
async def is_duplicate_state(
    controller: PageController,
    retriever: VectorRetriever,
    embedding: EmbeddingProvider,
    threshold: float = 0.92,
) -> State | None:
    """判断当前页面是否已存在于图中。"""
    state = await controller.get_browser_state()
    
    [query_vec] = await embedding.embed([
        f"url:{state.url} title:{state.title}\n{state.content[:2000]}"
    ])
    
    results = await retriever.search(VectorSearchRequest(
        collection="state_fingerprints",
        query_vector=query_vec,
        top_k=3,
        min_score=threshold,
    ))
    
    if results:
        return results[0]  # 返回最相似的已有状态
    return None             # 新状态
```

#### 场景 2: 意图合并（intent）

```python
async def find_similar_intents(
    new_intent: Intent,
    retriever: VectorRetriever,
    embedding: EmbeddingProvider,
) -> list[VectorResult]:
    """找到语义相似的已有意图, 判断是否合并。"""
    [vec] = await embedding.embed([f"{new_intent.verb} {new_intent.object}: {new_intent.summary}"])
    
    return await retriever.search(VectorSearchRequest(
        collection="intent_embeddings",
        query_vector=vec,
        top_k=5,
        min_score=0.85,
    ))
```

#### 场景 3: GraphRAG 混合查询

```python
async def hybrid_search(
    question: str,
    retriever: VectorRetriever,
    embedding: EmbeddingProvider,
    neo4j_driver,
) -> list[dict]:
    """
    自然语言问题 → 向量检索候选节点 → 图遍历扩展上下文。
    
    例: "怎么创建一个管理员用户" →
      1. 向量检索: 在 intent_embeddings 中找到 "创建用户" 意图 (score=0.91)
      2. 图遍历: 从该 Intent 出发, 沿 REALIZES← Transition ←FROM State 拿到路径
      3. 再从 Intent 出发找 REQUIRES_ENTITY 拿到前置条件
    """
    [q_vec] = await embedding.embed([question])
    
    # Phase 1: Vector retrieval
    intent_hits = await retriever.search(VectorSearchRequest(
        collection="intent_embeddings",
        query_vector=q_vec, top_k=3, min_score=0.7,
    ))
    
    state_hits = await retriever.search(VectorSearchRequest(
        collection="state_fingerprints",
        query_vector=q_vec, top_k=3, min_score=0.7,
    ))
    
    # Phase 2: Graph expansion (Cypher)
    candidate_ids = [h.id for h in intent_hits + state_hits]
    result = await neo4j_driver.execute_query("""
        UNWIND $ids AS nid
        MATCH (n {id: nid})
        OPTIONAL MATCH path = (n)-[*1..3]-()
        RETURN n, collect(path) AS context
    """, ids=candidate_ids)
    
    return result
```

#### 场景 4: 检查点模板复用（checkpoint）

```python
async def suggest_checkpoints(
    transition: Transition,
    retriever: VectorRetriever,
    embedding: EmbeddingProvider,
) -> list[Checkpoint]:
    """根据相似 Transition 的检查点, 推荐新 Transition 的检查点模板。"""
    [vec] = await embedding.embed([
        f"{transition.action} {transition.selector}: {transition.thought or ''}"
    ])
    
    similar_transitions = await retriever.search(VectorSearchRequest(
        collection="transition_embeddings",
        query_vector=vec, top_k=5, min_score=0.8,
    ))
    
    # 从相似 Transition 的 CHECK_AFTER 检查点中提取模板
    # ... Cypher 查询获取检查点并参数化替换
```

### 10.7 生命周期管理

```python
class VectorSyncManager:
    """
    保持 Neo4j 节点与 Vector DB 的同步。
    
    同步策略:
    - 写入时同步: 在 Neo4j 写入节点的同时, 调用 retriever.upsert 写入向量
    - 删除时同步: Neo4j 删除节点时, 同步删除向量
    - 全量重建: 从 Neo4j 读取所有节点, 重新 embed 并写入 (用于切换后端或模型)
    - 增量更新: 仅处理 last_modified > last_sync 的节点
    """
    
    def __init__(
        self,
        retriever: VectorRetriever,
        embedding: EmbeddingProvider,
        neo4j_driver,
    ): ...
    
    async def sync_node(self, label: str, node_id: str, text: str) -> None:
        """单节点同步 (写入/更新时调用)。"""
        ...
    
    async def rebuild_collection(self, collection: str) -> int:
        """全量重建某个集合。返回同步条数。"""
        ...
    
    async def incremental_sync(self) -> dict[str, int]:
        """增量同步所有集合。返回 {collection: synced_count}。"""
        ...
```

### 10.8 配置

```python
@dataclass
class VectorConfig:
    """VectorRetriever 统一配置。"""
    
    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_dimension: int = 1536
    embedding_cache_path: str | None = ".autotestagent/embedding_cache.db"
    
    # Retriever backend
    backend: str = "neo4j"                   # "neo4j" | "memory" | "chroma" | "qdrant" | "milvus"
    
    # Neo4j (复用主连接)
    # 无需额外配置, 使用 Neo4jConfig 中的连接
    
    # Chroma
    chroma_path: str | None = None           # 持久化目录, None=内存
    
    # Qdrant
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    
    # Milvus
    milvus_uri: str | None = None
    
    # 通用
    default_top_k: int = 10
    default_min_score: float = 0.7
    
    # 集合配置 (可按集合覆盖默认参数)
    collections: dict[str, CollectionConfig] = field(default_factory=dict)


@dataclass
class CollectionConfig:
    """单个集合的配置覆盖。"""
    min_score: float | None = None           # 覆盖默认 min_score
    top_k: int | None = None                 # 覆盖默认 top_k
    enabled: bool = True                     # 是否启用此集合
```

### 10.9 与 CognitiveProcessor 的关系

现有 `CognitiveProcessor` 将被**重构为 VectorRetriever 的消费者**，而非自己管理向量：

```
重构前:
  CognitiveProcessor
  ├── _get_embedding(text) → list[float]         # 直接调 API
  ├── calculate_similarity(vec1, vec2) → float   # 手动计算
  └── is_similar(vec1, vec2, threshold) → bool   # 内存比对

重构后:
  CognitiveProcessor
  ├── embedding_provider: EmbeddingProvider       # 注入
  ├── retriever: VectorRetriever                  # 注入
  ├── compute_fingerprint(state) → str            # 调 embedding_provider
  ├── find_similar_states(state) → list           # 调 retriever.search
  └── is_similar_state(s1, s2, threshold) → bool  # 调 retriever.search
```

`CognitiveProcessor` 保留为**业务门面**，封装"状态相似度判断"这一业务语义，底层存储和检索全部委托给注入的 `VectorRetriever` 和 `EmbeddingProvider`。

---

## 11. lib 模块 — 页面理解引擎

### 11.1 设计背景

`lib` 模块为 AutoTestAgent 提供**结构化的页面理解能力**。它不是 Agent 本身，而是 Agent 的"眼睛和手"。

**核心设计决策**：DOM 提取、iframe 处理、元素可见性判断等底层能力**完全委托给 browser-use 的 `DomService`**（基于 CDP 协议），不再自建 JS 注入或 Playwright frame API。lib 模块聚焦于**高层交互控制**（W3C 事件模拟、智能滚动、contenteditable 输入）和**页面状态管理**。

### 11.2 与 page-agent 的关系

page-agent 的核心交互逻辑已移植到 `PageController` 中，通过 `page.evaluate()` 注入 JS 执行：

| page-agent 原始逻辑 | lib 中的实现方式 |
|---------------------|---------------|
| W3C 事件模拟 (pointer/mouse/click) | `_CLICK_ELEMENT_JS`：scrollIntoView + elementFromPoint + 完整 pointer/mouse 事件序列 |
| contenteditable 输入 | `_INPUT_TEXT_JS`：多策略（synthetic InputEvent → execCommand fallback → native value setter） |
| 智能滚动 (vertical) | `_SCROLL_VERTICAL_JS`：查找最近可滚动祖先容器，含边缘检测 |
| 智能滚动 (horizontal) | `_SCROLL_HORIZONTAL_JS`：同上，水平方向 |
| DOM Patches | `_PATCH_REACT_JS`：标记 React root 等容器元素为 non-interactive |

### 11.3 当前模块结构

```
graph_agent/lib/
├── __init__.py              # 导出 PageInfo, BrowserState, ActionResult, PageController
├── types.py                 # 数据类型: PageInfo, BrowserState, ActionResult
├── page_controller.py       # 页面控制器 (656 行, 核心)
│   class PageController:
│       __init__(browser_session)     # 接受 browser-use BrowserSession
│       update_tree() -> str          # 调用 DomService 提取 DOM, 返回简化 HTML
│       click_element(index)          # W3C 事件模拟点击 (JS 注入)
│       input_text(index, text)       # 多策略文本输入 (含 contenteditable)
│       select_option(index, text)    # 下拉选择
│       scroll(direction, amount, index?)          # 智能垂直滚动
│       scroll_horizontally(direction, amount, index?)  # 智能水平滚动
│       execute_javascript(script)    # 执行自定义 JS
│       get_last_update_time()        # 上次 DOM 更新时间戳
│       dispose()                     # 清理内部状态
│
├── scope_manager.py         # Scope 过滤 (71 行)
│   class ScopeManager:
│       apply_scope(page, scope) -> None
│       clear_scope(page) -> None
│
└── tools.py                 # 内置工具定义 (170 行)
    TOOLS: dict[str, ToolDef]
    # done, wait, click_element_by_index, input_text,
    # select_dropdown_option, scroll, scroll_horizontally,
    # execute_javascript, go_back, close_overlay
```

### 11.4 DOM 提取委托 browser-use

lib 不再自建 DOM 提取。`PageController.update_tree()` 内部调用链：

```
PageController.update_tree()
  → DomService(browser_session).get_serialized_dom_tree()
    → CDP: DOM.getDocument + Accessibility tree + Layout snapshot
    → 递归处理同源/跨域 iframe (max_depth=5)
    → 坐标系转换 (total_frame_offset)
    → 可见性判断 (is_visible, paint order filtering)
    → 输出: simplified_html + selector_map
```

**browser-use `DomService` 已内置的能力（无需自建）**：

- 可交互元素检测 + 索引分配
- 可见性判断（display/visibility/opacity/面积/paint order）
- 多层级 iframe 递归提取（同源 `contentDocument` + 跨域 CDP Target）
- 坐标系自动转换（含 iframe 内滚动修正）
- 简化 HTML 序列化（`[index]<tag attrs>text />`）

### 11.5 核心类型

```python
@dataclass
class PageInfo:
    viewport_width: int
    viewport_height: int
    page_width: int
    page_height: int
    scroll_x: int
    scroll_y: int
    pixels_above: int
    pixels_below: int
    pages_above: float
    pages_below: float
    total_pages: float
    current_page_position: float

@dataclass
class BrowserState:
    url: str
    title: str
    header: str          # 页面信息 + 滚动提示
    content: str         # 简化 HTML (可交互元素带索引)
    footer: str          # 底部滚动提示
    frame_context: str | None = None

@dataclass
class ActionResult:
    success: bool
    message: str
```

### 11.6 ReAct 探索架构

PRD 原规划的 `lib/agent_loop.py` 已由 `cartography/react_explorer.py` + `react_schema.py` + `react_prompts.py` 三个文件实现：

```
cartography/
├── react_explorer.py     # ReAct 循环: observe → think → act (409 行)
│   class ReActExplorer:
│       explore_page(session, state_id, page_title) -> CartographyResult
│       _invoke_llm_with_retry(system_prompt, user_prompt) -> dict | None
│       _execute_action(controller, action) -> str
│
├── react_schema.py       # Pydantic 结构化输出 (167 行)
│   AgentOutput: evaluation_previous_goal, memory, next_goal, action
│   AgentAction: 判别联合类型 (click | input | scroll | done | ...)
│
└── react_prompts.py      # 系统/用户提示词 (143 行)
    CARTOGRAPHY_SYSTEM_PROMPT: 包含可用动作、输出格式说明
    build_user_prompt(): 组装 DOM + history + observations
```

**LLM 输出格式**：使用 Pydantic `AgentOutput` 模型 + `ainvoke_structured`（类似 page-agent 的 Zod schema），通过 OpenAI `response_format: json_schema` 严格验证，无正则回退。

### 11.7 与上层模块的集成

```python
# cartography/react_explorer.py — ReAct 探索
from graph_agent.lib.page_controller import PageController

controller = PageController(browser_session)
simplified_html = await controller.update_tree()  # DOM via browser-use DomService
# LLM 基于 simplified_html 决策
output = await ainvoke_structured(llm, system_prompt, user_prompt, AgentOutput)
# 通过 PageController 执行动作 (自动跨 iframe)
await controller.click_element(output.action.index)


# cartography/orchestrator.py — 测绘编排
from graph_agent.lib.page_controller import PageController

orchestrator = CartographyOrchestrator(neo4j_driver, browser_session)
report = await orchestrator.run_session(
    start_url="https://app.example.com",
    app_id="app:my-crm",
    app_name="My CRM",
)
```

---

## 12. 模块结构 (当前)

```
AutoTestAgent/
├── docker-compose.yml               # Neo4j 容器编排
├── neo4j_data/                      # Neo4j 持久化数据 (⚠️ .gitignore)
│
├── retriever/                       # 向量检索抽象层
│   ├── __init__.py
│   ├── types.py                     # VectorEntry, VectorResult, VectorSearchRequest
│   ├── base.py                      # EmbeddingProvider, VectorRetriever (ABC)
│   ├── embedding.py                 # OpenAIEmbeddingProvider, CachedEmbeddingProvider
│   ├── memory.py                    # InMemoryVectorRetriever
│   ├── neo4j_vector.py              # Neo4jVectorRetriever
│   ├── graphrag.py                  # GraphRAG hybrid 查询
│   ├── sync.py                      # VectorSyncManager
│   └── config.py                    # VectorConfig, CollectionConfig
│
├── graph_agent/                     # 业务层
│   ├── models_v2.py                 # 统一 Pydantic 数据模型 (App, State, Intent, GraphEdge...)
│   ├── llm.py                       # LLM 工具 (get_llm, ainvoke_prompt, ainvoke_structured)
│   │
│   ├── lib/                         # 页面理解引擎 (基于 browser-use CDP)
│   │   ├── __init__.py              # 导出: PageInfo, BrowserState, ActionResult, PageController
│   │   ├── types.py                 # PageInfo, BrowserState, ActionResult
│   │   ├── page_controller.py       # 核心: DOM 提取 + W3C 交互 + 智能滚动 (656 行)
│   │   ├── scope_manager.py         # Scope 过滤
│   │   └── tools.py                 # 内置工具定义 (ToolDef)
│   │
│   ├── neo4j/                       # Neo4j 存储层
│   │   ├── driver.py                # 连接管理 + Schema 初始化 (含 App 约束)
│   │   ├── repository.py            # CRUD: App, State, Transition, Zone, Intent...
│   │   └── queries.py               # Cypher 模板 (含 App 查询, 按 App 过滤)
│   │
│   ├── cartography/                 # 测绘引擎
│   │   ├── orchestrator.py          # 测绘编排 (支持 app_id)
│   │   ├── react_explorer.py        # ReAct 探索器 (LLM 驱动, Pydantic 输出)
│   │   ├── react_schema.py          # AgentOutput + AgentAction Pydantic 模型
│   │   ├── react_prompts.py         # 系统/用户提示词
│   │   ├── explorer.py              # 旧规则驱动探索器 (已被 react_explorer 替代)
│   │   ├── scope.py                 # 预定义 Scope 模板
│   │   ├── menu_extractor.py        # 菜单树提取
│   │   ├── zone_discoverer.py       # 功能区域发现
│   │   └── snapshot.py              # DOM 指纹
│   │
│   ├── checkpoint/
│   │   ├── generator.py             # Checkpoint 自动生成
│   │   ├── runner.py                # Checkpoint 执行
│   │   ├── constraint_prober.py     # 字段约束探测
│   │   └── test_deriver.py          # 边界测试派生
│   │
│   ├── intent/
│   │   ├── inferrer.py              # 意图推断
│   │   ├── dependency.py            # 意图依赖分析
│   │   ├── planner.py               # 意图规划
│   │   └── entity_pool.py           # 实体池管理
│   │
│   ├── coverage/
│   │   ├── analyzer.py              # 覆盖率计算
│   │   └── scheduler.py             # 探索任务调度
│   │
│   ├── graph/
│   │   ├── merger.py                # 图合并 (支持 app_id)
│   │   ├── io.py                    # NetworkX 图序列化
│   │   ├── pathfinding.py           # 路径查找
│   │   └── templates.py             # 业务模板生成
│   │
│   ├── playback/
│   │   └── engine.py                # Playwright 回放引擎
│   │
│   ├── mapping/                     # 保留兼容 (mapping 管线)
│   │   ├── parser.py                # browser-use AgentHistory → GraphEdge
│   │   ├── run.py                   # mapping 入口
│   │   ├── scout.py                 # URL 发现
│   │   ├── semantic_scout.py        # 语义 scout
│   │   └── cognitive_processor.py   # 认知处理 (待重构为 retriever 消费者)
│   │
│   ├── acceptance/                  # 验收测试工具
│   │   ├── playback_acceptance.py
│   │   ├── playback_diagnostics.py
│   │   └── failure_chain.py
│   │
│   └── web/
│       ├── app.py                   # FastAPI 服务
│       └── static/                  # 前端静态文件
│
└── tests/graph_agent/              # 测试
```

---

## 13. 里程碑

| 里程碑 | 内容 | 产出 | 状态 |
|--------|------|------|------|
| **M0** | **lib 页面理解引擎** | `lib/page_controller.py`（基于 browser-use CDP）、`scope_manager.py`、`tools.py` | ✅ 完成（架构已变：委托 browser-use DomService，不再自建 JS 注入） |
| **M1** | Neo4j 部署 + Schema + 数据模型 + 存储层 | `docker-compose.yml`, `models_v2.py`, `neo4j/` 模块 | ⚠️ 部分完成（新增 `Menu` 节点后的 Schema/约束/关系待补齐） |
| **M1.5** | **VectorRetriever 抽象层** | `retriever/` 全模块, InMemory + Neo4j 后端, GraphRAG hybrid | ✅ 完成 |
| **M2** | Scope 机制 + 菜单/Zone 发现 | `cartography/scope.py`, `menu_extractor.py`, `zone_discoverer.py` | ⚠️ 部分完成（菜单发现已完成，`Menu` 图建模写入待补齐） |
| **M3** | 多次测绘 + 图合并 + 覆盖率 | `coverage/`, `graph/merger.py`, `cartography/orchestrator.py` | ✅ 完成（含 app_id 隔离） |
| **M4** | Checkpoint 引擎 + 自动生成 | `checkpoint/` 全模块 | ✅ 完成 |
| **M5** | 意图依赖 + 实体管理 + 规划器 | `intent/` 全模块 | ✅ 完成 |
| **M6** | iframe 探索 | browser-use DomService 原生支持（删除了 FrameManager / FrameDiscoverer） | ✅ 完成（方案变更） |
| **M6.5** | ReAct 探索架构 | `react_explorer.py`, `react_schema.py`, `react_prompts.py`, Pydantic 结构化输出 | ✅ 完成（新增里程碑） |
| **M7** | 边界值/等价类测试派生 | `checkpoint/constraint_prober.py`, `checkpoint/test_deriver.py` | ✅ 完成 |
| **M8** | GraphRAG hybrid 查询 + 外部 VectorDB 适配器 | `retriever/graphrag.py`, hybrid 查询 API | ⚠️ 部分完成（外部 VectorDB 适配器待实现） |
| **M9** | 端到端验收 | 目标站点全链路测试通过 | 🔄 进行中 |
| **M10** | 回放验证器 | `playback/validator.py` | ❌ 未开始 |

---

## 14. 验收标准

### 14.1 功能验收

**lib 页面理解引擎 (基于 browser-use CDP):**
- [x] `PageController` 通过 `DomService` 提取简化 DOM，索引格式 `[N]<tag attrs>text />`
- [x] `PageController` 支持通过索引执行 click/input/select/scroll，W3C 事件模拟
- [x] browser-use `DomService` 原生支持多层级 iframe（同源 + 跨域），最深 5 层
- [x] 跨 frame 元素自动编入 selector_map，操作自动路由到正确 CDP session
- [x] `ScopeManager` 可通过 `data-scope-excluded` 属性过滤元素，不影响页面渲染
- [x] contenteditable 输入支持：synthetic InputEvent → execCommand fallback → value setter
- [x] 智能滚动：自动查找最近可滚动祖先容器，支持垂直和水平方向

**VectorRetriever 抽象层:**
- [x] `EmbeddingProvider` 支持 OpenAI 兼容 API, 可配置 model/base_url
- [x] `CachedEmbeddingProvider` 对相同文本不重复调用 API (SHA256 缓存命中)
- [x] `InMemoryVectorRetriever` 通过所有 VectorRetriever 协议测试
- [x] `Neo4jVectorRetriever` 通过所有 VectorRetriever 协议测试
- [ ] 7 个集合 (state/menu/intent/zone/transition/checkpoint/entity) 的 embedding 模板正确（`menu_embeddings` 待实现）
- [x] `VectorSyncManager` 支持单节点同步、全量重建、增量同步
- [x] GraphRAG hybrid 查询: 自然语言问题 → 向量检索 + 图遍历, 返回结构化上下文
- [x] `CognitiveProcessor` 重构后行为与旧版一致 (状态相似度判断)
- [x] 外部 VectorDB 适配器通过 extras 安装, 不影响核心依赖

**Neo4j 部署:**
- [x] `docker compose up -d` 一键启动 Neo4j 5.26.24-community
- [x] 数据持久化到 `neo4j_data/`, 容器重启后数据不丢失
- [x] `neo4j_data/` 已加入 `.gitignore`, 不会误提交
- [x] `graph_agent/neo4j/driver.py` 可从 `.env` 读取连接配置并成功连接

**业务层:**
- [ ] **Menu** 为独立节点：`CHILD_OF` 建菜单树，`LEADS_TO` 指向 **State**；`menu_path` 可与图一致冗余；测绘写入 `(:Session)-[:DISCOVERED]->(:Menu)`（见 3.1 / 3.5）
- [ ] 支持对含多级菜单的 SPA 进行自动测绘, 生成状态图存入 Neo4j（`Menu` 节点及关系落库待完成）
- [x] 支持含 2 级以上 iframe 嵌套的页面探索与回放
- [x] 支持多次 session 增量补全图谱, 覆盖率单调递增
- [x] Scope 机制有效限制 agent 在指定区域内操作
- [x] 自动生成的 Checkpoint 能检测操作是否生效
- [x] Entity Checkpoint 能验证前置条件和后置效果
- [x] 边界值测试用例能自动派生并执行

### 14.2 质量验收

- [ ] 单次测绘后 `menu_coverage >= 60%`（新增 `Menu` 节点后需重测）
- [x] 3 次测绘后 `zone_coverage >= 80%` (实测 100%)
- [x] Transition 平均 `confidence >= 0.6` (实测 0.70, 7 条已验证)
- [x] Checkpoint 自动生成覆盖率: 每条 Transition 至少 1 个 `CHECK_AFTER` (实测 7/7 = 100%)
- [x] 端到端: 至少 1 条长度 >= 3 的业务路径可完整回放并通过所有 Checkpoint (实测: 工作台→项目→产品→我的)

### 14.3 非功能验收

- [x] Neo4j 查询延迟 < 200ms (千节点规模)
- [x] 向量检索延迟 < 100ms (万级向量, Neo4j Vector Index)
- [x] Embedding 缓存命中率 > 80% (稳态运行时)
- [x] 单次 session 支持 10 分钟时间预算, 可中断恢复
- [ ] 所有模型字段变更有 Neo4j migration 脚本
- [x] 切换 VectorRetriever 后端无需修改业务代码

---

## 15. 风险与缓解

| 风险 | 缓解 |
|------|------|
| LLM 意图推断不稳定 | Pydantic 结构化输出 + 指数退避重试 + 多次验证提升 confidence |
| SPA 路由变化检测困难 | URL + DOM fingerprint 双重识别 |
| iframe cross-origin 限制 | browser-use CDP 原生支持跨域 iframe（需开启 `cross_origin_iframes=True`），默认关闭 |
| Neo4j 性能瓶颈 | 索引关键字段 (id, url, fingerprint), 分页查询 |
| Scope 过滤遗漏 | 多层防御: DOM 过滤 + Tool 拦截 + URL 守卫 |
| 多次测绘合并冲突 | 保留更短路径, 降低而非删除置信度 |
| Embedding 模型更新导致向量不一致 | `VectorSyncManager.rebuild_collection` 全量重建; embedding_model 版本记录在集合元数据中 |
| 向量与图数据不同步 | 写入时同步 (同事务/同回调); 定期增量同步兜底; 健康检查对比两侧计数 |
| Vector DB 选型锁定 | VectorRetriever 抽象层解耦; 业务代码仅依赖协议, 不依赖具体实现 |
