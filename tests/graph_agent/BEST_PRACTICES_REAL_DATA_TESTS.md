# 基于真实数据的测试最佳实践

## 1. 测试分层架构

```
tests/
├── unit/                    # 单元测试 - 纯内存，无外部依赖
│   ├── test_models.py
│   └── test_parser.py
├── integration/             # 集成测试 - 需要 Neo4j/LLM
│   ├── test_neo4j_repository.py
│   └── test_vector_retriever.py
├── e2e/                     # E2E 测试 - 需要浏览器
│   └── test_cartography.py
└── acceptance/              # 验收测试 - 真实数据验证
    └── test_real_app_mapping.py
```

## 2. 真实数据测试的核心原则

### 2.1 数据隔离
- 每个测试使用独立的 `app_id`
- 测试结束后清理数据
- 使用 fixture 管理数据生命周期

### 2.2 可重复性
- 使用固定的随机种子
- 记录 LLM 响应用于回放
- 版本控制测试数据快照

### 2.3 环境适配
- 通过环境变量配置
- 支持 .env 文件加载
- 提供降级方案（mock）

## 3. 真实数据测试编写模式

### 模式 A: 录制-回放模式
```python
# 1. 录制阶段（一次性）
async def test_record_real_session():
    # 运行真实测绘，保存结果
    result = await run_cartography(url)
    save_snapshot(result, "snapshots/session_v1.json")

# 2. 回放阶段（CI 中使用）
async def test_replay_recorded_session():
    # 加载录制结果，验证存储逻辑
    snapshot = load_snapshot("snapshots/session_v1.json")
    await store_to_neo4j(snapshot)
    # 验证存储正确性
```

### 模式 B: 属性测试
```python
@pytest.mark.asyncio
async def test_cartography_properties():
    """验证测绘结果满足特定属性，不依赖具体数据"""
    result = await run_cartography(TEST_URL)
    
    # 属性验证
    assert result.states  # 至少发现一些状态
    assert all(s.url for s in result.states)  # 所有状态有 URL
    assert all(t.selector for t in result.transitions)  # 所有转换有选择器
```

### 模式 C: 端到端完整流程
```python
@pytest.mark.asyncio
async def test_full_workflow():
    """测试完整链路：测绘 → 存储 → 查询 → 回放"""
    # 1. 测绘
    cartography_result = await run_cartography(TEST_URL)
    
    # 2. 存储到 Neo4j
    await store_result(cartography_result)
    
    # 3. 查询路径
    path = await find_path(start, end)
    assert len(path) > 0
    
    # 4. 回放验证
    playback_result = await run_playback(path)
    assert playback_result.success
```

## 4. 真实数据测试示例

### 示例 1: 基于真实网站的测绘测试
```python
import pytest
from graph_agent.cartography.orchestrator import CartographyOrchestrator
from graph_agent.neo4j.driver import Neo4jDriver

# 测试配置
TEST_APP_ID = "test:real:demo-app"
TEST_URL = "https://example.com/login"

@pytest.fixture
async def neo4j_driver():
    """提供 Neo4j 连接，测试后清理"""
    driver = Neo4jDriver()
    await driver.connect()
    yield driver
    # 清理测试数据
    await clean_test_data(driver, TEST_APP_ID)
    await driver.close()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_site_cartography(neo4j_driver):
    """对真实网站进行测绘，验证数据正确存储"""
    orchestrator = CartographyOrchestrator(
        neo4j_driver.driver,
        app_id=TEST_APP_ID,
        app_name="Demo App",
    )
    
    # 运行测绘
    report = await orchestrator.run_session(
        start_url=TEST_URL,
        time_budget_ms=60_000,
    )
    
    # 验证报告
    assert report.states_discovered >= 1
    assert report.transitions_discovered >= 1
    
    # 验证数据存储
    repo = GraphRepository(neo4j_driver.driver)
    states = await repo.get_app_states(TEST_APP_ID)
    assert len(states) == report.states_discovered
```

### 示例 2: Menu 发现测试（使用新实现的 Menu 功能）
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_menu_extraction_and_storage(neo4j_driver, browser_session):
    """测试菜单提取和存储"""
    from graph_agent.cartography.menu_extractor import MenuExtractor, store_menu_tree
    
    page = browser_session.page
    await page.goto(TEST_URL)
    
    # 提取菜单
    extractor = MenuExtractor()
    tree_data = await extractor.extract_tree(page)
    
    # 验证提取结果
    assert "items" in tree_data
    assert len(tree_data["items"]) > 0
    
    # 存储到 Neo4j
    repo = GraphRepository(neo4j_driver.driver)
    menus = await store_menu_tree(repo, tree_data, TEST_APP_ID, "test-session-1")
    
    # 验证存储
    stored_menus = await repo.get_app_menus(TEST_APP_ID)
    assert len(stored_menus) == len(menus)
    
    # 验证树结构
    menu_tree = await repo.get_menu_tree(TEST_APP_ID)
    assert len(menu_tree) > 0
```

### 示例 3: VectorRetriever 集成测试
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_vector_search_with_real_data(neo4j_driver):
    """测试向量检索与真实数据"""
    from retriever.neo4j_vector import Neo4jVectorRetriever
    from retriever.embedding import OpenAIEmbeddingProvider
    
    # 创建 retriever
    embedding = OpenAIEmbeddingProvider()
    retriever = Neo4jVectorRetriever(neo4j_driver.driver)
    
    # 准备测试数据
    test_intents = [
        {"id": "intent:login", "name": "用户登录", "summary": "使用用户名密码登录系统"},
        {"id": "intent:logout", "name": "用户登出", "summary": "退出当前登录状态"},
    ]
    
    # 生成 embedding
    texts = [f"{i['name']}: {i['summary']}" for i in test_intents]
    vectors = await embedding.embed(texts)
    
    # 存储向量
    from retriever.types import VectorEntry
    entries = [
        VectorEntry(id=i["id"], vector=v, metadata=i)
        for i, v in zip(test_intents, vectors)
    ]
    await retriever.ensure_collection("intent_embeddings", embedding.dimension())
    await retriever.upsert("intent_embeddings", entries)
    
    # 搜索验证
    query_vec = await embedding.embed(["如何登录系统"])
    results = await retriever.search(VectorSearchRequest(
        collection="intent_embeddings",
        query_vector=query_vec[0],
        top_k=2,
    ))
    
    assert len(results) > 0
    assert results[0].id == "intent:login"
```

## 5. 测试数据管理

### 5.1 使用 Factory 模式创建测试数据
```python
class MenuFactory:
    """创建测试用 Menu 数据"""
    
    @staticmethod
    def create_menu_tree(app_id: str, depth: int = 2) -> list[Menu]:
        """创建固定结构的菜单树用于测试"""
        menus = []
        now = datetime.utcnow()
        
        # 根菜单
        root = Menu(
            id=f"menu:{app_id}:root",
            label="系统管理",
            level=0,
            order_index=0,
            selector=".menu-root",
            menu_key="system",
            app_id=app_id,
            stable_path="system",
            first_discovered=now,
            last_seen=now,
        )
        menus.append(root)
        
        # 子菜单
        for i, label in enumerate(["用户管理", "角色管理", "权限管理"]):
            child = Menu(
                id=f"menu:{app_id}:system:{label}",
                label=label,
                level=1,
                order_index=i,
                selector=f".menu-{label}",
                menu_key=f"system.{label}",
                app_id=app_id,
                stable_path=f"system/{label}",
                first_discovered=now,
                last_seen=now,
            )
            menus.append(child)
        
        return menus
```

### 5.2 数据快照对比
```python
def assert_graph_structure_match(actual: dict, expected_snapshot_path: str):
    """验证图结构与快照匹配"""
    with open(expected_snapshot_path) as f:
        expected = json.load(f)
    
    # 比较关键指标
    assert actual["state_count"] == expected["state_count"]
    assert actual["transition_count"] == expected["transition_count"]
    
    # 比较关键路径存在性
    for path in expected["required_paths"]:
        assert path_exists(actual["graph"], path), f"Missing path: {path}"
```

## 6. CI/CD 集成建议

### 6.1 分层运行
```yaml
# .github/workflows/test.yml
test:
  stages:
    - unit-tests      # 快速，每次提交运行
    - integration     # 中等，PR 时运行
    - e2e             # 慢，合并前运行
    - acceptance      # 很慢，定期运行
```

### 6.2 测试标记
```python
# 使用 pytest 标记分类测试
@pytest.mark.unit           # 单元测试
@pytest.mark.integration    # 集成测试（需要 Neo4j）
@pytest.mark.e2e           # 端到端测试（需要浏览器）
@pytest.mark.acceptance    # 验收测试（需要真实网站）
@pytest.mark.slow          # 慢测试
```

### 6.3 并行测试
```python
# conftest.py
import pytest

@pytest.fixture(scope="session")
def event_loop():
    """提供事件循环用于异步测试"""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# 为每个测试生成唯一的 app_id
@pytest.fixture
async def test_app_id():
    import uuid
    return f"test:{uuid.uuid4().hex[:8]}"
```

## 7. 推荐测试清单

对于 AutoTestAgent，建议按以下优先级编写真实数据测试：

1. **高优先级（核心功能）**
   - [x] Neo4j 连接和基础 CRUD
   - [x] Menu 节点存储和查询
   - [x] VectorRetriever 基础功能
   - [ ] Cartography 会话存储
   - [ ] Checkpoint 生成和验证

2. **中优先级（集成点）**
   - [ ] 完整测绘流程（小规模网站）
   - [ ] 路径查找和回放
   - [ ] GraphRAG 查询

3. **低优先级（边界情况）**
   - [ ] 大规模数据处理性能
   - [ ] 并发会话处理
   - [ ] 错误恢复机制
