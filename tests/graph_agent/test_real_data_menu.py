"""
基于真实数据的 Menu 功能测试。

这些测试验证：
1. Menu 模型可以正确存储到 Neo4j
2. 菜单树结构可以正确重建
3. Menu 与 State 的关联关系

运行前需要：
    docker compose up -d neo4j
    python scripts/clean_neo4j.py  # 可选：清理旧数据
"""

import pytest
import uuid

from graph_agent.models import Menu, State
from graph_agent.neo4j.driver import Neo4jDriver
from graph_agent.neo4j.repository import GraphRepository
from graph_agent.cartography.menu_extractor import store_menu_tree


@pytest.fixture(scope="module")
async def neo4j_driver():
    """提供 Neo4j 驱动连接。"""
    driver = Neo4jDriver()
    await driver.connect()
    await driver.ensure_schema()
    yield driver
    await driver.close()


@pytest.fixture
async def test_repo(neo4j_driver):
    """提供 GraphRepository 并自动清理测试数据。"""
    repo = GraphRepository(neo4j_driver.driver)
    
    # 生成唯一的测试应用 ID
    test_app_id = f"test:menu:{uuid.uuid4().hex[:8]}"
    
    yield repo, test_app_id
    
    # 清理测试数据
    async with neo4j_driver.driver.session() as session:
        # 删除该应用的所有相关数据
        await session.run("""
            MATCH (a:App {id: $app_id})-[:HAS_MENU]->(m:Menu)
            OPTIONAL MATCH (m)-[r]-()
            DELETE r, m
        """, app_id=test_app_id)
        await session.run("""
            MATCH (a:App {id: $app_id})
            DELETE a
        """, app_id=test_app_id)


class TestMenuModel:
    """测试 Menu 模型的创建和属性。"""
    
    def test_menu_creation(self):
        """测试创建 Menu 实例。"""
        menu = Menu(
            id="menu:test:system:users",
            label="用户管理",
            level=1,
            order_index=0,
            selector="nav .menu-users",
            menu_key="system.users",
            app_id="app:test",
            stable_path="system/users",
        )
        
        assert menu.id == "menu:test:system:users"
        assert menu.label == "用户管理"
        assert menu.level == 1
        assert menu.order_index == 0
        assert menu.stable_path == "system/users"
    
    def test_menu_unique_id_per_app(self):
        """测试不同应用的 Menu ID 是独立的。"""
        menu1 = Menu(
            id="menu:app1:users",
            label="用户管理",
            app_id="app:app1",
        )
        menu2 = Menu(
            id="menu:app2:users",
            label="用户管理",
            app_id="app:app2",
        )
        
        assert menu1.id != menu2.id
        assert menu1.app_id != menu2.app_id


class TestMenuRepository:
    """测试 Menu 的存储和查询。"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_upsert_and_get_menu(self, test_repo):
        """测试 Menu 的增查。"""
        repo, app_id = test_repo
        
        menu = Menu(
            id=f"menu:{app_id}:test",
            label="测试菜单",
            level=0,
            app_id=app_id,
        )
        
        # 存储
        await repo.upsert_menu(menu)
        
        # 查询
        retrieved = await repo.get_menu(menu.id)
        assert retrieved is not None
        assert retrieved.id == menu.id
        assert retrieved.label == menu.label
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_menu_hierarchy(self, test_repo):
        """测试菜单层级关系。"""
        repo, app_id = test_repo
        
        # 创建父菜单
        parent = Menu(
            id=f"menu:{app_id}:parent",
            label="父菜单",
            level=0,
            app_id=app_id,
        )
        await repo.upsert_menu(parent)
        await repo.link_app_menu(app_id, parent.id)
        
        # 创建子菜单
        child = Menu(
            id=f"menu:{app_id}:child",
            label="子菜单",
            level=1,
            order_index=0,
            app_id=app_id,
        )
        await repo.upsert_menu(child)
        await repo.link_app_menu(app_id, child.id)
        await repo.link_menu_child_of(child.id, parent.id, order_index=0)
        
        # 查询树结构
        tree = await repo.get_menu_tree(app_id)
        assert len(tree) == 2
        
        # 验证父子关系
        child_entry = next(e for e in tree if e["menu"].id == child.id)
        assert child_entry["parent_id"] == parent.id
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_menu_leads_to_state(self, test_repo):
        """测试 Menu 与 State 的关联。"""
        repo, app_id = test_repo
        
        # 创建菜单
        menu = Menu(
            id=f"menu:{app_id}:users",
            label="用户管理",
            level=0,
            app_id=app_id,
        )
        await repo.upsert_menu(menu)
        await repo.link_app_menu(app_id, menu.id)
        
        # 创建 State
        state = State(
            id=f"state:{app_id}:users-list",
            url="https://example.com/users",
            title="用户列表",
            app_id=app_id,
        )
        await repo.upsert_state(state)
        await repo.link_app_state(app_id, state.id)
        
        # 关联
        await repo.link_menu_leads_to(menu.id, state.id, session_id="test-session")
        
        # 验证
        async with repo._driver.session() as session:
            result = await session.run("""
                MATCH (m:Menu {id: $menu_id})-[:LEADS_TO]->(s:State)
                RETURN s.id AS state_id
            """, menu_id=menu.id)
            record = await result.single()
            assert record["state_id"] == state.id


class TestMenuExtractor:
    """测试菜单提取功能。"""
    
    def test_build_menu_models(self):
        """测试从树数据构建 Menu 模型。"""
        from graph_agent.cartography.menu_extractor import build_menu_models
        
        tree_data = {
            "items": [
                {
                    "text": "系统管理",
                    "level": 0,
                    "path": "系统管理",
                    "order_index": 0,
                    "selector": ".menu-system",
                    "children": [
                        {
                            "text": "用户管理",
                            "level": 1,
                            "path": "系统管理/用户管理",
                            "order_index": 0,
                            "selector": ".menu-users",
                            "children": [],
                        }
                    ],
                }
            ]
        }
        
        app_id = "test:app"
        menus = build_menu_models(tree_data, app_id, "test-session")
        
        assert len(menus) == 2
        
        # 验证父菜单
        parent = next(m for m in menus if m.level == 0)
        assert parent.label == "系统管理"
        assert parent.stable_path == "系统管理"
        
        # 验证子菜单
        child = next(m for m in menus if m.level == 1)
        assert child.label == "用户管理"
        assert child.stable_path == "系统管理/用户管理"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_store_menu_tree(self, test_repo):
        """测试完整的菜单树存储流程。"""
        repo, app_id = test_repo
        
        tree_data = {
            "root_selector": "nav",
            "items": [
                {
                    "text": "系统管理",
                    "href": "",
                    "selector": ".menu-system",
                    "level": 0,
                    "path": "系统管理",
                    "order_index": 0,
                    "has_children": True,
                    "children": [
                        {
                            "text": "用户管理",
                            "href": "/users",
                            "selector": ".menu-users",
                            "level": 1,
                            "path": "系统管理/用户管理",
                            "order_index": 0,
                            "has_children": False,
                            "children": [],
                        },
                        {
                            "text": "角色管理",
                            "href": "/roles",
                            "selector": ".menu-roles",
                            "level": 1,
                            "path": "系统管理/角色管理",
                            "order_index": 1,
                            "has_children": False,
                            "children": [],
                        },
                    ],
                }
            ],
            "depth": 1,
        }
        
        # 存储菜单树
        menus = await store_menu_tree(repo, tree_data, app_id, "test-session-1")
        
        # 验证存储结果
        assert len(menus) == 3  # 1 个父菜单 + 2 个子菜单
        
        # 验证可以从 Neo4j 查询
        stored_menus = await repo.get_app_menus(app_id)
        assert len(stored_menus) == 3
        
        # 验证树结构
        menu_tree = await repo.get_menu_tree(app_id)
        assert len(menu_tree) == 3
        
        # 验证 CHILD_OF 关系
        parent_id = f"menu:{app_id}:系统管理"
        children_with_parent = [
            e for e in menu_tree 
            if e["parent_id"] == parent_id
        ]
        assert len(children_with_parent) == 2


class TestMenuVectorEmbedding:
    """测试 Menu 的向量嵌入功能。"""
    
    def test_menu_embedding_template(self):
        """测试 Menu embedding 文本模板。"""
        from retriever.sync import EMBEDDING_TEMPLATES
        
        menu_data = {
            "label": "用户管理",
            "stable_path": "system/users",
            "menu_key": "system.users",
        }
        
        text = EMBEDDING_TEMPLATES["menu_embeddings"](menu_data)
        assert "用户管理" in text
        assert "system/users" in text
        assert "system.users" in text
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_menu_vector_sync(self, neo4j_driver):
        """测试 Menu 向量同步。"""
        from retriever.neo4j_vector import Neo4jVectorRetriever
        from retriever.embedding import OpenAIEmbeddingProvider
        from retriever.types import VectorEntry
        
        # 创建测试数据
        app_id = f"test:menu:vector:{uuid.uuid4().hex[:8]}"
        menu = Menu(
            id=f"menu:{app_id}:users",
            label="用户管理系统",
            app_id=app_id,
            stable_path="system/users",
            menu_key="system.users",
        )
        
        # 存储到 Neo4j
        repo = GraphRepository(neo4j_driver.driver)
        await repo.upsert_menu(menu)
        
        # 创建向量索引
        embedding = OpenAIEmbeddingProvider()
        retriever = Neo4jVectorRetriever(neo4j_driver.driver)
        
        await retriever.ensure_collection("menu_embeddings", embedding.dimension())
        
        # 生成并存储向量
        text = f"{menu.label} path:{menu.stable_path} key:{menu.menu_key}"
        vectors = await embedding.embed([text])
        
        await retriever.upsert("menu_embeddings", [
            VectorEntry(id=menu.id, vector=vectors[0], metadata={"label": menu.label})
        ])
        
        # 验证可以搜索到
        query_vec = await embedding.embed(["用户管理"])
        from retriever.types import VectorSearchRequest
        results = await retriever.search(VectorSearchRequest(
            collection="menu_embeddings",
            query_vector=query_vec[0],
            top_k=5,
            min_score=0.7,
        ))
        
        assert len(results) > 0
        assert results[0].id == menu.id
        
        # 清理
        async with neo4j_driver.driver.session() as session:
            await session.run("MATCH (m:Menu {app_id: $app_id}) DELETE m", app_id=app_id)
