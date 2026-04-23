# Graph Agent 使用指南

Graph Agent 是一个基于图谱的 UI 自动化测试框架，通过「侦察 → 测绘 → 图谱 → 回放」实现低维护成本的自动化测试。

## 核心架构

1. **Scout（侦察）**：扫描目标页面，识别可交互元素，生成元素清单 `element_inventory.json`。
2. **Mapping（测绘）**：使用 AI Agent 探索业务流程（如登录），记录操作步骤并构建有向图。
3. **Graph（图谱）**：节点为页面/URL，边为交互动作（点击、输入），持久化为 Neo4j 图数据库。
4. **Playback（回放）**：按业务意图寻径，用 Playwright 执行回放。

## 快速开始

### 1. 环境准备

依赖与 LLM 配置（使用项目根目录 `.env`）：

```bash
# 安装依赖
uv sync --group graph_agent
playwright install chromium
```

在项目根目录的 `.env` 中配置（Graph Agent 会通过 `load_dotenv()` 自动加载）：

```bash
# LLM（支持 OpenRouter / 兼容 OpenAI 的 API）
LLM_API_KEY=sk-...
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=google/gemini-2.5-pro-preview

# 可选：测绘与回放
# MAPPING_URL=https://the-internet.herokuapp.com/login
# MAPPING_INVENTORY=graph_agent/data/element_inventory.json
# MAPPING_USERNAME=tomsmith
# MAPPING_PASSWORD=SuperSecretPassword!
# PLAYWRIGHT_CHANNEL=chrome
# PLAYWRIGHT_HEADLESS=false
```

### 2. 一键测绘（Scout + Mapping）

一条命令会先执行 Scout 生成元素清单，再执行 Mapping 探索流程并构建 Neo4j 图谱：

```bash
# 使用默认 URL（或 .env 中的 MAPPING_URL）
uv run python -m graph_agent.cartography.runner

# 指定 URL 与清单路径
uv run python -m graph_agent.cartography.runner \
  --url https://the-internet.herokuapp.com/login \
  --inventory graph_agent/data/element_inventory.json
```

**参数**：`--url` 起始页；`--inventory` 元素清单路径。图谱直接写入 Neo4j，无需 `--output`。

### 3. 可视化与回放（Web UI）

```bash
uv run uvicorn graph_agent.web.app:app --reload
```

打开 http://localhost:8000 可：

1. **查看图谱**：节点与边。
2. **选择意图**：在 Intent 下拉框选业务意图（如「点击登录」）。
3. **填写测试数据**：若路径包含表单，填写用户名、密码等。
4. **执行回放**：点击 Playback，查看日志与浏览器执行结果。

说明：当前 Web API 与页面默认无需登录即可使用；`/api/auth/login` 已移除。

若目标站点包含登录表单，可在 `.env` 配置 `MAPPING_USERNAME` / `MAPPING_PASSWORD`，Mapping 录制会自动将其作为登录提示词注入。若未配置（或 `.env` 不存在），不会添加任何登录提示词。

## 进阶

### 语义发现（Semantic Scout）

除 LLM 视觉发现外，支持基于 Accessibility Tree 的语义发现，速度更快、选择器更稳。可通过 `SemanticScout` 在代码中调用。

### 编程方式回放

```python
import asyncio
from graph_agent.graph.pathfinding import get_path_from_query
from graph_agent.models import GraphEdge
from graph_agent.playback.engine import run_playback

async def _load_edges_from_neo4j() -> list[GraphEdge]:
    from graph_agent.neo4j_client.driver import Neo4jDriver
    from graph_agent.graph.pathfinding import neo4j_transition_to_edge_data, _edge_to_model

    driver = Neo4jDriver()
    await driver.connect()
    try:
        async with driver.driver.session() as session:
            result = await session.run("""
                MATCH (a:App)
                WITH a ORDER BY coalesce(a.last_session_at, a.created_at) DESC LIMIT 1
                MATCH (a)-[:HAS_STATE]->(s:State)<-[:FROM]-(t:Transition)-[:TO]->(target:State)
                OPTIONAL MATCH (t)-[:REALIZES]->(i:Intent)
                RETURN t.id AS id, t.step_index AS step_index,
                       s.id AS from_state_id, target.id AS to_state_id,
                       t.selector AS selector, t.action AS action,
                       i{.*} AS intent
                ORDER BY t.step_index
            """)
            edges: list[GraphEdge] = []
            async for record in result:
                t = dict(record)
                data = neo4j_transition_to_edge_data(t)
                u = str(t.get("from_state_id", ""))
                v = str(t.get("to_state_id", ""))
                if u and v:
                    edges.append(_edge_to_model(u, v, data))
            return edges
    finally:
        await driver.close()

async def main():
    edges = await _load_edges_from_neo4j()
    path = get_path_from_query("login_success", edges)
    test_data = {"username": "tomsmith", "password": "SuperSecretPassword!"}

    result = await run_playback(
        path, test_data,
        start_url="https://the-internet.herokuapp.com/login",
    )

    if result["success"]:
        print("Test Passed!")
    else:
        print(f"Test Failed: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 常见问题

**Q: 图谱为空？**  
检查 Scout 是否生成了元素清单，以及 Mapping 阶段 Agent 的操作是否匹配到清单中的元素。

**Q: 回放报 "Selector not found"？**  
页面结构可能已变，重新运行测绘即可更新图谱，无需手改脚本。
