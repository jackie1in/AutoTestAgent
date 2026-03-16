# AutoTestAgent

基于 [Browser-Use](https://docs.browser-use.com/) 的智能 UI 自动化测试平台，核心组件：

**Graph Agent**: 基于图谱测绘的业务流程探索与回放工具。

---

## Graph Agent (图谱测绘)

通过“测绘 → 图谱 → 寻径 → 回放”的流程，实现低维护成本的自动化测试。

### 特性
- **🗺️ 自动测绘** - 自动探索业务流程，生成有向图谱 (DiGraph)
- **👁️ 元素发现** - 自动识别页面交互元素，生成清单
- **🧭 意图寻径** - 根据业务意图自动规划执行路径
- **⚡️ 高速回放** - 基于 Playwright 的本地高速回放 (默认使用 Chrome)

### 快速开始

**1. 测绘（Scout + Mapping 一条命令）**
```bash
uv run python -m graph_agent.mapping.run --url https://the-internet.herokuapp.com/login
```
默认生成 `graph_agent/data/element_inventory.json` 和 `graph_agent/data/graph.json`，可通过 `.env` 或 `--inventory` / `--output` 覆盖。

**2. 可视化与回放**
```bash
uv run uvicorn graph_agent.web.app:app --reload
```
访问 http://localhost:8000 使用 Web 界面。
Web API 默认无需登录即可访问与回放。

[查看详细文档](graph_agent/USAGE.md)

---

## 安装与配置

### 安装
```bash
# 使用 uv (推荐)
uv sync
playwright install chromium
```

### 配置环境变量

复制 `.env.example` 到 `.env`，按需填写。Graph Agent 自动从项目根目录加载 `.env`：

```bash
cp .env.example .env
```

```ini
# LLM 配置（OpenRouter 或兼容 OpenAI 的 API）
LLM_API_KEY=sk-xxxxx
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=google/gemini-2.5-pro-preview

# 可选：Graph Agent 测绘与回放
# MAPPING_URL=https://the-internet.herokuapp.com/login
# MAPPING_USERNAME=tomsmith
# MAPPING_PASSWORD=SuperSecretPassword!
# MAPPING_HEADLESS=false
# MAPPING_CHANNEL=chrome
# PLAYWRIGHT_CHANNEL=chrome
# PLAYWRIGHT_HEADLESS=false
```

## 项目结构

```
AutoTestAgent/
├── graph_agent/            # Graph Agent 核心模块
│   ├── mapping/            # 测绘与发现 (Scout)
│   ├── graph/              # 图谱数据结构与 I/O
│   ├── playback/           # 回放引擎
│   ├── web/                # 可视化 Web UI
│   └── USAGE.md            # Graph Agent 详细文档
├── skills/                 # 本地 Skills 目录
└── test_cases/             # 录制的测试用例
```

## 许可证
MIT
