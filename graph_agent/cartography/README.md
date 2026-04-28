# Cartography 模块分层说明

`graph_agent.cartography` 负责网站自动测绘，当前按单一职责拆分为以下层次：

- `runner.py`
  - 对外入口（`run_mapping` + CLI）
  - 负责串联配置、登录、pipeline 执行与结果持久化
- `mapping_pipeline.py`
  - 多页面探索执行引擎
  - 管理队列、预算、跨域恢复、页内探索调度
- `llm_planning.py`
  - 页面分析与下一步探索规划（结构化输出）
- `login.py`
  - 登录检测、表单填充、菜单文本点击、自动登录流程
- `captcha.py`
  - 验证码图像采集与识别路由（ddddocr 优先 + LLM fallback）
- `persistence.py`
  - 将 `CartographyResult` 持久化到 Neo4j
- `config.py`
  - URL/环境变量/timeout/inventory 等配置工具
- `browser_lifecycle.py`
  - 浏览器生命周期管理、信号处理与关停状态

## 典型调用链

1. `runner.run_mapping()` 读取配置并启动浏览器
2. `login` 执行预登录（含 `captcha`）
3. `mapping_pipeline.run_orchestrated_mapping()` 执行多页探索
4. `persistence.persist_mapping_result()` 写入图数据库

## 维护约定

- 新增能力时优先落到对应职责层，不直接堆到 `runner.py`
- `mapping_pipeline.py` 专注流程控制，不直接承担落库逻辑
- 登录与验证码相关逻辑统一走 `login.py` / `captcha.py`
