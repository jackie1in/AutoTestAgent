# UI Automation Testing Agent

基于 [Browser-Use](https://docs.browser-use.com/) 的智能 UI 自动化测试工具，支持自然语言录制、智能回放和自动修正。

## 特性

### 核心功能

- **🎬 智能录制** - 使用自然语言描述任务，AI 自动执行并录制操作
- **▶️ 智能回放** - 回放录制的测试用例，自动适应页面变化
- **🔧 自动修正** - 当元素位置变化时，利用 AI 思考过程自动定位并修正
- **📝 用例管理** - 创建、查看、搜索、删除测试用例
- **🤖 多模型支持** - 支持 OpenAI、OpenRouter、Anthropic 等多种 LLM

### 录制增强

录制时保存 AI 的完整思考过程，用于智能回放：

| 字段 | 说明 |
|------|------|
| `thinking` | AI 对当前状态的评估 |
| `goal` | AI 的下一步目标意图 |
| `memory` | AI 的上下文记忆 |
| `element_description` | 元素特征 (id, name, class, xpath) |

### 自动修正原理

```
录制: 点击 index=108 的按钮
         ↓ 保存元素特征
      id='su', value='百度一下', xpath='...'

回放: index 变成 120
         ↓ 使用元素特征重新定位
      找到 id='su' 的按钮并点击
```

## 安装

```bash
# 使用 uv (推荐) https://docs.astral.sh/uv/getting-started/installation/
uv sync

# 或使用 pip
pip install -e .
```

## 配置

1. 复制环境变量模板：
```bash
cp .env.example .env
```

2. 配置 LLM (选择一种):

**OpenAI:**
```bash
OPENAI_API_KEY=sk-xxxxx
```

**OpenRouter (推荐，支持多种模型):**
```bash
LLM_MODEL=google/gemini-2.5-pro-preview
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-v1-xxxxx
```

## 使用方法

### 命令行

```bash
# 录制测试用例
uv run python main.py record "在百度搜索 ai skills"
uv run python main.py record -p "点击登录" -u "https://example.com" -n "登录测试"

# 回放测试用例 (带自动修正)
uv run python main.py replay                    # 交互式选择
uv run python main.py replay 20240101_120000    # 指定用例

# 管理测试用例
uv run python main.py list                      # 列出所有用例
uv run python main.py view 20240101_120000      # 查看详情
uv run python main.py delete 20240101_120000    # 删除用例
```

### 命令参数

**record 命令:**
| 参数 | 说明 |
|------|------|
| `-p, --prompt` | 任务描述 |
| `-u, --url` | 起始 URL |
| `-n, --name` | 用例名称 |
| `--headless` | 无头模式 (不显示浏览器) |

### 编程接口

```python
import asyncio
from ui_test_agent import UITestRecorder, UITestPlayer, LLMConfig

async def main():
    # 配置 LLM
    llm_config = LLMConfig(
        model="google/gemini-2.5-pro-preview",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-v1-xxxxx"
    )
    
    # 录制
    recorder = UITestRecorder(headless=False, llm_config=llm_config)
    test_case = await recorder.record(
        task="在百度搜索 ai skills",
        start_url="https://www.baidu.com",
        test_name="百度搜索测试"
    )
    
    # 回放 (带自动修正)
    player = UITestPlayer(llm_config=llm_config)
    results = await player.replay(test_case.id, auto_correct=True)
    
    print(f"回放结果: {'成功' if results['success'] else '失败'}")

asyncio.run(main())
```

## 测试用例格式

测试用例保存在 `test_cases/` 目录，格式为 JSON：

```json
{
  "id": "20240101_120000",
  "name": "百度搜索测试",
  "description": "在百度搜索 ai skills",
  "start_url": "https://www.baidu.com",
  "actions": [
    {
      "action_type": "type",
      "action_name": "input",
      "parameters": {
        "index": 4,
        "text": "ai skills",
        "_element": {
          "attributes": {"id": "kw", "name": "wd", "class": "s_ipt"},
          "x_path": "html/body/div/div/div[3]/div/div[1]/form/span[1]/input"
        }
      },
      "thinking": "Successfully navigated to the Baidu homepage.",
      "goal": "Input 'ai skills' into the search bar and click the search button.",
      "memory": "I am on the Baidu homepage.",
      "element_description": "input id='kw' name='wd' class='s_ipt'"
    }
  ],
  "metadata": {
    "task": "在百度搜索 ai skills",
    "is_successful": true
  }
}
```

## 支持的操作类型

| 类型 | 说明 |
|------|------|
| `navigate` | 导航到 URL |
| `click` | 点击元素 |
| `type` | 输入文本 |
| `scroll` | 滚动页面 |
| `wait` | 等待 |
| `send_keys` | 发送按键 (Enter, Tab 等) |
| `select` | 选择下拉选项 |
| `upload` | 上传文件 |
| `extract` | 提取内容 |
| `go_back` | 返回上一页 |
| `go_forward` | 前进 |
| `refresh` | 刷新页面 |

## 项目结构

```
advanced/
├── main.py              # CLI 入口
├── ui_test_agent.py     # 核心模块
├── pyproject.toml       # 项目配置
├── .env                 # 环境变量 (需创建)
├── .env.example         # 环境变量模板
├── test_cases/          # 测试用例存储目录
└── replay_reports/      # 回放失败报告目录
```

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     UITestRecorder                          │
├─────────────────────────────────────────────────────────────┤
│  • 接收自然语言任务描述                                      │
│  • 调用 Browser-Use Agent 执行任务                          │
│  • 通过 on_step_end hook 捕获每个操作                       │
│  • 保存 AI 思考过程 (thinking/goal/memory)                  │
│  • 保存元素特征 (id/name/class/xpath)                       │
│  • 输出 TestCase JSON 文件                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      UITestPlayer                           │
├─────────────────────────────────────────────────────────────┤
│  • 加载 TestCase 文件                                       │
│  • 生成带元素描述的回放任务                                  │
│  • 执行回放，监控每步结果                                    │
│  • 失败时利用 thinking/goal 生成修正任务                     │
│  • 利用 element_description 重新定位元素                     │
│  • 保存失败报告供调试                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 待优化项 (Roadmap)

### 🔴 高优先级

#### 1. 截图功能
每一步操作保存截图，用于可视化调试和对比。

```python
# 计划实现
@dataclass
class RecordedAction:
    before_screenshot: Optional[str] = None  # 操作前截图 (base64)
    after_screenshot: Optional[str] = None   # 操作后截图
```

**价值**: 直观查看每一步的页面状态，快速定位问题。

#### 2. HTML 测试报告
生成美观的 HTML 报告，包含：
- 测试概览 (成功/失败统计)
- 每步操作详情和截图
- 错误信息和修正记录
- 执行时间分析

```bash
# 计划命令
uv run python main.py report 20240101_120000 --output report.html
```

#### 3. 断言/验证机制
支持在录制时添加验证点：

```python
# 计划实现
assertions = [
    {"type": "text_exists", "value": "搜索结果"},
    {"type": "element_visible", "selector": "#result"},
    {"type": "url_contains", "value": "/search"},
]
```

**验证类型**:
- `text_exists` - 页面包含指定文本
- `element_visible` - 元素可见
- `element_count` - 元素数量检查
- `url_contains` - URL 包含指定字符串
- `title_equals` - 页面标题匹配

### 🟡 中优先级

#### 4. 导出为标准脚本
将录制的测试用例导出为可独立运行的脚本：

```bash
# 计划命令
uv run python main.py export 20240101_120000 --format playwright
uv run python main.py export 20240101_120000 --format selenium
```

**支持格式**:
- Playwright (Python)
- Selenium (Python)
- Puppeteer (JavaScript)

#### 5. 参数化测试 (数据驱动)
支持使用不同数据运行同一测试：

```python
# 计划实现
test_data = [
    {"search_term": "ai skills", "expected": "人工智能"},
    {"search_term": "machine learning", "expected": "机器学习"},
]

await player.replay_with_data(test_id, test_data)
```

#### 6. 智能等待策略
替代固定等待，使用智能等待：

```python
# 计划实现
wait_strategies = {
    "wait_for_element": "#search-results",
    "wait_for_url": "/search",
    "wait_for_text": "搜索结果",
    "timeout": 30,
}
```

### 🟢 低优先级

#### 7. 并行测试执行
同时运行多个测试用例：

```bash
# 计划命令
uv run python main.py batch-run --parallel 3 --tag smoke
```

#### 8. 测试标签和过滤
支持按标签组织和运行测试：

```python
test_case.tags = ["smoke", "login", "critical"]

# 只运行 smoke 测试
await player.run_by_tag("smoke")
```

#### 9. 环境配置管理
支持不同测试环境：

```bash
# 计划配置
environments:
  dev:
    base_url: https://dev.example.com
  staging:
    base_url: https://staging.example.com
  prod:
    base_url: https://example.com
```

#### 10. CI/CD 集成
提供 GitHub Actions / GitLab CI 模板：

```yaml
# .github/workflows/ui-test.yml
- name: Run UI Tests
  run: uv run python main.py batch-run --headless --report
```

#### 11. Web UI 管理界面
提供可视化管理界面：
- 测试用例列表和详情
- 截图对比查看
- 实时执行日志
- 报告查看和下载

#### 12. 视频录制
录制完整的测试执行视频：

```python
recorder = UITestRecorder(record_video=True)
# 输出 .mp4 文件
```

---

## 环境要求

- Python 3.11+
- Browser-Use 0.11.8+
- Chrome/Chromium 浏览器

## 许可证

MIT

## 贡献

欢迎提交 Issue 和 Pull Request！

## 相关链接

- [Browser-Use 文档](https://docs.browser-use.com/)
- [OpenRouter](https://openrouter.ai/)
