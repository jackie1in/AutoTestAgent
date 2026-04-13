# GLM-5 Thinking Mode 使用指南

AutoTestAgent 提供专门的 `ChatGLM` provider 来支持 GLM-5 的全部功能，包括 thinking/reasoning 模式。

## 架构

```
graph_agent/llm/
├── __init__.py           # 主入口：get_llm(), ChatGLM, ainvoke_structured
├── glm_chat.py           # ChatGLM provider（原生 GLM 支持）
└── glm_with_thinking.py  # ChatOpenRouterWithThinking（OpenRouter 兼容）
```

## Provider 选择

| 使用场景 | Provider | 特点 |
|---------|----------|------|
| **推荐** Zhipu 直接 API | `ChatGLM` | 完整 thinking 控制、最低延迟 |
| OpenRouter 访问 GLM | `ChatOpenRouterWithThinking` | 兼容性好、thinking 只读 |
| 其他 OpenAI 兼容服务 | `ChatOpenAI` | 标准兼容 |

## 配置

### 方式 1：Zhipu 直接 API（推荐）

使用原生 `ChatGLM` provider，支持完整的 thinking 控制：

```bash
LLM_MODEL=glm-5
LLM_BASE_URL=https://api.z.ai/api/paas/v4/
LLM_API_KEY=your-zhipu-api-key
```

然后在代码中控制 thinking 模式：

```python
from graph_agent.llm import get_llm

# 启用 thinking 模式
llm = get_llm(use_thinking=True)

# 禁用 thinking 模式（默认，节省 token）
llm = get_llm(use_thinking=False)
```

### 方式 2：OpenRouter

通过 OpenRouter 访问 GLM-5，thinking 内容会被提取但无法通过参数控制：

```bash
LLM_MODEL=glm-5
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-v1-xxxxx
# thinking 模式由 GLM-5 默认行为决定
```

## 在代码中使用

### 使用 get_llm()（自动选择 Provider）

```python
from graph_agent.llm import get_llm, ainvoke_structured
from graph_agent.cartography.react_schema import AgentOutput

# 使用 use_thinking 参数控制 thinking 模式
llm = get_llm(use_thinking=True)

result = await ainvoke_structured(
    llm,
    system_prompt="...",
    user_prompt="...",
    output_format=AgentOutput
)
```

### 直接使用 ChatGLM

```python
from graph_agent.llm import ChatGLM

llm = ChatGLM(
    model="glm-5",
    api_key="your-api-key",
    thinking="enabled",           # 启用 thinking
    clear_thinking=False,          # 保留 thinking 跨轮次
    temperature=0.6,
)

# 调用 API
result = await llm.ainvoke(messages, output_format=AgentOutput)

# 访问 thinking
if result.thinking:
    print(f"Thinking: {result.thinking}")
```

### 使用 Helper 函数

```python
from graph_agent.llm import create_glm_llm

llm = create_glm_llm(
    model="glm-5",
    api_key="your-api-key",  # 或从环境变量 GLM_API_KEY 读取
    enable_thinking=True,
    preserve_thinking=True,   # 跨轮次保留 thinking
)
```

## Thinking 模式

### 1. 启用/禁用 Thinking

```python
llm = ChatGLM(
    model="glm-5",
    thinking="enabled",   # "enabled" 或 "disabled"
)
```

### 2. 保留式思考（Preserved Thinking）

跨轮次保留 reasoning_content，保持推理连贯性：

```python
llm = ChatGLM(
    model="glm-5",
    thinking="enabled",
    clear_thinking=False,  # False = 保留 thinking
)
```

### 3. 查看 Thinking 内容

```bash
# 设置日志级别
export LOG_LEVEL=INFO
```

日志输出：
```
INFO: [GLM-5 Thinking] Let me analyze the current page state. I can see there are 
5 interactive elements on this page. The first element [0] appears to be a login 
button, which I should click to explore the authentication flow...
```

## ChatGLM Provider 特性

### 完整参数支持

```python
ChatGLM(
    model="glm-5",                    # 模型名称
    api_key="your-key",               # API 密钥
    base_url="https://api.z.ai/api/paas/v4/",
    temperature=0.6,                  # 温度
    top_p=None,                       # Top-p 采样
    max_tokens=4096,                  # 最大 token 数
    thinking="enabled",               # Thinking 模式
    clear_thinking=False,             # 是否清除 thinking
    timeout=60.0,                     # 超时时间
    max_retries=5,                    # 重试次数
)
```

### 结构化输出

支持 Pydantic schema 的 structured output：

```python
from pydantic import BaseModel

class AgentOutput(BaseModel):
    action: str
    params: dict

result = await llm.ainvoke(messages, output_format=AgentOutput)
# result.completion 是 AgentOutput 实例
```

### Token 使用统计

```python
result = await llm.ainvoke(messages)
if result.usage:
    print(f"Prompt tokens: {result.usage.prompt_tokens}")
    print(f"Completion tokens: {result.usage.completion_tokens}")
    print(f"Total tokens: {result.usage.total_tokens}")
```

## 故障排除

### Provider 选择不正确

检查 `LLM_BASE_URL`：
- `https://api.z.ai/api/paas/v4/` → 使用 `ChatGLM`
- `https://openrouter.ai/api/v1` → 使用 `ChatOpenRouterWithThinking`

### Thinking 不显示

1. **检查模型**：确保使用 `glm-5` 或支持 thinking 的模型
2. **检查日志级别**：`export LOG_LEVEL=INFO`
3. **检查配置**：
   - Zhipu API: `thinking="enabled"`
   - OpenRouter: thinking 由模型默认行为决定

### API 错误

```
APIStatusError: 400 - Invalid parameter
```

- 确保使用正确的 `base_url`
- 检查 `api_key` 是否有效
- 确认模型名称正确

## 参考

- [GLM-5 Thinking Mode 文档](https://docs.bigmodel.cn/cn/guide/capabilities/thinking-mode)
- [Zhipu AI API 文档](https://docs.bigmodel.cn/)
- [OpenRouter 文档](https://openrouter.ai/docs)
