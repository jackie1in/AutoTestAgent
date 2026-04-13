# GLM API 对照表

官方文档: https://docs.bigmodel.cn/cn/guide/capabilities/thinking-mode

## 参数映射

| 官方 API 参数 | ChatGLM 参数 | 类型 | 说明 |
|-------------|-------------|------|------|
| `model` | `model` | `str` | 模型名称，如 "glm-5", "glm-4.7" |
| `thinking.type` | `thinking` | `"enabled" \| "disabled" \| None` | 思考模式开关 |
| `thinking.clear_thinking` | `clear_thinking` | `bool` | False=保留思考，True=清除思考 |
| `messages` | `messages` | `list[BaseMessage]` | 消息列表 |
| `temperature` | `temperature` | `float` | 温度参数 |
| `max_tokens` | `max_tokens` | `int` | 最大生成 token 数 |

## 响应字段映射

| 官方 API 响应 | ChatInvokeCompletion 字段 | 说明 |
|-------------|--------------------------|------|
| `choices[0].message.content` | `completion` | 生成的内容 |
| `choices[0].message.reasoning_content` | `thinking` | 思考/推理内容 |
| `usage.prompt_tokens` | `usage.prompt_tokens` | 提示 token 数 |
| `usage.completion_tokens` | `usage.completion_tokens` | 生成 token 数 |
| `usage.total_tokens` | `usage.total_tokens` | 总 token 数 |

## 代码对比

### 官方 Python SDK 示例

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://api.z.ai/api/paas/v4/",
)

# 配置 thinking
response = client.chat.completions.create(
    model="glm-5",
    messages=[...],
    extra_body={
        "thinking": {
            "type": "enabled",
            "clear_thinking": False  # 保留思考
        }
    }
)

# 提取 reasoning_content
reasoning = ""
for chunk in response:
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        reasoning += delta.reasoning_content
```

### AutoTestAgent ChatGLM

```python
from graph_agent.llm.glm_chat import ChatGLM

llm = ChatGLM(
    model="glm-5",
    api_key="YOUR_API_KEY",
    base_url="https://api.z.ai/api/paas/v4/",
    thinking="enabled",      # 对应 thinking.type
    clear_thinking=False,    # 对应 thinking.clear_thinking
)

# 调用 API（非流式）
result = await llm.ainvoke(messages)

# 直接访问 reasoning_content
if result.thinking:
    print(result.thinking)
```

## Thinking 模式对比

### 1. 交错式思考 (Interleaved Thinking)

```python
# 官方 API
extra_body = {
    "thinking": {"type": "enabled"}
}

# ChatGLM
llm = ChatGLM(
    model="glm-5",
    thinking="enabled",
)
```

### 2. 保留式思考 (Preserved Thinking)

```python
# 官方 API
extra_body = {
    "thinking": {
        "type": "enabled",
        "clear_thinking": False  # 保留
    }
}

# ChatGLM
llm = ChatGLM(
    model="glm-5",
    thinking="enabled",
    clear_thinking=False,  # 保留
)
```

### 3. 禁用思考

```python
# 官方 API
extra_body = {
    "thinking": {"type": "disabled"}
}

# ChatGLM
llm = ChatGLM(
    model="glm-5",
    thinking="disabled",
)
```

## 多轮对话保留 thinking

### 官方方式

```python
# 第一轮
response1 = client.chat.completions.create(...)
reasoning1 = extract_reasoning(response1)
content1 = extract_content(response1)

# 第二轮 - 手动添加 reasoning_content
messages.append({
    "role": "assistant",
    "content": content1,
    "reasoning_content": reasoning1  # 保留思考
})
messages.append({"role": "user", "content": "..."})

response2 = client.chat.completions.create(...)
```

### ChatGLM 方式

```python
llm = ChatGLM(
    model="glm-5",
    thinking="enabled",
    clear_thinking=False,  # 服务端自动保留
)

# 第一轮
result1 = await llm.ainvoke(messages)
reasoning1 = result1.thinking
content1 = result1.completion

# 第二轮 - 同样需要手动添加 reasoning_content
from browser_use.llm.messages import AssistantMessage

messages.extend([
    {"role": "assistant", "content": content1, "reasoning_content": reasoning1},
    UserMessage(content="..."),
])

result2 = await llm.ainvoke(messages)
```

## 关键差异

| 特性 | 官方 SDK | ChatGLM |
|-----|---------|---------|
| 流式输出 | ✅ 支持 | ❌ 暂不支持（使用非流式） |
| 自动重试 | ❌ 需自行实现 | ✅ 内置 5 次重试 |
| 结构化输出 | ⚠️ 需手动解析 JSON | ✅ Pydantic 模型直接验证 |
| Token 统计 | ✅ 支持 | ✅ 支持 |
| 错误处理 | ⚠️ 需自行处理 | ✅ 统一异常类 |

## 完全等价示例

### 场景：带工具调用的思考

官方文档示例简化版：
```python
# 官方
response = client.chat.completions.create(
    model="glm-5",
    messages=messages,
    tools=tools,
    extra_body={
        "thinking": {
            "type": "enabled",
            "clear_thinking": False
        }
    }
)
reasoning = response.choices[0].message.reasoning_content
```

ChatGLM 等价实现：
```python
from graph_agent.llm.glm_chat import ChatGLM
from browser_use.llm.messages import SystemMessage, UserMessage

llm = ChatGLM(
    model="glm-5",
    api_key="...",
    thinking="enabled",
    clear_thinking=False,
)

messages = [
    SystemMessage(content="You are an assistant."),
    UserMessage(content="What's the weather?"),
]

result = await llm.ainvoke(messages)
reasoning = result.thinking  # 等价于官方 reasoning_content
content = result.completion  # 等价于官方 content
```
