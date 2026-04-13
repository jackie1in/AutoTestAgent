# GLM-5 快速参考

基于官方文档: https://docs.bigmodel.cn/cn/guide/develop/langchain/introduction

## 1. 最简配置

```python
from graph_agent.llm import ChatGLM

llm = ChatGLM(
    model="glm-5",
    api_key="your-zhipu-api-key",
    base_url="https://open.bigmodel.cn/api/paas/v4/",  # 官方推荐地址
)
```

## 2. 环境变量配置

```bash
# .env
LLM_MODEL=glm-5
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
LLM_API_KEY=your-zhipu-api-key
```

```python
from graph_agent.llm import get_llm

llm = get_llm()  # 自动读取环境变量
```

## 3. 消息类型（多种选择）

### 方式 1: browser-use 消息（推荐）

```python
from browser_use.llm.messages import SystemMessage, UserMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    UserMessage(content="What is AI?"),
]

result = await llm.ainvoke(messages)
```

### 方式 2: LangChain 消息

```python
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is AI?"),
]

result = await llm.ainvoke(messages)  # ✅ 自动转换
```

### 方式 3: 原始 dict

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is AI?"},
]

result = await llm.ainvoke(messages)
```

### 方式 4: 混合使用

```python
from browser_use.llm.messages import SystemMessage
from langchain_core.messages import HumanMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),  # browser-use
    HumanMessage(content="What is AI?"),                   # LangChain
]

result = await llm.ainvoke(messages)  # ✅ 全部支持
```

## 4. 结构化输出

```python
from pydantic import BaseModel

class Output(BaseModel):
    answer: str
    confidence: float

result = await llm.ainvoke(messages, output_format=Output)
output = result.completion  # Output 实例
print(output.answer)
```

## 5. 参数速查

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `model` | `str` | 必填 | 模型名称，如 "glm-5" |
| `api_key` | `str` | 必填 | Zhipu AI API Key |
| `base_url` | `str` | `https://open.bigmodel.cn/api/paas/v4/` | API 地址 |
| `temperature` | `float` | `0.6` | 温度 |
| `max_tokens` | `int` | `4096` | 最大 token |

## 6. 与 LangChain 对比

| 功能 | LangChain (官方) | AutoTestAgent |
|-----|-----------------|---------------|
| 导入 | `from langchain_openai import ChatOpenAI` | `from graph_agent.llm import ChatGLM` |
| 实例化 | `ChatOpenAI(model="glm-5", openai_api_key=..., openai_api_base=...)` | `ChatGLM(model="glm-5", api_key=..., base_url=...)` |
| 调用 | `llm(messages)` | `await llm.ainvoke(messages)` |
| 消息 | LangChain 类型 | ✅ LangChain + browser-use + dict |
| Base URL | `https://open.bigmodel.cn/api/paas/v4/` | `https://open.bigmodel.cn/api/paas/v4/` |

## 7. 官方文档参考

- [LangChain 集成指南](https://docs.bigmodel.cn/cn/guide/develop/langchain/introduction)
- [API 文档](https://docs.bigmodel.cn/)
