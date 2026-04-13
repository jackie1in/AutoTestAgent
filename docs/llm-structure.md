# LLM 模块结构

## 目录结构

```
graph_agent/llm/
├── __init__.py          # 公共 API 导出
├── llm.py               # 工厂函数 (get_llm, create_glm_llm)
├── utils.py             # 工具函数 (ainvoke_structured)
├── views.py             # 视图模型
└── glm/
    ├── __init__.py
    └── chat.py          # ChatGLM 实现

# ChatOpenRouter 由 browser-use 直接提供
```

## ChatGLM 实现

参考 `browser-use/llm/openai/chat.py` 的实现方式：

### 核心设计

```python
@dataclass
class ChatGLM(BaseChatModel):
    # 标准 OpenAI 参数
    model: str
    temperature: float | None = 0.6
    max_tokens: int | None = 4096
    
    # GLM 特有的 thinking 控制
    thinking: Literal["enabled", "disabled"] | None = None
    clear_thinking: bool = True
    
    # 客户端参数
    api_key: str | None = None
    base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
```

### 关键方法

| 方法 | 说明 |
|-----|------|
| `_get_client_params()` | 构建 AsyncOpenAI 客户端参数 |
| `_build_model_params()` | 构建 API 调用参数 |
| `_build_extra_body()` | 构建 GLM 特有的 extra_body (thinking) |
| `ainvoke()` | 调用 GLM API |

### extra_body 格式

```python
# Thinking enabled
{
    "thinking": {
        "type": "enabled",
        "clear_thinking": false  # Preserved Thinking
    }
}

# Thinking disabled
{
    "thinking": {
        "type": "disabled"
    }
}
```

### 使用方式

```python
from graph_agent.llm import ChatGLM

# 基础使用
llm = ChatGLM(model="glm-5", api_key="your-key")

# 启用 thinking
llm = ChatGLM(
    model="glm-5",
    api_key="your-key",
    thinking="enabled",
    clear_thinking=False,
)

# 工厂函数
from graph_agent.llm import get_llm
llm = get_llm()  # 自动根据环境变量创建
```

## References

- LangChain: https://docs.bigmodel.cn/cn/guide/develop/langchain/introduction
- Thinking: https://docs.bigmodel.cn/cn/guide/capabilities/thinking-mode
