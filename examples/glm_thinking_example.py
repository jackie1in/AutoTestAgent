"""GLM-5 Thinking Mode 使用示例

对应文档: https://docs.bigmodel.cn/cn/guide/capabilities/thinking-mode

功能演示:
1. 启用/禁用 Thinking
2. 保留式思考 (Preserved Thinking)
3. 提取 reasoning_content
4. 结构化输出 (Structured Output)
"""

import asyncio
import os

from pydantic import BaseModel


# 定义结构化输出模型
class AgentOutput(BaseModel):
    """Agent 输出格式"""
    evaluation: str
    memory: str
    next_action: str


async def example_1_basic_thinking():
    """示例 1: 基础 Thinking 模式"""
    from graph_agent.llm.zhipu.chat import ChatZhiPu

    print("=== 示例 1: 基础 Thinking 模式 ===")

    # 创建启用 thinking 的客户端
    llm = ChatZhiPu(
        model="glm-5",
        api_key=os.getenv("GLM_API_KEY"),
        base_url="https://api.z.ai/api/paas/v4/",
        thinking="enabled",  # 启用 thinking
    )

    # 检查 extra_body 是否正确构建
    extra_body = llm._build_extra_body()
    print(f"extra_body: {extra_body}")
    # 输出: {'thinking': {'type': 'enabled', 'clear_thinking': True}}

    # 调用 API
    from browser_use.llm.messages import SystemMessage, UserMessage

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What is 2+2?"),
    ]

    result = await llm.ainvoke(messages)

    print(f"Content: {result.completion}")
    if result.thinking:
        print(f"Thinking: {result.thinking[:200]}...")
    if result.usage:
        print(f"Tokens: {result.usage.total_tokens}")


async def example_2_preserved_thinking():
    """示例 2: 保留式思考 (Preserved Thinking)

n    跨轮次保留 reasoning_content，保持推理连贯性
    对应文档中的: clear_thinking: False
    """
    from graph_agent.llm.zhipu.chat import ChatZhiPu

    print("\n=== 示例 2: 保留式思考 (Preserved Thinking) ===")

    llm = ChatZhiPu(
        model="glm-5",
        api_key=os.getenv("GLM_API_KEY"),
        thinking="enabled",
        clear_thinking=False,  # False = 保留思考 (Preserved Thinking)
    )

    extra_body = llm._build_extra_body()
    print(f"extra_body: {extra_body}")
    # 输出: {'thinking': {'type': 'enabled', 'clear_thinking': False}}

    print("✓ 配置完成: reasoning_content 将在多轮对话中保留")


async def example_3_disable_thinking():
    """示例 3: 禁用 Thinking (减少 token 消耗)"""
    from graph_agent.llm.zhipu.chat import ChatZhiPu

    print("\n=== 示例 3: 禁用 Thinking ===")

    llm = ChatZhiPu(
        model="glm-5",
        api_key=os.getenv("GLM_API_KEY"),
        thinking="disabled",  # 禁用 thinking
    )

    extra_body = llm._build_extra_body()
    print(f"extra_body: {extra_body}")
    # 输出: {'thinking': {'type': 'disabled'}}

    print("✓ 配置完成: 已禁用 thinking，减少约 20-30% token 消耗")


async def example_4_structured_output():
    """示例 4: 结构化输出 + Thinking"""
    from graph_agent.llm.zhipu.chat import ChatZhiPu

    print("\n=== 示例 4: 结构化输出 + Thinking ===")

    llm = ChatZhiPu(
        model="glm-5",
        api_key=os.getenv("GLM_API_KEY"),
        thinking="enabled",
        clear_thinking=False,
    )

    from browser_use.llm.messages import SystemMessage, UserMessage

    messages = [
        SystemMessage(content="You are an agent that analyzes web pages."),
        UserMessage(content="Analyze this page: <button>Login</button>"),
    ]

    # 使用 Pydantic 模型作为输出格式
    result = await llm.ainvoke(messages, output_format=AgentOutput)

    # result.completion 是 AgentOutput 实例
    output = result.completion
    print(f"Evaluation: {output.evaluation}")
    print(f"Memory: {output.memory}")
    print(f"Next Action: {output.next_action}")

    if result.thinking:
        print(f"\nThinking Process:\n{result.thinking[:500]}...")


async def example_5_multi_turn_with_thinking():
    """示例 5: 多轮对话 + 保留式思考

n    模拟文档中的多轮 tool calling 示例
    """
    from graph_agent.llm.zhipu.chat import ChatZhiPu

    print("\n=== 示例 5: 多轮对话 + 保留式思考 ===")

    llm = ChatZhiPu(
        model="glm-5",
        api_key=os.getenv("GLM_API_KEY"),
        thinking="enabled",
        clear_thinking=False,  # 关键：保留思考以保持连贯性
    )

    from browser_use.llm.messages import SystemMessage, UserMessage

    # 第一轮
    print("\n--- Round 1 ---")
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What's the weather like in Beijing?"),
    ]

    result1 = await llm.ainvoke(messages)
    content1 = result1.completion
    thinking1 = result1.thinking or ""

    print(f"Content: {content1}")
    print(f"Thinking: {thinking1[:200]}...")

    # 第二轮 - 保留上一轮思考
    print("\n--- Round 2 ---")
    # 在实际应用中，你需要手动将 reasoning_content 添加到消息历史
    # GLM 会自动在服务端保留，但客户端也可以访问

    messages.extend([
        {"role": "assistant", "content": content1, "reasoning_content": thinking1},
        UserMessage(content="What about Shanghai?"),
    ])

    result2 = await llm.ainvoke(messages)
    print(f"Content: {result2.completion}")
    if result2.thinking:
        print(f"Thinking: {result2.thinking[:200]}...")


async def example_6_code_config():
    """示例 6: 使用代码参数控制 Thinking"""
    from graph_agent.llm import get_llm

    print("\n=== 示例 6: 使用代码参数控制 Thinking ===")

    # 设置环境变量（基础配置）
    os.environ["LLM_API_KEY"] = os.getenv("GLM_API_KEY", "your-api-key")
    os.environ["LLM_BASE_URL"] = "https://api.z.ai/api/paas/v4/"
    os.environ["LLM_MODEL"] = "glm-5"

    # 通过 use_thinking 参数控制 thinking 模式
    # 启用 thinking
    llm = get_llm(use_thinking=True)
    print(f"✓ Provider: {type(llm).__name__}")
    print(f"✓ Model: {llm.model}")
    if hasattr(llm, "thinking"):
        print(f"✓ Thinking: {llm.thinking}")
    
    # 禁用 thinking（默认行为，节省 20-30% token）
    llm_no_think = get_llm(use_thinking=False)
    print("\n✓ Disabled thinking mode")
    if hasattr(llm_no_think, "thinking"):
        print(f"✓ Thinking: {llm_no_think.thinking}")


async def main():
    """运行所有示例"""
    print("GLM-5 Thinking Mode Examples")
    print("=" * 50)
    print()

    # 检查 API key
    if not os.getenv("GLM_API_KEY"):
        print("⚠️  Warning: GLM_API_KEY not set in environment")
        print("Set it with: export GLM_API_KEY=your-api-key")
        print()

    # 运行示例
    await example_1_basic_thinking()
    await example_2_preserved_thinking()
    await example_3_disable_thinking()
    await example_4_structured_output()
    await example_5_multi_turn_with_thinking()
    await example_6_code_config()

    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
