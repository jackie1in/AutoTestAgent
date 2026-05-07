#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2025/8/26 11:26
# @Author  : guoxz
# @Site    : 
# @File    : t_wenyu.py
# @Software: PyCharm
# @Description

import json
import logging
import sys

from openai import OpenAI

API_KEY = '-'
BASE_URL = 'http://47.109.179.115:8088/v1'

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def f_stream(thinking=True):
    client = OpenAI(api_key=API_KEY,
                    base_url=BASE_URL,
                    timeout=3600)
    # /think:high
    question = '''
你好
    '''.strip()

    stream = client.chat.completions.create(
        model='GLM-5.1-FP8',  # "wenyu3.0_14B_beta",  # or another reasoning-capable model
        messages=[{"role": "user", "content": question}],
        temperature=0.6,
        # frequency_penalty=0.05,
        stream=True,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": thinking
            }
        }
    )

    reasoning_first = True
    content_first = True
    for chunk in stream:
        if chunk.choices[0].delta.content:
            if content_first:
                sys.stdout.write('\n</think>\n')
                sys.stdout.flush()
                content_first = False
            sys.stdout.write(chunk.choices[0].delta.content)
            sys.stdout.flush()
        if hasattr(chunk.choices[0].delta, "reasoning") and chunk.choices[0].delta.reasoning:
            if reasoning_first:
                reasoning_first = False
                sys.stdout.write('<think>\n')
                sys.stdout.flush()
            sys.stdout.write(f"{chunk.choices[0].delta.reasoning}")
            sys.stdout.flush()


def f_stream_tools(thinking=True):
    client = OpenAI(api_key=API_KEY,
                    base_url=BASE_URL,
                    timeout=3600)

    messages = [{'role': 'system',
                 'content': 'You are an expert search agent. Your goal is to answer the user\'s question efficiently using verifiable online sources.\n\nExecution Rules:\n1. ** Seach Results ** - Each search query returns 10 detailed results or sources. Use as many tool calls as required but do not spam searches.\n2. ** Planning ** - Start by breaking the problem into multiple key sub-questions (as many as required only). If you aren\'t confident about something, break it down into smaller sub-questions.\n2. ** Tool Use ** - You MUST make at least 3 distinct tool calls.\n3. ** Avoid Duplicates ** - If a search fails, pivot your strategy instantly.\n4. ** MAXIMUM 25 Calls ** - You MUST finish within 25 tool calls. Keep a counter in your thought process to track the number of tool calls you have made. If you hit 25, stop searching and derive the best answer immediately.\n5. ** Verification ** - Cross-check critical facts (dates, names, locations) with a second source.\n6. ** Output Format ** - The last line of your output must be EXACTLY: "Final Answer: <The Entity>" (replace <The Entity> with the actual answer).\n7. ** Response Format ** - The response format must be as follows, the angular brackets should be replaced with the actual content:\n  - Thought: <Explain your next step>\n  - Search: <Tool Call>\n  - Observation: <Concise summary of findings>\n  ... (repeat thought, search, observation as needed) ...\n  - Final Answer: <The Entity>, replace <The Entity> with the actual answer.\n'},
                {'role': 'user',
                 'content': 'Given that lead exposure primarily harms a particular organ within the visual system, which exact arterial branch supplies that organ?'}]

    default_tools = [
        {'type': 'function', 'function': {'name': 'web-search', 'description': 'Search the web for a query.',
                                          'parameters': {'type': 'object', 'properties': {
                                              'query': {'type': 'string', 'description': 'Search query.'}},
                                                         'required': ['query']}}}]

    stream = client.chat.completions.create(
        model='GLM-5.1-FP8',  # "wenyu3.0_14B_beta",  # or another reasoning-capable model
        messages=messages,
        tools=default_tools,
        temperature=0.6,
        stream=True,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": thinking
            }
        }
    )

    content = ''
    reasoning_content = ''

    tools = []
    tool = None
    reasoning_first = True
    content_first = True
    for chunk in stream:
        if chunk.choices[0].delta.content:
            if content_first:
                sys.stdout.write('\n</think>\n')
                sys.stdout.flush()
                content_first = False
            content += chunk.choices[0].delta.content
            sys.stdout.write(chunk.choices[0].delta.content)
            sys.stdout.flush()
        if hasattr(chunk.choices[0].delta, "reasoning") and chunk.choices[0].delta.reasoning:
            if reasoning_first:
                sys.stdout.write('<think>\n')
                sys.stdout.flush()
                reasoning_first = False
            reasoning_content += chunk.choices[0].delta.reasoning
            sys.stdout.write(f"{chunk.choices[0].delta.reasoning}")
            sys.stdout.flush()
        if hasattr(chunk.choices[0].delta, "tool_calls") and chunk.choices[0].delta.tool_calls:
            if content_first:
                sys.stdout.write('\n</think>\n')
                sys.stdout.flush()
                content_first = False
            t = chunk.choices[0].delta.tool_calls[0]
            if chunk.choices[0].delta.tool_calls[0].id is not None:
                if tool is not None:
                    tools.append(tool)
                tool = {
                    'id': t.id,
                    'type': t.type,
                    'function': {
                        'name': t.function.name,
                        'arguments': t.function.arguments,
                    },
                }
            else:
                arguments = chunk.choices[0].delta.tool_calls[0].function.arguments
                tool['function']['arguments'] += arguments
    if tool is not None:
        tools.append(tool)

    for tool in tools:
        sys.stdout.write(
            f'\n<tool_call>\n{json.dumps(tool, ensure_ascii=False, indent=2)}\n</tool_call>\n'
        )
        sys.stdout.flush()


def f(thinking=True):
    client = OpenAI(api_key=API_KEY,
                    base_url=BASE_URL,
                    timeout=3600)
    # 一个底面半径4cm，高为9cm的封闭圆柱形容器（容器壁厚度忽略不计）内有两个半径相等的铁球，则铁球半径最大值为多少cm？
    question = '''
你好
        '''.strip()

    rsp = client.chat.completions.create(
        model="GLM-5.1-FP8",  # "wenyu3.0_8B_beta",  # or another reasoning-capable model
        messages=[{"role": "user", "content": question}],
        n=1,
        temperature=0.6,
        stream=False,
        # reasoning_effort='high'
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": thinking
            }
        }
    )

    response = rsp.choices[0].message.content

    finish_reason = rsp.choices[0].finish_reason
    try:
        reasoning = rsp.choices[0].message.model_extra['reasoning']
    except:
        reasoning = None

    num_prompt_tokens = rsp.usage.prompt_tokens
    num_rsp_tokens = rsp.usage.completion_tokens

    logger.info("num_prompt_tokens: %s", num_prompt_tokens)
    logger.info("num_rsp_tokens: %s", num_rsp_tokens)
    logger.info("finish_reason: %s", finish_reason)
    logger.info("%s", "#" * 50)
    logger.info("<think>\n%s\n</think>", reasoning)
    logger.info("%s", response)


def f_tools(thinking=True):
    client = OpenAI(api_key=API_KEY,
                    base_url=BASE_URL,
                    timeout=3600)

    messages = [{'role': 'system',
                 'content': 'You are an expert search agent. Your goal is to answer the user\'s question efficiently using verifiable online sources.\n\nExecution Rules:\n1. ** Seach Results ** - Each search query returns 10 detailed results or sources. Use as many tool calls as required but do not spam searches.\n2. ** Planning ** - Start by breaking the problem into multiple key sub-questions (as many as required only). If you aren\'t confident about something, break it down into smaller sub-questions.\n2. ** Tool Use ** - You MUST make at least 3 distinct tool calls.\n3. ** Avoid Duplicates ** - If a search fails, pivot your strategy instantly.\n4. ** MAXIMUM 25 Calls ** - You MUST finish within 25 tool calls. Keep a counter in your thought process to track the number of tool calls you have made. If you hit 25, stop searching and derive the best answer immediately.\n5. ** Verification ** - Cross-check critical facts (dates, names, locations) with a second source.\n6. ** Output Format ** - The last line of your output must be EXACTLY: "Final Answer: <The Entity>" (replace <The Entity> with the actual answer).\n7. ** Response Format ** - The response format must be as follows, the angular brackets should be replaced with the actual content:\n  - Thought: <Explain your next step>\n  - Search: <Tool Call>\n  - Observation: <Concise summary of findings>\n  ... (repeat thought, search, observation as needed) ...\n  - Final Answer: <The Entity>, replace <The Entity> with the actual answer.\n'},
                {'role': 'user',
                 'content': 'Given that lead exposure primarily harms a particular organ within the visual system, which exact arterial branch supplies that organ?'}]

    default_tools = [
        {'type': 'function', 'function': {'name': 'web-search', 'description': 'Search the web for a query.',
                                          'parameters': {'type': 'object', 'properties': {
                                              'query': {'type': 'string', 'description': 'Search query.'}},
                                                         'required': ['query']}}}]

    rsp = client.chat.completions.create(
        model="GLM-5.1-FP8",  # "wenyu3.0_8B_beta",  # or another reasoning-capable model
        messages=messages,
        tools=default_tools,
        temperature=0.6,
        stream=False,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": thinking
            }
        }
    )

    finish_reason = rsp.choices[0].finish_reason

    response = rsp.choices[0].message.content  # 函数调用这个可能是空

    if 'reasoning' in rsp.choices[0].message.model_extra:
        reasoning_content = rsp.choices[0].message.model_extra['reasoning']
    elif 'reasoning_content' in rsp.choices[0].message.model_extra:
        reasoning_content = rsp.choices[0].message.model_extra['reasoning_content']
    else:
        reasoning_content = None  # 函数调用这个可能是空

    if rsp.choices[0].message.tool_calls is not None:
        tool_calls = []
        for t in rsp.choices[0].message.tool_calls:
            tool_calls.append({
                'id': t.id,
                'type': t.type,
                'function': {
                    'name': t.function.name,
                    'arguments': t.function.arguments,
                },
            })
        if not tool_calls:
            tool_calls = None
    else:
        tool_calls = None

    num_prompt_tokens = rsp.usage.prompt_tokens
    num_rsp_tokens = rsp.usage.completion_tokens

    logger.info("num_prompt_tokens: %s", num_prompt_tokens)
    logger.info("num_rsp_tokens: %s", num_rsp_tokens)
    logger.info("finish_reason: %s", finish_reason)
    logger.info("%s", "#" * 50)
    logger.info("<think>\n%s\n</think>", reasoning_content)

    if response is not None:
        logger.info("%s", response)

    if tool_calls is not None:
        for t in tool_calls:
            logger.info(
                "<tool_call>\n%s\n</tool_call>", json.dumps(t, ensure_ascii=False, indent=2)
            )


if __name__ == '__main__':
    # f()
    f_stream_tools()
    # f_tools()

    """
curl http://47.109.179.115:8088/v1/chat/completions \
-H "content-type: application/json" \
-H "Authorization: Bearer -" \
-d '{"model":"GLM-5.1-FP8","messages":[{"role":"user","content":"Hi"}]}'
    """
