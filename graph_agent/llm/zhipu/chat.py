"""GLM (Zhipu AI) LLM provider.

Reference: https://docs.bigmodel.cn/cn/guide/develop/langchain/introduction
Thinking: https://docs.bigmodel.cn/cn/guide/capabilities/thinking-mode

Uses OpenAI SDK with GLM's OpenAI-compatible API.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast, overload

import httpx
from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import (
    BaseMessage,
    ContentPartTextParam,
    SystemMessage,
    UserMessage,
)
from browser_use.llm.openai.serializer import OpenAIMessageSerializer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel, ValidationError

from graph_agent.llm.utils import get_format_instructions

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
Thinking = Literal["enabled", "disabled"]


def _looks_like_schema_object(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    keys = set(value.keys())
    return {"properties", "required", "type"}.issubset(keys) and not {
        "key",
        "confidence",
        "summary",
    }.issubset(keys)


@dataclass
class ChatZhiPu(BaseChatModel):
    """ZhiPu provider using OpenAI-compatible API.

    Supports GLM-specific features:
    - Thinking mode control via extra_body
    - reasoning_content extraction
    - Structured output via format_instructions in prompt

    Example:
        llm = ChatZhiPu(
            model="glm-5",
            api_key="your-api-key",
            thinking="enabled",
            clear_thinking=False,
        )
    """

    # Model configuration
    model: str

    # Model params (standard OpenAI)
    temperature: float | None = 0.2
    max_tokens: int | None = 4096
    top_p: float | None = None

    # GLM-specific: Thinking mode
    # WARNING: thinking='enabled' causes timeouts with long prompts
    thinking: Thinking | None = "disabled"
    clear_thinking: bool = True  # False = Preserved Thinking

    # Client params
    api_key: str | None = None
    base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    timeout: float | httpx.Timeout | None = 3600  # Increased for thinking mode
    max_retries: int = 3
    http_client: httpx.AsyncClient | None = None

    stream: bool = False

    _client: AsyncOpenAI | None = None

    @property
    def provider(self) -> str:
        return "zhipu"

    @property
    def name(self) -> str:
        return str(self.model)

    def _get_client_params(self) -> dict[str, Any]:
        """Prepare client parameters."""
        params: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "max_retries": self.max_retries,
        }
        if self.timeout is not None:
            params["timeout"] = self.timeout
        if self.http_client is not None:
            params["http_client"] = self.http_client
        return params

    def get_client(self) -> AsyncOpenAI:
        """Get or create AsyncOpenAI client."""
        if self._client is not None:
            return self._client
        self._client = AsyncOpenAI(**self._get_client_params())
        return self._client

    def _build_extra_body(self) -> dict[str, Any] | None:
        """Build extra_body with GLM-specific parameters."""
        extra: dict[str, Any] = {}

        if self.thinking is not None:
            if self.model.lower().startswith("glm-5.1"):
                extra["chat_template_kwargs"] = {
                    "enable_thinking": self.thinking == "enabled"
                }

            else:
                thinking_dict: dict[str, Thinking | bool] = {"type": self.thinking}
                if self.thinking == "enabled":
                    thinking_dict["clear_thinking"] = self.clear_thinking
                extra["thinking"] = thinking_dict

        return extra if extra else None

    def _build_model_params(self) -> dict[str, Any]:
        """Build model parameters for API call."""
        params: dict[str, Any] = {}

        if self.temperature is not None:
            params["temperature"] = max(0.01, min(0.99, self.temperature))

        # Adjust max_tokens for thinking mode to avoid timeout
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        if self.top_p is not None:
            params["top_p"] = max(0.01, min(0.99, self.top_p))

        # Add GLM-specific extra_body
        extra_body = self._build_extra_body()
        if extra_body is not None:
            params["extra_body"] = extra_body

        return params

    def _get_usage(self, response: ChatCompletion) -> ChatInvokeUsage | None:
        """Extract usage from response."""
        if response.usage is None:
            return None

        prompt_details = getattr(response.usage, "prompt_tokens_details", None)
        cached_tokens = prompt_details.cached_tokens if prompt_details else None

        return ChatInvokeUsage(
            prompt_tokens=response.usage.prompt_tokens,
            prompt_cached_tokens=cached_tokens,
            prompt_cache_creation_tokens=None,
            prompt_image_tokens=None,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    def _extract_thinking(self, response: ChatCompletion) -> str | None:
        """Extract reasoning_content from response.

        GLM-5.1 uses message.model_extra['reasoning'] or message.model_extra['reasoning_content']
        """
        if not response.choices:
            return None

        message = response.choices[0].message

        # Method 1: Direct attribute (some providers)
        _rc = getattr(message, "reasoning_content", None)
        if _rc:
            thinking = _rc
            logger.debug(f"[GLM Thinking] {str(thinking)[:200]}...")
            return thinking

        # Method 2: model_extra['reasoning'] (GLM-5.1 vLLM)
        if hasattr(message, "model_extra") and message.model_extra:
            if "reasoning" in message.model_extra:
                thinking = message.model_extra["reasoning"]
                logger.debug(f"[GLM Thinking] {thinking[:200]}...")
                return thinking
            if "reasoning_content" in message.model_extra:
                thinking = message.model_extra["reasoning_content"]
                logger.debug(f"[GLM Thinking] {thinking[:200]}...")
                return thinking

        return None

    def _inject_format_instructions(
        self,
        messages: list[BaseMessage],
        output_format: type[BaseModel],
    ) -> list[BaseMessage]:
        """Inject format instructions into the last user message.

        Similar to LangChain's approach: embed format instructions in the prompt.

        Args:
            messages: Original messages list
            output_format: Pydantic model for structured output

        Returns:
            Modified messages with format instructions injected
        """
        if not messages:
            return messages

        format_instructions = get_format_instructions(output_format)

        # Find the last user message and inject format instructions
        new_messages: list[BaseMessage] = []
        last_user_idx = -1

        for i, msg in enumerate(messages):
            if isinstance(msg, UserMessage):
                last_user_idx = i

        if last_user_idx == -1:
            # No user message found, append format instructions as system message
            new_messages = list(messages)
            new_messages.append(SystemMessage(content=format_instructions))
            return new_messages

        # Inject format instructions into the last user message
        for i, msg in enumerate(messages):
            if i == last_user_idx and isinstance(msg, UserMessage):
                # When content is a multimodal list (text + image parts), append
                # the format instructions as a new text part to preserve images.
                # When content is plain text, fall back to string concatenation.
                if isinstance(msg.content, list):
                    new_parts = list(msg.content) + [
                        ContentPartTextParam(text=f"\n{format_instructions}")
                    ]
                    new_messages.append(UserMessage(content=new_parts))
                else:
                    new_content = f"{msg.content}\n\n{format_instructions}\n"
                    new_messages.append(UserMessage(content=new_content))
            else:
                new_messages.append(msg)

        return new_messages

    def _parse_json_output(self, content: str, output_format: type[T]) -> T:
        """Parse LLM output into Pydantic model.

        Handles JSON extraction from markdown code blocks and filters
        extra fields to handle models with extra="forbid".

        Args:
            content: Raw LLM output
            output_format: Pydantic model class

        Returns:
            Validated Pydantic model instance
        """
        # Try to extract JSON from markdown code blocks
        content = content.strip()

        # Handle ```json ... ``` blocks
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
            if content.endswith("```"):
                content = content[:-3]  # Remove trailing ```
            content = content.strip()
        elif content.startswith("```"):
            content = content[3:]  # Remove ```
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        # Try to find JSON object in the content
        try:
            # Try direct JSON parsing first
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON object between braces
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and start < end:
                try:
                    data = json.loads(content[start : end + 1])
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Could not parse JSON from output: {content[:200]}"
                    ) from e
            else:
                raise ValueError(f"No JSON object found in output: {content[:200]}")

        # Filter out extra fields not defined in the model schema
        # This handles models with extra="forbid" that reject unknown fields
        data = self._filter_extra_fields(data, output_format)

        # Validate with Pydantic model
        return output_format.model_validate(data)

    def _filter_extra_fields(self, data: Any, output_format: type[BaseModel]) -> Any:
        """Filter out extra fields not defined in the model schema.

        Args:
            data: Parsed JSON data
            output_format: Pydantic model class

        Returns:
            Filtered data dict
        """
        if not isinstance(data, dict):
            return data

        # Get the model's JSON schema
        schema = output_format.model_json_schema()

        # Get allowed properties at root level
        allowed_props = set(schema.get("properties", {}).keys())
        if "$defs" in schema:
            # Handle nested $defs for complex models
            allowed_props.update(self._get_all_defined_properties(schema))

        # Filter root level
        filtered = {}
        for key, value in data.items():
            if key in allowed_props:
                # Recursively filter nested objects
                prop_schema = schema.get("properties", {}).get(key, {})
                if prop_schema.get("type") == "object" and isinstance(value, dict):
                    # Check if this property has a $ref to another model
                    if "$ref" in prop_schema:
                        ref_name = prop_schema["$ref"].split("/")[-1]
                        nested_model = self._get_model_from_ref(output_format, ref_name)
                        if nested_model:
                            filtered[key] = self._filter_extra_fields(
                                value, nested_model
                            )
                        else:
                            filtered[key] = value
                    else:
                        filtered[key] = value
                elif prop_schema.get("type") == "array" and isinstance(value, list):
                    # Filter array items
                    item_schema = prop_schema.get("items", {})
                    filtered_items = []
                    for item in value:
                        if isinstance(item, dict):
                            if "$ref" in item_schema:
                                ref_name = item_schema["$ref"].split("/")[-1]
                                nested_model = self._get_model_from_ref(
                                    output_format, ref_name
                                )
                                if nested_model:
                                    filtered_items.append(
                                        self._filter_extra_fields(item, nested_model)
                                    )
                                else:
                                    filtered_items.append(item)
                            else:
                                # For anyOf union types, try each option
                                if "anyOf" in item_schema:
                                    filtered_item = self._filter_union_type(
                                        item, item_schema, output_format
                                    )
                                    filtered_items.append(filtered_item)
                                else:
                                    filtered_items.append(item)
                        else:
                            filtered_items.append(item)
                    filtered[key] = filtered_items
                else:
                    filtered[key] = value
            # Silently drop unknown fields

        return filtered

    def _get_all_defined_properties(self, schema: dict[str, Any]) -> set[str]:
        """Get all property names defined in schema including $defs."""
        props = set(schema.get("properties", {}).keys())
        for def_name, def_schema in schema.get("$defs", {}).items():
            props.update(def_schema.get("properties", {}).keys())
        return props

    def _get_model_from_ref(
        self, output_format: type[BaseModel], ref_name: str
    ) -> type[BaseModel] | None:
        """Get a nested model class from $defs by reference name."""
        # Try to find the model in annotations
        for anno in output_format.__mro__:
            if hasattr(anno, "__annotations__"):
                for field_name, field_type in getattr(
                    anno, "__annotations__", {}
                ).items():
                    if (
                        hasattr(field_type, "__name__")
                        and field_type.__name__ == ref_name
                    ):
                        return field_type
        return None

    def _filter_union_type(
        self,
        data: dict[str, Any],
        schema: dict[str, Any],
        parent_format: type[BaseModel],
    ) -> dict[str, Any]:
        """Filter data for union types (anyOf) - keep only fields from the matching option."""
        if not isinstance(data, dict):
            return data

        any_of = schema.get("anyOf", [])
        for option in any_of:
            if option.get("type") != "object":
                continue

            option_props = set(option.get("properties", {}).keys())
            data_keys = set(data.keys())

            # If this option's properties match the data (at least partially), use it
            if option_props & data_keys:  # Intersection
                # Filter to only include properties from this option
                filtered = {k: v for k, v in data.items() if k in option_props}
                return filtered

        # No matching option found, return data as-is
        return data

    async def _parse_stream_response(
        self,
        stream: Any,
        output_format: type[T] | None = None,
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        """Parse streaming response from GLM.

        Collects all chunks and assembles the final response.

        Args:
            stream: Async iterator of ChatCompletionChunk
            output_format: Optional Pydantic model for structured output

        Returns:
            ChatInvokeCompletion with assembled completion
        """
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        finish_reason: str | None = None

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Collect content
            if delta.content:
                content_parts.append(delta.content)

            # Collect thinking/reasoning (GLM-specific)
            # GLM-5.1 vLLM uses delta.reasoning
            if hasattr(delta, "reasoning") and delta.reasoning:
                thinking_parts.append(delta.reasoning)
            elif hasattr(delta, "reasoning_content") and delta.reasoning_content:
                thinking_parts.append(delta.reasoning_content)

            # Capture finish reason from last chunk
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        full_content = "".join(content_parts)
        full_thinking = "".join(thinking_parts) if thinking_parts else None

        if output_format is not None:
            # Parse structured output
            parsed = self._parse_json_output(full_content, output_format)
            return ChatInvokeCompletion[T](
                completion=parsed,
                thinking=full_thinking,
                usage=None,  # Stream mode doesn't have usage info
                stop_reason=finish_reason,
            )
        else:
            return ChatInvokeCompletion(
                completion=full_content,
                thinking=full_thinking,
                usage=None,  # Stream mode doesn't have usage info
                stop_reason=finish_reason,
            )

    def _parse_non_stream_response(
        self,
        response: ChatCompletion,
        output_format: type[T] | None = None,
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        """Parse non-streaming response from GLM."""
        choice = response.choices[0] if response.choices else None
        if choice is None:
            raise ModelProviderError(
                message="GLM response missing choices",
                status_code=502,
                model=self.name,
            )

        if output_format is not None:
            # Structured output
            if choice.message.content is None:
                raise ModelProviderError(
                    message="GLM response missing content",
                    status_code=500,
                    model=self.name,
                )

            raw_content = choice.message.content
            try:
                parsed = self._parse_json_output(raw_content, output_format)
            except Exception as e:
                schema_like_output = False
                try:
                    raw_obj = json.loads(raw_content)
                    if isinstance(raw_obj, dict):
                        schema_like_output = _looks_like_schema_object(raw_obj)
                except Exception:
                    schema_like_output = False
                required_fields = [
                    name
                    for name, field in output_format.model_fields.items()
                    if field.is_required()
                ]
                logger.warning(
                    "[GLM Structured Debug] parse failed | model=%s schema=%s required=%s raw_output=%r err=%s",
                    self.model,
                    output_format.__name__,
                    required_fields,
                    raw_content,
                    e,
                )
                if schema_like_output:
                    raise ModelProviderError(
                        message="SCHEMA_ECHO_STRUCTURED_OUTPUT",
                        model=self.name,
                    ) from e
                raise
            return ChatInvokeCompletion[T](
                completion=parsed,
                thinking=self._extract_thinking(response),
                usage=self._get_usage(response),
                stop_reason=choice.finish_reason,
            )
        else:
            # Non-structured output
            return ChatInvokeCompletion(
                completion=choice.message.content or "",
                thinking=self._extract_thinking(response),
                usage=self._get_usage(response),
                stop_reason=choice.finish_reason,
            )

    @overload
    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion[str]: ...

    @overload
    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type[T],
        **kwargs: Any,
    ) -> ChatInvokeCompletion[T]: ...

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type[T] | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        """Invoke GLM API with stream or non-stream mode.

        Args:
            messages: List of chat messages
            output_format: Optional Pydantic model for structured output.
                          When provided, format_instructions are injected into the prompt.

        Returns:
            ChatInvokeCompletion with completion, thinking, and usage
        """
        # Build model params
        model_params = self._build_model_params()
        print(f"[DEBUG] model_params: {model_params}, stream={self.stream}")

        try:
            # Prepare messages
            if output_format is None:
                glm_messages = OpenAIMessageSerializer.serialize_messages(messages)
                response_format = None
            else:
                modified_messages = self._inject_format_instructions(
                    messages, output_format
                )
                glm_messages = OpenAIMessageSerializer.serialize_messages(
                    modified_messages
                )
                response_format = {"type": "json_object"}

            typed_rf = cast(Any, response_format)

            # Make API call (stream or non-stream)
            if self.stream:
                # ===== STREAM MODE =====
                print("[GLM DEBUG] Using STREAM mode")
                stream = await self.get_client().chat.completions.create(
                    model=self.model,
                    messages=glm_messages,
                    response_format=typed_rf,
                    stream=True,
                    **model_params,
                )
                return await self._parse_stream_response(stream, output_format)
            else:
                # ===== NON-STREAM MODE =====
                print("[GLM DEBUG] Using NON-STREAM mode")
                response = await self.get_client().chat.completions.create(
                    model=self.model,
                    messages=glm_messages,
                    response_format=typed_rf,
                    stream=False,
                    **model_params,
                )
                return self._parse_non_stream_response(response, output_format)

        except RateLimitError as e:
            raise ModelRateLimitError(message=e.message, model=self.name) from e
        except APIConnectionError as e:
            raise ModelProviderError(message=str(e), model=self.name) from e
        except APIStatusError as e:
            raise ModelProviderError(
                message=e.message, status_code=e.status_code, model=self.name
            ) from e
        except ValidationError as e:
            raise ModelProviderError(
                message=f"Failed to parse structured output: {str(e)}",
                model=self.name,
            ) from e
        except Exception as e:
            raise ModelProviderError(message=str(e), model=self.name) from e
