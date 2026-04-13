"""GLM (Zhipu AI) LLM provider.

Reference: https://docs.bigmodel.cn/cn/guide/develop/langchain/introduction
Thinking: https://docs.bigmodel.cn/cn/guide/capabilities/thinking-mode

Uses OpenAI SDK with GLM's OpenAI-compatible API.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, overload

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel, ValidationError
from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import BaseMessage, SystemMessage, UserMessage
from browser_use.llm.openai.serializer import OpenAIMessageSerializer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

from graph_agent.llm.utils import get_format_instructions

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


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
    temperature: float | None = 0.6
    max_tokens: int | None = 4096
    top_p: float | None = None

    # GLM-specific: Thinking mode
    thinking: Literal["enabled", "disabled"] | None = None
    clear_thinking: bool = True  # False = Preserved Thinking

    # Client params
    api_key: str | None = None
    base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    timeout: float | httpx.Timeout | None = 60.0
    max_retries: int = 3
    http_client: httpx.AsyncClient | None = None

    _client: AsyncOpenAI | None = None

    @property
    def provider(self) -> str:
        return "glm"

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
            extra["thinking"] = {"type": self.thinking}
            if self.thinking == "enabled":
                extra["thinking"]["clear_thinking"] = self.clear_thinking
        
        return extra if extra else None

    def _build_model_params(self) -> dict[str, Any]:
        """Build model parameters for API call."""
        params: dict[str, Any] = {}
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        
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
        """Extract reasoning_content from response."""
        if not response.choices:
            return None
        
        message = response.choices[0].message
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            thinking = message.reasoning_content
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
                # Inject format instructions
                new_content = f"""{msg.content}

{format_instructions}
"""
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
                    data = json.loads(content[start:end+1])
                except json.JSONDecodeError as e:
                    raise ValidationError(f"Could not parse JSON from output: {content[:200]}") from e
            else:
                raise ValidationError(f"No JSON object found in output: {content[:200]}")
        
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
                            filtered[key] = self._filter_extra_fields(value, nested_model)
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
                                nested_model = self._get_model_from_ref(output_format, ref_name)
                                if nested_model:
                                    filtered_items.append(self._filter_extra_fields(item, nested_model))
                                else:
                                    filtered_items.append(item)
                            else:
                                # For anyOf union types, try each option
                                if "anyOf" in item_schema:
                                    filtered_item = self._filter_union_type(item, item_schema, output_format)
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

    def _get_model_from_ref(self, output_format: type[BaseModel], ref_name: str) -> type[BaseModel] | None:
        """Get a nested model class from $defs by reference name."""
        # Try to find the model in annotations
        for anno in output_format.__mro__:
            if hasattr(anno, "__annotations__"):
                for field_name, field_type in getattr(anno, "__annotations__", {}).items():
                    if hasattr(field_type, "__name__") and field_type.__name__ == ref_name:
                        return field_type
        return None

    def _filter_union_type(self, data: dict[str, Any], schema: dict[str, Any], parent_format: type[BaseModel]) -> dict[str, Any]:
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
        """Invoke GLM API.
        
        Args:
            messages: List of chat messages
            output_format: Optional Pydantic model for structured output.
                          When provided, format_instructions are injected into the prompt.
            
        Returns:
            ChatInvokeCompletion with completion, thinking, and usage
        """
        # Build model params
        model_params = self._build_model_params()
        
        try:
            if output_format is None:
                # Non-structured output - use original messages
                glm_messages = OpenAIMessageSerializer.serialize_messages(messages)
                
                # DEBUG: Log messages being sent
                print(f"\n[GLM DEBUG] ===== SENDING {len(glm_messages)} MESSAGES =====")
                for i, msg in enumerate(glm_messages):
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        print(f"[GLM DEBUG] Message {i} ({msg.get('role')}):")
                        print(f"{content[:1000]}...")
                        print()
                    else:
                        print(f"[GLM DEBUG] Message {i} ({msg.get('role')}): [complex content]")
                print("[GLM DEBUG] ===== END MESSAGES =====\n")
                
                response = await self.get_client().chat.completions.create(
                    model=self.model,
                    messages=glm_messages,
                    **model_params,
                )
                
                choice = response.choices[0] if response.choices else None
                if choice is None:
                    raise ModelProviderError(
                        message="GLM response missing choices",
                        status_code=502,
                        model=self.name,
                    )
                
                return ChatInvokeCompletion(
                    completion=choice.message.content or "",
                    thinking=self._extract_thinking(response),
                    usage=self._get_usage(response),
                    stop_reason=choice.finish_reason,
                )
            
            else:
                # Structured output - inject format_instructions into prompt
                modified_messages = self._inject_format_instructions(messages, output_format)
                glm_messages = OpenAIMessageSerializer.serialize_messages(modified_messages)
                
                # DEBUG: Log messages being sent
                print(f"\n[GLM DEBUG] ===== SENDING {len(glm_messages)} MESSAGES (STRUCTURED) =====")
                for i, msg in enumerate(glm_messages):
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        print(f"[GLM DEBUG] Message {i} ({msg.get('role')}):")
                        print(f"{content[:1000]}...")
                        print()
                    else:
                        print(f"[GLM DEBUG] Message {i} ({msg.get('role')}): [complex content]")
                print("[GLM DEBUG] ===== END MESSAGES =====\n")
                
                # Use json_object mode to encourage JSON output
                response = await self.get_client().chat.completions.create(
                    model=self.model,
                    messages=glm_messages,
                    response_format={"type": "json_object"},
                    **model_params,
                )
                
                choice = response.choices[0] if response.choices else None
                if choice is None or choice.message.content is None:
                    raise ModelProviderError(
                        message="GLM response missing content",
                        status_code=500,
                        model=self.name,
                    )
                
                # Parse and validate the output
                parsed = self._parse_json_output(choice.message.content, output_format)
                
                return ChatInvokeCompletion[T](
                    completion=parsed,
                    thinking=self._extract_thinking(response),
                    usage=self._get_usage(response),
                    stop_reason=choice.finish_reason,
                )
        
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


def create_glm_llm(
    model: str = "glm-5",
    api_key: str | None = None,
    enable_thinking: bool = False,
    preserve_thinking: bool = True,
    **kwargs,
) -> ChatZhiPu:
    """Create a ChatGLM instance.
    
    Args:
        model: Model name
        api_key: API key (falls back to GLM_API_KEY or LLM_API_KEY env var)
        enable_thinking: Enable thinking mode
        preserve_thinking: Preserve thinking across turns
        **kwargs: Additional args for ChatGLM
    """
    import os
    
    api_key = api_key or os.getenv("GLM_API_KEY") or os.getenv("LLM_API_KEY")
    
    return ChatZhiPu(
        model=model,
        api_key=api_key,
        thinking="enabled" if enable_thinking else "disabled",
        clear_thinking=not preserve_thinking,
        **kwargs,
    )
