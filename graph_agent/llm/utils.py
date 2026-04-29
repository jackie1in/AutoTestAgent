"""LLM utility functions.

Helper functions for common LLM operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from browser_use.llm.base import BaseChatModel
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


def get_format_instructions(output_format: type[BaseModel]) -> str:
    """Generate format instructions from a Pydantic model.
    
    Similar to LangChain's PydanticOutputParser.get_format_instructions().
    Creates detailed JSON schema instructions for LLM structured output.
    
    Args:
        output_format: Pydantic model class defining the output structure
        
    Returns:
        Format instructions string to embed in prompts
    """
    # Get optimized schema with all details preserved
    schema = SchemaOptimizer.create_optimized_json_schema(output_format)
    
    # Build comprehensive format instructions
    lines = [
        "You must respond with a valid JSON object that follows this exact schema:",
        "",
        "```json",
        json.dumps(schema, indent=2, ensure_ascii=False),
        "```",
        "",
        "CRITICAL INSTRUCTIONS:",
        "1. Output ONLY the JSON object, with NO markdown code blocks, NO explanations, NO comments",
        "2. The JSON must be valid and parseable",
        "3. Include ALL required fields marked in the schema",
        "4. For fields with 'anyOf' (union types), choose ONE of the options and include ONLY that option's fields",
        "5. Do NOT include extra fields not defined in the schema",
        "6. Field types must match exactly (string, number, boolean, object, array)",
        "7. Do NOT output the schema itself (no 'properties', 'required', '$defs', or 'additionalProperties' as top-level output)",
        "8. Your output must be an INSTANCE object, not a schema/definition/template",
    ]
    
    # Add required fields info
    required = schema.get("required", [])
    if required:
        lines.append(f"9. Required top-level fields: {', '.join(required)}")
    
    # Add anyOf guidance if present
    anyof_fields = _find_anyof_fields(schema)
    if anyof_fields:
        lines.append("")
        lines.append("UNION TYPE FIELDS (choose exactly one option):")
        for field_name, options in anyof_fields.items():
            lines.append(f"- '{field_name}': choose one of [{', '.join(options)}]")
    
    return "\n".join(lines)


def _find_anyof_fields(schema: dict[str, Any], prefix: str = "") -> dict[str, list[str]]:
    """Find fields with anyOf union types and their options.
    
    Args:
        schema: JSON schema dict
        prefix: Field name prefix for nested fields
        
    Returns:
        Dict mapping field paths to list of option names
    """
    result: dict[str, list[str]] = {}
    
    properties = schema.get("properties", {})
    for prop_name, prop_schema in properties.items():
        full_name = f"{prefix}.{prop_name}" if prefix else prop_name
        
        # Check for anyOf at this level
        if "anyOf" in prop_schema:
            options = []
            for option in prop_schema["anyOf"]:
                if option.get("type") == "object" and "properties" in option:
                    # Get the first property name as the discriminator
                    prop_keys = list(option["properties"].keys())
                    if prop_keys:
                        options.append(prop_keys[0])
                elif "title" in option:
                    options.append(option["title"])
            if options:
                result[full_name] = options
        
        # Recurse into nested objects
        if prop_schema.get("type") == "object" and "properties" in prop_schema:
            result.update(_find_anyof_fields(prop_schema, full_name))
        
        # Recurse into array items
        if prop_schema.get("type") == "array" and "properties" in prop_schema.get("items", {}):
            result.update(_find_anyof_fields(prop_schema["items"], full_name))
    
    return result


def _format_schema_for_instructions(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON schema to a simplified, human-readable format.
    
    Args:
        schema: JSON schema dict
        
    Returns:
        Simplified schema for human reading
    """
    result: dict[str, Any] = {}
    
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    for prop_name, prop_schema in properties.items():
        result[prop_name] = _describe_property(prop_schema, prop_name in required)
    
    return result


def _describe_property(prop_schema: dict[str, Any], is_required: bool) -> Any:
    """Describe a single property in human-readable format.
    
    Args:
        prop_schema: Property schema dict
        is_required: Whether this property is required
        
    Returns:
        Description string or nested dict
    """
    prop_type = prop_schema.get("type", "any")
    description = prop_schema.get("description", "")
    enum = prop_schema.get("enum")
    
    # Handle array type
    if prop_type == "array":
        items = prop_schema.get("items", {})
        item_desc = _describe_property(items, False)
        if isinstance(item_desc, str):
            base_desc = f"array of {item_desc}"
        else:
            base_desc = "array of objects"
    
    # Handle object type (nested)
    elif prop_type == "object":
        nested_props = prop_schema.get("properties", {})
        if nested_props:
            nested_required = prop_schema.get("required", [])
            return {
                k: _describe_property(v, k in nested_required)
                for k, v in nested_props.items()
            }
        else:
            base_desc = "object"
    
    # Handle anyOf (union types)
    elif "anyOf" in prop_schema:
        types = []
        for option in prop_schema["anyOf"]:
            if option.get("type") and option.get("type") != "null":
                types.append(option.get("type"))
        base_desc = f"{' or '.join(types)}"
    
    # Simple types
    elif enum:
        base_desc = f"enum: {', '.join(repr(e) for e in enum)}"
    else:
        base_desc = prop_type
    
    # Add description if available
    if description:
        base_desc = f"{base_desc} - {description}"
    
    # Mark optional fields
    if not is_required:
        base_desc = f"{base_desc} (optional)"
    
    return base_desc


async def ainvoke_structured(
    llm: BaseChatModel,
    system_prompt: str,
    user_prompt: str,
    output_format: type[T],
    *,
    timeout_ms: int | None = None,
    max_retries: int = 3,
    **kwargs: Any,
) -> T:
    """Invoke LLM with structured output validation.
    
    Args:
        llm: LLM instance
        system_prompt: System-level instructions
        user_prompt: User prompt
        output_format: Pydantic model class for output validation
        timeout_ms: Optional timeout in milliseconds
        max_retries: Max retries for validation errors
        **kwargs: Additional arguments for LLM
        
    Returns:
        Validated instance of output_format
    """
    from browser_use.llm.messages import SystemMessage, UserMessage

    messages = [
        SystemMessage(content=system_prompt),
        UserMessage(content=user_prompt),
    ]
    
    last_error: Exception | None = None
    
    for attempt in range(max_retries):
        try:
            # Calculate timeout
            timeout_s = timeout_ms / 1000.0 if timeout_ms else None
            
            # Make the call
            if timeout_s:
                coro = llm.ainvoke(messages, output_format=output_format, **kwargs)
                result = await asyncio.wait_for(coro, timeout=timeout_s)
            else:
                result = await llm.ainvoke(messages, output_format=output_format, **kwargs)

            # Return the parsed completion
            if isinstance(result, ChatInvokeCompletion):
                return result.completion
            return result
            
        except ValidationError as e:
            last_error = e
            logger.warning(
                "Validation failed (attempt %d/%d): %s",
                attempt + 1, max_retries, str(e)[:200]
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
                
        except asyncio.TimeoutError:
            logger.warning("LLM timeout (attempt %d/%d)", attempt + 1, max_retries)
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))
                
        except Exception as e:
            last_error = e
            required_fields = []
            if hasattr(output_format, "model_fields"):
                required_fields = [
                    name
                    for name, field in output_format.model_fields.items()
                    if getattr(field, "is_required", lambda: False)()
                ]
            logger.warning(
                "LLM error (attempt %d/%d) | schema=%s required=%s | err=%s",
                attempt + 1,
                max_retries,
                output_format.__name__,
                required_fields,
                str(e)[:600],
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
    
    # All retries exhausted
    raise RuntimeError(
        f"LLM structured output failed after {max_retries} attempts. "
        f"Last error: {last_error}"
    ) from last_error


async def ainvoke_prompt(llm: BaseChatModel, prompt: str) -> str:
    """Simple prompt invocation returning string response.
    
    Args:
        llm: LLM instance
        prompt: User prompt
        
    Returns:
        String response
    """
    from browser_use.llm.messages import UserMessage

    result = await llm.ainvoke([UserMessage(content=prompt)])

    if isinstance(result, ChatInvokeCompletion):
        return result.completion
    return str(result)
