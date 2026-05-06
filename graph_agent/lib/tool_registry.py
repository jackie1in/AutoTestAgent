"""Tool registry for ReAct agent - inspired by browser-use's registry.

Provides @action decorator for registering tools with automatic:
- Pydantic schema generation
- System prompt injection
- Action dispatch
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, get_type_hints

from pydantic import BaseModel, Field, create_model

T = TypeVar("T")


@dataclass
class RegisteredAction:
    """A registered action with its metadata."""

    name: str
    description: str
    param_model: type[BaseModel]
    handler: Callable
    terminates: bool = False  # If True, ends the exploration


class ToolRegistry:
    """Registry for ReAct agent actions.

    Usage:
        registry = ToolRegistry()

        @registry.action("Click an element by index")
        def click_element_by_index(index: int, controller: PageController):
            return controller.click(index)
    """

    def __init__(self):
        self.actions: dict[str, RegisteredAction] = {}
        self._special_params = {"controller", "context", "session"}

    def action(
        self,
        description: str,
        param_model: type[BaseModel] | None = None,
        terminates: bool = False,
    ) -> Callable:
        """Decorator to register an action.

        Args:
            description: Human-readable description for LLM
            param_model: Optional Pydantic model for parameters
            terminates: If True, this action ends the exploration
        """
        # Capture param_model in closure
        _param_model = param_model

        def decorator(func: Callable) -> Callable:
            name = func.__name__

            # Create param model from function signature if not provided
            final_param_model = _param_model
            if final_param_model is None:
                final_param_model = self._create_param_model(func)

            # Register the action
            self.actions[name] = RegisteredAction(
                name=name,
                description=description,
                param_model=final_param_model,
                handler=func,
                terminates=terminates,
            )

            # Return original function unchanged
            return func

        return decorator

    def _create_param_model(self, func: Callable) -> type[BaseModel]:
        """Create Pydantic model from function signature."""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        fields = {}
        for param_name, param in sig.parameters.items():
            # Skip special injected params
            if param_name in self._special_params:
                continue

            # Get type annotation
            annotation = type_hints.get(param_name, str)

            # Get default value
            if param.default is inspect.Parameter.empty:
                default = ...
            else:
                default = param.default

            fields[param_name] = (annotation, default)

        # Create dynamic model
        model_name = f"{func.__name__}_params"
        return create_model(model_name, **fields)

    def create_action_union(self) -> type[BaseModel]:
        """Create a discriminated union of all registered actions.

        Returns a Pydantic model that can be used as output_format.
        """
        from typing import Annotated, Literal, Union

        action_models = []
        for name, action in self.actions.items():
            # Create action model with action_type discriminator
            action_model = create_model(
                f"{name}_action",
                action_type=(Literal[name], name),  # type: ignore
                __base__=action.param_model,
            )
            action_models.append(action_model)

        # Create union
        if len(action_models) == 1:
            return action_models[0]

        # Annotated union with discriminator
        union_type = Union[tuple(action_models)]  # type: ignore
        return Annotated[union_type, Field(discriminator="action_type")]  # type: ignore[return-value]

    def format_for_prompt(self) -> str:
        """Format all actions for system prompt."""
        lines = ["Available actions (set action_type to the name):"]

        for name, action in self.actions.items():
            # Format parameters
            schema = action.param_model.model_json_schema()
            props = schema.get("properties", {})
            param_strs = []

            for prop_name, prop_info in props.items():
                ptype = prop_info.get("type", "any")
                enum = prop_info.get("enum")
                default = prop_info.get("default")

                parts = [prop_name]
                if ptype != "object":
                    parts.append(f": {ptype}")
                if enum:
                    parts.append(f" ({'|'.join(enum)})")
                if default is not None:
                    parts.append(f" = {default}")

                param_strs.append("".join(parts))

            params_desc = f" ({', '.join(param_strs)})" if param_strs else ""
            lines.append(f"- {name}: {action.description}{params_desc}")

        return "\n".join(lines)

    async def execute(
        self,
        action_name: str,
        params: dict[str, Any],
        controller: Any,
        context: Any = None,
    ) -> Any:
        """Execute a registered action.

        Args:
            action_name: Name of the action to execute
            params: Parameters for the action
            controller: PageController instance (injected)
            context: Optional context (injected)
        """
        if action_name not in self.actions:
            raise ValueError(f"Unknown action: {action_name}")

        action = self.actions[action_name]

        # Validate params
        validated = action.param_model.model_validate(params)

        # Build kwargs
        kwargs = validated.model_dump()

        # Inject special params
        sig = inspect.signature(action.handler)
        if "controller" in sig.parameters:
            kwargs["controller"] = controller
        if "context" in sig.parameters:
            kwargs["context"] = context
        if "session" in sig.parameters:
            kwargs["session"] = (
                controller._session if hasattr(controller, "_session") else None
            )

        # Call handler
        if inspect.iscoroutinefunction(action.handler):
            return await action.handler(**kwargs)
        else:
            return action.handler(**kwargs)

    def get_action_schemas(self) -> dict[str, type[BaseModel]]:
        """Get mapping of action names to their param models."""
        return {name: action.param_model for name, action in self.actions.items()}


# Global registry instance
_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def action(
    description: str,
    param_model: type[BaseModel] | None = None,
    terminates: bool = False,
):
    """Convenience decorator using global registry.

    Usage:
        @action("Click an element by index")
        def click_element_by_index(index: int, controller: PageController):
            return controller.click(index)
    """
    return get_registry().action(description, param_model, terminates)


def reset_registry():
    """Reset the global registry (useful for testing)."""
    global _registry
    _registry = ToolRegistry()
