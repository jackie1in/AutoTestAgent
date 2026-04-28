from __future__ import annotations

import logging
import os
import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_initialized = False
_lmnr_available = False


def _no_op_observe(*_args: Any, **_kwargs: Any) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await func(*args, **kwargs)

            return cast(F, async_wrapper)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return cast(F, sync_wrapper)

    return decorator


try:
    from lmnr import Laminar as _Laminar
    from lmnr import observe as _observe

    _lmnr_available = True
except Exception:  # pragma: no cover - optional dependency
    _Laminar = None
    _observe = _no_op_observe


def observe(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Safe observe decorator; no-op when lmnr is unavailable."""
    return _observe(*args, **kwargs)


def initialize_laminar() -> bool:
    """Initialize Laminar once from environment variables.

    Returns:
        True when Laminar is initialized in this process, else False.
    """
    global _initialized
    if _initialized:
        return _lmnr_available
    _initialized = True

    if not _lmnr_available or _Laminar is None:
        logger.info("Laminar SDK not installed; observability decorators run as no-op.")
        return False

    project_api_key = (os.getenv("LMNR_PROJECT_API_KEY") or "").strip()
    if not project_api_key:
        logger.info("LMNR_PROJECT_API_KEY is empty; Laminar initialization skipped.")
        return False

    kwargs: dict[str, Any] = {"project_api_key": project_api_key}
    base_url = (os.getenv("LMNR_BASE_URL") or "").strip()
    if base_url:
        kwargs["base_url"] = base_url

    try:
        _Laminar.initialize(**kwargs)
        logger.info("Laminar initialized (base_url=%s).", kwargs.get("base_url", "default"))
        return True
    except Exception as exc:  # pragma: no cover - runtime env dependent
        logger.warning("Laminar initialize failed: %s", exc)
        return False
