"""ReAct-based page explorer — inherits BaseAgent ReAct loop.

Eliminates duplicated loop code by reusing BaseAgent's observe→think→act
infrastructure, while keeping PageController for browser interactions and
CartographyResult for output.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from typing import cast
from urllib.parse import parse_qsl, urlparse

from browser_use.browser.session import BrowserSession as Browser

from graph_agent.cartography.base_agent import BaseAgent
from graph_agent.cartography.captcha import (
    normalize_manual_captcha_code,
    resolve_captcha_solve_mode,
    solve_captcha_from_page,
)
from graph_agent.cartography.inference_core import (
    SemanticInferenceInput,
    infer_transition_semantics,
)
from graph_agent.cartography.react_prompts import (
    build_system_prompt,
    build_user_prompt,
)
from graph_agent.cartography.react_schema import (
    AgentOutput,
)
from graph_agent.graph.merger import CartographyResult
from graph_agent.lib.page_controller import PageController
from graph_agent.llm import get_llm
from graph_agent.models import (
    ActionType,
    Checkpoint,
    CheckpointExpect,
    CheckpointLayer,
    CheckpointTiming,
    Severity,
    State,
    Transition,
    TransitionStep,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_STEPS = 500
_LLM_MAX_RETRIES = 3
_LLM_TIMEOUT_MS = 60_000


def _extract_spa_route(url: str) -> str:
    parsed = urlparse(url or "")
    return parsed.fragment or parsed.path or "/"


def _build_data_signature(url: str) -> str:
    parsed = urlparse(url or "")
    query_keys = sorted(k for k, _ in parse_qsl(parsed.query, keep_blank_values=True))
    raw = "|".join(query_keys)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _build_state_identity(url: str, spa_route: str, view_fingerprint: str) -> str:
    """Build a stable state identity for cross-session reuse."""
    parsed = urlparse(url or "")
    origin = f"{parsed.scheme}://{parsed.netloc}".lower()
    route = (spa_route or "/").strip() or "/"
    view = (view_fingerprint or "").strip() or "no-view-fp"
    digest = hashlib.md5(f"{origin}|{route}|{view}".encode("utf-8")).hexdigest()[:16]
    return f"state:{digest}"


