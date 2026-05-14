"""ReAct-based page explorer — inherits BaseAgent ReAct loop.

Eliminates duplicated loop code by reusing BaseAgent's observe→think→act
infrastructure, while keeping PageController for browser interactions and
CartographyResult for output.
"""

from __future__ import annotations

import hashlib
import logging
from urllib.parse import parse_qsl, urlparse



logger = logging.getLogger(__name__)

_DEFAULT_MAX_STEPS = 500
_LLM_MAX_RETRIES = 3
_LLM_TIMEOUT_MS = 60_000


def _extract_spa_route(url: str) -> str:
    parsed = urlparse(url or "")
    return parsed.fragment or parsed.path or "/"


def _build_data_signature(url: str, dom_text: str = "") -> str:
    parsed = urlparse(url or "")
    query_keys = sorted(k for k, _ in parse_qsl(parsed.query, keep_blank_values=True))
    parts = list(query_keys)
    if dom_text and dom_text.strip():
        # Include a sample of visible DOM text so SPA views get distinct signatures
        text_sample = " ".join(dom_text.strip().split()[:40])
        parts.append(hashlib.md5(text_sample.encode("utf-8"), usedforsecurity=False).hexdigest()[:8])
    raw = "|".join(parts)
    return hashlib.md5(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def _build_state_identity(url: str, spa_route: str, view_fingerprint: str) -> str:
    """Build a stable state identity for cross-session reuse."""
    parsed = urlparse(url or "")
    origin = f"{parsed.scheme}://{parsed.netloc}".lower()
    route = (spa_route or "/").strip() or "/"
    view = (view_fingerprint or "").strip() or "no-view-fp"
    digest = hashlib.md5(f"{origin}|{route}|{view}".encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return f"state:{digest}"


