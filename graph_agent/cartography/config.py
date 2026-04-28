from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse


def clean_url(url: str) -> str:
    """Strip query parameters and fragments for stable identity."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    except Exception:
        return url


def resolve_mapping_url(url: str | None) -> str:
    """Resolve mapping URL from arg or env."""
    value = (url or "").strip()
    if value:
        return value
    env_value = (os.getenv("MAPPING_URL") or "").strip()
    if env_value:
        return env_value
    raise ValueError("url is required. Provide --url or set MAPPING_URL.")


def resolve_mapping_headless() -> bool:
    raw = (os.getenv("MAPPING_HEADLESS") or "").strip().lower()
    if raw in {"", "1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def resolve_mapping_channel() -> str | None:
    raw = (os.getenv("MAPPING_CHANNEL") or "").strip()
    return raw or None


def setup_browser_use_timeouts() -> None:
    mapping_timeout = os.getenv("MAPPING_TIMEOUT", "").strip()
    if mapping_timeout:
        try:
            timeout_val = float(mapping_timeout)
            os.environ.setdefault("TIMEOUT_NavigateToUrlEvent", str(timeout_val))
            os.environ.setdefault("TIMEOUT_BrowserStateRequestEvent", str(timeout_val))
            os.environ.setdefault("TIMEOUT_BrowserStartEvent", str(timeout_val))
        except ValueError:
            pass


def is_http_url(value: str) -> bool:
    v = (value or "").strip()
    return v.startswith("http://") or v.startswith("https://")


def same_origin(a: str, b: str) -> bool:
    if not a or not b:
        return False
    try:
        pa, pb = urlparse(a), urlparse(b)
        if not pa.scheme or not pb.scheme or not pa.netloc or not pb.netloc:
            return False
        return pa.scheme.lower() == pb.scheme.lower() and pa.netloc.lower() == pb.netloc.lower()
    except Exception:
        return False


def env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def load_inventory(inventory_path: str | Path) -> list[dict]:
    path = Path(inventory_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Inventory file not found: {path}. Run scout first: uv run python -m graph_agent.cartography.runner --url <url> (with --inventory and --output)."
        )
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    elements = raw.get("elements") if isinstance(raw, dict) else []
    return elements if isinstance(elements, list) else []


def resolve_layout_aware_enabled() -> bool:
    """Enable layout-aware pipeline by default; allow env rollback."""
    return env_bool("CARTOGRAPHY_LAYOUT_AWARE_ENABLED", True)


def resolve_layout_snapshot_limit() -> int:
    raw = (os.getenv("CARTOGRAPHY_LAYOUT_SNAPSHOT_LIMIT") or "").strip()
    if not raw:
        return 200
    try:
        val = int(raw)
    except ValueError:
        return 200
    return max(20, min(1000, val))


def resolve_layout_confidence_retry_enabled() -> bool:
    """Enable low-layout-confidence compensation strategy."""
    return env_bool("CARTOGRAPHY_LAYOUT_CONFIDENCE_RETRY_ENABLED", False)


def resolve_layout_confidence_threshold() -> float:
    raw = (os.getenv("CARTOGRAPHY_LAYOUT_CONFIDENCE_THRESHOLD") or "").strip()
    if not raw:
        return 0.45
    try:
        val = float(raw)
    except ValueError:
        return 0.45
    return max(0.1, min(0.95, val))
