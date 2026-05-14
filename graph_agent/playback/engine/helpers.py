"""Playwright playback engine: run edge list (async, no LLM)."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from time import monotonic
from typing import Any
from urllib.parse import urlparse


from graph_agent.models import (
    ElementConstraints,
    FrameLocatorSnapshot,
    GraphEdge,
)

DEFAULT_TIMEOUT_MS = 60_000
NETWORK_OBSERVE_MS = 150
NETWORK_POLL_MS = 25
PLAYBACK_RETRY_COUNT = 2
PLAYBACK_RETRY_DELAY_S = 0.3

logger = logging.getLogger(__name__)


_LOGIN_URL_TOKENS: frozenset[str] = frozenset(
    (
        "login",
        "signin",
        "sign-in",
        "sign_in",
        "/sso",
        "/oauth/authorize",
        "/oauth2/authorize",
    )
)


def _is_login_like_url(url: str) -> bool:
    """Best-effort login page detection for enhanced auth failure diagnostics."""
    lowered = url.lower()
    return any(token in lowered for token in _LOGIN_URL_TOKENS)


def _urls_same_page(a: str, b: str) -> bool:
    """Compare two URLs ignoring fragment and trivial query differences."""
    pa, pb = urlparse(a), urlparse(b)
    return (
        pa.scheme == pb.scheme
        and pa.netloc == pb.netloc
        and pa.path.rstrip("/") == pb.path.rstrip("/")
    )


def _extract_login_error_message(payload_text: str) -> str | None:
    """Parse backend login response text and return a human-readable error message."""
    if not payload_text:
        return None
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("serviceSuccess") is True:
        return None
    errors = payload.get("errors")
    if not isinstance(errors, list) or not errors:
        return None
    first = errors[0]
    if not isinstance(first, dict):
        return None
    msg = str(first.get("msg") or "").strip()
    return msg or None


def _extract_login_error_message_from_page_text(page_text: str) -> str | None:
    """Best-effort extraction from rendered login page messages."""
    if not page_text:
        return None
    candidates = (
        "帐号已过期",
        "账号已过期",
        "密码错误",
        "登录失败",
        "用户不存在",
    )
    for line in page_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(token in stripped for token in candidates):
            return stripped
    return None


def _generate_value_from_constraints(constraints: ElementConstraints) -> str:
    """Generate a dummy value based on constraints."""
    fmt = constraints.format
    if fmt == "email":
        return f"test-{uuid.uuid4()}@example.com"
    if fmt == "password":
        return "Test@1234"  # Static strong password
    if fmt == "phone":
        return "13800138000"
    return "test_value"


def _is_http_request(request: Any) -> bool:
    """Track only HTTP(S) requests and ignore websocket traffic."""
    url = str(getattr(request, "url", "") or "")
    if not url.startswith(("http://", "https://")):
        return False
    resource_type = str(getattr(request, "resource_type", "") or "").lower()
    return resource_type != "websocket"


def _resolve_navigate_url(edge: GraphEdge) -> str | None:
    """Best-effort URL resolution for explicit navigate steps."""
    for candidate in (edge.action_value, edge.target, edge.source):
        value = str(candidate or "").strip()
        if value.startswith(("http://", "https://", "data:")):
            return value
    return None


def _remove_listener(page: Any, event: str, handler: Callable[[Any], None]) -> None:
    """Detach event listener across Playwright versions and test doubles."""
    if hasattr(page, "off"):
        page.off(event, handler)
        return
    if hasattr(page, "remove_listener"):
        page.remove_listener(event, handler)


def _is_closed_context_error(exc: BaseException) -> bool:
    """Detect Playwright 'page/context/browser has been closed' errors.

    Also walks the exception chain (``__cause__``) so that wrapped errors
    (e.g. ``ValueError`` from ``_resolve_playback_context``) are recognised.
    """
    cur: BaseException | None = exc
    while cur is not None:
        msg = str(cur).lower()
        if "has been closed" in msg or "target closed" in msg:
            return True
        cur = cur.__cause__
    return False


def _is_transient_error(exc: BaseException) -> bool:
    """Detect transient failures (async load, timing) that may succeed on retry."""
    if _is_closed_context_error(exc):
        return False
    msg = str(exc).lower()
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    return any(
        token in msg
        for token in ("timed out", "timeout", "waiting for selector", "locator")
    )


async def _retry_action(
    action: Callable[[], Awaitable[None]],
    *,
    max_retries: int = PLAYBACK_RETRY_COUNT,
    retry_delay_s: float = PLAYBACK_RETRY_DELAY_S,
) -> None:
    """Retry an action on transient failures to improve playback stability."""
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            await action()
            return
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries and _is_transient_error(exc):
                await asyncio.sleep(retry_delay_s)
                continue
            raise
    if last_exc is not None:
        raise last_exc


def _frame_selector_candidates(frame: FrameLocatorSnapshot, index: int) -> list[str]:
    """Return ordered list of selector candidates for a frame (stability fallback)."""
    candidates: list[str] = []
    primary = str(frame.selector or "").strip()
    if primary:
        candidates.append(primary)
    css = str(frame.css_selector or "").strip()
    if css and css != primary:
        candidates.append(css)
    xp = frame.xpath or frame.x_path
    xp_str = str(xp or "").strip()
    if xp_str and xp_str != primary:
        candidates.append(
            f"xpath={xp_str}" if not xp_str.startswith("xpath=") else xp_str
        )
    if not candidates:
        raise ValueError(f"Missing selector for iframe level {index}")
    return candidates


def _resolve_playback_context(page: Any, frame_path: list[FrameLocatorSnapshot]) -> Any:
    """Resolve the page or nested frame context for an edge."""
    context = page
    for index, frame in enumerate(frame_path, start=1):
        candidates = _frame_selector_candidates(frame, index)
        last_exc: Exception | None = None
        for frame_selector in candidates:
            try:
                context = context.frame_locator(frame_selector)
                break
            except Exception as exc:
                last_exc = exc
                continue
        else:
            raise ValueError(
                f"Failed to locate iframe level {index}: {candidates[0]}"
            ) from last_exc
    return context


def _element_selector_candidates(edge: GraphEdge) -> list[str]:
    """Return ordered selector candidates for element (PRD: id > css > semantic > xpath)."""
    candidates: list[str] = []
    primary = str(edge.selector or "").strip()
    elem = edge.element

    # 1. Prefer id (most stable)
    if elem and elem.id:
        id_sel = f"#{elem.id}"
        if id_sel not in candidates:
            candidates.append(id_sel)

    # 2. Semantic selectors from attributes
    if elem and elem.attributes:
        attrs = elem.attributes
        if attrs.get("href"):
            href = str(attrs["href"])
            candidates.append(f'a[href="{href}"]')
        if attrs.get("type") == "submit":
            candidates.append('button[type="submit"]')
        if attrs.get("name") and attrs.get("type"):
            candidates.append(f'input[type="{attrs["type"]}"][name="{attrs["name"]}"]')
        elif attrs.get("name"):
            candidates.append(f'[name="{attrs["name"]}"]')

    # 3. Stable css_selector from element snapshot
    if elem and elem.css_selector and elem.css_selector != primary:
        candidates.append(elem.css_selector)

    # 4. Primary selector (may be xpath)
    if primary and primary not in candidates:
        candidates.append(primary)

    return candidates if candidates else [primary] if primary else []


async def _ensure_frame_attached(
    page: Any, frame_path: list[FrameLocatorSnapshot], timeout_ms: int = 10_000
) -> None:
    """Wait for iframe element to appear in the DOM **and** for the inner
    frame document to reach a usable state.

    In SPAs the content-area iframe often keeps the same ``id``/``name`` but
    its ``src`` is swapped by JS when the user clicks a different menu item.
    The iframe *element* reappears almost immediately, but the inner document
    may still be loading.  Without the second wait the subsequent
    ``frame_locator().locator().click()`` will hit a detached frame and
    throw ``Target page, context or browser has been closed``.
    """
    if not frame_path or not hasattr(page, "wait_for_selector"):
        return

    matched_sel: str | None = None
    for frame in frame_path:
        candidates = _frame_selector_candidates(frame, 1)
        for sel in candidates:
            try:
                await page.wait_for_selector(sel, state="attached", timeout=timeout_ms)
                matched_sel = sel
                break
            except Exception:
                continue
        if matched_sel:
            break

    if matched_sel is None:
        return

    # After the iframe element is attached, wait for its inner frame to be
    # navigated (domcontentloaded).  frame_locator() is lazy so we must
    # resolve the actual Frame and call wait_for_load_state on it.
    try:
        fl = page.frame_locator(matched_sel)
        inner_locator = fl.locator(":root")
        await inner_locator.wait_for(state="attached", timeout=min(timeout_ms, 10_000))
    except Exception:
        pass


def _locator_in_context(
    page: Any, selector: str, frame_path: list[FrameLocatorSnapshot]
) -> Any:
    """Build a locator in the target page/frame context."""
    return _resolve_playback_context(page, frame_path).locator(selector)


async def _try_action_with_selector_fallback(
    page_for_edge: Any,
    edge: GraphEdge,
    action_fn: Callable[[Any], Awaitable[None]],
) -> None:
    """Try action with each selector candidate until one succeeds (PRD: id > css > xpath)."""
    candidates = _element_selector_candidates(edge)
    last_exc: Exception | None = None
    for sel in candidates:
        try:
            loc = _locator_in_context(page_for_edge, sel, edge.frame_path or [])
            await action_fn(loc)
            return
        except Exception as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise ValueError("No selector candidates for edge")


def _format_replay_error(
    step_error: Exception,
    edge: GraphEdge,
    tab_id: str,
) -> str:
    """Format playback error with context for diagnosis (tab/iframe/selector/async_load)."""
    orig = str(step_error)
    # Tab errors already have clear message from _page_for_tab
    if "Target tab does not exist" in orig:
        return f"tab: {orig}"
    # Iframe errors from _resolve_playback_context
    if "iframe" in orig.lower() or "Failed to locate iframe" in orig:
        return f"iframe: {orig}"
    # Async load / timing: element or frame not ready yet
    if _is_transient_error(step_error):
        prefix = "async_load: "
    else:
        prefix = ""
    # Selector/element errors: add context for diagnosis
    parts = [f"tab={tab_id}"]
    if edge.frame_path:
        frame_summary = " > ".join(
            f"level{i}={f.selector}" for i, f in enumerate(edge.frame_path, 1)
        )
        parts.append(f"frame_path={frame_summary}")
    if edge.selector:
        parts.append(f"selector={edge.selector}")
    ctx = ", ".join(parts)
    return f"{prefix}selector: {orig} (context: {ctx})"


def summarize_transition_evidence(evidence_items: list[dict[str, Any]]) -> str:
    """Build a compact summary string from transition evidence items."""
    if not evidence_items:
        return ""
    counts: dict[str, int] = {}
    for item in evidence_items:
        ev_type = str(item.get("evidence_type") or item.get("type") or "unknown")
        counts[ev_type] = counts.get(ev_type, 0) + 1
    parts = [f"{name}:{counts[name]}" for name in sorted(counts)]
    return ", ".join(parts)


async def _lookup_transition_evidence_summary(transition_id: str | None) -> str:
    if not transition_id:
        return ""
    try:
        from graph_agent.neo4j_client.manager import GraphManager

        async with GraphManager() as manager:
            evidence_items = await manager.get_transition_evidence(transition_id)
        payload = [item.model_dump() for item in evidence_items]
        summary = summarize_transition_evidence(payload)
        return summary
    except Exception:
        return ""


def _page_for_tab(
    pages_by_tab_id: dict[str, Any],
    tab_id: str,
    fallback_tab_by_closed_tab_id: dict[str, str] | None = None,
    redirected_tab_by_tab_id: dict[str, str] | None = None,
) -> Any:
    """Return the page for a tab or raise a readable error."""
    current_tab_id = tab_id
    visited_tab_ids: set[str] = set()
    fallback_tab_by_closed_tab_id = fallback_tab_by_closed_tab_id or {}
    redirected_tab_by_tab_id = redirected_tab_by_tab_id or {}

    while True:
        if current_tab_id in visited_tab_ids:
            break
        visited_tab_ids.add(current_tab_id)

        redirected_tab_id = redirected_tab_by_tab_id.get(current_tab_id)
        if redirected_tab_id is not None:
            current_tab_id = redirected_tab_id
            continue

        page = pages_by_tab_id.get(current_tab_id)
        if page is not None:
            return page
        fallback_tab_id = fallback_tab_by_closed_tab_id.get(current_tab_id)
        if fallback_tab_id is None:
            break
        current_tab_id = fallback_tab_id

    raise ValueError(f"Target tab does not exist: {tab_id}")


async def _run_with_http_wait(
    page: Any,
    action: Callable[[], Awaitable[None]],
    *,
    timeout_ms: int,
) -> None:
    """Run an action and wait for the resulting HTTP request batch to settle."""
    pending: set[int] = set()
    saw_request = False
    changed = asyncio.Event()

    def _on_request(request: Any) -> None:
        nonlocal saw_request
        if not _is_http_request(request):
            return
        saw_request = True
        pending.add(id(request))
        changed.set()

    def _on_request_done(request: Any) -> None:
        if not _is_http_request(request):
            return
        pending.discard(id(request))
        changed.set()

    page.on("request", _on_request)
    page.on("requestfinished", _on_request_done)
    page.on("requestfailed", _on_request_done)
    try:
        await action()

        observe_deadline = monotonic() + (NETWORK_OBSERVE_MS / 1000)
        while not saw_request and monotonic() < observe_deadline:
            timeout = max(
                0.0, min(observe_deadline - monotonic(), NETWORK_POLL_MS / 1000)
            )
            if timeout == 0:
                break
            try:
                await asyncio.wait_for(changed.wait(), timeout=timeout)
            except TimeoutError:
                continue
            changed.clear()

        request_deadline = monotonic() + (timeout_ms / 1000)
        while saw_request and pending:
            timeout = max(
                0.0, min(request_deadline - monotonic(), NETWORK_POLL_MS / 1000)
            )
            if timeout == 0:
                raise TimeoutError("Timed out waiting for HTTP requests to finish")
            try:
                await asyncio.wait_for(changed.wait(), timeout=timeout)
            except TimeoutError:
                continue
            changed.clear()
    finally:
        _remove_listener(page, "request", _on_request)
        _remove_listener(page, "requestfinished", _on_request_done)
        _remove_listener(page, "requestfailed", _on_request_done)




async def _do_playback_action(loc, action_type: str, value: str = "", page=None) -> None:
    if action_type == "fill":
        await loc.fill(value)
    elif action_type == "click":
        await loc.click()
    elif action_type == "select":
        await loc.select_option(value)
    elif action_type == "rich_text":
        await loc.click()
        kb = page.keyboard  # type: ignore[union-attr]
        await kb.press("Control+a")
        await kb.type(value)

def _capture_login_response_sync(response, login_error_message: list) -> None:
    url = str(getattr(response, "url", "") or "").lower()
    if "/project/login" not in url:
        return
    try:
        body = response.text()
    except Exception:
        return
    parsed = _extract_login_error_message(body)
    if parsed:
        login_error_message[0] = parsed
