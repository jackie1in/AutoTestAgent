from __future__ import annotations

import pytest

def test_format_replay_error_tab():
    """Tab errors should be prefixed with 'tab:'."""
    err = ValueError("Target tab does not exist: tab-x")
    edge = GraphEdge(source="a", target="b", selector="#btn", action=ActionType.CLICK)
    assert (
        _format_replay_error(err, edge, "tab-x")
        == "tab: Target tab does not exist: tab-x"
    )


def test_format_replay_error_iframe():
    """Iframe errors should be prefixed with 'iframe:'."""
    err = ValueError("Failed to locate iframe level 2: iframe[name='inner']")
    edge = GraphEdge(
        source="a",
        target="b",
        selector="#submit",
        action=ActionType.CLICK,
        frame_path=[
            FrameLocatorSnapshot(selector="iframe[name='outer']"),
            FrameLocatorSnapshot(selector="iframe[name='inner']"),
        ],
    )
    assert "iframe:" in _format_replay_error(err, edge, "tab-0")


def test_format_replay_error_selector_with_context():
    """Selector errors should include tab, frame_path, selector context."""
    err = RuntimeError("Locator timed out: waiting for selector '#missing'")
    edge = GraphEdge(
        source="a",
        target="b",
        selector="#missing",
        action=ActionType.CLICK,
        tab_id="tab-1",
        frame_path=[FrameLocatorSnapshot(selector="iframe#f1")],
    )
    msg = _format_replay_error(err, edge, "tab-1")
    assert "selector:" in msg
    assert "tab=tab-1" in msg
    assert "frame_path=level1=iframe#f1" in msg
    assert "selector=#missing" in msg


@pytest.mark.asyncio
async def test_playback_reports_missing_target_tab(monkeypatch: pytest.MonkeyPatch):
    """引用不存在的 tab_id 时，应返回明确错误而不是裸 KeyError。"""
    page = _FakePage()
    _install_fake_playwright(monkeypatch, page)
    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#confirm",
            action=ActionType.CLICK,
            tab_id="tab-x",
        )
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is False
    assert result["error"] == "tab: Target tab does not exist: tab-x"


@pytest.mark.asyncio
async def test_playback_reports_selector_not_found_with_context(
    monkeypatch: pytest.MonkeyPatch,
):
    """Selector not found should return error with tab/frame_path/selector context."""
    page = _FakePage()
    _install_fake_playwright(monkeypatch, page)

    async def _raise_selector_error(fake_page: _FakePage) -> None:
        raise RuntimeError("Locator timed out: waiting for selector '#nonexistent'")

    page.click_hooks["#nonexistent"] = _raise_selector_error

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#nonexistent",
            action=ActionType.CLICK,
            tab_id="tab-0",
        )
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is False
    assert "selector:" in result["error"]
    assert "tab=tab-0" in result["error"]
    assert "selector=#nonexistent" in result["error"]


@pytest.mark.asyncio
async def test_playback_close_child_tab_restores_parent_for_followup_click(
    monkeypatch: pytest.MonkeyPatch,
):
    """CLOSE 子标签后，后续普通动作应恢复到父标签页执行。"""
    page = _FakePage()
    popup = _FakePage()
    popup.url = "https://a.com/popup"
    page.popup_page = popup

    async def _parent_after_close(fake_page: _FakePage) -> None:
        if not popup.closed:
            raise RuntimeError("popup still open; close/recover logic missing")
        fake_page.url = "https://a.com/parent/after-close"

    page.click_hooks["#back-on-parent"] = _parent_after_close
    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#open-popup",
            action=ActionType.CLICK,
            tab_id="tab-0",
            target_tab_id="tab-1",
            tab_action=TabActionType.OPEN,
        ),
        GraphEdge(
            source="b",
            target="c",
            selector="#confirm",
            action=ActionType.CLICK,
            tab_id="tab-1",
        ),
        GraphEdge(
            source="c",
            target="d",
            selector="",
            action=ActionType.CLICK,
            tab_id="tab-1",
            tab_action=TabActionType.CLOSE,
        ),
        GraphEdge(
            source="d",
            target="e",
            selector="#back-on-parent",
            action=ActionType.CLICK,
            tab_id="tab-0",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert ("click", "#open-popup") in page.events
    assert ("click", "#confirm") in popup.events
    assert ("close",) in popup.events
    assert ("click", "#back-on-parent") in page.events
    assert result["actual_url"] == "https://a.com/parent/after-close"


@pytest.mark.asyncio
async def test_playback_close_child_tab_falls_back_to_parent_when_followup_still_uses_child_tab_id(
    monkeypatch: pytest.MonkeyPatch,
):
    """CLOSE 子标签后，后续仍写 child tab_id 也应自动回落到父标签页执行。"""
    page = _FakePage()
    popup = _FakePage()
    popup.url = "https://a.com/popup"
    page.popup_page = popup

    async def _parent_after_close(fake_page: _FakePage) -> None:
        if not popup.closed:
            raise RuntimeError("popup still open; close/recover logic missing")
        fake_page.url = "https://a.com/parent/fallback-after-close"

    page.click_hooks["#back-on-parent"] = _parent_after_close
    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#open-popup",
            action=ActionType.CLICK,
            tab_id="tab-0",
            target_tab_id="tab-1",
            tab_action=TabActionType.OPEN,
        ),
        GraphEdge(
            source="b",
            target="c",
            selector="#confirm",
            action=ActionType.CLICK,
            tab_id="tab-1",
        ),
        GraphEdge(
            source="c",
            target="d",
            selector="",
            action=ActionType.CLICK,
            tab_id="tab-1",
            tab_action=TabActionType.CLOSE,
        ),
        GraphEdge(
            source="d",
            target="e",
            selector="#back-on-parent",
            action=ActionType.CLICK,
            tab_id="tab-1",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert ("click", "#open-popup") in page.events
    assert ("click", "#confirm") in popup.events
    assert ("close",) in popup.events
    assert ("click", "#back-on-parent") in page.events
    assert ("click", "#back-on-parent") not in popup.events
    assert result["actual_url"] == "https://a.com/parent/fallback-after-close"


@pytest.mark.asyncio
async def test_playback_fails_fast_when_login_submit_does_not_leave_login_page(
    monkeypatch: pytest.MonkeyPatch,
):
    """When auth submit keeps user on login page, playback should fail before downstream steps."""
    page = _FakePage()
    _install_fake_playwright(monkeypatch, page)
    edge_list = [
        GraphEdge(
            source="login",
            target="login-user",
            selector="#username",
            action=ActionType.FILL,
            intent=_make_intent("Fill username", key="auth.fill.username"),
            param_name="username",
        ),
        GraphEdge(
            source="login-user",
            target="login-pass",
            selector="#password",
            action=ActionType.FILL,
            intent=_make_intent("Fill password", key="auth.fill.password"),
            param_name="password",
        ),
        GraphEdge(
            source="login-pass",
            target="secure",
            source_url="https://example.com/login",
            target_url="https://example.com/secure",
            selector="#submit",
            action=ActionType.CLICK,
            intent=_make_intent("Submit login", key="auth.submit.login"),
        ),
        GraphEdge(
            source="secure",
            target="dashboard",
            selector="#dashboard-entry",
            action=ActionType.CLICK,
            intent=_make_intent("Open dashboard", key="project.dashboard.open"),
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={"username": "alice", "password": "secret"},
        start_url="https://example.com/login",
        wait_for_network=False,
    )

    assert result["success"] is False
    assert "login" in (result["error"] or "").lower()
    # Guard should stop playback before first post-login business click.
    assert ("click", "#dashboard-entry") not in page.events


@pytest.mark.asyncio
async def test_playback_navigate_waits_for_http_requests_before_next_action(
    monkeypatch: pytest.MonkeyPatch,
):
    """NAVIGATE should wait for spawned HTTP requests before continuing."""
    page = _FakePage()
    _install_fake_playwright(monkeypatch, page)

    async def _navigate_hook(fake_page: _FakePage) -> None:
        req = _FakeRequest("https://a.com/bootstrap")
        fake_page.events.append(("request-start", req.url))
        fake_page.emit("request", req)

        async def _finish() -> None:
            await asyncio.sleep(0.01)
            fake_page.events.append(("request-finish", req.url))
            fake_page.emit("requestfinished", req)

        asyncio.create_task(_finish())

    page.goto_hooks["https://a.com/secure"] = _navigate_hook

    edge_list = [
        GraphEdge(
            source="state-start",
            target="https://a.com/secure",
            selector="",
            action=ActionType.NAVIGATE,
            intent=_make_intent("Navigate to secure page", key="navigation.secure"),
        ),
        GraphEdge(
            source="state-secure",
            target="state-filled",
            selector="#user",
            action=ActionType.FILL,
            intent=_make_intent("Fill username", key="auth.fill.username"),
            param_name="username",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={"username": "alice"},
        start_url="https://a.com/start",
        wait_for_network=True,
    )

    assert result["success"] is True
    assert ("goto", "https://a.com/secure") in page.events
    assert page.events.index(
        ("request-finish", "https://a.com/bootstrap")
    ) < page.events.index(("fill", "#user", "alice"))


@pytest.mark.asyncio
async def test_playback_click_waits_for_triggered_http_requests(
    monkeypatch: pytest.MonkeyPatch,
):
    """CLICK should wait for its HTTP request batch before the next step."""
    page = _FakePage()
    _install_fake_playwright(monkeypatch, page)

    async def _click_hook(fake_page: _FakePage) -> None:
        req = _FakeRequest("https://a.com/api/projects")
        fake_page.events.append(("request-start", req.url))
        fake_page.emit("request", req)

        async def _finish() -> None:
            await asyncio.sleep(0.01)
            fake_page.events.append(("request-finish", req.url))
            fake_page.emit("requestfinished", req)

        asyncio.create_task(_finish())

    page.click_hooks["#btn"] = _click_hook

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#btn",
            action=ActionType.CLICK,
            intent=_make_intent("Open projects", key="project.open"),
        ),
        GraphEdge(
            source="b",
            target="c",
            selector="#user",
            action=ActionType.FILL,
            intent=_make_intent("Fill username", key="auth.fill.username"),
            param_name="username",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={"username": "alice"},
        start_url="https://a.com/start",
        wait_for_network=True,
    )

    assert result["success"] is True
    assert page.events.index(
        ("request-finish", "https://a.com/api/projects")
    ) < page.events.index(("fill", "#user", "alice"))


@pytest.mark.asyncio
async def test_playback_no_http_request_continues_without_error(
    monkeypatch: pytest.MonkeyPatch,
):
    """When an action triggers no HTTP request, playback should continue normally."""
    page = _FakePage()
    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#btn",
            action=ActionType.CLICK,
            intent=_make_intent("Open modal", key="modal.open"),
        ),
        GraphEdge(
            source="b",
            target="c",
            selector="#user",
            action=ActionType.FILL,
            intent=_make_intent("Fill username", key="auth.fill.username"),
            param_name="username",
        ),
    ]

    result = await run_playback(
        edge_list,
        test_data={"username": "alice"},
        start_url="https://a.com/start",
        wait_for_network=True,
    )

    assert result["success"] is True
    assert page.events.index(("click", "#btn")) < page.events.index(
        ("fill", "#user", "alice")
    )


def test_is_transient_error_timeout():
    """TimeoutError should be classified as transient."""
    assert _is_transient_error(TimeoutError("timed out")) is True
    assert _is_transient_error(asyncio.TimeoutError()) is True


def test_is_transient_error_locator_message():
    """Locator timeout messages should be classified as transient."""
    assert (
        _is_transient_error(RuntimeError("Locator timed out: waiting for selector"))
        is True
    )
    assert _is_transient_error(RuntimeError("Timeout 30000ms exceeded")) is True


def test_is_transient_error_non_transient():
    """Non-timeout errors should not be classified as transient."""
    assert _is_transient_error(ValueError("Target tab does not exist")) is False
    assert _is_transient_error(RuntimeError("missing frame: iframe#x")) is False


def test_is_transient_error_excludes_closed_context():
    """Closed-context errors must NOT be transient — they need recovery, not retry."""
    assert (
        _is_transient_error(
            RuntimeError(
                "Locator.click: Target page, context or browser has been closed"
            )
        )
        is False
    )
    assert _is_transient_error(RuntimeError("target closed")) is False


def test_format_replay_error_async_load_prefix():
    """Timeout/async errors should be prefixed with async_load."""
    err = RuntimeError("Locator timed out: waiting for selector '#btn'")
    edge = GraphEdge(source="a", target="b", selector="#btn", action=ActionType.CLICK)
    msg = _format_replay_error(err, edge, "tab-0")
    assert msg.startswith("async_load: ")
    assert "selector:" in msg
    assert "tab=tab-0" in msg
