# Nested Iframe Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 `graph_agent` 增加嵌套 iframe 的录制、图存储与回放支持，并用自动化测试覆盖顶层、单层、多层 frame 路径。

**Architecture:** 通过在模型层引入显式 `frame_path`，把目标元素所属的 frame 父链从录制端写入 `GraphEdge` 和 `ElementSnapshot`。回放时不再只依赖顶层 `page.locator()`，而是先按 `frame_path` 逐层进入正确的 frame 上下文，再执行已有的 `fill` / `click` 流程。

**Tech Stack:** Python, Pydantic, NetworkX, Playwright async API, pytest

---

### Task 1: Frame Path Models

**Files:**
- Modify: `graph_agent/models.py`
- Test: `tests/graph_agent/test_models.py`

**Step 1: Write the failing test**

```python
def test_graph_edge_defaults_frame_path_to_empty_list():
    edge = GraphEdge(
        source="a",
        target="b",
        selector="#submit",
        action=ActionType.CLICK,
    )
    assert edge.frame_path == []


def test_element_snapshot_and_edge_preserve_nested_frame_path():
    frame = FrameLocatorSnapshot(
        selector="iframe[name='outer']",
        name="outer",
    )
    edge = GraphEdge(
        source="a",
        target="b",
        selector="#submit",
        action=ActionType.CLICK,
        frame_path=[frame],
        element=ElementSnapshot(
            selector="#submit",
            frame_path=[frame],
        ),
    )
    assert edge.frame_path[0].selector == "iframe[name='outer']"
    assert edge.element is not None
    assert edge.element.frame_path[0].name == "outer"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph_agent/test_models.py -k frame_path -v`

Expected: FAIL with `FrameLocatorSnapshot` or `frame_path` field missing.

**Step 3: Write minimal implementation**

```python
class FrameLocatorSnapshot(BaseModel):
    selector: str
    xpath: str | None = None
    x_path: str | None = None
    css_selector: str | None = None
    name: str | None = None
    id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class ElementSnapshot(BaseModel):
    ...
    frame_path: list[FrameLocatorSnapshot] = Field(default_factory=list)


class GraphEdge(BaseModel):
    ...
    frame_path: list[FrameLocatorSnapshot] = Field(default_factory=list)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph_agent/test_models.py -k frame_path -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/graph_agent/test_models.py graph_agent/models.py
git commit -m "feat: add frame path models for iframe replay"
```

### Task 2: Graph IO Persistence

**Files:**
- Modify: `graph_agent/graph/io.py`
- Test: `tests/graph_agent/test_io.py`

**Step 1: Write the failing test**

```python
def test_save_and_load_graph_preserves_edge_frame_path(tmp_path):
    graph = nx.MultiDiGraph()
    frame = FrameLocatorSnapshot(selector="iframe#iframe-outer")
    graph.add_node("a", url="https://example.com/a")
    graph.add_node("b", url="https://example.com/b")
    graph.add_edge(
        "a",
        "b",
        key="step-1",
        edge_id="step-1",
        selector="#submit",
        action=ActionType.CLICK,
        frame_path=[frame],
        element=ElementSnapshot(selector="#submit", frame_path=[frame]),
    )

    path = tmp_path / "graph.json"
    save_graph(graph, path)
    loaded = load_graph(path)
    loaded_edge = next(iter(loaded.edges(data=True)))[2]

    assert loaded_edge["frame_path"][0].selector == "iframe#iframe-outer"
    assert loaded_edge["element"].frame_path[0].selector == "iframe#iframe-outer"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph_agent/test_io.py -k frame_path -v`

Expected: FAIL because `save_graph()` / `load_graph()` drop `frame_path`.

**Step 3: Write minimal implementation**

```python
from graph_agent.models import FrameLocatorSnapshot

...
if isinstance(element_data, dict):
    element = ElementSnapshot(**element_data)

edge = GraphEdge(
    ...
    frame_path=data.get("frame_path", []),
)

G.add_edge(
    ...
    frame_path=edge.frame_path,
)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph_agent/test_io.py -k frame_path -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/graph_agent/test_io.py graph_agent/graph/io.py
git commit -m "feat: persist iframe frame paths in graph io"
```

### Task 3: Parser Frame Path Extraction

**Files:**
- Modify: `graph_agent/mapping/parser.py`
- Test: `tests/graph_agent/test_parser.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_parse_browser_use_step_extracts_nested_frame_path():
    action = {
        "click": {"element": "button"},
        "interacted_element": {
            "css_selector": "#submit",
            "attributes": {"id": "submit"},
            "frame_path": [
                {
                    "css_selector": "iframe[name='outer']",
                    "attributes": {"name": "outer"},
                },
                {
                    "css_selector": "iframe[name='inner']",
                    "attributes": {"name": "inner"},
                },
            ],
        },
    }

    edge = await parse_browser_use_step(
        action=action,
        thought={"next_goal": "Submit form"},
        source_url="https://a.com/form",
        target_url="https://a.com/done",
    )

    assert [frame.selector for frame in edge.frame_path] == [
        "iframe[name='outer']",
        "iframe[name='inner']",
    ]
    assert edge.element is not None
    assert [frame.name for frame in edge.element.frame_path] == ["outer", "inner"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph_agent/test_parser.py -k nested_frame_path -v`

Expected: FAIL because `parse_browser_use_step()` does not populate `frame_path`.

**Step 3: Write minimal implementation**

```python
def _frame_locator_snapshot_from_mapping(raw: Mapping[str, Any]) -> FrameLocatorSnapshot:
    attrs_raw = raw.get("attributes", {})
    attrs = dict(attrs_raw) if isinstance(attrs_raw, Mapping) else {}
    selector = _selector_from_element(raw)
    return FrameLocatorSnapshot(
        selector=selector,
        xpath=str(raw.get("xpath")) if raw.get("xpath") else None,
        x_path=str(raw.get("x_path")) if raw.get("x_path") else None,
        css_selector=str(raw.get("css_selector")) if raw.get("css_selector") else None,
        name=str(attrs.get("name")) if attrs.get("name") is not None else None,
        id=str(attrs.get("id")) if attrs.get("id") is not None else None,
        attributes=attrs,
    )


def _extract_frame_path_from_interacted(interacted: dict | list | object) -> list[FrameLocatorSnapshot]:
    element = _normalize_interacted_element(interacted)
    raw_frames = element.get("frame_path", [])
    if not isinstance(raw_frames, list):
        return []
    return [
        _frame_locator_snapshot_from_mapping(item)
        for item in raw_frames
        if isinstance(item, Mapping) and _selector_from_element(item)
    ]


def _element_snapshot_from_interacted(...):
    ...
    frame_path = _extract_frame_path_from_interacted(interacted)
    return ElementSnapshot(..., frame_path=frame_path)


async def parse_browser_use_step(...):
    ...
    frame_path = element.frame_path if element is not None else []
    return GraphEdge(..., element=element, frame_path=frame_path)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph_agent/test_parser.py -k nested_frame_path -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/graph_agent/test_parser.py graph_agent/mapping/parser.py
git commit -m "feat: capture iframe frame paths during mapping"
```

### Task 4: Playback Context Resolution

**Files:**
- Modify: `graph_agent/playback/engine.py`
- Test: `tests/graph_agent/test_playback.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_playback_click_uses_nested_frame_path(monkeypatch: pytest.MonkeyPatch):
    page = _FakePage()
    _install_fake_playwright(monkeypatch, page)
    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#submit",
            action=ActionType.CLICK,
            frame_path=[
                FrameLocatorSnapshot(selector="iframe[name='outer']"),
                FrameLocatorSnapshot(selector="iframe[name='inner']"),
            ],
        )
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is True
    assert page.events == [
        ("goto", "https://a.com/start"),
        ("frame", "iframe[name='outer']"),
        ("frame", "iframe[name='inner']"),
        ("click", "#submit"),
        ("browser-close",),
    ]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph_agent/test_playback.py -k nested_frame_path -v`

Expected: FAIL because `_FakePage` and `run_playback()` have no frame context support.

**Step 3: Write minimal implementation**

```python
def _locator_in_context(page: Any, selector: str, frame_path: list[FrameLocatorSnapshot]) -> Any:
    context = page
    for index, frame in enumerate(frame_path, start=1):
        frame_selector = str(frame.selector or "").strip()
        if not frame_selector:
            raise ValueError(f"Missing selector for iframe level {index}")
        context = context.frame_locator(frame_selector)
    return context.locator(selector)


...
loc = _locator_in_context(page, selector, edge.frame_path)
```

并扩展测试桩：

```python
class _FakeFrameContext:
    def frame_locator(self, selector: str) -> "_FakeFrameContext":
        self.page.events.append(("frame", selector))
        return _FakeFrameContext(self.page)

    def locator(self, selector: str) -> _FakeLocator:
        return _FakeLocator(self.page, selector)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph_agent/test_playback.py -k nested_frame_path -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/graph_agent/test_playback.py graph_agent/playback/engine.py
git commit -m "feat: replay recorded actions inside nested iframes"
```

### Task 5: Playback Error Semantics

**Files:**
- Modify: `graph_agent/playback/engine.py`
- Test: `tests/graph_agent/test_playback.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_playback_reports_missing_iframe_level(monkeypatch: pytest.MonkeyPatch):
    page = _FakePage(missing_frames={"iframe[name='inner']"})
    _install_fake_playwright(monkeypatch, page)
    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#submit",
            action=ActionType.CLICK,
            frame_path=[
                FrameLocatorSnapshot(selector="iframe[name='outer']"),
                FrameLocatorSnapshot(selector="iframe[name='inner']"),
            ],
        )
    ]

    result = await run_playback(
        edge_list,
        test_data={},
        start_url="https://a.com/start",
        wait_for_network=False,
    )

    assert result["success"] is False
    assert "iframe level 2" in (result["error"] or "")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph_agent/test_playback.py -k missing_iframe_level -v`

Expected: FAIL because frame resolution errors are not surfaced with level detail.

**Step 3: Write minimal implementation**

```python
def _resolve_playback_context(page: Any, frame_path: list[FrameLocatorSnapshot]) -> Any:
    context = page
    for index, frame in enumerate(frame_path, start=1):
        frame_selector = str(frame.selector or "").strip()
        if not frame_selector:
            raise ValueError(f"Missing selector for iframe level {index}")
        try:
            context = context.frame_locator(frame_selector)
        except Exception as exc:
            raise ValueError(f"Failed to locate iframe level {index}: {frame_selector}") from exc
    return context
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph_agent/test_playback.py -k missing_iframe_level -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/graph_agent/test_playback.py graph_agent/playback/engine.py
git commit -m "fix: clarify iframe resolution failures during replay"
```

### Task 6: End-to-End Regression Sweep

**Files:**
- Modify: `tests/graph_agent/test_parser.py`
- Modify: `tests/graph_agent/test_playback.py`
- Modify: `tests/graph_agent/test_io.py`
- Modify: `tests/graph_agent/test_models.py`

**Step 1: Run focused suites before any cleanup**

```bash
uv run pytest \
  tests/graph_agent/test_models.py \
  tests/graph_agent/test_io.py \
  tests/graph_agent/test_parser.py \
  tests/graph_agent/test_playback.py -v
```

**Step 2: Fix any remaining red tests with minimal code**

```python
# Only make changes required by failing assertions.
# Do not refactor unrelated playback, parser, or template logic.
```

**Step 3: Run the focused suites again**

Run: `uv run pytest tests/graph_agent/test_models.py tests/graph_agent/test_io.py tests/graph_agent/test_parser.py tests/graph_agent/test_playback.py -v`

Expected: PASS

**Step 4: Run lints for touched files**

Run: `ReadLints` on:
- `graph_agent/models.py`
- `graph_agent/graph/io.py`
- `graph_agent/mapping/parser.py`
- `graph_agent/playback/engine.py`
- `tests/graph_agent/test_models.py`
- `tests/graph_agent/test_io.py`
- `tests/graph_agent/test_parser.py`
- `tests/graph_agent/test_playback.py`

Expected: No new diagnostics caused by this feature.

**Step 5: Commit**

```bash
git add graph_agent/models.py graph_agent/graph/io.py graph_agent/mapping/parser.py graph_agent/playback/engine.py tests/graph_agent/test_models.py tests/graph_agent/test_io.py tests/graph_agent/test_parser.py tests/graph_agent/test_playback.py
git commit -m "feat: support nested iframe capture and replay"
```
