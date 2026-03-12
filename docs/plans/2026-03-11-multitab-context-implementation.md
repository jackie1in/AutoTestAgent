# Multi-Tab Context Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 `graph_agent` 增加多标签页上下文支持，使系统能够录制、存储并回放标签页的打开、切换、关闭，以及在指定 tab 内继续执行普通页面动作。

**Architecture:** 通过在 `GraphEdge` 中显式加入 `tab_id`、`target_tab_id`、`tab_action` 与 `TabSnapshot`，把标签页生命周期作为一等事件建模。回放引擎维护 `tab_id -> page` 的运行时映射，并在普通动作执行前先选定正确 tab，再叠加已有的 `frame_path + selector` 定位。

**Tech Stack:** Python, Pydantic, NetworkX, Playwright async API, pytest

---

### Task 1: Tab Context Models

**Files:**
- Modify: `graph_agent/models.py`
- Test: `tests/graph_agent/test_models.py`

**Step 1: Write the failing test**

```python
def test_graph_edge_defaults_to_primary_tab_context():
    edge = GraphEdge(
        source="a",
        target="b",
        selector="#submit",
        action=ActionType.CLICK,
    )
    assert edge.tab_id == "tab-0"
    assert edge.target_tab_id is None
    assert edge.tab_action is None


def test_graph_edge_supports_tab_lifecycle_metadata():
    edge = GraphEdge(
        source="a",
        target="b",
        selector="",
        action=ActionType.UNKNOWN,
        tab_id="tab-0",
        target_tab_id="tab-1",
        tab_action=TabActionType.OPEN,
        tab=TabSnapshot(tab_id="tab-1", opener_tab_id="tab-0", url="https://example.com/popup"),
    )
    assert edge.tab_action == TabActionType.OPEN
    assert edge.tab is not None
    assert edge.tab.opener_tab_id == "tab-0"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph_agent/test_models.py -k tab_context -v`

Expected: FAIL with missing `TabActionType`, `TabSnapshot`, or missing `GraphEdge` tab fields.

**Step 3: Write minimal implementation**

```python
class TabActionType(str, Enum):
    OPEN = "open"
    SWITCH = "switch"
    CLOSE = "close"


class TabSnapshot(BaseModel):
    tab_id: str
    opener_tab_id: str | None = None
    url: str | None = None
    title: str | None = None


class GraphEdge(BaseModel):
    ...
    tab_id: str = "tab-0"
    target_tab_id: str | None = None
    tab_action: TabActionType | None = None
    tab: TabSnapshot | None = None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph_agent/test_models.py -k tab_context -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/graph_agent/test_models.py graph_agent/models.py
git commit -m "feat: add tab context models for multi-tab playback"
```

### Task 2: Graph IO for Tab Lifecycle

**Files:**
- Modify: `graph_agent/graph/io.py`
- Test: `tests/graph_agent/test_io.py`

**Step 1: Write the failing test**

```python
def test_save_load_roundtrip_preserves_tab_context():
    graph = nx.MultiDiGraph()
    graph.add_node("a", url="https://example.com/a")
    graph.add_node("b", url="https://example.com/b")
    graph.add_edge(
        "a",
        "b",
        key="step-1",
        edge_id="step-1",
        selector="",
        action=ActionType.UNKNOWN,
        tab_id="tab-0",
        target_tab_id="tab-1",
        tab_action=TabActionType.OPEN,
        tab={"tab_id": "tab-1", "opener_tab_id": "tab-0", "url": "https://example.com/b"},
    )

    ...
    assert data["tab_id"] == "tab-0"
    assert data["target_tab_id"] == "tab-1"
    assert data["tab_action"] == TabActionType.OPEN
    assert data["tab"].opener_tab_id == "tab-0"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph_agent/test_io.py -k tab_context -v`

Expected: FAIL because graph JSON save/load drops tab fields.

**Step 3: Write minimal implementation**

```python
from graph_agent.models import TabSnapshot

...
tab_data = data.get("tab")
if isinstance(tab_data, dict):
    tab = TabSnapshot(**tab_data)
elif isinstance(tab_data, TabSnapshot):
    tab = tab_data
else:
    tab = None

edge = GraphEdge(..., tab_id=data.get("tab_id", "tab-0"), target_tab_id=data.get("target_tab_id"), tab_action=data.get("tab_action"), tab=tab)

G.add_edge(..., tab_id=edge.tab_id, target_tab_id=edge.target_tab_id, tab_action=edge.tab_action, tab=edge.tab)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph_agent/test_io.py -k tab_context -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/graph_agent/test_io.py graph_agent/graph/io.py
git commit -m "feat: persist tab lifecycle metadata in graph io"
```

### Task 3: Playback Open and Switch Tab

**Files:**
- Modify: `graph_agent/playback/engine.py`
- Test: `tests/graph_agent/test_playback.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_playback_opens_popup_and_uses_new_tab_context(monkeypatch):
    page = _FakePage()
    popup = _FakePage()
    page.popup_page = popup
    _install_fake_playwright(monkeypatch, page)

    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#quality",
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
    ]

    result = await run_playback(...)

    assert result["success"] is True
    assert ("click", "#quality") in page.events
    assert ("click", "#confirm") in popup.events
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph_agent/test_playback.py -k open_popup_tab -v`

Expected: FAIL because playback only manages one `page`.

**Step 3: Write minimal implementation**

```python
pages_by_tab_id = {"tab-0": page}

def _page_for_tab(tab_id: str) -> Any:
    if tab_id not in pages_by_tab_id:
        raise ValueError(f"Target tab does not exist: {tab_id}")
    return pages_by_tab_id[tab_id]

...
if edge.tab_action == TabActionType.OPEN:
    opener_page = _page_for_tab(edge.tab_id)
    popup = await _open_popup_from_action(opener_page, ...)
    pages_by_tab_id[edge.target_tab_id] = popup
    continue

page_for_edge = _page_for_tab(edge.tab_id)
loc = _locator_in_context(page_for_edge, selector, edge.frame_path)
```

并扩展 fake Playwright 测试桩，支持 popup/new page 返回。

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph_agent/test_playback.py -k open_popup_tab -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/graph_agent/test_playback.py graph_agent/playback/engine.py
git commit -m "feat: replay popup flows with explicit tab context"
```

### Task 4: Playback Close and Recover Active Tab

**Files:**
- Modify: `graph_agent/playback/engine.py`
- Test: `tests/graph_agent/test_playback.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_playback_closes_child_tab_and_returns_to_parent(monkeypatch):
    ...
    edge_list = [
        GraphEdge(..., tab_id="tab-0", target_tab_id="tab-1", tab_action=TabActionType.OPEN),
        GraphEdge(..., tab_id="tab-1", selector="#confirm", action=ActionType.CLICK),
        GraphEdge(..., tab_id="tab-1", target_tab_id="tab-1", tab_action=TabActionType.CLOSE, selector="", action=ActionType.UNKNOWN),
        GraphEdge(..., tab_id="tab-0", selector="#back-on-parent", action=ActionType.CLICK),
    ]
    ...
    assert ("click", "#back-on-parent") in parent_page.events
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph_agent/test_playback.py -k close_child_tab -v`

Expected: FAIL because closed tabs are not tracked and active page is not restored.

**Step 3: Write minimal implementation**

```python
opener_by_tab_id: dict[str, str | None] = {"tab-0": None}

if edge.tab_action == TabActionType.CLOSE:
    tab_to_close = edge.target_tab_id or edge.tab_id
    closing_page = _page_for_tab(tab_to_close)
    await closing_page.close()
    pages_by_tab_id.pop(tab_to_close, None)
```

并补上“关闭后再用父 tab 继续执行”的最小恢复逻辑。

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph_agent/test_playback.py -k close_child_tab -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/graph_agent/test_playback.py graph_agent/playback/engine.py
git commit -m "feat: close tabs and recover parent tab context"
```

### Task 5: Playback Tab Error Semantics

**Files:**
- Modify: `graph_agent/playback/engine.py`
- Test: `tests/graph_agent/test_playback.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_playback_reports_missing_target_tab(monkeypatch):
    page = _FakePage()
    _install_fake_playwright(monkeypatch, page)
    edge_list = [
        GraphEdge(
            source="a",
            target="b",
            selector="#submit",
            action=ActionType.CLICK,
            tab_id="tab-9",
        )
    ]
    result = await run_playback(...)
    assert result["success"] is False
    assert "Target tab does not exist: tab-9" in (result["error"] or "")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph_agent/test_playback.py -k missing_target_tab -v`

Expected: FAIL because tab lookup errors are not explicit.

**Step 3: Write minimal implementation**

```python
def _page_for_tab(tab_id: str) -> Any:
    page = pages_by_tab_id.get(tab_id)
    if page is None:
        raise ValueError(f"Target tab does not exist: {tab_id}")
    return page
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph_agent/test_playback.py -k missing_target_tab -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/graph_agent/test_playback.py graph_agent/playback/engine.py
git commit -m "fix: clarify multi-tab replay errors"
```

### Task 6: Parser and Mapping Tab Metadata

**Files:**
- Modify: `graph_agent/mapping/parser.py`
- Modify: `graph_agent/mapping/run.py`
- Test: `tests/graph_agent/test_parser.py`
- Test: `tests/graph_agent/test_run.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_parse_browser_use_step_extracts_tab_context():
    action = {
        "click": {"element": "button"},
        "interacted_element": {"css_selector": "#quality"},
        "tab_id": "tab-0",
        "target_tab_id": "tab-1",
        "tab_action": "open",
        "tab": {"tab_id": "tab-1", "opener_tab_id": "tab-0", "url": "https://example.com/quality"},
    }
    edge = await parse_browser_use_step(...)
    assert edge.tab_id == "tab-0"
    assert edge.target_tab_id == "tab-1"
    assert edge.tab_action == TabActionType.OPEN
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph_agent/test_parser.py -k tab_context -v`

Expected: FAIL because parser ignores tab metadata.

**Step 3: Write minimal implementation**

```python
tab_id = str(action.get("tab_id") or "tab-0")
target_tab_id = action.get("target_tab_id")
tab_action = action.get("tab_action")
tab_data = action.get("tab")
```

并让 `GraphEdge(...)` 写入这些字段。

**Step 4: Run parser tests to verify they pass**

Run: `uv run pytest tests/graph_agent/test_parser.py -k tab_context -v`

Expected: PASS

**Step 5: Add mapping graph propagation test**

```python
def test_build_graph_from_history_preserves_tab_context():
    ...
    assert data["tab_id"] == "tab-1"
    assert data["tab_action"] == TabActionType.OPEN
```

Run: `uv run pytest tests/graph_agent/test_run.py -k tab_context -v`

Expected: PASS

**Step 6: Commit**

```bash
git add tests/graph_agent/test_parser.py tests/graph_agent/test_run.py graph_agent/mapping/parser.py graph_agent/mapping/run.py
git commit -m "feat: capture tab lifecycle metadata during mapping"
```

### Task 7: Tab Plus Iframe Integration

**Files:**
- Modify: `tests/graph_agent/test_playback.py`
- Modify: `graph_agent/playback/engine.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_playback_uses_tab_and_nested_iframe_context(monkeypatch):
    ...
    edge_list = [
        GraphEdge(..., tab_id="tab-0", target_tab_id="tab-1", tab_action=TabActionType.OPEN),
        GraphEdge(
            source="b",
            target="c",
            selector="#submit",
            action=ActionType.CLICK,
            tab_id="tab-1",
            frame_path=[
                FrameLocatorSnapshot(selector="iframe[name='outer']"),
                FrameLocatorSnapshot(selector="iframe[name='inner']"),
            ],
        ),
    ]
    ...
    assert ("frame", "iframe[name='outer']") in popup.events
    assert ("click", "#submit") in popup.events
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph_agent/test_playback.py -k tab_and_nested_iframe -v`

Expected: FAIL if popup tab routing and iframe routing do not compose correctly.

**Step 3: Write minimal implementation**

```python
page_for_edge = _page_for_tab(edge.tab_id)
loc = _locator_in_context(page_for_edge, selector, edge.frame_path)
```

只修复组合路径需要的最小问题，不顺手重构其他逻辑。

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph_agent/test_playback.py -k tab_and_nested_iframe -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/graph_agent/test_playback.py graph_agent/playback/engine.py
git commit -m "feat: combine tab and iframe replay context"
```

### Task 8: Focused Regression and Lint Sweep

**Files:**
- Modify: `graph_agent/models.py`
- Modify: `graph_agent/graph/io.py`
- Modify: `graph_agent/mapping/parser.py`
- Modify: `graph_agent/mapping/run.py`
- Modify: `graph_agent/playback/engine.py`
- Modify: `tests/graph_agent/test_models.py`
- Modify: `tests/graph_agent/test_io.py`
- Modify: `tests/graph_agent/test_parser.py`
- Modify: `tests/graph_agent/test_run.py`
- Modify: `tests/graph_agent/test_playback.py`

**Step 1: Run focused suites**

Run:

```bash
uv run pytest \
  tests/graph_agent/test_models.py \
  tests/graph_agent/test_io.py \
  tests/graph_agent/test_parser.py \
  tests/graph_agent/test_run.py \
  tests/graph_agent/test_playback.py -v
```

Expected: PASS

**Step 2: Fix any remaining red tests with minimal code**

```python
# Only address failures directly caused by tab-context support.
# Do not refactor unrelated template, login, or iframe logic.
```

**Step 3: Run the focused suites again**

Run:

```bash
uv run pytest tests/graph_agent/test_models.py tests/graph_agent/test_io.py tests/graph_agent/test_parser.py tests/graph_agent/test_run.py tests/graph_agent/test_playback.py -v
```

Expected: PASS

**Step 4: Run lints for touched files**

Run `ReadLints` on:
- `graph_agent/models.py`
- `graph_agent/graph/io.py`
- `graph_agent/mapping/parser.py`
- `graph_agent/mapping/run.py`
- `graph_agent/playback/engine.py`
- `tests/graph_agent/test_models.py`
- `tests/graph_agent/test_io.py`
- `tests/graph_agent/test_parser.py`
- `tests/graph_agent/test_run.py`
- `tests/graph_agent/test_playback.py`

Expected: No new diagnostics introduced by this feature.

**Step 5: Commit**

```bash
git add graph_agent/models.py graph_agent/graph/io.py graph_agent/mapping/parser.py graph_agent/mapping/run.py graph_agent/playback/engine.py tests/graph_agent/test_models.py tests/graph_agent/test_io.py tests/graph_agent/test_parser.py tests/graph_agent/test_run.py tests/graph_agent/test_playback.py
git commit -m "feat: support multi-tab context capture and replay"
```
