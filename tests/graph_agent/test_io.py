"""Tests for Graph I/O (T3: intent=null compatibility)."""

import tempfile
from pathlib import Path

import networkx as nx

from graph_agent.graph.io import save_graph, load_graph
from graph_agent.models import (
    ActionType,
    BusinessTemplate,
    BusinessTemplateStep,
    ElementSnapshot,
    FrameLocatorSnapshot,
    Intent,
    TabActionType,
    TabSnapshot,
)


def test_save_load_roundtrip_intent_null():
    """Round-trip: save graph with intent=null edge, load and verify fields preserved."""
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com", title="A")
    G.add_node("b", url="https://b.com", title="B")
    G.add_edge(
        "a",
        "b",
        selector="#btn",
        action=ActionType.CLICK,
        intent=None,
        intent_failure_reason="LLM output format invalid",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "graph.json"
        save_graph(G, path)

        loaded = load_graph(path)
        assert loaded.number_of_nodes() == 2
        assert loaded.number_of_edges() == 1

        u, v, data = next(iter(loaded.edges(data=True)))
        assert data["intent"] is None
        assert data["intent_failure_reason"] == "LLM output format invalid"


def test_save_load_roundtrip_mixed_intents():
    """Round-trip: graph with both intent=null and intent=Intent edges."""
    intent = Intent(
        raw="click login",
        verb="Click",
        object="Login button",
        summary="Click login button",
    )
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_node("c", url="https://c.com")
    G.add_edge("a", "b", selector="#x", action=ActionType.CLICK, intent=intent, intent_failure_reason=None)
    G.add_edge("b", "c", selector="#y", action=ActionType.FILL, intent=None, intent_failure_reason="parse failed")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "graph.json"
        save_graph(G, path)

        loaded = load_graph(path)
        edges = list(loaded.edges(data=True))
        assert len(edges) == 2

        # Find edges by target
        by_target = {e[1]: e[2] for e in edges}
        assert by_target["b"]["intent"] is not None
        assert by_target["b"]["intent"].summary == "Click login button"
        assert by_target["b"]["intent_failure_reason"] is None

        assert by_target["c"]["intent"] is None
        assert by_target["c"]["intent_failure_reason"] == "parse failed"


def test_load_json_with_intent_null():
    """load_graph correctly parses JSON file containing intent=null."""
    json_content = """{
  "nodes": [
    {"id": "a", "url": "https://a.com", "title": null},
    {"id": "b", "url": "https://b.com", "title": null}
  ],
  "edges": [
    {
      "source": "a",
      "target": "b",
      "selector": "#btn",
      "action": "click",
      "intent": null,
      "intent_failure_reason": "parse failed"
    }
  ],
  "metadata": {}
}"""

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "graph.json"
        path.write_text(json_content, encoding="utf-8")

        loaded = load_graph(path)
        assert loaded.number_of_edges() == 1
        u, v, data = next(iter(loaded.edges(data=True)))
        assert data["intent"] is None
        assert data["intent_failure_reason"] == "parse failed"


def test_save_preserves_intent_null_in_json():
    """Saved JSON file explicitly contains intent=null."""
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_edge("a", "b", selector="#x", action=ActionType.CLICK, intent=None, intent_failure_reason="reason")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "graph.json"
        save_graph(G, path)

        content = path.read_text(encoding="utf-8")
        assert '"intent": null' in content
        assert '"intent_failure_reason": "reason"' in content


def test_save_load_roundtrip_t4_metadata():
    """T4: metadata with intent_missing_count, filtered_non_ui_edges, mapping_stopped, stop_reason round-trips."""
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_node("c", url="https://c.com")
    G.add_edge(
        "a",
        "b",
        selector="#x",
        action=ActionType.CLICK,
        intent=Intent(summary="Click login", raw="click", verb="Click", object="Login"),
        intent_failure_reason=None,
    )
    G.add_edge("b", "c", selector="#y", action=ActionType.FILL, intent=None, intent_failure_reason="parse failed")

    G.graph["filtered_non_ui_edges"] = 3
    G.graph["mapping_stopped"] = True
    G.graph["stop_reason"] = "unfillable form"
    G.graph["intent_missing_count"] = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "graph.json"
        save_graph(G, path)

        loaded = load_graph(path)
        assert loaded.graph["filtered_non_ui_edges"] == 3
        assert loaded.graph["mapping_stopped"] is True
        assert loaded.graph["stop_reason"] == "unfillable form"
        assert loaded.graph["intent_missing_count"] == 1


def test_save_load_roundtrip_preserves_param_name_action_value_and_element_attrs():
    """Round-trip should preserve new fill metadata on graph edges."""
    G = nx.DiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    G.add_edge(
        "a",
        "b",
        selector="[name='username']",
        action=ActionType.FILL,
        intent=None,
        intent_failure_reason=None,
        param_name="username",
        action_value="tomsmith",
        element={
            "selector": "[name='username']",
            "css_selector": "input[name='username']",
            "name": "username",
            "id": "user",
            "type": "text",
            "attributes": {"name": "username", "id": "user", "type": "text"},
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "graph.json"
        save_graph(G, path)

        loaded = load_graph(path)
        _, _, data = next(iter(loaded.edges(data=True)))
        assert data["param_name"] == "username"
        assert data["action_value"] == "tomsmith"
        assert data["element"] is not None
        assert data["element"].attributes["id"] == "user"


def test_save_load_roundtrip_preserves_edge_and_element_frame_path():
    """Round-trip should preserve nested iframe paths on edge and element snapshot."""
    G = nx.MultiDiGraph()
    G.add_node("a", url="https://a.com")
    G.add_node("b", url="https://b.com")
    frame_path = [
        FrameLocatorSnapshot(selector="iframe[name='outer']", name="outer"),
        FrameLocatorSnapshot(selector="iframe[name='inner']", name="inner"),
    ]
    G.add_edge(
        "a",
        "b",
        key="step-1",
        edge_id="step-1",
        selector="#submit",
        action=ActionType.CLICK,
        intent=None,
        frame_path=frame_path,
        element=ElementSnapshot(
            selector="#submit",
            frame_path=frame_path,
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "graph.json"
        save_graph(G, path)

        loaded = load_graph(path)
        _, _, _, data = next(iter(loaded.edges(keys=True, data=True)))
        assert [frame.selector for frame in data["frame_path"]] == [
            "iframe[name='outer']",
            "iframe[name='inner']",
        ]
        assert data["element"] is not None
        assert [frame.name for frame in data["element"].frame_path] == ["outer", "inner"]


def test_save_load_roundtrip_business_templates_metadata():
    """Business templates stored in metadata should survive graph JSON round-trip."""
    G = nx.DiGraph()
    G.add_node("login", url="https://example.com/login")
    G.add_node("secure", url="https://example.com/secure")
    G.add_edge("login", "secure", selector="#submit", action=ActionType.CLICK, intent=None)
    template = BusinessTemplate(
        template_id="tpl-auth-login",
        business_key="auth.login",
        summary="用户登录流程",
        entry_node="login",
        exit_node="secure",
        path_length=2,
        confidence=0.92,
        steps=[
            BusinessTemplateStep(
                edge_id="step-1",
                source="login",
                target="secure",
                selector="#submit",
                action=ActionType.CLICK,
                intent_key="auth.submit.login",
                param_name=None,
            )
        ],
        slots={"submit": "step-1"},
        evidence={"intent_keys": ["auth.submit.login"]},
    )
    G.graph["business_templates"] = [template.model_dump(mode="json")]
    G.graph["business_template_count"] = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "graph.json"
        save_graph(G, path)

        loaded = load_graph(path)
        assert loaded.graph["business_template_count"] == 1
        assert len(loaded.graph["business_templates"]) == 1
        restored = loaded.graph["business_templates"][0]
        assert restored["business_key"] == "auth.login"
        assert restored["steps"][0]["edge_id"] == "step-1"


def test_save_load_roundtrip_preserves_template_dependencies():
    """Business template prerequisites should survive graph JSON round-trip."""
    G = nx.DiGraph()
    G.add_node("secure", url="https://example.com/secure")
    G.add_node("dashboard", url="https://example.com/dashboard")
    G.add_edge("secure", "dashboard", selector="#project", action=ActionType.CLICK, intent=None)
    template = BusinessTemplate(
        template_id="tpl-project-dashboard",
        business_key="project.dashboard.open",
        summary="进入项目看板",
        entry_node="secure",
        exit_node="dashboard",
        path_length=1,
        confidence=0.9,
        steps=[
            BusinessTemplateStep(
                edge_id="step-10",
                source="secure",
                target="dashboard",
                selector="#project",
                action=ActionType.CLICK,
                intent_key="project.dashboard.open",
                param_name=None,
            )
        ],
        slots={},
        evidence={},
        depends_on=["auth.login"],
    )
    G.graph["business_templates"] = [template.model_dump(mode="json")]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "graph.json"
        save_graph(G, path)

        loaded = load_graph(path)
        restored = loaded.graph["business_templates"][0]
        assert restored["depends_on"] == ["auth.login"]


def test_save_load_roundtrip_preserves_opaque_state_ids_with_url_metadata():
    """Graph node ids may differ from URLs and should survive graph JSON round-trip."""
    G = nx.DiGraph()
    G.add_node("state-home", url="https://example.com/")
    G.add_node("state-login-empty", url="https://example.com/login")
    G.add_edge("state-home", "state-login-empty", selector="#enter", action=ActionType.CLICK, intent=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "graph.json"
        save_graph(G, path)

        loaded = load_graph(path)
        assert "state-home" in loaded.nodes
        assert loaded.nodes["state-home"]["url"] == "https://example.com/"
        assert loaded.nodes["state-login-empty"]["url"] == "https://example.com/login"


def test_save_load_roundtrip_preserves_tab_context():
    """Round-trip should preserve tab lifecycle metadata on multigraph edges."""
    G = nx.MultiDiGraph()
    G.add_node("a", url="https://example.com/a")
    G.add_node("b", url="https://example.com/b")
    G.add_edge(
        "a",
        "b",
        key="step-1",
        edge_id="step-1",
        selector="",
        action=ActionType.UNKNOWN,
        tab_id="tab-3",
        target_tab_id="tab-1",
        tab_action=TabActionType.OPEN,
        tab={
            "tab_id": "tab-1",
            "opener_tab_id": "tab-0",
            "url": "https://example.com/b",
            "title": "Popup",
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "graph.json"
        save_graph(G, path)

        loaded = load_graph(path)
        _, _, _, data = next(iter(loaded.edges(keys=True, data=True)))
        assert data["tab_id"] == "tab-3"
        assert data["target_tab_id"] == "tab-1"
        assert data["tab_action"] == TabActionType.OPEN
        assert data["tab"] == TabSnapshot(
            tab_id="tab-1",
            opener_tab_id="tab-0",
            url="https://example.com/b",
            title="Popup",
        )
