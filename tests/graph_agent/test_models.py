"""Tests for GraphEdge model (T1: intent nullable + intent_failure_reason)."""

import pytest
from graph_agent.models import (
    ActionType,
    BusinessTemplate,
    BusinessTemplateStep,
    ElementSnapshot,
    FrameLocatorSnapshot,
    GraphEdge,
    GraphNode,
    GraphData,
    Intent,
    TabActionType,
    TabSnapshot,
)


def test_graph_edge_with_intent_none_new_sample():
    """New sample: intent=None with intent_failure_reason should construct."""
    edge = GraphEdge(
        source="https://a.com",
        target="https://b.com",
        selector="#btn",
        action=ActionType.CLICK,
        intent=None,
        intent_failure_reason="LLM output format invalid",
    )
    assert edge.intent is None
    assert edge.intent_failure_reason == "LLM output format invalid"


def test_graph_edge_with_intent_old_sample():
    """Old sample: intent=Intent(...) without failure reason should construct."""
    intent = Intent(
        raw="click login button",
        verb="Click",
        object="Login button",
        summary="Click login button",
    )
    edge = GraphEdge(
        source="https://a.com",
        target="https://b.com",
        selector="#login",
        action=ActionType.CLICK,
        intent=intent,
        intent_failure_reason=None,
    )
    assert edge.intent is not None
    assert edge.intent.summary == "Click login button"
    assert edge.intent_failure_reason is None


def test_graph_edge_from_json_with_intent_null():
    """Pydantic must accept intent=null in JSON."""
    json_str = """{
        "source": "https://a.com",
        "target": "https://b.com",
        "selector": "#btn",
        "action": "click",
        "intent": null,
        "intent_failure_reason": "parse failed"
    }"""
    edge = GraphEdge.model_validate_json(json_str)
    assert edge.intent is None
    assert edge.intent_failure_reason == "parse failed"


def test_graph_edge_json_roundtrip_with_null_intent():
    """Round-trip: serialize and deserialize preserves intent=null."""
    edge = GraphEdge(
        source="a",
        target="b",
        selector="button",
        action=ActionType.CLICK,
        intent=None,
        intent_failure_reason="reason",
    )
    json_str = edge.model_dump_json()
    restored = GraphEdge.model_validate_json(json_str)
    assert restored.intent is None
    assert restored.intent_failure_reason == "reason"


def test_graph_data_with_null_intent_edges():
    """GraphData containing edges with intent=null should construct."""
    nodes = [
        GraphNode(id="a", url="https://a.com", title="A"),
        GraphNode(id="b", url="https://b.com", title="B"),
    ]
    edges = [
        GraphEdge(
            source="a",
            target="b",
            selector="#x",
            action=ActionType.CLICK,
            intent=None,
            intent_failure_reason="failed",
        ),
    ]
    data = GraphData(nodes=nodes, edges=edges)
    assert len(data.edges) == 1
    assert data.edges[0].intent is None
    assert data.edges[0].intent_failure_reason == "failed"


def test_graph_edge_defaults():
    """Optional fields default to None when omitted."""
    edge = GraphEdge(
        source="a",
        target="b",
        selector="x",
        action=ActionType.NAVIGATE,
    )
    assert edge.intent is None
    assert edge.intent_failure_reason is None
    assert edge.param_name is None
    assert edge.action_value is None
    assert edge.element is None
    assert edge.constraints is None


def test_graph_edge_supports_element_snapshot_and_recorded_value():
    """GraphEdge should persist param_name, action_value and element snapshot."""
    edge = GraphEdge(
        source="a",
        target="b",
        selector="[name='username']",
        action=ActionType.FILL,
        param_name="username",
        action_value="tomsmith",
        element=ElementSnapshot(
            selector="[name='username']",
            css_selector="input[name='username']",
            name="username",
            id="user",
            type="text",
            attributes={"name": "username", "id": "user", "type": "text"},
        ),
    )
    assert edge.param_name == "username"
    assert edge.action_value == "tomsmith"
    assert edge.element is not None
    assert edge.element.name == "username"
    assert edge.element.attributes["id"] == "user"


def test_graph_edge_defaults_frame_path_to_empty_list():
    """GraphEdge should default frame_path to an empty list."""
    edge = GraphEdge(
        source="a",
        target="b",
        selector="#submit",
        action=ActionType.CLICK,
    )

    assert edge.frame_path == []


def test_graph_edge_tab_context_defaults_to_primary_tab():
    """GraphEdge should default to the primary tab context."""
    edge = GraphEdge(
        source="a",
        target="b",
        selector="#submit",
        action=ActionType.CLICK,
    )

    assert edge.tab_id == "tab-0"
    assert edge.target_tab_id is None
    assert edge.tab_action is None
    assert edge.tab is None


def test_graph_edge_supports_tab_context_lifecycle_metadata():
    """GraphEdge should preserve tab lifecycle metadata."""
    edge = GraphEdge(
        source="a",
        target="b",
        selector="a[target='_blank']",
        action=ActionType.CLICK,
        tab_id="tab-0",
        target_tab_id="tab-2",
        tab_action=TabActionType.OPEN,
        tab=TabSnapshot(
            tab_id="tab-2",
            opener_tab_id="tab-0",
            url="https://example.com/reports",
            title="Reports",
        ),
    )

    assert edge.tab_id == "tab-0"
    assert edge.target_tab_id == "tab-2"
    assert edge.tab_action == TabActionType.OPEN
    assert edge.tab is not None
    assert edge.tab.tab_id == "tab-2"
    assert edge.tab.opener_tab_id == "tab-0"
    assert edge.tab.url == "https://example.com/reports"
    assert edge.tab.title == "Reports"


def test_graph_edge_tab_context_rejects_open_without_target_tab_id():
    """OPEN tab actions should require target_tab_id."""
    with pytest.raises(ValueError, match="target_tab_id"):
        GraphEdge(
            source="a",
            target="b",
            selector="a[target='_blank']",
            action=ActionType.CLICK,
            tab_action=TabActionType.OPEN,
            target_tab_id=None,
        )


def test_graph_edge_tab_context_rejects_switch_without_target_tab_id():
    """SWITCH tab actions should require target_tab_id."""
    with pytest.raises(ValueError, match="target_tab_id"):
        GraphEdge(
            source="a",
            target="b",
            selector="#tab-2",
            action=ActionType.CLICK,
            tab_action=TabActionType.SWITCH,
            target_tab_id=None,
        )


def test_graph_edge_tab_context_rejects_mismatched_tab_snapshot_target():
    """tab.tab_id should match target_tab_id when both are present."""
    with pytest.raises(ValueError, match="target_tab_id"):
        GraphEdge(
            source="a",
            target="b",
            selector="a[target='_blank']",
            action=ActionType.CLICK,
            tab_action=TabActionType.OPEN,
            target_tab_id="tab-2",
            tab=TabSnapshot(
                tab_id="tab-3",
                opener_tab_id="tab-0",
                url="https://example.com/reports",
                title="Reports",
            ),
        )


def test_element_snapshot_and_edge_preserve_nested_frame_path():
    """ElementSnapshot and GraphEdge should preserve nested iframe paths."""
    frame_path = [
        FrameLocatorSnapshot(
            selector="iframe[name='outer']",
            name="outer",
        ),
        FrameLocatorSnapshot(
            selector="iframe[name='inner']",
            name="inner",
        ),
    ]

    element = ElementSnapshot(
        selector="#submit",
        frame_path=frame_path,
    )
    edge = GraphEdge(
        source="a",
        target="b",
        selector="#submit",
        action=ActionType.CLICK,
        frame_path=frame_path,
        element=element,
    )

    assert [frame.selector for frame in edge.frame_path] == [
        "iframe[name='outer']",
        "iframe[name='inner']",
    ]
    assert edge.element is not None
    assert [frame.name for frame in edge.element.frame_path] == ["outer", "inner"]


def test_graph_edge_rejects_mismatched_element_frame_path():
    """GraphEdge should reject divergent frame paths from its element snapshot."""
    with pytest.raises(ValueError, match="frame_path"):
        GraphEdge(
            source="a",
            target="b",
            selector="#submit",
            action=ActionType.CLICK,
            frame_path=[FrameLocatorSnapshot(selector="iframe[name='outer']")],
            element=ElementSnapshot(
                selector="#submit",
                frame_path=[FrameLocatorSnapshot(selector="iframe[name='inner']")],
            ),
        )


def test_business_template_step_constructs():
    """BusinessTemplateStep should capture replay-critical edge fields."""
    step = BusinessTemplateStep(
        edge_id="step-3",
        source="login",
        target="secure",
        selector="button[type='submit']",
        action=ActionType.CLICK,
        intent_key="auth.submit.login",
        param_name=None,
    )
    assert step.edge_id == "step-3"
    assert step.action == ActionType.CLICK
    assert step.intent_key == "auth.submit.login"


def test_business_template_json_roundtrip():
    """BusinessTemplate should round-trip nested steps, slots and confidence."""
    template = BusinessTemplate(
        template_id="tpl-auth-login-001",
        business_key="auth.login",
        summary="用户登录流程",
        entry_node="https://example.com/login",
        exit_node="https://example.com/secure",
        path_length=3,
        confidence=0.91,
        steps=[
            BusinessTemplateStep(
                edge_id="step-1",
                source="https://example.com/login",
                target="State 1",
                selector="#username",
                action=ActionType.FILL,
                intent_key="auth.fill.username",
                param_name="username",
            ),
            BusinessTemplateStep(
                edge_id="step-2",
                source="State 1",
                target="State 2",
                selector="#password",
                action=ActionType.FILL,
                intent_key="auth.fill.password",
                param_name="password",
            ),
            BusinessTemplateStep(
                edge_id="step-3",
                source="State 2",
                target="https://example.com/secure",
                selector="button[type='submit']",
                action=ActionType.CLICK,
                intent_key="auth.submit.login",
                param_name=None,
            ),
        ],
        slots={"username": 0, "password": 1, "submit": "step-3"},
        evidence={
            "selectors": ["#username", "#password", "button[type='submit']"],
            "intent_keys": [
                "auth.fill.username",
                "auth.fill.password",
                "auth.submit.login",
            ],
        },
    )
    restored = BusinessTemplate.model_validate_json(template.model_dump_json())
    assert restored.business_key == "auth.login"
    assert restored.path_length == 3
    assert restored.confidence == 0.91
    assert len(restored.steps) == 3
    assert restored.steps[0].param_name == "username"
    assert restored.slots["submit"] == "step-3"
    assert restored.evidence["intent_keys"][-1] == "auth.submit.login"


def test_business_template_json_roundtrip_preserves_dependencies():
    """BusinessTemplate should preserve explicit prerequisite template keys."""
    template = BusinessTemplate(
        template_id="tpl-project-dashboard",
        business_key="project.dashboard.open",
        summary="进入项目看板",
        entry_node="secure",
        exit_node="dashboard",
        path_length=1,
        confidence=0.88,
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

    restored = BusinessTemplate.model_validate_json(template.model_dump_json())

    assert restored.depends_on == ["auth.login"]


def test_graph_node_supports_opaque_state_id_with_real_url():
    """GraphNode should allow state id and page url to differ."""
    node = GraphNode(
        id="state-login-password-filled",
        url="https://example.com/login",
        title="Login",
    )
    assert node.id == "state-login-password-filled"
    assert node.url == "https://example.com/login"
