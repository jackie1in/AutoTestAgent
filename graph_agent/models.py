from __future__ import annotations
import json
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class ActionType(str, Enum):
    CLICK = "click"
    FILL = "fill"
    SELECT = "select"
    RICH_TEXT = "rich_text"
    NAVIGATE = "navigate"
    UNKNOWN = "unknown"


class EvidenceType(str, Enum):
    DOM_DIFF = "dom_diff"
    NETWORK = "network"
    CONSOLE = "console"
    SCREENSHOT = "screenshot"
    URL_CHANGE = "url_change"
    LAYOUT = "layout"


class TabActionType(str, Enum):
    OPEN = "open"
    SWITCH = "switch"
    CLOSE = "close"


class TransitionSourceType(str, Enum):
    AUTO = "auto"
    MANUAL_GRAPH_ASSISTED = "manual_graph_assisted"
    MANUAL_RAW = "manual_raw"


class ZoneType(str, Enum):
    SEARCH_FORM = "search_form"
    DATA_TABLE = "data_table"
    DETAIL_FORM = "detail_form"
    ACTION_BAR = "action_bar"
    TAB_PANEL = "tab_panel"
    TREE_PANEL = "tree_panel"
    MODAL = "modal"


class ExplorationStatus(str, Enum):
    UNDISCOVERED = "undiscovered"
    DISCOVERED = "discovered"
    PARTIAL = "partial"
    EXPLORED = "explored"
    VALIDATED = "validated"
    STALE = "stale"


class CheckpointLayer(str, Enum):
    STRUCTURAL = "structural"
    DATA = "data"
    BEHAVIORAL = "behavioral"
    SEMANTIC = "semantic"
    ENTITY = "entity"
    LIFECYCLE = "lifecycle"


class CheckpointTiming(str, Enum):
    BEFORE = "before"
    IMMEDIATE = "immediate"
    AFTER_BLUR = "after_blur"
    AFTER_SUBMIT = "after_submit"
    AFTER = "after"


class CheckpointExpect(str, Enum):
    SHOULD_PASS = "should_pass"
    SHOULD_FAIL = "should_fail"


class Severity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class CheckpointOrigin(str, Enum):
    CARTOGRAPHY = "cartography"
    STRATEGY = "strategy"
    MANUAL = "manual"
    INFERRED = "inferred"


class TestCaseCategory(str, Enum):
    BOUNDARY = "boundary"
    EQUIVALENCE = "equivalence"
    NEGATIVE = "negative"
    FORMAT = "format"
    REQUIRED = "required"
    POSITIVE = "positive"
    CHAOS = "chaos"


class SessionFocus(str, Enum):
    BREADTH = "breadth"
    DEPTH = "depth"
    VALIDATION = "validation"


# ===== Node Models =====

class App(BaseModel):
    """Top-level application node that owns all States, Sessions, etc."""
    id: str
    name: str = ""
    entry_url: str = ""
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_session_at: datetime | None = None
    total_sessions: int = 0
    total_states: int = 0
    total_transitions: int = 0


class State(BaseModel):
    id: str
    url: str
    title: str = ""
    fingerprint: str = ""
    menu_path: list[str] = Field(default_factory=list)
    first_discovered: datetime = Field(default_factory=datetime.utcnow)
    last_visited: datetime = Field(default_factory=datetime.utcnow)
    visit_count: int = 1
    spa_route: str | None = None
    is_modal: bool = False
    parent_state_id: str | None = None
    app_id: str | None = None
    view_fingerprint: str | None = None
    data_signature: str | None = None
    ingest_version_id: str | None = None


class Transition(BaseModel):
    id: str
    selector: str = ""
    action: ActionType = ActionType.CLICK
    action_value: str | None = None
    param_name: str | None = None
    element_snapshot: str | None = None  # JSON
    frame_path: str | None = None  # JSON
    tab_id: str = "tab-0"
    target_tab_id: str | None = None
    tab_action: TabActionType | None = None
    thought: str | None = None
    confidence: float = 0.5
    first_discovered: datetime = Field(default_factory=datetime.utcnow)
    last_validated: datetime | None = None
    session_id: str | None = None
    validation_count: int = 0
    step_index: int | None = None

    # Intent inference (unified across cartography and mapping)
    intent: "Intent | None" = None
    intent_failure_reason: str | None = None
    selector_chain: list[str] = Field(default_factory=list)
    semantic_action_key: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    failed_requests: list[dict[str, Any]] = Field(default_factory=list)
    source_type: TransitionSourceType = TransitionSourceType.AUTO
    operator_id: str = "agent"
    ingest_version_id: str | None = None

    # Relationship endpoints (not stored as properties, used for graph construction)
    from_state_id: str | None = None
    to_state_id: str | None = None


class Zone(BaseModel):
    id: str
    zone_type: ZoneType = ZoneType.SEARCH_FORM
    root_selector: str = ""
    summary: str = ""
    interactive_count: int = 0
    exploration_status: ExplorationStatus = ExplorationStatus.UNDISCOVERED
    last_explored: datetime | None = None
    ingest_version_id: str | None = None


class FrameNode(BaseModel):
    id: str
    selector: str = ""
    xpath: str | None = None
    name: str | None = None
    src: str | None = None
    depth: int = 0
    parent_frame_id: str | None = None


class Entity(BaseModel):
    id: str
    name: str
    key_fields: list[str] = Field(default_factory=list)
    description: str = ""


class EntityInstance(BaseModel):
    id: str
    data: str = "{}"  # JSON
    status: str = "available"  # available | consumed | expired
    session_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def get_data(self) -> dict:
        return json.loads(self.data)


class Intent(BaseModel):
    id: str = ""
    name: str = ""
    raw: str = ""
    summary: str = ""
    verb: str = ""
    object: str = ""
    key: str = ""
    confidence: float | None = None


class Checkpoint(BaseModel):
    id: str
    layer: CheckpointLayer = CheckpointLayer.STRUCTURAL
    timing: CheckpointTiming = CheckpointTiming.AFTER
    expect: CheckpointExpect = CheckpointExpect.SHOULD_PASS
    severity: Severity = Severity.MAJOR
    rule_type: str = ""
    rule: str = "{}"  # JSON
    description: str = ""
    origin_type: CheckpointOrigin = CheckpointOrigin.CARTOGRAPHY
    session_id: str | None = None
    total_runs: int = 0
    pass_count: int = 0
    fail_count: int = 0
    flaky: bool = False
    last_result: str | None = None  # pass | fail | skip
    ingest_version_id: str | None = None

    def get_rule(self) -> dict:
        return json.loads(self.rule)


class FieldConstraint(BaseModel):
    id: str
    selector: str = ""
    field_name: str = ""
    input_type: str = "text"
    required: bool = False
    min_length: int | None = None
    max_length: int | None = None
    min_value: float | None = None
    max_value: float | None = None
    pattern: str | None = None
    step: float | None = None
    learned_constraints: str = "[]"  # JSON array
    valid_examples: list[str] = Field(default_factory=list)


class TestCase(BaseModel):
    id: str
    name: str = ""
    category: TestCaseCategory = TestCaseCategory.POSITIVE
    description: str = ""
    field_overrides: str = "{}"  # JSON
    total_runs: int = 0
    pass_count: int = 0
    fail_count: int = 0
    last_run: datetime | None = None
    last_result: str | None = None


class Session(BaseModel):
    id: str
    app_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: int = 0
    focus: SessionFocus = SessionFocus.BREADTH
    states_discovered: int = 0
    states_updated: int = 0
    transitions_discovered: int = 0
    transitions_validated: int = 0
    transitions_invalidated: int = 0
    checkpoints_generated: int = 0


class Menu(BaseModel):
    """Navigation menu item node - separate from State."""
    id: str
    label: str
    level: int = 0
    order_index: int = 0
    selector: str = ""
    menu_key: str | None = None
    app_id: str = ""
    stable_path: str | None = None
    first_discovered: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    ingest_version_id: str | None = None


class Evidence(BaseModel):
    """Observational evidence supporting a transition decision."""

    id: str
    transition_id: str
    session_id: str
    evidence_type: EvidenceType = EvidenceType.DOM_DIFF
    summary: str = ""
    payload: str = "{}"  # JSON
    confidence: float = 0.5
    created_at: datetime = Field(default_factory=datetime.utcnow)
    ingest_version_id: str | None = None


class IngestionRun(BaseModel):
    """One mapping ingestion batch for audit and replay."""

    id: str
    app_id: str
    session_id: str
    mode: str = "auto"
    source: str = "cartography"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "completed"


class TransitionEntity(BaseModel):
    """Stable transition identity across revisions."""

    stable_key: str
    app_id: str | None = None
    from_state_id: str | None = None
    to_state_id: str | None = None
    action: str = ""
    semantic_action_key: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TransitionRevision(BaseModel):
    """Versioned transition snapshot."""

    revision_id: str
    stable_key: str
    transition_id: str
    confidence: float = 0.5
    intent_key: str | None = None
    source_type: TransitionSourceType = TransitionSourceType.AUTO
    operator_id: str = "agent"
    selector: str = ""
    action: str = ""
    from_state_id: str = ""
    to_state_id: str = ""
    session_id: str = ""
    ingest_version_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True


class GraphRelease(BaseModel):
    """Consumable graph snapshot composed from ingestion/revisions."""

    id: str
    app_id: str
    base_ingest_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "active"


class CoverageSnapshot(BaseModel):
    """Per-session coverage snapshot. Persisted so SkipAdvisor / Scheduler can
    consult historical coverage without recomputing from raw zones every time."""

    id: str
    app_id: str
    session_id: str
    release_id: str = ""
    captured_at: datetime = Field(default_factory=datetime.utcnow)
    menu_coverage: float = 0.0
    zone_coverage: float = 0.0
    interaction_coverage: float = 0.0
    state_coverage: float = 0.0
    overall_completeness: float = 0.0
    transition_high: int = 0
    transition_medium: int = 0
    transition_low: int = 0
    recommendation: str = "needs_more"


# ===== Coverage Model =====

class TransitionConfidenceDistribution(BaseModel):
    high: int = 0  # >= 0.8
    medium: int = 0  # 0.4 - 0.8
    low: int = 0  # < 0.4


class CoverageReport(BaseModel):
    menu_coverage: float = 0.0
    zone_coverage: float = 0.0
    interaction_coverage: float = 0.0
    transition_confidence: TransitionConfidenceDistribution = Field(
        default_factory=TransitionConfidenceDistribution
    )
    overall_completeness: float = 0.0
    recommendation: Literal["complete", "needs_more", "needs_validation"] = "needs_more"


# ===== Mapping / Playback Models =====
# (Migrated from the former graph_agent/models.py)

class ElementConstraints(BaseModel):
    format: str | None = None
    masked: bool = False


class FrameLocatorSnapshot(BaseModel):
    selector: str
    xpath: str | None = None
    x_path: str | None = None
    css_selector: str | None = None
    name: str | None = None
    id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class ElementSnapshot(BaseModel):
    selector: str
    xpath: str | None = None
    x_path: str | None = None
    css_selector: str | None = None
    name: str | None = None
    id: str | None = None
    class_name: str | None = None
    type: str | None = None
    tag_name: str | None = None                 # HTML tag name (button, input, etc.)
    text_content: str | None = None             # Element text content
    inner_text: str | None = None               # Visible text (innerText)
    placeholder: str | None = None              # Input placeholder
    aria_label: str | None = None               # Accessibility label
    value: str | None = None                    # Current value (for inputs)
    href: str | None = None                     # For links
    title: str | None = None                    # Title attribute
    attributes: dict[str, Any] = Field(default_factory=dict)
    frame_path: list[FrameLocatorSnapshot] = Field(default_factory=list)


class TabSnapshot(BaseModel):
    tab_id: str
    opener_tab_id: str | None = None
    url: str | None = None
    title: str | None = None


class GraphEdge(BaseModel):
    """Edge representing an interaction in the mapping graph."""
    edge_id: str | None = None
    step_index: int | None = None
    source: str
    target: str
    source_url: str | None = None
    target_url: str | None = None
    selector: str
    action: ActionType
    tab_id: str = "tab-0"
    target_tab_id: str | None = None
    tab_action: TabActionType | None = None
    tab: TabSnapshot | None = None
    frame_path: list[FrameLocatorSnapshot] = Field(default_factory=list)
    intent: Intent | None = None
    context_level_used: str | None = None
    intent_failure_reason: str | None = None
    param_name: str | None = None
    action_value: str | None = None
    thought: str | None = None
    element: ElementSnapshot | None = None
    constraints: ElementConstraints | None = None

    @model_validator(mode="after")
    def _validate_frame_path_consistency(self) -> "GraphEdge":
        if (
            self.tab_action in {TabActionType.OPEN, TabActionType.SWITCH}
            and self.target_tab_id is None
        ):
            raise ValueError(
                "target_tab_id is required when tab_action is OPEN or SWITCH"
            )
        if (
            self.tab is not None
            and self.target_tab_id is not None
            and self.tab.tab_id != self.target_tab_id
        ):
            raise ValueError(
                "tab.tab_id must match target_tab_id when both are provided"
            )
        if self.element is None or not self.element.frame_path or not self.frame_path:
            return self
        if self.frame_path != self.element.frame_path:
            raise ValueError(
                "frame_path must match element.frame_path when both are provided"
            )
        return self


class GraphNode(BaseModel):
    id: str
    url: str
    title: str | None = None


class GraphData(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    metadata: dict[str, Any] = {}
