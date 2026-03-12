from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, model_validator

class ActionType(str, Enum):
    CLICK = "click"
    FILL = "fill"
    NAVIGATE = "navigate"
    UNKNOWN = "unknown"


class TabActionType(str, Enum):
    OPEN = "open"
    SWITCH = "switch"
    CLOSE = "close"


class ElementConstraints(BaseModel):
    format: str | None = None  # email, phone, password
    masked: bool = False


class FrameLocatorSnapshot(BaseModel):
    """Captured iframe locator metadata for nested frame replay."""

    selector: str
    xpath: str | None = None
    x_path: str | None = None
    css_selector: str | None = None
    name: str | None = None
    id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class ElementSnapshot(BaseModel):
    """Captured interacted element metadata for replay/debugging."""

    selector: str
    xpath: str | None = None
    x_path: str | None = None
    css_selector: str | None = None
    name: str | None = None
    id: str | None = None
    class_name: str | None = None
    type: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    frame_path: list[FrameLocatorSnapshot] = Field(default_factory=list)


class TabSnapshot(BaseModel):
    tab_id: str
    opener_tab_id: str | None = None
    url: str | None = None
    title: str | None = None


class Intent(BaseModel):
    """Structured business intent."""
    raw: str = Field(..., description="Original thought from agent")
    verb: str = Field(..., description="Action verb (e.g. Login, Submit, Search)")
    object: str = Field(..., description="Target object (e.g. Form, Button, Item)")
    summary: str = Field(..., description="Human-readable summary")
    key: str | None = Field(default=None, description="Normalized intent key (e.g. fill_username, submit_login)")
    confidence: float | None = Field(default=None, description="Intent confidence score in [0, 1]")


class BusinessTemplateStep(BaseModel):
    """One replayable step inside a higher-level business flow template."""

    edge_id: str | None = None
    source: str
    target: str
    selector: str
    action: ActionType
    intent_key: str | None = None
    param_name: str | None = None


class BusinessTemplate(BaseModel):
    """Derived high-level business path built from multiple atomic edges."""

    template_id: str
    business_key: str
    summary: str
    entry_node: str
    exit_node: str
    path_length: int
    confidence: float | None = None
    steps: list[BusinessTemplateStep] = Field(default_factory=list)
    slots: dict[str, int | str] = Field(default_factory=dict)
    evidence: dict[str, list[str] | str] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class GraphEdge(BaseModel):
    """Edge data representing an interaction."""
    edge_id: str | None = None
    step_index: int | None = None
    source: str
    target: str
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
    element: ElementSnapshot | None = None
    constraints: ElementConstraints | None = None

    @model_validator(mode="after")
    def _validate_frame_path_consistency(self) -> "GraphEdge":
        if self.tab_action in {TabActionType.OPEN, TabActionType.SWITCH} and self.target_tab_id is None:
            raise ValueError("target_tab_id is required when tab_action is OPEN or SWITCH")
        if self.tab is not None and self.target_tab_id is not None and self.tab.tab_id != self.target_tab_id:
            raise ValueError("tab.tab_id must match target_tab_id when both are provided")
        if self.element is None or not self.element.frame_path or not self.frame_path:
            return self
        if self.frame_path != self.element.frame_path:
            raise ValueError("frame_path must match element.frame_path when both are provided")
        return self

class GraphNode(BaseModel):
    """Node data representing a page state."""
    id: str
    url: str
    title: str | None = None

class GraphData(BaseModel):
    """Full graph structure for serialization."""
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    metadata: dict[str, Any] = {}
