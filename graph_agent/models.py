from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

class ActionType(str, Enum):
    CLICK = "click"
    FILL = "fill"
    NAVIGATE = "navigate"
    UNKNOWN = "unknown"

class ElementConstraints(BaseModel):
    format: str | None = None  # email, phone, password
    masked: bool = False

class Intent(BaseModel):
    """Structured business intent."""
    raw: str = Field(..., description="Original thought from agent")
    verb: str = Field(..., description="Action verb (e.g. Login, Submit, Search)")
    object: str = Field(..., description="Target object (e.g. Form, Button, Item)")
    summary: str = Field(..., description="Human-readable summary")
    key: str | None = Field(default=None, description="Normalized intent key (e.g. fill_username, submit_login)")
    confidence: float | None = Field(default=None, description="Intent confidence score in [0, 1]")

class GraphEdge(BaseModel):
    """Edge data representing an interaction."""
    source: str
    target: str
    selector: str
    action: ActionType
    intent: Intent | None = None
    intent_failure_reason: str | None = None
    data_key: str | None = None
    constraints: ElementConstraints | None = None

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
