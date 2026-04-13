from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorEntry:
    """Entry to be upserted into vector storage."""
    id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorResult:
    """Single vector search result."""
    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: list[float] | None = None


@dataclass
class VectorSearchRequest:
    """Search request parameters."""
    collection: str
    query_vector: list[float]
    top_k: int = 10
    min_score: float = 0.0
    filter: dict[str, Any] | None = None
    include_vectors: bool = False
