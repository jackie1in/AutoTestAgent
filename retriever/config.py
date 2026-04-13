from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CollectionConfig:
    min_score: float | None = None
    top_k: int | None = None
    enabled: bool = True


@dataclass
class VectorConfig:
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_dimension: int = 1536
    embedding_cache_path: str | None = ".autotestagent/embedding_cache.db"
    
    backend: str = "neo4j"  # "neo4j" | "memory" | "chroma" | "qdrant" | "milvus"
    
    chroma_path: str | None = None
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    milvus_uri: str | None = None
    
    default_top_k: int = 10
    default_min_score: float = 0.7
    
    collections: dict[str, CollectionConfig] = field(default_factory=dict)
