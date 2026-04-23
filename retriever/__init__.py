from retriever.types import VectorEntry, VectorResult, VectorSearchRequest
from retriever.base import EmbeddingProvider, VectorRetriever
from retriever.embedding import OpenAIEmbeddingProvider, CachedEmbeddingProvider
from retriever.memory import InMemoryVectorRetriever
from retriever.config import VectorConfig, CollectionConfig

try:
    from retriever.graphrag import (
        GraphRAGQuery,
        GraphRAGQueryEngine,
        GraphRAGIndexManager,
        GraphRAGEmbedder,
        IntentBasedRetriever,
    )
except Exception:
    GraphRAGQuery = None  # type: ignore[misc,assignment]
    GraphRAGQueryEngine = None  # type: ignore[misc,assignment]
    GraphRAGIndexManager = None  # type: ignore[misc,assignment]
    GraphRAGEmbedder = None  # type: ignore[misc,assignment]
    IntentBasedRetriever = None  # type: ignore[misc,assignment]

# Optional external Vector DB adapters
# Usage:
#   from retriever.chroma_vector import ChromaVectorRetriever
#   from retriever.qdrant_vector import QdrantVectorRetriever
#   from retriever.milvus_vector import MilvusVectorRetriever

__all__ = [
    "VectorEntry", "VectorResult", "VectorSearchRequest",
    "EmbeddingProvider", "VectorRetriever",
    "OpenAIEmbeddingProvider", "CachedEmbeddingProvider",
    "InMemoryVectorRetriever",
    "VectorConfig", "CollectionConfig",
    # GraphRAG (neo4j-graphrag based)
    "GraphRAGQuery",
    "GraphRAGQueryEngine",
    "GraphRAGIndexManager",
    "GraphRAGEmbedder",
    "IntentBasedRetriever",
]
