from __future__ import annotations
from abc import ABC, abstractmethod
from retriever.types import VectorEntry, VectorResult, VectorSearchRequest


class EmbeddingProvider(ABC):
    """Converts text to vectors. Decouples embedding model from vector storage."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def dimension(self) -> int:
        ...


class VectorRetriever(ABC):
    """Storage-agnostic vector retrieval. Backend can be Neo4j Vector Index, Chroma, Qdrant, etc."""

    @abstractmethod
    async def upsert(self, collection: str, entries: list[VectorEntry]) -> int:
        ...

    @abstractmethod
    async def search(self, request: VectorSearchRequest) -> list[VectorResult]:
        ...

    @abstractmethod
    async def delete(self, collection: str, ids: list[str]) -> int:
        ...

    @abstractmethod
    async def count(self, collection: str) -> int:
        ...

    async def ensure_collection(self, collection: str, dimension: int, **kwargs) -> None:
        pass

    async def drop_collection(self, collection: str) -> None:
        remaining = await self.count(collection)
        if remaining > 0:
            # Subclasses should override for atomic drop
            pass
