from __future__ import annotations
import logging
from typing import Any

from retriever.base import VectorRetriever
from retriever.types import VectorEntry, VectorResult, VectorSearchRequest

logger = logging.getLogger(__name__)


class QdrantVectorRetriever(VectorRetriever):
    """
    Qdrant adapter for vector retrieval.
    Suitable for: high-performance, filtered large-scale retrieval.
    
    Install: pip install qdrant-client
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ):
        """
        Args:
            url: Qdrant server URL (e.g., "http://localhost:6333")
            api_key: API key for authentication
            client: Optional existing Qdrant client instance
        """
        self._url = url
        self._api_key = api_key
        self._client = client
        self._dim: int | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError:
                raise ImportError("qdrant-client package required: pip install qdrant-client")
            
            kwargs = {}
            if self._url:
                kwargs["url"] = self._url
            if self._api_key:
                kwargs["api_key"] = self._api_key
            
            self._client = QdrantClient(**kwargs)
        return self._client

    async def ensure_collection(self, collection: str, dimension: int, **kwargs) -> None:
        """Ensure Qdrant collection exists with proper configuration."""
        client = self._get_client()
        self._dim = dimension
        
        try:
            from qdrant_client.models import Distance, VectorParams
            
            # Check if collection exists
            collections = client.get_collections().collections
            exists = any(c.name == collection for c in collections)
            
            if not exists:
                # Create new collection
                distance = kwargs.get("distance", Distance.COSINE)
                client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=dimension, distance=distance),
                )
                logger.info("Created Qdrant collection %s with dimension %d", collection, dimension)
        except Exception as e:
            logger.error("Failed to ensure collection %s: %s", collection, e)
            raise

    async def upsert(self, collection: str, entries: list[VectorEntry]) -> int:
        if not entries:
            return 0
        
        try:
            from qdrant_client.models import PointStruct
        except ImportError:
            raise ImportError("qdrant-client package required: pip install qdrant-client")
        
        client = self._get_client()
        
        points = [
            PointStruct(
                id=e.id,
                vector=e.vector,
                payload=e.metadata,
            )
            for e in entries
        ]
        
        client.upsert(
            collection_name=collection,
            points=points,
        )
        
        return len(entries)

    async def search(self, request: VectorSearchRequest) -> list[VectorResult]:
        try:
            from qdrant_client.models import Filter
        except ImportError:
            raise ImportError("qdrant-client package required: pip install qdrant-client")
        
        client = self._get_client()
        
        # Build filter if provided
        qdrant_filter = None
        if request.filter:
            qdrant_filter = Filter(**request.filter)
        
        results = client.search(
            collection_name=request.collection,
            query_vector=request.query_vector,
            limit=request.top_k,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=request.include_vectors,
        )
        
        vector_results: list[VectorResult] = []
        for result in results:
            # Qdrant score is already normalized for cosine similarity
            if result.score < request.min_score:
                continue
            
            vector_results.append(VectorResult(
                id=result.id,
                score=result.score,
                metadata=result.payload or {},
                vector=result.vector if request.include_vectors else None,
            ))
        
        return vector_results

    async def delete(self, collection: str, ids: list[str]) -> int:
        client = self._get_client()
        
        # Qdrant uses UUIDs or integers, convert string IDs
        point_ids = ids
        
        client.delete(
            collection_name=collection,
            points_selector=point_ids,
        )
        
        return len(ids)

    async def count(self, collection: str) -> int:
        client = self._get_client()
        result = client.count(collection_name=collection)
        return result.count

    async def drop_collection(self, collection: str) -> None:
        client = self._get_client()
        try:
            client.delete_collection(collection_name=collection)
            logger.info("Dropped Qdrant collection %s", collection)
        except Exception as e:
            logger.warning("Failed to drop collection %s: %s", collection, e)
