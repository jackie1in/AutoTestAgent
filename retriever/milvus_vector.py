from __future__ import annotations
import logging
from typing import Any

from retriever.base import VectorRetriever
from retriever.types import VectorEntry, VectorResult, VectorSearchRequest

logger = logging.getLogger(__name__)


class MilvusVectorRetriever(VectorRetriever):
    """
    Milvus adapter for vector retrieval.
    Suitable for: ultra-large scale (million+) vector retrieval.
    
    Install: pip install pymilvus
    """

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        token: str | None = None,
        client: Any | None = None,
    ):
        """
        Args:
            uri: Milvus server URI
            token: Authentication token
            client: Optional existing Milvus client instance
        """
        self._uri = uri
        self._token = token
        self._client = client
        self._dim: int | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from pymilvus import MilvusClient
            except ImportError:
                raise ImportError("pymilvus package required: pip install pymilvus")
            
            kwargs = {"uri": self._uri}
            if self._token:
                kwargs["token"] = self._token
            
            self._client = MilvusClient(**kwargs)
        return self._client

    async def ensure_collection(self, collection: str, dimension: int, **kwargs) -> None:
        """Ensure Milvus collection exists."""
        client = self._get_client()
        self._dim = dimension
        
        try:
            # Check if collection exists
            if not client.has_collection(collection_name=collection):
                # Create new collection with auto_id=False to allow custom IDs
                from pymilvus import DataType, FieldSchema, CollectionSchema
                
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                ]
                
                # Add metadata fields
                schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
                
                client.create_collection(
                    collection_name=collection,
                    schema=schema,
                )
                
                # Create index for vector field
                client.create_index(
                    collection_name=collection,
                    index_params={
                        "metric_type": "COSINE",
                        "index_type": "IVF_FLAT",
                        "params": {"nlist": 128},
                    },
                )
                
                logger.info("Created Milvus collection %s with dimension %d", collection, dimension)
        except Exception as e:
            logger.error("Failed to ensure collection %s: %s", collection, e)
            raise

    async def upsert(self, collection: str, entries: list[VectorEntry]) -> int:
        if not entries:
            return 0
        
        client = self._get_client()
        
        # Prepare data for Milvus
        data = []
        for e in entries:
            record = {
                "id": e.id,
                "vector": e.vector,
                **{k: v for k, v in e.metadata.items() if isinstance(v, (str, int, float, bool))}
            }
            data.append(record)
        
        client.upsert(
            collection_name=collection,
            data=data,
        )
        
        return len(entries)

    async def search(self, request: VectorSearchRequest) -> list[VectorResult]:
        client = self._get_client()
        
        # Build filter expression if provided
        filter_expr = None
        if request.filter:
            # Convert dict filter to Milvus expression
            conditions = []
            for key, value in request.filter.items():
                if isinstance(value, str):
                    conditions.append(f'{key} == "{value}"')
                else:
                    conditions.append(f"{key} == {value}")
            if conditions:
                filter_expr = " and ".join(conditions)
        
        results = client.search(
            collection_name=request.collection,
            data=[request.query_vector],
            limit=request.top_k,
            filter=filter_expr,
            output_fields=["*"],  # Return all fields
        )
        
        vector_results: list[VectorResult] = []
        for result_group in results:
            for result in result_group:
                score = result.get("distance", 0)
                if score < request.min_score:
                    continue
                
                # Extract metadata (excluding id and vector)
                metadata = {k: v for k, v in result.items() if k not in ("id", "distance", "entity")}
                
                vector_results.append(VectorResult(
                    id=result.get("id"),
                    score=score,
                    metadata=metadata,
                    vector=None,  # Milvus doesn't return vectors by default in search
                ))
        
        # Sort by score descending
        vector_results.sort(key=lambda x: x.score, reverse=True)
        return vector_results

    async def delete(self, collection: str, ids: list[str]) -> int:
        client = self._get_client()
        
        # Milvus 2.3+ supports string IDs
        client.delete(
            collection_name=collection,
            ids=ids,
        )
        
        return len(ids)

    async def count(self, collection: str) -> int:
        client = self._get_client()
        result = client.query(
            collection_name=collection,
            filter="",
            output_fields=["count(*)"],
        )
        return result[0].get("count(*)", 0) if result else 0

    async def drop_collection(self, collection: str) -> None:
        client = self._get_client()
        try:
            client.drop_collection(collection_name=collection)
            logger.info("Dropped Milvus collection %s", collection)
        except Exception as e:
            logger.warning("Failed to drop collection %s: %s", collection, e)
