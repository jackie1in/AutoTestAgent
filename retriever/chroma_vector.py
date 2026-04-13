from __future__ import annotations
import logging
from typing import Any

from retriever.base import VectorRetriever
from retriever.types import VectorEntry, VectorResult, VectorSearchRequest

logger = logging.getLogger(__name__)


class ChromaVectorRetriever(VectorRetriever):
    """
    Chroma DB adapter for vector retrieval.
    Suitable for: local deployment, rapid prototyping.
    
    Install: pip install chromadb
    """

    def __init__(self, client: Any | None = None, persist_directory: str | None = None):
        """
        Args:
            client: Optional existing Chroma client instance
            persist_directory: Directory to persist Chroma data (None = in-memory)
        """
        self._client = client
        self._persist_directory = persist_directory
        self._collections: dict[str, Any] = {}

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import chromadb
            except ImportError:
                raise ImportError("chromadb package required: pip install chromadb")
            
            if self._persist_directory:
                self._client = chromadb.PersistentClient(path=self._persist_directory)
            else:
                self._client = chromadb.Client()
        return self._client

    def _get_collection(self, name: str):
        if name not in self._collections:
            client = self._get_client()
            self._collections[name] = client.get_or_create_collection(name=name)
        return self._collections[name]

    async def ensure_collection(self, collection: str, dimension: int, **kwargs) -> None:
        """Chroma collections are created on-demand, but we pre-create here."""
        client = self._get_client()
        try:
            # Try to get existing collection
            coll = client.get_collection(name=collection)
            self._collections[collection] = coll
        except Exception:
            # Create new collection
            coll = client.create_collection(
                name=collection,
                metadata={"dimension": dimension, **kwargs}
            )
            self._collections[collection] = coll
            logger.info("Created Chroma collection %s with dimension %d", collection, dimension)

    async def upsert(self, collection: str, entries: list[VectorEntry]) -> int:
        if not entries:
            return 0
        
        coll = self._get_collection(collection)
        
        ids = [e.id for e in entries]
        embeddings = [e.vector for e in entries]
        metadatas = [
            {k: v for k, v in e.metadata.items() if isinstance(v, (str, int, float, bool))}
            for e in entries
        ]
        
        coll.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
        return len(entries)

    async def search(self, request: VectorSearchRequest) -> list[VectorResult]:
        coll = self._get_collection(request.collection)
        
        where_filter = None
        if request.filter:
            # Convert dict filter to Chroma where format
            where_filter = request.filter
        
        results = coll.query(
            query_embeddings=[request.query_vector],
            n_results=request.top_k,
            where=where_filter,
            include=["metadatas", "distances"],
        )
        
        # Convert Chroma results to VectorResult
        vector_results: list[VectorResult] = []
        
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        for idx, (id_, distance, metadata) in enumerate(zip(ids, distances, metadatas)):
            # Chroma returns L2 distance by default, convert to similarity score (0-1)
            # Using a simple conversion: score = 1 / (1 + distance)
            score = 1.0 / (1.0 + distance) if distance is not None else 0.0
            
            if score < request.min_score:
                continue
            
            vector_results.append(VectorResult(
                id=id_,
                score=score,
                metadata=metadata or {},
                vector=None,  # Chroma doesn't return vectors by default in query
            ))
        
        # Sort by score descending
        vector_results.sort(key=lambda x: x.score, reverse=True)
        return vector_results

    async def delete(self, collection: str, ids: list[str]) -> int:
        coll = self._get_collection(collection)
        coll.delete(ids=ids)
        return len(ids)

    async def count(self, collection: str) -> int:
        coll = self._get_collection(collection)
        return coll.count()

    async def drop_collection(self, collection: str) -> None:
        client = self._get_client()
        try:
            client.delete_collection(name=collection)
            if collection in self._collections:
                del self._collections[collection]
            logger.info("Dropped Chroma collection %s", collection)
        except Exception as e:
            logger.warning("Failed to drop collection %s: %s", collection, e)
