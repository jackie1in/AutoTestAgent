from __future__ import annotations
import math
from retriever.base import VectorRetriever
from retriever.types import VectorEntry, VectorResult, VectorSearchRequest


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class InMemoryVectorRetriever(VectorRetriever):
    """
    In-memory vector retriever using brute-force cosine similarity.
    Suitable for testing and small datasets (< 1000 entries).
    Data is lost when the process exits.
    """

    def __init__(self):
        self._collections: dict[str, dict[str, VectorEntry]] = {}

    async def upsert(self, collection: str, entries: list[VectorEntry]) -> int:
        if collection not in self._collections:
            self._collections[collection] = {}
        store = self._collections[collection]
        count = 0
        for entry in entries:
            store[entry.id] = entry
            count += 1
        return count

    async def search(self, request: VectorSearchRequest) -> list[VectorResult]:
        store = self._collections.get(request.collection, {})
        if not store:
            return []
        
        scored: list[tuple[float, VectorEntry]] = []
        for entry in store.values():
            if request.filter:
                skip = False
                for k, v in request.filter.items():
                    if entry.metadata.get(k) != v:
                        skip = True
                        break
                if skip:
                    continue
            
            score = _cosine_similarity(request.query_vector, entry.vector)
            if score >= request.min_score:
                scored.append((score, entry))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        results: list[VectorResult] = []
        for score, entry in scored[: request.top_k]:
            results.append(VectorResult(
                id=entry.id,
                score=score,
                metadata=entry.metadata,
                vector=entry.vector if request.include_vectors else None,
            ))
        return results

    async def delete(self, collection: str, ids: list[str]) -> int:
        store = self._collections.get(collection, {})
        count = 0
        for id_ in ids:
            if id_ in store:
                del store[id_]
                count += 1
        return count

    async def count(self, collection: str) -> int:
        return len(self._collections.get(collection, {}))

    async def ensure_collection(self, collection: str, dimension: int, **kwargs) -> None:
        if collection not in self._collections:
            self._collections[collection] = {}

    async def drop_collection(self, collection: str) -> None:
        self._collections.pop(collection, None)
