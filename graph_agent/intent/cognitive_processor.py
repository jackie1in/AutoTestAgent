from __future__ import annotations

import logging
import os
from typing import Any

from retriever.base import EmbeddingProvider, VectorRetriever
from retriever.embedding import CachedEmbeddingProvider, OpenAIEmbeddingProvider
from retriever.memory import InMemoryVectorRetriever
from retriever.types import VectorSearchRequest

logger = logging.getLogger(__name__)


class CognitiveProcessor:
    """Computes semantic fingerprints and similarity.

    Refactored to delegate to VectorRetriever and EmbeddingProvider
    instead of managing vectors directly.
    """

    def __init__(
        self,
        api_key: str | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        retriever: VectorRetriever | None = None,
    ):
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv(
            "LLM_API_KEY", ""
        )

        if embedding_provider:
            self._embedding = embedding_provider
        else:
            inner = OpenAIEmbeddingProvider(api_key=self.api_key)
            cache_path = os.getenv(
                "EMBEDDING_CACHE_PATH", ".autotestagent/embedding_cache.db"
            )
            self._embedding = CachedEmbeddingProvider(inner, cache_path=cache_path)

        self._retriever = retriever or InMemoryVectorRetriever()

    def compute_fingerprint(self, envelopes: list[dict[str, Any]]) -> dict[str, Any]:
        """Computes a semantic fingerprint for a list of semantic envelopes."""
        text_parts = []
        for env in envelopes:
            role = env.get("role", "")
            name = env.get("name", "")
            if role and name:
                text_parts.append(f"{role}:{name}")

        text_rep = " | ".join(sorted(text_parts))

        # Synchronous compatibility: return text representation
        # Actual vector computed lazily via embed()
        return {
            "text_representation": text_rep,
            "vector": [0.0] * self._embedding.dimension(),
        }

    async def compute_fingerprint_async(
        self, envelopes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Async version that actually computes embeddings."""
        text_parts = []
        for env in envelopes:
            role = env.get("role", "")
            name = env.get("name", "")
            if role and name:
                text_parts.append(f"{role}:{name}")

        text_rep = " | ".join(sorted(text_parts))
        [vector] = await self._embedding.embed([text_rep])
        return {"text_representation": text_rep, "vector": vector}

    async def find_similar_states(
        self,
        state_text: str,
        threshold: float = 0.92,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Find similar states using vector retrieval."""
        [query_vec] = await self._embedding.embed([state_text])
        results = await self._retriever.search(
            VectorSearchRequest(
                collection="state_fingerprints",
                query_vector=query_vec,
                top_k=top_k,
                min_score=threshold,
            )
        )
        return [{"id": r.id, "score": r.score, "metadata": r.metadata} for r in results]

    async def is_similar_state(
        self,
        text1: str,
        text2: str,
        threshold: float = 0.95,
    ) -> bool:
        """Check if two state descriptions are semantically similar."""
        vecs = await self._embedding.embed([text1, text2])
        if len(vecs) < 2:
            return False

        import math

        v1, v2 = vecs[0], vecs[1]
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1))
        n2 = math.sqrt(sum(b * b for b in v2))
        if n1 == 0 or n2 == 0:
            return False
        similarity = dot / (n1 * n2)
        return similarity >= threshold

    # Backward compatibility
    def _get_embedding(self, text: str) -> list[float]:
        return [0.0] * self._embedding.dimension()

    def calculate_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        import math

        v1, v2 = vec1, vec2
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1))
        n2 = math.sqrt(sum(b * b for b in v2))
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)

    def is_similar(
        self, vec1: list[float], vec2: list[float], threshold: float = 0.95
    ) -> bool:
        return self.calculate_similarity(vec1, vec2) >= threshold
