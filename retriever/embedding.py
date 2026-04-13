from __future__ import annotations
import hashlib
import json
import logging
import os
import sqlite3
from typing import Any

from retriever.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible embedding provider. Supports custom base_url for local deployments."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 100,
    ):
        self._model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self._api_key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY", "")
        self._base_url = base_url or os.getenv("EMBEDDING_BASE_URL") or os.getenv("LLM_BASE_URL")
        self._batch_size = batch_size
        self._client: Any = None
        self._dim: int | None = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("openai package required: pip install openai")
            
            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        all_vectors: list[list[float]] = []
        
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = await client.embeddings.create(
                model=self._model,
                input=batch,
            )
            batch_vectors = [item.embedding for item in response.data]
            all_vectors.extend(batch_vectors)
            
            if self._dim is None and batch_vectors:
                self._dim = len(batch_vectors[0])
        
        return all_vectors

    def dimension(self) -> int:
        if self._dim is not None:
            return self._dim
        model_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return model_dims.get(self._model, 1536)


class CachedEmbeddingProvider(EmbeddingProvider):
    """Caching decorator for EmbeddingProvider. Uses SHA256 of text as cache key."""

    def __init__(
        self,
        inner: EmbeddingProvider,
        cache_path: str | None = None,
    ):
        self._inner = inner
        self._memory_cache: dict[str, list[float]] = {}
        self._db: sqlite3.Connection | None = None
        
        if cache_path:
            self._db = sqlite3.connect(cache_path)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS embeddings "
                "(hash TEXT PRIMARY KEY, vector TEXT, model TEXT)"
            )
            self._db.commit()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_cached(self, text_hash: str) -> list[float] | None:
        if text_hash in self._memory_cache:
            return self._memory_cache[text_hash]
        if self._db:
            row = self._db.execute(
                "SELECT vector FROM embeddings WHERE hash = ?", (text_hash,)
            ).fetchone()
            if row:
                vec = json.loads(row[0])
                self._memory_cache[text_hash] = vec
                return vec
        return None

    def _set_cached(self, text_hash: str, vector: list[float]) -> None:
        self._memory_cache[text_hash] = vector
        if self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO embeddings (hash, vector) VALUES (?, ?)",
                (text_hash, json.dumps(vector)),
            )
            self._db.commit()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        
        for i, text in enumerate(texts):
            h = self._hash_text(text)
            cached = self._get_cached(h)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        if uncached_texts:
            new_vectors = await self._inner.embed(uncached_texts)
            for idx, vec in zip(uncached_indices, new_vectors):
                text_hash = self._hash_text(texts[idx])
                self._set_cached(text_hash, vec)
                results[idx] = vec
        
        return [r for r in results if r is not None]

    def dimension(self) -> int:
        return self._inner.dimension()
