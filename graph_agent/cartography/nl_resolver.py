"""Natural language intent resolver using GraphRAG vector search.

Bridges cartography's GraphRAG indexes (intent_embeddings) with playback's
intent-based pathfinding, enabling semantic NL-to-intent resolution.

This module uses raw Neo4j vector index queries (db.index.vector.queryNodes)
instead of neo4j-graphrag retrievers, because neo4j-graphrag 1.14.1 only
supports synchronous neo4j.Driver while this codebase uses AsyncDriver.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, TypedDict

from neo4j import AsyncDriver

logger = logging.getLogger(__name__)


class EmbedderLike(Protocol):
    model: str

    async def aembed(self, text: str) -> list[float]:
        ...


class TransitionRef(TypedDict):
    transition_id: str
    action: str
    from_state: str
    to_state: str


class NLIntentCandidate(TypedDict):
    intent_id: str
    key: str
    summary: str
    name: str
    score: float
    transitions: list[TransitionRef]


class _EmbedderAdapter:
    """Adapter to make our EmbeddingProvider compatible with neo4j-graphrag."""

    def __init__(self, provider):
        self._provider = provider

    @property
    def model(self) -> str:
        inner = getattr(self._provider, "_inner", self._provider)
        return getattr(inner, "_model", "unknown")

    async def aembed(self, text: str) -> list[float]:
        vectors = await self._provider.embed([text])
        return vectors[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._provider.embed(texts)


def _get_embedder() -> _EmbedderAdapter:
    from retriever.embedding import CachedEmbeddingProvider, OpenAIEmbeddingProvider

    inner = OpenAIEmbeddingProvider(model="embedding-3", batch_size=64)
    cached = CachedEmbeddingProvider(inner)
    return _EmbedderAdapter(cached)


def _resolve_vector_index_name(base_name: str, model: str | None = None) -> str:
    """Append sanitized model slug to index name to avoid dimension collisions."""
    if model:
        slug = model.replace("-", "_").replace("/", "_").lower()
        return f"{base_name}_{slug}"
    return base_name


class NLResolver:
    """Resolves natural language queries to intent keys using GraphRAG vector search.

    Uses Neo4j async driver directly with db.index.vector.queryNodes,
    avoiding neo4j-graphrag retrievers which require sync driver.

    Usage::

        resolver = NLResolver(driver)
        results = await resolver.resolve("建任务")
        # [{"key": "create_task", "summary": "创建新任务", "score": 0.92, ...}]
    """

    def __init__(self, driver: AsyncDriver, embedder: EmbedderLike | None = None):
        self._driver = driver
        self._embedder = embedder or _get_embedder()

    async def resolve(self, query: str, top_k: int = 3) -> list[NLIntentCandidate]:
        """Resolve NL query to matching intents via vector search on intent_embeddings.

        Args:
            query: Natural language description, e.g. "建任务", "登录并创建项目"
            top_k: Maximum number of intent candidates to return.

        Returns:
            List of matching intent dicts with keys:
            - intent_id: str
            - key: str (the intent key used by pathfinding)
            - summary: str
            - name: str
            - score: float (similarity score)
            - transitions: list[dict] (related transitions from Neo4j)

        Returns empty list if vector index is unavailable or no match found.
        """
        try:
            # 1. Embed query text
            query_vector = await self._embedder.aembed(query)
        except Exception as e:
            logger.warning("Embedding failed for NL query '%s': %s", query, e)
            return []

        # 2. Determine index name
        model = getattr(self._embedder, "model", None)
        index_name = _resolve_vector_index_name("intent_embeddings", model)

        # 3. Query Neo4j vector index
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    f"""
                    CALL db.index.vector.queryNodes($index_name, $top_k, $vector)
                    YIELD node, score
                    RETURN node.id AS intent_id,
                           node.key AS key,
                           node.summary AS summary,
                           node.name AS name,
                           score
                    """,
                    index_name=index_name,
                    top_k=top_k,
                    vector=query_vector,
                )
                records = await result.data()
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
            return []

        if not records:
            return []

        # 4. Enrich with related transitions
        normalized: list[NLIntentCandidate] = []
        for rec in records:
            intent_id = rec.get("intent_id", "")
            transitions: list[TransitionRef] = []
            try:
                async with self._driver.session() as session:
                    t_result = await session.run(
                        """
                        MATCH (i:Intent {id: $intent_id})
                        OPTIONAL MATCH (t:Transition)-[:REALIZES]->(i)
                        OPTIONAL MATCH (t)-[:FROM]->(from_state:State)
                        OPTIONAL MATCH (t)-[:TO]->(to_state:State)
                        RETURN collect(DISTINCT {
                            transition_id: t.id,
                            action: t.action,
                            from_state: from_state.url,
                            to_state: to_state.url
                        }) AS transitions
                        """,
                        intent_id=intent_id,
                    )
                    t_record = await t_result.single()
                    if t_record:
                        raw = t_record.get("transitions", [])
                        transitions = [
                            {
                                "transition_id": str(item.get("transition_id", "")),
                                "action": str(item.get("action", "")),
                                "from_state": str(item.get("from_state", "")),
                                "to_state": str(item.get("to_state", "")),
                            }
                            for item in raw
                            if isinstance(item, dict)
                        ]
            except Exception:
                pass

            normalized.append(
                {
                    "intent_id": intent_id,
                    "key": rec.get("key", ""),
                    "summary": rec.get("summary", ""),
                    "name": rec.get("name", ""),
                    "score": float(rec.get("score", 0.0)),
                    "transitions": transitions,
                }
            )
        return normalized
