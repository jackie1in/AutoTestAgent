from __future__ import annotations
import logging
from typing import Any

from retriever.base import VectorRetriever
from retriever.types import VectorEntry, VectorResult, VectorSearchRequest

logger = logging.getLogger(__name__)

COLLECTION_LABEL_MAP = {
    "state_fingerprints": "State",
    "menu_embeddings": "Menu",
    "intent_embeddings": "Intent",
    "zone_embeddings": "Zone",
    "transition_embeddings": "Transition",
    "checkpoint_embeddings": "Checkpoint",
    "entity_embeddings": "Entity",
}


class Neo4jVectorRetriever(VectorRetriever):
    """
    Uses Neo4j 5.x+ built-in Vector Index.
    Suitable for graph+vector unified scheme, medium scale (< 100K vectors).
    """

    def __init__(self, driver: Any):
        self._driver = driver

    async def ensure_collection(self, collection: str, dimension: int, **kwargs) -> None:
        label = COLLECTION_LABEL_MAP.get(collection, collection)
        query = """
        CALL db.index.vector.createNodeIndex(
            $index_name, $label, 'embedding', toInteger($dimension), 'cosine'
        )
        """
        async with self._driver.session() as session:
            try:
                await session.run(
                    query,
                    index_name=collection,
                    label=label,
                    dimension=dimension,
                )
                logger.info("Created vector index %s on label %s", collection, label)
            except Exception as e:
                if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
                    logger.debug("Vector index %s already exists", collection)
                else:
                    raise

    async def upsert(self, collection: str, entries: list[VectorEntry]) -> int:
        label = COLLECTION_LABEL_MAP.get(collection, collection)
        count = 0
        async with self._driver.session() as session:
            for entry in entries:
                props = {k: v for k, v in entry.metadata.items() if isinstance(v, (str, int, float, bool))}
                await session.run(
                    f"MERGE (n:{label} {{id: $id}}) "
                    f"SET n.embedding = $embedding, n += $props",
                    id=entry.id,
                    embedding=entry.vector,
                    props=props,
                )
                count += 1
        return count

    async def search(self, request: VectorSearchRequest) -> list[VectorResult]:
        async with self._driver.session() as session:
            result = await session.run(
                "CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector) "
                "YIELD node, score "
                "WHERE score >= $min_score "
                "RETURN node.id AS id, score, properties(node) AS props "
                "ORDER BY score DESC",
                index_name=request.collection,
                top_k=request.top_k,
                query_vector=request.query_vector,
                min_score=request.min_score,
            )
            records = await result.data()
            
            results: list[VectorResult] = []
            for r in records:
                props = dict(r.get("props", {}))
                embedding = props.pop("embedding", None)
                results.append(VectorResult(
                    id=r["id"],
                    score=r["score"],
                    metadata=props,
                    vector=embedding if request.include_vectors else None,
                ))
            return results

    async def delete(self, collection: str, ids: list[str]) -> int:
        label = COLLECTION_LABEL_MAP.get(collection, collection)
        async with self._driver.session() as session:
            result = await session.run(
                f"MATCH (n:{label}) WHERE n.id IN $ids "
                f"REMOVE n.embedding "
                f"RETURN count(n) AS deleted",
                ids=ids,
            )
            record = await result.single()
            return record["deleted"] if record else 0

    async def count(self, collection: str) -> int:
        label = COLLECTION_LABEL_MAP.get(collection, collection)
        async with self._driver.session() as session:
            result = await session.run(
                f"MATCH (n:{label}) WHERE n.embedding IS NOT NULL "
                f"RETURN count(n) AS cnt"
            )
            record = await result.single()
            return record["cnt"] if record else 0

    async def drop_collection(self, collection: str) -> None:
        label = COLLECTION_LABEL_MAP.get(collection, collection)
        async with self._driver.session() as session:
            await session.run(
                f"MATCH (n:{label}) WHERE n.embedding IS NOT NULL "
                f"REMOVE n.embedding"
            )
            try:
                await session.run(f"DROP INDEX {collection} IF EXISTS")
            except Exception:
                pass
