from __future__ import annotations
import logging
from typing import Any, Callable

from retriever.base import EmbeddingProvider, VectorRetriever
from retriever.types import VectorEntry

logger = logging.getLogger(__name__)

EMBEDDING_TEMPLATES: dict[str, Callable] = {
    "state_fingerprints": lambda s: f"url:{s.get('url','')} title:{s.get('title','')}\n{s.get('simplified_html','')[:2000]}",
    "menu_embeddings": lambda m: f"{m.get('label','')} path:{m.get('stable_path','')} key:{m.get('menu_key','') or m.get('id','')}",
    "intent_embeddings": lambda i: f"{i.get('verb','')} {i.get('object','')}: {i.get('summary','')}",
    "zone_embeddings": lambda z: f"[{z.get('zone_type','')}] {z.get('summary','')}",
    "transition_embeddings": lambda t: f"{t.get('action','')} {t.get('selector','')}: {t.get('thought','')}",
    "checkpoint_embeddings": lambda c: f"[{c.get('layer','')}/{c.get('rule_type','')}] {c.get('description','')}",
    "entity_embeddings": lambda e: f"{e.get('name','')}: {e.get('description','')} fields={','.join(e.get('key_fields',[]))}",
}

COLLECTION_CYPHER_MAP = {
    "state_fingerprints": ("State", ["id", "url", "title"]),
    "menu_embeddings": ("Menu", ["id", "label", "stable_path", "menu_key"]),
    "intent_embeddings": ("Intent", ["id", "verb", "object", "summary"]),
    "zone_embeddings": ("Zone", ["id", "zone_type", "summary"]),
    "transition_embeddings": ("Transition", ["id", "action", "selector", "thought"]),
    "checkpoint_embeddings": ("Checkpoint", ["id", "layer", "rule_type", "description"]),
    "entity_embeddings": ("Entity", ["id", "name", "description", "key_fields"]),
}


class VectorSyncManager:
    """Keeps Neo4j nodes and Vector DB in sync."""

    def __init__(
        self,
        retriever: VectorRetriever,
        embedding: EmbeddingProvider,
        neo4j_driver: Any,
    ):
        self._retriever = retriever
        self._embedding = embedding
        self._neo4j = neo4j_driver

    async def sync_node(self, collection: str, node_id: str, text: str) -> None:
        [vec] = await self._embedding.embed([text])
        await self._retriever.upsert(
            collection,
            [VectorEntry(id=node_id, vector=vec)],
        )

    async def rebuild_collection(self, collection: str) -> int:
        if collection not in COLLECTION_CYPHER_MAP:
            raise ValueError(f"Unknown collection: {collection}")
        
        label, fields = COLLECTION_CYPHER_MAP[collection]
        template_fn = EMBEDDING_TEMPLATES[collection]
        
        async with self._neo4j.session() as session:
            field_str = ", ".join(f"n.{f} AS {f}" for f in fields)
            result = await session.run(f"MATCH (n:{label}) RETURN {field_str}")
            records = await result.data()
        
        if not records:
            return 0
        
        await self._retriever.ensure_collection(
            collection, self._embedding.dimension()
        )
        
        texts = [template_fn(r) for r in records]
        vectors = await self._embedding.embed(texts)
        
        entries = [
            VectorEntry(id=r["id"], vector=vec, metadata={k: v for k, v in r.items() if k != "id" and v is not None})
            for r, vec in zip(records, vectors)
        ]
        
        count = await self._retriever.upsert(collection, entries)
        logger.info("Rebuilt collection %s: %d entries", collection, count)
        return count

    async def incremental_sync(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for collection in COLLECTION_CYPHER_MAP:
            count = await self.rebuild_collection(collection)
            result[collection] = count
        return result
