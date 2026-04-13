"""
GraphRAG implementation using neo4j-graphrag library.

Requires:
    uv pip install neo4j-graphrag

This module provides GraphRAG capabilities using the official neo4j-graphrag library:
- VectorRetriever: Pure vector search
- HybridRetriever: Vector + Fulltext search  
- VectorCypherRetriever: Vector search with custom Cypher post-processing
- Text2CypherRetriever: Natural language to Cypher query
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncDriver
from neo4j_graphrag.embeddings import OpenAIEmbeddings as Neo4jOpenAIEmbeddings
from neo4j_graphrag.retrievers import (
    VectorRetriever,
    HybridRetriever,
    VectorCypherRetriever,
)

logger = logging.getLogger(__name__)


class GraphRAGEmbedder:
    """Adapter for neo4j-graphrag embedding interface."""
    
    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small"):
        self._embedder = Neo4jOpenAIEmbeddings(
            api_key=api_key,
            model=model,
        )
    
    async def embed(self, text: str) -> list[float]:
        """Embed single text."""
        return await self._embedder.aembed(text)
    
    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        return await self._embedder.aembed_documents(texts)


class GraphRAGIndexManager:
    """Manages Neo4j vector indexes using neo4j-graphrag."""
    
    def __init__(self, driver: AsyncDriver):
        self._driver = driver
    
    async def create_vector_index(
        self,
        name: str,
        label: str,
        property_name: str = "embedding",
        dimensions: int = 1536,
        similarity_fn: str = "cosine",
    ) -> None:
        """Create a vector index.
        
        Args:
            name: Index name
            label: Node label (e.g., "State", "Menu", "Intent")
            property_name: Property storing the vector
            dimensions: Vector dimensions
            similarity_fn: Similarity function (cosine, euclidean)
        """
        async with self._driver.session() as session:
            try:
                await session.run(
                    f"CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (n:{label}) ON (n.{property_name}) "
                    f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimensions}, `vector.similarity_function`: '{similarity_fn}'}}}}"
                )
                logger.info(f"Created vector index: {name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.debug(f"Vector index {name} already exists")
                else:
                    raise
    
    async def upsert_node_vector(
        self,
        node_id: str,
        vector: list[float],
        id_property: str = "id",
        embedding_property: str = "embedding",
    ) -> None:
        """Upsert vector on a node.
        
        Args:
            node_id: Node identifier value
            vector: The embedding vector
            id_property: Property name for node identification (default: "id")
            embedding_property: Property to store the vector (default: "embedding")
        """
        async with self._driver.session() as session:
            await session.run(
                f"MATCH (n {{{id_property}: $node_id }}) "
                f"SET n.{embedding_property} = $vector",
                node_id=node_id,
                vector=vector,
            )


class GraphRAGQueryEngine:
    """GraphRAG query engine using neo4j-graphrag retrievers."""
    
    def __init__(
        self,
        driver: AsyncDriver,
        embedder: Any | None = None,
    ):
        """
        Args:
            driver: Neo4j async driver
            embedder: Embedding provider (if None, must provide for each query)
        """
        self._driver = driver
        self._embedder = embedder
    
    async def vector_search(
        self,
        query_text: str,
        index_name: str,
        top_k: int = 5,
        embedder: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Pure vector search.
        
        Args:
            query_text: Natural language query
            index_name: Name of the vector index
            top_k: Number of results
            embedder: Optional embedder override
            
        Returns:
            List of results with node info and score
        """
        emb = embedder or self._embedder
        if emb is None:
            raise ValueError("Embedder required")
        
        # Create retriever
        retriever = VectorRetriever(
            driver=self._driver,
            index_name=index_name,
            embedder=emb,
            return_properties=["id", "url", "title", "label", "name", "summary"],
        )
        
        # Search
        result = await retriever.asearch(query_text=query_text, top_k=top_k)
        
        # Convert to dict format
        return [
            {
                "id": item.metadata.get("id", ""),
                "score": item.score,
                "metadata": {k: v for k, v in item.metadata.items() if k != "id"},
            }
            for item in result.items
        ]
    
    async def hybrid_search(
        self,
        query_text: str,
        vector_index_name: str,
        fulltext_index_name: str,
        top_k: int = 5,
        embedder: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search: vector + fulltext.
        
        Args:
            query_text: Natural language query
            vector_index_name: Name of vector index
            fulltext_index_name: Name of fulltext index
            top_k: Number of results
            embedder: Optional embedder override
            
        Returns:
            List of results with combined scores
        """
        emb = embedder or self._embedder
        if emb is None:
            raise ValueError("Embedder required")
        
        # Create hybrid retriever
        retriever = HybridRetriever(
            driver=self._driver,
            vector_index_name=vector_index_name,
            fulltext_index_name=fulltext_index_name,
            embedder=emb,
            return_properties=["id", "url", "title", "label", "name", "summary"],
        )
        
        result = await retriever.asearch(query_text=query_text, top_k=top_k)
        
        return [
            {
                "id": item.metadata.get("id", ""),
                "score": item.score,
                "metadata": {k: v for k, v in item.metadata.items() if k != "id"},
            }
            for item in result.items
        ]
    
    async def vector_cypher_search(
        self,
        query_text: str,
        index_name: str,
        retrieval_query: str,
        top_k: int = 5,
        embedder: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Vector search with custom Cypher for graph traversal.
        
        Args:
            query_text: Natural language query
            index_name: Name of vector index
            retrieval_query: Custom Cypher query for post-processing
                Use $node AS node and $score AS score as parameters
            top_k: Number of results
            embedder: Optional embedder override
            
        Example retrieval_query::
            
                MATCH (n:State {{id: node.id}})-[:HAS_ZONE]->(z:Zone)
                RETURN n.id AS state_id, n.url AS url, collect(z.summary) AS zones, $score AS score
        """
        emb = embedder or self._embedder
        if emb is None:
            raise ValueError("Embedder required")
        
        retriever = VectorCypherRetriever(
            driver=self._driver,
            index_name=index_name,
            embedder=emb,
            retrieval_query=retrieval_query,
        )
        
        result = await retriever.asearch(query_text=query_text, top_k=top_k)
        
        return [
            {
                "content": item.content,
                "score": item.score,
                "metadata": item.metadata,
            }
            for item in result.items
        ]


class IntentBasedRetriever:
    """Specialized retriever for finding intents and related paths."""
    
    def __init__(
        self,
        driver: AsyncDriver,
        embedder: Any,
    ):
        self._driver = driver
        self._embedder = embedder
    
    async def find_intent(
        self,
        description: str,
        intent_index: str = "intent_embeddings",
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Find matching intents by description.
        
        Args:
            description: Natural language description of intent
            intent_index: Vector index name for intents
            top_k: Number of results
            
        Returns:
            List of matching intents with related transitions
        """
        engine = GraphRAGQueryEngine(self._driver, self._embedder)
        
        # Custom query to get intent with related transitions
        retrieval_query = """
        MATCH (i:Intent {id: node.id})
        OPTIONAL MATCH (t:Transition)-[:REALIZES]->(i)
        OPTIONAL MATCH (t)-[:FROM]->(from_state:State)
        OPTIONAL MATCH (t)-[:TO]->(to_state:State)
        RETURN i.id AS intent_id,
               i.name AS name,
               i.summary AS summary,
               i.key AS key,
               collect(DISTINCT {
                   transition_id: t.id,
                   action: t.action,
                   from_state: from_state.url,
                   to_state: to_state.url
               }) AS transitions,
               $score AS score
        """
        
        return await engine.vector_cypher_search(
            query_text=description,
            index_name=intent_index,
            retrieval_query=retrieval_query,
            top_k=top_k,
        )
    
    async def find_path_between_states(
        self,
        from_description: str,
        to_description: str,
        state_index: str = "state_fingerprints",
    ) -> list[dict[str, Any]]:
        """Find path between two states described in natural language.
        
        Args:
            from_description: Description of starting state
            to_description: Description of target state
            state_index: Vector index name for states
            
        Returns:
            Path as list of state nodes
        """
        engine = GraphRAGQueryEngine(self._driver, self._embedder)
        
        # Find start state
        start_results = await engine.vector_search(
            from_description,
            state_index,
            top_k=1,
        )
        if not start_results:
            return []
        start_id = start_results[0]["id"]
        
        # Find end state
        end_results = await engine.vector_search(
            to_description,
            state_index,
            top_k=1,
        )
        if not end_results:
            return []
        end_id = end_results[0]["id"]
        
        # Query shortest path
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH path = shortestPath(
                    (s1:State {id: $start_id})((<-[:FROM]-(:Transition)-[:TO]->)(:State))+
                    (s2:State {id: $end_id})
                )
                RETURN [n IN nodes(path) | {
                    id: n.id,
                    url: n.url,
                    title: n.title
                }] AS path_nodes
                """,
                start_id=start_id,
                end_id=end_id,
            )
            record = await result.single()
            if record and record["path_nodes"]:
                return list(record["path_nodes"])
        return []


class GraphRAGQuery:
    """High-level GraphRAG query interface using neo4j-graphrag."""
    
    def __init__(
        self,
        retriever: Any,  # Not used, kept for API consistency
        embedding: Any,  # Our embedding provider
        neo4j_driver: AsyncDriver,
    ):
        """
        Args:
            retriever: Ignored, kept for compatibility
            embedding: Embedding provider (our retriever.embedding type)
            neo4j_driver: Neo4j async driver
        """
        self._neo4j = neo4j_driver
        # Adapt our embedding provider to neo4j-graphrag interface
        self._neo4j_embedder = _EmbedderAdapter(embedding)
        self._engine = GraphRAGQueryEngine(neo4j_driver, self._neo4j_embedder)
    
    async def hybrid_search(
        self,
        question: str,
        top_k: int = 3,
        min_score: float = 0.7,
        graph_depth: int = 2,
    ) -> list[dict[str, Any]]:
        """Execute hybrid vector + graph search.
        
        Args:
            question: Natural language question
            top_k: Number of results
            min_score: Minimum similarity score (filtering done post-search)
            graph_depth: Graph expansion depth (currently for interface compatibility)
            
        Returns:
            List of results with node info, scores, and graph context
        """
        # Search intent embeddings
        intent_results = await self._engine.vector_search(
            question, "intent_embeddings", top_k
        )
        
        # Search state embeddings
        state_results = await self._engine.vector_search(
            question, "state_fingerprints", top_k
        )
        
        # Combine and deduplicate
        seen_ids = set()
        combined = []
        for r in intent_results + state_results:
            if r["id"] not in seen_ids and r["score"] >= min_score:
                seen_ids.add(r["id"])
                combined.append(r)
        
        # Graph expansion for additional context
        if graph_depth > 0 and combined:
            ids = [r["id"] for r in combined]
            async with self._neo4j.session() as session:
                result = await session.run(
                    """
                    UNWIND $ids AS nid
                    MATCH (n {id: nid})
                    OPTIONAL MATCH (n)-[r1]-(m1)
                    OPTIONAL MATCH (m1)-[r2]-(m2) WHERE m2 <> n
                    RETURN n.id AS node_id, labels(n) AS labels, properties(n) AS props,
                           collect(DISTINCT {type: type(r1), target_id: m1.id, target_labels: labels(m1)}) AS neighbors,
                           collect(DISTINCT {type: type(r2), target_id: m2.id}) AS second_hop
                    """,
                    ids=ids,
                )
                records = await result.data()
            
            # Merge graph info with vector results
            record_map = {r["node_id"]: r for r in records}
            for r in combined:
                rec = record_map.get(r["id"])
                if rec:
                    r["labels"] = rec["labels"]
                    r["properties"] = rec["props"]
                    r["neighbors"] = rec.get("neighbors", [])
                    r["second_hop"] = rec.get("second_hop", [])
        
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined
    
    async def find_path(
        self, from_description: str, to_description: str
    ) -> list[dict[str, Any]]:
        """Find shortest path between two described states.
        
        Args:
            from_description: Description of starting state
            to_description: Description of target state
            
        Returns:
            Path as list of state nodes
        """
        retriever = IntentBasedRetriever(self._neo4j, self._neo4j_embedder)
        return await retriever.find_path_between_states(
            from_description, to_description
        )


class _EmbedderAdapter:
    """Adapter to make our EmbeddingProvider compatible with neo4j-graphrag."""
    
    def __init__(self, provider):
        self._provider = provider
    
    async def aembed(self, text: str) -> list[float]:
        """Embed single text."""
        vectors = await self._provider.embed([text])
        return vectors[0]
    
    async def aembed_query(self, query: str) -> list[list[float]]:
        """Embed query (for compatibility)."""
        return await self._provider.embed([query])
