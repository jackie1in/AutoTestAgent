import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

logger = logging.getLogger(__name__)

_neo4j_driver: Any | None = None
_embedder: Any | None = None
_graphrag_available: bool | None = None


@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    try:
        yield
    finally:
        global _neo4j_driver
        if _neo4j_driver is not None:
            try:
                await _neo4j_driver.close()
                logger.info("Neo4j driver closed on shutdown")
            except Exception as e:
                logger.warning("Error closing Neo4j driver: %s", e)


async def _ensure_graphrag() -> tuple[Any, Any] | None:
    global _neo4j_driver, _embedder, _graphrag_available

    if _graphrag_available is not None:
        return (_neo4j_driver, _embedder) if _graphrag_available else None

    try:
        from graph_agent.cartography.nl_resolver import _get_embedder
        from graph_agent.neo4j_client.driver import Neo4jDriver

        _neo4j_driver = Neo4jDriver()
        await _neo4j_driver.connect()
        _embedder = _get_embedder()
        _graphrag_available = True
        logger.info("GraphRAG initialized for NL playback")
        return (_neo4j_driver, _embedder)
    except Exception as e:
        logger.warning("GraphRAG initialization failed: %s", e)
        _graphrag_available = False
        return None


async def _get_driver() -> Any:
    graphrag = await _ensure_graphrag()
    if graphrag is not None:
        return graphrag[0].driver
    raise RuntimeError("Neo4j driver not available")
