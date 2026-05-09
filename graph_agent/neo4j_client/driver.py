from __future__ import annotations

import logging
import os

from neo4j import AsyncDriver, AsyncGraphDatabase

logger = logging.getLogger(__name__)


class Neo4jDriver:
    """Manages Neo4j async driver lifecycle and schema initialization."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        logging.getLogger("neo4j.notifications").setLevel(logging.DEBUG)
        self._uri: str = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._user: str = user or os.getenv("NEO4J_USER", "neo4j")
        self._password: str = password or os.getenv("NEO4J_PASSWORD", "autotestagent")
        self._driver: AsyncDriver | None = None

    async def connect(self) -> AsyncDriver:
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self._uri, auth=(self._user, self._password)
            )
            await self._driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", self._uri)
        return self._driver

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._driver

    async def ensure_schema(self) -> None:
        """Create indexes and constraints. Idempotent."""
        driver = await self.connect()
        async with driver.session() as session:
            # Constraints
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:App) REQUIRE a.id IS UNIQUE")
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:State) REQUIRE s.id IS UNIQUE")
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Transition) REQUIRE t.id IS UNIQUE")
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (z:Zone) REQUIRE z.id IS UNIQUE")
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (f:Frame) REQUIRE f.id IS UNIQUE")
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (ei:EntityInstance) REQUIRE ei.id IS UNIQUE")
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Intent) REQUIRE i.id IS UNIQUE")
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Checkpoint) REQUIRE c.id IS UNIQUE")
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (fc:FieldConstraint) REQUIRE fc.id IS UNIQUE")
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (tc:TestCase) REQUIRE tc.id IS UNIQUE")
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE")
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Menu) REQUIRE m.id IS UNIQUE")
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (ev:Evidence) REQUIRE ev.id IS UNIQUE")
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (run:IngestionRun) REQUIRE run.id IS UNIQUE")
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (ent:TransitionEntity) REQUIRE ent.stable_key IS UNIQUE")
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (rev:TransitionRevision) REQUIRE rev.revision_id IS UNIQUE")
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (gr:GraphRelease) REQUIRE gr.id IS UNIQUE")
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (cov:CoverageSnapshot) REQUIRE cov.id IS UNIQUE")

            # Indexes
            await session.run("CREATE INDEX IF NOT EXISTS FOR (a:App) ON (a.entry_url)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (s:State) ON (s.url)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (s:State) ON (s.fingerprint)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (s:State) ON (s.app_id)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (t:Transition) ON (t.confidence)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (t:Transition) ON (t.session_id)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (i:Intent) ON (i.key)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (c:Checkpoint) ON (c.layer)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (c:Checkpoint) ON (c.session_id)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (z:Zone) ON (z.exploration_status)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (sess:Session) ON (sess.app_id)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (ei:EntityInstance) ON (ei.status)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (ei:EntityInstance) ON (ei.session_id)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (tc:TestCase) ON (tc.category)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (fc:FieldConstraint) ON (fc.field_name)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (m:Menu) ON (m.app_id)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (m:Menu) ON (m.level)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (m:Menu) ON (m.menu_key)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (m:Menu) ON (m.stable_path)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (ev:Evidence) ON (ev.evidence_type)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (ev:Evidence) ON (ev.transition_id)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (run:IngestionRun) ON (run.app_id)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (run:IngestionRun) ON (run.session_id)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (rev:TransitionRevision) ON (rev.transition_id, rev.is_active)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (rev:TransitionRevision) ON (rev.stable_key)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (gr:GraphRelease) ON (gr.app_id)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (cov:CoverageSnapshot) ON (cov.app_id)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (cov:CoverageSnapshot) ON (cov.session_id)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (cov:CoverageSnapshot) ON (cov.captured_at)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (ent:TransitionEntity) ON (ent.confirmed_session_count)")

    async def __aenter__(self) -> Neo4jDriver:
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
