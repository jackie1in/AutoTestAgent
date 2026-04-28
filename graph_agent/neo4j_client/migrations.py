"""
Neo4j Migration Manager

Handles schema migrations for Neo4j graph database.
All model field changes should be tracked here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """A single database migration."""
    version: str
    description: str
    apply: Callable[[Any], Awaitable[None]]


class MigrationManager:
    """Manages Neo4j schema migrations."""

    MIGRATIONS_TABLE = "__migrations"

    def __init__(self, driver: Any):
        self._driver = driver
        self._migrations: list[Migration] = []
        self._register_builtin_migrations()

    def _register_builtin_migrations(self) -> None:
        """Register all built-in migrations."""
        # Migration 001: Initial schema (App, State, Transition, etc.)
        self.register(Migration(
            version="001",
            description="Initial schema with App, State, Transition, Zone, Frame, Entity, Intent, Checkpoint, FieldConstraint, TestCase, Session",
            apply=self._migration_001_initial_schema,
        ))

        # Migration 002: Add Menu node type
        self.register(Migration(
            version="002",
            description="Add Menu node type with CHILD_OF, LEADS_TO, NAVIGATED_VIA relationships",
            apply=self._migration_002_add_menu_nodes,
        ))

        # Migration 003: Add embedding property for vector search
        self.register(Migration(
            version="003",
            description="Add embedding property to State, Menu, Intent, Zone, Transition, Checkpoint, Entity for vector search",
            apply=self._migration_003_add_embedding_properties,
        ))

        # Migration 004: Add Zone and Menu indexes for M2 cartography
        self.register(Migration(
            version="004",
            description="Add Zone type/indexes and Menu page_url/indexes for M2 cartography features",
            apply=self._migration_004_add_zone_menu_indexes,
        ))

        # Migration 005: Add ingestion/revision/release versioning layers
        self.register(Migration(
            version="005",
            description="Add IngestionRun, TransitionEntity/TransitionRevision, GraphRelease constraints and indexes",
            apply=self._migration_005_add_versioning_layers,
        ))

    def register(self, migration: Migration) -> None:
        """Register a new migration."""
        self._migrations.append(migration)
        self._migrations.sort(key=lambda m: m.version)

    async def ensure_migrations_table(self) -> None:
        """Create migrations tracking table if not exists."""
        async with self._driver.session() as session:
            await session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (m:__Migration) REQUIRE m.version IS UNIQUE
            """)
            await session.run("""
                CREATE INDEX IF NOT EXISTS FOR (m:__Migration) ON (m.applied_at)
            """)

    async def get_applied_migrations(self) -> set[str]:
        """Get set of already applied migration versions."""
        async with self._driver.session() as session:
            result = await session.run("""
                MATCH (m:__Migration)
                RETURN m.version AS version
            """)
            records = await result.data()
            return {r["version"] for r in records}

    async def apply_migration(self, migration: Migration) -> None:
        """Apply a single migration and record it."""
        logger.info("Applying migration %s: %s", migration.version, migration.description)
        
        await migration.apply(self._driver)
        
        async with self._driver.session() as session:
            await session.run("""
                CREATE (m:__Migration {
                    version: $version,
                    description: $description,
                    applied_at: datetime()
                })
            """, version=migration.version, description=migration.description)
        
        logger.info("Migration %s applied successfully", migration.version)

    async def migrate(self, target_version: str | None = None) -> list[str]:
        """Run all pending migrations up to target_version.
        
        Args:
            target_version: If provided, stop at this version. If None, run all.
            
        Returns:
            List of applied migration versions
        """
        await self.ensure_migrations_table()
        applied = await self.get_applied_migrations()
        
        applied_versions: list[str] = []
        
        for migration in self._migrations:
            if migration.version in applied:
                continue
            
            if target_version and migration.version > target_version:
                break
            
            await self.apply_migration(migration)
            applied_versions.append(migration.version)
        
        if applied_versions:
            logger.info("Applied %d migrations: %s", len(applied_versions), applied_versions)
        else:
            logger.info("No pending migrations")
        
        return applied_versions

    async def rollback(self, version: str) -> list[str]:
        """Rollback migrations to a specific version.
        
        Args:
            version: Target version to rollback to
            
        Returns:
            List of rolled back migration versions
        """
        # Note: Neo4j doesn't support true DDL rollback
        # This is a best-effort implementation
        logger.warning("Rollback is not fully supported in Neo4j. Manual intervention may be required.")
        
        async with self._driver.session() as session:
            result = await session.run("""
                MATCH (m:__Migration)
                WHERE m.version > $version
                RETURN m.version AS version
                ORDER BY m.version DESC
            """, version=version)
            records = await result.data()
            
            rolled_back = []
            for r in records:
                v = r["version"]
                await session.run("""
                    MATCH (m:__Migration {version: $version})
                    DELETE m
                """, version=v)
                rolled_back.append(v)
            
            return rolled_back

    async def status(self) -> dict[str, Any]:
        """Get current migration status."""
        await self.ensure_migrations_table()
        applied = await self.get_applied_migrations()
        
        pending = [m.version for m in self._migrations if m.version not in applied]
        
        return {
            "current_version": max(applied) if applied else None,
            "applied_count": len(applied),
            "pending_count": len(pending),
            "applied": sorted(applied),
            "pending": pending,
        }

    # ========== Built-in Migrations ==========

    async def _migration_001_initial_schema(self, driver: Any) -> None:
        """Create initial schema constraints and indexes."""
        async with driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (a:App) REQUIRE a.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:State) REQUIRE s.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Transition) REQUIRE t.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (z:Zone) REQUIRE z.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Frame) REQUIRE f.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (ei:EntityInstance) REQUIRE ei.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Intent) REQUIRE i.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Checkpoint) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (fc:FieldConstraint) REQUIRE fc.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (tc:TestCase) REQUIRE tc.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE",
            ]
            
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (a:App) ON (a.entry_url)",
                "CREATE INDEX IF NOT EXISTS FOR (s:State) ON (s.url)",
                "CREATE INDEX IF NOT EXISTS FOR (s:State) ON (s.fingerprint)",
                "CREATE INDEX IF NOT EXISTS FOR (s:State) ON (s.app_id)",
                "CREATE INDEX IF NOT EXISTS FOR (t:Transition) ON (t.confidence)",
                "CREATE INDEX IF NOT EXISTS FOR (t:Transition) ON (t.session_id)",
                "CREATE INDEX IF NOT EXISTS FOR (i:Intent) ON (i.key)",
                "CREATE INDEX IF NOT EXISTS FOR (c:Checkpoint) ON (c.layer)",
                "CREATE INDEX IF NOT EXISTS FOR (c:Checkpoint) ON (c.session_id)",
                "CREATE INDEX IF NOT EXISTS FOR (z:Zone) ON (z.exploration_status)",
                "CREATE INDEX IF NOT EXISTS FOR (sess:Session) ON (sess.app_id)",
                "CREATE INDEX IF NOT EXISTS FOR (ei:EntityInstance) ON (ei.status)",
                "CREATE INDEX IF NOT EXISTS FOR (ei:EntityInstance) ON (ei.session_id)",
                "CREATE INDEX IF NOT EXISTS FOR (tc:TestCase) ON (tc.category)",
                "CREATE INDEX IF NOT EXISTS FOR (fc:FieldConstraint) ON (fc.field_name)",
            ]
            
            for stmt in constraints + indexes:
                await session.run(stmt)

    async def _migration_002_add_menu_nodes(self, driver: Any) -> None:
        """Add Menu node type with related constraints and indexes."""
        async with driver.session() as session:
            # Create Menu constraint
            await session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Menu) REQUIRE m.id IS UNIQUE"
            )
            
            # Create Menu indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (m:Menu) ON (m.app_id)",
                "CREATE INDEX IF NOT EXISTS FOR (m:Menu) ON (m.level)",
                "CREATE INDEX IF NOT EXISTS FOR (m:Menu) ON (m.menu_key)",
                "CREATE INDEX IF NOT EXISTS FOR (m:Menu) ON (m.stable_path)",
            ]
            for stmt in indexes:
                await session.run(stmt)

    async def _migration_003_add_embedding_properties(self, driver: Any) -> None:
        """Add embedding property support for vector search.
        
        Note: This only adds the property. Vector indexes are created separately
        via VectorRetriever.ensure_collection() for each collection.
        """
        # Vector indexes are created dynamically via db.index.vector.createNodeIndex
        # This migration ensures the properties exist
        async with driver.session() as session:
            # Set empty embedding property on existing nodes to ensure schema
            # This is a no-op for existing nodes but ensures property exists
            labels = ["State", "Menu", "Intent", "Zone", "Transition", "Checkpoint", "Entity"]
            for label in labels:
                try:
                    await session.run(f"""
                        MATCH (n:{label})
                        WHERE n.embedding IS NULL
                        SET n.embedding = null
                    """)
                except Exception as e:
                    logger.warning("Could not initialize embedding property for %s: %s", label, e)

    async def _migration_004_add_zone_menu_indexes(self, driver: Any) -> None:
        """Add Zone and Menu indexes for M2 cartography features."""
        async with driver.session() as session:
            # Zone indexes for filtering by type and lookup
            zone_indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (z:Zone) ON (z.type)",
                "CREATE INDEX IF NOT EXISTS FOR (z:Zone) ON (z.selector)",
                "CREATE INDEX IF NOT EXISTS FOR (z:Zone) ON (z.element_count)",
            ]
            
            # Menu indexes for filtering by page_url and level
            menu_indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (m:Menu) ON (m.page_url)",
                "CREATE INDEX IF NOT EXISTS FOR (m:Menu) ON (m.href)",
                "CREATE INDEX IF NOT EXISTS FOR (m:Menu) ON (m.is_active)",
            ]
            
            for stmt in zone_indexes + menu_indexes:
                await session.run(stmt)

    async def _migration_005_add_versioning_layers(self, driver: Any) -> None:
        """Add three-layer versioning schema and indexes."""
        async with driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (run:IngestionRun) REQUIRE run.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (ent:TransitionEntity) REQUIRE ent.stable_key IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (rev:TransitionRevision) REQUIRE rev.revision_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (r:GraphRelease) REQUIRE r.id IS UNIQUE",
            ]
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (run:IngestionRun) ON (run.app_id)",
                "CREATE INDEX IF NOT EXISTS FOR (run:IngestionRun) ON (run.session_id)",
                "CREATE INDEX IF NOT EXISTS FOR (rev:TransitionRevision) ON (rev.transition_id, rev.is_active)",
                "CREATE INDEX IF NOT EXISTS FOR (rev:TransitionRevision) ON (rev.stable_key)",
                "CREATE INDEX IF NOT EXISTS FOR (rev:TransitionRevision) ON (rev.ingest_version_id)",
                "CREATE INDEX IF NOT EXISTS FOR (r:GraphRelease) ON (r.app_id)",
                "CREATE INDEX IF NOT EXISTS FOR (s:State) ON (s.ingest_version_id)",
                "CREATE INDEX IF NOT EXISTS FOR (t:Transition) ON (t.ingest_version_id)",
                "CREATE INDEX IF NOT EXISTS FOR (e:Evidence) ON (e.ingest_version_id)",
                "CREATE INDEX IF NOT EXISTS FOR (m:Menu) ON (m.ingest_version_id)",
                "CREATE INDEX IF NOT EXISTS FOR (z:Zone) ON (z.ingest_version_id)",
            ]
            for stmt in constraints + indexes:
                await session.run(stmt)


# Convenience functions for CLI usage

async def migrate_database(driver: Any, target_version: str | None = None) -> list[str]:
    """Run all pending migrations.
    
    Usage:
        from graph_agent.neo4j_client.migrations import migrate_database
        applied = await migrate_database(driver)
    """
    manager = MigrationManager(driver)
    return await manager.migrate(target_version)


async def rollback_database(driver: Any, version: str) -> list[str]:
    """Rollback to specific version.
    
    Usage:
        from graph_agent.neo4j_client.migrations import rollback_database
        rolled_back = await rollback_database(driver, "001")
    """
    manager = MigrationManager(driver)
    return await manager.rollback(version)


async def migration_status(driver: Any) -> dict[str, Any]:
    """Get current migration status.
    
    Usage:
        from graph_agent.neo4j_client.migrations import migration_status
        status = await migration_status(driver)
        print(f"Current version: {status['current_version']}")
        print(f"Pending: {status['pending']}")
    """
    manager = MigrationManager(driver)
    return await manager.status()
