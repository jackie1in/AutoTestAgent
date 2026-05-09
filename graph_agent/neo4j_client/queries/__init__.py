from graph_agent.neo4j_client.queries.app import AppQueriesMixin
from graph_agent.neo4j_client.queries.state import StateQueriesMixin
from graph_agent.neo4j_client.queries.transition import TransitionQueriesMixin
from graph_agent.neo4j_client.queries.zone import ZoneQueriesMixin
from graph_agent.neo4j_client.queries.frame import FrameQueriesMixin
from graph_agent.neo4j_client.queries.intent import IntentQueriesMixin
from graph_agent.neo4j_client.queries.entity import EntityQueriesMixin
from graph_agent.neo4j_client.queries.checkpoint import CheckpointQueriesMixin
from graph_agent.neo4j_client.queries.field_constraint import FieldConstraintQueriesMixin
from graph_agent.neo4j_client.queries.testcase import TestCaseQueriesMixin
from graph_agent.neo4j_client.queries.evidence import EvidenceQueriesMixin
from graph_agent.neo4j_client.queries.session import SessionQueriesMixin
from graph_agent.neo4j_client.queries.menu import MenuQueriesMixin
from graph_agent.neo4j_client.queries.legacy_menu import LegacyMenuQueriesMixin
from graph_agent.neo4j_client.queries.graph import GraphQueriesMixin
from graph_agent.neo4j_client.queries.skip_advisor import SkipAdvisorQueriesMixin
from graph_agent.neo4j_client.queries.knowledge import KnowledgeQueriesMixin
from graph_agent.neo4j_client.queries.runner import RunnerQueriesMixin
from graph_agent.neo4j_client.queries.semantic import SemanticQueriesMixin
from graph_agent.neo4j_client.queries.retention import RetentionQueriesMixin


class CypherQueries(
    AppQueriesMixin,
    StateQueriesMixin,
    TransitionQueriesMixin,
    ZoneQueriesMixin,
    FrameQueriesMixin,
    IntentQueriesMixin,
    EntityQueriesMixin,
    CheckpointQueriesMixin,
    FieldConstraintQueriesMixin,
    TestCaseQueriesMixin,
    EvidenceQueriesMixin,
    SessionQueriesMixin,
    MenuQueriesMixin,
    LegacyMenuQueriesMixin,
    GraphQueriesMixin,
    SkipAdvisorQueriesMixin,
    KnowledgeQueriesMixin,
    RunnerQueriesMixin,
    SemanticQueriesMixin,
    RetentionQueriesMixin,
):
    """Centralized Cypher query templates."""

