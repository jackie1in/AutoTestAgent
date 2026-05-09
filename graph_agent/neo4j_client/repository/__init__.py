from graph_agent.neo4j_client.repository.core import GraphRepositoryCore
from graph_agent.neo4j_client.repository.extended import GraphRepositoryExtended


class GraphRepository(GraphRepositoryCore, GraphRepositoryExtended):
    pass

