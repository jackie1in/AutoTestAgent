from graph_agent.cartography.persistence.core import _PersistenceSessionCore as _PersistenceSessionCore
from graph_agent.cartography.persistence.finalize_mixin import _FinalizeMixin as _FinalizeMixin
from graph_agent.cartography.persistence.one_shot import persist_mapping_result as persist_mapping_result


class PersistenceSession(_FinalizeMixin, _PersistenceSessionCore):
    pass
