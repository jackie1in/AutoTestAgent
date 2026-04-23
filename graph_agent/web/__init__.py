"""FastAPI web app for graph and intents API."""

# Do NOT re-export as 'app' to avoid shadowing the graph_agent.web.app module.
# Patches like ``patch("graph_agent.web.app._get_driver")`` must resolve to
# the module, not a FastAPI instance attribute on the parent package.
from graph_agent.web.app import app as _fastapi_app

__all__ = []
