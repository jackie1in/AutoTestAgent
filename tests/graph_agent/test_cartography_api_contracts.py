from __future__ import annotations

from pathlib import Path


_CARTOGRAPHY_FILES = [
    "graph_agent/cartography/runner.py",
    "graph_agent/cartography/skip_advisor.py",
    "graph_agent/cartography/knowledge_broker.py",
    "graph_agent/cartography/react_explorer.py",
    "graph_agent/cartography/persistence.py",
]


def test_cartography_modules_do_not_use_graph_manager_private_api():
    repo_root = Path(__file__).resolve().parents[2]
    forbidden_tokens = ("._run_read(", "._driver.driver")
    for rel_path in _CARTOGRAPHY_FILES:
        content = (repo_root / rel_path).read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in content, f"{rel_path} should not contain {token}"


def test_graph_manager_exposes_stable_public_read_api():
    from graph_agent.neo4j_client.manager import GraphManager

    assert hasattr(GraphManager, "run_read")
    assert hasattr(GraphManager, "get_driver")
