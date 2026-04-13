"""Tests for Task 1: 固化失败链路 - failure chain registry."""

import json
from pathlib import Path

from graph_agent.acceptance.failure_chain import (
    FailureChainRegistry,
    FailureRootCause,
    TargetChainDescription,
    load_registry,
)


def test_failure_root_cause_enum_values():
    """FailureRootCause 应包含 PRD 规定的五类根因。"""
    assert FailureRootCause.DEPENDENCY.value == "dependency"
    assert FailureRootCause.TAB.value == "tab"
    assert FailureRootCause.IFRAME.value == "iframe"
    assert FailureRootCause.SELECTOR.value == "selector"
    assert FailureRootCause.ASYNC_LOAD.value == "async_load"


def test_target_chain_description_model():
    """TargetChainDescription 应正确解析。"""
    chain = TargetChainDescription(
        chain_id="test-chain",
        summary="登录 -> 首页",
        start_url="https://example.com/",
        intent_keys=["fill_username", "submit_login"],
        failure_root_causes=[FailureRootCause.SELECTOR],
        notes="测试",
    )
    assert chain.chain_id == "test-chain"
    assert chain.summary == "登录 -> 首页"
    assert chain.failure_root_causes == [FailureRootCause.SELECTOR]


def test_load_registry_returns_default_when_file_missing(tmp_path):
    """当注册表文件不存在时，load_registry 应返回默认注册表。"""
    registry = load_registry(tmp_path / "nonexistent.json")
    assert isinstance(registry, FailureChainRegistry)
    assert registry.default_graph_source == "uv run python -m graph_agent.cartography.runner"
    assert registry.default_graph_path == "graph_agent/data/graph.json"
    assert registry.target_chains == []


def test_load_registry_loads_existing_file(tmp_path):
    """当注册表文件存在时，应正确加载。"""
    data = {
        "default_graph_source": "uv run python -m graph_agent.cartography.runner",
        "default_graph_path": "graph_agent/data/graph.json",
        "target_chains": [
            {
                "chain_id": "login-flow",
                "summary": "登录流程",
                "start_url": "https://example.com/",
                "intent_keys": ["fill_username", "submit_login"],
                "failure_root_causes": ["selector", "iframe"],
                "notes": "",
            }
        ],
        "metadata": {},
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    registry = load_registry(path)
    assert len(registry.target_chains) == 1
    chain = registry.target_chains[0]
    assert chain.chain_id == "login-flow"
    assert chain.failure_root_causes == [
        FailureRootCause.SELECTOR,
        FailureRootCause.IFRAME,
    ]


def test_load_registry_default_path():
    """load_registry() 无参调用时使用 graph_agent/acceptance/failure_chain_registry.json。"""
    pkg_root = Path(__file__).resolve().parent.parent.parent
    expected = pkg_root / "graph_agent" / "acceptance" / "failure_chain_registry.json"
    if expected.exists():
        registry = load_registry()
        assert len(registry.target_chains) >= 1
        assert (
            registry.default_graph_source == "uv run python -m graph_agent.cartography.runner"
        )
        assert registry.default_graph_path == "graph_agent/data/graph.json"


def test_registry_roundtrip(tmp_path):
    """FailureChainRegistry 应可序列化并反序列化。"""
    registry = FailureChainRegistry(
        target_chains=[
            TargetChainDescription(
                chain_id="roundtrip",
                summary="测试",
                start_url="https://a.com/",
                failure_root_causes=[FailureRootCause.TAB],
            )
        ],
    )
    path = tmp_path / "registry.json"
    path.write_text(registry.model_dump_json(exclude_none=True), encoding="utf-8")
    loaded = load_registry(path)
    assert loaded.target_chains[0].chain_id == "roundtrip"
    assert loaded.target_chains[0].failure_root_causes == [FailureRootCause.TAB]
