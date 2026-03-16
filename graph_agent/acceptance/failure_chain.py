"""Task 1: 固化失败链路 - 失败根因分类与目标业务链描述。

PRD 9.1: 梳理真实失败链路并固化验收样本。
- 失败根因分类: dependency / tab / iframe / selector / async_load
- 目标业务链说明
- 验收默认使用 uv run python -m graph_agent.mapping.run 输出的数据文件
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class FailureRootCause(str, Enum):
    """失败根因分类 (PRD 9.1)。"""

    DEPENDENCY = "dependency"
    TAB = "tab"
    IFRAME = "iframe"
    SELECTOR = "selector"
    ASYNC_LOAD = "async_load"


def classify_root_cause_detail(error: str | None) -> tuple[FailureRootCause, str]:
    """Return (primary root cause, secondary detail) from playback error text."""
    text = str(error or "")
    lower = text.lower()

    if lower.startswith("tab:"):
        return FailureRootCause.TAB, "tab.missing"

    if lower.startswith("iframe:"):
        if "failed to locate iframe" in lower:
            return FailureRootCause.IFRAME, "iframe.not_found"
        if "has been closed" in lower or "target closed" in lower:
            return FailureRootCause.IFRAME, "iframe.not_attached"
        return FailureRootCause.IFRAME, "iframe.unknown"

    if lower.startswith("async_load:"):
        if "has been closed" in lower or "target closed" in lower:
            return FailureRootCause.IFRAME, "iframe.not_attached"
        return FailureRootCause.ASYNC_LOAD, "async_load.timeout"

    if "login flow did not leave" in lower:
        return FailureRootCause.DEPENDENCY, "dependency.nav_failed"
    if "click was expected to navigate" in lower:
        return FailureRootCause.DEPENDENCY, "dependency.nav_failed"
    if "iframe for the next step did not appear" in lower:
        return FailureRootCause.DEPENDENCY, "dependency.lookahead"

    if lower.startswith("selector:"):
        if "not attached" in lower or "detached" in lower:
            return FailureRootCause.SELECTOR, "selector.stale"
        if "waiting for selector" in lower or "no element" in lower:
            return FailureRootCause.SELECTOR, "selector.not_found"
        if "timeout" in lower:
            return FailureRootCause.SELECTOR, "selector.timeout"
        return FailureRootCause.SELECTOR, "selector.unknown"

    return FailureRootCause.DEPENDENCY, "dependency.unknown"


class TargetChainDescription(BaseModel):
    """目标业务链说明，供后续任务复用。"""

    chain_id: str = Field(..., description="链路唯一标识")
    summary: str = Field(..., description="业务链摘要")
    start_url: str = Field(..., description="起始 URL")
    intent_keys: list[str] = Field(default_factory=list, description="意图 key 序列")
    failure_root_causes: list[FailureRootCause] = Field(
        default_factory=list,
        description="该链路上已识别的失败根因",
    )
    notes: str = Field(default="", description="备注")


class FailureChainRegistry(BaseModel):
    """失败链路注册表，固化可重复验收的样本。"""

    default_graph_source: str = Field(
        default="uv run python -m graph_agent.mapping.run",
        description="本轮验收默认数据来源：mapping.run 输出",
    )
    default_graph_path: str = Field(
        default="graph_agent/data/graph.json",
        description="默认图谱路径",
    )
    target_chains: list[TargetChainDescription] = Field(
        default_factory=list,
        description="目标业务链列表",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


def load_registry(path: Path | None = None) -> FailureChainRegistry:
    """加载失败链路注册表。若路径不存在则返回默认注册表。"""
    if path is None:
        pkg_root = Path(__file__).resolve().parent
        path = pkg_root / "failure_chain_registry.json"
    if not path.exists():
        return FailureChainRegistry()
    data = path.read_text(encoding="utf-8")
    return FailureChainRegistry.model_validate_json(data)
