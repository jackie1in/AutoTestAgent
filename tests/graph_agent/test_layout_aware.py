from __future__ import annotations

import pytest

from graph_agent.cartography.layout_snapshot import (
    build_layout_summary,
    estimate_layout_confidence,
    compute_layout_fingerprint,
)
from graph_agent.cartography.llm_planning import (
    LLMPageAnalysis,
    analyze_page_with_llm,
)
from graph_agent.cartography.mapping_pipeline import (
    collect_layout_context,
    is_low_layout_confidence,
    summarize_layout_metrics,
)


class _FakePage:
    async def evaluate(self, _script, _limit):
        return {
            "viewport": {"width": 1440, "height": 900},
            "element_count": 3,
            "elements": [
                {
                    "tag": "nav",
                    "role": "",
                    "selector": "nav.main",
                    "bbox": {"x": 0, "y": 0, "width": 1200, "height": 80},
                    "is_fixed": False,
                    "is_sticky": False,
                    "z_index": "10",
                },
                {
                    "tag": "aside",
                    "role": "",
                    "selector": "aside.side",
                    "bbox": {"x": 0, "y": 90, "width": 220, "height": 700},
                    "is_fixed": False,
                    "is_sticky": True,
                    "z_index": "5",
                },
                {
                    "tag": "table",
                    "role": "",
                    "selector": "table.orders",
                    "bbox": {"x": 260, "y": 140, "width": 900, "height": 400},
                    "is_fixed": False,
                    "is_sticky": False,
                    "z_index": "auto",
                },
            ],
        }


class _FakeBrowser:
    def __init__(self):
        self.page = _FakePage()

    async def get_current_page(self):
        return self.page


@pytest.mark.asyncio
async def test_collect_layout_context_respects_switch(monkeypatch: pytest.MonkeyPatch):
    calls = {"n": 0}

    async def _fake_capture(_page, limit=200):
        calls["n"] += 1
        return {"elements": [], "element_count": 0, "viewport": {}, "limit": limit}

    monkeypatch.setattr(
        "graph_agent.cartography.mapping_pipeline.capture_layout_snapshot",
        _fake_capture,
    )

    browser = _FakeBrowser()
    summary, fp, conf = await collect_layout_context(browser, enabled=False, limit=200)
    assert summary == ""
    assert fp == ""
    assert conf == 0.0
    assert calls["n"] == 0

    _summary2, _fp2, conf2 = await collect_layout_context(browser, enabled=True, limit=200)
    assert calls["n"] == 1
    assert conf2 >= 0.0


def test_layout_summary_and_fingerprint_are_stable():
    snapshot = {
        "viewport": {"width": 1440, "height": 900},
        "element_count": 3,
        "elements": [
            {
                "tag": "nav",
                "role": "",
                "selector": "nav.main",
                "bbox": {"x": 0, "y": 0, "width": 1200, "height": 80},
                "is_fixed": False,
                "is_sticky": False,
                "z_index": "10",
            },
            {
                "tag": "aside",
                "role": "",
                "selector": "aside.side",
                "bbox": {"x": 0, "y": 100, "width": 200, "height": 700},
                "is_fixed": False,
                "is_sticky": True,
                "z_index": "5",
            },
            {
                "tag": "table",
                "role": "",
                "selector": "table.orders",
                "bbox": {"x": 260, "y": 140, "width": 900, "height": 400},
                "is_fixed": False,
                "is_sticky": False,
                "z_index": "auto",
            },
        ],
    }

    summary = build_layout_summary(snapshot)
    assert "top_nav:" in summary
    assert "sidebar:" in summary
    assert "tables:" in summary

    fp1 = compute_layout_fingerprint(snapshot)
    fp2 = compute_layout_fingerprint(snapshot)
    assert fp1
    assert fp1 == fp2
    conf = estimate_layout_confidence(snapshot)
    assert 0.1 <= conf <= 1.0


@pytest.mark.asyncio
async def test_analyze_page_with_llm_includes_layout_summary(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, str] = {}

    async def _fake_ainvoke(_llm, system_prompt, user_prompt, _schema, **_kwargs):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return LLMPageAnalysis(page_type="dashboard")

    monkeypatch.setattr(
        "graph_agent.cartography.llm_planning.ainvoke_structured",
        _fake_ainvoke,
    )

    result = await analyze_page_with_llm(
        llm=object(),
        dom_text="<div>hello</div>",
        current_url="https://example.com/app",
        page_title="Dashboard",
        layout_summary="top_nav: count=1 sample=nav.main",
    )

    assert result.page_type == "dashboard"
    assert "Layout summary (visual structure):" in captured["user"]
    assert "top_nav: count=1 sample=nav.main" in captured["user"]


def test_low_layout_confidence_gate():
    assert is_low_layout_confidence(0.2, enabled=True, threshold=0.45) is True
    assert is_low_layout_confidence(0.6, enabled=True, threshold=0.45) is False
    assert is_low_layout_confidence(0.2, enabled=False, threshold=0.45) is False


def test_summarize_layout_metrics():
    metrics = summarize_layout_metrics(
        samples=[0.2, 0.6, 0.4],
        low_confidence_hits=2,
        low_confidence_page_types={"dashboard": 1, "list": 1},
        evidence_count=3,
    )
    assert metrics["layout_confidence_samples"] == 3
    assert metrics["layout_confidence_avg"] == 0.4
    assert metrics["layout_confidence_min"] == 0.2
    assert metrics["layout_confidence_max"] == 0.6
    assert metrics["layout_low_confidence_hits"] == 2
    assert metrics["layout_low_confidence_page_types"]["dashboard"] == 1
    assert metrics["layout_evidence_count"] == 3
