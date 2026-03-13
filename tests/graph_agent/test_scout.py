"""Tests for scout structured-first parsing and fallback behavior."""

import json

import pytest

from graph_agent.mapping.scout import (
    _aggregate_elements_with_sources,
    _build_scout_metadata,
    _heuristic_extract_elements_from_text,
    _normalize_type,
    _parse_elements_from_llm_response,
    _resolve_multi_page_urls,
    extract_derived_urls,
    extract_derived_urls_from_elements,
    normalize_elements,
    run_scout_multi,
)


def test_parse_elements_supports_structured_object():
    response = """
    {
      "elements": [
        {"selector": "#username", "type": "input", "label": "Username"},
        {"selector": "#login", "type": "button", "label": "Login"}
      ]
    }
    """
    elements = _parse_elements_from_llm_response(response)
    assert len(elements) == 2
    assert elements[0]["selector"] == "#username"
    assert elements[1]["type"] == "button"


def test_heuristic_extract_elements_from_text():
    raw = "Found selectors: #login [name='username'] xpath=//button[@id='submit']"
    elements = _heuristic_extract_elements_from_text(raw)
    selectors = {item["selector"] for item in elements}
    assert "#login" in selectors
    assert "[name='username']" in selectors
    assert "xpath=//button[@id='submit']" in selectors


def test_normalize_elements_dedup_and_drop_empty_selector():
    raw = [
        {"selector": " #username ", "type": "textbox", "label": " Username "},
        {"selector": "#username", "type": "input", "label": "Username"},
        {"selector": "", "type": "button", "label": "Login"},
    ]
    out = normalize_elements(raw)
    assert len(out) == 1
    assert out[0]["selector"] == "#username"
    assert out[0]["type"] == "input"
    assert out[0]["label"] == "Username"


def test_normalize_type_maps_alias_and_infers_from_selector():
    assert (
        _normalize_type("anchor", "text=Form Authentication", "Form Authentication")
        == "link"
    )
    assert (
        _normalize_type("unknown", "xpath=//button[@id='submit']", "Submit") == "button"
    )
    assert _normalize_type("other", "[name='password']", "Password") == "input"


def test_build_scout_metadata_includes_type_counts():
    elements = [
        {"selector": "#username", "type": "input", "label": "Username"},
        {"selector": "#login", "type": "button", "label": "Login"},
        {"selector": "text=Home", "type": "link", "label": "Home"},
    ]
    metadata = _build_scout_metadata(
        extraction_path="llm_extract",
        raw_report="abc",
        extraction_text="xyz",
        elements=elements,
    )
    assert metadata["element_count"] == 3
    assert metadata["type_counts"] == {"input": 1, "button": 1, "link": 1}
    assert metadata["extraction_path"] == "llm_extract"


def test_resolve_multi_page_urls_supports_path_and_absolute():
    urls = _resolve_multi_page_urls(
        start_url="https://the-internet.herokuapp.com",
        page_hints=[
            "/login",
            "https://the-internet.herokuapp.com/dropdown",
            "checkboxes",
        ],
    )
    assert urls == [
        "https://the-internet.herokuapp.com",
        "https://the-internet.herokuapp.com/login",
        "https://the-internet.herokuapp.com/dropdown",
        "https://the-internet.herokuapp.com/checkboxes",
    ]


def test_aggregate_elements_with_sources_dedup_and_source_urls():
    per_page = {
        "https://the-internet.herokuapp.com": [
            {
                "selector": "text=Form Authentication",
                "type": "link",
                "label": "Form Authentication",
            },
            {"selector": "#shared", "type": "button", "label": "Shared"},
        ],
        "https://the-internet.herokuapp.com/login": [
            {"selector": "#username", "type": "input", "label": "Username"},
            {"selector": "#shared", "type": "button", "label": "Shared"},
        ],
    }
    out = _aggregate_elements_with_sources(per_page)
    by_selector = {item["selector"]: item for item in out}
    assert len(out) == 3
    assert by_selector["#username"]["type"] == "input"
    assert by_selector["#shared"]["source_urls"] == [
        "https://the-internet.herokuapp.com",
        "https://the-internet.herokuapp.com/login",
    ]


def test_extract_derived_urls_excludes_start_and_deduplicates():
    urls = [
        "https://example.com/",
        "https://example.com/login",
        "https://example.com/dashboard",
        "https://example.com/login",
        "",
        "not-http",
    ]
    derived = extract_derived_urls(urls, "https://example.com/", exclude_start=True)
    assert "https://example.com/" not in derived
    assert "https://example.com/login" in derived
    assert "https://example.com/dashboard" in derived
    assert derived.count("https://example.com/login") == 1


def test_extract_derived_urls_resolves_relative_paths():
    urls = ["/login", "/dashboard"]
    derived = extract_derived_urls(urls, "https://example.com/", exclude_start=False)
    assert "https://example.com/login" in derived
    assert "https://example.com/dashboard" in derived


def test_extract_derived_urls_from_elements_parses_href():
    elements = [
        {"selector": "a[href='/login']", "type": "link", "label": "Login"},
        {"selector": 'a[href="/dashboard"]', "type": "link", "label": "Dashboard"},
        {
            "selector": "xpath=//a[@href='/settings']",
            "type": "link",
            "label": "Settings",
        },
        {"selector": "#submit", "type": "button", "label": "Submit"},
        {"selector": "a[href='#anchor']", "type": "link", "label": "Anchor"},
    ]
    derived = extract_derived_urls_from_elements(elements, "https://example.com/")
    assert "https://example.com/login" in derived
    assert "https://example.com/dashboard" in derived
    assert "https://example.com/settings" in derived
    assert "#anchor" not in str(derived)


@pytest.mark.asyncio
async def test_run_scout_multi_aggregates_and_writes_metadata(monkeypatch, tmp_path):
    async def _fake_run_scout(url, output_path=None):
        if url.endswith("/login"):
            return [{"selector": "#username", "type": "input", "label": "Username"}]
        return [
            {
                "selector": "text=Form Authentication",
                "type": "link",
                "label": "Form Authentication",
            }
        ]

    monkeypatch.setattr("graph_agent.mapping.scout.run_scout", _fake_run_scout)
    output = tmp_path / "inventory.json"
    elements = await run_scout_multi(
        start_url="https://the-internet.herokuapp.com",
        page_hints=["/login"],
        output_path=output,
    )
    assert len(elements) == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["mode"] == "multi_page"
    assert payload["metadata"]["page_count"] == 2
    assert payload["metadata"]["aggregated_element_count"] == 2
