from __future__ import annotations

from typing import Literal, TypedDict


class BBox(TypedDict):
    x: int
    y: int
    width: int
    height: int


class LayoutElement(TypedDict, total=False):
    tag: str
    role: str
    selector: str
    text_hint: str
    aria_label: str
    bbox: BBox
    z_index: str
    is_fixed: bool
    is_sticky: bool
    in_scroll_container: bool


class LayoutViewport(TypedDict):
    width: int
    height: int
    scrollX: int
    scrollY: int


class LayoutSnapshot(TypedDict, total=False):
    viewport: LayoutViewport
    element_count: int
    elements: list[LayoutElement]


class LayoutEvidenceItem(TypedDict):
    url: str
    step: int
    layout_fingerprint: str
    layout_summary: str
    layout_confidence: float


class LayoutMetrics(TypedDict):
    layout_confidence_samples: int
    layout_confidence_avg: float
    layout_confidence_min: float
    layout_confidence_max: float
    layout_low_confidence_hits: int
    layout_low_confidence_page_types: dict[str, int]
    layout_evidence_count: int


class LoginInfo(TypedDict, total=False):
    hasLogin: bool
    hasCaptcha: bool
    captchaTag: str
    captchaSrc: str
    captchaId: str
    hasCaptchaInput: bool
    captchaInputName: str
    captchaInputId: str


class FillResult(TypedDict, total=False):
    success: bool
    userFilled: bool
    pwdFilled: bool
    captchaFilled: bool
    submitClicked: bool
    reason: str


class LLMTransitionHint(TypedDict, total=False):
    action: str
    selector: str
    from_url: str
    to_url: str


TransitionSourceType = Literal["auto", "manual_graph_assisted", "manual_raw"]


class EvidenceBundleItem(TypedDict, total=False):
    evidence_type: Literal[
        "layout",
        "dom_diff",
        "url_change",
        "screenshot",
        "manual_note",
        "network",
    ]
    summary: str
    payload: str
    confidence: float


class TransitionCandidate(TypedDict, total=False):
    source_type: TransitionSourceType
    operator_id: str
    session_id: str
    trace_id: str
    step_index: int
    action: str
    selector: str
    action_value: str
    param_name: str
    url_before: str
    url_after: str
    from_state_hint: str
    to_state_hint: str
    thought: str
    confidence_hint: float
    evidence_bundle: list[EvidenceBundleItem]
