from __future__ import annotations

from typing import TypedDict


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
