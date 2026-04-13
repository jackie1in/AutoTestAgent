from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PageInfo:
    viewport_width: int
    viewport_height: int
    page_width: int
    page_height: int
    scroll_x: int
    scroll_y: int
    pixels_above: int
    pixels_below: int
    pages_above: float
    pages_below: float
    total_pages: float
    current_page_position: float


@dataclass
class BrowserState:
    url: str
    title: str
    header: str
    content: str
    footer: str
    frame_context: str | None = None


@dataclass
class ActionResult:
    success: bool
    message: str
