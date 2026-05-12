from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def _rank_selector_chain(
    selector: str, attrs: dict[str, object], tag: str
) -> list[str]:
    chain: list[str] = []
    test_id = attrs.get("data-testid") or attrs.get("data-test") or attrs.get("data-qa")
    if test_id:
        chain.append(f'[data-testid="{test_id}"]')
    element_id = attrs.get("id")
    if element_id:
        chain.append(f"#{element_id}")
    role = attrs.get("role")
    name = attrs.get("name")
    if role and name:
        chain.append(f'{tag}[role="{role}"][name="{name}"]')
    elif role:
        chain.append(f'{tag}[role="{role}"]')
    if name:
        chain.append(f'{tag}[name="{name}"]')
    if selector and selector not in chain:
        chain.append(selector)
    if not chain:
        chain.append(f"[selector:{selector or '?'}]")
    return chain


