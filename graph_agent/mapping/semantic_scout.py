from typing import List, Dict, Any, Set, Optional


class SemanticScout:
    """
    A scout that explores the semantic structure of a web page using accessibility snapshots.
    """

    INTERACTIVE_ROLES: Set[str] = {
        "button",
        "link",
        "textbox",
        "combobox",
        "checkbox",
        "radio",
        "menuitem",
    }
    SEMANTIC_CONTAINERS: Set[str] = {
        "form",
        "dialog",
        "nav",
        "region",
        "main",
        "header",
        "footer",
    }

    def __init__(self, page: Any):
        """
        Initialize the SemanticScout with a Playwright page.

        Args:
            page: The Playwright page object.
        """
        self.page = page

    async def get_aom_snapshot(self) -> Dict[str, Any]:
        """
        Get the accessibility object model (AOM) snapshot of the current page.

        Returns:
            Dict[str, Any]: The accessibility snapshot as a dictionary.
        """
        snapshot = await self.page.accessibility.snapshot(interesting_only=False)
        self._enrich_with_parents(snapshot)
        return snapshot

    def filter_interactive_elements(
        self, aom_node: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Recursively filter the AOM tree to find interactive elements.

        Args:
            aom_node (Dict[str, Any]): The root node of the AOM tree or subtree to search.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing interactive elements found.
        """
        elements: List[Dict[str, Any]] = []

        if aom_node.get("role") in self.INTERACTIVE_ROLES:
            elements.append(aom_node)

        children = aom_node.get("children", [])
        for child in children:
            elements.extend(self.filter_interactive_elements(child))

        return elements

    def _enrich_with_parents(
        self, aom_node: Dict[str, Any], parent: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Recursively adds parent references to AOM nodes.

        Note: This modifies the aom_node dictionary in-place.
        """
        if parent:
            aom_node["parent"] = parent

        for child in aom_node.get("children", []):
            self._enrich_with_parents(child, aom_node)

    def _get_semantic_context(self, aom_node: Dict[str, Any]) -> str:
        """Bubbles up the tree to find semantic containers."""
        context_parts = []
        current = aom_node.get("parent")

        while current:
            role = current.get("role", "")
            name = current.get("name", "")

            if role in self.SEMANTIC_CONTAINERS:
                if name:
                    context_parts.append(f"{role}:{name}")
                else:
                    context_parts.append(role)

            current = current.get("parent")

        return " | ".join(reversed(context_parts))

    def generate_selectors(self, aom_node: Dict[str, Any]) -> List[str]:
        """Generates a list of robust selectors for an AOM node."""
        selectors = []
        role = aom_node.get("role", "")
        name = aom_node.get("name", "")

        # 1. Playwright Role Selector (Most Robust)
        if role and name:
            # Escape quotes in name if necessary
            safe_name = name.replace("'", "\\'")
            selectors.append(f"role={role}[name='{safe_name}']")

        # 2. Text Content (Exact)
        if name:
            safe_name = name.replace("'", "\\'")
            selectors.append(f"text='{safe_name}'")

        # 3. Role Only (if unique - context dependent, but we add it as candidate)
        if role:
            selectors.append(f"role={role}")

        return selectors
