from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any
from datetime import datetime

if TYPE_CHECKING:
    from browser_use.actor.page import Page

from graph_agent.models import Menu

logger = logging.getLogger(__name__)

_EXTRACT_MENU_TREE_JS = """() => {
    const menuSelectors = [
        'nav', '.sidebar', '.menu', '[role="navigation"]',
        '.ant-menu', '.el-menu', '[class*="sidebar"]', '[class*="nav"]'
    ];
    
    function buildTree(container, depth = 0, parentPath = '') {
        const items = [];
        const selectors = ['a', 'li', '.menu-item', '.ant-menu-item', '.el-menu-item'];
        
        for (const sel of selectors) {
            const elements = container.querySelectorAll(sel);
            elements.forEach((el, idx) => {
                const text = el.textContent?.trim();
                if (!text || text.length > 50) return;
                
                const href = el.href || el.closest('a')?.href || '';
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return;
                
                // Generate unique selector for this element
                let selector = '';
                if (el.id) {
                    selector = `#${el.id}`;
                } else if (el.className && typeof el.className === 'string') {
                    const classes = el.className.split(' ').filter(c => c).join('.');
                    selector = `.${classes}`;
                } else {
                    selector = `${sel}:nth-of-type(${idx + 1})`;
                }
                
                // Find submenu children
                const submenu = el.querySelector('ul, .submenu, .ant-menu-submenu, .el-menu--inline') 
                    || el.closest('li')?.querySelector('ul, .submenu, .ant-menu-submenu, .el-menu--inline');
                
                const path = parentPath ? `${parentPath}/${text}` : text;
                
                items.push({
                    text: text,
                    href: href,
                    selector: selector,
                    level: depth,
                    path: path,
                    order_index: idx,
                    has_children: !!submenu,
                    children: submenu ? buildTree(submenu, depth + 1, path) : []
                });
            });
        }
        
        // Remove duplicates based on text
        const seen = new Set();
        return items.filter(item => {
            if (seen.has(item.text)) return false;
            seen.add(item.text);
            return true;
        });
    }
    
    for (const sel of menuSelectors) {
        const container = document.querySelector(sel);
        if (container) {
            const tree = buildTree(container, 0);
            if (tree.length > 0) {
                return {
                    root_selector: sel,
                    items: tree,
                    depth: Math.max(...tree.map(i => {
                        let maxDepth = i.level;
                        function findDepth(items, d) {
                            for (const item of items) {
                                maxDepth = Math.max(maxDepth, d);
                                if (item.children?.length) {
                                    findDepth(item.children, d + 1);
                                }
                            }
                        }
                        findDepth(i.children || [], i.level + 1);
                        return maxDepth;
                    }))
                };
            }
        }
    }
    
    // Fallback: extract flat menu items
    const fallbackItems = [];
    document.querySelectorAll('a').forEach((el, idx) => {
        const text = el.textContent?.trim();
        const href = el.href || '';
        if (text && text.length > 0 && text.length < 30) {
            const rect = el.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
                fallbackItems.push({
                    text: text,
                    href: href,
                    selector: `a:nth-of-type(${idx + 1})`,
                    level: 0,
                    path: text,
                    order_index: idx,
                    has_children: false,
                    children: []
                });
            }
        }
    });
    
    return {
        root_selector: 'body',
        items: fallbackItems,
        depth: 0
    };
}"""


class MenuExtractor:
    """Extracts hierarchical menu tree from SPA without LLM."""

    async def extract(self, page: "Page") -> list[dict[str, Any]]:
        """Extract flat list of menu items (legacy method)."""
        tree = await self.extract_tree(page)
        items = []
        
        def flatten(node_list, parent_id=None):
            for item in node_list:
                menu_item = {
                    "text": item["text"],
                    "href": item["href"],
                    "selector": item["selector"],
                    "level": item["level"],
                    "path": item["path"],
                    "order_index": item["order_index"],
                    "parent_id": parent_id,
                }
                items.append(menu_item)
                current_id = item["path"]
                if item.get("children"):
                    flatten(item["children"], current_id)
        
        flatten(tree.get("items", []))
        return items

    async def extract_tree(self, page: "Page") -> dict[str, Any]:
        """Extract hierarchical menu tree structure."""
        raw = await page.evaluate(_EXTRACT_MENU_TREE_JS)
        if isinstance(raw, str):
            return json.loads(raw)
        return raw


def build_menu_models(
    tree_data: dict[str, Any],
    app_id: str,
    session_id: str | None = None,
) -> list[Menu]:
    """Convert extracted tree data to Menu model instances."""
    menus = []
    now = datetime.utcnow()
    
    def process_item(item: dict, parent_path: str = "") -> Menu:
        menu_id = f"menu:{app_id}:{item['path'].replace('/', ':').replace(' ', '_')}"
        stable_path = item["path"].replace(" ", "_").lower()
        
        menu = Menu(
            id=menu_id,
            label=item["text"],
            level=item["level"],
            order_index=item["order_index"],
            selector=item["selector"],
            menu_key=stable_path,
            app_id=app_id,
            stable_path=stable_path,
            first_discovered=now,
            last_seen=now,
        )
        return menu
    
    def traverse(items: list[dict], parent_path: str = ""):
        for item in items:
            menu = process_item(item, parent_path)
            menus.append(menu)
            if item.get("children"):
                traverse(item["children"], item["path"])
    
    traverse(tree_data.get("items", []))
    return menus


async def _link_parent_child_recursive(
    repo: Any,
    items: list[dict],
    menu_by_path: dict[str, Menu]
) -> None:
    """Recursively link parent-child menu relationships."""
    for item in items:
        current_path = item["path"].replace(" ", "_").lower()
        current_menu = menu_by_path.get(current_path)
        
        if current_menu and item.get("children"):
            for child_item in item["children"]:
                child_path = child_item["path"].replace(" ", "_").lower()
                child_menu = menu_by_path.get(child_path)
                
                if child_menu and current_menu:
                    await repo.link_menu_child_of(
                        child_menu.id,
                        current_menu.id,
                        child_item["order_index"]
                    )
            
            # Recurse into children
            await _link_parent_child_recursive(repo, item["children"], menu_by_path)


async def store_menu_tree(
    repo: Any,
    tree_data: dict[str, Any],
    app_id: str,
    session_id: str | None = None,
) -> list[Menu]:
    """Store extracted menu tree to Neo4j.
    
    Args:
        repo: GraphRepository instance
        tree_data: Output from MenuExtractor.extract_tree()
        app_id: Application ID
        session_id: Current session ID for provenance tracking
        
    Returns:
        List of created Menu models
    """
    menus = build_menu_models(tree_data, app_id, session_id)
    
    # Create menus and build parent-child relationships
    menu_by_path: dict[str, Menu] = {}
    
    for menu in menus:
        await repo.upsert_menu(menu)
        await repo.link_app_menu(app_id, menu.id)
        if session_id:
            await repo.link_session_discovered_menu(session_id, menu.id)
        menu_by_path[menu.stable_path] = menu
    
    # Create parent-child relationships
    await _link_parent_child_recursive(repo, tree_data.get("items", []), menu_by_path)
    
    logger.info(
        "Stored %d menu items for app %s (session: %s)",
        len(menus), app_id, session_id
    )
    return menus
