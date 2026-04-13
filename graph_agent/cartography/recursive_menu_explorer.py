"""Recursive Menu Explorer - 递归菜单深度探索

核心功能：
1. 递归展开所有菜单层级
2. 记录菜单到页面的映射
3. 支持返回后重新探索
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from graph_agent.models import State, Transition, ActionType

logger = logging.getLogger(__name__)


@dataclass
class MenuNode:
    """菜单节点"""
    text: str
    level: int
    selector: str | None = None
    href: str | None = None
    is_leaf: bool = False
    children: list[MenuNode] = field(default_factory=list)
    parent: MenuNode | None = None
    
    @property
    def path(self) -> list[str]:
        """获取菜单路径"""
        if self.parent:
            return self.parent.path + [self.text]
        return [self.text]
    
    @property
    def path_key(self) -> str:
        """获取唯一路径键"""
        return "/".join(self.path)


@dataclass
class MenuExplorationState:
    """菜单探索状态（断点续传）"""
    root_url: str
    explored_paths: set[str] = field(default_factory=set)
    pending_paths: list[list[str]] = field(default_factory=list)
    state_id_map: dict[str, str] = field(default_factory=dict)  # path_key -> state_id
    
    def mark_explored(self, path: list[str]):
        """标记路径已探索"""
        self.explored_paths.add("/".join(path))
    
    def is_explored(self, path: list[str]) -> bool:
        """检查路径是否已探索"""
        return "/".join(path) in self.explored_paths
    
    def add_pending(self, path: list[str]):
        """添加待探索路径"""
        key = "/".join(path)
        if key not in self.explored_paths:
            self.pending_paths.append(path)
    
    def get_next_pending(self) -> list[str] | None:
        """获取下一个待探索路径"""
        while self.pending_paths:
            path = self.pending_paths.pop(0)
            if not self.is_explored(path):
                return path
        return None


class RecursiveMenuExplorer:
    """递归菜单探索器"""
    
    def __init__(
        self,
        session,
        neo4j_driver,
        max_depth: int = 5,
        exploration_state: MenuExplorationState | None = None,
    ):
        self._session = session
        self._driver = neo4j_driver
        self._max_depth = max_depth
        self._state = exploration_state
        self._discovered_transitions: list[Transition] = []
        
    async def explore_menu_recursive(
        self,
        start_url: str,
        app_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        """
        递归探索所有菜单
        
        算法：
        1. 扫描当前页面菜单
        2. 对每个未探索的叶子节点：
           - 点击菜单
           - 记录新 State
           - 在新页面继续扫描菜单
           - 返回父页面
        3. 递归直到所有路径探索完成
        """
        if self._state is None:
            self._state = MenuExplorationState(root_url=start_url)
        
        results = {
            'states': [],
            'transitions': [],
            'menu_tree': None,
        }
        
        # 1. 构建完整菜单树（不点击，只扫描 DOM）
        menu_tree = await self._build_menu_tree()
        results['menu_tree'] = menu_tree
        
        # 2. 收集所有叶子节点路径
        leaf_paths = self._collect_leaf_paths(menu_tree)
        logger.info(f"Found {len(leaf_paths)} leaf menu items to explore")
        
        # 3. 逐个探索叶子节点
        for path in leaf_paths:
            if self._state.is_explored(path):
                logger.debug(f"Skipping already explored: {'/'.join(path)}")
                continue
            
            try:
                state, transitions = await self._explore_menu_path(
                    path, start_url, app_id, session_id
                )
                if state:
                    results['states'].append(state)
                results['transitions'].extend(transitions)
                self._state.mark_explored(path)
            except Exception as e:
                logger.error(f"Failed to explore menu path {'/'.join(path)}: {e}")
        
        return results
    
    async def _build_menu_tree(self) -> MenuNode:
        """构建菜单树结构"""
        page = await self._session.must_get_current_page()
        
        # JavaScript 扫描菜单结构
        menu_data = await page.evaluate("""
        () => {
            const menuItems = [];
            
            // 检测常见菜单模式
            const selectors = [
                '.ant-menu-item',
                '.el-menu-item',
                '.ant-layout-sider .ant-menu-submenu-title',
                '.el-submenu__title',
                'nav li',
                'aside a',
                '[role="menuitem"]',
            ];
            
            for (const sel of selectors) {
                const items = document.querySelectorAll(sel);
                items.forEach((el, idx) => {
                    const text = el.textContent?.trim();
                    const href = el.getAttribute('href');
                    const hasSubmenu = el.querySelector('.ant-menu-sub, .el-menu') !== null ||
                                      el.nextElementSibling?.classList?.contains('ant-menu-sub') ||
                                      el.nextElementSibling?.classList?.contains('el-menu');
                    
                    if (text) {
                        menuItems.push({
                            text: text,
                            index: idx,
                            selector: `${sel}:nth-of-type(${idx + 1})`,
                            href: href,
                            hasSubmenu: hasSubmenu,
                            level: el.closest('.ant-menu-sub, .el-menu') ? 1 : 0,
                        });
                    }
                });
            }
            
            return menuItems;
        }
        """)
        
        # 构建树结构
        root = MenuNode(text="root", level=-1)
        
        for item in menu_data:
            node = MenuNode(
                text=item['text'],
                level=item['level'],
                selector=item['selector'],
                href=item['href'],
                is_leaf=not item['hasSubmenu'] and not item['href'],
            )
            # 简化：扁平结构
            root.children.append(node)
            node.parent = root
        
        return root
    
    def _collect_leaf_paths(self, root: MenuNode) -> list[list[str]]:
        """收集所有叶子节点路径"""
        paths = []
        
        def traverse(node: MenuNode):
            if node.is_leaf or not node.children:
                if node.text != "root":
                    paths.append(node.path)
            for child in node.children:
                traverse(child)
        
        traverse(root)
        return paths
    
    async def _explore_menu_path(
        self,
        path: list[str],
        start_url: str,
        app_id: str,
        session_id: str,
    ) -> tuple[State | None, list[Transition]]:
        """
        探索单个菜单路径
        
        1. 返回到起始页面
        2. 依次点击菜单
        3. 记录每个步骤的状态变化
        """
        page = await self._session.must_get_current_page()
        transitions = []
        
        # 确保在起始页面
        current_url = await page.get_url()
        if current_url != start_url:
            await page.goto(start_url)
            await asyncio.sleep(2)
        
        previous_state_id = None
        
        # 依次点击菜单项
        for i, menu_text in enumerate(path):
            # 点击菜单
            clicked = await self._click_menu_by_text(page, menu_text)
            if not clicked:
                logger.warning(f"Failed to click menu: {menu_text}")
                break
            
            await asyncio.sleep(1.5)
            
            # 记录状态
            from graph_agent.cartography.snapshot import capture_dom_fingerprint
            new_url = await page.get_url()
            fingerprint = await capture_dom_fingerprint(page)
            
            state_id = f"state:{app_id}:menu:{'/'.join(path[:i+1])}"
            state = State(
                id=state_id,
                url=new_url,
                title=menu_text,
                fingerprint=fingerprint,
                menu_path=path[:i+1],
            )
            
            # 创建 Transition
            if previous_state_id:
                transition = Transition(
                    id=f"t:{previous_state_id}:{menu_text}",
                    selector=f"menu:{menu_text}",
                    action=ActionType.CLICK,
                    from_state_id=previous_state_id,
                    to_state_id=state_id,
                    intent=None,  # 可以从 menu_text 推断
                )
                transitions.append(transition)
            
            previous_state_id = state_id
        
        return state, transitions
    
    async def _click_menu_by_text(self, page, text: str) -> bool:
        """通过文本点击菜单项"""
        try:
            result = await page.evaluate(
                """(targetText) => {
                    const selectors = [
                        '.ant-menu-item',
                        '.el-menu-item',
                        'nav li a',
                        'aside a',
                        '[role="menuitem"]',
                    ];
                    
                    for (const sel of selectors) {
                        const items = document.querySelectorAll(sel);
                        for (const el of items) {
                            if (el.textContent?.trim() === targetText) {
                                el.click();
                                return true;
                            }
                        }
                    }
                    return false;
                }""",
                text,
            )
            return result is True
        except Exception as e:
            logger.error(f"Click menu failed: {e}")
            return False


class MenuAwareScheduler:
    """菜单感知的调度器 - 优先探索菜单可达的页面"""
    
    async def schedule_with_menu_priority(
        self,
        neo4j_driver,
        menu_exploration_result: dict,
    ) -> list[dict]:
        """
        基于菜单探索结果生成调度任务
        
        策略：
        1. 优先探索菜单直接可达的页面（breadth）
        2. 然后深度探索每个页面的 Zones（depth）
        3. 最后验证关键路径（validation）
        """
        tasks = []
        
        # 从菜单结果提取所有 States
        for state in menu_exploration_result.get('states', []):
            tasks.append({
                'type': 'explore_state_zones',
                'priority': 100,
                'state_id': state.id,
                'state': state,
            })
        
        # 排序：菜单路径短的优先
        tasks.sort(key=lambda t: len(t.get('state', {}).menu_path or []))
        
        return tasks
