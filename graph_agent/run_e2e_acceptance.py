"""T10 端到端验收脚本（目标站点）: 验证 测绘 -> 意图 -> 寻径 -> 回放 链路可执行。

用法:
  uv run python graph_agent/run_e2e_acceptance.py [--graph PATH] [--re-infer] [--url URL]

  1. 若提供 --graph，从该文件加载图；否则使用 graph_agent/data/graph.json
  2. 若提供 --re-infer，对 intent=null 边执行重推
  3. 统计 edge_count、intent_missing_count、intent_success_rate
  4. 选择至少 1 条长度 >= 2 的路径完成回放
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Ensure project root on path (required before graph_agent imports)
_root = Path(__file__).resolve().parent.parent
if _root not in sys.path:
    sys.path.insert(0, str(_root))

from graph_agent.graph.io import load_graph  # noqa: E402
from graph_agent.graph.pathfinding import get_path_from_query  # noqa: E402
from graph_agent.mapping.run import re_infer_missing_intents  # noqa: E402
from graph_agent.playback.engine import run_playback  # noqa: E402

TARGET_SITE = "https://the-internet.herokuapp.com"
DEFAULT_GRAPH = Path(__file__).resolve().parent / "data" / "graph.json"


def _compute_stats(graph) -> dict:
    """计算图的质量指标。"""
    edge_count = graph.number_of_edges()
    intent_missing = sum(1 for _u, _v, d in graph.edges(data=True) if d.get("intent") is None)
    intent_success_rate = 1.0 - (intent_missing / edge_count) if edge_count else 0.0
    return {
        "edge_count": edge_count,
        "intent_missing_count": intent_missing,
        "intent_success_rate": intent_success_rate,
    }


def _find_path_length_ge_2(graph, intent_queries: list[str] | None = None) -> tuple[list, str | None]:
    """查找长度 >= 2 的路径。返回 (path, intent_used) 或 ([], None)。"""
    if intent_queries is None:
        intent_queries = ["auth.login", "login", "fill_username", "fill_password", "submit_login", "click", "fill"]
    for q in intent_queries:
        path = get_path_from_query(q, graph)
        if len(path) >= 2:
            return path, q
    return [], None


async def run_acceptance(
    graph_path: str | Path,
    re_infer: bool = False,
    start_url: str | None = None,
) -> bool:
    """执行端到端验收。返回是否通过。"""
    path_obj = Path(graph_path)
    if not path_obj.exists():
        print(f"图文件不存在: {graph_path}")
        return False

    graph = load_graph(path_obj)
    if graph.number_of_edges() == 0:
        print("图为空，无法验收。请先运行 mapping: uv run python -m graph_agent.mapping.run --url <url>")
        return False

    # 1. 统计
    stats_before = _compute_stats(graph)
    print(f"统计: edge_count={stats_before['edge_count']}, "
          f"intent_missing={stats_before['intent_missing_count']}, "
          f"intent_success_rate={stats_before['intent_success_rate']:.2%}")

    # 2. 可选：re-infer-missing
    if re_infer and stats_before["intent_missing_count"] > 0:
        print("执行 --re-infer-missing...")
        re_stats = await re_infer_missing_intents(path_obj)
        print(f"Re-infer: total={re_stats['total']}, succeeded={re_stats['succeeded']}, failed={re_stats['failed']}")
        graph = load_graph(path_obj)
        stats_after = _compute_stats(graph)
        print(f"改进后: intent_missing={stats_after['intent_missing_count']}, "
              f"intent_success_rate={stats_after['intent_success_rate']:.2%}")

    # 3. 寻径
    path, intent_used = _find_path_length_ge_2(graph)
    if not path:
        print("未找到长度 >= 2 的可执行路径。")
        return False
    print(f"找到路径 (intent={intent_used}, len={len(path)}): {path[0].source} -> ... -> {path[-1].target}")

    # 4. 回放
    url = start_url or _get_start_url_from_graph(graph)
    if not url:
        url = TARGET_SITE + "/"
    os.environ["PLAYWRIGHT_HEADLESS"] = "true"
    try:
        result = await run_playback(
            path,
            test_data={"username": "tomsmith", "password": "SuperSecretPassword!"},
            start_url=url,
        )
        if result["success"]:
            print(f"回放成功: {result['actual_url']}")
            return True
        print(f"回放失败: {result.get('error')}")
        return False
    finally:
        os.environ.pop("PLAYWRIGHT_HEADLESS", None)


def _get_start_url_from_graph(graph) -> str | None:
    """从图中推断起始 URL（in_degree=0 的节点）。"""
    graph_start_url = graph.graph.get("start_url") if hasattr(graph, "graph") else None
    if isinstance(graph_start_url, str) and graph_start_url.startswith("http"):
        return graph_start_url
    entries = [n for n in graph if graph.in_degree(n) == 0]
    for n in entries:
        url = graph.nodes[n].get("url") if isinstance(n, str) else None
        if isinstance(url, str) and url.startswith("http"):
            return url
    return None


def main() -> None:
    import argparse
    from dotenv import load_dotenv

    load_dotenv()
    pkg_root = Path(__file__).resolve().parent
    default_graph = str(pkg_root / "data" / "graph.json")

    parser = argparse.ArgumentParser(description="T10 端到端验收（目标站点）")
    parser.add_argument("--graph", default=default_graph, help="图 JSON 路径")
    parser.add_argument("--re-infer", action="store_true", help="对 intent=null 边执行重推")
    parser.add_argument("--url", default=None, help="回放起始 URL（默认从图推断）")
    args = parser.parse_args()

    ok = asyncio.run(run_acceptance(args.graph, re_infer=args.re_infer, start_url=args.url))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
