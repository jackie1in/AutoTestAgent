"""CLI entry for mapping: allows `uv run python graph_agent/run_mapping.py` with --url, --output."""

import sys
from pathlib import Path

# When run as script, ensure project root is on path so "graph_agent" package resolves
_root = Path(__file__).resolve().parent.parent
if _root not in sys.path:
    sys.path.insert(0, str(_root))

from graph_agent.mapping.run import main

if __name__ == "__main__":
    main()
