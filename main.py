#!/usr/bin/env python3
"""
UI Automation Testing Agent - Main Entry Point

A browser automation testing tool based on Browser-Use that can:
- Record UI interactions via AI agent
- Replay recorded test cases
- Manage test cases

Usage:
  python main.py record [options]   - Record a test case (use --skill-creator to generate a skill)
  python main.py replay [test_id]   - Replay a saved test case
  python main.py list               - List all saved test cases
  python main.py list-skills        - List local skills
  python main.py view <id>          - View details of a test case
  python main.py delete <id>        - Delete a test case
  python main.py help               - Show help message
"""

# Suppress langchain_core Pydantic V1 warning on Python 3.14+ (upstream compatibility)
import warnings
warnings.filterwarnings("ignore", message=".*Pydantic V1.*isn't compatible with Python 3.14.*", module="langchain_core")

from ui_test_agent import main

if __name__ == "__main__":
    main()
