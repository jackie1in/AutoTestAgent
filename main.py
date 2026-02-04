#!/usr/bin/env python3
"""
UI Automation Testing Agent - Main Entry Point

A browser automation testing tool based on Browser-Use that can:
- Record UI interactions via AI agent
- Replay recorded test cases
- Manage test cases

Usage:
  python main.py record     - Record a new test case
  python main.py replay     - Replay a saved test case
  python main.py list       - List all saved test cases
  python main.py view <id>  - View details of a test case
  python main.py delete <id> - Delete a test case
  python main.py help       - Show help message
"""

from ui_test_agent import main

if __name__ == "__main__":
    main()
