from __future__ import annotations

import signal
import sys
from contextlib import asynccontextmanager
from types import FrameType

from browser_use.browser.session import BrowserSession as Browser

_active_browsers: list[Browser] = []
_shutdown_requested = False


def is_shutdown_requested() -> bool:
    return _shutdown_requested


def request_shutdown() -> None:
    global _shutdown_requested
    _shutdown_requested = True


def register_browser(browser: Browser) -> None:
    if browser not in _active_browsers:
        _active_browsers.append(browser)


def unregister_browser(browser: Browser) -> None:
    if browser in _active_browsers:
        _active_browsers.remove(browser)


async def cleanup_all_browsers() -> None:
    global _active_browsers
    if not _active_browsers:
        return

    print(f"\n[INFO] Cleaning up {len(_active_browsers)} browser session(s)...")
    for browser in list(_active_browsers):
        try:
            if hasattr(browser, "kill"):
                await browser.kill()
                print("  [OK] Browser killed")
            elif hasattr(browser, "stop"):
                await browser.stop()
                print("  [OK] Browser stopped")
            elif hasattr(browser, "close"):
                await browser.close()
                print("  [OK] Browser closed")
        except Exception as e:
            print(f"  [WARN] Error during browser cleanup: {e}")
        finally:
            unregister_browser(browser)
    print("[INFO] Browser cleanup complete")


def _signal_handler(signum: int, _frame: FrameType | None) -> None:
    global _shutdown_requested
    if _shutdown_requested:
        print("\n[FORCE] Force exit requested")
        sys.exit(1)
    _shutdown_requested = True
    signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
    print(f"\n[INFO] Received {signal_name}, shutting down gracefully...")
    print("[INFO] Press Ctrl+C again to force exit")


def install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


@asynccontextmanager
async def managed_browser(browser: Browser):
    register_browser(browser)
    try:
        yield browser
    finally:
        try:
            if hasattr(browser, "kill"):
                await browser.kill()
            elif hasattr(browser, "stop"):
                await browser.stop()
            elif hasattr(browser, "close"):
                await browser.close()
        except Exception as e:
            print(f"[WARN] Browser cleanup error: {e}")
        finally:
            unregister_browser(browser)
