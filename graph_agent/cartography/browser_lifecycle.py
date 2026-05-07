from __future__ import annotations

import logging
import signal
import sys
from contextlib import asynccontextmanager
from types import FrameType

from browser_use.browser.session import BrowserSession as Browser

logger = logging.getLogger(__name__)

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

    logger.info("Cleaning up %d browser session(s)...", len(_active_browsers))
    for browser in list(_active_browsers):
        try:
            if hasattr(browser, "kill"):
                await browser.kill()
                logger.info("Browser killed")
            elif hasattr(browser, "stop"):
                await browser.stop()
                logger.info("Browser stopped")
            elif hasattr(browser, "close"):
                await browser.close()
                logger.info("Browser closed")
        except Exception as e:
            logger.warning("Error during browser cleanup: %s", e)
        finally:
            unregister_browser(browser)
    logger.info("Browser cleanup complete")


def _signal_handler(signum: int, _frame: FrameType | None) -> None:
    global _shutdown_requested
    if _shutdown_requested:
        logger.warning("Force exit requested")
        sys.exit(1)
    _shutdown_requested = True
    signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
    logger.info("Received %s, shutting down gracefully...", signal_name)
    logger.info("Press Ctrl+C again to force exit")


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
            logger.warning("Browser cleanup error: %s", e)
        finally:
            unregister_browser(browser)
