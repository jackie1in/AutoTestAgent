"""ReAct agent tools - registered via @action decorator.

These tools are automatically discovered and injected into the agent.
"""

from graph_agent.lib.tool_registry import action, get_registry

__all__ = ["register_default_tools", "get_registry"]


@action("Click element by index")
def click_element_by_index(index: int, controller):
    """Click an interactive element by its index."""
    return controller.click(index)


@action("Input text into a field")
def input_text(index: int, text: str, controller):
    """Click and type text into an interactive input element."""
    return controller.input_text(index, text)


@action("Select dropdown option")
def select_dropdown_option(index: int, option_text: str, controller):
    """Select a dropdown option by its text."""
    return controller.select_option(index, option_text)


@action("Scroll vertically")
def scroll(direction: str = "down", amount: int = 500, index: int | None = None, controller=None):
    """Scroll vertically. Without index: scrolls the document."""
    return controller.scroll(direction, amount, index)


@action("Scroll horizontally")
def scroll_horizontally(direction: str = "right", amount: int = 300, index: int | None = None, controller=None):
    """Scroll horizontally. Without index: scrolls the document."""
    return controller.scroll_horizontal(direction, amount, index)


@action("Wait for page load")
def wait(seconds: int = 1):
    """Wait for x seconds until the page or data is fully loaded."""
    import asyncio
    return asyncio.sleep(seconds)


@action("Navigate back")
def go_back(controller):
    """Navigate back to the previous page."""
    return controller.go_back()


@action("Close overlay/modal")
def close_overlay(controller):
    """Close any visible overlay (modal, drawer, dialog)."""
    return controller.close_overlay()


@action("Execute JavaScript")
def execute_javascript(script: str, controller):
    """Execute JavaScript code on the current page."""
    return controller.execute_js(script)


@action("Complete exploration", terminates=True)
def done(text: str, success: bool = True):
    """Complete the exploration task."""
    return {"text": text, "success": success}


def register_default_tools():
    """Register all default tools.
    
    This is called automatically when tools module is imported.
    The @action decorator already registers tools, so this is a no-op
    but kept for explicitness.
    """
    pass  # Tools are auto-registered by @action decorator
