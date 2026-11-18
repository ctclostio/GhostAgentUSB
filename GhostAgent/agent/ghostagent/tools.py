"""The tools the model is offered (names follow H Company's browser action set), their arguments
under either schema, and a one-line description of a call."""

import copy
import json

from .settings import FALSE_WORDS, TRUE_WORDS


# --------------------------------------------------------------------------------------------
# Tools (names follow H Company's browser action set)
# --------------------------------------------------------------------------------------------

def fn(name, description, required, /, **properties):
    # (positional-only, so a tool may have arguments called "name" or "description")
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


ELEMENT = {"type": "string", "description": "Detailed description of the target UI element"}
X = {"type": "integer", "description": "X coordinate as integer in [0, 1000]"}
Y = {"type": "integer", "description": "Y coordinate as integer in [0, 1000]"}

TOOLS = [
    fn("click_web", "Click at (x, y) coordinates on the current page.", ["element", "x", "y"],
       element=ELEMENT, x=X, y=Y,
       button={"type": "string", "enum": ["left", "right"], "description": "Mouse button (default left)"},
       double_click={"type": "boolean", "description": "Double-click instead of a single click"}),
    fn("write", "Click the input field at (x, y) and type text into it. Replaces the field's current text unless clear is false.",
       ["element", "x", "y", "text"],
       element=ELEMENT, x=X, y=Y, text={"type": "string", "description": "Text to type"},
       press_enter={"type": "boolean", "description": "Press Enter after typing"},
       clear={"type": "boolean", "description": "Clear the field before typing (default true)"}),
    fn("select_option", "Pick an option from the native <select> dropdown at (x, y).", ["element", "x", "y", "option"],
       element=ELEMENT, x=X, y=Y, option={"type": "string", "description": "Visible text of the option to select"}),
    fn("move_mouse_web", "Move the mouse to (x, y) to reveal hovers, tooltips or menus.", ["element", "x", "y"],
       element=ELEMENT, x=X, y=Y),
    fn("press_keys_web", "Press a key or keyboard shortcut, for example Enter, Escape, Tab, PageDown, 'ctrl + a', 'shift + tab'. Separate several presses with spaces.",
       ["keys"], keys={"type": "string", "description": "Keys to press"}),
    fn("scroll_web", "Scroll the page, or the scrollable container at (x, y).", ["direction"],
       direction={"type": "string", "enum": ["up", "down", "left", "right"]},
       x={"type": "integer", "description": "Optional X of the container to scroll, in [0, 1000]"},
       y={"type": "integer", "description": "Optional Y of the container to scroll, in [0, 1000]"}),
    fn("go_to_web", "Navigate the current tab to a URL.", ["url"], url={"type": "string"}),
    fn("go_back_web", "Go back in the browser history.", []),
    fn("refresh_web", "Reload the current page.", []),
    fn("switch_tab_web", "Switch to another open tab by its index, or open a new tab at a URL.", [],
       tab_index={"type": "integer", "description": "Index of the tab to switch to"},
       new_tab_url={"type": "string", "description": "URL to open in a new tab"}),
    fn("close_tab_web", "Close a tab (default: the current one).", [],
       tab_index={"type": "integer", "description": "Index of the tab to close"}),
    fn("ctrl_f_web", "Find text on the page and scroll to the next match.", ["query"], query={"type": "string"}),
    fn("reader_mode", "Return the main content of the current page as markdown text with links, to read long pages "
       "without scrolling. Long pages come in parts: the result says which characters it shows and where the next "
       "part starts.", [],
       start={"type": "integer", "description": "First character to return (default 0)"},
       end={"type": "integer", "description": "Character to stop at (default start + 8000)"}),
    fn("wait_web", "Wait for the page to settle.", ["seconds"],
       seconds={"type": "number", "description": "Seconds to wait, at most 60"}),
    fn("answer", "Provide a final answer to the user, or ask the user a question. Ends the task.", ["content"],
       content={"type": "string", "description": "The answer content"}),
]

# answer's optional outcome (tool_schema "h"): how the task went, shown in the console.
ANSWER_OUTCOMES = ("success", "partial", "infeasible", "blocked")


def _enter_text(enter_default, schema):
    """The description of write's Enter switch, which states its default."""
    if schema == "h":
        if enter_default:
            return "Press Enter after typing (default true); set it to false so that a form is not submitted"
        return "Press Enter after typing, for example to send a search (default false)"
    return "Press Enter after typing (default true)" if enter_default else "Press Enter after typing"


# tool_schema "h": the names, arguments and defaults of H Company's Holo4 web tools (the visual set),
# described in GhostAgent's own words. Holo4 was trained on these, so matching them keeps the model
# on familiar ground; tool_choice "required" lets it use only the names declared here.
def h_tools(enter_default=False):
    return [
        fn("click_web", "Click at (x, y) on the current page.", ["element", "x", "y"],
           element=ELEMENT, x=X, y=Y,
           button={"type": "string", "enum": ["left", "right"], "description": "Mouse button (default left)"},
           double_click={"type": "boolean", "description": "Double-click instead of a single click"}),
        fn("write", "Click the input field at (x, y) to focus it, then type text into it.",
           ["element", "x", "y", "text"],
           element=ELEMENT, x=X, y=Y, text={"type": "string", "description": "Text to type"},
           overwrite={"type": "boolean", "description": "Replace the field's current text (default true); "
                                                        "false adds to it"},
           enter={"type": "boolean", "description": _enter_text(enter_default, "h")}),
        fn("select_option", "Pick an option from the native <select> dropdown at (x, y).", ["element", "x", "y", "option"],
           element=ELEMENT, x=X, y=Y, option={"type": "string", "description": "Visible text of the option to select"}),
        fn("move_mouse_web", "Move the mouse to (x, y) to reveal hovers, tooltips or menus.", ["element", "x", "y"],
           element=ELEMENT, x=X, y=Y),
        fn("press_keys_web", "Press keys on the keyboard: modifiers first, then the key, joined by +. Separate several "
           "presses with spaces. Examples: pagedown, down, shift + tab, end, home, ctrl + c, enter.",
           ["keys"], keys={"type": "string", "description": "The keys to press"}),
        fn("scroll_web", "Scroll the page, or the scrollable container given in scroll_element.", ["direction"],
           direction={"type": "string", "enum": ["up", "down", "left", "right"]},
           scroll_element={"type": "object", "description": "The container to scroll; leave it out to scroll the page",
                           "properties": {"element": ELEMENT, "x": X, "y": Y}, "required": ["x", "y"]},
           num_pixels={"type": "integer", "description": "How far to scroll, in pixels (default: most of the visible area)"}),
        fn("go_to_web", "Open a URL in the current tab.", ["url"], url={"type": "string"}),
        fn("go_back_web", "Go back in the browser history.", []),
        fn("refresh_web", "Reload the current page.", []),
        fn("switch_tab_web", "Switch to an open tab by its id (the number in brackets in the list of tabs), or open a "
           "URL in a new tab.", [],
           tab_id={"type": "integer", "description": "Id of the tab to switch to"},
           url={"type": "string", "description": "URL to open in a new tab"}),
        fn("close_tab_web", "Close a tab by its id (default: the current tab).", [],
           tab_id={"type": "integer", "description": "Id of the tab to close"}),
        fn("drag_and_drop_web", "Press the mouse button at (x1, y1), move to (x2, y2) and release it: for sliders, "
           "moving items and reordering lists.", ["description", "x1", "y1", "x2", "y2"],
           description={"type": "string", "description": "What is dragged, and where to"},
           x1=X, y1=Y, x2=X, y2=Y),
        fn("ctrl_f_web", "Find text on the page and scroll to the next match.", ["query"], query={"type": "string"}),
        fn("reader_mode", "Return the main content of the current page as markdown text with links, to read long pages "
           "without scrolling. Long pages come in parts: the result says which characters it shows and where the next "
           "part starts.", [],
           start={"type": "integer", "description": "First character to return (default 0)"},
           end={"type": "integer", "description": "Character to stop at (default start + 8000)"}),
        fn("wait_web", "Wait for the page to settle.", ["seconds"],
           seconds={"type": "number", "description": "Seconds to wait, at most 60"}),
        fn("answer", "Give the final answer to the user, or ask the user a question. Ends the task.", ["content"],
           content={"type": "string", "description": "The answer content"},
           outcome={"type": "string", "enum": list(ANSWER_OUTCOMES),
                    "description": "How the task went: success, partial (only part of it is done), infeasible (it "
                                   "cannot be done) or blocked (something such as a sign-in or a CAPTCHA stops you)"}),
    ]


TOOLS_H = h_tools()


def tool_list(settings):
    """The tools the model is offered: tool_schema "classic" (TOOLS) or "h" (h_tools), with write's
    Enter default as write_enter_default sets it."""
    enter_default = bool(settings.get("write_enter_default", False))
    if str(settings.get("tool_schema") or "classic") == "h":
        return h_tools(enter_default)
    if not enter_default:
        return TOOLS
    tools = copy.deepcopy(TOOLS)
    for t in tools:
        if t["function"]["name"] == "write":
            t["function"]["parameters"]["properties"]["press_enter"]["description"] = _enter_text(True, "classic")
    return tools


def _flag(value, default):
    """A tool argument meant as true/false: models sometimes send "true" or 1 instead of true."""
    if value is None:
        return default
    if isinstance(value, str):
        low = value.strip().lower()
        return low in TRUE_WORDS if low in TRUE_WORDS | FALSE_WORDS else default
    return bool(value)


def canonical_args(name, args, enter_default=False):
    """A tool call's arguments under the names the executor uses, whichever tool schema the model
    was given: the "h" names (write's overwrite/enter, tab_id/url, scroll_element/num_pixels) and the
    classic ones (clear/press_enter, tab_index/new_tab_url, x/y on scroll_web) both work, so a model
    that mixes them, or an older test, still gets what it meant. write's enter is always filled in
    (write_enter_default when the model said nothing). Returns a new dict."""
    a = dict(args or {})

    def take(new, old):
        value = a.pop(new, None)
        legacy = a.pop(old, None)
        return value if value is not None else legacy

    if name == "write":
        a["overwrite"] = _flag(take("overwrite", "clear"), True)
        a["enter"] = _flag(take("enter", "press_enter"), enter_default)
    elif name in ("switch_tab_web", "close_tab_web"):
        tab = take("tab_id", "tab_index")
        if tab is not None:
            a["tab_id"] = tab
        if name == "switch_tab_web":
            url = take("url", "new_tab_url")
            if url not in (None, ""):
                a["url"] = url
    elif name == "scroll_web":
        target = a.pop("scroll_element", None)
        if isinstance(target, str):
            try:
                target = json.loads(target)
            except ValueError:
                target = None
        if isinstance(target, dict) and target.get("x") is not None and target.get("y") is not None:
            a["x"], a["y"] = target["x"], target["y"]
            if target.get("element"):
                a["element"] = target["element"]
    elif name == "drag_and_drop_web":
        if "description" not in a and "element" in a:
            a["description"] = a.pop("element")
    elif name == "answer":
        outcome = str(a.get("outcome") or "").strip().lower()
        if outcome in ANSWER_OUTCOMES:
            a["outcome"] = outcome
        else:
            a.pop("outcome", None)
    return a


def describe_call(name, args):
    """One line saying what a tool call does, for the console, the trace and the notes kept when a
    long history is shortened. Takes either schema's argument names (see canonical_args)."""
    if name in ("click_web", "write", "select_option", "move_mouse_web"):
        bits = f"'{args.get('element', '')}' at ({args.get('x')}, {args.get('y')})"
        if name == "write":
            enter = args.get("enter", args.get("press_enter"))
            bits += f" <- \"{args.get('text', '')}\"" + (" + Enter" if _flag(enter, False) else "")
        if name == "select_option":
            bits += f" -> \"{args.get('option', '')}\""
        return f"{name} {bits}"
    if name == "drag_and_drop_web":
        return (f"{name} '{args.get('description', args.get('element', ''))}' from ({args.get('x1')}, {args.get('y1')}) "
                f"to ({args.get('x2')}, {args.get('y2')})")
    return f"{name} {json.dumps(args, ensure_ascii=False)}" if args else name
