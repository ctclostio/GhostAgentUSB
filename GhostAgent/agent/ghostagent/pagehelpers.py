"""Helpers for the browser: page JavaScript with a time limit, stall detection, the observation's
header and footer, reader_mode slices and downloads."""

import json
import os
import re
import urllib.parse
from pathlib import Path

from .console import clean, one_line
from .guards import untrusted, url_scheme
from .pagejs import READER_CHUNK, READER_MAX_CHUNK


# --------------------------------------------------------------------------------------------
# Page JavaScript with a time limit
# --------------------------------------------------------------------------------------------

PAGE_JS_SECONDS = 3.0      # a page that has not answered by then is busy (a long or endless script)
READER_JS_SECONDS = 10.0   # reader_mode walks the whole page
BUSY_RESULT = ("The page is busy (its scripts are not responding), so nothing was done. Use wait_web to give it "
               "time, or go to another page.")


class PageBusy(Exception):
    """A page or frame did not run a short script within its time limit."""


def page_js(target, fn, arg=None, seconds=PAGE_JS_SECONDS, handle=False):
    """Like target.evaluate(fn, arg) for a page or frame, but PageBusy after `seconds`: Playwright's
    evaluate has no time limit, so a page stuck in a long script would hold the step forever (and a
    Ctrl+C there would then hit Playwright). wait_for_function does have one; it waits for a truthy
    value, so the result travels inside an object. handle=True returns the result as a JSHandle (for
    elements)."""
    try:
        box = target.wait_for_function("(a) => ({v: (" + fn + ")(a)})", arg=arg, timeout=seconds * 1000, polling=100)
    except Exception as e:
        if type(e).__name__ == "TimeoutError":
            raise PageBusy() from None
        raise
    if handle:
        try:
            return box.get_property("v")
        finally:
            try:
                box.dispose()
            except Exception:
                pass
    try:
        return (box.json_value() or {}).get("v")
    finally:
        try:
            box.dispose()
        except Exception:
            pass


# --------------------------------------------------------------------------------------------
# Stalls: actions that keep leaving the page as it was
# --------------------------------------------------------------------------------------------

STALL_SAME_ACTION = 3      # the same action on the same unchanged page this many times ...
STALL_UNCHANGED_STEPS = 5  # ... or this many actions in a row that changed nothing: tell the model
STALL_ALERT = ("<system_alert>Your last {n} actions did not change the page. Do not repeat them. Try a fundamentally "
               "different approach (another element, go_to_web a known URL, ctrl_f_web, reader_mode, another site). "
               "If something blocks you, call answer and say what.</system_alert>")
CHECKIN_ALERT = ("<system_alert>{n} steps so far. Check your strategy: if the last steps did not bring you closer to "
                 "the goal, change your approach instead of repeating it. If something blocks you, call answer and "
                 "say what.</system_alert>")


class StallDetector:
    """Counts actions that left the page unchanged. record() returns the number of such actions in a
    row once an alert is due (the same action on the same unchanged page STALL_SAME_ACTION times, or
    STALL_UNCHANGED_STEPS unchanged actions in a row), else 0; the counts start over after an alert
    and after any action that changed the page. Actions whose effect is unknown count neither way."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.unchanged = 0
        self.repeats = {}

    def record(self, action, page, changed):
        if changed is None:
            return 0
        if changed:
            self.reset()
            return 0
        self.unchanged += 1
        key = (action, page)
        self.repeats[key] = self.repeats.get(key, 0) + 1
        if self.repeats[key] >= STALL_SAME_ACTION or self.unchanged >= STALL_UNCHANGED_STEPS:
            n = self.unchanged
            self.reset()
            return n
        return 0


def action_signature(name, args):
    """What makes two actions 'the same' for the stall check: the tool, the element it names, the
    point in 25-unit squares (a few pixels apart is the same target) and the other arguments."""
    def bucket(v):
        try:
            return int(float(v)) // 25
        except (TypeError, ValueError):
            return None
    rest = {k: v for k, v in sorted(args.items()) if k not in ("element", "x", "y")}
    return (name, str(args.get("element", "")).strip().lower(), bucket(args.get("x")), bucket(args.get("y")),
            json.dumps(rest, sort_keys=True, ensure_ascii=False, default=str))


def with_alert(observation, text):
    """The observation message with a harness alert in front of its first text (still starting the
    message, so is_observation recognises it)."""
    parts = observation.get("content")
    if not text or not isinstance(parts, list) or not parts or parts[0].get("type") != "text":
        return observation
    parts[0] = dict(parts[0], text=text + "\n" + parts[0]["text"])
    return observation


# --------------------------------------------------------------------------------------------
# observation_format "header": the page's scroll position and extent around the screenshot
# --------------------------------------------------------------------------------------------

HEADER_RULE = "=" * 20


def page_title(text, limit=120):
    """A tab title for the model: control characters removed, one line, at most `limit` characters."""
    return one_line(clean(str(text or "")), limit)


def title_is_address(title, url):
    """True when a tab's title is only its address, which is what the browser reports for a page
    without a title: the URL without its scheme, unescaped ('a b' for 'a%20b'), and for a bare host
    without the slash ('127.0.0.1:8765' for http://127.0.0.1:8765/; Chromium keeps any other scheme,
    so 'https://example.com' for https://example.com/). A real title that is only the start of the
    address (a page titled 'localhost' on localhost) is kept."""
    if not title:
        return False
    bare = url.split("://", 1)[-1]
    forms = {url, bare}
    try:
        parts = urllib.parse.urlsplit(url)
        if parts.path == "/" and not parts.query and not parts.fragment and bare.endswith("/"):
            forms |= {bare[:-1], url[:-1]}
    except ValueError:
        pass
    forms |= {urllib.parse.unquote(f) for f in forms}
    return title in forms


def observation_header(url, title, size, metrics, tabs, notes):
    """The text before the screenshot. size: the screenshot's (w, h); metrics: METRICS_JS's result or
    None; tabs: [(index, title, current, opened_by_page)], listed only when there are several; notes:
    sentences about what happened meanwhile."""
    lines = ["<observation>", HEADER_RULE, f"URL: {url}", f"Title: {untrusted(page_title(title))}"]
    w, h = size if size and size[0] else ((metrics or {}).get("iw"), (metrics or {}).get("ih"))
    if w and h:
        lines.append(f"Viewport: {w}\u00d7{h}px")
    if metrics:
        sh, ch, y = max(1, metrics["sh"]), max(1, metrics["ch"]), max(0, metrics["y"])
        room = sh - ch
        lines.append(f"Position: {100.0 * min(y, room) / room if room > 0 else 0.0:.1f}% ({y}px / {sh}px)")
        lines.append(f"Visible: {min(100, round(100 * ch / sh))}% of page")
    else:
        lines.append("Position: unknown (the page did not answer in time)")
    if len(tabs) > 1:
        lines.append(f"Open Tabs ({len(tabs)}):")
        for i, t, current, by_page in tabs:
            mark = " (current)" if current else " (opened by the page)" if by_page else ""
            lines.append(f"  [{i}] {untrusted(page_title(t))}{mark}")
    if notes:
        lines.append("Note: " + " ".join(notes))
    lines.append(HEADER_RULE)
    return "\n".join(lines) + "\n"


def observation_footer(metrics):
    """The text after the screenshot: kept to a few tokens (llama.cpp handles a text piece of more than
    32 tokens after an image as an extra batch, which costs time on every step)."""
    if not metrics:
        return "\n</observation>"
    y, x = max(0, metrics["y"]), max(0, metrics["x"])
    above = y
    below = max(0, metrics["sh"] - y - metrics["ch"])
    if above <= 1 and below <= 1:
        bits = ["Entire page visible (do not scroll down)"]
    else:
        bits = []
        if above > 1:
            bits.append(f"\u2191 {above}px above")
        bits.append(f"\u2193 {below}px below (scroll down to continue)" if below > 1 else "bottom of the page")
    if x > 1:
        bits.append("\u2190 More left")
    if metrics["sw"] - x - metrics["cw"] > 1:
        bits.append("\u2192 More right")
    return "\n" + " | ".join(bits) + "\n</observation>"


# --------------------------------------------------------------------------------------------
# reader_mode pages and downloads
# --------------------------------------------------------------------------------------------

def reader_slice(text, start=None, end=None):
    """(chunk, a, b, total): characters a to b of the page's text. start defaults to 0 and end to
    start + READER_CHUNK; one call returns at most READER_MAX_CHUNK characters."""
    total = len(text)

    def number(v, default):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default
    a = min(max(0, number(start, 0)), total)
    b = number(end, a + READER_CHUNK)
    if b <= a:
        b = a + READER_CHUNK
    b = min(b, a + READER_MAX_CHUNK, total)
    return text[a:b], a, b, total


WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}


def safe_download_name(name):
    """A download's suggested name as a plain Windows file name: no folders, no characters Windows
    forbids, no reserved device names, at most 150 characters."""
    name = re.split(r"[\\/]", str(name or ""))[-1]
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name)
    # Direction marks could make "invoice<RLO>fdp.exe" look like a PDF in Explorer.
    name = re.sub(r"[\x7f-\x9f\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", name).strip(" .")
    stem, ext = os.path.splitext(name)
    if not stem or stem.split(".")[0].upper() in WINDOWS_RESERVED_NAMES:
        stem = "download" + ("-" + stem if stem else "")
    return (stem[:150 - len(ext[:20])] + ext[:20]) or "download"


def unique_path(folder, name):
    """folder/name, or 'name (1).ext', 'name (2).ext' ... when that is taken: a download never
    replaces a file that is already there."""
    folder = Path(folder)
    stem, ext = os.path.splitext(name)
    target, n = folder / name, 1
    while target.exists():
        target = folder / f"{stem} ({n}){ext}"
        n += 1
    return target


def mark_of_the_web(path, url):
    """Write the Zone.Identifier stream Windows reads to know a file came from the internet (SmartScreen
    and Office Protected View then warn before it is opened), as browsers do. Needs NTFS. Returns True
    when written."""
    host = url if url_scheme(url) in ("http", "https") else "about:internet"
    try:
        with open(str(path) + ":Zone.Identifier", "wb") as f:
            f.write(f"[ZoneTransfer]\r\nZoneId=3\r\nHostUrl={host}\r\n".encode("ascii", "replace"))
        return True
    except OSError:
        return False


def size_text(n):
    return f"{n / 2**20:.1f} MB" if n >= 2**20 else f"{max(1, round(n / 1024))} KB" if n >= 1024 else f"{n} bytes"
