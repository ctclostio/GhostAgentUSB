"""Questions in the console, read key by key with a time limit: a page's confirm() or prompt(), and
the user's yes before a risky click."""

import os
import sys
import time

from . import runstate
from .console import c, clean, console_only, focus_console, warn
from .modelclient import ctrl_c_pressed, ctrl_c_stops_task
from .paths import log_only
from .runstate import STOP


# --------------------------------------------------------------------------------------------
# Questions in the console: a page's confirm() or prompt(), a click on Delete, Send or Pay. The
# answer is read key by key, so the wait has a time limit and Ctrl+C stops the task cleanly.
# Nobody at the console (or an unattended run) always means no.
# --------------------------------------------------------------------------------------------

ASK_SECONDS = 120


def _keyboard():
    """The console's key buffer (msvcrt), or None when nobody can be asked: an unattended run, or no
    console to type in (input redirected, or started without a window)."""
    if runstate.HEADLESS or os.name != "nt":
        return None
    try:
        if sys.stdin is None or not sys.stdin.isatty():
            return None
        import msvcrt
        return msvcrt
    except Exception:
        return None


def read_console_line(keyboard, timeout, prefill=""):
    """A line typed in the console: (text, "answered"); (None, "cancelled") for Escape; (None,
    "timeout") after `timeout` seconds; (None, "stopped") once Ctrl+C has been pressed. Keys typed
    before the question appeared are dropped, so an Enter meant for something else is no answer."""
    while keyboard.kbhit():
        keyboard.getwch()
    buf = list(prefill or "")
    if buf:
        console_only("".join(buf), end="")
    end = time.monotonic() + timeout
    try:
        while True:
            if STOP.is_set():
                return None, "stopped"
            if time.monotonic() >= end:
                return None, "timeout"
            if not keyboard.kbhit():
                time.sleep(0.05)
                continue
            ch = keyboard.getwch()
            if ch in ("\r", "\n"):
                return "".join(buf), "answered"
            if ch == "\x03":  # Ctrl+C arriving as a key instead of a signal: the same as the signal
                ctrl_c_pressed()  # (STOP is clear here, so this is a first press: noted for a second one)
                STOP.set()
                return None, "stopped"
            if ch == "\x1b":
                return None, "cancelled"
            if ch in ("\x00", "\xe0"):  # arrows and function keys come as two codes: ignore both
                keyboard.getwch()
                continue
            if ch == "\x08":
                if buf:
                    buf.pop()
                    console_only("\b \b", end="")
                continue
            if ch.isprintable():
                buf.append(ch)
                console_only(ch, end="")
    finally:
        console_only("")


def ask_console(question, prefill="", timeout=None, private=False):
    """Ask the user in the console and wait up to ASK_SECONDS. Returns (text or None, how), where how
    is "answered", "cancelled", "timeout", "stopped" (Ctrl+C) or "unattended" (nobody to ask).
    Ctrl+C here goes the usual way (ctrl_c_stops_task): the task stops, nothing is interrupted.
    private: the console log gets only the length of the answer."""
    keyboard = _keyboard()
    if keyboard is None:
        return None, "unattended"
    if STOP.is_set():
        return None, "stopped"
    timeout = ASK_SECONDS if timeout is None else timeout
    # The prefilled text can come from a page (a prompt()'s default): cleaned like everything else that
    # is printed, and on one line, so the line editing below stays on the line it started on.
    prefill = clean(prefill or "").replace("\n", " ").replace("\t", " ")
    with ctrl_c_stops_task(clear=False):
        focus_console()
        console_only("\a", end="")  # the bell: the user may be looking at the browser
        console_only(c("  " + clean(question) + " ", "1;33"), end="")
        text, how = read_console_line(keyboard, timeout, prefill)
    shown = f"<{len(text)} chars>" if private and text is not None else text if text is not None else f"<{how}>"
    log_only("  " + clean(question) + " " + clean(shown))
    if how == "timeout":
        warn("No answer in time, so the answer is no.")
    return text, how


def ask_yes_no(question):
    """(True only for y or yes, how): see ask_console."""
    text, how = ask_console(question + " [y/N]")
    return (text or "").strip().lower() in ("y", "yes"), how
