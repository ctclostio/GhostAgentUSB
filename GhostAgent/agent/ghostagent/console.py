"""Console output: colours (plain lines when the output is not a console), progress lines redrawn
in place, text from pages and the model made safe to print, and the console window's title and focus.

PLAIN_OUTPUT is set at start-up (_enable_ansi); other modules read it as console.PLAIN_OUTPUT."""

import ctypes
import os
import re
import sys
import time

from . import runstate


# --------------------------------------------------------------------------------------------
# Console helpers
# --------------------------------------------------------------------------------------------

# True when the output is not a console (piped or redirected to a file, as unattended runs often are):
# then no colour codes and no lines redrawn in place, only plain lines (see LiveLine).
PLAIN_OUTPUT = False
# How often a progress line is written when it cannot be redrawn in place (seconds).
PLAIN_STATUS_SECONDS = 15


def output_is_console(stream=None):
    stream = sys.stdout if stream is None else stream
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError, ValueError):
        return False


def _enable_ansi():
    global PLAIN_OUTPUT
    PLAIN_OUTPUT = not output_is_console()
    if os.name == "nt" and not PLAIN_OUTPUT:
        os.system("")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def c(text, code):
    return str(text) if PLAIN_OUTPUT else f"\033[{code}m{text}\033[0m"


class LiveLine:
    """A progress line redrawn in place on the console ('\\r'). When the output is not a console,
    nothing is redrawn: a plain line is written at most every `every` seconds (never with every=None),
    so a log gets readable one-line progress messages instead of carriage returns."""

    def __init__(self, prefix="  ", every=PLAIN_STATUS_SECONDS):
        self.prefix = prefix
        self.every = every
        self.last = None
        self.open = False  # a redrawn line is on the console and not ended yet

    def show(self, text):
        if PLAIN_OUTPUT:
            now = time.monotonic()
            if self.every is not None and (self.last is None or now - self.last >= self.every):
                self.last = now
                print(self.prefix + text, flush=True)
            return
        print(f"\r{self.prefix}{text}   ", end="", flush=True)
        self.open = True

    def close(self, text=None):
        """End the line. With `text`, that is its final state, shown either way."""
        if text is not None:
            print((self.prefix if PLAIN_OUTPUT else f"\r{self.prefix}") + text + ("" if PLAIN_OUTPUT else "   "),
                  flush=True)
        elif self.open:
            print(flush=True)
        self.open = False

    def clear(self, width):
        """Wipe the redrawn line so it leaves no trace (nothing to do without a console)."""
        if self.open:
            print("\r" + " " * width + "\r", end="", flush=True)
        self.open = False


# Escape sequences a terminal acts on: OSC (window title, links) and the other string commands,
# CSI (colours, cursor moves, erasing lines), and any other ESC pair. Page text and model output
# could otherwise retitle the console, hide lines or fake a question once they are printed.
_ESCAPE_RE = re.compile(
    r"(?:\x1b\]|\x9d)[^\x07\x1b\x9c]*(?:\x07|\x1b\\|\x9c)?"
    r"|(?:\x1b[PX^_]|[\x90\x98\x9e\x9f])[^\x1b\x9c]*(?:\x1b\\|\x9c)?"
    r"|(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]?"
    r"|\x1b[ -/]*[0-~]?")
# C0 and C1 control characters except tab and newline, plus the Unicode controls that reverse or
# isolate the direction of text (they can make a line read differently from what it says).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def clean(text):
    """Text from a web page or the model, made safe to print: no escape sequences and no control
    characters other than tab and newline."""
    return _CONTROL_RE.sub("", _ESCAPE_RE.sub("", str(text)))


def info(msg):
    print(c("  " + clean(msg), "90"), flush=True)


def head(msg):
    print(c(clean(msg), "1;36"), flush=True)


def warn(msg):
    print(c("  ! " + clean(msg), "33"), flush=True)


def err(msg):
    print(c("  x " + clean(msg), "31"), flush=True)


def console_only(text, end="\n"):
    """Write to the console without copying it to the console log (a secret such as the web UI's
    key, or the characters echoed while the user answers a question)."""
    stream = getattr(sys.stdout, "_stream", sys.stdout)
    try:
        stream.write(text + end)
        stream.flush()
    except (OSError, ValueError):
        pass


def first_line(e):
    """The first line of an exception's message (or its repr when it has none)."""
    text = str(e).strip()
    return text.splitlines()[0] if text else repr(e)


def one_line(text, limit=220):
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def set_console_title(text):
    if os.name == "nt" and not runstate.HEADLESS:
        try:
            ctypes.windll.kernel32.SetConsoleTitleW(text)
        except Exception:
            pass


def focus_console():
    """Best effort: bring the console back in front of the browser so the answer is visible.
    Never in unattended runs: those must not take the focus from whatever the user is doing."""
    if os.name != "nt" or runstate.HEADLESS:
        return
    try:
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            user32 = ctypes.windll.user32
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
    except Exception:
        pass
