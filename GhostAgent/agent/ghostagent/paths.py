"""Where the agent writes: the folders next to the agent (or under --data-dir), the console log in
logs\\, log rotation, and the folders it keeps on this PC.

LOGS, DOWNLOADS, PROFILE, SETTINGS_FILE and _CONSOLE_LOG change at run time (--data-dir, a
read-only drive, profile_on_stick, /forget-pc), so other modules read them as paths.NAME."""

import ctypes
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


# --------------------------------------------------------------------------------------------
# Where the agent writes. Everything lives next to the agent, unless the drive is read-only.
# --------------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent  # the GhostAgent folder (this is agent\ghostagent\paths.py)
BIN = ROOT / "bin"
LOGS = ROOT / "logs"
# Where the agent's own browser saves downloads. None = the PC's Downloads\GhostAgent folder (see
# download_folder); --data-dir (tests) sets a folder of its own.
DOWNLOADS = None
PROFILE = ROOT / "browser-profile"
SETTINGS_FILE = ROOT / "settings.json"            # the user's own settings (only what they changed)
DEFAULTS_FILE = ROOT / "settings.default.json"    # every setting with its note; replaced by updates
PROFILES_FILE = ROOT.parent / "tools" / "profiles.json"
BENCH_DIR = ROOT.parent / "tools" / "bench"

# Always the browsers on this drive: a PC that sets PLAYWRIGHT_BROWSERS_PATH for its own projects
# would otherwise make the agent start that folder's browser instead of the one built with it.
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(ROOT / "browsers")


def set_data_dir(path):
    """Keep logs, downloads, the browser profile and the user's settings.json in another folder
    (tests and benchmarks use a scratch folder, so they never touch the real ones)."""
    global LOGS, DOWNLOADS, PROFILE, SETTINGS_FILE
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    LOGS, DOWNLOADS, PROFILE = path / "logs", path / "downloads", path / "browser-profile"
    SETTINGS_FILE = path / "settings.json"


def ensure_writable_data():
    """Test-write the logs folder. A read-only stick (BitLocker To Go or a 'deny write access to
    removable disks' policy) would otherwise crash the agent at its first log line. Returns a
    notice when the logs and the browser profile had to move to this PC, else None. (Downloads are
    saved on this PC anyway: see download_folder.)"""
    global LOGS, PROFILE

    def writable(folder):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / f".write-test-{os.getpid()}"
            probe.write_bytes(b"")
            probe.unlink()
            return True
        except OSError:
            return False

    if writable(LOGS):
        return None
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "GhostAgent"
    LOGS, PROFILE = base / "logs", base / "browser-profile"
    if not writable(LOGS):
        return "Neither this drive nor this PC's user folder can be written to, so no logs are kept."
    return (f"This drive cannot be written to, so the logs and the browser profile are kept on "
            f"this PC instead, in {base}.")


def download_folder():
    """Where the agent's own browser saves downloads: the PC's Downloads\\GhostAgent folder. A PC's
    disk is NTFS, so the Mark of the Web can be written (an exFAT stick cannot hold it), and a file a
    page pushes does not fill the stick. --data-dir (tests) moves it."""
    if DOWNLOADS is not None:
        return Path(DOWNLOADS)
    return Path(os.environ.get("USERPROFILE") or Path.home()) / "Downloads" / "GhostAgent"


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CONSOLE_LOG = None


def short_path(path):
    """The 8.3 short form of an existing path (C:\\Users\\LONGNA~1 for C:\\Users\\Long Name), or the
    path itself when it has none or it cannot be read."""
    path = str(path)
    if os.name != "nt":
        return path
    try:
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        k32.GetShortPathNameW.restype = wintypes.DWORD
        need = k32.GetShortPathNameW(path, None, 0)
        if need:
            buf = ctypes.create_unicode_buffer(need)
            got = k32.GetShortPathNameW(path, buf, need)
            if 0 < got < need:
                return buf.value
    except Exception:
        pass
    return path


_HOME_FORMS = {}


def _home_pattern(home):
    """The regular expression for every way a log can spell the home folder: its long and its 8.3
    short form (a program that got the short form from Windows writes that), each with \\ or /.
    Worked out once per home folder."""
    if home not in _HOME_FORMS:
        forms = {f.rstrip("\\/") for f in (home, short_path(home))}
        forms |= {f.replace("\\", "/") for f in forms}
        forms = sorted((f for f in forms if len(f) >= 4), key=len, reverse=True)
        _HOME_FORMS[home] = "|".join(re.escape(f) for f in forms)
    return _HOME_FORMS[home]


def hide_home(text):
    """text with this account's home folder (C:\\Users\\<name>, or its short form C:\\Users\\NAME~1)
    written as %USERPROFILE%, so the logs kept on the stick do not name the Windows account of every
    PC it was used on."""
    home = str(Path.home()).rstrip("\\/")
    if len(home) < 4 or not text:
        return text
    pattern = _home_pattern(home)
    return re.sub(f"(?:{pattern})(?=[\\\\/\"'\\s]|$)", "%USERPROFILE%", text, flags=re.I)


def hide_home_in_file(path):
    """Rewrite a finished log file with the home folder written as %USERPROFILE% (see hide_home)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        hidden = hide_home(text)
        if hidden != text:
            Path(path).write_text(hidden, encoding="utf-8")
    except OSError:
        pass


class _ConsoleLogFile:
    """logs\\agent-<time>.log: what the console showed, as plain text (no colour codes)."""

    def __init__(self, path):
        self.path = path
        # Re-entrant: the Ctrl+C handler prints, and it can run on the main thread while that thread
        # is in the middle of writing a line here. A plain Lock would then wait on itself forever.
        self.lock = threading.RLock()
        self.f = open(path, "a", encoding="utf-8", errors="replace")

    def write_line(self, text):
        with self.lock:
            try:
                self.f.write(hide_home(ANSI_RE.sub("", text)).rstrip() + "\n")
                self.f.flush()
            except (OSError, ValueError):
                pass

    def close(self):
        with self.lock:
            try:
                self.f.close()
            except OSError:
                pass


class _Tee:
    """Wraps sys.stdout or sys.stderr: everything still goes to the console, and every finished
    line also goes to the console log. A carriage return starts the line over, so the live
    progress line ('step 3: thinking ... 12s') only leaves its final state in the log."""

    def __init__(self, stream, log):
        self._stream = stream
        self._log = log
        self._line = ""

    def write(self, text):
        n = self._stream.write(text)
        try:
            for piece in re.split(r"(\r\n|\n|\r)", text):
                if piece in ("\n", "\r\n"):
                    self._log.write_line(self._line)
                    self._line = ""
                elif piece == "\r":
                    self._line = ""
                elif piece:
                    self._line += piece
        except Exception:
            pass  # the log must never break the console
        return n

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        # fileno() and isatty() come from the real console, so input() keeps its line editing.
        return getattr(self._stream, name)


def install_console_log():
    """Mirror the console into logs\\agent-YYYYMMDD-HHMMSS.log (prints, warnings and tracebacks).
    Returns the log's path, or None when it cannot be written."""
    global _CONSOLE_LOG
    if _CONSOLE_LOG is not None:
        return _CONSOLE_LOG.path
    try:
        LOGS.mkdir(parents=True, exist_ok=True)
        log = _ConsoleLogFile(LOGS / f"agent-{datetime.now():%Y%m%d-%H%M%S}.log")
    except OSError:
        return None
    _CONSOLE_LOG = log
    sys.stdout = _Tee(sys.stdout, log)
    sys.stderr = _Tee(sys.stderr, log)
    return log.path


def log_only(text):
    """A line for the console log that the console itself does not need (for example the task
    that was just typed, which the console already shows)."""
    if _CONSOLE_LOG is not None:
        _CONSOLE_LOG.write_line(text)


LOG_PATTERNS = ("llama-server-*.log", "agent-*.log", "steps-*.jsonl")


def prune_logs(folder=None, keep=20):
    """Keep only the newest `keep` files of each kind of log, so the stick does not fill up."""
    folder = Path(folder or LOGS)
    for pattern in LOG_PATTERNS:
        files = []
        for p in folder.glob(pattern):
            try:
                files.append((p.stat().st_mtime, p.name, p))
            except OSError:
                pass
        files.sort(reverse=True)
        for _, _, old in files[keep:]:
            try:
                old.unlink()
            except OSError:
                pass


def pc_data_dir():
    """%LOCALAPPDATA%\\GhostAgent: what the agent keeps on this PC rather than on the stick."""
    return Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")) / "GhostAgent"


def screenshots_dir():
    """Where save_screenshots puts its files: on this PC, not on a stick that can be lost."""
    return pc_data_dir() / "screenshots"


def prune_screenshots(days=7):
    """Delete saved screenshots older than `days` days: they show the pages that were visited."""
    folder = screenshots_dir()
    if not folder.is_dir():
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    for p in folder.glob("*.png"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def use_profile_location(settings):
    """profile_on_stick false: the bundled/Edge browser profile lives on this PC
    (%LOCALAPPDATA%\\GhostAgent\\browser-profile) instead of on the stick. Cookie sign-ins are tied to
    the PC where they were made anyway (Windows encrypts them for that PC and account); a lost stick
    then carries no site data at all, but every PC needs its own sign-ins."""
    global PROFILE
    if not settings.get("profile_on_stick", True):
        PROFILE = pc_data_dir() / "browser-profile"
