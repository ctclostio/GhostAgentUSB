"""The user's own Google Chrome or Microsoft Edge (attached to over remote debugging), and the
Windows clipboard."""

import ctypes
import json
import os
import re
import subprocess
import time
from pathlib import Path

from .llamaserver import port_is_free
from .winproc import CREATE_BREAKAWAY_FROM_JOB, CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW


# --------------------------------------------------------------------------------------------
# The user's own Google Chrome or Microsoft Edge. Since version 136 their normal profile ignores
# --remote-debugging-port; instead the user turns on "Allow remote debugging for this browser
# instance" once at chrome://inspect/#remote-debugging (edge://inspect/... in Edge; it stays on),
# the browser then writes the DevTools address to DevToolsActivePort, and asks "Allow remote
# debugging?" on every connection.
# --------------------------------------------------------------------------------------------

# GHOST_CHROME_USER_DATA / GHOST_EDGE_USER_DATA point the agent at another profile folder (tests).
CHROME_USER_DATA_OVERRIDE = os.environ.get("GHOST_CHROME_USER_DATA")
CHROME_USER_DATA = Path(CHROME_USER_DATA_OVERRIDE or Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data")
EDGE_USER_DATA_OVERRIDE = os.environ.get("GHOST_EDGE_USER_DATA")
EDGE_USER_DATA = Path(EDGE_USER_DATA_OVERRIDE or Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Edge" / "User Data")
INSPECT_URL = "chrome://inspect/#remote-debugging"
# The first version whose inspect page has the "Allow remote debugging for this browser instance"
# switch: older ones cannot be attached to at all, so the agent does not wait for them.
MIN_ATTACH_VERSION = 144
# How the browser answers a connection the user refused (or could not be asked about): HTTP 403. A
# port or an id in the address can contain the digits too, so only a 403 on its own counts.
REFUSED_RE = re.compile(r"(?<![\w:/.-])403(?!\w)")


class UseBundled(Exception):
    """The user chose the browser on this drive while the agent waited for their Chrome or Edge."""


class BrowserGone(Exception):
    """The connection to the user's Chrome or Edge was lost in the middle of a task."""


class UserBrowser:
    """An installed browser the agent attaches to, with the user's own profile: Google Chrome
    (browser "chrome") or Microsoft Edge (browser "user_edge")."""

    def __init__(self, name, long_name, folder, exe_name, policy_key, user_data, override, inspect_url):
        self.name = name                # in messages: "Chrome"
        self.long_name = long_name      # "Google Chrome"
        self.folder = folder            # ("Google", "Chrome"): under Program Files or LOCALAPPDATA
        self.exe_name = exe_name
        self.policy_key = policy_key
        self.user_data = Path(user_data)
        self.override = override
        self.inspect_url = inspect_url

    def find(self):
        """The browser's program file, or None when it is not installed."""
        candidates = []
        if os.name == "nt":
            import winreg
            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    with winreg.OpenKey(hive, "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\" + self.exe_name) as k:
                        candidates.append(winreg.QueryValue(k, None))
                except OSError:
                    pass
        for var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            if os.environ.get(var):
                candidates.append(os.path.join(os.environ[var], *self.folder, "Application", self.exe_name))
        for cand in candidates:
            if cand and Path(cand.strip('"')).is_file():
                return Path(cand.strip('"'))
        return None

    @staticmethod
    def version(exe):
        """The installed version as a tuple of numbers, from the version folder next to the program,
        or None. After an update that waits for a restart, the newer one."""
        best = None
        try:
            for d in Path(exe).parent.iterdir():
                if d.is_dir() and re.fullmatch(r"\d+(\.\d+){3}", d.name):
                    v = tuple(int(x) for x in d.name.split("."))
                    best = v if best is None else max(best, v)
        except OSError:
            return None
        return best

    def policy_blocks(self):
        """True if an administrator turned remote debugging off (policy RemoteDebuggingAllowed)."""
        if os.name != "nt":
            return False
        import winreg
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(hive, self.policy_key) as k:
                    if winreg.QueryValueEx(k, "RemoteDebuggingAllowed")[0] == 0:
                        return True
            except OSError:
                pass
        return False

    def running(self):
        """The browser keeps the "lockfile" in its profile folder open (unreadable) while it runs."""
        lock = self.user_data / "lockfile"
        if not lock.exists():
            return False
        try:
            os.close(os.open(lock, os.O_RDONLY))
            return False
        except OSError:
            return True

    def start(self, exe):
        """Start the browser outside this process's job, so it stays open when the agent exits."""
        args = [str(exe)] + ([f"--user-data-dir={self.user_data}"] if self.override else [])
        try:
            subprocess.Popen(args, cwd=str(exe.parent), close_fds=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=CREATE_BREAKAWAY_FROM_JOB | CREATE_NEW_PROCESS_GROUP)
        except OSError:
            if self.override:
                raise
            # Some consoles run their programs in a job that forbids breaking away; Explorer starts it instead.
            subprocess.Popen(["explorer.exe", str(exe)], creationflags=CREATE_NO_WINDOW)

    def endpoint(self):
        """WebSocket address of the running browser's DevTools, or None while remote debugging is off."""
        try:
            port, path = (self.user_data / "DevToolsActivePort").read_text().split()[:2]
            port = int(port)
        except (OSError, ValueError):
            return None
        if port_is_free(port):
            return None  # left behind by a browser that crashed or had it on before
        return f"ws://127.0.0.1:{port}{path}"

    def profile_name(self):
        """The name of the profile the user used last (as on the browser's profile button), or None."""
        try:
            state = json.loads((self.user_data / "Local State").read_text(encoding="utf-8"))
            profiles = state.get("profile") or {}
            last = profiles.get("last_used") or "Default"
            name = ((profiles.get("info_cache") or {}).get(last) or {}).get("name")
            return str(name) if name else None
        except (OSError, ValueError, AttributeError):
            return None


USER_BROWSERS = {
    "chrome": UserBrowser("Chrome", "Google Chrome", ("Google", "Chrome"), "chrome.exe",
                          "SOFTWARE\\Policies\\Google\\Chrome", CHROME_USER_DATA, CHROME_USER_DATA_OVERRIDE, INSPECT_URL),
    "user_edge": UserBrowser("Edge", "Microsoft Edge", ("Microsoft", "Edge"), "msedge.exe",
                             "SOFTWARE\\Policies\\Microsoft\\Edge", EDGE_USER_DATA, EDGE_USER_DATA_OVERRIDE,
                             "edge://inspect/#remote-debugging"),
}


def attach_problem(target):
    """(program path, None) when the agent can try to attach to this browser, else (None, the reason)."""
    exe = target.find()
    if exe is None:
        return None, f"{target.long_name} is not installed on this computer."
    if target.policy_blocks():
        return None, (f"This computer's {target.name} policy (RemoteDebuggingAllowed) does not let programs use "
                      f"{target.name}.")
    version = target.version(exe)
    if version and version[0] < MIN_ATTACH_VERSION:
        return None, (f"{target.name} {'.'.join(map(str, version))} is too old for the agent to use it: version "
                      f"{MIN_ATTACH_VERSION} or newer is needed (update it from {target.name}'s menu: Help > "
                      f"About {target.long_name}).")
    return exe, None


def pick_auto_browser():
    """browser "auto": the user's Chrome when it can be used, else their Edge, else the bundled browser."""
    for mode in ("chrome", "user_edge"):
        exe, _ = attach_problem(USER_BROWSERS[mode])
        if exe is not None:
            return mode
    return "bundled"


CF_UNICODETEXT = 13


def _clipboard_dlls():
    from ctypes import wintypes
    u32 = ctypes.WinDLL("user32", use_last_error=True)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    u32.OpenClipboard.argtypes, u32.OpenClipboard.restype = [wintypes.HWND], wintypes.BOOL
    u32.CloseClipboard.argtypes, u32.CloseClipboard.restype = [], wintypes.BOOL
    u32.EmptyClipboard.argtypes, u32.EmptyClipboard.restype = [], wintypes.BOOL
    u32.IsClipboardFormatAvailable.argtypes, u32.IsClipboardFormatAvailable.restype = [wintypes.UINT], wintypes.BOOL
    u32.GetClipboardData.argtypes, u32.GetClipboardData.restype = [wintypes.UINT], wintypes.HANDLE
    u32.SetClipboardData.argtypes, u32.SetClipboardData.restype = [wintypes.UINT, wintypes.HANDLE], wintypes.HANDLE
    k32.GlobalAlloc.argtypes, k32.GlobalAlloc.restype = [wintypes.UINT, ctypes.c_size_t], wintypes.HGLOBAL
    k32.GlobalLock.argtypes, k32.GlobalLock.restype = [wintypes.HGLOBAL], ctypes.c_void_p
    k32.GlobalUnlock.argtypes, k32.GlobalUnlock.restype = [wintypes.HGLOBAL], wintypes.BOOL
    k32.GlobalFree.argtypes, k32.GlobalFree.restype = [wintypes.HGLOBAL], wintypes.HGLOBAL
    return u32, k32


def _open_clipboard(u32):
    for _ in range(10):  # another program may hold it for a moment
        if u32.OpenClipboard(None):
            return True
        time.sleep(0.05)
    return False


def get_clipboard_text():
    """The text on the clipboard, or None (no text on it, or it could not be read)."""
    if os.name != "nt":
        return None
    try:
        u32, k32 = _clipboard_dlls()
        if not u32.IsClipboardFormatAvailable(CF_UNICODETEXT) or not _open_clipboard(u32):
            return None
        try:
            handle = u32.GetClipboardData(CF_UNICODETEXT)
            if not handle:
                return None
            ptr = k32.GlobalLock(handle)
            if not ptr:
                return None
            try:
                return ctypes.wstring_at(ptr)
            finally:
                k32.GlobalUnlock(handle)
        finally:
            u32.CloseClipboard()
    except Exception:
        return None


def set_clipboard_text(text):
    """Put text on the clipboard (any language). True when it worked."""
    if os.name != "nt":
        return False
    try:
        u32, k32 = _clipboard_dlls()
        data = str(text).encode("utf-16-le") + b"\x00\x00"
        handle = k32.GlobalAlloc(0x0002, len(data))  # GMEM_MOVEABLE
        if not handle:
            return False
        ptr = k32.GlobalLock(handle)
        if not ptr:
            k32.GlobalFree(handle)
            return False
        ctypes.memmove(ptr, data, len(data))
        k32.GlobalUnlock(handle)
        if not _open_clipboard(u32):
            k32.GlobalFree(handle)
            return False
        try:
            u32.EmptyClipboard()
            if u32.SetClipboardData(CF_UNICODETEXT, handle):
                return True  # the clipboard owns the memory now
            k32.GlobalFree(handle)
            return False
        finally:
            u32.CloseClipboard()
    except Exception:
        return False


def copy_to_clipboard(text):
    if set_clipboard_text(text):
        return True
    try:
        subprocess.run(["clip"], input=text.encode("ascii"), timeout=5, creationflags=CREATE_NO_WINDOW)
        return True
    except Exception:
        return False


def lend_clipboard(text):
    """Put text on the clipboard for a moment (the chrome://inspect address during the first setup).
    Returns a function that puts back the text that was there before, unless the user has copied
    something else since; None when nothing could be copied."""
    before = get_clipboard_text()
    if not copy_to_clipboard(text):
        return None

    def give_back():
        if before is not None and get_clipboard_text() == text:
            set_clipboard_text(before)
    return give_back
