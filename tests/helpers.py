"""Shared test helpers: repo paths, in-process fake servers, headless Chromium, isolated folders.

Importing this module also isolates the test process from the user's own machine state:
  * LOCALAPPDATA points at a temporary folder (deleted at exit), so nothing reads or writes the
    real %LOCALAPPDATA%\\GhostAgent;
  * GHOST_CHROME_USER_DATA and GHOST_EDGE_USER_DATA point at empty temporary folders, so no code
    path can find the user's Chrome or Edge profile;
  * PLAYWRIGHT_BROWSERS_PATH points at the repo's bundled browsers;
  * NO_PROXY covers 127.0.0.1 and localhost (a test of proxy handling must clear it itself).
"""

import atexit
import base64
import contextlib
import http.client
import inspect
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
import zlib
from pathlib import Path

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
AGENT_DIR = REPO / "GhostAgent" / "agent"
BENCH = REPO / "tools" / "bench"
SITE = BENCH / "site"
SUITE = BENCH / "suite.json"
PYTHON = REPO / "GhostAgent" / "python" / "python.exe"
BROWSERS = REPO / "GhostAgent" / "browsers"
CREATE_NO_WINDOW = 0x08000000

for _p in (str(TESTS), str(BENCH), str(AGENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TMP = Path(tempfile.mkdtemp(prefix="ghostagent-tests-"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)
(TMP / "LocalAppData").mkdir()
os.environ["LOCALAPPDATA"] = str(TMP / "LocalAppData")
os.environ["GHOST_CHROME_USER_DATA"] = str(TMP / "no-chrome-profile")
os.environ["GHOST_EDGE_USER_DATA"] = str(TMP / "no-edge-profile")
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(BROWSERS)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

import fake_llama  # noqa: E402  (tools\bench, on sys.path above)
import bench_server  # noqa: E402


# --------------------------------------------------------------------------------------------
# Folders and environment
# --------------------------------------------------------------------------------------------

def temp_dir(test=None, prefix="t-"):
    """A new empty folder under the test run's temporary folder (removed with it, or at test cleanup)."""
    d = Path(tempfile.mkdtemp(prefix=prefix, dir=TMP))
    if test is not None:
        test.addCleanup(shutil.rmtree, d, ignore_errors=True)
    return d


@contextlib.contextmanager
def isolated_env(**extra):
    """Fresh LOCALAPPDATA, APPDATA and USERPROFILE folders (plus any extra variables) for one block.
    Yields the base folder. Everything is restored afterwards."""
    base = Path(tempfile.mkdtemp(prefix="env-", dir=TMP))
    values = {"LOCALAPPDATA": base / "LocalAppData", "APPDATA": base / "AppData", "USERPROFILE": base / "Home"}
    for p in values.values():
        p.mkdir()
    values = {k: str(v) for k, v in values.items()}
    values.update({k: str(v) for k, v in extra.items()})
    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield base
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(base, ignore_errors=True)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------------------------
# The agent module and the fake servers
# --------------------------------------------------------------------------------------------

def import_agent():
    """The repo's ghost_agent module (GhostAgent\\agent\\ghost_agent.py)."""
    import ghost_agent
    assert Path(ghost_agent.__file__).resolve() == (AGENT_DIR / "ghost_agent.py").resolve(), ghost_agent.__file__
    return ghost_agent


def call_stream_chat(base, body, on_event=None, api_key=None):
    """ghost_agent.stream_chat, passing an API key only if the function takes one."""
    agent = import_agent()
    on_event = on_event or (lambda kind, value: None)
    params = inspect.signature(agent.stream_chat).parameters
    if api_key is not None and "api_key" in params:
        return agent.stream_chat(base, body, on_event, api_key=api_key)
    return agent.stream_chat(base, body, on_event)


def start_fake_llama(test=None, script=None, **kwargs):
    """fake_llama on a free port, stopped at the end of the test when one is given."""
    fake = fake_llama.start(script=script, port=0, **kwargs)
    if test is not None:
        test.addCleanup(fake.stop)
    return fake


def start_bench_server(test=None, **kwargs):
    """bench_server on a free port, stopped at the end of the test when one is given."""
    site = bench_server.start(port=0, **kwargs)
    if test is not None:
        test.addCleanup(site.stop)
    return site


def suite_tasks(port=None):
    return bench_server.load_suite(SUITE, port)


# --------------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------------

def http_request(method, url, body=None, headers=None, timeout=30):
    """(status, headers dict, body bytes) without following redirects or using any proxy."""
    u = urllib.parse.urlsplit(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    data = body
    hdrs = dict(headers or {})
    if isinstance(body, (dict, list)):
        data = json.dumps(body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    elif isinstance(body, str):
        data = body.encode()
    try:
        conn.request(method, (u.path or "/") + (f"?{u.query}" if u.query else ""), data, hdrs)
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read()
    finally:
        conn.close()


def http_json(method, url, body=None, headers=None, timeout=30):
    status, hdrs, data = http_request(method, url, body, headers, timeout)
    try:
        return status, json.loads(data) if data else None
    except ValueError:
        return status, data.decode("utf-8", "replace")


def sse_request(base, body, api_key=None, path="/v1/chat/completions", timeout=30):
    """POST and read a whole SSE reply. Returns {status, events, done, complete, error_body, text}:
    events are the parsed JSON chunks; done is True when "data: [DONE]" arrived; complete is False
    when the chunked stream broke off (the connection was cut mid-stream)."""
    u = urllib.parse.urlsplit(base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    out = {"status": None, "events": [], "done": False, "complete": True, "error_body": None, "text": ""}
    try:
        conn.request("POST", path, json.dumps(body).encode(), headers)
        resp = conn.getresponse()
        out["status"] = resp.status
        try:
            raw = resp.read()
        except http.client.IncompleteRead as e:
            raw = e.partial
            out["complete"] = False
        text = raw.decode("utf-8", "replace")
        out["text"] = text
        if resp.status != 200:
            try:
                out["error_body"] = json.loads(text)
            except ValueError:
                out["error_body"] = text
            return out
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                out["done"] = True
            else:
                out["events"].append(json.loads(data))
        return out
    finally:
        conn.close()


# --------------------------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------------------------

def make_png(width=64, height=40, rgb=(40, 90, 160)):
    """A small solid-colour PNG, built with the standard library."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    row = b"\x00" + bytes(rgb) * width
    raw = zlib.compress(row * height, 9)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", raw) + chunk(b"IEND", b""))


def png_data_url(png):
    return "data:image/png;base64," + base64.b64encode(png).decode()


# --------------------------------------------------------------------------------------------
# Headless Chromium (the repo's bundled build; never the user's Chrome or Edge)
# --------------------------------------------------------------------------------------------

CHROMIUM_ARGS = ["--disable-gpu", "--no-first-run", "--no-default-browser-check", "--mute-audio"]


def launch_chromium(playwright, **kwargs):
    """The bundled Chromium, headless, with the GPU off. Never shows a window."""
    args = CHROMIUM_ARGS + list(kwargs.pop("args", []))
    return playwright.chromium.launch(channel="chromium", headless=True, args=args, **kwargs)


@contextlib.contextmanager
def headless_chromium(**kwargs):
    """Yields (playwright, browser); both are closed afterwards."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = launch_chromium(pw, **kwargs)
        try:
            yield pw, browser
        finally:
            browser.close()


def watch_errors(page):
    """A list that collects the page's uncaught errors and console errors as they happen."""
    errors = []
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
    return errors


def wait_until(predicate, timeout=5.0, interval=0.05):
    """Poll predicate() until it is truthy; returns its last value."""
    end = time.time() + timeout
    while True:
        value = predicate()
        if value or time.time() >= end:
            return value
        time.sleep(interval)


# --------------------------------------------------------------------------------------------
# Command-line runs (no console windows)
# --------------------------------------------------------------------------------------------

def run_python(args, timeout=120, **kwargs):
    """Run a script with the stick's Python (-I), without a console window. Returns CompletedProcess."""
    return subprocess.run([str(PYTHON), "-I"] + [str(a) for a in args], capture_output=True, text=True,
                          timeout=timeout, creationflags=CREATE_NO_WINDOW, cwd=str(REPO), **kwargs)


def start_python(args, **kwargs):
    """Start a script with the stick's Python (-I) in the background, without a console window."""
    return subprocess.Popen([str(PYTHON), "-I"] + [str(a) for a in args], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, creationflags=CREATE_NO_WINDOW,
                            cwd=str(REPO), **kwargs)


def wait_for_http(url, timeout=15.0):
    """True once url answers with any HTTP status."""
    def up():
        try:
            http_request("GET", url, timeout=2)
            return True
        except OSError:
            return False
    return wait_until(up, timeout, 0.1)
