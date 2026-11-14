"""Shared pieces for the agent's own tests (test_agent_*.py): a stand-in browser for the agent loop,
scratch folders for everything the agent writes, quiet output, and running the agent's command line.
Nothing here needs a GPU, a model file or a network beyond 127.0.0.1."""

import contextlib
import copy
import io
import json
import subprocess
import time
import unittest.mock as mock
from pathlib import Path

import helpers

AGENT = helpers.import_agent()
AGENT_PY = helpers.AGENT_DIR / "ghost_agent.py"
TINY_PNG = helpers.make_png(64, 40)


def settings(**over):
    """The default settings with some changed (no settings file is read)."""
    s = copy.deepcopy(AGENT.DEFAULT_SETTINGS)
    s.update(over)
    return s


def scratch_paths(test, folder=None):
    """Point everything the agent writes (logs, downloads, browser profile, settings.json) at a
    scratch folder for the length of one test. Returns the folder."""
    folder = Path(folder or helpers.temp_dir(test))
    for name, value in (("LOGS", folder / "logs"), ("DOWNLOADS", folder / "downloads"),
                        ("PROFILE", folder / "browser-profile"), ("SETTINGS_FILE", folder / "settings.json"),
                        # The test runner's console is the user's: never retitle it or bring it forward.
                        ("set_console_title", lambda text: None), ("focus_console", lambda: None)):
        patcher = mock.patch.object(AGENT, name, value)
        patcher.start()
        test.addCleanup(patcher.stop)
    return folder


@contextlib.contextmanager
def quiet():
    """Swallow what the agent prints; yields the buffer, so a test can still read it."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        yield out


def reset_stop(test):
    """Make sure STOP is clear before and after a test that touches Ctrl+C handling."""
    AGENT.STOP.clear()
    test.addCleanup(AGENT.STOP.clear)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def steps_file(folder):
    files = sorted((Path(folder) / "logs").glob("steps-*.jsonl"))
    return files[-1] if files else None


class StubBrowser:
    """Just enough of ghost_agent.Browser for Agent.run: every observation is a small PNG of a
    fixed page, every action is recorded and answered with 'Executed.'. navigate_on: {tool name:
    address} the page moves to when that tool runs; refuse_on: tools answered like a refusal (not
    done: last_ok False); results: {tool name: text} returned instead of 'Executed.'."""

    prompt_note = ""

    def __init__(self, url="https://example.test/search?q=secret+words#frag", execute_delay=0.0, fail_on=None,
                 navigate_on=None, refuse_on=None, results=None):
        self.url = url
        self.calls = []
        self.execute_delay = execute_delay
        self.fail_on = fail_on or set()
        self.navigate_on = navigate_on or {}
        self.refuse_on = refuse_on or set()
        self.results = results or {}
        self.last_shot = {"png_bytes": None, "png_w": None, "png_h": None}
        self.last_url = None
        self.last_timings = {"execute": None, "settle": None}
        self.last_ok = True
        self.overlays = []

    def location(self):
        return 0, self.url

    def observation(self, bring_to_front=False):
        self.last_shot = {"png_bytes": len(TINY_PNG), "png_w": 64, "png_h": 40}
        self.last_url = self.url
        return {"role": "user", "content": [
            {"type": "text", "text": "<observation>\n"},
            {"type": "image_url", "image_url": {"url": helpers.png_data_url(TINY_PNG)}},
            {"type": "text", "text": f"\nURL: {self.url}\nOpen tabs:\n  [0] Page - {self.url} (current)\n</observation>"},
        ]}

    def overlay(self, kind, title="", text=""):
        self.overlays.append(kind)

    def execute(self, name, args):
        self.calls.append((name, args))
        self.last_timings = {"execute": None, "settle": None}
        self.last_ok = True
        t0 = time.perf_counter()
        if name in self.fail_on:
            self.last_timings["execute"] = round(time.perf_counter() - t0, 3)
            raise RuntimeError(f"{name} failed on purpose\nsecond line")
        if self.execute_delay:
            time.sleep(self.execute_delay)
        self.last_timings = {"execute": round(time.perf_counter() - t0, 3), "settle": 0.01}
        if name in self.refuse_on:
            self.last_ok = False
            return "The user did not allow this action."
        if name in self.navigate_on:
            self.url = self.navigate_on[name]
        if name in self.results:
            return self.results[name]
        return f"Executed. Current URL: {self.url}"


class StubServer:
    def __init__(self, base):
        self.base = base
        self.api_key = None

    def alive(self):
        return True


def make_agent(test, script=None, browser=None, fake=None, **setting_over):
    """(agent, fake_llama, scratch folder): the real Agent loop against fake_llama and a stub browser.
    The fake checks every request strictly (each tool call answered by a tool message with its id
    before the conversation goes on), so an agent that breaks the pairing fails the request."""
    folder = scratch_paths(test)
    fake = fake or helpers.start_fake_llama(test, script=script, validate="strict")
    agent = AGENT.Agent(settings(**setting_over), StubServer(fake.base), browser or StubBrowser())
    return agent, fake, folder


def pairing_problems(messages):
    """What is wrong with a message list's tool-call pairing, as a list of sentences (empty = fine):
    the system prompt comes first and only there; each assistant tool call has a unique id and its
    arguments are a JSON object; the calls are answered, each exactly once, by the tool messages that
    directly follow; no tool message answers anything else; and the list does not end with calls
    still open."""
    problems = []
    open_ids = None
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "system" and i != 0:
            problems.append(f"message {i}: a system message that is not first")
        if role == "tool":
            if open_ids is None or m.get("tool_call_id") not in open_ids:
                problems.append(f"message {i}: tool message for {m.get('tool_call_id')!r}, which no open call has")
            else:
                open_ids.discard(m.get("tool_call_id"))
            continue
        if open_ids:
            problems.append(f"message {i}: calls {sorted(open_ids)} were not answered before a {role} message")
        open_ids = None
        if role == "assistant" and m.get("tool_calls"):
            ids = [tc.get("id") for tc in m["tool_calls"]]
            if len(set(ids)) != len(ids) or not all(ids):
                problems.append(f"message {i}: tool call ids missing or repeated: {ids}")
            for tc in m["tool_calls"]:
                try:
                    args = json.loads((tc.get("function") or {}).get("arguments") or "{}")
                except ValueError:
                    args = None
                if not isinstance(args, dict):
                    problems.append(f"message {i}: arguments of {tc.get('id')!r} are not a JSON object")
            open_ids = set(ids)
    if open_ids:
        problems.append(f"the list ends with calls {sorted(open_ids)} unanswered")
    return problems


def assert_valid_pairing(test, messages, where=""):
    problems = pairing_problems(messages)
    test.assertEqual(problems, [], f"tool-call pairing broken{' in ' + where if where else ''}")


def assert_requests_valid(test, fake):
    """Every chat request the fake received keeps a valid pairing, and none was refused."""
    chats = fake.chat_requests()
    test.assertTrue(chats, "no chat request was made")
    for n, r in enumerate(chats, 1):
        assert_valid_pairing(test, r["body"]["messages"], f"request {n}")
        test.assertEqual(r["body"]["messages"][-1]["role"], "user", f"request {n} must end with the user's turn")


def call(name, **args):
    return {"name": name, "arguments": args}


def run_cli(args, timeout=180, **kwargs):
    """Run ghost_agent.py with the stick's Python, without a console window. Returns CompletedProcess."""
    return helpers.run_python([AGENT_PY] + [str(a) for a in args], timeout=timeout, encoding="utf-8",
                              errors="replace", **kwargs)


def start_cli(args):
    return subprocess.Popen([str(helpers.PYTHON), "-I", str(AGENT_PY)] + [str(a) for a in args],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
                            creationflags=helpers.CREATE_NO_WINDOW, cwd=str(helpers.REPO))


def element_centre(site, path, selector, width=1280, height=800):
    """Where an element's centre is on a site page, in the model's 0-1000 coordinates."""
    with helpers.headless_chromium() as (pw, browser):
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(site.url(path))
        box = page.locator(selector).first.bounding_box()
        page.close()
    return (round((box["x"] + box["width"] / 2) / width * 1000),
            round((box["y"] + box["height"] / 2) / height * 1000))
