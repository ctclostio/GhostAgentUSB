"""The user's own Chrome or Edge (browser "chrome" / "user_edge" / "auto") and the installed Chrome with
the agent's own profile ("chrome-agent").

Nothing here touches the user's own browser profiles. The attach code runs against stubs, and once
against a real Microsoft Edge that the test starts headless with a brand-new throwaway profile in a
temporary folder (remote debugging on a free port, as a debugging-enabled browser would have it).
"chrome-agent" runs headless with LOCALAPPDATA pointing at a temporary folder. Both are skipped when
the browser is not installed."""

import json
import os
import subprocess
import time
import unittest
import unittest.mock as mock
from pathlib import Path

import helpers
from agent_support import AGENT, patch_agent, quiet, reset_stop, scratch_paths, settings


def user_browser(folder, name="Chrome"):
    return AGENT.UserBrowser(name, "Google " + name, ("Google", name), "chrome.exe", "SOFTWARE\\Policies\\None",
                             folder, str(folder), "chrome://inspect/#remote-debugging")


class Keyboard:
    """msvcrt's stand-in: the keys typed so far."""

    def __init__(self, keys=""):
        self.keys = list(keys)
        self.polls = 0

    def kbhit(self):
        self.polls += 1
        return self.polls > 1 and bool(self.keys)  # typed after the first look

    def getwch(self):
        return self.keys.pop(0)


class FakeChromium:
    """playwright's chromium.connect_over_cdp, answering from a list (an exception is raised)."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.endpoints = []

    def connect_over_cdp(self, endpoint, **kw):
        self.endpoints.append(endpoint)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


class FakePW:
    def __init__(self, answers):
        self.chromium = FakeChromium(answers)


class UserBrowserTests(unittest.TestCase):
    def setUp(self):
        scratch_paths(self)
        reset_stop(self)
        self.folder = helpers.temp_dir(self)

    def test_refusals_are_told_from_other_errors(self):
        for text in ("BrowserType.connect_over_cdp: Unexpected status 403 when connecting to ws://127.0.0.1:9222/x",
                     "WebSocket error: 403 Forbidden"):
            self.assertTrue(AGENT.REFUSED_RE.search(text), text)
        for text in ("connect ECONNREFUSED ws://127.0.0.1:54031/devtools/browser/9a403b",
                     "WebSocket error: ws://127.0.0.1:4031/devtools/browser/x 404 Not Found"):
            self.assertFalse(AGENT.REFUSED_RE.search(text), text)

    def test_version_profile_and_policy(self):
        app = self.folder / "Application"
        for d in ("143.0.7499.40", "157.0.8103.51", "Dictionaries", "SetupMetrics"):
            (app / d).mkdir(parents=True)
        (app / "chrome.exe").write_bytes(b"MZ")
        self.assertEqual(AGENT.UserBrowser.version(app / "chrome.exe"), (157, 0, 8103, 51))
        self.assertIsNone(AGENT.UserBrowser.version(self.folder / "missing" / "chrome.exe"))
        data = self.folder / "User Data"
        data.mkdir()
        target = user_browser(data)
        self.assertIsNone(target.profile_name())
        (data / "Local State").write_text(json.dumps({"profile": {"last_used": "Profile 2", "info_cache": {
            "Default": {"name": "Person 1"}, "Profile 2": {"name": "Work"}}}}), encoding="utf-8")
        self.assertEqual(target.profile_name(), "Work")
        self.assertFalse(target.running())
        self.assertIsNone(target.endpoint())

    def test_attach_problems(self):
        target = user_browser(self.folder)
        with mock.patch.object(target, "find", return_value=None):
            self.assertIn("not installed", AGENT.attach_problem(target)[1])
        exe = self.folder / "Application" / "chrome.exe"
        (self.folder / "Application" / "120.0.1.2").mkdir(parents=True)
        exe.write_bytes(b"MZ")
        with mock.patch.object(target, "find", return_value=exe):
            with mock.patch.object(target, "policy_blocks", return_value=True):
                self.assertIn("RemoteDebuggingAllowed", AGENT.attach_problem(target)[1])
            exe_found, problem = AGENT.attach_problem(target)
            self.assertIsNone(exe_found)
            self.assertIn("Chrome 120.0.1.2 is too old", problem)
            (self.folder / "Application" / "150.0.1.2").mkdir()
            self.assertEqual(AGENT.attach_problem(target), (exe, None))

    def test_auto_picks_chrome_then_edge_then_bundled(self):
        answers = {}
        with patch_agent("attach_problem",
                               lambda t: answers[t.name]):
            answers.update(Chrome=(Path("c.exe"), None), Edge=(Path("e.exe"), None))
            self.assertEqual(AGENT.pick_auto_browser(), "chrome")
            answers["Chrome"] = (None, "not installed")
            self.assertEqual(AGENT.pick_auto_browser(), "user_edge")
            answers["Edge"] = (None, "too old")
            self.assertEqual(AGENT.pick_auto_browser(), "bundled")

    def started(self, mode, **over):
        """Browser.start with Playwright and the launches replaced: (browser, what was launched, output)."""
        b = AGENT.Browser(settings(browser=mode, **over))
        launched = []

        class PW:
            def start(self):
                return object()

        with mock.patch("playwright.sync_api.sync_playwright", lambda: PW()), \
                mock.patch.object(b, "_launch_own", lambda url=None: launched.append(("own", b.mode))), \
                mock.patch.object(b, "_attach_chrome", lambda url=None: launched.append(("attach", b.mode))), \
                mock.patch.object(b, "_launch_chrome_agent", lambda url=None: launched.append(("agent", b.mode))), \
                quiet() as out:
            b.start(start_url="about:blank")
        return b, launched, out.getvalue()

    def test_modes_at_start(self):
        with patch_agent("pick_auto_browser", return_value="bundled"):
            b, launched, out = self.started("auto")
        self.assertEqual(launched, [("own", "bundled")])
        self.assertIn("browser auto", out)
        with patch_agent("pick_auto_browser", return_value="user_edge"), \
                patch_agent("attach_problem", return_value=(Path("msedge.exe"), None)):
            b, launched, out = self.started("auto")
        self.assertEqual(launched, [("attach", "user_edge")])
        self.assertTrue(b.attached)
        self.assertIn("everyday Edge", b.prompt_note)
        with patch_agent("attach_problem", return_value=(None, "Google Chrome is not installed on this computer.")):
            b, launched, out = self.started("chrome")
        self.assertEqual(launched, [("own", "bundled")])
        self.assertIn("not installed on this computer. Using the browser on this drive instead.", out)
        b, launched, out = self.started("chrome-agent")
        self.assertEqual(launched, [("agent", "chrome-agent")])
        self.assertFalse(b.attached)
        self.assertEqual(b.prompt_note, "")

    def waiting_browser(self, answers, keys="", endpoints=("ws://127.0.0.1:9/a",)):
        b = AGENT.Browser(settings(browser="chrome"))
        b.pw = FakePW(answers)
        b.target = user_browser(self.folder)
        b._exe = Path("chrome.exe")
        endpoints = list(endpoints)
        starts = []
        patches = [mock.patch.object(b.target, "running", return_value=True),
                   mock.patch.object(b.target, "endpoint", lambda: endpoints[0] if len(endpoints) == 1 else endpoints.pop(0)),
                   mock.patch.object(b.target, "start", lambda exe: starts.append(exe)),
                   patch_agent("_keyboard", return_value=Keyboard(keys) if keys is not None else None),
                   patch_agent("lend_clipboard", return_value=None),
                   mock.patch.object(AGENT.time, "sleep", lambda s: None)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return b, starts

    def test_b_switches_to_the_bundled_browser(self):
        b, starts = self.waiting_browser([], keys="xb", endpoints=(None,))
        with quiet() as out, self.assertRaises(AGENT.UseBundled):
            b._wait_for_user_browser()
        self.assertIn("press B to use the browser on this drive", out.getvalue())
        launched = []
        b, starts = self.waiting_browser([], keys="b", endpoints=(None,))
        with mock.patch.object(b, "_launch_own", lambda url=None: launched.append(url)), quiet() as out:
            b._attach_or_bundled("about:blank")
        self.assertEqual((b.mode, b.target, launched), ("bundled", None, ["about:blank"]))
        self.assertIsNone(b.quit_note)

    def test_quick_refusals_open_a_window_and_repeat_the_hint(self):
        refused = Exception("Unexpected status 403 when connecting")
        connected = object()
        b, starts = self.waiting_browser([refused] * 6 + [connected])
        with quiet() as out:
            b._wait_for_user_browser()
        self.assertIs(b.browser, connected)
        text = out.getvalue()
        self.assertEqual(text.count("needs an open window to ask in"), 2)  # after 3 and after 6
        self.assertEqual(len(starts), 1)  # one window opened, not one per hint
        self.assertIn('Chrome asks "Allow remote debugging?"', text)

    def test_other_errors_are_shown_once_and_a_new_address_is_tried(self):
        stale = Exception("connect ECONNREFUSED 127.0.0.1:9\nCall log: ...")
        connected = object()
        b, starts = self.waiting_browser([stale, connected], endpoints=("ws://a", "ws://a", "ws://b"))
        with quiet() as out:
            b._wait_for_user_browser()
        self.assertEqual(b.pw.chromium.endpoints, ["ws://a", "ws://b"])
        self.assertEqual(out.getvalue().count("Could not connect to Chrome: connect ECONNREFUSED 127.0.0.1:9"), 1)
        self.assertNotIn("Call log", out.getvalue())

    def test_cancel_can_switch_to_the_bundled_browser(self):
        cancelled = Exception("Unexpected status 403 when connecting")
        b, starts = self.waiting_browser([cancelled], keys=None)
        times = [0.0, 5.0]  # Cancel came after 5 seconds: a person clicked it

        def clock():
            return times.pop(0) if times else 10.0

        with mock.patch.object(AGENT.time, "time", clock), \
                mock.patch("builtins.input", return_value="b"), quiet() as out, self.assertRaises(AGENT.UseBundled):
            b._wait_for_user_browser()
        self.assertIn("Cancel was clicked", out.getvalue())

    def test_a_lost_connection_ends_the_task_and_reconnects_between_tasks(self):
        b = AGENT.Browser(settings(browser="user_edge"))
        b.target = AGENT.USER_BROWSERS["user_edge"]

        class Gone:
            def is_connected(self):
                return False

        b.browser = Gone()
        with self.assertRaises(AGENT.BrowserGone) as cm:
            b._live_page()
        self.assertEqual(str(cm.exception), "Edge was closed; start the next task to reconnect.")
        calls = []
        with mock.patch.object(b, "_attach_chrome", lambda url=None: calls.append(url)), quiet() as out:
            b.ensure_connected()
        self.assertEqual(calls, [None])
        self.assertIn("The connection to Edge was lost", out.getvalue())
        b._was_attached = True
        self.assertIn("edge://inspect/#remote-debugging", b.quit_note)

    def test_chrome_agent_falls_back_without_chrome(self):
        b = AGENT.Browser(settings(browser="chrome-agent"))
        launched = []
        with mock.patch.object(AGENT.USER_BROWSERS["chrome"], "find", return_value=None), \
                mock.patch.object(b, "_launch_own", lambda url=None: launched.append(url)), quiet() as out:
            b._launch_chrome_agent("about:blank")
        self.assertEqual((b.mode, launched), ("bundled", ["about:blank"]))
        self.assertIn("Google Chrome is not installed", out.getvalue())


EDGE = AGENT.USER_BROWSERS["user_edge"].find()
CHROME = AGENT.USER_BROWSERS["chrome"].find()


class RealBrowserTests(unittest.TestCase):
    """Real browsers, headless, each with a brand-new profile in a temporary folder."""

    def setUp(self):
        scratch_paths(self)
        reset_stop(self)
        self.local = helpers.temp_dir(self)
        patcher = mock.patch.dict(os.environ, {"LOCALAPPDATA": str(self.local)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.site = helpers.start_bench_server(self)

    @unittest.skipUnless(EDGE, "Microsoft Edge is not installed")
    def test_attach_to_a_running_edge(self):
        version = AGENT.UserBrowser.version(EDGE)
        if version and version[0] < AGENT.MIN_ATTACH_VERSION:
            self.skipTest("Edge is older than the agent supports")
        profile = helpers.temp_dir(self) / "edge-profile"
        profile.mkdir()
        edge = subprocess.Popen(
            [str(EDGE), "--headless=new", f"--user-data-dir={profile}", "--remote-debugging-port=0", "--no-first-run",
             "--no-default-browser-check", "--disable-gpu", "--mute-audio", "--disable-background-networking",
             "--disable-component-update", "--disable-sync", "--disable-default-apps", "--disable-extensions",
             "--no-pings", "about:blank"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=helpers.CREATE_NO_WINDOW)

        class ThrowawayEdge(AGENT.UserBrowser):
            def start(self, exe):
                raise AssertionError("the test would open a visible Edge window")

        target = ThrowawayEdge("Edge", "Microsoft Edge", ("Microsoft", "Edge"), "msedge.exe",
                               "SOFTWARE\\Policies\\Microsoft\\Edge", profile, str(profile),
                               "edge://inspect/#remote-debugging")
        b = AGENT.Browser(settings(browser="user_edge", start_url=self.site.base + "/"))

        def end_edge():
            # msedge.exe hands over to another process and exits, so the browser is closed through
            # DevTools; anything of this throwaway profile still running after that is ended by name.
            endpoint = target.endpoint()
            if endpoint and b.pw is not None:
                try:
                    other = b.pw.chromium.connect_over_cdp(endpoint, timeout=10000)
                    other.new_browser_cdp_session().send("Browser.close")
                except Exception:
                    pass
            helpers.wait_until(lambda: not target.running(), timeout=20)
            # Whatever of this throwaway profile still runs is ended by its profile path, also an Edge that
            # never got as far as its lockfile (only processes started with this temporary folder match).
            subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                            "Get-CimInstance Win32_Process -Filter \"Name='msedge.exe'\" | Where-Object { "
                            f"$_.CommandLine -like '*{profile}*' }} | ForEach-Object {{ Stop-Process -Id "
                            "$_.ProcessId -Force }"], capture_output=True, timeout=60,
                           creationflags=helpers.CREATE_NO_WINDOW)
            edge.wait(20)

        self.addCleanup(b.close)
        self.addCleanup(end_edge)  # runs first, while Playwright is still there to close Edge with
        if not helpers.wait_until(lambda: (profile / "DevToolsActivePort").exists(), timeout=30):
            # The installed Edge, not the agent: it sometimes does not start headless at all (for example
            # while it is updating itself). The attach code is covered by the stub tests either way.
            self.skipTest("the installed Edge did not start headless with a DevTools port within 30 s")
        with mock.patch.dict(AGENT.USER_BROWSERS, {"user_edge": target}), quiet() as out:
            b.start()
        self.assertIn("connected to your Edge", out.getvalue())
        self.assertTrue(b.attached and b._connected())
        self.assertEqual(len(b._tabs()), 1)  # only the agent's own window counts
        obs = b.observation()
        self.assertTrue(any(part.get("type") == "image_url" for part in obs["content"]))
        self.assertTrue(b.last_url.startswith(self.site.base))
        self.assertIn("edge://inspect/#remote-debugging", b.quit_note)
        end_edge()
        self.assertTrue(helpers.wait_until(lambda: not b._connected(check=True), timeout=20, interval=0.2))
        with self.assertRaises(AGENT.BrowserGone):
            b._live_page()

    @unittest.skipUnless(CHROME, "Google Chrome is not installed")
    def test_chrome_with_the_agents_own_profile(self):
        b = AGENT.Browser(settings(browser="chrome-agent"))
        self.addCleanup(b.close)
        from playwright.sync_api import sync_playwright
        b.pw = sync_playwright().start()
        opts = b.chrome_agent_options(headless=True)
        self.assertEqual(Path(opts["user_data_dir"]), self.local / "GhostAgent" / "chrome-profile")
        self.assertEqual(opts["channel"], "chrome")
        self.assertIn("--disable-extensions", opts["ignore_default_args"])  # extensions work in it
        self.assertTrue(b.chrome_agent_options()["ignore_default_args"][:4] == AGENT.UNSAFE_DEFAULT_SWITCHES)
        b.ctx = b._launch(opts)
        b._use_context(b._start_page(self.site.base + "/"))
        obs = b.observation()
        self.assertTrue(any(part.get("type") == "image_url" for part in obs["content"]))
        self.assertTrue((self.local / "GhostAgent" / "chrome-profile").is_dir())
        self.assertIn(("the agent's Chrome profile on this PC (browser chrome-agent)",
                       self.local / "GhostAgent" / "chrome-profile"), AGENT.pc_leftovers())


if __name__ == "__main__":
    unittest.main()
