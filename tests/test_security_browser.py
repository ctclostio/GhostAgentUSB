"""The bundled browser runs with Chromium's sandbox and with the protections Playwright normally turns
off (the pop-up blocker, phishing detection, HTTPS upgrades). When the sandbox cannot start, Edge takes
over, or the bundled browser starts without it only if the user allowed that.

The real browser here is the headless test one; the windowed start is checked with a stand-in for
Playwright, so nothing ever shows on screen."""

import os
import re
import unittest
import unittest.mock as mock
from pathlib import Path

import helpers
from agent_support import AGENT, quiet, scratch_paths, settings
from security_support import SiteBrowserCase


class SandboxTests(SiteBrowserCase):
    def test_the_renderers_run_in_the_sandbox(self):
        b = self.start("/fixtures/long.html")
        b.page.goto("chrome://sandbox")
        b.page.wait_for_function("document.body && document.body.innerText.includes('Untrusted')", timeout=15000)
        text = b.page.inner_text("body")
        self.assertRegex(text, r"Renderer[\s\S]*Lockdown[\s\S]*Untrusted")

    def test_the_command_line_keeps_chrome_protections(self):
        b = self.start("/fixtures/long.html")
        b.page.goto("chrome://version")
        cmd = b.page.inner_text("#command_line")
        for switch in ("--no-sandbox", "--disable-popup-blocking", "--disable-client-side-phishing-detection"):
            self.assertNotIn(switch, cmd)
        switches = AGENT.playwright_feature_switches()
        if switches:
            disabled = re.findall(r"--disable-features=(\S+)", cmd)
            enabled = re.findall(r"--enable-features=(\S+)", cmd)
            self.assertNotIn("HttpsUpgrades", disabled[-1].split(","))  # Chromium reads the last copy only
            self.assertIn("HttpsUpgrades", enabled[-1].split(","))

    def test_pop_ups_need_a_click(self):
        b = self.start("/fixtures/popups.html?timer=200")
        b.page.wait_for_timeout(1200)
        self.assertEqual(len(b._tabs()), 1)  # the timer's window.open was blocked
        result = self.click(b, "#open", "Open help in a new tab")
        self.assertEqual(len(b._tabs()), 2)
        self.assertIn("A new tab opened", result)


class FeatureSwitchTests(unittest.TestCase):
    def test_switches_from_the_installed_playwright(self):
        switches = AGENT.playwright_feature_switches()
        self.assertEqual(len(switches), 2)
        disable, enable = switches
        self.assertTrue(disable.startswith("--disable-features="))
        self.assertTrue(enable.startswith("--enable-features="))
        disabled, enabled = disable.split("=", 1)[1].split(","), enable.split("=", 1)[1].split(",")
        self.assertNotIn("HttpsUpgrades", disabled)
        self.assertIn("HttpsUpgrades", enabled)
        self.assertGreater(len(disabled), 3)  # Playwright's other choices are kept
        self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9_]+", f) for f in disabled + enabled))

    def test_an_unexpected_playwright_leaves_its_defaults_alone(self):
        with mock.patch.object(Path, "read_text", return_value="something else entirely"):
            self.assertEqual(AGENT.playwright_feature_switches(), [])
        with mock.patch.object(Path, "read_text", side_effect=OSError("gone")):
            self.assertEqual(AGENT.playwright_feature_switches(), [])


class FakePage:
    def __init__(self):
        self.url = "about:blank"

    def on(self, *args):
        pass

    def goto(self, url, **kwargs):
        self.url = url

    def is_closed(self):
        return False


class FakeContext:
    def __init__(self):
        self.pages = [FakePage()]

    def on(self, *args):
        pass

    def new_page(self):
        self.pages.append(FakePage())
        return self.pages[-1]

    def close(self):
        pass


class FakePlaywright:
    """Records every launch; each entry of `failures` is the error for one launch (None = it works)."""

    def __init__(self, *failures):
        self.failures = list(failures)
        self.calls = []
        self.chromium = self

    def launch_persistent_context(self, **kwargs):
        self.calls.append(kwargs)
        failure = self.failures.pop(0) if self.failures else None
        if failure:
            raise Exception(failure)
        return FakeContext()

    def stop(self):
        pass


class WindowedLaunchTests(unittest.TestCase):
    def setUp(self):
        scratch_paths(self)
        self.temp = helpers.temp_dir(self)
        patcher = mock.patch.dict(os.environ, {"TEMP": str(self.temp), "TMP": str(self.temp)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def launch(self, *failures, **over):
        b = AGENT.Browser(settings(**{"browser": "bundled", **over}))
        self.addCleanup(b.close)
        b.pw = FakePlaywright(*failures)
        with quiet() as out:
            b._launch_own("https://example.com/")
        return b, b.pw.calls, out.getvalue()

    def test_sandboxed_with_the_protections_on(self):
        cache = AGENT.browser_cache_dir()
        (cache / "left-over").mkdir(parents=True)  # from a window closed with its X button
        b, calls, out = self.launch()
        self.assertEqual(len(calls), 1)
        kw = calls[0]
        self.assertIs(kw["chromium_sandbox"], True)
        self.assertNotIn("channel", kw)
        self.assertEqual(kw["user_data_dir"], str(AGENT.PROFILE))
        self.assertEqual(kw["ignore_default_args"], AGENT.UNSAFE_DEFAULT_SWITCHES)
        self.assertIn(f"--disk-cache-dir={cache}", kw["args"])
        self.assertTrue(cache.is_relative_to(self.temp))
        self.assertFalse((cache / "left-over").exists())
        for switch in AGENT.playwright_feature_switches():
            self.assertIn(switch, kw["args"])
        self.assertEqual(b.page.url, "https://example.com/")
        cache.mkdir(parents=True, exist_ok=True)  # what the browser writes while it runs
        b.close()
        self.assertFalse(cache.exists())

    def test_edge_takes_over_when_the_sandbox_cannot_start(self):
        b, calls, out = self.launch("Browser closed: sandbox initialisation failed")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["channel"], "msedge")
        self.assertIs(calls[1]["chromium_sandbox"], True)
        self.assertEqual(calls[1]["user_data_dir"], str(AGENT.PROFILE) + "-edge")
        self.assertIn("trying Microsoft Edge instead", out)
        self.assertIn("allow_unsandboxed_browser", out)

    def test_without_the_sandbox_only_when_allowed(self):
        b, calls, out = self.launch("sandbox initialisation failed", allow_unsandboxed_browser=True)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("channel", calls[1])
        self.assertIs(calls[1]["chromium_sandbox"], False)
        self.assertIn("starts without it", out)

    def test_when_nothing_starts(self):
        with self.assertRaises(SystemExit) as cm:
            self.launch("sandbox initialisation failed", "msedge is not installed")
        self.assertIn("Microsoft Edge did not start either", str(cm.exception))

    def test_a_profile_in_use_is_not_retried(self):
        b = AGENT.Browser(settings(browser="bundled"))
        self.addCleanup(b.close)
        b.pw = FakePlaywright("Failed to create a ProcessSingleton: the profile is already in use")
        with quiet(), self.assertRaises(SystemExit) as cm:
            b._launch_own("https://example.com/")
        self.assertIn("profile is in use", str(cm.exception))
        self.assertEqual(len(b.pw.calls), 1)

    def test_edge_mode(self):
        b, calls, out = self.launch(browser="edge")
        self.assertEqual([c.get("channel") for c in calls], ["msedge"])
        self.assertIs(calls[0]["chromium_sandbox"], True)


if __name__ == "__main__":
    unittest.main()
