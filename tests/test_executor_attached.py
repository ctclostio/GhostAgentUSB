"""Batch D1 when the agent works in the user's own Chrome (browser "chrome"): download events from the
browser, file pickers, new tabs and their titles, and the window size. The "user's Chrome" here is
the bundled Chromium started headless with a brand-new throwaway profile, whose downloads go to a
temporary folder; the user's own browsers and profiles are never touched."""

import json
import subprocess
import unittest
import unittest.mock as mock
from pathlib import Path

import helpers
from agent_support import AGENT, quiet, reset_stop, scratch_paths, settings

CHROMIUM = next(iter(sorted(helpers.BROWSERS.glob("chromium-*/chrome-win64/chrome.exe"))), None)


@unittest.skipUnless(CHROMIUM, "the bundled Chromium is missing")
class AttachedBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.site = helpers.start_bench_server()

    @classmethod
    def tearDownClass(cls):
        cls.site.stop()

    def setUp(self):
        reset_stop(self)
        scratch_paths(self)

    def attach(self, path):
        """The agent's Browser attached to a throwaway headless Chromium standing in for the user's Chrome,
        with a task begun at a site page. Returns (browser, the folder its downloads go to)."""
        tmp = helpers.temp_dir(self)
        profile, downloads = tmp / "profile", tmp / "downloads"
        downloads.mkdir()
        (profile / "Default").mkdir(parents=True)
        # Chromium's own download folder would be the user's Downloads: this profile uses a temporary one.
        (profile / "Default" / "Preferences").write_text(json.dumps({
            "download": {"default_directory": str(downloads), "prompt_for_download": False, "directory_upgrade": True},
            "savefile": {"default_directory": str(downloads)}}), encoding="utf-8")
        proc = subprocess.Popen(
            [str(CHROMIUM), "--headless=new", f"--user-data-dir={profile}", "--remote-debugging-port=0",
             "--no-first-run", "--no-default-browser-check", "--disable-gpu", "--mute-audio",
             "--disable-background-networking", "--disable-component-update", "--disable-sync", "about:blank"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=helpers.CREATE_NO_WINDOW)

        class ThrowawayChrome(AGENT.UserBrowser):
            def find(self):
                return CHROMIUM

            def policy_blocks(self):
                return False

            def start(self, exe):
                raise AssertionError("the test would open a visible browser window")

        target = ThrowawayChrome("Chrome", "Google Chrome", ("Google", "Chrome"), "chrome.exe",
                                 "SOFTWARE\\Policies\\Google\\Chrome", profile, str(profile), AGENT.INSPECT_URL)
        b = AGENT.Browser(settings(browser="chrome", start_url=self.site.url(path), settle_seconds=0.2))

        def end_chromium():
            endpoint = target.endpoint()
            if endpoint and b.pw is not None and not b.playwright_dead():
                try:
                    other = b.pw.chromium.connect_over_cdp(endpoint, timeout=10000)
                    other.new_browser_cdp_session().send("Browser.close")
                except Exception:
                    pass
            try:
                proc.wait(20)
            except subprocess.TimeoutExpired:
                proc.kill()

        self.addCleanup(b.close)
        self.addCleanup(end_chromium)  # runs first, while Playwright is still there
        if not helpers.wait_until(lambda: (profile / "DevToolsActivePort").exists(), timeout=30):
            self.skipTest("the bundled Chromium did not start headless with a DevTools port")
        with mock.patch.dict(AGENT.USER_BROWSERS, {"chrome": target}), quiet():
            b.start()
        self.assertTrue(b.attached)
        b.begin_task("test")
        return b, downloads

    def act(self, b, name, **args):
        with quiet():
            return b.execute(name, args)

    def at(self, b, selector):
        box = b.page.locator(selector).first.bounding_box()
        return {"x": round((box["x"] + box["width"] / 2) / b.w * 1000),
                "y": round((box["y"] + box["height"] / 2) / b.h * 1000)}

    def test_downloads_and_file_pickers(self):
        b, downloads = self.attach("/fixtures/download.html")
        with quiet():
            b.observation()  # the page area's size, for the coordinates
        said = self.act(b, "click_web", element="small download", **self.at(b, "#small"))
        self.assertIn("A download started: ghostagent-small.txt. Chrome saves it as it saves your downloads; "
                      "GhostAgent never opens downloaded files.", said)
        if "finished" not in said:
            said += self.act(b, "wait_web", seconds=1)
        self.assertIn("The download of ghostagent-small.txt finished.", said)
        self.assertTrue(helpers.wait_until(lambda: (downloads / "ghostagent-small.txt").exists(), timeout=5))

        self.act(b, "go_to_web", url=self.site.url("/fixtures/upload.html"))
        said = self.act(b, "click_web", element="Browse", **self.at(b, "#pick"))
        self.assertIn("A file picker opened; uploading local files is not supported.", said)
        b.end_task()
        self.assertEqual(b._choosers, set())

    def test_tabs_titles_and_the_window(self):
        b, _ = self.attach("/fixtures/popups.html?timer=off&later=2000")
        first = b.page
        with quiet():
            b.observation()
        said = self.act(b, "click_web", element="Open help", **self.at(b, "#open"))
        self.assertIn("A new tab opened and is now the current tab.", said)
        self.assertIn("from=click", b.page.url)
        with quiet():
            obs = b.observation()
        tabs = obs["content"][-1]["text"]
        self.assertIn("<untrusted_page_text>Popups</untrusted_page_text>", tabs)  # titles from the browser
        self.assertIn("<untrusted_page_text>Popup page (click)</untrusted_page_text>", tabs)

        self.act(b, "switch_tab_web", tab_index=0)
        self.assertIs(b.page, first)
        said = self.act(b, "click_web", element="Open a tab later", **self.at(b, "#later"))
        said += self.act(b, "wait_web", seconds=3)
        self.assertIn("The page opened a new tab by itself; it was not switched to", said)
        self.assertIs(b.page, first)
        self.assertEqual(len(b._tabs()), 3)
        self.assertTrue(b.refit_window())
        with mock.patch.dict(b.s, {"refit_window_each_task": False}):
            self.assertFalse(b.refit_window())


if __name__ == "__main__":
    unittest.main()
