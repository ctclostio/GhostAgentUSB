"""Privacy: what pages, the console and the PC get to see. The status box in the page says nothing
about the task, pasting the clipboard is refused, escape sequences from pages or the model never
reach the console, and what GhostAgent keeps on a PC is limited, pruned and removable (/forget-pc).

Nothing here touches the real clipboard, the real %LOCALAPPDATA%\\GhostAgent or the user's TEMP."""

import builtins
import json
import os
import time
import unittest
import unittest.mock as mock
from pathlib import Path

import helpers
from agent_support import AGENT, StubBrowser, call, quiet, reset_stop, scratch_paths, settings
from security_support import SiteBrowserCase

EVIL = "\x1b]0;evil title\x07"


class PageSeesNothingTests(SiteBrowserCase):
    def test_the_page_never_sees_the_task_or_the_answer(self):
        b = self.start("/fixtures/mutation-log.html")
        fake = helpers.start_fake_llama(self, script=[
            {"tool_calls": [call("wait_web", seconds=0.6)]},
            {"tool_calls": [call("answer", content="The code word is PELICAN-77.")]}])
        agent = AGENT.Agent(settings(settle_seconds=0.1), AGENT.ExternalServer(fake.base), b)
        self.site.log_entries.clear()
        with quiet():
            self.assertIn("PELICAN-77", agent.run("Find the code word FLAMINGO-42 on this page."))
            b.overlay("answer", AGENT.ANSWER_OVERLAY)
        b.page.wait_for_timeout(700)  # the page reports in batches every 250 ms
        logged = json.dumps(self.site.entries("log"))
        self.assertIn("GhostAgent - step 1", logged)  # the page does see the status box ...
        self.assertIn("GhostAgent finished", logged)
        self.assertNotIn("FLAMINGO", logged)  # ... but nothing of the task or the answer
        self.assertNotIn("PELICAN", logged)

    def test_paste_chords_are_refused(self):
        b = self.start("/fixtures/keys.html")
        self.click(b, "#target", "Target field")
        b.page.evaluate("window.keys = []")
        for keys in ("ctrl+V", "control + v", "shift+insert", "Shift + Insert", "Meta+V", "Control+Shift+V", "CTRL+v",
                     "a ctrl+v"):
            self.assertEqual(self.act(b, "press_keys_web", keys=keys), AGENT.PASTE_REFUSAL, keys)
        self.assertEqual(b.page.evaluate("window.keys"), [])  # not even the 'a' before it
        result = self.act(b, "press_keys_web", keys="Shift+V ctrl+a")
        self.assertTrue(result.startswith("Executed."), result)
        self.assertEqual(b.page.input_value("#target").lower(), "v")  # typed, not refused

    def test_screenshots_are_kept_on_the_pc_not_the_drive(self):
        b = self.start("/fixtures/long.html", save_screenshots=True)
        with helpers.isolated_env() as base, quiet():
            b.observation()
            shots = list(AGENT.screenshots_dir().glob("*.png"))
            self.assertEqual(len(shots), 1)
            self.assertTrue(shots[0].is_relative_to(base / "LocalAppData" / "GhostAgent" / "screenshots"))
        self.assertFalse((self.folder / "logs" / "screenshots").exists())


class PasteKeyTests(unittest.TestCase):
    def test_which_chords_paste(self):
        for keys in ("ctrl+V", "control + v", "shift+insert", "Meta+V", "Control+Shift+V", "cmd+v", "Alt+Ctrl+V"):
            self.assertTrue(any(AGENT.is_paste(k) for k in AGENT.key_presses(keys)), keys)
        for keys in ("ctrl+c", "ctrl+a", "Shift+V", "v", "Control+Insert", "Enter", "Insert", "alt+v"):
            self.assertFalse(any(AGENT.is_paste(k) for k in AGENT.key_presses(keys)), keys)
        self.assertEqual(AGENT.key_presses("control + v  Enter"), ["Control+v", "Enter"])


class ConsoleTextTests(unittest.TestCase):
    def setUp(self):
        reset_stop(self)
        scratch_paths(self)

    def test_clean(self):
        self.assertEqual(AGENT.clean(EVIL), "")
        self.assertEqual(AGENT.clean("a\x1b[2Jb\x1b[1;31mc\x1b[0m"), "abc")
        self.assertEqual(AGENT.clean("\x1b]8;;http://x.example\x1b\\link\x1b]8;;\x1b\\"), "link")
        self.assertEqual(AGENT.clean("\x9b2Jx\x9d0;t\x9cy"), "xy")
        self.assertEqual(AGENT.clean("abc\u202edcba\u2066x\u2069"), "abcdcbax")
        self.assertEqual(AGENT.clean("bell\x07 back\x08 cr\r end"), "bell back cr end")
        self.assertEqual(AGENT.clean("tab\tnew\nline"), "tab\tnew\nline")
        self.assertEqual(AGENT.clean(None), "None")

    def test_messages_are_cleaned(self):
        with quiet() as out:
            AGENT.info("page " + EVIL + "title")
            AGENT.warn("x\x1b[2Ky")
            AGENT.err("\x1b]52;c;ZXZpbA==\x07z")
        text = out.getvalue()
        self.assertNotIn("\x07", text)
        self.assertNotIn("\x1b]", text)
        self.assertNotIn("\x1b[2K", text)
        self.assertIn("page title", text)

    def test_the_answer_cannot_drive_the_console(self):
        fake = helpers.start_fake_llama(self, script=[
            {"tool_calls": [call("answer", content="Result " + EVIL + "ok \x1b[2J\u202edone")]}])
        browser = StubBrowser()
        browser.mode = "bundled"
        browser.start = lambda start_url=None: None
        browser.close = lambda: None
        answers = ["a task " + EVIL, "/quit"]
        with quiet() as out, mock.patch.object(builtins, "input", lambda prompt="": answers.pop(0)):
            AGENT.run_interactive(settings(), AGENT.ExternalServer(fake.base), browser)
        text = out.getvalue()
        self.assertIn("Result ok done", text)
        for bad in ("\x1b]", "\x07", "\x1b[2J", "\u202e", "evil title"):
            self.assertNotIn(bad, text)
        self.assertIn("answer", browser.overlays)

    def test_the_clipboard_is_given_back(self):
        board = {"text": "the user's own text"}
        sets = []

        def put(text):
            board["text"] = text
            return True

        def put_back(text):
            sets.append(text)
            return put(text)

        with mock.patch.multiple(AGENT, get_clipboard_text=lambda: board["text"], copy_to_clipboard=put,
                                 set_clipboard_text=put_back):
            give_back = AGENT.lend_clipboard(AGENT.INSPECT_URL)
            self.assertEqual(board["text"], AGENT.INSPECT_URL)
            give_back()
            self.assertEqual(board["text"], "the user's own text")

            give_back = AGENT.lend_clipboard(AGENT.INSPECT_URL)
            board["text"] = "copied meanwhile"
            give_back()
            self.assertEqual(board["text"], "copied meanwhile")  # the user's newer copy stays

            board["text"] = None  # nothing (or no text) on the clipboard before
            give_back = AGENT.lend_clipboard(AGENT.INSPECT_URL)
            give_back()
            self.assertEqual(sets, ["the user's own text"])
        with mock.patch.object(AGENT, "copy_to_clipboard", lambda text: False):
            self.assertIsNone(AGENT.lend_clipboard("x"))


class DataAtRestTests(unittest.TestCase):
    def setUp(self):
        reset_stop(self)
        self.folder = scratch_paths(self)

    def test_logs_do_not_name_the_windows_account(self):
        with helpers.isolated_env() as base:
            home = str(Path.home())
            self.assertEqual(Path(home), base / "Home")
            self.assertEqual(AGENT.hide_home(home + "\\x\\y.txt"), "%USERPROFILE%\\x\\y.txt")
            self.assertEqual(AGENT.hide_home(home.upper().replace("\\", "/") + "/x"), "%USERPROFILE%/x")
            self.assertEqual(AGENT.hide_home(f'"{home}" and {home}'), '"%USERPROFILE%" and %USERPROFILE%')
            self.assertEqual(AGENT.hide_home(home + "2\\x"), home + "2\\x")  # only that exact folder
            log = AGENT._ConsoleLogFile(self.folder / "agent.log")
            log.write_line(f"  loading {home}\\AppData\\Local\\GhostAgent\\models\\m.gguf")
            log.close()
            text = (self.folder / "agent.log").read_text(encoding="utf-8")
            self.assertIn("%USERPROFILE%\\AppData", text)
            self.assertNotIn(str(base), text)

    def test_old_screenshots_are_pruned(self):
        with helpers.isolated_env():
            shots = AGENT.screenshots_dir()
            shots.mkdir(parents=True)
            old, new, other = shots / "old.png", shots / "new.png", shots / "notes.txt"
            for p in (old, new, other):
                p.write_bytes(b"x")
            week_ago = time.time() - 8 * 86400
            os.utime(old, (week_ago, week_ago))
            os.utime(other, (week_ago, week_ago))
            self.assertEqual(AGENT.prune_screenshots(), 1)
            self.assertEqual(sorted(p.name for p in shots.iterdir()), ["new.png", "notes.txt"])
        with helpers.isolated_env():
            self.assertEqual(AGENT.prune_screenshots(), 0)  # no folder at all

    def test_close_removes_the_browser_cache(self):
        b = AGENT.Browser(settings(browser="bundled"))
        cache = helpers.temp_dir(self) / "cache"
        (cache / "Cache_Data").mkdir(parents=True)
        (cache / "Cache_Data" / "data_1").write_bytes(b"page bytes")
        b._cache_dir = cache
        b.close()
        self.assertFalse(cache.exists())
        self.assertIsNone(b._cache_dir)

    def test_profile_kept_on_the_pc_when_asked(self):
        with helpers.isolated_env() as base:
            on_stick = AGENT.PROFILE
            AGENT.use_profile_location(settings())
            self.assertEqual(AGENT.PROFILE, on_stick)
            AGENT.use_profile_location(settings(profile_on_stick=False))
            self.assertEqual(AGENT.PROFILE, base / "LocalAppData" / "GhostAgent" / "browser-profile")


class ForgetPcTests(unittest.TestCase):
    def setUp(self):
        reset_stop(self)
        scratch_paths(self)
        self.base = self.enterContext(helpers.isolated_env(TEMP=helpers.temp_dir(self)))
        self.pc = AGENT.pc_data_dir()
        self.cache = AGENT.browser_cache_dir()

    def make_leftovers(self):
        for rel in ("models/Holo4.gguf", "screenshots/a.png", "browser-profile/Default/Cookies", "logs/agent-1.log",
                    "downloads/file.pdf", "pc.json"):
            path = self.pc / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
        (self.cache / "Cache_Data").mkdir(parents=True)
        (self.cache / "Cache_Data" / "data_0").write_bytes(b"x")
        old = self.pc.parent / "HoloAgent" / "models"
        old.mkdir(parents=True)
        return [self.pc / "models", self.pc / "screenshots", self.pc / "browser-profile", self.pc / "logs",
                self.pc / "downloads", self.pc / "pc.json", self.cache, old.parent]

    def forget(self, answer):
        later = []
        with quiet() as out, mock.patch.object(builtins, "input", lambda prompt="": answer):
            AGENT.forget_pc(settings(), later)
        return later, out.getvalue()

    def test_nothing_to_forget(self):
        later, out = self.forget("y")
        self.assertIn("keeps nothing on this PC", out)

    def test_no_keeps_everything(self):
        paths = self.make_leftovers()
        later, out = self.forget("")
        self.assertIn("Nothing was deleted", out)
        for p in paths:
            self.assertTrue(p.exists(), p)
            self.assertIn(str(p), out)

    def test_yes_deletes_everything(self):
        paths = self.make_leftovers()
        later, out = self.forget("y")
        self.assertEqual(later, [])
        for p in paths:
            self.assertFalse(p.exists(), p)
        self.assertFalse(self.pc.exists())

    def test_what_is_in_use_goes_when_ghostagent_closes(self):
        self.make_leftovers()
        busy = open(self.pc / "screenshots" / "a.png", "ab")  # held open, as a running program would
        try:
            later, out = self.forget("yes")
            self.assertEqual([p for _, p in later], [self.pc / "screenshots"])
            self.assertIn("is in use; it is deleted when GhostAgent closes", out)
            self.assertFalse((self.pc / "models").exists())
        finally:
            busy.close()
        with quiet():
            AGENT.finish_forget_pc(later)
        self.assertFalse(self.pc.exists())

    def test_the_command_in_the_console(self):
        self.make_leftovers()
        self.assertIn("/forget-pc", AGENT.HELP)
        browser = StubBrowser()
        browser.mode = "bundled"
        browser.start = lambda start_url=None: None
        browser.close = lambda: None
        answers = ["/forget-pc", "y", "/quit"]
        with quiet(), mock.patch.object(builtins, "input", lambda prompt="": answers.pop(0)):
            code = AGENT.run_interactive(settings(), AGENT.ExternalServer(helpers.start_fake_llama(self).base), browser)
        self.assertEqual(code, 0)
        self.assertFalse(self.pc.exists())
        self.assertFalse(self.cache.exists())


if __name__ == "__main__":
    unittest.main()
