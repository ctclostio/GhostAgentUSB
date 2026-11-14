"""Keyboard routes around the safety rails: other spellings of the paste chord and of typed keys,
key presses that set off a Delete or Send button, line breaks in written text, and text a page puts
into the console. Uses the agent's own headless browser on the benchmark site and a scripted console
(security_support); nothing reads the real console or clipboard."""

import os
import unittest
import unittest.mock as mock
from pathlib import Path

from agent_support import AGENT, quiet, reset_stop, scratch_paths, settings
from security_support import FakeKeyboard, SiteBrowserCase, use_keyboard


class KeyNameTests(unittest.TestCase):
    def test_every_spelling_of_the_paste_chord_is_refused(self):
        for keys in ("ControlOrMeta+v", "ControlOrMeta+V", "Control+KeyV", "control + keyv", "Meta+KeyV",
                     "Control+Shift+KeyV", "ctrl+V", "Shift+Insert"):
            presses = AGENT.key_presses(keys)
            self.assertTrue(any(AGENT.is_paste(k) for k in presses), keys)
        for keys in ("Control+KeyC", "KeyV", "Control+A", "Shift+KeyV", "Insert"):
            presses = AGENT.key_presses(keys)
            self.assertFalse(any(AGENT.is_paste(k) for k in presses), keys)

    def test_key_codes_count_as_typing(self):
        for combo in ("a", "Shift+A", "Space", "KeyA", "Shift+KeyB", "Digit1", "Numpad5", "Equal", "Slash", "NumpadAdd"):
            self.assertTrue(AGENT.types_text(AGENT.normalize_key(combo)), combo)
        for combo in ("Tab", "Enter", "Escape", "F5", "Control+A", "ArrowLeft", "Shift+Tab", "Backspace", "PageDown"):
            self.assertFalse(AGENT.types_text(AGENT.normalize_key(combo)), combo)

    def test_presses_that_set_off_a_button(self):
        for combo in ("Enter", "Control+Enter", "NumpadEnter", "Space"):
            self.assertTrue(AGENT.activates(AGENT.normalize_key(combo)), combo)
        for combo in ("Tab", "a", "Shift+Space", "Escape"):
            self.assertFalse(AGENT.activates(AGENT.normalize_key(combo)), combo)


class KeyGuardBrowserTests(SiteBrowserCase):
    def setUp(self):
        super().setUp()
        self.site.log_entries.clear()

    def test_key_codes_cannot_type_into_a_password_field(self):
        b = self.start("/fixtures/password.html", task="Sign in.")
        self.click(b, "#password", "Password field")
        for keys in ("KeyA", "Digit1 Equal", "Shift+KeyB"):
            self.assertEqual(self.act(b, "press_keys_web", keys=keys), AGENT.PASSWORD_REFUSAL, keys)
        self.assertEqual(b.page.input_value("#password"), "")

    def test_tab_into_a_password_field_then_typing_is_refused(self):
        b = self.start("/fixtures/password.html", task="Sign in.")
        self.click(b, "#username", "Username field")
        result = self.act(b, "press_keys_web", keys="Tab h u n t e r")
        self.assertEqual(result, AGENT.PASSWORD_REFUSAL)
        self.assertEqual(b.page.input_value("#password"), "")
        self.assertEqual(b.page.input_value("#username"), "")

    def test_enter_on_a_focused_delete_button_needs_the_same_yes_as_a_click(self):
        b = self.start("/fixtures/delete-repo.html", task="Summarise this page.")  # nobody to ask
        b.page.focus("#delete")
        for keys in ("Enter", "Space"):
            result = self.act(b, "press_keys_web", keys=keys)
            self.assertTrue(result.startswith(AGENT.RISKY_REFUSAL), (keys, result))
        b.page.focus("#star")
        result = self.act(b, "press_keys_web", keys="Tab Enter")  # Tab moves the focus to Delete first
        self.assertTrue(result.startswith(AGENT.RISKY_REFUSAL), result)
        b.page.wait_for_timeout(300)
        self.assertEqual(self.posts("/api/repo/delete"), [])

        b.begin_task("Delete the test repo.")
        b.page.focus("#delete")
        self.act(b, "press_keys_web", keys="Enter")
        b.page.wait_for_function("document.title === 'repo-deleted'")
        self.assertEqual(len(self.posts("/api/repo/delete")), 1)

    def test_enter_key_in_a_form_field_counts_as_its_submit_button(self):
        b = self.start("/fixtures/send-form.html", task="Find the form's fields.", ask=["n\r"])
        b.page.focus("#to")
        self.assertEqual(self.act(b, "press_keys_web", keys="Enter"), AGENT.RISKY_REFUSAL)
        self.assertIn('"Send"', self.asked[0])
        b.page.focus("#to")
        result = self.act(b, "press_keys_web", keys="a Space b")  # Space only types in a field
        self.assertTrue(result.startswith("Executed."), result)
        self.assertEqual(len(self.asked), 1)
        b.page.focus("#city")
        result = self.act(b, "press_keys_web", keys="Enter")  # Look up: harmless
        self.assertTrue(result.startswith("Executed."), result)
        self.assertEqual(len(self.asked), 1)
        self.assertEqual(self.posts("/submit/message"), [])
        self.assertEqual(len(self.posts("/submit/lookup")), 1)

    def test_an_address_without_slashes_still_opens(self):
        b = self.start("/fixtures/long.html", task="Read the page.")
        target = self.site.url("/fixtures/navigate.html")
        result = self.act(b, "go_to_web", url=target.replace("http://", "http:"))
        self.assertTrue(result.startswith("Executed."), result)
        self.assertEqual(b.page.url, target)

    def test_a_line_break_in_written_text_counts_as_press_enter(self):
        b = self.start("/fixtures/send-form.html", task="Find the form's fields.", ask=["n\r"])
        result = self.act(b, "write", text="anna@example.com\n", **self.at(b, "#to"))
        self.assertEqual(result, AGENT.RISKY_REFUSAL)
        self.assertEqual(len(self.asked), 1)
        b.page.wait_for_timeout(300)
        self.assertEqual(self.posts("/submit/message"), [])


class AddressSpellingTests(unittest.TestCase):
    """The local-network check reads an address the way the browser does, so another spelling of a
    router's address is refused before anything is sent to it."""

    def browser(self, task="Summarise this page."):
        b = AGENT.Browser(settings(), headless=True)  # not started: _vet needs no browser
        b.begin_task(task)
        b._local["example.com"] = False  # no DNS lookups in tests
        return b

    def test_other_spellings_of_a_local_address_are_refused(self):
        b = self.browser()
        for url in ("http:192.168.1.1", "http:/192.168.1.1/", "https:\\\\192.168.1.1\\admin",
                    "http://192.168.1.1\\@example.com/", "http://%31%39%32.168.1.1/", "http://127.1:8080/",
                    "http://0x7f.1/", "http://0300.0250.1.1/", "http://3232235777/",
                    "http://１９２.168.1.1/", "http://192。168。1。1/"):
            with self.assertRaises(AGENT.ToolRefused, msg=ascii(url)):
                b._vet(url)
        self.assertEqual(b._vet("http:example.com"), "http://example.com")
        self.assertEqual(self.browser("Open 192.168.1.1 and read the status.")._vet("http:192.168.1.1"),
                         "http://192.168.1.1")

    def test_addresses_are_written_the_way_the_browser_reads_them(self):
        self.assertEqual(AGENT.vet_url("HTTP:\\\\Example.com\\a\\b?q=\\x#\\y"), "http://Example.com/a/b?q=\\x#\\y")
        self.assertEqual(AGENT.vet_url("//example.com/x"), "https://example.com/x")
        self.assertEqual(AGENT.vet_url("localhost:8080/x"), "https://localhost:8080/x")
        self.assertEqual(AGENT.vet_url("https://example.com/a?b=c"), "https://example.com/a?b=c")

    def test_browser_ipv4(self):
        cases = {"127.1": "127.0.0.1", "0xc0.0xa8.1.1": "192.168.1.1", "3232235777": "192.168.1.1",
                 "0300.0250.1.1": "192.168.1.1", "10.1.2.3.": "10.1.2.3", "1.2.3.4.5": None, "example.com": None,
                 "256.1.1.1": None, "1.2.3.256": None, "": None}
        for host, expected in cases.items():
            ip = AGENT.browser_ipv4(host)
            self.assertEqual(str(ip) if ip else None, expected, host)


class ConsoleTextTests(unittest.TestCase):
    def setUp(self):
        reset_stop(self)
        scratch_paths(self)

    def test_a_pages_prefilled_text_is_cleaned_before_it_is_printed(self):
        use_keyboard(self, "\r")
        with quiet() as out:
            text, how = AGENT.ask_console("Text to send:", prefill="\x1b]0;evil title\x07DEL\nETE\x1b[2J")
        self.assertEqual((text, how), ("DEL ETE", "answered"))
        for bad in ("\x1b]", "evil title", "\x1b[2J", "\x07DEL"):
            self.assertNotIn(bad, out.getvalue())

    def test_no_bidirectional_controls_in_the_source(self):
        # They make code read differently from what it does (and GitHub flags such files).
        source = Path(AGENT.__file__).read_text(encoding="utf-8")
        self.assertFalse([c for c in source if 0x202A <= ord(c) <= 0x202E or 0x2066 <= ord(c) <= 0x2069])


class Proc:
    _handle = 0

    def poll(self):
        return None


class KeyFileTests(unittest.TestCase):
    def test_a_relative_key_file_is_read_from_the_servers_folder(self):
        folder = scratch_paths(self)
        exe = folder / "bin" / "cpu" / "llama-server.exe"
        exe.parent.mkdir(parents=True, exist_ok=True)
        (exe.parent / "keys.txt").write_text("# comment\nrelative-key\n", encoding="utf-8")
        server = AGENT.LlamaServer(settings(extra_server_args=["--api-key-file", "keys.txt"]))
        with mock.patch.object(AGENT, "pick_backend", return_value=("cpu", exe, None)), \
                mock.patch.object(AGENT, "model_files", return_value=[Path("m.gguf"), Path("p.gguf")]), \
                mock.patch.object(AGENT.subprocess, "Popen", lambda args, **kw: Proc()), \
                mock.patch.object(AGENT, "port_is_free", return_value=True), \
                mock.patch.dict(os.environ, {}), quiet() as out:
            server.start()
        self.assertEqual(server.api_key, "relative-key")
        self.assertNotIn("Could not read the key", out.getvalue())


if __name__ == "__main__":
    unittest.main()
