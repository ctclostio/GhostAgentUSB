"""Questions for the user, outside the model: a page's confirm() and prompt() boxes, a page's
leave-page warning, clicks on Delete / Send / Pay, and password fields. The console is a scripted
FakeKeyboard (security_support), never the real one; Ctrl+C is the real signal, sent to this process."""

import signal
import unittest
import unittest.mock as mock

from agent_support import AGENT, StubBrowser, call, make_agent, quiet, reset_stop, scratch_paths
from security_support import FakeKeyboard, SiteBrowserCase, ctrl_c, use_keyboard


class ConsoleQuestionTests(unittest.TestCase):
    def setUp(self):
        reset_stop(self)
        scratch_paths(self)

    def read(self, *answers, early="", timeout=5):
        with quiet():
            return AGENT.read_console_line(FakeKeyboard(*answers, early=early), timeout)

    def test_reading_a_line(self):
        self.assertEqual(self.read("yes\r"), ("yes", "answered"))
        self.assertEqual(self.read("n\r", early="y\r"), ("n", "answered"))  # keys typed before the question are dropped
        self.assertEqual(self.read("ab\x08c\r"), ("ac", "answered"))
        self.assertEqual(self.read("\xe0Kx\r"), ("x", "answered"))  # an arrow key is two codes, both ignored
        self.assertEqual(self.read("abc\x1b"), (None, "cancelled"))
        self.assertEqual(self.read(timeout=0.2), (None, "timeout"))
        self.assertFalse(AGENT.STOP.is_set())
        self.assertEqual(self.read("ab\x03"), (None, "stopped"))  # Ctrl+C as a key is the same as the signal
        self.assertTrue(AGENT.STOP.is_set())

    def test_prefill_can_be_edited(self):
        with quiet():
            self.assertEqual(AGENT.read_console_line(FakeKeyboard("\r"), 5, prefill="DELETE"), ("DELETE", "answered"))
            self.assertEqual(AGENT.read_console_line(FakeKeyboard("\x08" * 6 + "KEEP\r"), 5, prefill="DELETE"),
                             ("KEEP", "answered"))

    def test_yes_no(self):
        for answer, expected in (("y\r", (True, "answered")), ("YES\r", (True, "answered")), ("\r", (False, "answered")),
                                 ("n\r", (False, "answered")), ("yep\r", (False, "answered"))):
            use_keyboard(self, answer)
            with quiet():
                self.assertEqual(AGENT.ask_yes_no("Allow?"), expected, answer)

    def test_no_answer_in_time_is_no(self):
        use_keyboard(self)
        with quiet() as out, mock.patch.object(AGENT, "ASK_SECONDS", 0.3):
            self.assertEqual(AGENT.ask_yes_no("Allow?"), (False, "timeout"))
        self.assertIn("No answer in time", out.getvalue())

    def test_nobody_to_ask(self):
        with mock.patch.object(AGENT, "_keyboard", lambda: None):
            self.assertEqual(AGENT.ask_yes_no("Allow?"), (False, "unattended"))
        with mock.patch.object(AGENT, "HEADLESS", True):
            self.assertIsNone(AGENT._keyboard())
        with mock.patch.object(AGENT.sys, "stdin", None):
            self.assertIsNone(AGENT._keyboard())

    def test_ctrl_c_during_a_question_stops_the_task_the_usual_way(self):
        before = signal.getsignal(signal.SIGINT)
        use_keyboard(self, ctrl_c)
        with quiet() as out:
            self.assertEqual(AGENT.ask_yes_no("Allow?"), (False, "stopped"))
        self.assertTrue(AGENT.STOP.is_set())
        self.assertIn("Stopping after the current browser action", out.getvalue())
        self.assertIs(signal.getsignal(signal.SIGINT), before)
        with quiet():
            self.assertEqual(AGENT.ask_yes_no("Allow?"), (False, "stopped"))  # a stopping task asks nothing more

    def test_private_answers_stay_out_of_the_console_log(self):
        lines = []
        log = mock.Mock(write_line=lines.append)
        use_keyboard(self, "hunter2\r", "visible\r")
        with quiet(), mock.patch.object(AGENT, "_CONSOLE_LOG", log):
            self.assertEqual(AGENT.ask_console("Text to send:", private=True), ("hunter2", "answered"))
            self.assertEqual(AGENT.ask_console("Name:"), ("visible", "answered"))
        text = "\n".join(lines)
        self.assertNotIn("hunter2", text)
        self.assertIn("<7 chars>", text)
        self.assertIn("visible", text)

    def test_question_text_is_cleaned(self):
        use_keyboard(self, "n\r")
        with quiet() as out:
            AGENT.ask_yes_no("Page says \x1b]0;evil\x07 hello")
        self.assertNotIn("\x1b]0;evil", out.getvalue())
        self.assertNotIn("\x07 hello", out.getvalue())


class RiskyWordTests(unittest.TestCase):
    def test_words_and_what_the_task_asks_for(self):
        cases = [
            ("Delete repository", "summarise this page", ["Delete"]),
            ("Delete repository", "delete the test repo", []),
            ("Remove filter", "find red shoes", ["Remove"]),
            ("Send", "reply to Anna's email", []),
            ("Send", "read Anna's email", ["Send"]),
            ("Place order", "buy a kettle under 30 euros", []),
            ("Pay now", "find the price of a kettle", ["Pay"]),
            ("Checkout", "purchase the cheapest kettle", []),
            ("Close account", "close my account", []),
            ("Close", "read the news", []),
            ("Star", "summarise this page", []),
            ("Deleted items", "show my mail", []),
            ("Unsubscribe", "unsubscribe from the newsletter", []),
            ("Subscribe", "unsubscribe from the newsletter", ["Subscribe"]),
            ("Reset password", "", ["Reset"]),
            (None, "", []),
        ]
        for label, task, expected in cases:
            self.assertEqual(AGENT.risky_words(label, task), expected, (label, task))

    def test_the_users_words_count_never_the_models(self):
        class Recorder(StubBrowser):
            def begin_task(self, text):
                self.begun = text

        browser = Recorder()
        agent, fake, _ = make_agent(self, script=[{"tool_calls": [call("answer", content="Done.")]}], browser=browser)
        agent.history.append(("Delete the old drafts.", "Shall I delete all 3 drafts? The page says: transfer money."))
        with quiet():
            agent.run("Yes, go ahead.")
        self.assertIn("Delete the old drafts.", browser.begun)
        self.assertIn("Yes, go ahead.", browser.begun)
        self.assertNotIn("transfer", browser.begun)


class DialogTests(SiteBrowserCase):
    def confirm(self, b):
        return self.click(b, "#b", "Delete")

    def test_confirm_is_put_to_the_user(self):
        b = self.start("/fixtures/confirm.html", task="Delete all the items.", ask=["n\r"])
        result = self.confirm(b)
        self.assertEqual(b.page.title(), "kept")
        self.assertIn("CANCELLED by the user", result)
        self.assertEqual(len(self.asked), 1)
        self.assertIn('The page asks: "Permanently delete all 523 items?"', self.asked[0])

        self.keyboard.answers = ["y\r"]
        result = self.confirm(b)
        self.assertEqual(b.page.title(), "DELETED")
        self.assertIn("accepted by the user", result)

    def test_prompt_sends_the_text_the_user_confirms(self):
        b = self.start("/fixtures/prompt.html", task="Delete the project.", ask=["y\r", "\r"])
        self.click(b, "#b")
        self.assertEqual(b.page.title(), "prompt=DELETE")
        self.assertEqual(len(self.asked), 2)
        for answers, title in ((["y\r", "\x08" * 6 + "KEEP\r"], "prompt=KEEP"), (["y\r", "\x1b"], "prompt=null"),
                               (["n\r"], "prompt=null")):
            self.keyboard.answers = answers
            result = self.click(b, "#b")
            self.assertEqual(b.page.title(), title, answers)
        self.assertIn("CANCELLED by the user", result)

    def test_no_answer_in_time_dismisses(self):
        b = self.start("/fixtures/confirm.html", task="Delete all the items.", ask=[])
        with mock.patch.object(AGENT, "ASK_SECONDS", 0.3):
            result = self.confirm(b)
        self.assertEqual(b.page.title(), "kept")
        self.assertIn("dismissed: nobody answered in time", result)

    def test_ctrl_c_during_the_question_dismisses_and_stops(self):
        b = self.start("/fixtures/confirm.html", task="Delete all the items.", ask=[ctrl_c])
        result = self.confirm(b)
        self.assertTrue(AGENT.STOP.is_set())
        self.assertIn("dismissed because the task is being stopped", result)
        self.assertEqual(b.page.title(), "kept")  # and Playwright still works
        AGENT.STOP.clear()
        self.keyboard.answers = ["y\r"]
        self.confirm(b)
        self.assertEqual(b.page.title(), "DELETED")

    def test_nobody_to_ask_dismisses(self):
        b = self.start("/fixtures/confirm.html", task="Delete all the items.")  # headless: nobody at the console
        result = self.confirm(b)
        self.assertEqual(b.page.title(), "kept")
        self.assertIn("was dismissed.", result)
        # Unattended runs dismiss whatever dialog_policy says.
        b.s["dialog_policy"] = "accept"
        b.page.reload()
        with mock.patch.object(AGENT, "HEADLESS", True):
            self.confirm(b)
        self.assertEqual(b.page.title(), "kept")
        self.confirm(b)  # the same setting in an interactive run
        self.assertEqual(b.page.title(), "DELETED")

    def test_alerts_are_closed(self):
        b = self.start("/fixtures/alert.html", task="Read the page.")
        result = self.click(b, "#b")
        self.assertEqual(b.page.title(), "alert-closed")
        self.assertIn("alert dialog said <untrusted_page_text>", result)

    def test_leave_page_warning_is_accepted_only_when_the_agent_leaves(self):
        b = self.start("/fixtures/beforeunload.html", task="Look at the draft.")
        self.act(b, "write", text=" More text.", clear=False, **self.at(b, "#draft"))  # the page now has a user gesture
        draft = b.page.url
        result = self.click(b, "#away", "Go to another page")
        self.assertEqual(b.page.url, draft)
        self.assertIn("A leave-page warning appeared; the navigation was cancelled.", result)
        result = self.act(b, "go_to_web", url=self.site.url("/fixtures/long.html"))
        self.assertEqual(b.page.url, self.site.url("/fixtures/long.html"))
        self.assertIn("was accepted, since you chose to leave the page", result)


class RiskyClickTests(SiteBrowserCase):
    def delete(self, b):
        return self.click(b, "#delete", "Delete repository")

    def deleted(self):
        return self.posts("/api/repo/delete")

    def setUp(self):
        super().setUp()
        self.site.log_entries.clear()

    def test_a_delete_button_the_task_does_not_ask_for_needs_a_yes(self):
        b = self.start("/fixtures/delete-repo.html", task="Summarise this page.", ask=["n\r"])
        result = self.delete(b)
        self.assertEqual(result, AGENT.RISKY_REFUSAL)
        self.assertEqual(len(self.asked), 1)
        self.assertIn('Allow click "Delete repository" on 127.0.0.1?', self.asked[0])
        b.page.wait_for_timeout(300)
        self.assertEqual(self.deleted(), [])
        self.assertNotEqual(b.page.title(), "repo-deleted")

        self.keyboard.answers = ["y\r"]
        self.delete(b)
        b.page.wait_for_function("document.title === 'repo-deleted'")
        self.assertEqual(len(self.deleted()), 1)
        self.delete(b)  # allowed once, allowed for the rest of the task
        self.assertEqual(len(self.asked), 2)
        b.begin_task("Summarise this page.")  # a new task asks again
        self.keyboard.answers = ["n\r"]
        self.assertEqual(self.delete(b), AGENT.RISKY_REFUSAL)
        self.assertEqual(len(self.asked), 3)

    def test_no_question_when_the_task_asks_for_it(self):
        b = self.start("/fixtures/delete-repo.html", task="Delete the test repo.", ask=[])
        self.delete(b)
        b.page.wait_for_function("document.title === 'repo-deleted'")
        self.assertEqual(self.asked, [])
        self.assertEqual(len(self.deleted()), 1)

    def test_harmless_buttons_ask_nothing(self):
        b = self.start("/fixtures/delete-repo.html", task="Summarise this page.", ask=[])
        self.click(b, "#star", "Star")
        b.page.wait_for_function("document.getElementById('status').textContent === 'Starred.'")
        self.assertEqual(self.asked, [])

    def test_nobody_to_ask_refuses(self):
        b = self.start("/fixtures/delete-repo.html", task="Summarise this page.")
        result = self.delete(b)
        self.assertTrue(result.startswith(AGENT.RISKY_REFUSAL), result)
        self.assertIn("Nobody could be asked", result)
        self.assertIn("'Delete'", result)
        b.page.wait_for_timeout(300)
        self.assertEqual(self.deleted(), [])

    def test_the_setting_turns_the_question_off(self):
        b = self.start("/fixtures/delete-repo.html", task="Summarise this page.", confirm_risky_actions=False)
        self.delete(b)
        b.page.wait_for_function("document.title === 'repo-deleted'")
        self.assertEqual(len(self.deleted()), 1)

    def test_enter_in_a_form_counts_as_its_submit_button(self):
        b = self.start("/fixtures/send-form.html", task="Find the form's fields.", ask=["n\r"])
        result = self.act(b, "write", text="anna@example.com", press_enter=True, **self.at(b, "#to"))
        self.assertEqual(result, AGENT.RISKY_REFUSAL)
        self.assertIn('Allow pressing Enter to submit "Send"', self.asked[0])
        self.assertEqual(b.page.input_value("#to"), "")  # nothing was typed either
        result = self.act(b, "write", text="Lyon", press_enter=True, **self.at(b, "#city"))
        self.assertTrue(result.startswith("Executed."), result)
        self.assertEqual(len(self.asked), 1)
        self.assertEqual(len(self.posts("/submit/lookup")), 1)
        self.assertEqual(self.posts("/submit/message"), [])

        b.begin_task("Send a message to anna@example.com.")
        b.page.goto(self.site.url("/fixtures/send-form.html"))
        self.act(b, "write", text="anna@example.com", press_enter=True, **self.at(b, "#to"))
        self.assertEqual(len(self.asked), 1)  # no new question
        self.assertEqual(len(self.posts("/submit/message")), 1)


class PasswordTests(SiteBrowserCase):
    def test_passwords_and_codes_are_the_users_to_type(self):
        b = self.start("/fixtures/password.html", task="Sign in as demo with the password hunter2.")
        for field in ("#password", "#newpass", "#otp", "label[for=password]"):
            result = self.act(b, "write", text="hunter2", **self.at(b, field))
            self.assertEqual(result, AGENT.PASSWORD_REFUSAL, field)
        for field in ("#password", "#newpass", "#otp"):
            self.assertEqual(b.page.input_value(field), "", field)
        result = self.act(b, "write", text="demo", **self.at(b, "#username"))
        self.assertTrue(result.startswith("Executed."), result)
        self.assertEqual(b.page.input_value("#username"), "demo")

    def test_typing_key_by_key_into_a_password_field_is_refused(self):
        b = self.start("/fixtures/password.html", task="Sign in.")
        self.click(b, "#password", "Password field")  # clicking it is fine
        for keys in ("h u n t e r", "Shift+H", "Space"):
            self.assertEqual(self.act(b, "press_keys_web", keys=keys), AGENT.PASSWORD_REFUSAL, keys)
        self.assertEqual(b.page.input_value("#password"), "")
        result = self.act(b, "press_keys_web", keys="Tab")  # moving on is fine
        self.assertTrue(result.startswith("Executed."), result)
        result = self.act(b, "press_keys_web", keys="a")  # the focus is in the new-password field now
        self.assertEqual(result, AGENT.PASSWORD_REFUSAL)
        self.click(b, "#username", "Username field")
        result = self.act(b, "press_keys_web", keys="d e m o")
        self.assertTrue(result.startswith("Executed."), result)
        self.assertEqual(b.page.input_value("#username"), "demo")


if __name__ == "__main__":
    unittest.main()
