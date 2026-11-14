"""Shared pieces for the safety-rail tests (test_security_*.py): a scripted console keyboard for the
agent's questions, and the agent's own headless browser on the local benchmark site.

Nothing here reads the real console: the agent's key reader is pointed at FakeKeyboard instead."""

import _thread
import signal
import unittest
import unittest.mock as mock

import helpers
from agent_support import AGENT, quiet, reset_stop, scratch_paths, settings


def ctrl_c():
    """An answer for FakeKeyboard: Ctrl+C pressed while the question waits (the real signal, in the
    main thread, the way a key press delivers it)."""
    _thread.interrupt_main(signal.SIGINT)


class FakeKeyboard:
    """Stands in for msvcrt in the agent's console questions. Each question gets the next answer,
    typed only once the question is up: keys typed before it (`early`) are what the agent must drop.
    An answer can also be a function, which runs instead of typing (for example ctrl_c)."""

    def __init__(self, *answers, early=""):
        self.answers = list(answers)
        self.buf = list(early)
        self.drained = False

    def kbhit(self):
        if self.buf:
            return True
        if not self.drained:
            self.drained = True  # the agent has dropped what was typed before the question
            return False
        if self.answers:
            answer = self.answers.pop(0)
            if callable(answer):
                answer()
                return False
            self.buf = list(answer)
        return bool(self.buf)

    def getwch(self):
        return self.buf.pop(0)

    def new_question(self):
        """A question is about to be read: the agent first drops whatever keys are waiting."""
        self.drained = False


def use_keyboard(test, *answers, early=""):
    """Point the agent's console questions at a FakeKeyboard for one test. Returns (keyboard, the
    list of questions asked)."""
    kb = FakeKeyboard(*answers, early=early)
    asked = []
    real_ask = AGENT.ask_console

    def ask(question, *args, **kwargs):
        asked.append(question)
        kb.new_question()
        return real_ask(question, *args, **kwargs)

    for name, value in (("_keyboard", lambda: kb), ("ask_console", ask)):
        patcher = mock.patch.object(AGENT, name, value)
        patcher.start()
        test.addCleanup(patcher.stop)
    return kb, asked


class SiteBrowserCase(unittest.TestCase):
    """The agent's Browser (bundled Chromium, headless, a throwaway profile) on the benchmark site."""

    @classmethod
    def setUpClass(cls):
        cls.site = helpers.start_bench_server()

    @classmethod
    def tearDownClass(cls):
        cls.site.stop()

    def setUp(self):
        reset_stop(self)
        self.folder = scratch_paths(self)

    def start(self, path, task="", ask=None, **over):
        """A browser at a site page, with a task begun. ask: answers for the console questions (the
        browser then counts as having someone at the console); None = nobody to ask."""
        over.setdefault("settle_seconds", 0.1)
        b = AGENT.Browser(settings(**over), headless=True)
        self.addCleanup(b.close)
        with quiet():
            b.start(start_url=self.site.url(path))
        b.begin_task(task)
        self.asked = []
        if ask is not None:
            b.can_ask = True
            self.keyboard, self.asked = use_keyboard(self, *ask)
        return b

    def at(self, b, selector):
        """The centre of an element in the model's 0-1000 coordinates."""
        box = b.page.locator(selector).first.bounding_box()
        return {"x": round((box["x"] + box["width"] / 2) / b.w * 1000),
                "y": round((box["y"] + box["height"] / 2) / b.h * 1000)}

    def act(self, b, name, **args):
        with quiet():
            return b.execute(name, args)

    def click(self, b, selector, element="it"):
        return self.act(b, "click_web", element=element, **self.at(b, selector))

    def posts(self, path):
        return [e for e in self.site.entries("post") if e.get("path") == path]
