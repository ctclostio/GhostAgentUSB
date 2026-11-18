"""CS-03: Ctrl+C stops a task through a flag (STOP), so it never lands inside a Playwright call.

Ctrl+C is simulated with _thread.interrupt_main(SIGINT), which runs the installed SIGINT handler in
the main thread exactly as a real key press would. Browsers are the bundled Chromium, headless."""

import _thread
import builtins
import signal
import textwrap
import threading
import time
import types
import unittest
import unittest.mock as mock

import helpers
from agent_support import (AGENT, StubBrowser, call, make_agent, patch_agent, quiet, read_jsonl, reset_stop,
                           scratch_paths, settings, steps_file)


def press_ctrl_c(after):
    """Simulate Ctrl+C in the main thread after `after` seconds. Returns the timer."""
    t = threading.Timer(after, _thread.interrupt_main, args=(signal.SIGINT,))
    t.daemon = True
    t.start()
    return t


def set_stop(after):
    t = threading.Timer(after, AGENT.STOP.set)
    t.daemon = True
    t.start()
    return t


class HandlerTests(unittest.TestCase):
    def setUp(self):
        reset_stop(self)

    def test_first_press_sets_the_flag_and_second_press_forces(self):
        before = signal.getsignal(signal.SIGINT)
        clock = [100.0]
        fake_time = types.SimpleNamespace(monotonic=lambda: clock[0])
        with quiet() as out, patch_agent("time", fake_time):
            with AGENT.ctrl_c_stops_task():
                self.assertIsNot(signal.getsignal(signal.SIGINT), before)
                handler = signal.getsignal(signal.SIGINT)
                handler(signal.SIGINT, None)
                self.assertTrue(AGENT.STOP.is_set())
                self.assertIn("Stopping after the current browser action", out.getvalue())
                # A second press more than 3 seconds later is just another first press.
                clock[0] += 3.5
                handler(signal.SIGINT, None)
                self.assertIs(signal.getsignal(signal.SIGINT), handler)
                # Within 3 seconds: the previous handler comes back and the agent is told to quit.
                clock[0] += 2.0
                with self.assertRaises(AGENT.ForcedStop):
                    handler(signal.SIGINT, None)
                self.assertIs(signal.getsignal(signal.SIGINT), before)
        self.assertIs(signal.getsignal(signal.SIGINT), before)
        self.assertTrue(issubclass(AGENT.ForcedStop, KeyboardInterrupt))

    def test_the_previous_handler_is_restored_after_an_exception(self):
        before = signal.getsignal(signal.SIGINT)
        with self.assertRaises(ValueError):
            with AGENT.ctrl_c_stops_task():
                raise ValueError("x")
        self.assertIs(signal.getsignal(signal.SIGINT), before)

    def test_a_real_signal_only_sets_the_flag(self):
        with quiet():
            with AGENT.ctrl_c_stops_task():
                _thread.interrupt_main(signal.SIGINT)
                self.assertTrue(helpers.wait_until(AGENT.STOP.is_set, 2.0))
                time.sleep(0.05)  # no KeyboardInterrupt arrives
        self.assertTrue(AGENT.STOP.is_set())

    def test_clear_flag(self):
        AGENT.STOP.set()
        with AGENT.ctrl_c_stops_task(clear=False):
            self.assertTrue(AGENT.STOP.is_set())
        with AGENT.ctrl_c_stops_task():
            self.assertFalse(AGENT.STOP.is_set())

    def test_other_threads_leave_the_handler_alone(self):
        before = signal.getsignal(signal.SIGINT)
        seen = []

        def worker():
            with AGENT.ctrl_c_stops_task():
                seen.append(signal.getsignal(signal.SIGINT))

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        self.assertEqual(seen, [before])


class StreamCancelTests(unittest.TestCase):
    def setUp(self):
        reset_stop(self)

    def test_stop_cuts_a_waiting_request_at_once(self):
        fake = helpers.start_fake_llama(self, script=[{"delay": 20, "content": "late"}])
        set_stop(0.3)
        t0 = time.monotonic()
        with self.assertRaises(AGENT.StepCancelled):
            AGENT.stream_chat(fake.base, {"messages": [{"role": "user", "content": "x"}], "stream": True},
                              lambda k, v: None)
        self.assertLess(time.monotonic() - t0, 1.5)

    def test_stop_mid_stream_drops_the_request_and_the_server_is_free_again(self):
        fake = helpers.start_fake_llama(self, script=[{"reasoning": "word " * 400, "chunk_delay": 0.02},
                                                      {"content": "next one"}])
        events = []
        set_stop(0.4)
        t0 = time.monotonic()
        with self.assertRaises(AGENT.StepCancelled):
            AGENT.stream_chat(fake.base, {"messages": [{"role": "user", "content": "x"}], "stream": True},
                              lambda k, v: events.append(k))
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertIn("reasoning", events)  # it was cut mid-reply
        AGENT.STOP.clear()
        # The fake serves one request at a time, like llama-server's single slot: the next request is
        # answered promptly only if the cancelled one was really dropped.
        t1 = time.monotonic()
        msg = AGENT.stream_chat(fake.base, {"messages": [{"role": "user", "content": "y"}], "stream": True},
                                lambda k, v: None)
        self.assertEqual(msg["content"], "next one")
        self.assertLess(time.monotonic() - t1, 3.0)


class PlaywrightSurvivesTests(unittest.TestCase):
    """The reason for CS-03: a KeyboardInterrupt inside a Playwright call kills Playwright's
    dispatcher. With the flag handler, a Ctrl+C during a Playwright call leaves it working."""

    def setUp(self):
        reset_stop(self)

    def test_ctrl_c_during_a_playwright_call_leaves_playwright_working(self):
        with helpers.headless_chromium() as (pw, browser):
            page = browser.new_page()
            page.set_content("<p id=p>hello</p>")
            holder = AGENT.Browser(settings(browser="bundled"))
            holder.pw = pw
            self.assertIsNotNone(pw._impl_obj._connection._dispatcher_fiber)
            self.assertFalse(holder.playwright_dead())
            with quiet():
                with AGENT.ctrl_c_stops_task():
                    press_ctrl_c(0.3)
                    t0 = time.monotonic()
                    page.wait_for_timeout(1200)  # the call is allowed to finish
                    self.assertGreaterEqual(time.monotonic() - t0, 1.1)
            self.assertTrue(AGENT.STOP.is_set())
            self.assertFalse(holder.playwright_dead())
            # Later calls finish normally (before CS-03 they waited forever).
            self.assertEqual(page.evaluate("1 + 1"), 2)
            self.assertEqual(page.inner_text("#p"), "hello")
            page.set_content("<p id=p>again</p>")
            self.assertEqual(page.inner_text("#p"), "again")

    def test_playwright_dead_detects_a_real_interrupted_dispatcher(self):
        # In a separate process: after this, that process's Playwright can no longer be used or shut
        # down cleanly, so it leaves with os._exit.
        code = textwrap.dedent(f"""
            import _thread, os, signal, sys, threading
            sys.path.insert(0, {str(helpers.AGENT_DIR)!r})
            import ghost_agent  # the entry script, which puts the agent's package on sys.path
            from ghostagent.browser import Browser
            from ghostagent.settings import DEFAULT_SETTINGS
            from playwright.sync_api import sync_playwright
            pw = sync_playwright().start()
            browser = pw.chromium.launch(channel="chromium", headless=True, args={helpers.CHROMIUM_ARGS!r})
            page = browser.new_page()
            holder = Browser({{**DEFAULT_SETTINGS, "browser": "bundled"}})
            holder.pw = pw
            print("before", holder.playwright_dead(), flush=True)
            threading.Timer(0.3, _thread.interrupt_main, args=(signal.SIGINT,)).start()
            try:
                page.wait_for_timeout(3000)
            except KeyboardInterrupt:
                print("interrupted", flush=True)
            print("after", holder.playwright_dead(), flush=True)
            holder.close()  # must return at once instead of hanging
            holder.overlay("remove")
            print("closed", flush=True)
            os._exit(0)
        """)
        r = helpers.run_python(["-B", "-c", code], timeout=90, encoding="utf-8", errors="replace")
        lines = r.stdout.splitlines()
        self.assertEqual(lines, ["before False", "interrupted", "after True", "closed"], r.stdout + r.stderr)

    def test_close_and_overlay_skip_playwright_when_it_is_dead(self):
        calls = []

        class Boom:
            def __getattr__(self, name):
                calls.append(name)
                raise AssertionError(f"Playwright was used: {name}")

        b = AGENT.Browser(settings(browser="bundled"))
        b.pw = types.SimpleNamespace(_impl_obj=types.SimpleNamespace(_connection=types.SimpleNamespace(
            _dispatcher_fiber=types.SimpleNamespace(dead=True))), stop=lambda: calls.append("stop"))
        b.ctx = Boom()
        b.page = Boom()
        temp = helpers.temp_dir(self)
        b._temp_profile = temp
        self.assertTrue(b.playwright_dead())
        b.overlay("remove")
        b.close()
        self.assertEqual(calls, [])
        self.assertFalse(temp.exists())  # the throwaway profile is still removed
        # Nothing to inspect (no Playwright at all) is not "dead".
        self.assertFalse(AGENT.Browser(settings(browser="bundled")).playwright_dead())


class BrowserStopTests(unittest.TestCase):
    """wait_web and the agent loop with the real browser (bundled Chromium, headless)."""

    @classmethod
    def setUpClass(cls):
        cls.site = helpers.start_bench_server()

    @classmethod
    def tearDownClass(cls):
        cls.site.stop()

    def setUp(self):
        reset_stop(self)
        scratch_paths(self)
        # wait_web waits as long as asked unless stopped, which both tests here rely on.
        self.browser = AGENT.Browser(settings(settle_seconds=0.2), headless=True)
        self.addCleanup(self.browser.close)
        with quiet():
            self.browser.start(start_url=self.site.url("/search/"))

    def test_wait_web_ends_early_on_stop_without_spinning(self):
        set_stop(0.4)
        t0 = time.monotonic()
        result = self.browser.execute("wait_web", {"seconds": 30})
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertIn("Executed.", result)
        self.assertLess(self.browser.last_timings["settle"], 0.1)  # no settle wait once stopping
        AGENT.STOP.clear()
        cpu0, t0 = time.process_time(), time.monotonic()
        self.browser.execute("wait_web", {"seconds": 1.5})
        wall, cpu = time.monotonic() - t0, time.process_time() - cpu0
        self.assertGreaterEqual(wall, 1.5)
        self.assertLess(cpu, wall * 0.5, "waiting should not keep a CPU busy")

    def test_ctrl_c_during_wait_web_cancels_the_task_and_the_next_task_works(self):
        script = [{"tool_calls": [call("wait_web", seconds=30)]}, {"tool_calls": [call("answer", content="second ok")]}]
        agent, fake, folder = make_agent(self, script=script, browser=self.browser)
        t0 = time.monotonic()
        with quiet() as out:
            with AGENT.ctrl_c_stops_task():
                press_ctrl_c(0.6)
                with self.assertRaises(AGENT.StepCancelled):
                    agent.run("wait a long time")
        self.assertLess(time.monotonic() - t0, 6.0)
        self.assertIn("Stopping after the current browser action", out.getvalue())
        self.assertEqual(agent.last["outcome"], "cancelled")
        self.assertFalse(self.browser.playwright_dead())
        with quiet():
            with AGENT.ctrl_c_stops_task():  # cleared for the new task
                self.assertEqual(agent.run("now answer"), "second ok")
        lines = read_jsonl(steps_file(folder))
        self.assertEqual([l["outcome"] for l in lines if l["kind"] == "task"], ["cancelled", "answer"])


class InteractiveTests(unittest.TestCase):
    """run_interactive with a stub browser and the fake model server (ExternalServer)."""

    def setUp(self):
        reset_stop(self)
        scratch_paths(self)

    def run_prompt(self, answers, script):
        fake = helpers.start_fake_llama(self, script=script)
        browser = StubBrowser()
        browser.mode = "bundled"
        browser.start = lambda start_url=None: None
        browser.closed = False
        browser.close = lambda: setattr(browser, "closed", True)
        prompts = []

        def fake_input(prompt=""):
            prompts.append(prompt)
            value = answers.pop(0)
            if isinstance(value, BaseException) or (isinstance(value, type) and issubclass(value, BaseException)):
                raise value
            if callable(value):
                return value()
            return value

        with quiet() as out, mock.patch.object(builtins, "input", fake_input):
            code = AGENT.run_interactive(settings(), AGENT.ExternalServer(fake.base), browser)
        return code, out.getvalue(), prompts, browser, fake

    def test_ctrl_c_at_the_prompt_just_asks_again(self):
        code, out, prompts, browser, fake = self.run_prompt([KeyboardInterrupt, "/quit"], [])
        self.assertEqual(code, 0)
        self.assertEqual(len(prompts), 2)
        self.assertTrue(browser.closed)

    def test_ctrl_c_during_a_task_returns_to_the_prompt(self):
        def task_then_ctrl_c():
            press_ctrl_c(0.5)
            return "a slow task"

        # A long streamed reply (not a "delay": the fake only notices a dropped request when it writes).
        script = [{"reasoning": "word " * 400, "chunk_delay": 0.02, "tool_calls": [call("answer", content="too late")]},
                  {"tool_calls": [call("answer", content="the second answer")]}]
        t0 = time.monotonic()
        code, out, prompts, browser, fake = self.run_prompt([task_then_ctrl_c, "a quick task", "/quit"], script)
        self.assertLess(time.monotonic() - t0, 10)
        self.assertEqual(code, 0)
        self.assertEqual(len(prompts), 3)
        self.assertIn("Stopped.", out)
        self.assertIn("the second answer", out)
        self.assertIn("remove", browser.overlays)


class RunMainTests(unittest.TestCase):
    def run_main(self, exc, headless=False):
        asked = []

        def main(argv=None):
            raise exc

        with quiet() as out, patch_agent("main", main), patch_agent("HEADLESS", headless), \
                mock.patch.object(builtins, "input", lambda prompt="": asked.append(prompt) or ""):
            code = AGENT.run_main([])
        return code, out.getvalue(), asked

    def test_exit_codes_and_the_pause(self):
        code, out, asked = self.run_main(AGENT.ForcedStop())
        self.assertEqual(code, 1)
        self.assertIn("Stopped.", out)
        self.assertEqual(len(asked), 1)
        self.assertIn("Press Enter", asked[0])

        code, out, asked = self.run_main(KeyboardInterrupt())
        self.assertEqual((code, asked), (0, []))
        code, out, asked = self.run_main(KeyboardInterrupt(), headless=True)
        self.assertEqual((code, asked), (1, []))

        code, out, asked = self.run_main(SystemExit("Something is wrong."))
        self.assertEqual(code, 1)
        self.assertIn("Something is wrong.", out)
        self.assertEqual(len(asked), 1)
        self.assertEqual(self.run_main(SystemExit(0))[0], 0)
        code, out, asked = self.run_main(SystemExit(2), headless=True)
        self.assertEqual((code, asked), (2, []))

        code, out, asked = self.run_main(RuntimeError("bad"))
        self.assertEqual(code, 1)
        self.assertIn("RuntimeError: bad", out)
        self.assertEqual(len(asked), 1)
        code, out, asked = self.run_main(RuntimeError("bad"), headless=True)
        self.assertEqual((code, asked), (1, []))  # unattended runs never wait for a key


if __name__ == "__main__":
    unittest.main()
