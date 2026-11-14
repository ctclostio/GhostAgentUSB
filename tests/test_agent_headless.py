"""The agent end to end, unattended: ghost_agent.py --tasks-file ... --headless-browser against
tools\\bench\\fake_llama.py (the external_server setting) and the local benchmark site. Also the
headless browser itself (dialogs, tabs, timings, the throwaway profile). Nothing shows on screen."""

import _thread
import json
import os
import re
import signal
import threading
import time
import unittest

import helpers
from agent_support import (AGENT, element_centre, quiet, read_jsonl, reset_stop, run_cli, scratch_paths, settings)
from test_agent_trace import check_step_line, check_task_line

import fake_llama
import replay
import score

TYPED = "blue heron"


def tool(name, **args):
    return {"name": name, "arguments": args}


class HeadlessRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.site = helpers.start_bench_server()
        cls.suite = helpers.suite_tasks(cls.site.port)
        cls.q = element_centre(cls.site, "/search/index.html", "#q")
        cls.button = element_centre(cls.site, "/search/index.html", "form#search button")

    @classmethod
    def tearDownClass(cls):
        cls.site.stop()

    def setUp(self):
        self.d = helpers.temp_dir(self)
        self.temp = self.d / "temp"  # the agent's TEMP, to check that nothing is left behind there
        self.temp.mkdir()

    def write_tasks(self, tasks):
        path = self.d / "tasks.json"
        path.write_text(json.dumps(tasks), encoding="utf-8")
        return path

    def run_agent(self, tasks, override, extra=(), timeout=180):
        args = ["--headless-browser", "--data-dir", self.d, "--tasks-file", self.write_tasks(tasks),
                "--results", self.d / "results.jsonl", "--settings-override", json.dumps(override)] + list(extra)
        env = dict(os.environ, TEMP=str(self.temp), TMP=str(self.temp))
        t0 = time.monotonic()
        r = run_cli(args, timeout=timeout, env=env)
        return r, time.monotonic() - t0

    def test_a_suite_task_end_to_end(self):
        task = next(t for t in self.suite if t["name"] == "search-submit")
        script = [
            {"reasoning": "Type the words into the search box.",
             "tool_calls": [tool("write", element="search box", x=self.q[0], y=self.q[1], text=TYPED)]},
            {"reasoning": "Now press the Search button.",
             "tool_calls": [tool("click_web", element="Search button", x=self.button[0], y=self.button[1])]},
            {"reasoning": "The results list the article and its writer.",
             "tool_calls": [tool("answer", content="The article about the blue heron was written by Marta Okafor.")]},
        ]
        fake = helpers.start_fake_llama(self, script=script, validate="strict")
        override = {"external_server": fake.base, "settle_seconds": 0.2, "trace_messages": True, "seed": 11}
        r, took = self.run_agent([{"name": task["name"], "task": task["task"], "start_url": task["start_url"]}], override)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertLess(took, 60)

        # The results line.
        results = read_jsonl(self.d / "results.jsonl")
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(list(res), ["name", "task_id", "answer", "outcome", "steps", "wall_s", "s_per_step"])
        self.assertEqual((res["name"], res["outcome"], res["steps"]), ("search-submit", "answer", 3))
        self.assertIn("Marta Okafor", res["answer"])
        self.assertAlmostEqual(res["s_per_step"], res["wall_s"] / 3, places=2)

        # The step trace.
        logs = self.d / "logs"
        steps_files = list(logs.glob("steps-*.jsonl"))
        self.assertEqual(len(steps_files), 1)
        trace_text = steps_files[0].read_text(encoding="utf-8")
        lines = read_jsonl(steps_files[0])
        self.assertEqual([l["kind"] for l in lines], ["step", "step", "step", "task"])
        for line in lines[:-1]:
            check_step_line(self, line)
        check_task_line(self, lines[-1])
        self.assertEqual({l["task_id"] for l in lines}, {res["task_id"]})
        self.assertEqual([l["tool"] for l in lines[:-1]], ["write", "click_web", "answer"])
        self.assertEqual(lines[0]["url"], self.site.url("/search/index.html"))
        self.assertEqual(lines[2]["url"], self.site.url("/search/results.html"))  # the query is left out
        self.assertEqual((lines[0]["png_w"], lines[0]["png_h"]), (1280, 800))
        for line in lines[:2]:
            self.assertGreaterEqual(line["t_settle"], 0.2)
        self.assertNotIn("heron", trace_text.lower())  # neither the typed text, nor the task, nor the answer
        self.assertNotIn("okafor", trace_text.lower())

        # Scored as a success by the benchmark scorer, from the site's own log.
        summary = score.score(results, self.suite, self.site.entries(), lines)
        self.assertEqual((summary["runs"], summary["successes"]), (1, 1), summary["runs_detail"])
        self.assertEqual(summary["step_times_from"], "trace")

        # The request bodies saved for replay are the ones the server received.
        tasks = replay.discover(logs / "replay")
        self.assertEqual(len(tasks), 1)
        task_id, step_files = tasks[0]
        self.assertEqual(task_id, res["task_id"])
        chats = fake.chat_requests()
        self.assertEqual(len(step_files), len(chats))
        for f, received in zip(step_files, chats):
            body = replay.load_step(f)
            self.assertEqual(fake_llama.scrub_images(body), received["body"])
            self.assertEqual(body["seed"], 11)

        # The console log, without colour codes.
        agent_logs = list(logs.glob("agent-*.log"))
        self.assertEqual(len(agent_logs), 1)
        console = agent_logs[0].read_text(encoding="utf-8")
        self.assertNotIn("\x1b", console)
        for want in ("[1/1] search-submit", "Answer: The article about the blue heron", "1 of 1 tasks answered"):
            self.assertIn(want, console)

        # A first start leaves a settings.json that holds no settings, only a note.
        user = json.loads((self.d / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(list(user), ["_note"])
        # The headless browser used a throwaway profile, removed at the end, and not the stick's.
        self.assertFalse((self.d / "browser-profile").exists())
        self.assertEqual(list(self.temp.glob("ghost-agent-headless-*")), [])
        self.assertEqual(list(logs.glob("bench-server-*.jsonl")), [])  # the site was already running

    def test_the_benchmark_site_is_started_when_the_tasks_need_it(self):
        port = helpers.free_port()
        tasks = [t for t in helpers.suite_tasks(port) if t["name"] in ("search-submit", "copy-code")]
        self.assertEqual(len(tasks), 2)
        fake = helpers.start_fake_llama(self)  # no script: every reply answers at once
        r, took = self.run_agent([{k: t[k] for k in ("name", "task", "start_url")} for t in tasks],
                                 {"external_server": fake.base, "settle_seconds": 0.1})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        logs = list((self.d / "logs").glob(f"bench-server-*-{port}.jsonl"))
        self.assertEqual(len(logs), 1)
        bench_rows = read_jsonl(logs[0])
        self.assertEqual([e["task"] for e in bench_rows if e["kind"] == "start"], ["search-submit", "copy-code"])
        steps = read_jsonl(next((self.d / "logs").glob("steps-*.jsonl")))
        self.assertEqual([s["url"] for s in steps if s["kind"] == "step"],
                         [f"http://127.0.0.1:{port}/search/index.html", f"http://127.0.0.1:{port}/code.html"])
        # Every task starts fresh: the second request has no trace of the first task.
        second = fake.chat_requests()[1]["body"]["messages"]
        self.assertNotIn("Earlier in this session", json.dumps(second))
        self.assertEqual([m["role"] for m in second], ["system", "user", "user"])  # no follow-up, no summary
        self.assertNotIn(AGENT.SUMMARY_LABEL, json.dumps(second))
        self.assertNotIn(AGENT.FOLLOW_UP_PREFIX, json.dumps(second))
        self.assertTrue(helpers.wait_until(lambda: AGENT.port_is_free(port), 5))  # the site was stopped

    def test_no_model_server_means_exit_code_2(self):
        port = helpers.free_port()
        r, took = self.run_agent([{"name": "x", "task": "Anything."}],
                                 {"external_server": f"http://127.0.0.1:{port}", "start_url": "about:blank"})
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertLess(took, 40)
        self.assertIn(f"No model server answers at http://127.0.0.1:{port}/health", r.stdout)
        self.assertIn("no task was run", r.stdout)
        self.assertEqual(read_jsonl(self.d / "results.jsonl"), [])
        self.assertNotIn("Press Enter", r.stdout)

    def test_a_server_that_dies_mid_run_means_exit_code_2(self):
        script = self.d / "script.json"
        script.write_text(json.dumps([{"tool_calls": [tool("wait_web", seconds=1)]}]), encoding="utf-8")
        fake = helpers.start_python([helpers.BENCH / "fake_llama.py", "--port", "0", "--exit-after", "1",
                                     "--script", script])
        self.addCleanup(lambda: fake.poll() is None and fake.kill())
        first = fake.stdout.readline()
        m = re.search(r"listening on (http://127\.0\.0\.1:\d+)", first)
        self.assertIsNotNone(m, first)
        r, took = self.run_agent([{"name": "one", "task": "Wait."}, {"name": "two", "task": "Never run."}],
                                 {"external_server": m.group(1), "start_url": "about:blank", "settle_seconds": 0.1})
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("The model server has stopped", r.stdout)
        results = read_jsonl(self.d / "results.jsonl")
        self.assertEqual([(x["name"], x["outcome"], x["steps"]) for x in results], [("one", "error", 2)])
        self.assertTrue(results[0]["answer"].startswith("Error: Could not talk to the model server"))
        fake.wait(10)

    def test_ctrl_c_style_stop_of_a_run_means_exit_code_1(self):
        # A real console Ctrl+C cannot be sent to a windowless process, so this runs main() in-process
        # with run_tasks and a handler-installed STOP, the same path a key press takes.
        reset_stop(self)
        folder = scratch_paths(self)
        fake = helpers.start_fake_llama(self, script=[{"tool_calls": [tool("wait_web", seconds=30)]}])
        browser = AGENT.Browser(settings(), headless=True)
        opts = AGENT.parse_args(["--task", "one", "--task", "two", "--results", str(folder / "r.jsonl")])

        def press_when_waiting():
            # Only once the task is under way (its first request was answered and wait_web runs):
            # before that, the stop handler is not installed yet.
            if helpers.wait_until(lambda: len(fake.chat_requests()) >= 1, 60):
                time.sleep(0.7)
                _thread.interrupt_main(signal.SIGINT)

        threading.Thread(target=press_when_waiting, daemon=True).start()
        with quiet() as out:
            code = AGENT.run_tasks(opts, settings(settle_seconds=0.1, start_url="about:blank"),
                                   AGENT.ExternalServer(fake.base), browser, AGENT.load_tasks(opts))
        self.assertEqual(code, 1, out.getvalue())
        rows = read_jsonl(folder / "r.jsonl")
        self.assertEqual([(x["name"], x["outcome"], x["answer"]) for x in rows], [("task-1", "cancelled", None)])
        self.assertIn("the remaining tasks were not run", out.getvalue())
        self.assertIsNone(browser._temp_profile)  # closed


class HeadlessBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.site = helpers.start_bench_server()

    @classmethod
    def tearDownClass(cls):
        cls.site.stop()

    def setUp(self):
        reset_stop(self)
        scratch_paths(self)

    def start(self, **over):
        browser = AGENT.Browser(settings(**over), headless=True)
        self.addCleanup(browser.close)
        with quiet():
            browser.start(start_url=self.site.url("/fixtures/confirm.html"))
        return browser

    def test_headless_bundled_with_a_throwaway_profile(self):
        b = self.start(browser="chrome")
        self.assertEqual(b.mode, "bundled")  # the switch always means the bundled Chromium
        self.assertIn("HeadlessChrome", b.page.evaluate("navigator.userAgent"))
        profile = b._temp_profile
        self.assertTrue(profile.is_dir())
        self.assertFalse(AGENT.PROFILE.exists())
        b.close()
        self.assertFalse(profile.exists())

    def test_dialogs_dismissed_or_accepted_by_policy(self):
        for policy, title in (("dismiss", "kept"), ("accept", "DELETED")):
            b = self.start(dialog_policy=policy, settle_seconds=0.1)
            b.begin_task("Delete all the items.")  # the task asks for it, so the Delete button needs no question
            box = b.page.locator("#b").bounding_box()  # in the model's 0-1000 coordinates below
            x, y = round((box["x"] + box["width"] / 2) / b.w * 1000), round((box["y"] + box["height"] / 2) / b.h * 1000)
            result = b.execute("click_web", {"element": "Delete", "x": x, "y": y})
            self.assertEqual(b.page.title(), title, policy)
            self.assertIn("confirm dialog said", result)
            self.assertIn("<untrusted_page_text>Permanently delete all 523 items?</untrusted_page_text>", result)
            b.close()

    def test_action_timings_and_a_fresh_start_per_task(self):
        b = self.start(settle_seconds=0.3)
        b.execute("switch_tab_web", {"new_tab_url": self.site.url("/search/index.html")})
        self.assertGreaterEqual(b.last_timings["settle"], 0.3)
        self.assertGreaterEqual(b.last_timings["execute"], 0)
        self.assertEqual(len(b._tabs()), 2)
        b.execute("wait_web", {"seconds": 0})
        r = b.execute("reader_mode", {})
        self.assertIsNone(b.last_timings["settle"])  # nothing changed on the page, so no wait
        b.reset_for_task(self.site.url("/search/index.html"))
        self.assertEqual(len(b._tabs()), 1)
        self.assertEqual(b.page.url, self.site.url("/search/index.html"))
        obs = b.observation()
        self.assertEqual(b.last_url, self.site.url("/search/index.html"))
        self.assertEqual((b.last_shot["png_w"], b.last_shot["png_h"]), (1280, 800))
        self.assertTrue(obs["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
