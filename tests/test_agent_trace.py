"""CS-01: the per-step trace, the console log and its rotation, and the benchmark plumbing of the
model server (VRAM emulation, CPU affinity, seed). No GPU: llama-server is never started here."""

import io
import json
import os
import re
import subprocess
import sys
import time
import unittest
import unittest.mock as mock
from pathlib import Path

import helpers
from helpers import fake_llama
from agent_support import (AGENT, StubBrowser, call, make_agent, quiet, read_jsonl, reset_stop, scratch_paths,
                           settings, steps_file)

import replay

SECRET_TEXT = "my secret words 4417"
SECRET_QUERY = "private search 9902"
SECRET_ANSWER = "The account number is 55-1234."
SECRET_REASONING = "I can see the user's balance of 812 dollars"


def check_step_line(test, line):
    test.assertEqual(tuple(line), AGENT.STEP_FIELDS, "step lines carry exactly the documented fields, in order")
    test.assertEqual(line["kind"], "step")
    test.assertIsInstance(line["ts"], float)
    test.assertRegex(line["run_id"], r"^[0-9a-f]{8}$")
    test.assertRegex(line["task_id"], r"^[0-9a-f]{8}$")
    test.assertIsInstance(line["step"], int)
    test.assertGreaterEqual(line["step"], 1)
    test.assertIsInstance(line["evicted"], bool)
    test.assertIsInstance(line["trimmed"], bool)
    for key in ("url", "tool", "args_summary", "finish_reason", "error"):
        test.assertIsInstance(line[key], (str, type(None)), key)
    for key in ("t_observation", "t_model", "t_execute", "t_settle", "prompt_ms", "predicted_ms"):
        test.assertIsInstance(line[key], (int, float, type(None)), key)
        if line[key] is not None:
            test.assertGreaterEqual(line[key], 0, key)
    for key in ("png_bytes", "png_w", "png_h", "images_in_prompt", "cache_n", "prompt_n", "predicted_n",
                "reasoning_chars"):
        test.assertIsInstance(line[key], (int, type(None)), key)
    if line["url"]:
        test.assertNotIn("?", line["url"])
        test.assertNotIn("#", line["url"])


def check_task_line(test, line):
    test.assertEqual(tuple(line), AGENT.TASK_FIELDS)
    test.assertEqual(line["kind"], "task")
    test.assertIn(line["outcome"], ("answer", "error", "cancelled", "max_steps"))
    test.assertIsInstance(line["steps"], int)
    test.assertIsInstance(line["wall_s"], float)


class StepTraceTests(unittest.TestCase):
    def setUp(self):
        reset_stop(self)

    def run_task(self, script, browser=None, task="Do the thing.", **over):
        agent, fake, folder = make_agent(self, script=script, browser=browser, **over)
        with quiet():
            try:
                result = agent.run(task)
            except Exception as e:
                result = e
        return agent, fake, folder, result

    def test_every_line_follows_the_schema(self):
        script = [
            {"reasoning": SECRET_REASONING, "content": "Typing now.",
             "tool_calls": [call("write", element="search box", x=300, y=200, text=SECRET_TEXT, press_enter=True)]},
            {"reasoning": "Next.", "tool_calls": [call("go_to_web", url=f"example.test/find?q={SECRET_QUERY}")]},
            {"reasoning": "Find.", "tool_calls": [call("ctrl_f_web", query=SECRET_QUERY)]},
            {"reasoning": "Done.", "tool_calls": [call("answer", content=SECRET_ANSWER)]},
        ]
        agent, fake, folder, result = self.run_task(script)
        self.assertEqual(result, SECRET_ANSWER)
        lines = read_jsonl(steps_file(folder))
        self.assertEqual([l["kind"] for l in lines], ["step"] * 4 + ["task"])
        for line in lines[:-1]:
            check_step_line(self, line)
        check_task_line(self, lines[-1])
        task_id = agent.last["task_id"]
        self.assertTrue(all(l["task_id"] == task_id for l in lines))
        self.assertEqual({l["run_id"] for l in lines}, {AGENT.RUN_ID})
        self.assertEqual([l["step"] for l in lines[:-1]], [1, 2, 3, 4])
        self.assertEqual([l["tool"] for l in lines[:-1]], ["write", "go_to_web", "ctrl_f_web", "answer"])
        self.assertEqual(lines[-1]["outcome"], "answer")
        self.assertEqual(lines[-1]["steps"], 4)
        self.assertEqual(agent.last["outcome"], "answer")

        first = lines[0]
        self.assertEqual(first["url"], "https://example.test/search")
        self.assertEqual((first["png_bytes"], first["png_w"], first["png_h"]), (len(helpers.make_png(64, 40)), 64, 40))
        self.assertEqual([l["images_in_prompt"] for l in lines[:-1]], [1, 2, 3, 3])
        self.assertEqual([l["evicted"] for l in lines[:-1]], [False, False, False, True])
        self.assertFalse(any(l["trimmed"] for l in lines[:-1]))
        self.assertEqual(first["reasoning_chars"], len(SECRET_REASONING))
        self.assertEqual(first["finish_reason"], "tool_calls")
        self.assertIsNone(first["error"])
        self.assertEqual(first["t_settle"], 0.01)
        self.assertIsNotNone(first["t_execute"])
        self.assertIsNone(lines[3]["t_execute"])  # answer: nothing executed
        # The server's timings, step by step, as the fake reported them.
        timings = [r for r in fake.requests if r["path"] == "/v1/chat/completions"]
        self.assertEqual(len(timings), 4)
        self.assertEqual(first["cache_n"], 0)
        self.assertGreater(first["prompt_n"], 0)
        self.assertGreater(lines[1]["cache_n"], 0)  # the second request shares the first one's prefix
        for line in lines[:-1]:
            for key in ("prompt_ms", "predicted_n", "predicted_ms"):
                self.assertIsNotNone(line[key], key)

    def test_typed_text_answers_and_reasoning_never_reach_the_trace(self):
        script = [
            {"reasoning": SECRET_REASONING, "content": f"I will type {SECRET_TEXT}",
             "tool_calls": [call("write", element="search box", x=300, y=200, text=SECRET_TEXT)]},
            {"reasoning": SECRET_REASONING, "tool_calls": [call("go_to_web", url=f"https://example.test/a?q={SECRET_QUERY}")]},
            {"reasoning": "x", "tool_calls": [call("select_option", element="list", x=1, y=2, option=SECRET_QUERY)]},
            {"reasoning": "x", "tool_calls": [call("switch_tab_web", new_tab_url=f"example.test/b?t={SECRET_QUERY}")]},
            {"reasoning": "x", "tool_calls": [call("answer", content=SECRET_ANSWER)]},
        ]
        browser = StubBrowser(url=f"https://example.test/search?q={SECRET_QUERY.replace(' ', '+')}")
        agent, fake, folder, result = self.run_task(script, browser=browser)
        text = steps_file(folder).read_text(encoding="utf-8")
        for secret in (SECRET_TEXT, SECRET_QUERY, SECRET_QUERY.replace(" ", "+"), SECRET_ANSWER, SECRET_REASONING,
                       "I will type"):
            self.assertNotIn(secret, text)
        lines = read_jsonl(steps_file(folder))
        self.assertEqual(lines[0]["args_summary"], "write 'search box' at (300, 200) <- \"<20 chars>\"")
        self.assertIn("https://example.test/a", lines[1]["args_summary"])
        self.assertIn(f"<{len(SECRET_QUERY)} chars>", lines[2]["args_summary"])
        self.assertIn("https://example.test/b", lines[3]["args_summary"])
        self.assertEqual(lines[4]["args_summary"], f'answer {{"content": "<{len(SECRET_ANSWER)} chars>"}}')

    def test_trace_url(self):
        u = AGENT.trace_url
        self.assertEqual(u("https://user:pw@Example.test:8443/a/b?q=secret#frag"), "https://example.test:8443/a/b")
        self.assertEqual(u("http://127.0.0.1:8765/search/results.html?q=otters"), "http://127.0.0.1:8765/search/results.html")
        self.assertEqual(u("file:///C:/Windows/win.ini"), "file:///C:/Windows/win.ini")
        self.assertEqual(u("data:text/html,<b>secret</b>"), "data:")
        self.assertEqual(u("javascript:alert('secret')"), "javascript:")
        self.assertEqual(u("about:blank"), "about:blank")
        self.assertEqual(u("chrome://settings/passwords?search=x"), "chrome://settings/passwords")
        self.assertEqual(u("view-source:https://example.test/?q=secret"), "view-source:")
        self.assertIsNone(u(None))
        self.assertIsNone(u(""))
        self.assertIsNone(u("no scheme here"))
        self.assertIsNone(u("http://[broken"))

    def test_errors_cancels_and_step_limits_end_up_in_the_trace(self):
        # An action that fails is a step with an error; the task goes on.
        script = [{"tool_calls": [call("click_web", element="b", x=1, y=2)]},
                  {"tool_calls": [call("answer", content="ok")]}]
        agent, fake, folder, result = self.run_task(script, browser=StubBrowser(fail_on={"click_web"}))
        lines = read_jsonl(steps_file(folder))
        self.assertEqual(lines[0]["error"], "click_web failed on purpose")
        self.assertEqual(lines[-1]["outcome"], "answer")

        # The model server fails: the step and the task say so.
        agent, fake, folder, result = self.run_task([{"fault": "http", "status": 500, "message": "boom"}])
        self.assertIsInstance(result, RuntimeError)
        lines = read_jsonl(steps_file(folder))
        self.assertIn("boom", lines[0]["error"])
        self.assertEqual((lines[-1]["outcome"], lines[-1]["steps"]), ("error", 1))
        self.assertEqual(agent.last["outcome"], "error")

        # The step limit.
        agent, fake, folder, result = self.run_task([{"tool_calls": [call("wait_web", seconds=0)]}] * 2, max_steps=2)
        self.assertIn("I stopped after 2 steps", result)
        lines = read_jsonl(steps_file(folder))
        self.assertEqual((lines[-1]["outcome"], lines[-1]["steps"]), ("max_steps", 2))

        # Ctrl+C (STOP) during an action: the step is marked, the task is "cancelled".
        browser = StubBrowser(execute_delay=0.3)
        agent, fake, folder = make_agent(self, script=[{"tool_calls": [call("wait_web", seconds=1)]}], browser=browser)
        timer = __import__("threading").Timer(0.1, AGENT.STOP.set)
        timer.start()
        with quiet(), self.assertRaises(AGENT.StepCancelled):
            agent.run("x")
        lines = read_jsonl(steps_file(folder))
        self.assertEqual(lines[0]["error"], "cancelled")
        self.assertEqual(lines[-1]["outcome"], "cancelled")
        self.assertEqual(agent.last["outcome"], "cancelled")

    def test_trim_is_recorded(self):
        script = [{"tool_calls": [call("wait_web", seconds=0)], "timings": {"cache_n": 50000, "prompt_n": 100}}] * 4 \
            + [{"tool_calls": [call("answer", content="ok")]}]
        agent, fake, folder, result = self.run_task(script)
        lines = read_jsonl(steps_file(folder))
        self.assertEqual([l["trimmed"] for l in lines[:-1]], [False, False, False, False, True])

    def test_step_trace_off_writes_no_trace(self):
        agent, fake, folder, result = self.run_task([{"tool_calls": [call("answer", content="ok")]}], step_trace=False)
        self.assertEqual(result, "ok")
        self.assertIsNone(steps_file(folder))

    def test_trace_messages_records_the_exact_requests_for_replay(self):
        script = [{"tool_calls": [call("wait_web", seconds=0)]}] * 3 + [{"tool_calls": [call("answer", content="ok")]}]
        captured = []
        real = AGENT.stream_chat

        def spy(base, body, on_event, api_key=None):
            captured.append(json.loads(json.dumps(body)))
            return real(base, body, on_event, api_key=api_key)

        with mock.patch.object(AGENT, "stream_chat", spy):
            agent, fake, folder, result = self.run_task(script, trace_messages=True)
        self.assertEqual(result, "ok")
        task_dir = folder / "logs" / "replay" / agent.last["task_id"]
        tasks = replay.discover(folder / "logs" / "replay")
        self.assertEqual([t[0] for t in tasks], [agent.last["task_id"]])
        self.assertEqual([p.name for p in tasks[0][1]], ["1.json", "2.json", "3.json", "4.json"])
        self.assertEqual(sorted(p.name for p in task_dir.glob("*.png")),
                         ["1-1.png", "2-1.png", "2-2.png", "3-1.png", "3-2.png", "3-3.png", "4-1.png", "4-2.png", "4-3.png"])
        for i, body in enumerate(captured, 1):
            self.assertEqual(replay.load_step(task_dir / f"{i}.json"), body)
        stored = json.loads((task_dir / "2.json").read_text(encoding="utf-8"))
        urls = [p["image_url"]["url"] for m in stored["messages"] if isinstance(m["content"], list)
                for p in m["content"] if p.get("type") == "image_url"]
        self.assertEqual(urls, ["2-1.png", "2-2.png"])
        # The live conversation still has its images (the copy was made for the file only).
        self.assertTrue(captured[1]["messages"][2]["content"][1]["image_url"]["url"].startswith("data:image/png"))

    def test_request_body_sampling_seed_and_max_tokens(self):
        agent, fake, folder, result = self.run_task([{"tool_calls": [call("answer", content="ok")]}])
        body = fake.chat_requests()[-1]["body"]
        self.assertEqual((body["temperature"], body["top_k"], body["top_p"], body["min_p"], body["max_tokens"]),
                         (0.8, 20, 0.95, 0.0, 4096))
        self.assertNotIn("seed", body)
        self.assertEqual(body["tool_choice"], "required")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
        agent, fake, folder, result = self.run_task([{"tool_calls": [call("answer", content="ok")]}],
                                                    seed=7, max_tokens=0, top_k=40)
        body = fake.chat_requests()[-1]["body"]
        self.assertEqual((body["seed"], body["top_k"]), (7, 40))
        self.assertNotIn("max_tokens", body)


class ConsoleLogTests(unittest.TestCase):
    def test_tee_writes_plain_finished_lines(self):
        folder = helpers.temp_dir(self)
        log = AGENT._ConsoleLogFile(folder / "agent-x.log")
        screen = io.StringIO()
        tee = AGENT._Tee(screen, log)
        tee.write(AGENT.c("  step 1  (12s)", "90") + "\n")
        for i in range(3):
            tee.write(f"\r  step 2: thinking ... {i}s  {i} tok   ")
        tee.write("\r" + " " * 70 + "\r")
        tee.write(AGENT.c("Answer:", "1;35") + " forty-two\nsecond line\n")
        tee.write("  loading ... 1s ")
        tee.write("\r  loading ... 2s ")
        tee.write("\n")
        tee.flush()
        log.f.close()
        self.assertEqual((folder / "agent-x.log").read_text(encoding="utf-8").splitlines(),
                         ["  step 1  (12s)", "Answer: forty-two", "second line", "  loading ... 2s"])
        self.assertIn("\x1b[90m", screen.getvalue())  # the console itself is unchanged
        self.assertIn("thinking ... 1s", screen.getvalue())

    def test_console_log_holds_prints_warnings_and_tracebacks(self):
        folder = helpers.temp_dir(self)
        code = (
            "import sys, threading\n"
            f"sys.path.insert(0, {str(helpers.AGENT_DIR)!r})\n"
            "import ghost_agent as g\n"
            f"g.set_data_dir({str(folder)!r})\n"
            "path = g.install_console_log()\n"
            "print(g.c('coloured hello', '1;36'))\n"
            "g.warn('a warning')\n"
            "g.log_only('Task> only in the log')\n"
            "t = threading.Thread(target=lambda: 1 / 0); t.start(); t.join()\n"
            "raise ValueError('the end')\n")
        r = helpers.run_python(["-B", "-c", code], encoding="utf-8", errors="replace")
        self.assertNotEqual(r.returncode, 0)
        logs = list((folder / "logs").glob("agent-*.log"))
        self.assertEqual(len(logs), 1)
        self.assertRegex(logs[0].name, r"^agent-\d{8}-\d{6}\.log$")
        text = logs[0].read_text(encoding="utf-8")
        self.assertNotIn("\x1b", text)
        for want in ("coloured hello", "  ! a warning", "Task> only in the log", "ZeroDivisionError",
                     "Traceback", "ValueError: the end"):
            self.assertIn(want, text)
        self.assertNotIn("only in the log", r.stdout + r.stderr)

    def test_rotation_keeps_the_newest_twenty_of_each_kind(self):
        folder = helpers.temp_dir(self)
        now = time.time()
        made = {}
        for pattern, fmt in (("llama-server-*.log", "llama-server-202611{:02d}-120000.log"),
                             ("agent-*.log", "agent-202611{:02d}-120000.log"),
                             ("steps-*.jsonl", "steps-202611{:02d}.jsonl")):
            made[pattern] = []
            for i in range(1, 26):
                p = folder / fmt.format(i)
                p.write_text("x", encoding="utf-8")
                os.utime(p, (now - 1000 + i, now - 1000 + i))
                made[pattern].append(p.name)
        for other in ("bench-server-1.jsonl", "results-1.jsonl", "notes.txt"):
            (folder / other).write_text("keep", encoding="utf-8")
        AGENT.prune_logs(folder)
        for pattern, names in made.items():
            left = sorted(p.name for p in folder.glob(pattern))
            self.assertEqual(left, sorted(names[5:]), pattern)
        for other in ("bench-server-1.jsonl", "results-1.jsonl", "notes.txt"):
            self.assertTrue((folder / other).exists())
        AGENT.prune_logs(folder / "missing")  # no logs folder yet: nothing happens


class ServerPlumbingTests(unittest.TestCase):
    DEVICE = {"id": "CUDA0", "name": "NVIDIA GeForce RTX 4090", "total": 24564, "free": 22000, "integrated": False}

    def test_layout_and_vram_emulation(self):
        s = settings()
        args, explicit = AGENT.gpu_layout_args(self.DEVICE, s)
        self.assertTrue(explicit)
        self.assertEqual(args, ["--device", "CUDA0", "-ngl", "99", "--cpu-moe", "--fit", "off", "--moe-cache-mib", "4096"])
        # The benchmark plan's laptop emulation: 7070 MiB free gives the laptop's 1446 MiB expert cache.
        args, explicit = AGENT.gpu_layout_args(self.DEVICE, s, emulate_free_mib=7070)
        self.assertTrue(explicit)
        self.assertEqual(args[-2:], ["--moe-cache-mib", "1446"])
        # Too little for the explicit layout: llama.cpp fits it, leaving the real difference free too.
        args, explicit = AGENT.gpu_layout_args(self.DEVICE, s, emulate_free_mib=5000)
        self.assertFalse(explicit)
        self.assertEqual(args, ["--device", "CUDA0", "--fit-target", str(22000 - 5000 + 1024 + AGENT.VISION_RESERVE_MIB)])
        args, explicit = AGENT.gpu_layout_args(self.DEVICE, s, auto_fit=True, emulate_free_mib=7070)
        self.assertEqual(args[-2:], ["--fit-target", str(22000 - 7070 + 1024 + AGENT.VISION_RESERVE_MIB)])
        args, explicit = AGENT.gpu_layout_args(self.DEVICE, s, auto_fit=True)
        self.assertEqual(args[-2:], ["--fit-target", str(1024 + AGENT.VISION_RESERVE_MIB)])
        # More than the card has: the real figure counts.
        args, _ = AGENT.gpu_layout_args(self.DEVICE, s, auto_fit=True, emulate_free_mib=30000)
        self.assertEqual(args[-2:], ["--fit-target", str(1024 + AGENT.VISION_RESERVE_MIB)])
        # A small cache is left out; a set cache size is used as it is.
        args, _ = AGENT.gpu_layout_args({**self.DEVICE, "free": 5700}, s)
        self.assertNotIn("--moe-cache-mib", args)
        args, _ = AGENT.gpu_layout_args(self.DEVICE, settings(moe_cache_mib=2048), emulate_free_mib=7070)
        self.assertEqual(args[-2:], ["--moe-cache-mib", "2048"])
        self.assertEqual(AGENT.gpu_layout_args(None, s), ([], False))

    def test_affinity_mask_text(self):
        self.assertEqual(AGENT.parse_affinity_mask("ff"), 0xFF)
        self.assertEqual(AGENT.parse_affinity_mask("0xF0F"), 0xF0F)
        self.assertEqual(AGENT.parse_affinity_mask(" 0XfF "), 0xFF)
        for bad in ("0", "zz", "", "-1"):
            with self.assertRaises(ValueError):
                AGENT.parse_affinity_mask(bad)
        with quiet() as out, self.assertRaises(SystemExit) as cm:
            AGENT.parse_args(["--affinity-mask", "nope"])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("not a hex CPU mask", out.getvalue())
        self.assertEqual(AGENT.parse_args(["--affinity-mask", "f0ff"]).affinity_mask, 0xF0FF)

    def test_affinity_mask_on_a_real_child_process(self):
        child = subprocess.Popen([str(helpers.PYTHON), "-I", "-c", "import time; time.sleep(30)"],
                                 creationflags=helpers.CREATE_NO_WINDOW)
        self.addCleanup(child.wait)
        self.addCleanup(child.kill)
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                               ctypes.POINTER(ctypes.c_size_t)]

        def mask_of(proc):
            pm, sm = ctypes.c_size_t(), ctypes.c_size_t()
            self.assertTrue(k32.GetProcessAffinityMask(wintypes.HANDLE(int(proc._handle)), ctypes.byref(pm), ctypes.byref(sm)))
            return pm.value, sm.value

        _, system = mask_of(child)
        self.assertEqual(AGENT.set_process_affinity(child, 0x1), 0x1)
        self.assertEqual(mask_of(child)[0], 0x1)
        if system != (1 << 64) - 1:
            with quiet():
                self.assertEqual(AGENT.set_process_affinity(child, system | (1 << 63)), system)  # extra bits dropped
            if system >> 63 == 0:
                with quiet() as out:
                    self.assertIsNone(AGENT.set_process_affinity(child, 1 << 63))
                self.assertIn("selects none of this PC's CPUs", out.getvalue())

    def start_server(self, test_settings, device=..., emulate=None, mask=None, auto_fit=False):
        """LlamaServer.start with the backend probe, the model files and Popen replaced: returns
        (args, server, affinity calls)."""
        folder = scratch_paths(self)
        exe = folder / "bin" / "cuda13" / "llama-server.exe"
        exe.parent.mkdir(parents=True)
        device = self.DEVICE if device is ... else device
        launched, affinity = [], []

        class Proc:
            _handle = 0

            def poll(self):
                return None

        def popen(args, **kw):
            launched.append((args, kw))
            return Proc()

        server = AGENT.LlamaServer(test_settings)
        server.emulate_free_vram_mib, server.affinity_mask = emulate, mask
        with mock.patch.object(AGENT, "pick_backend", return_value=("cuda13", exe, device)), \
                mock.patch.object(AGENT, "model_files", return_value=[Path("m.gguf"), Path("p.gguf")]), \
                mock.patch.object(AGENT.subprocess, "Popen", popen), \
                mock.patch.object(AGENT, "set_process_affinity", lambda p, m: affinity.append(m) or m), \
                mock.patch.object(AGENT, "port_is_free", return_value=True), quiet():
            server.start(auto_fit=auto_fit)
        self.assertEqual(len(launched), 1)
        args, kw = launched[0]
        self.assertEqual(kw["creationflags"] & helpers.CREATE_NO_WINDOW, helpers.CREATE_NO_WINDOW)
        log = server.log_path.read_text(encoding="utf-8")
        self.assertTrue(log.startswith("command: "))
        return args, server, affinity

    def test_server_start_uses_emulation_and_affinity(self):
        args, server, affinity = self.start_server(settings(), emulate=7070, mask=0xF0FF)
        self.assertIn("--moe-cache-mib", args)
        self.assertEqual(args[args.index("--moe-cache-mib") + 1], "1446")
        self.assertTrue(server.explicit_layout)
        self.assertEqual(affinity, [0xF0FF])
        args, server, affinity = self.start_server(settings(), emulate=5000)
        self.assertEqual(args[args.index("--fit-target") + 1], str(22000 - 5000 + 1024 + AGENT.VISION_RESERVE_MIB))
        self.assertFalse(server.explicit_layout)
        self.assertEqual(affinity, [])

    def test_server_flags_file_lists_every_flag_the_agent_passes(self):
        doc = json.loads((helpers.REPO / "tools" / "server-flags.json").read_text(encoding="utf-8"))
        known = set(doc["required"]) | set(doc["optional"])
        self.assertIn("--list-devices", doc["required"])
        seen = set()
        for kwargs in ({}, {"emulate": 5000}, {"auto_fit": True}, {"device": None},
                       {"device": {**self.DEVICE, "free": 5700}}):
            args, _, _ = self.start_server(settings(), **kwargs)
            flags = {a for a in args[1:] if a.startswith("-") and not re.fullmatch(r"-\d+", a)}
            seen |= flags
            self.assertEqual(flags - known, set(), f"flags missing from tools\\server-flags.json ({kwargs})")
        for flag in doc["required"]:
            if flag != "--list-devices":
                self.assertIn(flag, seen, f"{flag} is listed as required but never passed")
        # extra_server_args are the user's own and go last, unchanged.
        args, _, _ = self.start_server(settings(extra_server_args=["--verbose-prompt"]))
        self.assertEqual(args[-1], "--verbose-prompt")


class CommandLineTests(unittest.TestCase):
    def test_options(self):
        a = AGENT.parse_args([])
        self.assertEqual((a.task, a.tasks_file, a.results, a.browser, a.headless_browser), ([], None, None, None, False))
        a = AGENT.parse_args(["--task", "one", "--task", "two", "--browser", "edge", "--emulate-free-vram-mib", "7070",
                              "--headless-browser", "--data-dir", "x", "--results", "r.jsonl"])
        self.assertEqual((a.task, a.browser, a.emulate_free_vram_mib, a.headless_browser, a.data_dir, a.results),
                         (["one", "two"], "edge", 7070, True, "x", "r.jsonl"))
        with quiet(), self.assertRaises(SystemExit):
            AGENT.parse_args(["--browser", "chrome"])
        with quiet(), self.assertRaises(SystemExit):
            AGENT.parse_args(["--emulate-free-vram-mib", "-5"])

    def test_test_switches_stay_out_of_the_help(self):
        with quiet() as out, self.assertRaises(SystemExit):
            AGENT.parse_args(["--help"])
        text = out.getvalue()
        for shown in ("--task", "--tasks-file", "--results", "--settings-override", "--browser",
                      "--emulate-free-vram-mib", "--affinity-mask"):
            self.assertIn(shown, text)
        self.assertNotIn("--headless-browser", text)
        self.assertNotIn("--data-dir", text)

    def test_settings_override_inline_or_file(self):
        self.assertEqual(AGENT.parse_override('{"max_steps": 4}'), {"max_steps": 4})
        f = helpers.temp_dir(self) / "o.json"
        f.write_text('\ufeff{"seed": 1}', encoding="utf-8")
        self.assertEqual(AGENT.parse_override(str(f)), {"seed": 1})
        self.assertIsNone(AGENT.parse_override(None))
        with self.assertRaises(SystemExit):
            AGENT.parse_override("{not json")
        with self.assertRaises(SystemExit):
            AGENT.parse_override("[1, 2]")

    def test_tasks_from_the_command_line_and_a_file(self):
        folder = helpers.temp_dir(self)
        f = folder / "tasks.json"
        f.write_text(json.dumps([{"name": "a", "task": "Do A", "start_url": "http://127.0.0.1:1/start/a"},
                                 {"task": "Do B"}, "Do C"]), encoding="utf-8")
        tasks = AGENT.load_tasks(AGENT.parse_args(["--task", "Do X", "--tasks-file", str(f)]))
        self.assertEqual([(t["name"], t["task"], t["start_url"]) for t in tasks],
                         [("task-1", "Do X", None), ("a", "Do A", "http://127.0.0.1:1/start/a"),
                          ("task-2", "Do B", None), ("task-3", "Do C", None)])
        # suite.json is a valid tasks file.
        suite = AGENT.load_tasks(AGENT.parse_args(["--tasks-file", str(helpers.SUITE)]))
        self.assertEqual(len(suite), 12)
        self.assertTrue(all(t["start_url"].startswith("http://127.0.0.1:8765/start/") for t in suite))
        f.write_text('[{"name": "x"}]', encoding="utf-8")
        with self.assertRaises(SystemExit):
            AGENT.load_tasks(AGENT.parse_args(["--tasks-file", str(f)]))


if __name__ == "__main__":
    unittest.main()
