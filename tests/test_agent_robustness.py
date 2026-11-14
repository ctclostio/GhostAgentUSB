"""Regression tests for problems found in review of the measurement and safety changes: the console
log under a Ctrl+C, a settings file in the new layout that must not be converted again, unusual
replies from the model server, and unattended runs that must never wait for a key press."""

import builtins
import json
import shutil
import threading
import unittest
import unittest.mock as mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import helpers
from agent_support import AGENT, quiet, read_jsonl, reset_stop, scratch_paths, steps_file

BODY = {"messages": [{"role": "user", "content": "hello"}], "stream": True}


def serve_lines(test, lines):
    """A tiny server whose chat reply is these raw SSE lines (bytes), then the end of the stream."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for line in lines:
                self.wfile.write(line + b"\n\n")

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    test.addCleanup(httpd.server_close)
    test.addCleanup(httpd.shutdown)
    return f"http://127.0.0.1:{httpd.server_address[1]}"


class ConsoleLogReentryTests(unittest.TestCase):
    def test_a_line_written_while_the_log_is_busy_on_the_same_thread_does_not_hang(self):
        # The Ctrl+C handler prints from the main thread, possibly while that thread is inside
        # write_line. Simulated here: the same thread writes while it already holds the lock.
        log = AGENT._ConsoleLogFile(helpers.temp_dir(self) / "agent-test.log")
        self.addCleanup(log.f.close)

        def nested():
            with log.lock:
                log.write_line("from the handler")

        t = threading.Thread(target=nested, daemon=True)
        t.start()
        t.join(5)
        self.assertFalse(t.is_alive(), "writing to the console log from inside a write deadlocked")
        self.assertIn("from the handler", log.path.read_text(encoding="utf-8"))


class NewLayoutIsNotConvertedTests(unittest.TestCase):
    def setUp(self):
        self.dir = helpers.temp_dir(self)
        self.user = self.dir / "settings.json"
        defaults = self.dir / "settings.default.json"
        shutil.copy(AGENT.DEFAULTS_FILE, defaults)
        for name, value in (("SETTINGS_FILE", self.user), ("DEFAULTS_FILE", defaults)):
            p = mock.patch.object(AGENT, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_settings_copied_with_their_notes_stay_as_they_are(self):
        # A user who copied lines from settings.default.json together with their notes, one of them
        # at the value that shipped before (which a later default could differ from).
        doc = {"_note": AGENT.USER_SETTINGS_NOTE,
               "image_budget": 3, "_image_budget": "copied note",
               "browser": "bundled", "_browser": "copied note",
               "backend": "auto", "_backend": "copied note",
               "max_steps": 40, "_max_steps": "copied note"}
        raw = json.dumps(doc, indent=2)
        self.user.write_text(raw, encoding="utf-8")
        self.assertFalse(AGENT.is_legacy_settings(doc))
        with quiet():
            s = AGENT.load_settings(interactive=False)
        self.assertEqual((s["browser"], s["max_steps"], s["image_budget"]), ("bundled", 40, 3))
        self.assertEqual(self.user.read_text(encoding="utf-8"), raw)
        self.assertFalse(list(self.dir.glob("settings.backup*")))

    def test_moe_cache_mib_message_names_auto(self):
        with self.assertRaises(AGENT.BadSetting) as cm:
            AGENT.coerce_setting("moe_cache_mib", "lots", "auto")
        self.assertIn('"auto"', str(cm.exception))
        self.assertEqual(AGENT.coerce_setting("moe_cache_mib", "2048", "auto"), 2048)
        self.assertEqual(AGENT.coerce_setting("moe_cache_mib", 1024.0, "auto"), 1024)


class UnusualStreamTests(unittest.TestCase):
    def setUp(self):
        reset_stop(self)

    def fails_with(self, lines):
        base = serve_lines(self, lines)
        with self.assertRaises(RuntimeError) as cm:
            AGENT.stream_chat(base, BODY, lambda k, v: None)
        return str(cm.exception)

    def test_an_error_event_line(self):
        text = self.fails_with([b'data: {"choices": [{"index": 0, "delta": {"content": "a"}}]}',
                                b'error: {"error": {"code": 500, "message": "the context is full"}}'])
        self.assertEqual(text, "The model server reported an error: the context is full")

    def test_a_chunk_that_is_json_but_not_an_object(self):
        text = self.fails_with([b"data: 5"])
        self.assertEqual(text, "The model server sent an unreadable reply: 5")
        text = self.fails_with([b'data: ["a"]'])
        self.assertTrue(text.startswith("The model server sent an unreadable reply"), text)


class TraceErrorTests(unittest.TestCase):
    def test_addresses_in_an_error_lose_their_query(self):
        scratch_paths(self)
        trace = AGENT.Trace({"step_trace": True})
        rec = AGENT.Trace.new_step("t1", 1)
        rec["error"] = ("Page.goto: net::ERR_NAME_NOT_RESOLVED at https://search.example/find?q=private+words#x "
                        "(from http://a.example/b?token=1 too)")
        trace.step(rec)
        line = read_jsonl(steps_file(AGENT.LOGS.parent))[-1]
        self.assertEqual(line["error"], "Page.goto: net::ERR_NAME_NOT_RESOLVED at https://search.example/find "
                                        "(from http://a.example/b too)")
        self.assertNotIn("private", json.dumps(line))
        self.assertIsNone(AGENT.trace_error(None))
        self.assertEqual(AGENT.trace_error("cancelled"), "cancelled")


class UnattendedNeverWaitsTests(unittest.TestCase):
    def test_a_mistyped_option_in_an_unattended_command_does_not_wait_for_enter(self):
        asked = []
        with quiet() as out, mock.patch.object(AGENT, "HEADLESS", False), \
                mock.patch.object(builtins, "input", lambda prompt="": asked.append(prompt) or ""):
            code = AGENT.run_main(["--tasks-file", "tasks.json", "--reslts", "out.jsonl"])
        self.assertEqual(code, 2)
        self.assertEqual(asked, [])
        self.assertIn("--reslts", out.getvalue())

    def test_a_mistyped_option_in_an_interactive_start_still_waits(self):
        asked = []
        with quiet(), mock.patch.object(AGENT, "HEADLESS", False), \
                mock.patch.object(builtins, "input", lambda prompt="": asked.append(prompt) or ""):
            code = AGENT.run_main(["--browsr", "bundled"])
        self.assertEqual(code, 2)
        self.assertEqual(len(asked), 1)


if __name__ == "__main__":
    unittest.main()
