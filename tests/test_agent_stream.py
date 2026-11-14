"""CS-10: the model request and its stream. Errors from the server reach the user as one readable
line; replies without a usable tool call never reach the history; local addresses never go through
a proxy. Everything runs against tools\\bench\\fake_llama.py."""

import json
import os
import threading
import time
import unittest
import unittest.mock as mock
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import helpers
from agent_support import (AGENT, StubBrowser, call, make_agent, quiet, read_jsonl, reset_stop, settings,
                           steps_file)

BODY = {"messages": [{"role": "user", "content": "hello"}], "stream": True, "return_progress": True}


def chat(base, body=None, events=None):
    return AGENT.stream_chat(base, body or BODY, (lambda k, v: events.append((k, v))) if events is not None
                             else (lambda k, v: None))


class StreamErrorTests(unittest.TestCase):
    def setUp(self):
        reset_stop(self)

    def fails_with(self, reply, body=None):
        fake = helpers.start_fake_llama(self, script=[reply])
        with self.assertRaises(RuntimeError) as cm:
            chat(fake.base, body)
        return str(cm.exception)

    def test_an_error_chunk_mid_stream(self):
        text = self.fails_with({"reasoning": "a b c d e f g h", "content": "x", "fault": "error_chunk",
                                "message": "the context is full", "fault_after": 2})
        self.assertEqual(text, "The model server reported an error: the context is full")

    def test_a_stream_that_ends_without_done(self):
        text = self.fails_with({"reasoning": "a b c d e f g h", "fault": "close"})
        self.assertEqual(text, "The model server closed the stream early.")

    def test_a_connection_cut_mid_reply(self):
        text = self.fails_with({"reasoning": "a b c d e f g h", "fault": "drop"})
        # Seen from the client, a cut chunked stream usually just ends; either message is right.
        self.assertIn(text, ("The model server closed the stream early.",
                             "The connection to the model server broke off in the middle of a reply."))

    def test_http_errors_show_only_the_message(self):
        text = self.fails_with({"fault": "http", "status": 400, "message": "the request exceeds the available context"})
        self.assertEqual(text, "The model server answered HTTP 400: the request exceeds the available context")
        self.assertNotIn("{", text)
        # Not JSON at all: the text itself, on one line.
        self.assertEqual(AGENT.server_error_message(b"Bad gateway\r\n  from a proxy"), "Bad gateway from a proxy")
        self.assertEqual(AGENT.server_error_message({"error": "plain"}), "plain")
        self.assertEqual(AGENT.server_error_message(b""), "no details")

    def test_nothing_listening(self):
        port = helpers.free_port()
        t0 = time.monotonic()
        with self.assertRaises(RuntimeError) as cm:
            chat(f"http://127.0.0.1:{port}")
        self.assertLess(time.monotonic() - t0, 5)
        self.assertTrue(str(cm.exception).startswith(f"Could not talk to the model server at http://127.0.0.1:{port} ("))

    def test_an_unreadable_chunk(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: {not json at all\n\n")

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        with self.assertRaises(RuntimeError) as cm:
            chat(f"http://127.0.0.1:{httpd.server_address[1]}")
        self.assertEqual(str(cm.exception), "The model server sent an unreadable reply: {not json at all")

    def test_finish_reason_and_progress(self):
        fake = helpers.start_fake_llama(self, script=[
            {"reasoning": "Looking.", "tool_calls": [{"name": "answer", "arguments": {"content": "hi"}}]},
            {"content": "just text"},
            {"reasoning": "word " * 50, "tool_calls": [{"name": "answer", "arguments": {"content": "hi"}}]},
        ])
        events = []
        msg = chat(fake.base, events=events)
        self.assertEqual(msg["finish_reason"], "tool_calls")
        self.assertEqual(msg["tool_calls"][0]["function"], {"name": "answer", "arguments": '{"content": "hi"}'})
        self.assertEqual(msg["reasoning"], "Looking.")
        self.assertTrue(any(k == "prompt" for k, v in events))
        self.assertTrue(msg["timings"]["prompt_n"] > 0)
        self.assertEqual(chat(fake.base)["finish_reason"], "stop")
        msg = chat(fake.base, {**BODY, "max_tokens": 10})
        self.assertEqual(msg["finish_reason"], "length")
        self.assertEqual(msg["tool_calls"], [])

    def test_the_api_key_is_sent(self):
        fake = helpers.start_fake_llama(self, script=[{"content": "ok"}, {"content": "ok"}], api_key="k-123")
        self.assertEqual(AGENT.stream_chat(fake.base, BODY, lambda k, v: None, api_key="k-123")["content"], "ok")
        with self.assertRaises(RuntimeError) as cm:
            chat(fake.base)
        self.assertEqual(str(cm.exception), "The model server answered HTTP 401: Invalid API Key")


class ProxyTests(unittest.TestCase):
    def setUp(self):
        reset_stop(self)

    def test_local_requests_ignore_a_proxy_in_the_environment(self):
        fake = helpers.start_fake_llama(self, script=[{"content": "through"}])
        env = {k: v for k, v in os.environ.items() if k.upper() not in ("NO_PROXY",)}
        env.update({"HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9"})
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertNotIn("NO_PROXY", os.environ)
            # What a plain urllib request would do here: go to the dead proxy and fail.
            with self.assertRaises(OSError):
                urllib.request.build_opener().open(fake.base + "/health", timeout=3)
            self.assertTrue(AGENT.server_healthy(fake.base))
            self.assertTrue(AGENT.ExternalServer(fake.base).alive())
            self.assertEqual(chat(fake.base)["content"], "through")
        self.assertTrue(all(r["path"] in ("/health", "/v1/chat/completions") for r in fake.requests))

    def test_no_proxy_gets_the_local_addresses(self):
        with mock.patch.dict(os.environ, {"NO_PROXY": "corp.example, 127.0.0.1"}):
            AGENT._bypass_proxy_for_localhost()
            self.assertEqual(os.environ["NO_PROXY"], "corp.example,127.0.0.1,localhost")
        env = {k: v for k, v in os.environ.items() if k.upper() != "NO_PROXY"}
        with mock.patch.dict(os.environ, env, clear=True):
            AGENT._bypass_proxy_for_localhost()
            self.assertEqual(os.environ["NO_PROXY"], "127.0.0.1,localhost")


class ReplyHandlingTests(unittest.TestCase):
    """Agent.run when a reply has no usable tool call."""

    def setUp(self):
        reset_stop(self)

    def run_agent(self, script, **over):
        agent, fake, folder = make_agent(self, script=script, **over)
        with quiet():
            try:
                result = agent.run("Find it.")
            except Exception as e:
                result = e
        return agent, fake, folder, result

    @staticmethod
    def after_reply(request):
        """The last assistant message of a request and the message after it (the note to the model);
        the request then ends with the next observation."""
        msgs = request["body"]["messages"]
        i = max(n for n, m in enumerate(msgs) if m["role"] == "assistant")
        return [{k: m.get(k) for k in ("role", "content", "tool_calls")} for m in msgs[i:i + 2]]

    def test_malformed_arguments_are_kept_out_of_the_history(self):
        script = [{"content": "Clicking.", "tool_calls": [{"name": "click_web", "arguments": '{"x": 1'}]},
                  {"tool_calls": [call("answer", content="found")]}]
        agent, fake, folder, result = self.run_agent(script)
        # fake_llama, like llama-server, fails any request whose history holds unparsable arguments.
        self.assertEqual(result, "found")
        second = fake.chat_requests()[1]
        reply, note = self.after_reply(second)
        self.assertEqual(reply, {"role": "assistant", "content": "Clicking.", "tool_calls": None})
        self.assertEqual(note["role"], "user")
        self.assertTrue(note["content"].startswith("Your tool call could not be parsed ("))
        self.assertTrue(note["content"].endswith("). Call a tool with valid JSON arguments."))
        self.assertEqual(second["body"]["messages"][-1]["role"], "user")  # then the next observation
        for m in second["body"]["messages"]:
            self.assertNotIn("tool_calls", m)
        lines = read_jsonl(steps_file(folder))
        self.assertTrue(lines[0]["error"].startswith("the tool call could not be parsed"))
        self.assertEqual(lines[0]["tool"], "click_web")
        self.assertIsNone(lines[0]["args_summary"])

    def test_arguments_that_are_not_an_object(self):
        script = [{"tool_calls": [{"name": "click_web", "arguments": "[1, 2]"}]},
                  {"tool_calls": [call("answer", content="ok")]}]
        agent, fake, folder, result = self.run_agent(script, browser=StubBrowser())
        self.assertEqual(result, "ok")
        reply, note = self.after_reply(fake.chat_requests()[1])
        self.assertIn("the arguments are not a JSON object", note["content"])

    def test_a_cut_off_reply_asks_for_a_short_one(self):
        script = [{"reasoning": "hmm " * 200, "tool_calls": [call("click_web", element="b", x=1, y=2)]},
                  {"tool_calls": [call("answer", content="ok")]}]
        agent, fake, folder, result = self.run_agent(script, max_tokens=40)
        self.assertEqual(result, "ok")
        first, second = fake.chat_requests()
        self.assertEqual(first["body"]["max_tokens"], 40)
        self.assertEqual(self.after_reply(second)[1], {"role": "user", "content": AGENT.CUT_OFF_NOTE, "tool_calls": None})
        lines = read_jsonl(steps_file(folder))
        self.assertEqual(lines[0]["finish_reason"], "length")
        self.assertEqual(lines[0]["error"], "the reply was cut off (max_tokens)")

    def test_a_reply_without_any_tool_call(self):
        script = [{"content": "I think it is done."}, {"tool_calls": [call("answer", content="ok")]}]
        agent, fake, folder, result = self.run_agent(script)
        self.assertEqual(result, "ok")
        self.assertEqual(self.after_reply(fake.chat_requests()[1]),
                         [{"role": "assistant", "content": "I think it is done.", "tool_calls": None},
                          {"role": "user", "content": AGENT.NO_TOOL_NOTE, "tool_calls": None}])

    def test_three_failed_replies_in_a_row_end_the_task(self):
        bad = {"tool_calls": [{"name": "click_web", "arguments": '{"x": '}]}
        cut = {"reasoning": "x", "finish_reason": "length"}
        agent, fake, folder, result = self.run_agent([bad, cut, bad, {"tool_calls": [call("answer", content="never")]}])
        self.assertIsInstance(result, RuntimeError)
        self.assertTrue(str(result).startswith("The model's last 3 replies had no usable tool call"))
        self.assertEqual(len(fake.chat_requests()), 3)
        lines = read_jsonl(steps_file(folder))
        self.assertEqual((lines[-1]["outcome"], lines[-1]["steps"]), ("error", 3))
        self.assertEqual(agent.last["outcome"], "error")

    def test_a_good_reply_resets_the_count(self):
        bad = {"tool_calls": [{"name": "click_web", "arguments": "nope"}]}
        good = {"tool_calls": [call("wait_web", seconds=0)]}
        agent, fake, folder, result = self.run_agent([bad, bad, good, bad, bad, {"tool_calls": [call("answer", content="ok")]}])
        self.assertEqual(result, "ok")
        self.assertEqual(len(fake.chat_requests()), 6)

    def test_the_request_carries_explicit_sampling(self):
        agent, fake, folder, result = self.run_agent([{"tool_calls": [call("answer", content="ok")]}],
                                                     top_k=30, top_p=0.9, min_p=0.05, max_tokens=2048)
        body = fake.chat_requests()[0]["body"]
        self.assertEqual({k: body[k] for k in ("top_k", "top_p", "min_p", "max_tokens", "stream")},
                         {"top_k": 30, "top_p": 0.9, "min_p": 0.05, "max_tokens": 2048, "stream": True})


if __name__ == "__main__":
    unittest.main()
