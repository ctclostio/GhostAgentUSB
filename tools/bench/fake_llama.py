"""A stand-in for llama.cpp's llama-server, for tests and benchmark plumbing. No model, no GPU.

It answers with the same JSON and SSE shapes llama-server uses, so GhostAgent's client code can
be tested end to end without loading a 20 GB model. What it "generates" comes from a script.

Command line (with the stick's Python):
    python\\python.exe -I tools\\bench\\fake_llama.py --port 18931 --script replies.json
        --record requests.jsonl --api-key KEY --log fake-server.log

In-process (unit tests):
    import fake_llama
    fake = fake_llama.start(script=[{"content": "hi"}], record="requests.jsonl")
    ... fake.port, fake.base, fake.requests ...
    fake.stop()

Endpoints: GET /health, GET /props, GET /v1/models, POST /tokenize (rough), and
POST /v1/chat/completions (streaming SSE or a plain JSON reply).

THE SCRIPT is a JSON list of replies, used one per chat request, in order:
    {
      "reasoning": "streamed as delta.reasoning_content",
      "content": "streamed as delta.content",
      "tool_calls": [{"name": "click_web", "arguments": "{\\"x\\": 1"}],
                       # arguments is the raw string the model "wrote" (it may be malformed),
                       # or an object, which is sent as its JSON text
      "finish_reason": "tool_calls" | "stop" | "length",
                       # default: tool_calls when there are calls, otherwise stop. A request's
                       # max_tokens also cuts the reply short and gives "length", as llama.cpp does.
      "timings": {...},        # values that replace the simulated timings
      "delay": 0.0,            # seconds before anything is sent
      "chunk_delay": 0.0,      # seconds between streamed chunks
      "fault": "error_chunk" | "close" | "drop" | "http",
          # error_chunk: an {"error": {...}} chunk mid-stream, then the stream ends (no [DONE])
          # close:       the stream ends cleanly mid-reply, without a final chunk or [DONE]
          # drop:        the connection is cut mid-reply (a broken chunked stream)
          # http:        an HTTP error status with llama.cpp's JSON error body, no stream
      "fault_after": 3,        # delta chunks sent before the fault (default: half of them)
      "status": 500,           # for "http"
      "message": "...",        # error text for "http" and "error_chunk"
      "log": ["line", ...]     # extra lines for the fake server log (for example "making room")
    }
When the script is used up, a chat request gets HTTP 500 ("the fake model's script is used up"),
unless loop is on, which starts the script again. With no script at all, every request gets a
short reply that calls the answer tool.

Requests: every request (method, path, headers, JSON body) is kept in memory and, with --record,
appended to a JSONL file. Images in the body are replaced by a marker such as
"<image/png 48213 bytes 1280x800>", so the file stays small; images_in() reads the markers back.

Validation: like the real server, a chat request whose history holds a tool call with arguments
that are not a JSON object fails with HTTP 500 (llama.cpp's chat template has to parse them).
validate="strict" also enforces OpenAI's pairing rules: every assistant tool call is answered by a
tool message with the same id before the conversation moves on. validate="off" checks nothing.

Timings are simulated: prompt tokens are rough (4 characters of text = 1 token, an image = about
one token per 32x32 pixel patch and at least 1024), and cache_n is the longest common prefix with
the previous request, the way a single llama.cpp slot reuses its cache.

--api-key KEY (or the LLAMA_API_KEY environment variable, as llama-server reads it): everything
except /health needs "Authorization: Bearer KEY" (or "X-Api-Key: KEY"); anything else gets
llama.cpp's 401 JSON error.

To stand in for llama-server itself, the command line ignores llama-server's other flags (-m,
--mmproj, -ngl ...), "--log -" writes the log lines to stdout (where the agent collects
llama-server's log), and --exit-after N makes the process exit after N chat requests, like a crash.
"""

import argparse
import base64
import json
import os
import re
import socket
import socketserver
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PUBLIC_PATHS = {"/health", "/v1/health"}
DEFAULT_REPLY = {"reasoning": "The task looks done.", "content": "",
                 "tool_calls": [{"name": "answer", "arguments": {"content": "Done (fake model)."}}]}
IMAGE_MARKER_RE = re.compile(r"^<(image/[\w.+-]+|image) (\d+) bytes(?: (\d+)x(\d+))?>$")
ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error", 404: "not_found_error",
               501: "not_supported_error", 503: "unavailable_error"}


# --------------------------------------------------------------------------------------------
# Helpers shared with tests and replay.py
# --------------------------------------------------------------------------------------------

def png_size(data):
    """(width, height) of PNG bytes, or None for anything else."""
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    return None


def decode_data_url(url):
    """(mime, bytes) for a base64 data: URL, or None."""
    if not isinstance(url, str) or not url.startswith("data:") or ";base64," not in url[:100]:
        return None
    head, _, payload = url.partition(",")
    try:
        return head[5:].split(";")[0] or "application/octet-stream", base64.b64decode(payload)
    except Exception:
        return None


def image_marker(url):
    decoded = decode_data_url(url)
    if not decoded:
        return url
    mime, data = decoded
    size = png_size(data)
    dims = f" {size[0]}x{size[1]}" if size else ""
    return f"<{mime} {len(data)} bytes{dims}>"


def scrub_images(obj):
    """A copy of a JSON value with every base64 data: URL replaced by an image marker."""
    if isinstance(obj, dict):
        return {k: scrub_images(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_images(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("data:"):
        return image_marker(obj)
    return obj


def images_in(obj):
    """Every image marker in a recorded body, as dicts {mime, bytes, width, height}."""
    found = []

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            m = IMAGE_MARKER_RE.match(o)
            if m:
                found.append({"mime": m.group(1), "bytes": int(m.group(2)),
                              "width": int(m.group(3)) if m.group(3) else None,
                              "height": int(m.group(4)) if m.group(4) else None})
    walk(obj)
    return found


def text_tokens(text):
    """Rough tokens: one per 4 characters, each a hash of its characters."""
    data = text.encode("utf-8", "replace")
    return [zlib.crc32(data[i:i + 4]) % 151000 for i in range(0, len(data), 4)]


def prompt_tokens(body):
    """A rough token list for a chat request, in template order: tools, then each message."""
    return _tokens_and_images(body)[0]


def _tokens_and_images(body):
    """(tokens, the token position where each image starts)."""
    toks = []
    image_starts = []
    if body.get("tools"):
        toks += text_tokens(json.dumps(body["tools"], sort_keys=True))
    for m in body.get("messages") or []:
        toks += text_tokens("<|im_start|>" + str(m.get("role")) + "\n")
        content = m.get("content")
        parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                decoded = decode_data_url(url)
                data = decoded[1] if decoded else str(url).encode()
                size = png_size(data)
                n = max(1024, (size[0] // 32) * (size[1] // 32)) if size else 1024
                h = zlib.crc32(data)
                image_starts.append(len(toks))
                toks += [(h + i) % 151000 for i in range(n)]
            else:
                toks += text_tokens(str(part.get("text") or ""))
        for tc in m.get("tool_calls") or []:
            f = tc.get("function") or {}
            toks += text_tokens(str(f.get("name")) + str(f.get("arguments")))
        toks += text_tokens("<|im_end|>\n")
    toks += text_tokens("<|im_start|>assistant\n")
    return toks, image_starts


def common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def check_messages(messages, mode):
    """Raise ValueError(status, message) the way llama.cpp (and, when strict, OpenAI) would."""
    if mode == "off":
        return
    if not isinstance(messages, list) or not messages:
        raise ValueError(400, "'messages' is required")
    open_calls = set()
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant", "tool"):
            raise ValueError(400, f"message {i} has no valid role")
        role = m["role"]
        if "content" not in m and not m.get("tool_calls"):
            raise ValueError(400, "Expected 'content' (ref: https://github.com/ggml-org/llama.cpp/issues/8367)")
        if role == "system" and i != 0:
            raise ValueError(500, "System message must be at the beginning.")
        for tc in m.get("tool_calls") or []:
            args = (tc.get("function") or {}).get("arguments")
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except ValueError as e:
                    raise ValueError(500, f"Failed to parse tool call arguments as JSON: {e}")
                if not isinstance(parsed, dict):
                    raise ValueError(500, "Tool call arguments must be a JSON object")
        if mode != "strict":
            continue
        if role == "tool":
            if m.get("tool_call_id") not in open_calls:
                raise ValueError(400, f"message {i}: a tool message must answer a preceding tool call "
                                      f"(unknown tool_call_id {m.get('tool_call_id')!r})")
            open_calls.discard(m.get("tool_call_id"))
        else:
            if open_calls:
                raise ValueError(400, f"message {i}: tool calls {sorted(open_calls)} were not answered "
                                      f"by tool messages")
            if role == "assistant":
                open_calls = {tc.get("id") for tc in m.get("tool_calls") or []}
    if mode == "strict" and open_calls:
        raise ValueError(400, f"tool calls {sorted(open_calls)} were not answered by tool messages")


def error_body(status, message):
    return {"error": {"code": status, "message": message, "type": ERROR_TYPES.get(status, "server_error")}}


# --------------------------------------------------------------------------------------------
# The server
# --------------------------------------------------------------------------------------------

class FakeLlama:
    """The fake server's state. start() returns one of these, already serving."""

    def __init__(self, script=None, record=None, api_key=None, loop=False, log=None, alias="holo4-35b-a3b",
                 ctx_size=65536, loading_s=0.0, validate="llama", prompt_tps=1500.0, gen_tps=25.0,
                 chunk_chars=4):
        self.script = list(script) if script is not None else None
        self.loop = loop or script is None
        self.record_path = record
        self.api_key = api_key
        self.log_path = log
        self.alias = alias
        self.ctx_size = ctx_size
        # The monotonic clock: a wall clock set back would make a fresh fake answer "Loading model".
        self.loading_until = time.monotonic() + float(loading_s)
        self.validate = validate
        self.prompt_tps = prompt_tps
        self.gen_tps = gen_tps
        self.chunk_chars = max(1, int(chunk_chars))
        self.requests = []
        self.chats_done = 0  # chat requests fully answered (or failed); --exit-after counts these
        self.chats_changed = threading.Condition()  # notified when chats_done goes up (see wait_for_chats)
        self.next_reply = 0
        self.task_id = 0
        self.slot_tokens = []
        self.lock = threading.Lock()
        self.chat_lock = threading.Lock()  # one slot: chat requests are served one at a time
        self.stopping = threading.Event()
        self.httpd = None
        self.thread = None
        self.port = None

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def set_script(self, script, loop=False):
        with self.lock:
            self.script = list(script)
            self.loop = loop
            self.next_reply = 0

    def reset_cache(self):
        self.slot_tokens = []

    def wait_for_chats(self, n, timeout=30.0):
        """Wait until n chat requests are done; returns chats_done then. A chat is counted just after
        its last byte has gone out, so a client that has its whole reply can still be a moment early:
        this waits for the count itself, not for a time that is usually long enough."""
        with self.chats_changed:
            self.chats_changed.wait_for(lambda: self.chats_done >= n, timeout)
            return self.chats_done

    def chat_requests(self):
        return [r for r in self.requests if r["path"] in ("/v1/chat/completions", "/chat/completions")]

    def take_reply(self):
        with self.lock:
            script = self.script if self.script is not None else [DEFAULT_REPLY]
            if self.next_reply >= len(script):
                if not self.loop or not script:
                    return None, None
                self.next_reply = 0
            index = self.next_reply
            self.next_reply += 1
            return index, dict(script[index])

    def record(self, entry):
        with self.lock:
            entry["n"] = len(self.requests) + 1
            self.requests.append(entry)
            if self.record_path:
                with open(self.record_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log(self, *lines):
        if not self.log_path:
            return
        with self.lock:
            if self.log_path == "-":  # standing in for llama-server, whose log is its stdout
                print("\n".join(lines), flush=True)
                return
            with open(self.log_path, "a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")

    def slot_line(self, func, text):
        return f"slot {func:>12.12}: id  0 | task {self.task_id} | {text}"

    def stop(self):
        self.stopping.set()
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread:
            self.thread.join(5)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "llama.cpp"
    sys_version = ""

    def log_message(self, *args):
        pass

    @property
    def fake(self):
        return self.server.fake

    # ---- plumbing ----

    def read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            return raw, (json.loads(raw) if raw else None)
        except ValueError:
            return raw, None

    def send_json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        if not self.fake.api_key or self.path.split("?")[0] in PUBLIC_PATHS:
            return True
        given = self.headers.get("Authorization", "")
        if given.startswith("Bearer "):
            given = given[7:]
        given = given or self.headers.get("X-Api-Key", "")
        return given == self.fake.api_key

    def handle_any(self, method):
        raw, body = self.read_body() if method == "POST" else (b"", None)
        path = self.path.split("?")[0]
        self.entry = {"ts": round(time.time(), 3), "method": method, "path": path,
                      "headers": dict(self.headers.items()),
                      "body": scrub_images(body) if body is not None else None, "body_bytes": len(raw)}
        self.fake.record(self.entry)
        if not self.authorized():
            return self.send_json(401, error_body(401, "Invalid API Key"))
        loading = time.monotonic() < self.fake.loading_until
        if path in ("/health", "/v1/health"):
            if loading:
                return self.send_json(503, error_body(503, "Loading model"))
            return self.send_json(200, {"status": "ok"})
        if loading:
            return self.send_json(503, error_body(503, "Loading model"))
        if method == "GET" and path == "/props":
            return self.send_json(200, {
                "default_generation_settings": {"n_ctx": self.fake.ctx_size, "params": {"temperature": 0.8}},
                "total_slots": 1, "model_alias": self.fake.alias, "model_path": "fake-model.gguf",
                "modalities": {"vision": True, "audio": False}, "chat_template": "(fake)",
                "build_info": "b0-fake_llama", "is_sleeping": False})
        if method == "GET" and path in ("/v1/models", "/models"):
            return self.send_json(200, {"object": "list", "data": [
                {"id": self.fake.alias, "object": "model", "created": int(time.time()), "owned_by": "llamacpp"}]})
        if method == "POST" and path == "/tokenize":
            if not isinstance(body, dict):
                return self.send_json(400, error_body(400, "invalid JSON body"))
            return self.send_json(200, {"tokens": text_tokens(str(body.get("content", "")))})
        if method == "POST" and path in ("/v1/chat/completions", "/chat/completions"):
            if not isinstance(body, dict):
                return self.send_json(400, error_body(400, "invalid JSON body"))
            with self.fake.chat_lock:
                try:
                    return self.chat(body)
                finally:
                    with self.fake.chats_changed:
                        self.fake.chats_done += 1
                        self.fake.chats_changed.notify_all()
        return self.send_json(404, error_body(404, "File Not Found"))

    def do_GET(self):
        self.handle_any("GET")

    def do_POST(self):
        self.handle_any("POST")

    # ---- chat completions ----

    def chat(self, body):
        fake = self.fake
        try:
            check_messages(body.get("messages"), fake.validate)
        except ValueError as e:
            status, message = e.args
            return self.send_json(status, error_body(status, message))
        index, reply = fake.take_reply()
        self.entry["script_index"] = index  # in memory only; the JSONL line is already written
        if reply is None:
            return self.send_json(500, error_body(500, "the fake model's script is used up"))
        if not self.wait(float(reply.get("delay", 0))):
            return
        if reply.get("fault") == "http":
            status = int(reply.get("status", 500))
            return self.send_json(status, error_body(status, reply.get("message", "fake server error")))

        # Prompt side: what a single llama.cpp slot would reuse and process.
        fake.task_id += 1
        toks, image_starts = _tokens_and_images(body)
        cache_n = min(common_prefix(fake.slot_tokens, toks), max(0, len(toks) - 1))
        prompt_n = len(toks) - cache_n
        n_images = sum(1 for pos in image_starts if pos >= cache_n)  # cached images are not re-encoded

        # Generation side: split the scripted text into "tokens" of a few characters.
        pieces = self.pieces(reply)
        limit = _max_tokens(body)
        finish = reply.get("finish_reason") or ("tool_calls" if reply.get("tool_calls") else "stop")
        if limit and len(pieces) > limit:
            pieces = pieces[:limit]
            finish = "length"
        timings = self.timings(cache_n, prompt_n, len(pieces), reply.get("timings"))
        fake.slot_tokens = toks + text_tokens("".join(str(p[1]) for p in pieces))
        self.write_log(cache_n, len(toks), n_images, timings, reply.get("log") or [])

        if body.get("stream"):
            self.stream(body, reply, pieces, finish, timings, cache_n, len(toks))
        else:
            self.whole(body, reply, pieces, finish, timings, cache_n, prompt_n)

    def wait(self, seconds):
        """Sleep in small slices so stop() is never held up. False when the server is stopping."""
        end = time.time() + seconds
        while time.time() < end:
            if self.fake.stopping.wait(min(0.05, max(0.0, end - time.time()))):
                return False
        return True

    def pieces(self, reply):
        """[(kind, text, call_index)] in llama.cpp's order: reasoning, content, then each tool call."""
        n = self.fake.chunk_chars
        out = []
        for kind in ("reasoning", "content"):
            text = reply.get(kind) or ""
            out += [(kind, text[i:i + n], None) for i in range(0, len(text), n)]
        for ci, call in enumerate(reply.get("tool_calls") or []):
            args = call.get("arguments", "")
            args = args if isinstance(args, str) else json.dumps(args)
            out.append(("call_start", call.get("name", ""), ci))
            out += [("args", args[i:i + n], ci) for i in range(0, len(args), n)]
        return out

    def timings(self, cache_n, prompt_n, predicted_n, override):
        fake = self.fake
        prompt_ms = prompt_n / fake.prompt_tps * 1000.0
        predicted_ms = predicted_n / fake.gen_tps * 1000.0
        t = {
            "cache_n": cache_n, "prompt_n": prompt_n, "prompt_ms": round(prompt_ms, 3),
            "prompt_per_token_ms": round(prompt_ms / prompt_n, 3) if prompt_n else 0.0,
            "prompt_per_second": round(prompt_n / prompt_ms * 1000.0, 3) if prompt_ms else 0.0,
            "predicted_n": predicted_n, "predicted_ms": round(predicted_ms, 3),
            "predicted_per_token_ms": round(predicted_ms / predicted_n, 3) if predicted_n else 0.0,
            "predicted_per_second": round(predicted_n / predicted_ms * 1000.0, 3) if predicted_ms else 0.0,
        }
        t.update(override or {})
        return t

    def write_log(self, cache_n, total, n_images, timings, extra):
        """llama-server's log lines for one request, in the b11669 wording replay.py parses."""
        fake = self.fake
        if not fake.log_path:
            return
        lines = [fake.slot_line("launch_slot_", "processing task"),
                 fake.slot_line("update_slots", f"new prompt, n_ctx_slot = {fake.ctx_size}, n_keep = 0, "
                                                f"task.n_tokens = {total}"),
                 fake.slot_line("update_slots", f"cached n_tokens = {cache_n}, memory_seq_rm [{cache_n}, end)")]
        done = cache_n
        while done < total:
            done = min(total, done + 2048)
            lines.append(fake.slot_line("update_slots", f"prompt processing, n_tokens = {done:6d}, "
                                                        f"progress = {done / total:.2f}, t = {0.5:6.2f} s / "
                                                        f"{fake.prompt_tps:.2f} tokens per second"))
        for _ in range(n_images):
            lines += ["srv  process_chun: processing image...", "image slice encoded in 812 ms",
                      "image decoded (batch 1/1) in 97 ms"]
        lines += list(extra)
        t = timings
        lines += [
            fake.slot_line("print_timing", f"prompt eval time = {t['prompt_ms']:10.2f} ms / {t['prompt_n']:5d} "
                                           f"tokens ({t['prompt_per_token_ms']:8.2f} ms per token, "
                                           f"{t['prompt_per_second']:8.2f} tokens per second)"),
            fake.slot_line("print_timing", f"       eval time = {t['predicted_ms']:10.2f} ms / "
                                           f"{t['predicted_n']:5d} tokens ({t['predicted_per_token_ms']:8.2f} ms "
                                           f"per token, {t['predicted_per_second']:8.2f} tokens per second)"),
            fake.slot_line("print_timing", f"      total time = {t['prompt_ms'] + t['predicted_ms']:10.2f} ms / "
                                           f"{t['prompt_n'] + t['predicted_n']:5d} tokens"),
            fake.slot_line("release", f"stop processing: n_tokens = {total + t['predicted_n']}, truncated = 0"),
            "srv  update_slots: all slots are idle",
        ]
        fake.log(*lines)

    def base_chunk(self, cid):
        return {"created": int(time.time()), "id": cid, "model": self.fake.alias,
                "system_fingerprint": "b0-fake_llama", "object": "chat.completion.chunk"}

    def send_chunk(self, data):
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def send_event(self, obj):
        self.send_chunk(b"data: " + json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode() + b"\n\n")

    def stream(self, body, reply, pieces, finish, timings, cache_n, total):
        fake = self.fake
        cid = f"chatcmpl-fake{fake.task_id:04d}"
        call_ids = [c.get("id") or f"fake{fake.task_id:04d}call{i}" for i, c in enumerate(reply.get("tool_calls") or [])]
        fault = reply.get("fault")
        fault_after = int(reply.get("fault_after", len(pieces) // 2))
        chunk_delay = float(reply.get("chunk_delay", 0))

        def delta_event(delta):
            ev = self.base_chunk(cid)
            ev["choices"] = [{"finish_reason": None, "index": 0, "delta": delta}]
            return ev

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            if body.get("return_progress"):
                done = cache_n
                while True:
                    done = min(total, done + 2048) if done < total else total
                    ev = delta_event({"role": "assistant", "content": None})
                    ev["prompt_progress"] = {"total": total, "cache": cache_n, "processed": done,
                                             "time_ms": round((done - cache_n) / fake.prompt_tps * 1000.0, 3)}
                    self.send_event(ev)
                    if done >= total:
                        break
            self.send_event(delta_event({"role": "assistant", "content": None}))
            for i, (kind, text, ci) in enumerate(pieces):
                if fault in ("error_chunk", "close", "drop") and i >= fault_after:
                    return self.fault(fault, reply)
                if chunk_delay and not self.wait(chunk_delay):
                    return
                if kind == "reasoning":
                    delta = {"reasoning_content": text}
                elif kind == "content":
                    delta = {"content": text}
                elif kind == "call_start":
                    delta = {"tool_calls": [{"index": ci, "id": call_ids[ci], "type": "function",
                                             "function": {"name": text, "arguments": ""}}]}
                else:
                    delta = {"tool_calls": [{"index": ci, "function": {"arguments": text}}]}
                self.send_event(delta_event(delta))
            if fault in ("error_chunk", "close", "drop"):
                return self.fault(fault, reply)
            final = self.base_chunk(cid)
            final["choices"] = [{"finish_reason": finish, "index": 0, "delta": {}}]
            if (body.get("stream_options") or {}).get("include_usage"):
                final["usage"] = {"completion_tokens": timings["predicted_n"],
                                  "prompt_tokens": timings["cache_n"] + timings["prompt_n"],
                                  "total_tokens": timings["cache_n"] + timings["prompt_n"] + timings["predicted_n"]}
            final["timings"] = timings
            self.send_event(final)
            self.send_chunk(b"data: [DONE]\n\n")
            self.send_chunk(b"")
        except (ConnectionError, OSError):
            self.close_connection = True

    def fault(self, fault, reply):
        if fault == "error_chunk":
            self.send_event(error_body(500, reply.get("message", "fake mid-stream error")))
            self.send_chunk(b"")
        elif fault == "close":
            self.send_chunk(b"")
        else:  # drop: cut the connection without ending the chunked stream
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def whole(self, body, reply, pieces, finish, timings, cache_n, prompt_n):
        fault = reply.get("fault")
        if fault == "error_chunk":
            return self.send_json(500, error_body(500, reply.get("message", "fake error")))
        if fault in ("close", "drop"):
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            return
        text = {"reasoning": "", "content": ""}
        calls = {}
        for kind, piece, ci in pieces:
            if kind in text:
                text[kind] += piece
            elif kind == "call_start":
                call = (reply.get("tool_calls") or [])[ci]
                calls[ci] = {"type": "function", "id": call.get("id") or f"fake{self.fake.task_id:04d}call{ci}",
                             "function": {"name": piece, "arguments": ""}}
            else:
                calls[ci]["function"]["arguments"] += piece
        message = {"role": "assistant", "content": text["content"] or None}
        if text["reasoning"]:
            message["reasoning_content"] = text["reasoning"]
        if calls:
            message["tool_calls"] = [calls[k] for k in sorted(calls)]
        self.send_json(200, {
            "choices": [{"finish_reason": finish, "index": 0, "message": message}],
            "created": int(time.time()), "model": self.fake.alias, "system_fingerprint": "b0-fake_llama",
            "object": "chat.completion", "id": f"chatcmpl-fake{self.fake.task_id:04d}",
            "usage": {"completion_tokens": timings["predicted_n"], "prompt_tokens": cache_n + prompt_n,
                      "total_tokens": cache_n + prompt_n + timings["predicted_n"]},
            "timings": timings})


def _max_tokens(body):
    for key in ("max_tokens", "max_completion_tokens", "n_predict"):
        v = body.get(key)
        if isinstance(v, int) and v > 0:
            return v
    return 0


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False  # on Windows this would let two servers share a port

    def server_bind(self):
        # HTTPServer.server_bind looks up the host name, which can stall for seconds on Windows.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            super().handle_error(request, client_address)


def start(script=None, record=None, api_key=None, port=0, host="127.0.0.1", **kwargs):
    """Start serving in a background thread. Returns the FakeLlama (see .port, .base, .stop())."""
    fake = FakeLlama(script=script, record=record, api_key=api_key, **kwargs)
    httpd = _Server((host, port), Handler)
    httpd.fake = fake
    fake.httpd = httpd
    fake.port = httpd.server_address[1]
    fake.log(f"main: server is listening on http://{host}:{fake.port}", "srv  update_slots: all slots are idle")
    fake.thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    fake.thread.start()
    return fake


def main(argv=None):
    # No short options and no abbreviations: llama-server's own flags (-ctk, --ctx-checkpoints ...)
    # must fall through to "ignored" rather than be mistaken for one of these.
    ap = argparse.ArgumentParser(description="A scripted stand-in for llama-server (no model, no GPU). "
                                             "Other llama-server flags (-m, --mmproj, -ngl ...) are accepted and ignored.",
                                 allow_abbrev=False)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18931, help="0 picks a free port (printed at start)")
    ap.add_argument("--script", help="JSON file: a list of scripted replies (see this file's docstring)")
    ap.add_argument("--record", help="append every request to this JSONL file")
    ap.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY"),
                    help="require this key on everything except /health (default: LLAMA_API_KEY, as llama-server)")
    ap.add_argument("--log", help="write llama-server style log lines to this file ('-' for stdout)")
    ap.add_argument("--loop", action="store_true", help="start the script again when it is used up")
    ap.add_argument("--loading-seconds", type=float, default=0.0, help="answer 503 'Loading model' at first")
    ap.add_argument("--validate", choices=["llama", "strict", "off"], default="llama")
    ap.add_argument("--alias", default="holo4-35b-a3b")
    ap.add_argument("--ctx-size", type=int, default=65536, help="also read from llama-server's -c N")
    ap.add_argument("--exit-after", type=int, default=0,
                    help="exit with code 1 after this many chat requests, like a crashed server")
    a, ignored = ap.parse_known_args(argv)
    if "-c" in ignored[:-1] and ignored[ignored.index("-c") + 1].isdigit():
        a.ctx_size = int(ignored[ignored.index("-c") + 1])
    script = None
    if a.script:
        with open(a.script, encoding="utf-8-sig") as f:
            script = json.load(f)
    fake = start(script=script, record=a.record, api_key=a.api_key, port=a.port, host=a.host, loop=a.loop,
                 log=a.log, loading_s=a.loading_seconds, validate=a.validate, alias=a.alias, ctx_size=a.ctx_size)
    print(f"fake_llama listening on {fake.base}" + (f" (ignoring {' '.join(ignored)})" if ignored else ""), flush=True)
    try:
        while True:
            time.sleep(0.1)
            # Counting finished replies, not received requests: a request is recorded a moment before
            # it is served, and exiting in that gap would cut off the reply that should still arrive.
            if a.exit_after and fake.chats_done >= a.exit_after:
                print("fake_llama: exiting on purpose (--exit-after)", flush=True)
                os._exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        fake.stop()


if __name__ == "__main__":
    sys.exit(main())
