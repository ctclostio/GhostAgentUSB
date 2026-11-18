"""Model client: the streaming chat request to the model server, and Ctrl+C, which stops a task at
the next safe point (a second press within 3 seconds quits)."""

import contextlib
import http.client
import json
import queue
import shutil
import signal
import socket
import threading
import time
import urllib.parse

from .console import c, first_line, one_line
from .runstate import STOP


# --------------------------------------------------------------------------------------------
# Model client (streaming, so the console shows progress and Ctrl+C stops a step quickly)
# --------------------------------------------------------------------------------------------

class StepCancelled(Exception):
    pass


class ForcedStop(KeyboardInterrupt):
    """Ctrl+C pressed twice within 3 seconds: leave right away, even mid browser action."""


FORCE_QUIT_SECONDS = 3.0
# When Ctrl+C was last pressed, shared by every handler: a console question (ask_console) puts its
# own handler over the task's, and two presses across the two must still count as a double press.
_LAST_CTRL_C = {"at": None}


def ctrl_c_pressed():
    """Note a Ctrl+C. True when it is the second press within FORCE_QUIT_SECONDS, with the task
    already told to stop: the user wants out now."""
    now = time.monotonic()
    last, _LAST_CTRL_C["at"] = _LAST_CTRL_C["at"], now
    return STOP.is_set() and last is not None and 0 <= now - last <= FORCE_QUIT_SECONDS


@contextlib.contextmanager
def ctrl_c_stops_task(clear=True):
    """While a task runs, Ctrl+C only sets STOP: the task then ends at the next safe point (the
    model request is cut at once, a browser action is allowed to finish). A KeyboardInterrupt
    raised inside a Playwright call kills Playwright's dispatcher, and every later browser call
    then hangs forever. A second Ctrl+C within 3 seconds is the way out of a browser call that
    hangs: it raises ForcedStop, which ends the agent."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    if clear:
        STOP.clear()
        _LAST_CTRL_C["at"] = None  # a new task: a press from the last one does not count

    def on_ctrl_c(signum, frame):
        if ctrl_c_pressed():
            signal.signal(signal.SIGINT, previous)
            raise ForcedStop()
        STOP.set()
        print(c("\n  Stopping after the current browser action... (press Ctrl+C again within "
                f"{FORCE_QUIT_SECONDS:.0f} s to quit GhostAgent now)", "33"), flush=True)

    previous = signal.signal(signal.SIGINT, on_ctrl_c)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


STEP_HINT = "   (Ctrl+C stops this task; the model stays loaded)"


def step_status_hint():
    """The reminder at the end of the step status line, only in a console wide enough for it: a line
    that wraps can no longer be redrawn in place."""
    try:
        columns = shutil.get_terminal_size((0, 0)).columns
    except Exception:
        columns = 0
    return STEP_HINT if columns >= 115 else ""


def server_error_message(data):
    """The readable part of an error from llama-server: error.message from its JSON, or the text."""
    obj = data
    if isinstance(data, (bytes, str)):
        text = data.decode("utf-8", "replace") if isinstance(data, bytes) else data
        try:
            obj = json.loads(text)
        except ValueError:
            return one_line(text, 2000) or "no details"
    if isinstance(obj, dict) and "error" in obj:
        e = obj["error"]
        if isinstance(e, dict):
            return str(e.get("message") or json.dumps(e)[:2000])
        return str(e)[:2000]
    return json.dumps(obj)[:2000]


def stream_chat(base, body, on_event, api_key=None):
    """POST /v1/chat/completions with stream=True. Calls on_event(kind, value) as data arrives.
    Returns the assembled message dict (content, reasoning, tool_calls, timings, finish_reason).
    The HTTP read runs in a worker thread, so the main thread keeps checking STOP; on Ctrl+C the
    socket is shut down at once (the server then drops the request) and StepCancelled is raised."""
    q = queue.Queue()
    holder = {"cancelled": False, "conn": None}
    u = urllib.parse.urlsplit(base)
    https = u.scheme == "https"
    host, port = u.hostname or "127.0.0.1", u.port or (443 if https else 80)
    path = u.path.rstrip("/") + "/v1/chat/completions"
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    def worker():
        # http.client never uses a proxy, and the connection object exists before anything is sent,
        # so a cancel can always reach it.
        conn = (http.client.HTTPSConnection if https else http.client.HTTPConnection)(host, port, timeout=900)
        holder["conn"] = conn
        try:
            conn.connect()
            conn.auto_open = 0  # after a cancel, never reconnect silently
            if holder["cancelled"]:
                return
            conn.request("POST", path, data, headers)
            resp = conn.getresponse()
            if resp.status != 200:
                q.put(("error", f"The model server answered HTTP {resp.status}: {server_error_message(resp.read())}"))
                return
            done = False
            while True:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("error:"):
                    # Some llama-server builds send a failure as an "error:" event instead of a data chunk.
                    q.put(("error", "The model server reported an error: " + server_error_message(line[6:].strip())))
                    return
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    chunk = None
                if not isinstance(chunk, dict):
                    q.put(("error", "The model server sent an unreadable reply: " + one_line(payload, 200)))
                    return
                if "error" in chunk:
                    # llama-server reports a failure mid-stream this way, then ends without [DONE].
                    q.put(("error", "The model server reported an error: " + server_error_message(chunk)))
                    return
                q.put(("chunk", chunk))
            q.put(("end", None) if done else ("error", "The model server closed the stream early."))
        except http.client.IncompleteRead:
            if not holder["cancelled"]:
                q.put(("error", "The connection to the model server broke off in the middle of a reply."))
        except Exception as e:
            if not holder["cancelled"]:
                q.put(("error", f"Could not talk to the model server at {base} ({first_line(e)})."))
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def cancel():
        holder["cancelled"] = True
        conn = holder.get("conn")
        if conn is None:
            return
        sock = conn.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            conn.close()
        except Exception:
            pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    content, reasoning = [], []
    calls = {}
    usage = {}
    finish = None
    while True:
        if STOP.is_set():
            cancel()
            raise StepCancelled()
        try:
            kind, val = q.get(timeout=0.25)
        except queue.Empty:
            continue
        if kind == "end":
            break
        if kind == "error":
            raise RuntimeError(val)
        if val.get("prompt_progress"):
            p = val["prompt_progress"]
            on_event("prompt", (p.get("processed", 0), p.get("total", 0)))
        if val.get("timings"):
            usage = val["timings"]
        for choice in val.get("choices") or []:
            if choice.get("finish_reason") and choice.get("index", 0) == 0:
                finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
                on_event("reasoning", delta["reasoning_content"])
            if delta.get("content"):
                content.append(delta["content"])
                on_event("content", delta["content"])
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
                on_event("tool", slot["name"])
    tool_calls = [{"id": v["id"] or f"call_{i}", "type": "function",
                   "function": {"name": v["name"], "arguments": v["arguments"] or "{}"}}
                  for i, v in sorted(calls.items())]
    return {"content": "".join(content), "reasoning": "".join(reasoning), "tool_calls": tool_calls,
            "timings": usage, "finish_reason": finish}
