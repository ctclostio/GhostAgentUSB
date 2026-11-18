"""Step trace: one JSON line per step in logs\\steps-YYYYMMDD.jsonl, and with trace_messages the
exact request of every step for tools\\bench\\replay.py."""

import base64
import json
import re
import time
import urllib.parse
import uuid
from datetime import datetime
from pathlib import Path

from . import paths
from .console import first_line, one_line, warn
from .tools import describe_call


# --------------------------------------------------------------------------------------------
# Step trace: one JSON line per step in logs\steps-YYYYMMDD.jsonl (timings, the action, the page
# address), and with trace_messages the exact request of every step in logs\replay\<task id>\ for
# tools\bench\replay.py. The trace never holds reasoning, page text or typed text.
# --------------------------------------------------------------------------------------------

# Identifies this run of the agent in the step trace; every task also gets its own id.
RUN_ID = uuid.uuid4().hex[:8]

STEP_FIELDS = ("ts", "run_id", "task_id", "kind", "step", "url", "tool", "args_summary",
               "t_observation", "t_model", "t_execute", "t_settle", "png_bytes", "png_w", "png_h",
               "images_in_prompt", "evicted", "trimmed", "cache_n", "prompt_n", "prompt_ms",
               "predicted_n", "predicted_ms", "finish_reason", "reasoning_chars", "error")
TASK_FIELDS = ("ts", "run_id", "task_id", "kind", "steps", "outcome", "wall_s")

# Tool arguments that hold typed or page text: only their length goes into the trace.
REDACTED_ARGS = ("text", "content", "query", "option")
URL_ARGS = ("url", "new_tab_url")
IMAGE_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


def trace_url(url):
    """Scheme, host and path of an address; the query and fragment (search words, tokens) are
    dropped, and so is everything after the scheme of data:, javascript: and similar addresses."""
    if not url:
        return None
    try:
        u = urllib.parse.urlsplit(str(url).strip())
        port = u.port
    except ValueError:
        return None
    scheme = u.scheme.lower()
    if not scheme:
        return None
    if u.netloc or scheme == "file":
        host = u.hostname or ""
        if ":" in host:
            host = f"[{host}]"  # an IPv6 address keeps its brackets, or its port could not be told apart
        host += f":{port}" if port else ""
        return f"{scheme}://{host}{u.path}"
    if scheme == "about":
        return f"about:{u.path}"
    return scheme + ":"


URL_IN_TEXT_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s\"'<>]+")


def trace_error(text):
    """An error line for the trace. Browser errors often quote the full address ('net::ERR_... at
    https://site/search?q=...'), so every address in it loses its query, as in the url field."""
    if not text:
        return text
    return one_line(URL_IN_TEXT_RE.sub(lambda m: trace_url(m.group(0)) or "<address>", str(text)), 300)


def trace_args_summary(name, args):
    """describe_call() for the trace: typed text, answers and search words become '<N chars>', and
    addresses lose their query."""
    safe = {}
    for k, v in args.items():
        if k in REDACTED_ARGS:
            safe[k] = f"<{len(str(v))} chars>"
        elif k in URL_ARGS:
            u = str(v).strip()
            if u and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", u):
                u = "https://" + u
            safe[k] = trace_url(u) or f"<{len(str(v))} chars>"
        else:
            safe[k] = v
    return one_line(describe_call(name, safe), 300)


def save_replay_step(folder, step, body):
    """Save one request body in the layout tools\\bench\\replay.py reads: <step>.json, with every
    image written beside it as <step>-<k>.png and referenced by that file name."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    body = json.loads(json.dumps(body))  # a deep copy: the live messages keep their images
    k = 0
    for m in body.get("messages") or []:
        if not isinstance(m.get("content"), list):
            continue
        for part in m["content"]:
            if not (isinstance(part, dict) and part.get("type") == "image_url" and isinstance(part.get("image_url"), dict)):
                continue
            url = part["image_url"].get("url", "")
            if not url.startswith("data:") or ";base64," not in url[:100]:
                continue
            head, _, payload = url.partition(",")
            k += 1
            name = f"{step}-{k}{IMAGE_EXT.get(head[5:].split(';')[0], '.png')}"
            (folder / name).write_bytes(base64.b64decode(payload))
            part["image_url"]["url"] = name
    path = folder / f"{step}.json"
    path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    return path


class Trace:
    def __init__(self, settings):
        self.steps_on = bool(settings.get("step_trace", True))
        self.messages_on = bool(settings.get("trace_messages", False))
        self.warned = False

    def _problem(self, e):
        if not self.warned:
            self.warned = True
            warn(f"Could not write the step trace ({first_line(e)}); the agent carries on without it.")

    def _write(self, entry):
        if not self.steps_on:
            return
        try:
            paths.LOGS.mkdir(parents=True, exist_ok=True)
            with open(paths.LOGS / f"steps-{datetime.now():%Y%m%d}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            self._problem(e)

    @staticmethod
    def new_step(task_id, step):
        rec = dict.fromkeys(STEP_FIELDS)
        rec.update(run_id=RUN_ID, task_id=task_id, kind="step", step=step, evicted=False, trimmed=False)
        return rec

    def step(self, rec):
        rec["ts"] = round(time.time(), 3)
        rec["error"] = trace_error(rec.get("error"))
        self._write(rec)

    def task(self, task_id, steps, outcome, wall_s):
        self._write({"ts": round(time.time(), 3), "run_id": RUN_ID, "task_id": task_id, "kind": "task",
                     "steps": steps, "outcome": outcome, "wall_s": wall_s})

    def save_request(self, task_id, step, body):
        if not self.messages_on:
            return
        try:
            save_replay_step(paths.LOGS / "replay" / task_id, step, body)
        except OSError as e:
            self._problem(e)
