"""Replay recorded GhostAgent requests against a llama-server, to measure prompt and generation cost.

    python\\python.exe -I tools\\bench\\replay.py --replay-dir logs\\replay --url http://127.0.0.1:18931
        [--api-key KEY] [--mode prompt|full|both] [--seed 1] [--repeats 3] [--server-log logs\\llama-server-X.log]
        [--server-pid PID] [--vram] [--label cms1536] [--out replay-cms1536.jsonl]

The recordings come from the agent with the setting trace_messages on:
    <replay dir>\\<task_id>\\<step>.json       the exact request body of that step
    <replay dir>\\<task_id>\\<step>-<k>.png    its images; in the body, the image_url "url" holds just
                                              the file name (write_step() below writes this layout)
Each task's steps are sent in order, the way the agent sent them, so the server's prompt cache sees
the same sequence. Two modes:
    prompt   max_tokens 1: measures prompt processing only (the cost the cache flags change)
    full     the recorded request with a fixed seed: prompt plus generation
For every step it records cache_n, prompt_n, prompt_ms, predicted_n and tokens per second from the
reply's timings, and, with --server-log, what the server logged while serving it: decode calls (prompt
batches plus image batches; batches over 32 tokens counted separately), image encode ms and
"making room" lines. --server-pid adds the server's peak RAM (working set); --vram samples its GPU
memory with nvidia-smi. Results go to a JSONL file plus a .summary.json next to it.

Compare like with like: a replay should start on a freshly started server. Within one replay, a
second repeat (or the full pass after the prompt pass with --mode both) sends prompts the server has
already seen, and with llama-server's RAM prompt cache on (--cache-ram above 0) it can restore them
instead of processing them, so later passes look faster than a real run. For comparable repeats,
restart the server between them (or run it with --cache-ram 0, when that is not what is measured).
Ctrl+C stops the replay; the steps done so far are kept and summarised.
"""

import argparse
import base64
import copy
import ctypes
import http.client
import json
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CREATE_NO_WINDOW = 0x08000000
MIME ={".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
EXT = {v: k for k, v in MIME.items() if k != ".jpeg"}


# --------------------------------------------------------------------------------------------
# The recording layout
# --------------------------------------------------------------------------------------------

def write_step(folder, step, body):
    """Save one request body in the replay layout: images become <step>-<k>.png beside <step>.json."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    body = copy.deepcopy(body)
    k = 0
    for part in _image_parts(body):
        url = part["image_url"].get("url", "")
        if not url.startswith("data:"):
            continue
        head, _, payload = url.partition(",")
        k += 1
        name = f"{step}-{k}{EXT.get(head[5:].split(';')[0], '.png')}"
        (folder / name).write_bytes(base64.b64decode(payload))
        part["image_url"]["url"] = name
    path = folder / f"{step}.json"
    path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    return path


def load_step(path):
    """A recorded body with its image file names turned back into data: URLs."""
    path = Path(path)
    body = json.loads(path.read_text(encoding="utf-8-sig"))
    for part in _image_parts(body):
        url = part["image_url"].get("url", "")
        if url.startswith("data:"):
            continue
        name = url[5:] if url.startswith("file:") else url
        image = path.parent / Path(urllib.parse.unquote(name)).name
        if not image.is_file():
            raise FileNotFoundError(f"{path.name} refers to a missing image file {image.name}")
        mime = MIME.get(image.suffix.lower(), "image/png")
        part["image_url"]["url"] = f"data:{mime};base64," + base64.b64encode(image.read_bytes()).decode()
    return body


def _image_parts(body):
    for m in body.get("messages") or []:
        if isinstance(m.get("content"), list):
            for part in m["content"]:
                if isinstance(part, dict) and part.get("type") == "image_url" and isinstance(part.get("image_url"), dict):
                    yield part


def discover(replay_dir, only=None):
    """[(task_id, [step paths in order])]. A folder of step files counts as one task."""
    root = Path(replay_dir)
    if not root.is_dir():
        return []

    def steps_in(d):
        files = [p for p in d.glob("*.json") if p.stem.isdigit()]
        return sorted(files, key=lambda p: int(p.stem))

    tasks = [(root.name, steps_in(root))] if steps_in(root) else []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if steps_in(d) and (not only or d.name in only):
            tasks.append((d.name, steps_in(d)))
    return tasks


def prepare(body, mode, seed):
    body = copy.deepcopy(body)
    body["stream"] = True
    if mode == "prompt":
        for k in ("max_completion_tokens", "n_predict"):
            body.pop(k, None)
        body["max_tokens"] = 1
    else:
        body["seed"] = seed
    return body


# --------------------------------------------------------------------------------------------
# Talking to the server
# --------------------------------------------------------------------------------------------

def post_stream(url, body, api_key=None, timeout=1800):
    """POST /v1/chat/completions and read the SSE stream. Returns what the reply said and how long it took."""
    u = urllib.parse.urlsplit(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    out = {"status": None, "timings": {}, "finish_reason": None, "done": False, "error": None,
           "first_token_s": None, "content_chars": 0, "reasoning_chars": 0, "tool_calls": 0}
    t0 = time.perf_counter()
    try:
        conn.request("POST", (u.path.rstrip("/") or "") + "/v1/chat/completions", json.dumps(body).encode(), headers)
        resp = conn.getresponse()
        out["status"] = resp.status
        if resp.status != 200:
            text = resp.read().decode("utf-8", "replace")
            try:
                out["error"] = json.loads(text)["error"]["message"]
            except Exception:
                out["error"] = f"HTTP {resp.status}: {text[:300]}"
            return out
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                out["done"] = True
                break
            chunk = json.loads(data)
            if "error" in chunk:
                err = chunk["error"]
                out["error"] = err.get("message") if isinstance(err, dict) else str(err)
                break
            if chunk.get("timings"):
                out["timings"] = chunk["timings"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                got = delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")
                if got and out["first_token_s"] is None:
                    out["first_token_s"] = time.perf_counter() - t0
                out["content_chars"] += len(delta.get("content") or "")
                out["reasoning_chars"] += len(delta.get("reasoning_content") or "")
                out["tool_calls"] += sum(1 for tc in delta.get("tool_calls") or [] if tc.get("id"))
                if choice.get("finish_reason"):
                    out["finish_reason"] = choice["finish_reason"]
        if not out["done"] and not out["error"]:
            out["error"] = "the stream ended without [DONE]"
    except (OSError, http.client.HTTPException, ValueError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        out["wall_s"] = time.perf_counter() - t0
        conn.close()
    return out


# --------------------------------------------------------------------------------------------
# The server log
# --------------------------------------------------------------------------------------------

PROMPT_BATCH_RE = re.compile(r"prompt processing(?: progress)?, n_tokens = \s*(\d+)")
CACHED_RE = re.compile(r"cached n_tokens = (\d+)")
IMAGE_ENCODE_RE = re.compile(r"slice encoded in (\d+) ms")
IMAGE_DECODE_RE = re.compile(r"decoded \(batch (\d+)/(\d+)\) in (\d+) ms")
MAKING_ROOM_RE = re.compile(r"making room")


class LogTail:
    """Reads what a growing log file gained since a mark."""

    def __init__(self, path):
        self.path = Path(path)

    def mark(self):
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def since(self, offset, wait_s=1.5):
        """Text added since offset. Waits briefly for the request's timing lines, which the server's
        log thread can write a moment after the reply has been sent."""
        end = time.time() + wait_s
        text = ""
        while True:
            try:
                with open(self.path, "rb") as f:
                    f.seek(offset)
                    text = f.read().decode("utf-8", "replace")
            except OSError:
                text = ""
            if "prompt eval time" in text or time.time() >= end:
                return text
            time.sleep(0.05)


def parse_log(text):
    """Per-request numbers from llama-server log lines. Prompt lines report the running token count,
    so each batch is the step from the previous count (starting at the cached count)."""
    m = CACHED_RE.search(text)
    last = int(m.group(1)) if m else 0
    batches = []
    for m in PROMPT_BATCH_RE.finditer(text):
        n = int(m.group(1))
        batches.append(n - last if n > last else n)
        last = n
    image_batches = IMAGE_DECODE_RE.findall(text)
    return {
        "decode_calls": len(batches) + len(image_batches),
        "decode_calls_gt32": sum(1 for b in batches if b > 32) + len(image_batches),
        "prompt_batches": batches,
        "images_encoded": len(IMAGE_ENCODE_RE.findall(text)),
        "image_encode_ms": sum(int(x) for x in IMAGE_ENCODE_RE.findall(text)),
        "image_decode_ms": sum(int(b[2]) for b in image_batches),
        "making_room": len(MAKING_ROOM_RE.findall(text)),
    }


# --------------------------------------------------------------------------------------------
# Memory: peak RAM of the server process, GPU memory via nvidia-smi
# --------------------------------------------------------------------------------------------

class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def process_memory_mib(pid):
    """(working set, peak working set) of a process in MiB, or None (Windows only)."""
    if sys.platform != "win32" or not pid:
        return None
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = ctypes.c_void_p
    handle = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        if not k32.K32GetProcessMemoryInfo(ctypes.c_void_p(handle), ctypes.byref(pmc), pmc.cb):
            return None
        return round(pmc.WorkingSetSize / 2 ** 20, 1), round(pmc.PeakWorkingSetSize / 2 ** 20, 1)
    finally:
        k32.CloseHandle(ctypes.c_void_p(handle))


def parse_nvidia_smi(text, pid):
    """MiB of GPU memory used by pid, from `nvidia-smi --query-compute-apps=pid,used_memory
    --format=csv,noheader,nounits` output."""
    total = 0
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(pid) and parts[1].isdigit():
            total += int(parts[1])
    return total


class VramSampler(threading.Thread):
    """Polls nvidia-smi once a second and keeps the peak for one process."""

    def __init__(self, pid, interval=1.0):
        super().__init__(daemon=True)
        self.pid, self.interval, self.peak = pid, interval, 0
        self.exe = shutil.which("nvidia-smi")
        self.halt = threading.Event()

    def run(self):
        while self.exe and not self.halt.is_set():
            try:
                out = subprocess.run([self.exe, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10, creationflags=CREATE_NO_WINDOW).stdout
                self.peak = max(self.peak, parse_nvidia_smi(out, self.pid))
            except (OSError, subprocess.SubprocessError):
                pass
            self.halt.wait(self.interval)

    def stop(self):
        self.halt.set()
        self.join(5)
        return self.peak if self.exe else None


# --------------------------------------------------------------------------------------------
# The replay
# --------------------------------------------------------------------------------------------

def replay(tasks, url, modes=("prompt",), seed=1, repeats=1, api_key=None, server_log=None, server_pid=None,
           label="", limit_steps=0, on_row=None):
    """Send every recorded step in order; returns one row per step."""
    tail = LogTail(server_log) if server_log else None
    rows = []
    for mode in modes:
        for rep in range(1, repeats + 1):
            for task_id, steps in tasks:
                for path in steps[:limit_steps or None]:
                    try:
                        body = prepare(load_step(path), mode, seed)
                    except (OSError, ValueError) as e:
                        # A damaged recording is reported like a failed request, and the replay goes on.
                        body = None
                        res = {"status": None, "timings": {}, "finish_reason": None, "first_token_s": None,
                               "wall_s": None, "error": f"could not load the recording: {e}"}
                    offset = tail.mark() if tail else 0
                    if body is not None:
                        res = post_stream(url, body, api_key)
                    t = res["timings"]
                    row = {"label": label, "mode": mode, "repeat": rep, "task_id": task_id, "step": int(path.stem),
                           "status": res["status"], "error": res["error"], "finish_reason": res["finish_reason"],
                           "cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
                           "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
                           "predicted_per_second": t.get("predicted_per_second"),
                           "first_token_s": _r(res["first_token_s"], 3), "wall_s": _r(res["wall_s"], 3)}
                    if tail and body is not None:
                        row.update(parse_log(tail.since(offset)))
                    if server_pid:
                        mem = process_memory_mib(server_pid)
                        row["rss_mib"] = mem[0] if mem else None
                    rows.append(row)
                    if on_row:
                        on_row(row)
    return rows


def _r(v, nd=1):
    return None if v is None else round(v, nd)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def summarize(rows):
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        rs = [r for r in rows if r["mode"] == mode]
        ok = [r for r in rs if not r["error"]]
        gen = [r for r in ok if r.get("predicted_n", 0) and r.get("predicted_n", 0) > 1]
        s = {
            "steps": len(rs), "errors": len(rs) - len(ok),
            "prompt_ms_mean": _r(_mean([r["prompt_ms"] for r in ok])),
            "prompt_ms_median": _r(_median([r["prompt_ms"] for r in ok])),
            "prompt_s_total": _r(sum(r["prompt_ms"] or 0 for r in ok) / 1000.0, 2),
            "prompt_n_mean": _r(_mean([r["prompt_n"] for r in ok])),
            "cache_n_mean": _r(_mean([r["cache_n"] for r in ok])),
            "predicted_n_mean": _r(_mean([r["predicted_n"] for r in gen])),
            "tok_per_s_mean": _r(_mean([r["predicted_per_second"] for r in gen]), 2),
            "tok_per_s_median": _r(_median([r["predicted_per_second"] for r in gen]), 2),
            "wall_s_mean": _r(_mean([r["wall_s"] for r in ok]), 3),
            "tasks": {},
        }
        if any("decode_calls" in r for r in ok):
            s["decode_calls_mean"] = _r(_mean([r.get("decode_calls") for r in ok]), 2)
            s["decode_calls_gt32_mean"] = _r(_mean([r.get("decode_calls_gt32") for r in ok]), 2)
            s["image_encode_ms_mean"] = _r(_mean([r.get("image_encode_ms") for r in ok]))
            s["making_room_total"] = sum(r.get("making_room") or 0 for r in ok)
        for task_id in dict.fromkeys(r["task_id"] for r in rs):
            tr = [r for r in ok if r["task_id"] == task_id]
            s["tasks"][task_id] = {"steps": len([r for r in rs if r["task_id"] == task_id]),
                                   "prompt_s_total": _r(sum(r["prompt_ms"] or 0 for r in tr) / 1000.0, 2),
                                   "prompt_ms_mean": _r(_mean([r["prompt_ms"] for r in tr]))}
        out[mode] = s
    return out


def table(rows, summary):
    lines = [f"{'mode':<7}{'task':<12}{'step':>5}{'cache_n':>9}{'prompt_n':>9}{'prompt_ms':>11}{'tok/s':>8}"
             f"{'decodes':>8}{'img_ms':>8}{'room':>5}  error"]
    for r in rows:
        tps = r.get("predicted_per_second")
        lines.append(f"{r['mode']:<7}{str(r['task_id'])[:11]:<12}{r['step']:>5}{_s(r['cache_n']):>9}{_s(r['prompt_n']):>9}"
                     f"{_s(r['prompt_ms']):>11}{_s(None if tps is None else round(tps, 1)):>8}"
                     f"{_s(r.get('decode_calls')):>8}{_s(r.get('image_encode_ms')):>8}{_s(r.get('making_room')):>5}"
                     f"  {r['error'] or ''}")
    for mode, s in summary.items():
        lines.append("")
        lines.append(f"[{mode}] {s['steps']} steps, {s['errors']} errors; prompt ms mean {s['prompt_ms_mean']} "
                     f"median {s['prompt_ms_median']} total {s['prompt_s_total']} s; prompt_n mean {s['prompt_n_mean']}, "
                     f"cache_n mean {s['cache_n_mean']}; generation {s['tok_per_s_mean']} tok/s mean")
        if "decode_calls_mean" in s:
            lines.append(f"        decode calls/step {s['decode_calls_mean']} (>32 tokens: {s['decode_calls_gt32_mean']}), "
                         f"image encode {s['image_encode_ms_mean']} ms/step, 'making room' lines {s['making_room_total']}")
    return "\n".join(lines)


def _s(v):
    return "-" if v is None else str(v)


def default_out(label):
    """replay-<label or time>.jsonl in the GhostAgent logs folder (repo or stick), else in a results folder here."""
    name = f"replay-{label or datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
    for logs in (HERE.parent.parent / "GhostAgent" / "logs", HERE.parent.parent / "logs"):
        if (logs.parent / "agent").is_dir():
            return logs / name
    return HERE / "results" / name


def main(argv=None):
    ap = argparse.ArgumentParser(description="Replay recorded GhostAgent requests against a llama-server.")
    ap.add_argument("--replay-dir", required=True, help="folder of <task_id> folders (or one task's folder)")
    ap.add_argument("--url", default="http://127.0.0.1:18931")
    ap.add_argument("--api-key")
    ap.add_argument("--mode", choices=["prompt", "full", "both"], default="prompt")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--tasks", help="comma-separated task ids to replay (default: all)")
    ap.add_argument("--limit-steps", type=int, default=0, help="replay only the first N steps of each task")
    ap.add_argument("--server-log", help="llama-server's log file, for decode calls, image encode and making-room lines")
    ap.add_argument("--server-pid", type=int, help="llama-server's process id, for its RAM use")
    ap.add_argument("--vram", action="store_true", help="sample the server's GPU memory with nvidia-smi")
    ap.add_argument("--label", default="", help="a name for this configuration, stored in every row")
    ap.add_argument("--out", help="results JSONL (default: replay-<label or time>.jsonl in the logs folder)")
    a = ap.parse_args(argv)
    if a.vram and not a.server_pid:
        ap.error("--vram needs --server-pid (the GPU memory is read for that process)")

    tasks = discover(a.replay_dir, set(a.tasks.split(",")) if a.tasks else None)
    if not tasks:
        print(f"No recorded steps under {a.replay_dir}.")
        return 1
    out = Path(a.out) if a.out else default_out(a.label)
    out.parent.mkdir(parents=True, exist_ok=True)
    modes = ["prompt", "full"] if a.mode == "both" else [a.mode]
    print(f"replaying {sum(len(s) for _, s in tasks)} steps from {len(tasks)} task(s), modes {modes}, "
          f"{a.repeats} repeat(s) -> {out}", flush=True)
    if a.repeats > 1 or len(modes) > 1:
        print("note: later passes reuse the same server, whose RAM prompt cache (--cache-ram) can make them "
              "look faster; restart the server between passes for comparable numbers.", flush=True)
    sampler = VramSampler(a.server_pid) if a.vram else None
    if sampler:
        sampler.start()
    rows = []
    stopped = False
    with open(out, "w", encoding="utf-8") as f:
        def on_row(row):
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"  {row['mode']} {row['task_id']} step {row['step']}: prompt_n {row['prompt_n']} "
                  f"cache_n {row['cache_n']} {row['prompt_ms']} ms {row['error'] or ''}", flush=True)
        try:
            replay(tasks, a.url, modes, a.seed, a.repeats, a.api_key, a.server_log, a.server_pid, a.label,
                   a.limit_steps, on_row)
        except KeyboardInterrupt:
            stopped = True
            print(f"\nStopped with Ctrl+C after {len(rows)} step(s); summarising those.", flush=True)
    summary = {"label": a.label, "url": a.url, "stopped_early": stopped, "modes": summarize(rows)}
    if a.server_pid:
        mem = process_memory_mib(a.server_pid)
        summary["peak_rss_mib"] = mem[1] if mem else None
    if sampler:
        summary["peak_vram_mib"] = sampler.stop()
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(table(rows, summary["modes"]))
    if "peak_rss_mib" in summary or "peak_vram_mib" in summary:
        print(f"peak RAM {summary.get('peak_rss_mib')} MiB, peak VRAM {summary.get('peak_vram_mib')} MiB")
    return 1 if stopped or any(r["error"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
