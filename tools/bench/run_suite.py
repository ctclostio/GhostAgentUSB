"""Run the local benchmark suite once and score it: one command to measure this PC, or to compare
settings and model profiles.

    python\\python.exe -I tools\\bench\\run_suite.py [--tasks a,b] [--label NAME] [agent options]

or "Run benchmark.cmd" next to "Start GhostAgent.cmd". Any option this script does not know goes to
the agent, for example --settings-override my-settings.json or --browser edge (the default here is
the bundled browser). It serves tools\\bench\\site on 127.0.0.1:8765, runs every task of suite.json
(or the --tasks named) in one agent session, then prints score.py's table. Its files go to
GhostAgent\\logs\\bench-<time>[-label]-*: the tasks, the agent's results, the site's log and the
score as JSON.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
GHOST = HERE.parent.parent / "GhostAgent"
PORT = 8765


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the GhostAgent benchmark suite and score it.")
    ap.add_argument("--tasks", help="comma-separated task names (default: all of suite.json)")
    ap.add_argument("--label", default="", help="a word added to the file names, e.g. the settings tried")
    ap.add_argument("--port", type=int, default=PORT, help=f"the site's port (default {PORT}; 0 = any free port)")
    a, agent_args = ap.parse_known_args(argv)

    sys.path.insert(0, str(HERE))
    import bench_server
    suite = bench_server.load_suite()
    if a.tasks:
        wanted = [t.strip() for t in a.tasks.split(",") if t.strip()]
        unknown = sorted(set(wanted) - {t["name"] for t in suite})
        if unknown:
            ap.error(f"unknown task(s): {', '.join(unknown)}")
    else:
        wanted = [t["name"] for t in suite]

    # The agent's --data-dir (tests) moves its logs folder; these files go next to them.
    data = agent_args[agent_args.index("--data-dir") + 1] if "--data-dir" in agent_args[:-1] else None
    logs = (Path(data) if data else GHOST) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    base = logs / ("bench-" + datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{a.label}" if a.label else ""))
    tasks_file, results, site_log = (Path(f"{base}-{x}") for x in ("tasks.json", "results.jsonl", "site.jsonl"))

    try:
        site = bench_server.start(port=a.port, log=site_log)
    except OSError as e:
        return f"Cannot serve the benchmark site on 127.0.0.1:{a.port} ({e}). Is another benchmark running?"
    started = time.time()
    try:
        suite = [t for t in bench_server.load_suite(port=site.port) if t["name"] in wanted]
        tasks_file.write_text(json.dumps([{k: t[k] for k in ("name", "task", "start_url")} for t in suite], indent=1),
                              encoding="utf-8")
        if "--browser" not in agent_args:
            agent_args = ["--browser", "bundled"] + agent_args
        cmd = [sys.executable, "-I", str(GHOST / "agent" / "ghost_agent.py"), "--tasks-file", str(tasks_file),
               "--results", str(results)] + agent_args
        print(f"Running {len(suite)} benchmark task(s); results in {results}", flush=True)
        rc = subprocess.call(cmd)
    finally:
        site.stop()

    # The agent's step trace (per-step times) is one file per day in its logs folder.
    steps = [p for p in logs.glob("steps-*.jsonl") if p.stat().st_mtime >= started]
    score = [sys.executable, "-I", str(HERE / "score.py"), "--results", str(results), "--bench-log", str(site_log),
             "--json", f"{base}-score.json"]
    if steps:
        score += ["--steps", str(max(steps, key=lambda p: p.stat().st_mtime))]
    if not results.exists():
        print("The agent wrote no results; nothing to score.", file=sys.stderr)
        return rc or 1
    return subprocess.call(score) or rc


if __name__ == "__main__":
    sys.exit(main())
