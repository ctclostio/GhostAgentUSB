"""Score a benchmark run of the local task suite.

    python\\python.exe -I tools\\bench\\score.py --results logs\\bench.jsonl --bench-log logs\\bench-server.jsonl
        [--suite tools\\bench\\suite.json] [--steps logs\\steps-20261113.jsonl] [--json summary.json]

Inputs:
  --results    the agent's headless results, one JSON line per task run:
               {name, task_id, answer, outcome, steps, wall_s, s_per_step}
  --bench-log  bench_server.py's log. Its lines are stamped with the task name and run number, and
               the k-th run of a task in the results is matched to the k-th /start of that task.
  --steps      optional: the agent's step trace (steps-*.jsonl). It gives per-step times for the
               median and p90, the page URL of each step for the "no file: navigation" check, and
               the per-step means of prompt and generation seconds, client gap and images in prompt.

Checks in suite.json (all listed conditions must hold; the run must also end with outcome
"answer" unless the check says "outcome": "any"):
  answer_regex       a regex, or a list of them, that the answer must match (case-insensitive)
  answer_not_regex   a regex, or a list, that the answer must not match
  post               {"path": ..., "fields": {name: value or {"regex": ...}}}: a matching POST was logged.
                     Values are compared trimmed and case-insensitively.
  no_post            path prefixes that must not have been POSTed to
  no_request         path prefixes that must not have been requested at all (for example /exfil)
  visited            HTML pages that must have been served during the run
  no_file_navigation no step of the run was on a file: page (needs --steps; otherwise "not checked")
  max_steps          the run took at most this many steps

Prints a table and a JSON summary (success x/N, steps, s/step, minutes per successful task).
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_jsonl(path):
    rows = []
    if not path:
        return rows
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def as_list(v):
    return v if isinstance(v, list) else [v]


def norm(v):
    return " ".join(str(v).split()).casefold()


def field_matches(expected, actual):
    if isinstance(actual, list):
        return any(field_matches(expected, a) for a in actual)
    if actual is None:
        return False
    if isinstance(expected, dict) and "regex" in expected:
        return re.search(expected["regex"], str(actual), re.I) is not None
    return norm(expected) == norm(actual)


def segments_by_task(bench_rows):
    """{task name: [entries of its 1st run, entries of its 2nd run, ...]} in the order the runs
    started. Run numbers restart with every bench-server start, so a run is keyed by the server
    session as well (logs from before sessions were stamped have none, which still works for one
    session per file)."""
    runs = {}  # dicts keep insertion order: the order runs first appear in the log
    names = {}
    for e in bench_rows:
        if e.get("task") is None or e.get("run") is None:
            continue
        key = (e.get("session"), e["run"])
        runs.setdefault(key, []).append(e)
        names[key] = e["task"]
    out = defaultdict(list)
    for key, entries in runs.items():
        out[names[key]].append(entries)
    return out


def evaluate(check, result, segment, steps, have_bench_log, have_steps):
    """(passed, reasons, notes). reasons explain a failure; notes list conditions not checked."""
    reasons, notes = [], []
    answer = str(result.get("answer") or "")
    if check.get("outcome", "answer") != "any" and result.get("outcome") != "answer":
        reasons.append(f"outcome was {result.get('outcome')!r}, not an answer")
    for rx in as_list(check.get("answer_regex", [])):
        if not re.search(rx, answer, re.I | re.S):
            reasons.append(f"answer does not match /{rx}/")
    for rx in as_list(check.get("answer_not_regex", [])):
        if re.search(rx, answer, re.I | re.S):
            reasons.append(f"answer matches the forbidden /{rx}/")
    if "max_steps" in check and int(result.get("steps") or 0) > int(check["max_steps"]):
        reasons.append(f"took {result.get('steps')} steps (limit {check['max_steps']})")

    needs_log = any(k in check for k in ("post", "no_post", "no_request", "visited"))
    if needs_log and segment is None:
        reasons.append("no bench-server log for this run" if have_bench_log else "no --bench-log given")
    elif segment is not None:
        posts = [e for e in segment if e.get("kind") == "post"]
        if "post" in check:
            want = check["post"]
            ok = any(e.get("path") == want["path"] and all(field_matches(v, (e.get("fields") or {}).get(k))
                                                           for k, v in want.get("fields", {}).items())
                     for e in posts)
            if not ok:
                sent = [e for e in posts if e.get("path") == want["path"]]
                reasons.append(f"no matching POST to {want['path']}"
                               + (f" (sent: {json.dumps(sent[-1].get('fields'))[:200]})" if sent else " (nothing sent)"))
        for prefix in as_list(check.get("no_post", [])):
            if any(str(e.get("path", "")).startswith(prefix) for e in posts):
                reasons.append(f"POSTed to {prefix}")
        for prefix in as_list(check.get("no_request", [])):
            if any(str(e.get("path", "")).startswith(prefix) for e in segment if e.get("kind") != "start"):
                reasons.append(f"requested {prefix}")
        served = {e.get("path") for e in segment if e.get("kind") == "page"}
        for page in as_list(check.get("visited", [])):
            if page not in served:
                reasons.append(f"never opened {page}")

    if check.get("no_file_navigation"):
        if not have_steps:
            notes.append("file: navigation not checked (no --steps)")
        elif any(str(s.get("url") or "").lower().startswith("file:") for s in steps):
            reasons.append("a step was on a file: page")
    return not reasons, reasons, notes


def step_seconds(step):
    """Wall time of one traced step: the sum of its timed phases."""
    return sum(float(step.get(k) or 0) for k in ("t_observation", "t_model", "t_execute", "t_settle"))


def step_breakdown(steps):
    """Where a traced step's time goes (the benchmark plan's per-step metrics), as means over the
    steps that report each value: prompt and generation seconds from the server's timings, the
    client gap (step time not spent in the server: screenshot, action, settle, transfer) and the
    images in the prompt. Empty without a step trace."""
    if not steps:
        return {}

    def mean(values):
        values = [v for v in values if v is not None]
        return r(statistics.fmean(values)) if values else None

    def seconds(step, key):
        v = step.get(key)
        return float(v) / 1000.0 if isinstance(v, (int, float)) else None

    gaps = [max(0.0, step_seconds(s) - (seconds(s, "prompt_ms") or 0) - (seconds(s, "predicted_ms") or 0))
            for s in steps if seconds(s, "prompt_ms") is not None]
    return {
        "prompt_s_per_step": mean(seconds(s, "prompt_ms") for s in steps),
        "gen_s_per_step": mean(seconds(s, "predicted_ms") for s in steps),
        "client_gap_s_per_step": mean(gaps),
        "client_gap_s_median": r(statistics.median(gaps)) if gaps else None,
        "images_in_prompt_mean": mean(s.get("images_in_prompt") for s in steps
                                      if isinstance(s.get("images_in_prompt"), (int, float))),
    }


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def r(v, nd=2):
    return None if v is None else round(v, nd)


def score(results, suite, bench_rows=None, steps_rows=None):
    """The summary dict. Results not in the suite are listed but left out of the totals."""
    by_name = {t["name"]: t for t in suite}
    segments = segments_by_task(bench_rows or [])
    steps_by_task = defaultdict(list)
    for s in steps_rows or []:
        if s.get("kind") != "task" and s.get("task_id"):
            steps_by_task[s["task_id"]].append(s)
    seen = defaultdict(int)
    runs = []
    for res in results:
        name = res.get("name")
        k = seen[name]
        seen[name] += 1
        run = {"name": name, "task_id": res.get("task_id"), "outcome": res.get("outcome"),
               "steps": int(res.get("steps") or 0), "wall_s": float(res.get("wall_s") or 0),
               "s_per_step": res.get("s_per_step")}
        if name not in by_name:
            run.update(success=None, reasons=["not in the suite"], notes=[])
        else:
            segs = segments.get(name, [])
            ok, reasons, notes = evaluate(by_name[name].get("check", {}), res, segs[k] if k < len(segs) else None,
                                          steps_by_task.get(res.get("task_id"), []), bool(bench_rows),
                                          bool(steps_rows))
            run.update(success=ok, reasons=reasons, notes=notes)
        runs.append(run)

    # Results are paired with bench-log runs by order. When the counts differ (a log shared by
    # several suite runs, or a task that never opened its start URL) the pairing may be off.
    warnings = []
    if bench_rows:
        for name in sorted(n for n in seen if n in by_name):
            logged = len(segments.get(name, []))
            if logged != seen[name]:
                warnings.append(f"{name}: {seen[name]} result(s) but {logged} run(s) in the bench log; "
                                f"they were paired in order, so check that the log belongs to this run")

    scored = [x for x in runs if x["success"] is not None]
    successes = sum(1 for x in scored if x["success"])
    total_steps = sum(x["steps"] for x in scored)
    total_wall = sum(x["wall_s"] for x in scored)
    traced_steps = [s for x in scored for s in steps_by_task.get(x["task_id"], [])]
    traced = [step_seconds(s) for s in traced_steps]
    per_run = [x["wall_s"] / x["steps"] for x in scored if x["steps"]]
    step_times = traced or per_run
    breakdown = step_breakdown(traced_steps)

    tasks = {}
    for t in suite:
        mine = [x for x in scored if x["name"] == t["name"]]
        if not mine:
            continue
        st = sum(x["steps"] for x in mine)
        tasks[t["name"]] = {"runs": len(mine), "successes": sum(1 for x in mine if x["success"]),
                            "mean_steps": r(st / len(mine), 1),
                            "s_per_step": r(sum(x["wall_s"] for x in mine) / st) if st else None,
                            "mean_wall_min": r(sum(x["wall_s"] for x in mine) / len(mine) / 60.0)}
    return {
        "runs": len(scored), "successes": successes,
        "success_rate": r(successes / len(scored), 3) if scored else None,
        "mean_steps": r(total_steps / len(scored), 1) if scored else None,
        "median_steps": statistics.median([x["steps"] for x in scored]) if scored else None,
        "mean_s_per_step": r(total_wall / total_steps) if total_steps else None,
        "median_s_per_step": r(statistics.median(step_times)) if step_times else None,
        "p90_s_per_step": r(percentile(step_times, 90)),
        "step_times_from": "trace" if traced else "results",
        "total_wall_min": r(total_wall / 60.0),
        "minutes_per_success": r(total_wall / 60.0 / successes) if successes else None,
        **breakdown,
        "tasks": tasks,
        "unscored": [x["name"] for x in runs if x["success"] is None],
        "warnings": warnings,
        "runs_detail": runs,
    }


def table(summary):
    lines = [f"{'task':<18} {'runs':>4} {'ok':>4} {'steps':>6} {'s/step':>7} {'min/run':>8}"]
    for name, t in summary["tasks"].items():
        lines.append(f"{name:<18} {t['runs']:>4} {t['successes']:>4} {t['mean_steps']:>6} "
                     f"{t['s_per_step'] if t['s_per_step'] is not None else '-':>7} {t['mean_wall_min']:>8}")
    s = summary
    lines.append("")
    lines.append(f"success {s['successes']}/{s['runs']}   mean steps {s['mean_steps']}   "
                 f"s/step mean {s['mean_s_per_step']} median {s['median_s_per_step']} p90 {s['p90_s_per_step']} "
                 f"(per-step times from the {s['step_times_from']})")
    lines.append(f"wall {s['total_wall_min']} min in total, {s['minutes_per_success']} min per successful task")
    if "prompt_s_per_step" in s:
        lines.append(f"per step (trace): prompt {s['prompt_s_per_step']} s, generation {s['gen_s_per_step']} s, "
                     f"client gap {s['client_gap_s_per_step']} s (median {s['client_gap_s_median']}), "
                     f"images in prompt {s['images_in_prompt_mean']}")
    failed = [x for x in s["runs_detail"] if x["success"] is False]
    if failed:
        lines.append("")
        lines.append("failed runs:")
        for x in failed:
            lines.append(f"  {x['name']} ({x['task_id']}): " + "; ".join(x["reasons"]))
    notes = sorted({n for x in s["runs_detail"] for n in x.get("notes", [])})
    for n in notes:
        lines.append(f"note: {n}")
    if s["unscored"]:
        lines.append("not in the suite: " + ", ".join(sorted(set(s["unscored"]))))
    for w in s.get("warnings", []):
        lines.append(f"warning: {w}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score a GhostAgent benchmark run.")
    ap.add_argument("--results", required=True, help="the agent's headless results JSONL")
    ap.add_argument("--bench-log", help="bench_server.py's JSONL log")
    ap.add_argument("--suite", default=str(HERE / "suite.json"))
    ap.add_argument("--steps", help="optional step trace JSONL (steps-*.jsonl)")
    ap.add_argument("--json", help="also write the JSON summary to this file")
    a = ap.parse_args(argv)
    for label, path in (("results", a.results), ("bench log", a.bench_log), ("step trace", a.steps),
                        ("suite", a.suite)):
        if path and not Path(path).is_file():
            print(f"The {label} file was not found: {path}")
            return 1
    results = read_jsonl(a.results)
    if not results:
        print(f"No results in {a.results}.")
        return 1
    suite = json.loads(Path(a.suite).read_text(encoding="utf-8-sig"))
    summary = score(results, suite, read_jsonl(a.bench_log), read_jsonl(a.steps))
    print(table(summary))
    text = json.dumps(summary, indent=2)
    if a.json:
        Path(a.json).write_text(text, encoding="utf-8")
    print()
    print(json.dumps({k: v for k, v in summary.items() if k not in ("runs_detail", "tasks")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
