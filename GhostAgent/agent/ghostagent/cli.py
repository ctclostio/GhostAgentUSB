"""Starting up: the command line, the interactive prompt and unattended runs (--task, --tasks-file)."""

import argparse
import json
import re
import sys
import time
import traceback
import urllib.parse
from datetime import datetime
from pathlib import Path

from . import paths, runstate
from .agentloop import ANSWER_OVERLAY, Agent, HELP, STOPPED_OVERLAY, read_task
from .browser import Browser
from .console import (_enable_ansi, c, clean, err, first_line, focus_console, head, info, one_line, set_console_title,
                      warn)
from .forgetpc import finish_forget_pc, forget_pc
from .llamaserver import ExternalServer, LlamaServer, MAX_SERVER_RESTARTS, parse_affinity_mask, port_is_free
from .modelclient import ForcedStop, StepCancelled, ctrl_c_stops_task
from .paths import (BENCH_DIR, ROOT, ensure_writable_data, hide_home, install_console_log, log_only, prune_logs,
                    prune_screenshots, set_data_dir, use_profile_location)
from .preflight import active_profile, check_model_files, check_ram
from .settings import load_settings
from .userbrowser import copy_to_clipboard
from .winproc import (already_running, bind_children_to_this_process, disable_quick_edit, keep_pc_awake,
                      quiet_crash_dialogs, restore_quick_edit)


# --------------------------------------------------------------------------------------------
# Starting up, the interactive prompt and unattended runs
# --------------------------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="ghost_agent.py",
        description="GhostAgent: a local vision model that drives a browser. Without arguments it asks for "
                    "tasks; with --task or --tasks-file it runs them unattended and writes one result line "
                    "per task (for benchmarks).")
    ap.add_argument("--task", action="append", default=[], metavar="TEXT",
                    help="run this task unattended (repeatable)")
    ap.add_argument("--tasks-file", metavar="PATH",
                    help='run the tasks in this JSON list of {"name", "task", "start_url"} unattended')
    ap.add_argument("--results", metavar="PATH",
                    help="unattended runs: add one JSON line per task to this file "
                         "(default: logs\\results-<date>-<time>.jsonl)")
    ap.add_argument("--settings-override", metavar="JSON",
                    help="settings applied over settings.json: a JSON object, or the path of a .json file")
    ap.add_argument("--browser", choices=["bundled", "edge"], help="use this browser, whatever the setting says")
    ap.add_argument("--emulate-free-vram-mib", type=int, metavar="N",
                    help="benchmarks: lay the model out as if the GPU had at most N MiB of free VRAM")
    ap.add_argument("--affinity-mask", metavar="HEX",
                    help="benchmarks: run llama-server only on the logical CPUs in this hex mask (for example ff)")
    ap.add_argument("--headless-browser", action="store_true", help=argparse.SUPPRESS)  # tests and CI only
    ap.add_argument("--data-dir", metavar="DIR", help=argparse.SUPPRESS)  # tests and CI only
    a = ap.parse_args(argv)
    if a.affinity_mask is not None:
        try:
            a.affinity_mask = parse_affinity_mask(a.affinity_mask)
        except ValueError:
            ap.error(f"--affinity-mask {a.affinity_mask!r} is not a hex CPU mask such as ff or 0xF0F")
    if a.emulate_free_vram_mib is not None and a.emulate_free_vram_mib < 0:
        ap.error("--emulate-free-vram-mib must be 0 or more")
    return a


def parse_override(text):
    """The --settings-override value: inline JSON, or the path of a JSON file."""
    if not text:
        return None
    raw = text
    if not text.lstrip().startswith("{") and Path(text).is_file():
        raw = Path(text).read_text(encoding="utf-8-sig")
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise SystemExit(f"--settings-override is not valid JSON ({e}).")
    if not isinstance(data, dict):
        raise SystemExit("--settings-override must be a JSON object, for example {\"max_steps\": 40}.")
    return data


def load_tasks(opts):
    """[{name, task, start_url}] from --task and --tasks-file."""
    tasks = [{"name": f"task-{i}", "task": t, "start_url": None} for i, t in enumerate(opts.task, 1) if t.strip()]
    if opts.tasks_file:
        path = Path(opts.tasks_file)
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as e:
            raise SystemExit(f"Could not read the tasks file {path} ({first_line(e)}).")
        if isinstance(data, dict) and isinstance(data.get("tasks"), list):
            data = data["tasks"]
        if not isinstance(data, list):
            raise SystemExit(f"{path} must hold a JSON list of tasks.")
        for i, t in enumerate(data, 1):
            if isinstance(t, str):
                t = {"task": t}
            if not isinstance(t, dict) or not str(t.get("task") or "").strip():
                raise SystemExit(f'Task {i} in {path} has no "task" text.')
            tasks.append({"name": str(t.get("name") or f"task-{i}"), "task": str(t["task"]),
                          "start_url": t.get("start_url") or None})
    return tasks


def start_bench_sites(tasks):
    """Start the local benchmark site (tools\\bench\\bench_server.py) when the tasks start on it and
    nothing serves that port yet, so a suite run is one command. Its log goes to the logs folder."""
    ports = set()
    for t in tasks:
        try:
            u = urllib.parse.urlsplit(t.get("start_url") or "")
            port = u.port
        except ValueError:
            continue
        if u.scheme == "http" and u.hostname in ("127.0.0.1", "localhost") and u.path.startswith("/start/") and port:
            ports.add(port)
    sites = []
    for port in sorted(ports):
        if not port_is_free(port):
            continue  # already running (started by hand)
        if not (BENCH_DIR / "bench_server.py").is_file():
            warn(f"The tasks start on the benchmark site (port {port}), but {BENCH_DIR} has no bench_server.py.")
            continue
        if str(BENCH_DIR) not in sys.path:
            sys.path.insert(0, str(BENCH_DIR))
        import bench_server
        log = paths.LOGS / f"bench-server-{datetime.now():%Y%m%d-%H%M%S}-{port}.jsonl"
        site = bench_server.start(port=port, log=log)
        info(f"benchmark site on {site.base} (log {log})")
        sites.append(site)
    return sites


def start_model_server(server, backend=None):
    """Wait until the model server answers, starting it first unless begin() already did (then
    `backend` is the build it chose). When the explicit GPU layout fails, try once more with llama.cpp
    choosing it; when a build cannot load the model at all (backend "auto"), go on with the next build
    that sees a GPU, and finally the CPU. Returns (backend, seconds) or (backend, None)."""
    if backend is None:
        backend = server.start()
    secs = server.wait_ready()
    while secs is None and not getattr(server, "quit_requested", False):
        if server.explicit_layout:
            warn("Retrying and letting llama.cpp decide how much of the model goes on the GPU ...")
            backend = server.start(auto_fit=True)
        elif getattr(server, "can_fall_back", lambda: False)():
            failed = backend
            backend = server.next_backend()
            warn(f"The {failed} build could not run the model; trying the {backend} build ...")
            server.start()
        else:
            break
        secs = server.wait_ready()
    if secs is not None:
        getattr(server, "remember_failed_backends", lambda: None)()
    return backend, secs


def recover_server(server, during_task=True):
    """The model server stopped (it crashed) during a task, or between tasks: show the end of its log
    and start it again, at most MAX_SERVER_RESTARTS times per session. True when it is ready again;
    False for a server the agent did not start (it leaves that alone). SystemExit, naming the log,
    when it cannot be started again."""
    log = server.log_path
    err("The model server has stopped. Last lines of its log:")
    for line in server.tail(20):
        print("    " + clean(line))
    if not getattr(server, "can_restart", lambda: False)():
        if getattr(server, "restarts", 0) >= MAX_SERVER_RESTARTS:
            raise SystemExit(f"The model server stopped again after {server.restarts} restarts, so GhostAgent ends "
                             f"here. Its log: {hide_home(str(log))}")
        return False
    info("starting the model server again ...")
    if server.restart() is None:
        raise SystemExit(f"The model server could not be started again. The log of the crash: {hide_home(str(log))}")
    info(f"The model server crashed and was restarted (the log of the crash: {log.name if log else '-'})."
         + (" Run the task again." if during_task else ""))
    return True


def run_interactive(settings, server, browser):
    set_console_title("GhostAgent - loading")
    forget_later = []  # what /forget-pc could not delete while it was in use
    last_answer = None
    try:
        # What needs the user comes first: the question about copying the model to this PC and which
        # GPU is used, then the model starts loading in the background while the browser starts, so
        # Chrome's "Allow" click and the load overlap instead of following each other.
        backend = server.begin() if hasattr(server, "begin") else None
        if getattr(browser, "attached", False):
            info(f"connecting to your {browser.user_browser_name} ...")
        else:
            info("starting the browser ...")
        try:
            browser.start()
        except StepCancelled:
            print()
            warn("Stopped.")
            return 0
        backend, secs = start_model_server(server, backend)
        if secs is None:
            raise SystemExit(1)
        info(f"model ready in {secs:.0f}s  (backend {backend}, {server.describe()})")
        agent = Agent(settings, server, browser)
        set_console_title("GhostAgent - ready")
        focus_console()
        print()
        print(HELP)
        while True:
            print()
            try:
                task = read_task(c("Task> ", "1;32")).strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print()
                continue
            if not task:
                continue
            log_only("Task> " + task)
            if "\n" in task:
                info(f"Task received ({task.count(chr(10)) + 1} lines)")
            if task.lower() in ("/quit", "/exit", "quit", "exit"):
                break
            if task.lower() == "/help":
                print(HELP)
                continue
            if task.lower() == "/new":
                agent.new_conversation()
                info("The next task starts a new conversation"
                     + (" (with a short summary of the earlier tasks)." if agent.history else "."))
                continue
            if task.lower() in ("/forget-pc", "/forgetpc"):
                forget_pc(settings, forget_later)
                continue
            if task.lower() == "/copy":
                if last_answer is None:
                    info("There is no answer to copy yet.")
                elif copy_to_clipboard(last_answer):
                    info("The last answer is on the clipboard.")
                else:
                    warn("Could not put the answer on the clipboard.")
                continue
            # Before the task: a server that crashed since the last one is started again, and a
            # connection to the user's browser that was lost is made again.
            if getattr(server, "proc", None) is not None and not server.alive():
                if not recover_server(server, during_task=False):
                    break
            try:
                with ctrl_c_stops_task(clear=False):
                    getattr(browser, "ensure_connected", lambda: None)()
            except StepCancelled:
                print()
                warn("Stopped.")
                continue
            t0 = time.time()
            server_died = False
            # Ctrl+C from here on only asks the task to stop (see ctrl_c_stops_task); the answer and
            # error overlays below are browser calls too, so they stay inside.
            with ctrl_c_stops_task(), keep_pc_awake():
                try:
                    result = agent.run(task)
                    last_answer = result
                    print()
                    outcome = (agent.last or {}).get("answer_outcome")
                    print(c(f"Answer ({outcome}):" if outcome else "Answer:", "1;35"), clean(result))
                    info(f"({time.time() - t0:.0f}s)")
                    browser.overlay("answer", ANSWER_OVERLAY)
                except StepCancelled:
                    print()
                    warn("Stopped.")
                    browser.overlay("remove")
                except Exception as e:
                    print()
                    err(f"Task failed: {e}")
                    # (Agent.run started a new conversation: say so, or a reply such as "yes, go on"
                    # would reach a model that no longer has the task in front of it.)
                    info("The next task starts a new conversation, with a short summary of the earlier tasks.")
                    server_died = not server.alive()
                    if not server_died:
                        browser.overlay("answer", STOPPED_OVERLAY)
            if server_died and not recover_server(server):
                break
            set_console_title("GhostAgent - ready")
            focus_console()
    finally:
        info("shutting down ...")
        note = getattr(browser, "quit_note", None)
        if note:
            info(note)
        browser.close()
        server.stop()
        finish_forget_pc(forget_later)
    return 0


def _server_start_failed(e):
    """run_tasks: the model server could not be started (SystemExit e). Shows the reason; exit code 2."""
    if e.code in (0, None):
        raise e
    if isinstance(e.code, str):
        err(e.code)
    err("The model server did not start, so no task was run.")
    return 2


def run_tasks(opts, settings, server, browser, tasks):
    """Unattended runs (benchmarks): every task starts fresh (no earlier tasks in its prompt, one
    tab at its start page), its result goes to the results file as one JSON line, and nothing waits
    for a key press. Exit code 2 when the model server fails, 1 when stopped with Ctrl+C."""
    results = Path(opts.results) if opts.results else paths.LOGS / f"results-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    results.parent.mkdir(parents=True, exist_ok=True)
    if results.exists() and results.stat().st_size:
        warn(f"{results} already has results; the new ones are added at the end. score.py pairs results "
             "with the benchmark site's log in order, so a new file per run is easier to score.")
    sites = start_bench_sites(tasks)
    code = 0
    try:
        server.unattended = True
        # A start that fails outright (a blocked program, no usable port) raises SystemExit with the
        # reason. Benchmark scripts tell a model-server failure apart by exit code 2, so it is mapped
        # here instead of ending as a generic error (1).
        try:
            backend = server.begin(interactive=False) if hasattr(server, "begin") else None
        except SystemExit as e:
            return _server_start_failed(e)
        info("starting the browser ...")
        browser.start(start_url="about:blank")
        try:
            backend, secs = start_model_server(server, backend)
        except SystemExit as e:
            return _server_start_failed(e)
        if secs is None:
            err("The model server did not start, so no task was run.")
            return 2
        info(f"model ready in {secs:.0f}s  (backend {backend}, {server.describe()})")
        agent = Agent(settings, server, browser)
        totals = {"answer": 0, "steps": 0, "wall": 0.0}
        with keep_pc_awake():
            for i, t in enumerate(tasks, 1):
                print()
                head(f"[{i}/{len(tasks)}] {t['name']}")
                print(c("  task: " + one_line(clean(t["task"]), 300), "37"))
                agent.reset()  # a new conversation with nothing from the earlier tasks
                agent.last = None
                answer, cancelled = None, False
                with ctrl_c_stops_task():
                    try:
                        browser.reset_for_task(t.get("start_url") or settings["start_url"])
                        answer = agent.run(t["task"])
                    except StepCancelled:
                        cancelled = True
                    except Exception as e:
                        answer = f"Error: {first_line(e)}"
                        err(f"Task failed: {first_line(e)}")
                last = agent.last or {"task_id": None, "steps": 0, "outcome": "error", "wall_s": 0.0}
                outcome = "cancelled" if cancelled else last["outcome"]
                row = {"name": t["name"], "task_id": last["task_id"], "answer": answer, "outcome": outcome,
                       "steps": last["steps"], "wall_s": last["wall_s"],
                       "s_per_step": round(last["wall_s"] / last["steps"], 3) if last["steps"] else None}
                if last.get("answer_outcome"):
                    row["answer_outcome"] = last["answer_outcome"]  # tool_schema "h": the model's own verdict
                with open(results, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                totals["answer"] += outcome == "answer"
                totals["steps"] += last["steps"]
                totals["wall"] += last["wall_s"]
                if outcome == "answer":
                    verdict = last.get("answer_outcome")
                    print(c(f"Answer ({verdict}):" if verdict else "Answer:", "1;35"), clean(answer))
                info(f"{outcome} after {last['steps']} steps in {last['wall_s']:.1f}s")
                if cancelled:
                    warn("Stopped: the remaining tasks were not run.")
                    code = 1
                    break
                if outcome == "error" and not server.alive():
                    err("The model server has stopped. Last lines of its log:")
                    for line in server.tail(20):
                        print("    " + clean(line))
                    code = 2
                    break
        print()
        info(f"{totals['answer']} of {len(tasks)} tasks answered, {totals['steps']} steps, "
             f"{totals['wall'] / 60:.1f} min. Results: {results}")
        return code
    finally:
        browser.close()
        server.stop()
        for site in sites:
            site.stop()


def main(argv=None):
    opts = parse_args(argv)
    tasks = load_tasks(opts) if (opts.task or opts.tasks_file) else None
    runstate.HEADLESS = tasks is not None
    if runstate.HEADLESS and not tasks:
        raise SystemExit("There are no tasks to run.")
    _enable_ansi()
    quiet_crash_dialogs()
    if not runstate.HEADLESS:
        disable_quick_edit()
    if opts.data_dir:
        set_data_dir(opts.data_dir)
    moved = ensure_writable_data()
    install_console_log()
    if moved:
        warn(moved)
    prune_logs()
    prune_screenshots()
    settings = load_settings(parse_override(opts.settings_override), interactive=not runstate.HEADLESS)
    use_profile_location(settings)
    if opts.browser:
        settings["browser"] = opts.browser
    if runstate.HEADLESS:
        # Nobody is there to click Chrome's "Allow", to judge a page's question or to wait for a copy.
        if settings["browser"] in ("chrome", "user_edge", "auto"):
            settings["browser"] = "bundled"
        settings["dialog_policy"] = "dismiss"
        settings["copy_model_to_pc"] = False
    external = settings["external_server"]
    if not external and already_running():
        raise SystemExit("GhostAgent is already running in another window.")
    bind_children_to_this_process()
    print()
    head(f"  GhostAgent{release_version()} - local browser agent  ({model_label(settings)} on llama.cpp + Playwright, "
         "runs entirely on this computer)")
    print()

    if external:
        info(f"using the model server at {external} (external_server); no model is loaded here")
        server = ExternalServer(external, settings["external_server_key"])
    else:
        check_model_files(settings)
        check_ram(settings, interactive=not runstate.HEADLESS)
        server = LlamaServer(settings)
        server.emulate_free_vram_mib = opts.emulate_free_vram_mib
        server.affinity_mask = opts.affinity_mask
    browser = Browser(settings, headless=opts.headless_browser)
    if runstate.HEADLESS:
        return run_tasks(opts, settings, server, browser, tasks)
    return run_interactive(settings, server, browser)


def release_version():
    """The release from the VERSION file next to the GhostAgent folder, with a space in front (" 0.2.0"),
    or "" without one."""
    try:
        text = (ROOT.parent / "VERSION").read_text(encoding="utf-8-sig").strip().splitlines()[0].strip()
    except (OSError, IndexError):
        return ""
    return " " + text if re.fullmatch(r"[0-9A-Za-z.+-]{1,32}", text) else ""


def model_label(settings):
    """The model in the banner: the profile's label without its note in brackets."""
    _, prof = active_profile(settings)
    own = Path(str(settings.get("model") or "")).name
    if own and own.lower() != str((prof.get("model") or {}).get("name", "")).lower():
        return Path(own).stem  # a model file of the user's own
    label = str(prof.get("label") or "Holo4-35B-A3B")
    return re.sub(r"\s*\([^)]*\)\s*$", "", label) or label


def _pause():
    if runstate.HEADLESS:
        return
    try:
        input("\n  Press Enter to close this window.")
    except BaseException:
        pass


def run_main(argv=None):
    """main(), plus what happens when it ends badly: the message stays on screen until Enter is
    pressed (the window would otherwise close at once). Returns the process exit code."""
    # An unattended run is known before the command line is checked, so a mistyped option in a
    # benchmark script ends with its exit code instead of waiting for Enter (main() decides again).
    if any(str(a).startswith("--task") for a in (sys.argv[1:] if argv is None else argv)):
        runstate.HEADLESS = True
    try:
        return _run_main(argv)
    finally:
        restore_quick_edit()


def _run_main(argv):
    try:
        return main(argv) or 0
    except ForcedStop:
        print()
        warn("Stopped.")
        _pause()
        return 1
    except KeyboardInterrupt:
        return 1 if runstate.HEADLESS else 0
    except SystemExit as e:
        if e.code in (0, None):
            return 0
        if isinstance(e.code, str):
            err(e.code)
        _pause()
        return e.code if isinstance(e.code, int) else 1
    except BaseException:
        traceback.print_exc()
        _pause()
        return 1
