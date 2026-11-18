"""GhostAgent: a local vision model (llama.cpp) that drives a real browser (Playwright).

Everything runs from this folder. Nothing is sent to any cloud service: the model is served by
llama-server on 127.0.0.1. The browser, by setting, is a window in the user's own running Chrome
or Edge (the default: their Chrome), the installed Chrome with a separate profile of the agent's
own, or a browser whose profile is kept on this drive (a portable Chromium from this folder, or
Edge). Only the websites you ask the agent to visit need internet.

The model is a profile from tools\\profiles.json: Holo4-35B-A3B at Q4_K_M ("fast", the default) or
Q8_0 ("fast-q8"), or Holo4-27B ("accurate", a non-commercial license). The agent loop follows H
Company's documented Holo4 conventions:
  * screenshots go in a user message wrapped in <observation> tags, at most the last 3 kept
    (image_eviction: the oldest one each step, or several at once so the prompt only grows in between)
  * coordinates are integers in [0, 1000] relative to the screenshot
  * thinking on, temperature 0.8, top_k 20, top_p 0.95 and no min_p; every tool call of a reply
    runs, in order, and the rest of a step is skipped once a call fails or the page changes
  * a task ends when `answer` is called; the user's next task continues the same conversation
    (/new starts another), and a history that outgrows the context keeps one line per older step
  * the model's earlier reasoning goes back into its context only with preserve_reasoning (on for
    the accurate profile; off for the fast ones until a benchmark decides); otherwise only its
    message text and tool calls do

Several choices that change what the model sees or how it acts are benchmark switches whose
defaults keep the original behaviour until a benchmark decides: image_eviction,
observation_format, prompt_version, tool_schema, write_enter_default, settle_mode and
snap_viewport. DECISIONS.md (next to the GhostAgent folder) records why each server flag and
default is what it is.

Settings: settings.default.json lists every setting with a note; settings.json (created on the
first start) holds only the ones the user changed.

Command line. Without arguments the agent is interactive (the Task> prompt). For benchmarks and
tests it also runs a list of tasks unattended and writes one result line per task:
    ghost_agent.py --tasks-file tools\\bench\\suite.json --results logs\\bench.jsonl --browser bundled
    ghost_agent.py --task "find ..." [--task ...] [--settings-override JSON-or-file]
  --emulate-free-vram-mib N   lay the model out as if the GPU had at most N MiB free (benchmarks)
  --affinity-mask HEX         keep llama-server on these logical CPUs (benchmarks)
  --headless-browser          tests only: the bundled browser without a window, throwaway profile
  --data-dir DIR              tests only: logs, downloads, browser profile and settings.json in DIR

Safety rails outside the model: the browser opens only http(s) pages (and addresses on this PC or the
local network only when the task names them, allowed_hosts lists them or the user opened them),
native confirm/prompt dialogs and clicks on buttons such as Delete, Send or Pay are put to the user
in the console, password and one-time-code fields are never typed into, pasting from the clipboard
is refused, and the model server takes requests only with a key made for this run.

The code is in the ghostagent package next to this file (its __init__.py lists the modules and
what each one holds); this script only starts it.
"""

import sys
from pathlib import Path

# Started as "python -I ghost_agent.py": isolated mode leaves this script's folder off sys.path, so it
# goes first here (it is the agent's own folder on the drive, which holds the ghostagent package). No
# bytecode is written, so no __pycache__ folders appear on the drive.
sys.dont_write_bytecode = True
AGENT_DIR = str(Path(__file__).resolve().parent)
if sys.path[:1] != [AGENT_DIR]:
    sys.path.insert(0, AGENT_DIR)

from ghostagent.cli import run_main  # noqa: E402  (after the sys.path line above)

if __name__ == "__main__":
    sys.exit(run_main())
