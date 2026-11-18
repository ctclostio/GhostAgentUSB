"""GhostAgent's agent. GhostAgent\\agent\\ghost_agent.py starts it (cli.run_main); the README and
DECISIONS.md (next to the GhostAgent folder) describe what it does and why.

The modules, from the bottom up: each one imports only modules listed above it.

  runstate      run-wide state: HEADLESS (an unattended run) and STOP (Ctrl+C asked the task to stop)
  console       console output: colours or plain lines, progress lines, safe printing, the window
  paths         where the agent writes: logs, downloads, browser profile, settings.json, the console log
  browsercache  the agent's browser's disk cache in TEMP, and its lock
  settings      the defaults, settings.default.json under settings.json, tools\\profiles.json
  preflight     the active model profile and its files, the RAM check
  winproc       child processes and Windows: job object, blocked programs, sleep, Quick Edit
  backends      which llama.cpp build and which GPU
  serverflags   the llama-server flags the installed build knows
  gguf          the model's layout from its GGUF header
  modelclient   the streaming chat request, and Ctrl+C during a task
  questions     questions in the console (confirm(), prompt(), a risky click)
  modelcopy     the model copy on this PC (copy_model_to_pc)
  screenshot    screenshot size in Holo4's 32-pixel tiles
  llamaserver   llama-server (and external_server): start, load, watch, restart
  tools         the tools the model is offered, and their arguments under either schema
  prompts       the system prompt and the message that starts a task
  keys          press_keys_web: key names and what a press would do
  pagejs        JavaScript the browser runs in pages
  userbrowser   the user's own Chrome or Edge, and the clipboard
  guards        where the agent may go, and what needs the user's yes
  pagehelpers   page scripts with a time limit, stalls, the observation header, reader_mode, downloads
  browser       the browser the agent drives (Playwright)
  history       the conversation's history: screenshots leave, old results are shortened
  steptrace     the step trace (logs\\steps-*.jsonl) and replay recordings
  agentloop     the agent loop: one task, step by step
  forgetpc      /forget-pc
  cli           start-up: the command line, the Task> prompt and unattended runs

A name the agent changes while it runs (runstate.HEADLESS, console.PLAIN_OUTPUT, paths.LOGS,
paths.PROFILE, ...) is read through its module, never imported by name: a from-import would keep
the value it had when the importing module was loaded.
"""

import sys

# No __pycache__ folders on the drive (ghost_agent.py sets this before the first import, too).
sys.dont_write_bytecode = True
