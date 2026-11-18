"""Run-wide state that every part of the agent reads.

HEADLESS is set at start-up (cli.main), so the other modules read it as runstate.HEADLESS at call
time; a from-import would keep the value it had when the module was loaded."""

import threading


# True for unattended runs (--task / --tasks-file): nothing waits for Enter, nothing grabs focus.
HEADLESS = False

# Set by Ctrl+C while a task runs. The agent stops at the next safe point instead of having a
# KeyboardInterrupt land inside a Playwright call, which would leave Playwright unusable.
STOP = threading.Event()
