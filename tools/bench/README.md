# Benchmark and test tools

Everything here runs with the stick's Python (`python\python.exe -I ...`), uses only the standard
library (plus Playwright for browser tests), and talks only to 127.0.0.1. None of it needs a GPU or
a model file.

| File | What it is |
| --- | --- |
| `fake_llama.py` | A scripted stand-in for `llama-server`: the same `/health`, `/props`, `/tokenize` and streaming `/v1/chat/completions` replies, fault modes (an error chunk mid-stream, a stream that ends without `[DONE]`, a cut connection, an HTTP error, delays), an optional API key, a request recorder and llama.cpp-style log lines. |
| `bench_server.py` | Serves `site\` on 127.0.0.1:8765 and logs form POSTs, `/log` beacons (click targets), any hit on `/exfil` and downloads to a JSONL file. |
| `site\` | The 12 benchmark task pages, plus `site\fixtures\` (small pages for unit tests: dialogs, scheme links, popups, downloads, a busy page, and so on; see `/fixtures/`). |
| `suite.json` | The 12 tasks: `name`, `task` (what a user would type), `start_url`, `page` and a deterministic `check`. |
| `score.py` | Scores a run from the agent's headless results plus the bench server log. |
| `replay.py` | Replays recorded request bodies against a llama-server to measure prompt and generation cost. |

The unit tests for all of this are in `tests\` at the repo root:

    GhostAgent\python\python.exe -I tests\run_tests.py

## A suite run

1. Start the site: `python\python.exe -I tools\bench\bench_server.py --log logs\bench-server.jsonl`
   (use a new log file for every suite run; without `--log`, each server start gets its own
   time-stamped file in the logs folder)
2. Run the agent headless over `suite.json` (one result line per task: `name, task_id, answer, outcome, steps, wall_s, s_per_step`).
3. Score it: `python\python.exe -I tools\bench\score.py --results logs\bench.jsonl --bench-log logs\bench-server.jsonl [--steps logs\steps-YYYYMMDD.jsonl]`

Every task starts at `/start/<name>`, which resets the server's state, opens a new run in its log and
redirects to the task's first page. That is how server events are matched to agent results, so a
runner must open each task's `start_url` exactly once per run. The k-th result of a task is paired
with the k-th run of that task in the log, and `score.py` prints a warning when the counts differ
(for example, a log that also holds an earlier suite run). On a port other than 8765,
`bench_server.load_suite(path, port)` rewrites the start URLs.

## Replay recordings

`replay.py` reads this layout (the agent writes it when `trace_messages` is on):

    <replay dir>\<task_id>\<step>.json       the exact request body of that step
    <replay dir>\<task_id>\<step>-<k>.png    the k-th image in that body

Inside `<step>.json`, each image part keeps its normal shape, but its URL is just the file name:
`{"type": "image_url", "image_url": {"url": "7-3.png"}}`. `replay.write_step()` is the reference
writer. Run, for example:

    python\python.exe -I tools\bench\replay.py --replay-dir logs\replay --url http://127.0.0.1:18931 ^
        --api-key KEY --mode both --server-log logs\llama-server-X.log --server-pid PID --label baseline

`--mode prompt` sends each step with `max_tokens` 1 (prompt cost only); `--mode full` sends it as
recorded with a fixed `--seed`. Results go to a JSONL file and a `.summary.json` beside it.

Start each replay on a freshly started server. A second pass in the same replay (`--repeats` above
1, or `--mode both`) sends prompts the server has already seen, and with its RAM prompt cache on
(`--cache-ram` above 0) it can restore them instead of processing them, which makes that pass look
faster than a real run.
