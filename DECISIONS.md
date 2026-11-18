# Decisions

Why GhostAgent runs the model server and the agent the way it does: every llama-server flag, the request settings, the main design choices, and the options that were tried or considered and rejected, so they are not tried again blindly. Each entry says what it rests on. "Not measured yet" means the choice waits for the benchmark; when a benchmark changes a default, update the entry with the result and the date.


## Where the numbers come from

| Source | When | What |
|---|---|---|
| Laptop logs | October 2026 | llama-server logs of 27 real tasks (598 model requests) on a laptop with an 8 GB NVIDIA GPU and 64 GB of RAM: fast profile, llama.cpp b11669 CUDA 13 build, 8 threads, model loaded from a copy on the SSD, the agent as it was before version 0.2.0. |
| Desktop benches | October 2026 | Flag comparisons during the original build, on a desktop with a large NVIDIA GPU, before the laptop runs. |
| Desktop GPU checks | 13 November 2026 | The three model profiles on a desktop GPU held to the laptop's free VRAM (7070 MiB) with `--emulate-free-vram-mib`, the agent before 0.2.0: probe prompts, a 10k-token prompt with a screenshot, and 20 recorded steps replayed (10 for the 27B at `medium`). |
| Source reading | November 2026 | llama.cpp b11669 and b12359 source and `--help`, the Holo4 GGUF headers and chat templates, H Company's published docs. |

Baseline from the laptop logs: a median of 22.9 s of server time per step, 70% of it spent reading the prompt; generation at a median 25.8 tokens/s; reading the prompt at about 375-390 tokens/s; a median of 26 steps per task, with 10 of the 27 tasks reaching the 40-step limit that existed then; 40-step tasks took 16 to 21 minutes. The largest context any step used was about 18.8k tokens.

The benchmark that will settle the open items is in `tools/bench`: a replay of recorded requests for speed-only changes, and 12 local tasks with automatic checks, run with 3 seeds each, for anything that changes what the model sees. A speed-only change is adopted when mean seconds per step improve by at least 3% with no task getting worse beyond noise. A change to what the model sees becomes a default only if, paired by task and seed, successes drop by at most 1 run in 36 and no kind of task goes from mostly passing to always failing; otherwise it stays an opt-in setting with the measured trade-off in its note. Final defaults are confirmed on the laptop.


## llama.cpp version

- **Default: b11669.** The build the laptop numbers were measured with.
- **Available: b12359 (the v1.0.0 release)**, with `"Build GhostAgent.cmd" -LlamaTag b12359`; run again with `-LlamaTag b11669` to go back. It brings:
  - an image resize that matches the reference Qwen-VL preprocessing (b11669 pads screenshots with black bars to fit the patch grid, which can shift clicks near the edges);
  - faster flash attention for the vision encoder's head size, fused MoE kernels, a better expert-cache eviction policy and cheaper checkpoints (upstream measurements, on other hardware);
  - tool-argument sanitising, vision-projector validation, long-path fixes, and a fix for stale experts on Vulkan.

  Its CUDA 13 build is CUDA 13.5, which needs NVIDIA driver 580 or newer; the agent then uses the CUDA 12 build. It becomes the default once a replay shows it is not slower and a grounding check (clicks on labelled screenshot targets, including near the edges) and the tool-call parse rate show no regression. **Not measured yet.**
- **Not b12360 to b12366 yet.** They add a rewrite of the expert cache and a grammar change that touches forced tool calls (`tool_choice: required`). Wait a release or two and re-run the replay first.
- Every flag the agent passes by default is accepted by both builds; `--moe-cache-cpu-share` exists only from b12359 on. `tools/server-flags.json` lists every flag the agent may pass. The build's smoke test checks each llama.cpp build's `--help` against it (a missing required flag fails the build; a missing optional one is a note, or a warning for the lock-down flags), and at run time the agent leaves out an optional flag a build does not know, with a one-line notice. `--no-webui` is passed by default but listed as optional, like the other lock-down flags: `web_ui` leaves it out, and a build without it can still run the agent.


## Server flags

### Always passed

| Flag | Why | Rests on |
|---|---|---|
| `-m`, `--mmproj` | The model and its vision projector, from the chosen profile in `tools/profiles.json`. | Design |
| `--alias <profile alias>` | Requests name the model by profile, so the log says which model answered. | Design |
| `--host 127.0.0.1` | The model server is never reachable from the network. | Design |
| `--port 18931` | GhostAgent's own port. If it is taken, the next port that can be bound and has no listener is used, up to 50 tries: the bind test catches ports Windows reserves, the connect test catches programs listening on all addresses. | Design |
| `-c 65536` (fast profiles), `-c 32768` (accurate) | Plenty for real tasks (largest seen: about 18.8k tokens); the history is condensed at three quarters full. The 27B's KV cache costs about three times as much per token (at 64k and q8_0, about 2.3 GB against 0.7 GB), so it gets 32k. `ctx_size` overrides both. | Laptop logs; model configs |
| `-np 1` | One agent, one conversation, one slot. | Design |
| `-fa on` | Flash attention, which llama.cpp needs for a quantized V cache. | llama.cpp |
| `-ctk q8_0 -ctv q8_0` | Half the KV cache memory of 16-bit, which matters with 8 GB of VRAM. | Original build |
| `--reasoning-format deepseek` | The model's thinking comes back separately from its message, so the agent can show it and decide whether to send it back. | Design |
| `--load-mode none` | The weights are read into the server's own memory instead of memory-mapping the file. The model is normally loaded from the PC's SSD copy; with a mapped file on a USB stick, pages would be read back from the stick. | Original build |
| `--image-min-tokens 1024` | H Company's llama.cpp recipe; llama.cpp itself warns that Qwen-VL models need about 1024 image tokens to locate things reliably. A 1280x800 screenshot is enlarged slightly to reach it (1066 tokens); `snap_viewport` (1280x832, 1040 tokens) avoids the resample. | H Company docs; llama.cpp source |
| `--image-max-tokens 1280` | A cap just above a normal screenshot (a 1280x800 one is 1089 prompt tokens), so normal screenshots are untouched; it matches the 1.3-megapixel big-screenshot warning. In the desktop check on 13 November it shrank the vision warm-up buffer by about 106 MiB on both models (35B: 5920 to 5814 MiB; 27B: 7168 to 7062 MiB after one screenshot) with identical answers. Clicks are in 0-1000 coordinates relative to the screenshot, so a bigger screenshot that the server scales down still clicks the right place. Optional: a build without the flag starts without the cap, with a notice. | Desktop GPU checks; llama.cpp source |
| `-b 2048 -ub 2048` | One micro-batch holds a whole screenshot (about 1,040 to 1,100 tokens) plus its text. With the experts in system RAM, the experts a micro-batch needs are copied to the GPU once per micro-batch, so bigger micro-batches mean fewer copies. | Desktop benches |
| `-t <threads>` | One thread per performance core on CPUs that mix performance and efficiency cores (at most 8), otherwise 8. On the desktop, 8 threads were as fast as 16 or 24. On the laptop's hybrid CPU, generation (25.8 tokens/s, about half the desktop's) looks bound by latency rather than memory bandwidth, so 6, 8, 10 and 12 threads and performance-core-only masks are to be compared there. **Not measured yet.** | Desktop benches |

### Prompt cache

Holo4 is a hybrid model: the 35B-A3B mixes 30 Gated DeltaNet layers, whose state is recurrent, with 10 attention layers. A recurrent state cannot be wound back token by token, so llama-server can resume a prompt only from a saved checkpoint at or before the first token that changed, and it drops every checkpoint after that point. When an old screenshot leaves the context, the prompt changes near its start.

In the laptop logs, from the first dropped screenshot (step 4) onward, the only usable checkpoint was the one at the start of the task (about 2.1k tokens of system prompt and task), so every step read the rest of the history again: the prompt grew from about 4k to about 12k tokens over a 40-step task, and in one such task steps went from about 17 s to about 33 s. That re-reading is most of the 70% of server time spent on prompts.

| Flag | Why | Rests on |
|---|---|---|
| `--checkpoint-min-step 1536` (`checkpoint_min_step`) | With llama.cpp's default spacing of 8192 tokens, the only checkpoint before a dropped screenshot is the one at the task start. A checkpoint every 1536 tokens keeps one close behind the change. 1536 is provisional: two simulations disagreed, one showed 1024 slowing tasks under 15 steps, and too small a spacing adds GPU micro-batches (about 0.5 s each). The sweep is 0, 256, 1024, 1536, 2048 and 8192, each with 8 and 32 checkpoints. **Not measured yet.** | Laptop logs; llama.cpp source |
| `--ctx-checkpoints 8` (`ctx_checkpoints`) | The simulations showed 8 and 32 giving the same step cost. Each checkpoint takes about 63 MiB of RAM (about 150 MiB for the 27B), so 32 would cost about 1.5 GB more. llama.cpp's own default is 32. | Simulation |
| `--cache-ram 0 --no-cache-idle-slots` (`cache_ram_mib`) | llama.cpp's RAM prompt cache (8 GiB by default) saves a copy of the slot whenever a request keeps less than half of it. For one agent on one conversation, an older saved state can never beat the live one, so it is pure overhead: in the laptop logs it filled up and kept evicting 380-570 MiB entries, and each save took a median of about 150 ms on a quarter of the steps. It also competes with the expert weights for RAM on a 32 GB PC. `--no-cache-idle-slots` only silences the warning that `--cache-ram 0` would otherwise log. H Company's recipe uses `--cache-ram 32768`, which is not adopted here for the same reasons. | Laptop logs; llama.cpp source |

### Model layout

**Fast profiles (mixture of experts), with enough free VRAM:** `-ngl 99 --cpu-moe --fit off --moe-cache-mib <auto>`. Every layer is on the GPU except the expert weights, which stay in system RAM; the VRAM left over caches the most-used experts. This measured faster than a static split (see Rejected). `auto` is free VRAM minus a reserve minus `fit_target_mib` (1024), at most `moe_cache_cap_mib` (4096).

- The reserve was a fixed 4600 MiB. Version 0.2.0 works it out from the GGUF header: everything that is not an expert, the vision projector, the KV cache, the recurrent state and 1024 MiB of compute buffers. For the fast model at 64k it comes to 4326 MiB, close to the roughly 4500 MiB measured during the original build.
- On the laptop (7070 MiB free) the old formula gave a 1446 MiB expert cache. In the desktop check, the fast profile peaked at about 6 GB of VRAM under that budget.
- The cache serves only batches of 32 tokens or fewer, so it speeds up generation, not prompt reading.
- Whether a bigger cache pays off on 12-24 GB cards (and the 4096 cap should scale with VRAM), and whether a smaller safety margin is safe on 8 GB: **not measured yet**.

With too little free VRAM for that layout, llama.cpp fits the layers itself (`--fit-target`), and a load that fails with the explicit layout is retried that way once. That margin includes the vision projector and 384 MiB more, as for the accurate profile below: llama.cpp's fit never counts the projector, whichever model it lays out.

**Accurate profile (dense):** `--fit-target <fit_target_mib + projector size + 384 MiB>`, and llama.cpp decides how many layers fit. Its fit does not count the vision projector: in the desktop check the process ended about 1.1 GB above the fit budget (the 16-bit projector, the vision buffers and the CUDA context), which put the 27B over the laptop's budget. The margin now includes the projector and 384 MiB more.

### Reasoning

| Flag | Why | Rests on |
|---|---|---|
| `--no-reasoning-preserve` (fast profiles only, unless `preserve_reasoning` is on) | The 35B's chat template shows no past reasoning by default, and the flag matches that. The 27B's template renders a thinking block for every past turn (empty when nothing is sent back), which is how H Company serves it, so the dense profile keeps the server's default. With `preserve_reasoning` on, the flag is left off and the agent sends the reasoning back. | Chat templates |

### Lock-down

| Flag | Why | Rests on |
|---|---|---|
| `LLAMA_API_KEY` in the server's environment | Before 0.2.0 the server answered any request, from any web page's origin, with no key, while the agent browsed arbitrary sites in the same browser. Now a random key is made at every start and handed over in the environment, so it is never on the command line or in a log. `/health` stays open, so loading can be watched. | Security review |
| `--cors-origins localhost` | Browsers may call the server only from pages on localhost. | Security review |
| `--no-slots` | The slot API (which can save and restore state) is off. | Security review |
| `--no-webui` (unless `web_ui`) | The chat page is off unless asked for; with `web_ui` the console shows the key it needs. | Security review |
| Cleared `LLAMA_*` and `AIP_*` variables | llama-server reads settings from environment variables, so a PC-wide one could turn on tools or loosen the server. | Security review |

Checked on b11669 without a model: `/health` answers without a key, chat requests without the key are refused, a foreign origin gets no CORS header, a `LLAMA_ARG_*` variable set before the start does not reach the server, and the key is not in the log. `extra_server_args` that loosen any of this (or set a key) get a warning.

### Tuning knobs, not passed by default

| Setting | What it does | Status |
|---|---|---|
| `moe_cache_cpu_share` | `--moe-cache-cpu-share` (b12359 and later): runs some missed experts on the CPU. b12359's docs say it rarely helps when the experts sit in pinned memory, as here. | Measure only |
| `op_offload_min_batch` | `GGML_OP_OFFLOAD_MIN_BATCH`: prompt micro-batches of at least this many tokens (default 32) copy the CPU-side experts to the GPU; smaller ones run on the CPU. On the laptop a GPU micro-batch cost about 0.75 s fixed, against about 0.17 s for one under 32 tokens, so the small text batches between screenshots (100 to 250 tokens) might be cheaper on the CPU. The sweep is 32, 48, 64 and 96. | Measure only |
| `graceful_server_stop` | Stops the server with Ctrl+Break first, so it can write its closing expert-cache statistics. For tuning runs. | Tooling |


## Request settings

| Setting | Value | Why |
|---|---|---|
| `temperature` | 0.8 | What H Company uses for Holo4 agent loops. |
| `top_k`, `top_p`, `min_p` | 20, 0.95, 0 | Holo4's generation config. Sent explicitly, because llama.cpp otherwise adds its own `min_p` of 0.05. |
| `max_tokens` | 4096 | In the laptop logs the median step generated 128 tokens, the 99th percentile about 920 and the longest 2888, so 4096 would have cut none of them. Without a cap, a reasoning loop runs until the context is full (estimated at over half an hour on the laptop). A reply cut off by the cap gets a note asking the model to think briefly and act; three unusable replies in a row end the task. |
| thinking | on | How H Company runs Holo4. |
| `tool_choice` | `required` | Every step must act or answer. |
| `reasoning_effort` | template default (xhigh) | Accurate profile only; the 35B's template has no such switch, so it is never sent there. In the desktop check a 27B step took a median of about 80 s at xhigh and about 50 s at `medium` (H Company's Agents API default); both are far slower than the fast profile's 11 s on the same desktop. |
| `seed` | random | A fixed seed for benchmark runs. |


## Agent design

| Choice | Why | Status |
|---|---|---|
| At most 3 screenshots in context (`image_budget`) | H Company's advice is at most 3. | Kept |
| Screenshot eviction: `rolling` (one leaves per step) vs `block` (several leave at once, so the prompt only grows in between) | Block mode keeps the history append-only between drops, so the server reuses its work; estimated mean prompt time for a 40-step task of about 8.6 s per step (keep 2) or 5.8 s (keep 1), against about 16 s with the checkpoint change alone. The model then sees 2 to 3 (or 1 to 3) earlier screenshots instead of always 3, and loses the before-and-after view right after a drop. | Opt-in until the A/B |
| Viewport 1280x800; `snap_viewport` for 1280x832 | Screenshots up to about 1280x832 cost the same (about 1040 to 1066 tokens), so a smaller window does not help; bigger ones cost more. In the laptop logs most screenshots were 1104 tokens, a page area wider than configured, so the window is now refitted at each task start and a warning appears above 1.3 megapixels. 1280x832 is whole 32-pixel tiles, so no resample. | Snapping opt-in until the A/B; refit and warning on |
| Every tool call of a reply runs, in order | The old loop silently kept only the first call. The rest of a step is skipped after a failure, a refusal or a change of page or tab. Holo4 rarely sends more than one call per step. | On |
| Next task continues the conversation | So the agent can act on "yes, send it" after asking for confirmation, and the server can reuse the cached prefix. `/new` starts over with a labelled summary; a task that ended in an error starts a new conversation. | On |
| Condensing a long history | At three quarters of the context, the older half of the steps becomes one notes block (one line per step and the model's own notes, at most 6000 characters). Old tool results (3 or more steps back; 2 with `image_budget` 2) are cut to 300 characters, but only on steps where screenshots leave anyway and never before the first screenshot that leaves, so the server can still reuse the prompt up to there and block eviction stays append-only. No logged task came near the threshold. | On |
| Stall alerts | 10 of 27 laptop tasks ran into the old 40-step cap, and `max_steps` is now 0 (no limit) by the user's choice. When the same action is repeated 3 times on an unchanged page, or 5 steps change nothing, the next observation carries a system alert asking for a different approach. | On |
| Honest action feedback | Results say "Executed." and add "The page did not change." when nothing changed; typing is read back; a scroll that moved nothing says so. | On |
| Settle after actions: `fixed` 1.0 s vs `adaptive` | Adaptive waits only as long as each kind of action needs and stops once the page is quiet; a page that changes again a moment later may be captured half-drawn. | Opt-in until the A/B |
| Observation `header` format | H Company's agent puts the address, title and scroll position around each screenshot. Keeping the text after the image under 32 tokens avoids one extra GPU micro-batch per step (estimated 0.4 to 0.5 s). | Opt-in until the A/B |
| `prompt_version h`, `tool_schema h`, `write_enter_default` | Closer to how Holo4 was trained (H Company's published guidance, tool names and defaults). The longer prompt costs a few seconds once per conversation; Enter after typing could send a message by mistake, which the Send confirmation guards. | Opt-in until the A/B |
| `preserve_reasoning` | H Company's own documents disagree: their model API page says to drop past reasoning, their Holo4 agent keeps it. Keeping it adds about 100 tokens per step and could anchor the model on a stale plan. `auto` keeps it for the accurate profile (its template is built for it) and drops it for the fast ones. | Fast profiles: off until the A/B |
| `reader_mode` as markdown with paging | Headings, lists, tables and links survive, and long pages come in 8000-character slices with a footer saying where the slice is. | On |
| Default browser `chrome` | Uses the user's own sign-ins with no extra setup per site. The remote-debugging switch it needs is a real risk (any local program can ask to control Chrome), so `chrome-agent` is offered as the safer alternative, at the cost of separate sign-ins; the choice is the user's. | Kept |
| Model copy to the PC: `ask` | Loading from a USB stick took about 5 minutes; from the SSD copy, 10 to 25 s. Asked once per PC, because it takes 22 GB of that PC's disk. | On |
| Downloads to the PC's `Downloads\GhostAgent` | An exFAT stick cannot hold the "downloaded from the internet" mark, and a page should not be able to fill the stick. | On |
| Browser profile on the stick (`profile_on_stick`) | Keeps the portable design; cookie sign-ins are tied to the PC anyway, so moving the profile to the PC is offered as a setting rather than changed silently. | Kept |
| Server crash recovery | At most 2 restarts per session; a llama.cpp build that cannot run the model is skipped and remembered per GPU, and the next one is tried. | On |
| Default model profile `fast` | The 27B is the more capable model but several times slower per step on an 8 GB GPU; the default is the user's call once laptop success rates and minutes per task are measured. Q8_0 becomes the fast default only if it wins by at least 2 tasks out of 51 paired runs and the slowdown is acceptable. | Kept until the benchmark |

The desktop check also ran the 27B through the correctness gate for llama.cpp #30906 (repetition loops reported on this model family when part of it runs on the CPU): a repetition-prone prompt at temperature 0.2, a 10k-token prompt with a screenshot, and the replayed steps. Neither model looped, and every replayed step produced a valid tool call.


## Build

| Choice | Why |
|---|---|
| Every part pinned and checked by SHA-256 | llama.cpp archives, Python and pip, the Playwright and greenlet wheels (`--require-hashes`, pip settings on the PC ignored), Chromium, and the models (pinned to a Hugging Face commit). Before 0.2.0, Playwright's dependencies and Chromium were not checked and the model URL followed the repository's latest version. |
| Chromium downloaded directly from Chrome for Testing | Playwright's own installer does not verify what it downloads. The pinned build must match the one the pinned Playwright expects, or the build stops. |
| FAT32 refused before any download | FAT32 cannot hold files over 4 GB; before 0.2.0 the build retried the model download forever. |
| Model staged on the PC's disk | Downloading straight to a slow stick was slow and failed on dropped connections. The model is now downloaded to `%LOCALAPPDATA%\GhostAgent\models` when there is room (resumable, up to 30 tries with growing pauses), hashed once there, copied to the stick and hashed once more. A mismatch stops the build instead of downloading again. |
| One check record per model folder (`.verified`) | Later runs compare size and time instead of hashing 22 GB again; the build and the agent share the format. |
| Component stamps and swapped-in folders | A part is unpacked to a temporary folder and swapped in whole, so an interrupted build never leaves a half-installed CUDA build that later runs skip. |
| Unused tools pruned | Every llama.cpp program except `llama-server.exe`, Playwright's ffmpeg and winldd, and old Chromium revisions: about 108 files the agent never uses. |
| Download mark cleared, updater runs from a copy | Files from a downloaded ZIP carry a mark that makes Smart App Control or SmartScreen block or warn about the `.cmd` and `.ps1` files. `Update GhostAgent.cmd` copies itself to `%TEMP%` because the update replaces it while it runs; the copy deletes itself at the end. |
| Proxy, HTTPS inspection, Constrained Language Mode | The build reads `-Proxy`, the proxy variables or the Windows proxy setting, uses best-effort certificate revocation checks (TLS-inspecting company proxies often block them), allows HTTPS only, and avoids .NET calls where PowerShell's Constrained Language Mode forbids them. |
| ARM64 parts | Pinned like the rest, but untested: no ARM64 hardware was available. |


## Code layout

Up to 0.2.0 the agent was one file, `GhostAgent/agent/ghost_agent.py`, of almost 9,000 lines. It is now the package `GhostAgent/agent/ghostagent/`, split along the old file's section banners, and `ghost_agent.py` only starts it. The split moved code and changed nothing else: every statement is the same apart from imports and the names below that are now read through their module, and `--help`, the exit codes and the output are unchanged.

| Choice | Why |
|---|---|
| One module per part, imported bottom up | `runstate`, `console`, `paths`, `browsercache`, `settings`, `preflight`, `winproc`, `backends`, `serverflags`, `gguf`, `modelclient`, `questions`, `modelcopy`, `screenshot`, `llamaserver`, `tools`, `prompts`, `keys`, `pagejs`, `userbrowser`, `guards`, `pagehelpers`, `browser`, `history`, `steptrace`, `agentloop`, `forgetpc`, `cli`. Each imports only modules before it in this list, so there are no import cycles; `ghostagent/__init__.py` says what each one holds. |
| Names that change at run time are read through their module | `runstate.HEADLESS`, `console.PLAIN_OUTPUT`, `paths.LOGS`, `paths.DOWNLOADS`, `paths.PROFILE`, `paths.SETTINGS_FILE`, `paths._CONSOLE_LOG` and a few handles and caches in `winproc` and `backends` are set while the agent runs (an unattended run, the console, the data folder). A `from ... import` would keep the value from when the importing module loaded, so other modules write `paths.LOGS`, never a copy. |
| `ghost_agent.py` puts its own folder first on `sys.path` | The agent runs in isolated mode (`python -I ghost_agent.py`; the embeddable Python's `._pth` file turns it on as well, so `Start GhostAgent.cmd` gets it too), and isolated mode leaves the script's folder off `sys.path`. The `._pth` lists `..\agent`, but the entry script does not rely on it, so a test or a hand-started copy elsewhere finds the package too. |
| `sys.dont_write_bytecode = True` before the first import | With 28 modules, compiled copies would otherwise appear in `__pycache__` folders on the stick (slow writes, stale files after an update). The package's `__init__.py` sets it again for anything that imports the package directly, so such an import (without `-B`) leaves at most the compiled copy of `__init__.py` itself, which is compiled before it runs. |
| An update mirrors the package | `Update GhostAgent.cmd` copies files without deleting anything (models and downloaded parts stay), but a module that a new version drops must not stay behind next to the new ones, so `GhostAgent\agent\ghostagent` alone is mirrored (robocopy `/MIR`). |
| Tests patch by name, not by module | `tests/agent_support.py` has `patch_agent`, `patch_agents` and `set_agent`: each replaces a name in the module that defines it and in every module that imported it, so a test reaches every place the agent looks the name up. `helpers.import_agent()` gives tests a read-only view of all modules by name; writing to it fails, so a test cannot set a copy the agent never reads. |


## Rejected or deferred

Runtime and flags:

- **`--cache-reuse`**: llama.cpp turns it off when the prompt contains images or the model's memory cannot be shifted, both true here.
- **`--swa-full`**: Holo4 has no sliding-window attention layers.
- **A micro-batch (`-ub`) above 2048**: on the desktop only the first request of a task got faster; later steps send small chunks.
- **`--mtmd-batch-max-tokens`**: the vision encoder processes one image at a time, so there is nothing to batch.
- **`--n-cpu-moe 33` (a static split)**: on the desktop it generated faster (61 tokens/s) but read prompts at only 275-395 tokens/s; reading prompts is most of a step, so `--cpu-moe` with the expert cache won.
- **`--no-op-offload`**: prompt reading dropped to about 170 tokens/s on the desktop. Copying the CPU-side experts to the GPU for prompt batches is essential.
- **The Vulkan build on NVIDIA cards**: about 30 tokens/s against about 50 with CUDA on the desktop. Vulkan is kept for AMD and Intel cards.
- **More than 8 threads**: 16 and 24 threads were no faster than 8 on the desktop.
- **`-lm dio` (unbuffered direct reads)**: meant for slow loads, but the model normally loads from the PC's SSD copy. Measure only if loading from the stick matters again.
- **Lowering `--image-min-tokens`, or an `--image-max-tokens` below a normal screenshot's size to shrink screenshots**: a departure from H Company's recipe that trades accuracy for speed; snapping the viewport removes the resample at no cost instead. (The adopted cap of 1280 sits above a normal screenshot, so it shrinks only oversized ones; see Always passed.)
- **H Company's `--cache-ram 32768`**: see Prompt cache.
- **Speculative decoding**: neither the 35B's GGUF nor its original weights contain MTP layers; the available draft models target the base Qwen3.6 model, with unknown acceptance on the Holo4 fine-tune, and would add little while the experts run from system RAM; llama.cpp #31472 (wrong output with drafting on this architecture) is open; and generation is only about 22 to 37 percent of a step. H Company's announced DSpark draft models have not been published. For the 27B, a community MTP repack is worth one A/B now that it passed its correctness gate.
- **A reasoning budget below 1024 tokens**: at 1024 it would trim 0.7% of steps and save about 0.2 s per step; lower caps trade accuracy for speed, and `max_tokens` already stops runaway steps.
- **llama.cpp b12360 to b12366**: see llama.cpp version.

Agent:

- **JPEG screenshots**: encoding and uploading PNGs costs at most about 0.37 s per step in total, and JPEG is lossy.
- **Starting to read the prompt while the browser acts**: the tool result is only known once the action finishes, so at most about 0.1 s could be saved.
- **Caching image embeddings across requests**: needs a patched llama.cpp for four backends with its own pins. It would save about 1.6 s per re-encoded screenshot, but block eviction removes most re-encoding. Propose it upstream if it is ever needed.
- **An LLM summary when the history is condensed** (as H Company's agent does): a 1-2 minute generation per summary on the laptop, and no logged task came near the threshold. The verbatim notes block is the cheap equivalent.
- **Page text as markdown with every screenshot**: 300 to 1,500 extra tokens per step, the main cost driver. `reader_mode` covers reading.
- **Plan and zoom tools from H Company's training harness**: they add tokens and images per step; deferred to a later A/B.
- **Separate double-click and right-click tools**: not in H Company's documented Holo4 action set, so they would push the model outside what it was trained on.
- **Defuddle for `reader_mode`**: about 650 KB of JavaScript with its own license; a small built-in walker does the job.
- **Checking a growing download while the agent waits at the `Task>` prompt**: the prompt waits in `input()`, which keeps the console's own line editing and history; the keyboard is polled only for a moment after a line, to tell a pasted block from typing. A check once a second would need the prompt rebuilt on raw key reads, and stopping a download is a Playwright call that only the main thread may make (Playwright also handles browser events only during a call), so a background thread could watch the size but not stop it. A download that grows past `download_max_mb` while the agent is idle is still stopped and deleted at the next step or when it finishes; until then it sits in Playwright's temporary folder on the PC, not in `Downloads\GhostAgent`.
- **Keeping llama-server running across agent restarts**: it conflicts with ending the server together with the window, and leaving a 20 GB process behind on a work laptop is worse than a 10-25 s reload.
- **Turning the Chrome switch off by script at exit**: unverified on branded Chrome, it would have to be the very last action, and it adds clicks at every start. `chrome-agent` mode is the clean alternative.
- **ROCm, SYCL and OpenVINO builds, and offloading to Intel Arc integrated graphics**: they need benchmarks on hardware that was not available. Revisit if users report AMD or Intel GPUs.

Models (see also README.md, Model profiles):

- **gpt-oss-120b**: text only (no screenshots, so no computer use), and its weights alone are about 61 GB.
- **Holotron4-30B-A3B**: far lower computer-use scores, no GGUF vision projector, NVIDIA's model license.
- **Holo3.1, Holo3, Holo2**: superseded by Holo4 at the same sizes.
- **Qwen3.8-27B, Fara1.5-27B, UI-Mate-27B, Gemma 4, EvoCUA, GUI-Owl, MiniCPM-V 4.7 and older models**: lower scores where a comparison exists, a different harness, or no vision GGUF.
- **Qwen-CUA and Qwen3.8-Flash-Next**: do not fit in 64 GB of RAM.
- **Smaller models** (for example Mage-VL-Cowork-4B, UI-Mate-9B): weaker; the project's rule is the most capable model the hardware can carry, spilling into RAM rather than shrinking.
- **MXFP4, IQ and "abliterated" variants**: lower precision, or safety training removed.
- **Converting H Company's BF16 weights to Q6_K or Q8_0 ourselves**: ready-made files exist. Only worth it with H Company's importance matrix, if fast-q8 shows a clear gain and the community file's provenance becomes a concern.
- **H Company's hosted agent tooling**: depends on their cloud service, which GhostAgent avoids by design.


## Watch list

- H Company's DSpark draft models, and any new Holo release (check before each build).
- llama.cpp [#30906](https://github.com/ggml-org/llama.cpp/issues/30906): repetition loops reported on this model family on the CPU backend. Not seen in the desktop check, but the accurate profile runs part of the model on the CPU.
- llama.cpp [#31472](https://github.com/ggml-org/llama.cpp/issues/31472): drafting gives different output on this architecture.
- llama.cpp [#32041](https://github.com/ggml-org/llama.cpp/issues/32041): the expert-cache tuning script does not run on Windows.
- llama.cpp releases after b12366.
