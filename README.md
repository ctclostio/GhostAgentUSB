# GhostAgentUSB

A local AI that uses your web browser, carried on a USB stick.

Plug the stick into a Windows PC, double-click the launcher and type a task like *"find the weather in Denver this weekend"*. A vision model looks at the browser window, then clicks, types and scrolls until the task is done and gives you the answer. No cloud AI is involved. The model runs on the PC itself through llama.cpp, listens only on `127.0.0.1`, and answers only requests that carry a key made for that run.

This is version 0.2.0. The speed figures on this page were measured **before** this version's changes; new numbers will be measured with the benchmark in `tools/bench`, and this page will be updated. Several new behaviours are built in but stay switched off until that benchmark decides (see [Settings that wait for a benchmark](#settings-that-wait-for-a-benchmark)). [DECISIONS.md](DECISIONS.md) records why each server flag and default is what it is.

| Part | What it is |
|---|---|
| Model | [Holo4](https://huggingface.co/Hcompany) by H Company, vision models trained for computer use. Three profiles: **fast** (Holo4-35B-A3B Q4_K_M, the default), **fast-q8** (the same model at Q8_0) and **accurate** (Holo4-27B Q4_K_M, non-commercial license). See [Model profiles](#model-profiles). |
| Runtime | [llama.cpp](https://github.com/ggml-org/llama.cpp) b11669 (b12359, the v1.0.0 release, is one build option away), with CUDA 13, CUDA 12, Vulkan and CPU builds. The best one is picked automatically, and the next one is tried if it cannot run the model. |
| Browser | By default a window in **your own Google Chrome** (Chrome 144 or newer), so it has your sign-ins. Your own Edge, the installed Chrome with a separate profile, or a portable Chromium or Edge with a profile on the stick are one setting away. |
| Harness | `GhostAgent/agent/ghost_agent.py`, which starts the `ghostagent` package next to it (plain Python, one module per part): a Playwright agent loop that follows H Company's published Holo4 conventions, plus safety checks that do not depend on the model. |
| Python | 3.14 embeddable with Playwright 1.65.1, so nothing is installed on the PC. |


## Requirements

- Windows 10 or 11, 64-bit. On ARM PCs the build fetches ARM64 parts, but those are untested.
- RAM: the model needs about **22 GB free** (fast), 19 GB (accurate) or 40 GB (fast-q8). A PC with 32 GB can run fast or accurate when little else is open; 64 GB is recommended, and fast-q8 needs it. The agent warns before loading, and asks first on a PC with clearly too little.
- GPU: an NVIDIA card with 8 GB or more works best. The CUDA 13 build needs driver 580 or newer and a GTX 16 / RTX 20-series card or newer (compute capability 7.5); other NVIDIA cards and drivers use the CUDA 12 build (driver 550 or newer). AMD and Intel Arc cards use Vulkan. Integrated graphics are not used. CPU-only works, but each step takes several times longer.
- The Microsoft Visual C++ Redistributable (x64; ARM64 for a stick built with ARM64 parts), which llama.cpp needs and which cannot be carried on the stick. Most PCs have it; the build and the agent say so if one does not.
- A USB stick formatted **NTFS or exFAT**, in a **USB 3** port. FAT32 cannot hold files over 4 GB, so the build refuses it. A USB 3 portable SSD or an SSD-class stick makes starts much faster (see [Speed](#speed)).
- Room on the stick: the program parts take about 3 GB, plus the model files.

  | Profile | Model files | Stick size |
  |---|---|---|
  | fast (default) | 22.2 GB | 32 GB |
  | accurate | 17.8 GB | 32 GB |
  | fast-q8 | 37.8 GB | 64 GB |
  | fast and accurate | 40.0 GB | 64 GB |
  | all three | 76.9 GB | 128 GB |

- Internet for the one-time build (about 24 GB of downloads for the fast profile) and for the websites you send the agent to.
- Ideally, room on the building PC's own disk: when that disk has the file's size plus 2 GB free, the model is downloaded to `%LOCALAPPDATA%\GhostAgent\models` first, checked there, then copied to the stick. That copy also lets the agent start quickly on the PC that built the stick (`/forget-pc` removes it). Without the room, the model goes straight to the stick.


## Build a stick

1. **Get the files onto the stick.** Either:
   - `git clone https://github.com/ctclostio/GhostAgentUSB` on the stick (updates then use `git pull`), or
   - **Code > Download ZIP**. Before you extract it, right-click the ZIP > **Properties** > tick **Unblock** > **OK**. Otherwise Windows marks every extracted file as downloaded from the internet, and Smart App Control or SmartScreen blocks or warns about the `.cmd` files. Then right-click the ZIP > **Extract All** and pick the stick. Windows suggests a folder named after the ZIP, and the ZIP holds a folder of the same name, so you get `GhostAgentUSB-main\GhostAgentUSB-main`. That is fine; the files only need to stay together.
2. Double-click **`Build GhostAgent.cmd`**. It clears the download mark from the folder (in case the Unblock step was skipped), checks the stick's file system and free space, downloads every part from its official source at a pinned version, checks each one, and ends with a smoke test. Writing 22 GB to a slow stick takes a long time (at 5 MB/s, over an hour); a USB 3 SSD-class stick takes minutes. Running it again is safe: finished parts are skipped and downloads resume where they stopped.
3. Double-click **`Start GhostAgent.cmd`**. `README.txt` (on the stick) explains the first start, including the one Chrome setting it needs.

Build options go after the file name in a Command Prompt, for example `"Build GhostAgent.cmd" -Profile accurate`:

| Option | What it does |
|---|---|
| `-Profile fast`, `fast-q8`, `accurate` or `all` | Which model to set up (default `fast`). To use another profile, also set `"model_profile"` in `GhostAgent\settings.json`. |
| `-ModelFrom "E:\old\GhostAgent\models"` | Take the model files from an older copy: hard-linked when on the same drive (instant), otherwise copied, then checked. A copy under the project's old name (`HoloAgent\models` in or next to this folder) is found without this option. |
| `-MoveModel` | With `-ModelFrom` on the same drive: move the files instead of linking them. |
| `-SkipModel` | Set up everything except the model. |
| `-SkipChromium` | No portable Chromium (the `bundled` browser mode then does not work). |
| `-LlamaTag b12359` | llama.cpp b12359 (v1.0.0) instead of b11669. Run again with `-LlamaTag b11669` to switch back; only the llama.cpp folders are replaced. |
| `-NoStaging` | Download the model straight to the stick instead of first to this PC's disk. |
| `-KeepDownloads` | Keep the downloaded archives in `%TEMP%\GhostAgent-build`. |
| `-Proxy http://host:port` | Download through this proxy. Without it the build uses `HTTPS_PROXY` / `HTTP_PROXY` or the Windows proxy setting. |
| `-Arch x64` or `arm64` | Override the detected architecture (mostly useful with `-DryRun`). |
| `-DryRun` | List what would be downloaded (address, size, SHA-256, destination) and change nothing. |

### What the build pins and checks

Every download is pinned to an exact version and checked against a SHA-256 sum stored in this repo before it is used:

- the llama.cpp release archives for the chosen tag (`tools/Build-GhostAgent.ps1`);
- the Python 3.14.8 embeddable package and pip;
- the Playwright 1.65.1 and greenlet wheels, installed with `pip --isolated --require-hashes --no-deps` from `tools/requirements.lock`, so pip settings on the building PC are ignored;
- Chromium 158.0.8127.18 (Chrome for Testing), the build this Playwright version expects (the build stops if the two disagree), downloaded directly so its archive can be checked;
- the model files, each pinned to a Hugging Face commit and a SHA-256 in `tools/profiles.json`. A model file is hashed once; later runs compare its size and time with `GhostAgent/models/.verified`. A file whose sum does not match is deleted, and the build stops instead of downloading it again in a loop.

Each installed part carries a stamp naming the archives it came from, so a later run replaces only parts whose pins changed or that turned out broken. The smoke test runs every llama.cpp build (`--version`, and `--help` against the flag list in `tools/server-flags.json`), imports Playwright, and starts the bundled Chromium headless with its sandbox on.


## Updating

Double-click **`Update GhostAgent.cmd`**. In a git clone it runs `git pull --ff-only`; otherwise it downloads the latest release ZIP from GitHub (or the main branch when there is no release) and copies it over the folder. Then it runs the build, which replaces only parts whose pinned versions changed, and offers to delete model files that no profile uses any more. It never overwrites `GhostAgent\settings.json`, the browser profiles, downloads or logs.

- Without access to GitHub from that PC: `"Update GhostAgent.cmd" -UpdateFrom "D:\Downloads\GhostAgentUSB-main.zip"` (a ZIP or an extracted folder).
- Build options pass through, for example `"Update GhostAgent.cmd" -Profile accurate`.
- Copies made before version 0.2.0 have no updater. Copy the new files over the old ones (unblock the ZIP first) and run `Build GhostAgent.cmd`. On the next start an old full `settings.json` is converted: only the values you changed are kept, and the old file is saved as `settings.backup.json`. A copy from before the rename keeps its files in a `HoloAgent` folder: the build reuses the model from there, but copy any settings you changed into `GhostAgent\settings.json` yourself.


## Using it

`README.txt` on the stick is the user guide: the first start, commands, settings and troubleshooting. In short:

- Type a task at `Task>`. Text pasted as several lines counts as one task.
- After an answer, your next task continues the same conversation, so you can answer a question the agent asked ("yes, send it") or follow up. `/new` starts a new conversation.
- **Ctrl+C** stops the current task once the browser action in progress finishes; the model stays loaded. Pressing it twice within 3 seconds closes GhostAgent.
- `/copy` puts the last answer on the clipboard, `/forget-pc` removes what GhostAgent keeps on this PC, `/help` lists the commands and `/quit` exits.
- The agent works only in its own window. Your other windows and tabs are left alone, and the model never sees them. You can use the agent's window yourself between tasks, for example to sign in to a site.
- Some questions come to you in the console, with a beep: a page's confirm or prompt box, or a click on a button such as Delete, Send or Pay that your task did not ask for. No answer within 2 minutes counts as no.
- When several actions in a row change nothing on the page, the model is told so and asked to change its approach.


## Model profiles

GhostAgent runs H Company's Holo4 models. They are vision models trained for computer use: they read a screenshot and reply with tool calls (click at a point, type, scroll and so on) in the format this harness uses. All three profiles use the same tools and the same coordinates, so switching is one setting.

| Profile | Model | Files | License | Notes |
|---|---|---|---|---|
| `fast` (default) | [Holo4-35B-A3B](https://huggingface.co/Hcompany/Holo4-35B-A3B-GGUF) Q4_K_M | 22.2 GB | Apache-2.0 | A mixture of experts: only about 3B of its 35B parameters work on each token, so most of it can sit in system RAM while the GPU does the rest. The fastest option on an 8 GB GPU. |
| `fast-q8` | [Holo4-35B-A3B Q8_0](https://huggingface.co/bartowski/Hcompany_Holo4-35B-A3B-GGUF), a community quantization by bartowski, with H Company's vision projector | 37.8 GB | Apache-2.0 | Closer to the original weights. Explicit trade-off: it needs about 40 GB of RAM, and each step is slower (estimated at half again to three quarters longer; not measured yet). Whether the extra precision helps the agent is unmeasured. |
| `accurate` | [Holo4-27B](https://huggingface.co/Hcompany/Holo4-27B-GGUF) Q4_K_M | 17.8 GB | **CC BY-NC 4.0** | H Company's most capable model. Their published benchmark results put it clearly ahead of the 35B-A3B, most of all on long multi-step tasks. It is dense (every parameter works on every token), so on an 8 GB GPU much of it runs on the CPU and each step takes several times longer (see [Speed](#speed)). Its `reasoning_effort` defaults to the most thorough level; `medium` or `low` is faster. |

The accurate profile's license does not allow commercial use, which can include work done for an employer: check before using it at work. It is downloaded only with `-Profile accurate` or `-Profile all`, and its license is in `GhostAgent/licenses/Holo4-27B-LICENSE.txt`.

To switch: `"Build GhostAgent.cmd" -Profile accurate`, then put `"model_profile": "accurate"` in `GhostAgent\settings.json`. The default stays `fast` until the benchmark has measured each profile's success rate and time per task on an 8 GB laptop; whether the more capable but slower 27B is worth it is then your call, with those numbers in hand.

### Why Holo4, and what else was considered

Holo4 is trained for exactly this job, ships official GGUF files with its vision projector, and runs in stock llama.cpp. When this version was planned (November 2026), no open computer-use model outside the Holo4 family scored higher while fitting a PC with 64 GB of RAM and an 8 GB GPU. Considered and not used:

- **gpt-oss-120b**: text only, so it cannot see the screen, and its weights alone are about 61 GB.
- **Holotron4-30B-A3B**: scores far lower on computer-use benchmarks, has no GGUF vision projector, and uses NVIDIA's model license.
- **Older Holo releases** (Holo3.1, Holo3, Holo2): superseded by Holo4 at the same sizes.
- **Other open computer-use models** (UI-Mate-27B, Fara1.5-27B, Qwen3.8-27B, Gemma 4, GUI-Owl, EvoCUA and others): lower scores where a comparison exists, a different tool format, or no vision GGUF.
- **Qwen-CUA and Qwen3.8-Flash-Next**: too large for 64 GB of RAM.
- **Smaller models**: weaker by design. GhostAgent aims for the most capable model the hardware can carry, spilling into system RAM rather than shrinking the model.
- **Converting H Company's BF16 weights ourselves**: ready-made Q8_0 files exist.

Watch list: H Company's announced draft models for speculative decoding, any new Holo release, and llama.cpp [#30906](https://github.com/ggml-org/llama.cpp/issues/30906) (repetition loops reported on this model family) and [#31472](https://github.com/ggml-org/llama.cpp/issues/31472) (speculative decoding output on this architecture). DECISIONS.md has the details.


## Speed

### Measured before version 0.2.0

On a laptop with an 8 GB NVIDIA GPU and 64 GB of RAM, running the fast model from a copy on its SSD (the model server's logs of 27 real tasks, about 600 steps, October 2026):

- A step (one look at the page plus one action) took **about 23 seconds** of model time at the median. Generation ran at about 26 tokens/s and reading the prompt at 375-390 tokens/s; reading the prompt took 70% of the time.
- Steps got slower as a task went on, because the server re-read most of the task's history at every step: in one 40-step task they went from about 17 s to about 33 s.
- The median task took about 26 steps. 10 of the 27 tasks reached the 40-step limit that existed then, and **a 40-step task took 16 to 21 minutes**.
- Loading the model from the USB stick (about 70 MB/s) took about 5 minutes; from the copy on the PC's SSD, 10 to 25 seconds.

### Not measured yet

Version 0.2.0 changes the model server's prompt-cache settings so it can reuse more of its earlier work at each step, and adds opt-in switches that should save more (block screenshot eviction, adaptive settling). None of this has been timed on the laptop yet. The benchmark in `tools/bench` (12 local tasks with automatic checks, plus a replay of recorded steps) will measure it, and this section will be updated with the new numbers.

### The accurate profile

In a first check on a desktop whose GPU was held to the laptop's 8 GB budget, replaying the same 20 recorded steps with the version before 0.2.0, a Holo4-27B step took about 7 times as long as a fast step at its default `reasoning_effort`, and about 4.5 times as long at `medium`. A desktop CPU is faster than a laptop's, so laptop times will be longer than the desktop's. Both models produced a valid tool call on every replayed step.

### Tips

- Starts: `copy_model_to_pc` (asked once per PC when the model is on a USB stick) keeps a copy on the PC's disk, so later starts take seconds instead of minutes.
- A maximized agent window makes every screenshot bigger and every step slower, so the agent resizes its window at the start of each task. A smaller window does not make steps faster: screenshots up to about 1280x832 cost the model the same.
- Games and other programs that use the GPU slow the agent down a lot.


## Settings

`GhostAgent\settings.default.json` lists every setting with its default and a note that explains it. Do not edit it; updates replace it. Put your changes in `GhostAgent\settings.json` (created on the first start), then restart the agent. A value of the wrong type or out of range is reported and its default used instead; a JSON mistake is reported with its line and column.

The ones most people need:

| Setting | Default | What it does |
|---|---|---|
| `model_profile` | `fast` | `fast`, `fast-q8` or `accurate` (download it first with `-Profile`). |
| `reasoning_effort` | `""` | Accurate profile only: `medium` or `low` is faster per step but may make more mistakes. Empty = the model's own default (the most thorough). |
| `browser` | `chrome` | Which browser the agent uses; see [Browser modes](#browser-modes). |
| `copy_model_to_pc` | `ask` | Load the model from a copy on the PC's disk (22 GB for fast). `ask` asks once per PC; `true` / `false` decide for every PC. |
| `start_url` | Bing | The page the browser opens at start. |
| `max_steps` | `0` | Steps before the agent gives up on a task; 0 = no limit (Ctrl+C stops it). |
| `progress_checkin_steps` | `0` | Every N steps, remind the model to check its strategy; 0 = never. |
| `backend` | `auto` | Force `cuda13`, `cuda12`, `vulkan` or `cpu`. A forced GPU build that is not on the drive (an ARM64 drive has no `cuda12`) falls back to `auto`. |
| `viewport_width`, `viewport_height` | 1280, 800 | The agent's page area. |
| `save_screenshots` | `false` | Keep the screenshots the model sees, on the PC, for 7 days. They show private pages. |
| `step_trace` | `true` | One line per step in `logs\steps-*.jsonl` (see [Data at rest](#data-at-rest)). |
| `web_ui` | `false` | Turn on llama-server's own chat page; the console then shows its address and the key it asks for. |

The safety settings (`dialog_policy`, `confirm_risky_actions`, `allowed_hosts`, `allow_file_urls`, `allow_internal_pages`, `allow_unsandboxed_browser`, `downloads`, `download_max_mb`, `profile_on_stick`) are described under [Safety](#safety). The model server's tuning settings are in DECISIONS.md.

### Settings that wait for a benchmark

These change what the model sees or how it acts. Each is fully built, but its default keeps the behaviour from before 0.2.0 until the benchmark compares the two on the same tasks and seeds. A new default is adopted only if it costs at most 1 success in 36 runs.

| Setting | Default now | Alternative | Trade-off |
|---|---|---|---|
| `image_eviction` | `rolling` | `block` (with `image_keep_after_evict` 2 or 1) | Block mode lets the server reuse its work between screenshot drops, so steps should be much faster; the model then sees 2 to 3 (or 1 to 3) earlier screenshots instead of always 3. |
| `observation_format` | `classic` | `header` | The address, title, page size and scroll position before each screenshot, and how much of the page lies above and below after it, as H Company's own agent does. |
| `prompt_version` | `classic` | `h` | Longer instructions written along H Company's published Holo4 browser guidance. Costs a few seconds once per conversation. |
| `tool_schema` | `classic` | `h` | H Company's Holo4 tool names and arguments, a drag-and-drop tool, and an outcome with each answer (success, partial, infeasible or blocked). |
| `write_enter_default` | `false` | `true` | Enter after typing unless the model says otherwise, as Holo4 was trained. A message could then be sent by mistake; the Send confirmation guards against that. |
| `preserve_reasoning` | `auto` (on for accurate, off for fast) | `true`, `false` | The model keeps its own earlier reasoning in context: about 100 more tokens per step. |
| `settle_mode` | `fixed` | `adaptive` | Waits after an action only until the page is quiet. Faster, but a page that changes again a moment later may be captured half-drawn. |
| `snap_viewport` | `false` | `true` | The page area in whole 32-pixel tiles (1280x800 becomes 1280x832), so the model server reads each screenshot without resampling it. |

Also waiting for measurements: the default model profile, the llama.cpp version (b11669 or b12359), and the server tuning values described in DECISIONS.md. Two changes ship switched on: every tool call in a reply now runs, in order (the rest of a step is skipped after a failure or a page change), and the next task continues the conversation.


## Browser modes

| `browser` | What the agent uses | Sign-ins |
|---|---|---|
| `chrome` (default) | A new window in your running Google Chrome (144 or newer). You turn on remote debugging once at `chrome://inspect/#remote-debugging` and click Allow when the agent connects. | Yours |
| `user_edge` | The same in your Microsoft Edge (`edge://inspect/#remote-debugging`). | Yours |
| `auto` | Your Chrome if it can be used, else your Edge, else `bundled`. | |
| `chrome-agent` | The installed Chrome with a separate profile kept on this PC (`%LOCALAPPDATA%\GhostAgent\chrome-profile`). No switch, no Allow click, and your own profile is never touched. | Separate, kept on this PC |
| `bundled` | The portable Chromium on the stick. | Kept on the stick |
| `edge` | Microsoft Edge with its own profile on the stick. | Kept on the stick |

If Chrome is missing, older than 144, or an administrator turned off its remote debugging (the `RemoteDebuggingAllowed` policy), the agent says so and uses the bundled browser. While it waits for Chrome, pressing **B** switches to the bundled browser for that session. Unattended runs (`--task`, `--tasks-file`) never use your own Chrome or Edge; they switch to the bundled browser.


## Safety

In `chrome` and `user_edge` modes the agent acts **with your accounts, signed in as you**. A local model can be fooled by a malicious page (prompt injection), so GhostAgent does not rely on the model alone:

- **Instructions.** The model is told to ask before buying, sending, posting, deleting or changing settings, and that text copied from web pages (marked as untrusted) is data, not instructions.
- **Page dialogs** (`dialog_policy`, default `ask`). A page's confirm or prompt box ("Delete all items?") is put to you in the console; no answer within 2 minutes counts as no. `dismiss` always cancels such boxes and `accept` always agrees, including to ones you would have refused. Simple alerts are closed, and a "leave this page?" warning is accepted only when the agent itself navigates away. Unattended runs always dismiss.
- **Risky clicks** (`confirm_risky_actions`, default on). Before the agent clicks a button or link labelled Delete, Remove, Send, Post, Pay, Buy, Place order, Checkout, Transfer, Subscribe, Reset and similar words (or presses Enter in a form whose submit button says so), the console asks you, unless your task asked for that action. A yes holds for that button on that site until the task ends. Harmless buttons such as "Remove filter" cost one question.
- **Passwords and codes** are never typed: the agent refuses password and one-time-code fields and asks you to type them. Pasting from the clipboard is refused too, so a page cannot make the agent paste what you copied.
- **Addresses.** Only `http(s)` pages and `about:blank` open. `file:` addresses and the browser's own pages (`chrome://`, extensions, `view-source:`) are blocked unless `allow_file_urls` or `allow_internal_pages` is on; `data:` and `javascript:` are always blocked. A tab that ends up on a blocked address is sent back to a blank page. Addresses on this PC or your local network (a router, a NAS, intranet sites, `localhost`) open only when your task names them, `allowed_hosts` lists them, or you had them open yourself.
- **Tabs, uploads and downloads.** A tab that a page opens on its own does not take the agent's focus. File pickers are cancelled during tasks, and the model is told to ask you to choose the file. In the agent's own browsers (`bundled`, `edge`, `chrome-agent`), downloads go to `Downloads\GhostAgent` on this PC under a new name when one is taken, are marked as downloaded from the internet, are stopped above `download_max_mb` (500), and are never opened by GhostAgent; `"downloads": "off"` cancels them. In your own Chrome or Edge the browser handles downloads as usual, and the agent is only told about them.
- **The model server** listens only on 127.0.0.1 and answers only requests that carry a random key made at each start. The key reaches the server through its environment, never its command line or a log. The server accepts browser requests only from pages on localhost, its chat page and slot API are off (`web_ui` turns the chat page on), and environment variables that could change its behaviour are cleared before it starts.
- **The agent's own browsers** run with Chromium's sandbox on. Playwright's switches that turn off pop-up blocking, phishing protection and component updates are removed, and HTTPS upgrades are on. If the bundled browser cannot start with its sandbox (some USB drives), the agent uses Edge rather than run without it, unless you set `allow_unsandboxed_browser`.
- **The on-page status box** says only "GhostAgent - step N", never your task or the answer, because a page's own scripts could read it. Text from pages is cleaned of control sequences before the console shows it.

### The remote-debugging switch

To work in your everyday Chrome or Edge, GhostAgent needs its "Allow remote debugging for this browser instance" switch, which stays on until you turn it off. While it is on, any program running under your Windows account can ask to control the browser, with full access to your profile: cookies, saved data and open tabs. The browser asks "Allow remote debugging?" each time, but the question does not say which program is asking. So:

- click Allow only right after you start GhostAgent;
- turn the switch off at `chrome://inspect/#remote-debugging` (or `edge://inspect/#remote-debugging`) when you are done; GhostAgent reminds you when it exits;
- or use `"browser": "chrome-agent"`: the installed Chrome with a separate profile, no switch and no Allow click, at the cost of signing in to sites there separately.

On a non-English Chrome the switch's label is translated; it is the switch on the `chrome://inspect/#remote-debugging` page.

### Data at rest

- `bundled` and `edge` keep the browser profile **on the stick** (`GhostAgent\browser-profile`, `browser-profile-edge`). Windows encrypts cookie sign-ins for the PC and account where they were made, so those do not work on another PC, but history, site storage and the tokens some sites keep outside cookies are not encrypted. Treat the stick like a key, and consider encrypting it with BitLocker To Go. `"profile_on_stick": false` keeps the profile on the PC instead.
- The logs on the stick (`GhostAgent\logs`, the newest 20 of each kind are kept): `agent-*.log` is a copy of the console, so it includes your tasks and the answers; `steps-*.jsonl` has each step's timings, action and page address without its query part, with typed text and answers only as their length (`"step_trace": false` turns it off); `llama-server-*.log` is the model server's log. Your Windows user folder is written as `%USERPROFILE%` in them.
- `trace_messages` (off) saves every request, screenshots included, for benchmarks. `save_screenshots` (off) keeps screenshots on the PC, not the stick, and deletes them after 7 days.

### What GhostAgent leaves on a PC

`/forget-pc` lists these and deletes them after you confirm; anything still in use goes when GhostAgent closes.

- The model copy in `%LOCALAPPDATA%\GhostAgent\models`, from `copy_model_to_pc` or left by the build on the PC that built the stick.
- Saved screenshots, and the bundled browser's cache in `%TEMP%\ghost-agent-browser-cache-<id>`, one folder per browser profile (normally removed when the browser closes; one left by a crash goes at the next start, unless another GhostAgent window is using it).
- Browser profiles kept on the PC (`chrome-agent`, or `"profile_on_stick": false`).
- Logs and the browser profile kept on the PC when the stick could not be written to.
- `pc.json`, this PC's answers to GhostAgent's questions.

`/forget-pc` does not delete files the agent downloaded (`Downloads\GhostAgent`), and it does not turn off the remote-debugging switch in your Chrome or Edge.


## Managed PCs and Smart App Control

- If Windows says your organization blocked a program (App Control, AppLocker, a group policy, or Microsoft Defender), GhostAgent cannot run on that PC. It names the part that was blocked and stops; it does not try to get around the rule. Ask whoever manages the PC.
- **Smart App Control.** llama.cpp's Windows builds are not code-signed, so a PC that enforces Smart App Control blocks them, and Smart App Control has no per-program exception. Turning it off is a change for the whole PC, and turning it back on afterwards may be hard or impossible depending on the Windows version. GhostAgent does not ask you to turn it off; use another PC instead. (SmartScreen is different: if it warns about a `.cmd` file, the ZIP was not unblocked. "More info > Run anyway", or unblocking the ZIP and extracting it again, is fine for files you got from this repo.)
- If an administrator turned off remote debugging in Chrome or Edge, the agent uses the bundled browser.
- If a policy makes USB drives read-only, GhostAgent keeps its logs and the browser profile in `%LOCALAPPDATA%\GhostAgent` and says so.
- Proxies: the build uses `-Proxy`, the `HTTPS_PROXY` / `HTTP_PROXY` variables, or a fixed proxy in the Windows proxy setting (an automatic-configuration script is not read). It tolerates HTTPS inspection that blocks revocation checks, and runs under PowerShell's Constrained Language Mode. The agent's own requests to its model server never go through a proxy.


## Troubleshooting

`README.txt` has the full list. The most common ones:

- **The build stops**: read the red message. It names the problem (a FAT32 stick, too little free space, a download that failed, a part Windows would not run). Run it again once that is sorted out; it continues where it stopped.
- **"llama-server.exe cannot start without the Microsoft Visual C++ ... runtime"**: install the "Microsoft Visual C++ Redistributable (x64)" from https://aka.ms/vs/17/release/vc_redist.x64.exe (administrator rights needed; the agent names the ARM64 one for a stick built with ARM64 parts), then start GhostAgent again.
- **Chrome never asks "Allow remote debugging?"**: check the switch at `chrome://inspect/#remote-debugging` and that a normal Chrome window is open (pick your profile if Chrome shows its profile picker). Press B to use the bundled browser instead.
- **The model loads slowly**: below about 45 MB/s the agent suggests a USB 3 port. `copy_model_to_pc` makes later starts fast.
- **Wrong GPU, or a crash while loading**: in `auto` the next llama.cpp build is tried by itself, and the failing one is skipped next time (`logs\backend-cache.json`). Set `backend` to force one.
- **Logs**: `GhostAgent\logs\agent-*.log` (the console) and `llama-server-*.log` (the model server).


## Benchmarks and tests

`tools/bench` holds the benchmark: a local website with 12 tasks and automatic checks, a scripted stand-in for llama-server, a scorer and a replay tool (see [tools/bench/README.md](tools/bench/README.md)). One command runs the suite unattended with the bundled browser and starts the local site by itself:

```
GhostAgent\python\python.exe -I GhostAgent\agent\ghost_agent.py --tasks-file tools\bench\suite.json --results GhostAgent\logs\bench.jsonl --browser bundled
```

`--emulate-free-vram-mib N` lays the model out as if the GPU had only N MiB free (to reproduce an 8 GB laptop on a bigger card), and `--affinity-mask HEX` keeps the model server on chosen CPUs. When the output goes to a file or a pipe rather than a console, it is plain text: no colours, and progress is written as an occasional line instead of one line redrawn in place.

The tests need no GPU, no model and no network beyond localhost, and use a headless browser:

```
GhostAgent\python\python.exe -I tests\run_tests.py
```


## Layout

```
Build GhostAgent.cmd       downloads everything that is not in git (safe to run again)
Start GhostAgent.cmd       the launcher (works from any drive letter)
Update GhostAgent.cmd      gets the newest version, then runs the build
README.txt                 the user guide that lives on the stick
DECISIONS.md               why each server flag and default is what it is
VERSION                    this release's version number
tools/
  Build-GhostAgent.ps1     the build and update script
  profiles.json            model profiles: files, Hugging Face revisions, sizes, SHA-256
  requirements.lock        Playwright and greenlet wheels, pinned by SHA-256
  server-flags.json        every llama-server flag the agent uses (checked by the build)
  bench/                   benchmark site, stand-in model server, scorer, replay tool
tests/                     unit and headless browser tests (no GPU or model needed)
GhostAgent/
  agent/ghost_agent.py     starts the agent
  agent/ghostagent/        the agent's modules (its __init__.py lists them)
  settings.default.json    every setting with its default and a note (replaced by updates)
  settings.json            your own changes (created on the first start; not in git)
  models/                  SOURCE.txt and SHA256SUMS.txt in git; the .gguf files are downloaded
  licenses/                licenses of the downloaded parts
  bin/ python/ browsers/   downloaded by the build (not in git)
  logs/ browser-profile*/  created while the agent runs (not in git)
```


## License

The code in this repo is MIT-licensed (see `LICENSE`). The parts that `Build GhostAgent.cmd` downloads keep their own licenses, listed in `GhostAgent/licenses/`:

- Holo4-35B-A3B (the fast and fast-q8 profiles): Apache-2.0
- Holo4-27B (the accurate profile, downloaded only on request): CC BY-NC 4.0, non-commercial use only
- llama.cpp: MIT
- Playwright: Apache-2.0
- Python: PSF
- Chromium (Chrome for Testing): its notices are on the `chrome://credits` page of the bundled browser
