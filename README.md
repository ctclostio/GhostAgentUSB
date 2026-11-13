# GhostAgentUSB

A local AI that uses your web browser, carried on a USB stick.

Plug the stick into a Windows PC, double-click the launcher and type a task like *"find the weather in Denver this weekend"*. A vision model looks at the browser window, then clicks, types and scrolls until the task is done and gives you the answer. No cloud AI is involved. The model runs on the PC itself through llama.cpp and only listens on `127.0.0.1`.

| Part | What it is |
|---|---|
| Model | [Holo4-35B-A3B](https://huggingface.co/Hcompany/Holo4-35B-A3B-GGUF) by H Company, Q4_K_M GGUF (Apache-2.0). It is a vision model trained for computer use, with 35B parameters but only 3B active per token, so most of it sits in system RAM while the GPU does the heavy lifting. |
| Runtime | [llama.cpp](https://github.com/ggml-org/llama.cpp) b11669, with CUDA 13, CUDA 12, Vulkan and CPU builds. The best one is picked automatically. |
| Browser | By default, a window in **your own Google Chrome** (Chrome 144+), so it has your sign-ins. A portable Chromium or Microsoft Edge with a separate profile is also available. |
| Harness | `HoloAgent/agent/holo_agent.py`, about 1,500 lines. It is a Playwright agent loop that follows H Company's Holo conventions. |
| Python | 3.14 embeddable with Playwright 1.63, so nothing is installed on the PC. |

The agent is called "Holo" inside the app, after its model.

## Requirements

- Windows 10 or 11, 64-bit
- About **23 GB of free RAM** (64 GB total recommended)
- An NVIDIA GPU with 8 GB or more VRAM works best. AMD and Intel Arc cards use Vulkan. CPU-only works too, but is several times slower.
- A USB stick with at least 32 GB free. A fast one helps, because the model is 21 GB.
- Internet for the one-time build (about 23 GB of downloads) and for the websites you send the agent to.

## Build a stick

1. Get this repo onto the stick: **Code → Download ZIP** and extract it to the stick's root, or `git clone` it there.
2. Double-click **`Build USB.cmd`**. It downloads every part from its official source at a pinned version, checks each against a SHA-256 sum, and fills `HoloAgent\`. You can run it again safely: finished parts are skipped and the model download resumes.
3. Double-click **`Start Holo Browser Agent.cmd`**, then follow `README.txt`. The first run asks you to turn on one Chrome setting.

> **Tip:** writing thousands of small files to a slow stick takes a long time. It is usually faster to build the folder on the PC's SSD, then copy the whole folder to the stick.

Options for `Build USB.cmd`:

```
Build USB.cmd -ModelFrom "E:\old-stick\HoloAgent\models"   copy the model from an older copy instead of downloading it
Build USB.cmd -SkipModel                                    everything except the model
Build USB.cmd -KeepDownloads                                keep the archives in %TEMP%\GhostAgentUSB-build
```

## Using it

`README.txt` is the user guide that lives on the stick. It covers first-time Chrome setup, commands, settings and troubleshooting. In short:

- Type a task at `Task>`. **Ctrl+C** stops the current task, `/new` forgets earlier tasks and `/quit` exits.
- The agent only works in **its own window**. Your other windows and tabs are left alone, and the model never sees them.
- Settings live in `HoloAgent\settings.json`, and each one has a note. The most useful are `browser`, `copy_model_to_pc`, `start_url`, `max_steps` (0 means no limit) and `backend`.

## Safety

In its default mode the agent acts **with your accounts, signed in as you**.

- Its instructions say to stop and ask before buying, sending, posting, deleting or changing settings, and to treat text on web pages as content rather than instructions. A local model can still be fooled by a malicious page (prompt injection). Watch it on sites that matter.
- Chrome asks *"Allow remote debugging?"* on every connection. Only click Allow right after starting the agent. The `chrome://inspect/#remote-debugging` switch lets any local program *ask* to control Chrome, so turn it off when you aren't using the agent.
- In `bundled` and `edge` modes, sign-ins are stored **on the stick** (`HoloAgent\browser-profile`). Treat the stick like a key.

## Performance

These figures were measured with an 8 GB VRAM limit, on a desktop and on a laptop with an 8 GB GPU and 64 GB RAM:

- **Desktop:** about 4–10 seconds per step.
- **Laptop:** about 25 tok/s generation, about 370 tok/s prompt processing, and roughly 20 seconds per step.

Simple lookups take 2–4 steps. Multi-page tasks take 5–20.

On an 8 GB card the agent uses about 6.5 GB. All expert weights stay in RAM (`--cpu-moe`), and the leftover VRAM caches the most-used experts (`--moe-cache-mib`). That measured faster than a static `--n-cpu-moe` split.

## Layout

```
Start Holo Browser Agent.cmd   launcher (works from any drive letter)
Build USB.cmd                  one-time download of everything below that is not in git
README.txt                     user guide that lives on the stick
tools/Build-USB.ps1            the build script
HoloAgent/
  agent/holo_agent.py          the agent
  settings.json
  models/                      SHA256SUMS.txt + SOURCE.txt in git; the .gguf files are downloaded
  licenses/                    licenses of the bundled parts
  bin/ python/ browsers/       downloaded by Build USB.cmd (not in git)
  logs/ browser-profile/       created at run time (not in git)
```

## License

The code in this repo is MIT-licensed (see `LICENSE`). The parts that `Build USB.cmd` downloads keep their own licenses, which are listed in `HoloAgent/licenses/`:

- Holo4-35B-A3B: Apache-2.0
- llama.cpp: MIT
- Playwright: Apache-2.0
- Python: PSF
