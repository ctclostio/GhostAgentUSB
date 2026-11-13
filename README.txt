HOLO BROWSER AGENT - a local AI that uses your web browser, run from this USB drive
===================================================================================

WHAT IT IS
  You type a task ("find the weather in Denver this weekend"). A vision model looks at
  the browser window, then clicks, types and scrolls until the task is done, and gives
  you the answer.

  Model    Holo4-35B-A3B by H Company (Apache 2.0), Q4_K_M GGUF. Built on Qwen3.6-35B-A3B
           and trained for computer use. 35B parameters but only 3B active per token, so
           most of it sits in system RAM while the GPU does the heavy lifting.
  Runtime  llama.cpp b11669 (v0.7.0), CUDA 13 / CUDA 12 / Vulkan / CPU builds included
  Browser  Your own Google Chrome: the agent opens its own window in the Chrome you already
           use, so it has your sign-ins, bookmarks and extensions. (A separate portable
           Chromium is included too, see "browser" under SETTINGS.)
  Python   3.14 embeddable with Playwright 1.63 (nothing gets installed on the laptop)

  No cloud AI service is used. The model server only listens on 127.0.0.1.
  The websites you send it to still need internet, of course.

FIRST TIME ONLY: LET THE AGENT USE YOUR CHROME (needs Chrome 144 or newer)
  1. In Chrome, open a new tab and go to   chrome://inspect/#remote-debugging
  2. Turn on "Allow remote debugging for this browser instance". It stays on.
  If you forget, the agent shows these steps and copies the address for you.

START
  Double-click "Start Holo Browser Agent.cmd". If Chrome is not open, it starts it.
  Chrome asks "Allow remote debugging?" - click Allow (it asks every time the agent starts).
  A new Chrome window opens for the agent. Then the model loads: from this USB drive that
  takes about 5 minutes (22 GB at ~70 MB/s). The console then shows "Task>". Type a task,
  press Enter.

  Faster starts: set "copy_model_to_pc": true in HoloAgent\settings.json. The next start
  copies the model to the laptop's drive once (%LOCALAPPDATA%\HoloAgent, 22 GB, about the
  same 5 minutes); after that it loads in well under a minute. To undo, delete that folder.

USING IT
  - The agent only works in its own Chrome window. Your other windows and tabs are left
    alone and the model never sees them. Keep using Chrome normally meanwhile.
  - Watch its window while it works. A small box in the bottom-right corner shows the
    step; the answer appears there and in the console window.
  - Ctrl+C in the console window stops the current task.
  - Follow-ups work ("now open the second result"). /new forgets earlier tasks.
  - Between tasks you can use the agent's window yourself, for example to sign in to a
    site; it continues from whatever page is showing there.
  - It works with YOUR accounts, signed in as you. It is told to ask you before buying,
    sending, posting, deleting or changing settings, but watch it on sites that matter.
  - Downloads go to Chrome's normal Downloads folder.
  - /quit exits, or just close the console window. Chrome and the agent's window stay open.
  - While connected, Chrome shows "Chrome is being controlled by automated test software".
  - While it runs, the model also has a plain chat page in any browser at the server
    address shown in the console (normally http://127.0.0.1:18931).

ABOUT THE CHROME SWITCH
  "Allow remote debugging" lets programs on this laptop ask to control Chrome. Chrome asks
  you (Allow / Cancel) every time one tries, so only click Allow right after starting the
  agent. To switch it off again, use the same chrome://inspect/#remote-debugging page.

HARDWARE
  Set up for an 8 GB NVIDIA GPU + 64 GB RAM. It needs about 23 GB of free RAM.
  The GPU is picked automatically: NVIDIA (CUDA 13, else CUDA 12 for older drivers), then
  AMD/Intel cards (Vulkan), then CPU only (works, but each step is several times slower).
  With 8 GB of VRAM it uses about 6.5 GB: all expert weights stay in RAM and the rest of
  the VRAM caches the most used experts. Games or other GPU programs slow it down a lot.
  Speed: about 4-10 s per step measured on a desktop with the same 8 GB limit; expect
  roughly 2x that on a laptop. Simple lookups take 2-4 steps, multi-page tasks 5-20.

SETTINGS
  HoloAgent\settings.json (each setting has a note). Most useful: copy_model_to_pc,
  browser, start_url, max_steps, backend, save_screenshots.
  "browser":  "chrome"  = a window in your own Chrome (default)
              "bundled" = the separate portable Chromium on this drive, with its own
                          profile in HoloAgent\browser-profile (sign-ins are stored ON THIS
                          DRIVE) and downloads in HoloAgent\downloads
              "edge"    = Microsoft Edge with its own profile on this drive

TROUBLESHOOTING
  - Logs: HoloAgent\logs\llama-server-*.log (the console prints the end of it on errors).
  - "already running": another Holo window is open; close it first.
  - Chrome never asks "Allow remote debugging?": make sure the switch above is on and that
    a normal Chrome window is open (if Chrome shows its profile picker, pick your profile).
  - Clicked Cancel by mistake: press Enter in the console and Chrome asks again.
  - No such switch on the chrome://inspect page: Chrome is older than version 144 (update
    it), or the laptop is managed and its administrator turned remote debugging off. Use
    "browser": "bundled" instead. (When Chrome is not installed, or its policy forbids
    remote debugging, the agent notices and uses the bundled browser by itself.)
  - Windows refuses to run llama-server.exe ("Smart App Control" or SmartScreen): these are
    the official, unsigned llama.cpp builds. SmartScreen: "More info" > "Run anyway".
    Smart App Control has no per-app exception; it must be off to run unsigned programs.
  - Wrong GPU or crash at load: set "backend" in settings.json to "cuda12", "vulkan" or "cpu".
  - Out of memory: close other programs (the model needs ~23 GB of RAM), or lower
    "moe_cache_mib" if the GPU runs out.
  - The bundled browser does not start: set "browser" to "edge".

WHY THIS MODEL (and not gpt-oss-120b)
  gpt-oss-120b cannot see the screen (text only) and its weights alone are ~61 GB, which
  does not leave room for Windows and a browser in 64 GB of RAM. Holo4-35B-A3B is a vision
  model trained for exactly this job, fits with plenty of headroom, and generates ~50
  tokens/s with this setup.

FOLDERS
  HoloAgent\agent      the agent (holo_agent.py)
  HoloAgent\bin        llama.cpp builds (cuda13, cuda12, vulkan, cpu)
  HoloAgent\models     model + vision projector (SHA-256 sums in models\SHA256SUMS.txt)
  HoloAgent\python     portable Python with Playwright
  HoloAgent\browsers   portable Chromium (used when "browser" is "bundled")
  HoloAgent\licenses   licenses of the bundled parts
