GHOSTAGENT - a local AI that uses your web browser, run from this USB drive
===========================================================================

WHAT IT IS
  You type a task ("find the weather in Denver this weekend"). A vision model looks at
  the browser window, then clicks, types and scrolls until the task is done, and gives
  you the answer.

  Model    Holo4 by H Company, trained for computer use. Three profiles (see MODEL
           PROFILES): fast = Holo4-35B-A3B Q4_K_M (the default), fast-q8 = the same model
           at Q8_0, accurate = Holo4-27B Q4_K_M (non-commercial license).
  Runtime  llama.cpp b11669 (b12359 optional), with CUDA 13 / CUDA 12 / Vulkan / CPU builds
  Browser  By default a window in your own Google Chrome, so it has your sign-ins. Your
           own Edge, Chrome with a separate profile, or a portable browser on this drive
           are settings away (see BROWSER MODES).
  Python   3.14 embeddable with Playwright (nothing gets installed on the PC)

  No cloud AI service is used. The model server only listens on 127.0.0.1 and only
  answers requests that carry a key made for this run. The websites you send the agent
  to still need internet, of course.

  This is version 0.2.0 (see the VERSION file). README.md has the longer technical
  description; DECISIONS.md explains why the model server runs the way it does.

WHAT IT NEEDS
  - Windows 10 or 11, 64-bit.
  - Free RAM for the model: about 22 GB (fast), 19 GB (accurate) or 40 GB (fast-q8).
    64 GB of RAM in total is recommended. The agent warns you before loading when the
    PC has too little.
  - Ideally an NVIDIA graphics card with 8 GB or more. AMD and Intel Arc cards use
    Vulkan. Without a usable graphics card it runs on the CPU, several times slower.
  - The Microsoft Visual C++ Redistributable (x64; ARM64 for a drive built with
    ARM64 parts), which llama.cpp needs. Most PCs have it; the agent says so if
    this one does not (see TROUBLESHOOTING).
  - A USB 3 port (often blue, or marked SS). A USB 2 port makes every start much slower.

FIRST START
  1. Double-click "Start GhostAgent.cmd". A console window opens; that is where you type
     tasks and where the agent asks you things.
  2. When the model is on a USB drive, the first start on a PC may ask:
       Copy the 22.2 GB model to this PC, so the agent starts in 10-25 s instead of
       about N min? [Y/n]
     Yes copies it to %LOCALAPPDATA%\GhostAgent\models in the background (checked as it
     copies, and resumed if interrupted). The answer is kept for that PC. Say no on a PC
     that is not yours, or that is short of disk space.
  3. The model starts loading while the browser connects. With the default browser
     (your Chrome, version 144 or newer), the agent needs one Chrome setting, once:
       a. In Chrome, open a new tab and go to   chrome://inspect/#remote-debugging
          (the agent copies this address for you; paste it with Ctrl+V).
       b. Turn on "Allow remote debugging for this browser instance". It stays on.
       c. Chrome asks "Allow remote debugging?" - click Allow. Chrome asks this every
          time the agent starts.
     On a non-English Chrome the labels are translated: it is the switch on that page.
     While the agent waits for Chrome, press B to use the browser on this drive instead,
     for this session. If you would rather not turn the switch on at all, see BROWSER
     MODES ("chrome-agent" or "bundled").
  4. A new Chrome window opens for the agent. The console shows how far the model has
     loaded (MB/s and time left). From a USB drive that takes minutes, from a copy on
     the PC's SSD seconds.
  5. The console shows "Task>". Type a task and press Enter.

USING IT
  - The agent only works in its own window. Your other windows and tabs are left alone,
    and the model never sees them. Keep using Chrome normally meanwhile.
  - Watch its window while it works. A small box in the bottom-right corner shows
    "GhostAgent - step N"; the answer appears in the console.
  - After an answer, your next task continues the same conversation, so you can reply to
    a question the agent asked ("yes, send it") or follow up ("now open the second
    result"). /new starts a new conversation.
  - Text pasted as several lines counts as one task.
  - Ctrl+C in the console stops the current task (as soon as the browser action in
    progress finishes); the model stays loaded. Ctrl+C twice within 3 seconds closes
    GhostAgent.
  - Between tasks you can use the agent's window yourself, for example to sign in to a
    site; it continues from whatever page is showing there.
  - When several actions in a row change nothing on the page, the model is told so and
    asked to try another way. If it keeps going round in circles, press Ctrl+C and
    describe the task differently.
  - While connected, Chrome shows "Chrome is being controlled by automated test
    software". That is expected.

  Commands at the Task> prompt:
    /new        start a new conversation (earlier tasks are kept only as a short summary)
    /copy       copy the last answer to the clipboard
    /forget-pc  remove what GhostAgent keeps on this PC (see PRIVACY)
    /help       the short help
    /quit       exit (or just close the window). With your own Chrome or Edge, the
                browser and the agent's window stay open.

QUESTIONS THE AGENT ASKS YOU
  The console beeps, comes to the front and asks; type y and Enter for yes. No answer
  within 2 minutes counts as no.
  - When a web page shows a confirm or prompt box (for example "Delete all items?").
    Setting: dialog_policy.
  - Before the agent clicks a button or link that says Delete, Remove, Send, Post, Pay,
    Buy, Place order, Checkout, Transfer, Subscribe, Reset or similar (or presses Enter
    in a form whose button says so), unless your task asked for exactly that. A yes
    holds for that button on that site until the task ends. Harmless buttons such as
    "Remove filter" also cost one question. Setting: confirm_risky_actions.

WHAT THE AGENT WILL NOT DO (do these yourself between tasks)
  - Type into password or one-time-code fields: it asks you to type them.
  - Paste from the clipboard.
  - Choose a file to upload: file pickers are closed, and it asks you to choose the file.
  - Open file: addresses, or the browser's own pages (chrome://, extensions), unless
    you turn on allow_file_urls or allow_internal_pages.
  - Open addresses on this PC or your local network (a router, a NAS, an intranet site,
    localhost), unless your task names them, allowed_hosts lists them, or you had them
    open yourself.
  - Open files it downloaded.

DOWNLOADS
  - In your own Chrome or Edge, downloads go wherever your browser puts them; the agent
    is only told that a download started and finished.
  - In the agent's own browsers (bundled, edge, chrome-agent), downloads go to your
    Downloads\GhostAgent folder on this PC, under a new name when the name is taken, and
    are marked as downloaded from the internet. A download over 500 MB is stopped and
    deleted (download_max_mb). "downloads": "off" cancels every download.

SPEED (WHAT TO EXPECT)
  Measured before version 0.2.0, on a laptop with an 8 GB NVIDIA GPU and 64 GB of RAM,
  with the fast model loaded from a copy on the laptop's SSD (27 real tasks, October
  2026):
  - About 23 seconds per step at the median (one look at the page plus one action).
    Steps got slower as a task went on: in one 40-step task, from about 17 to about 33
    seconds.
  - The median task took about 26 steps, and a 40-step task took 16 to 21 minutes.
  - Loading the model took about 5 minutes from the USB drive and 10-25 seconds from a
    copy on the PC.
  Version 0.2.0 changes how the model server reuses its earlier work, which should make
  long tasks faster, but that has not been measured yet. New numbers will be measured
  and this section updated.
  The accurate profile is several times slower per step: in a first check on a desktop
  held to an 8 GB GPU budget, about 7 times as slow as fast (4.5 times with
  "reasoning_effort": "medium"). Laptops are slower than that desktop.
  Tips:
  - Keep the agent's window at its size. A maximized window makes every screenshot, and
    so every step, slower (the agent resizes its window at the start of each task).
    A smaller window does not make it faster.
  - Games and other programs that use the graphics card slow it down a lot.

MODEL PROFILES
  fast      Holo4-35B-A3B Q4_K_M by H Company, 22.2 GB, Apache 2.0. A mixture-of-experts
            model: only about 3B of its 35B parameters work on each token, so most of it
            can sit in system RAM. The fastest option on an 8 GB graphics card.
  fast-q8   The same model at Q8_0 (a community file by bartowski), 37.8 GB, Apache 2.0.
            Closer to the original weights, but needs about 40 GB of free RAM and each
            step is slower. Whether that helps the agent has not been measured.
  accurate  Holo4-27B Q4_K_M by H Company, 17.8 GB, CC BY-NC 4.0: NO COMMERCIAL USE,
            which can include work done for an employer. H Company's most capable model,
            noticeably better on long tasks, but every part of it works on every token,
            so on an 8 GB card it is several times slower per step. reasoning_effort
            "medium" or "low" makes it faster, at some cost in care.
  To use another profile: run   "Build GhostAgent.cmd" -Profile accurate   (or fast-q8,
  or all) to download it, then put "model_profile": "accurate" in settings.json.
  The default is fast. The other two profiles have not been benchmarked; "Run
  benchmark.cmd" measures them on this PC (see SETTINGS).

SETTINGS
  GhostAgent\settings.default.json lists every setting with its default value and a
  note. Do not edit that file: updates replace it. To change a setting, copy its line
  into GhostAgent\settings.json (created on the first start), change the value, save,
  and restart the agent. A wrong value is reported and its default used instead.
  Example settings.json:
    {
      "browser": "bundled",
      "copy_model_to_pc": true
    }

  Most useful:
    model_profile       fast, fast-q8 or accurate (see MODEL PROFILES)
    browser             which browser the agent uses (see BROWSER MODES)
    copy_model_to_pc    "ask" (default), true or false: load the model from a copy on
                        this PC (seconds) instead of from this drive (minutes)
    start_url           the page the browser opens at start
    max_steps           0 = no limit (default); otherwise give up after this many steps
    progress_checkin_steps   every N steps, remind the model to check its strategy
    backend             auto (default), cuda13, cuda12, vulkan or cpu
    reasoning_effort    accurate profile only: "" (thorough), "medium" or "low"
    save_screenshots    true = keep screenshots for troubleshooting (on this PC, 7 days)
    web_ui              true = llama-server's own chat page in a browser, at the address
                        the console shows; it asks for the key the console prints
  Safety (see SAFETY):
    dialog_policy       ask (default), dismiss or accept
    confirm_risky_actions   true (default) = ask before Delete/Send/Pay-type clicks
    allowed_hosts       local-network addresses the agent may open, e.g. ["192.168.1.1"]
    allow_file_urls, allow_internal_pages, allow_unsandboxed_browser   (all false)
    downloads, download_max_mb, profile_on_stick

  Settings chosen by benchmark (12 local tasks on a GPU held to 8 GB, November 2026):
    image_eviction      "block" (default) or "rolling": block lets the model server reuse
                        its work, so steps are faster (reading the prompt took 2.8 s per
                        step instead of 5.4 s); the model sees 1 to 3 earlier screenshots
                        instead of 3 (image_keep_after_evict, default 1, says how many stay)
    prompt_version      "h" (default) or "classic": instructions along H Company's
                        published guidance
    tool_schema         "h" (default) or "classic": H Company's tool names and arguments,
                        a drag-and-drop tool and an outcome with each answer
    settle_mode         "adaptive" (default) or "fixed": waits after an action only until
                        the page is quiet; a page may be captured half-drawn
    observation_format  "classic" (default) or "header": page title, size and scroll
                        position around each screenshot (measured slower, so off)
    preserve_reasoning  "auto" (default: on for accurate, off for fast), true or false
    snap_viewport       false (default) or true: screenshots sized so the model server
                        does not resample them (measured no faster, so off)
    write_enter_default false (default) or true: Enter after typing unless told not to
                        (a message could be sent by mistake; the Send question guards it)
  "Run benchmark.cmd" runs the same 12 tasks on this PC with the bundled browser and
  prints the successes and the time per step. Add --settings-override try.json (a file
  shaped like settings.json) to try other settings or a model profile.
  Tuning settings for the model server (ctx_checkpoints, checkpoint_min_step,
  cache_ram_mib, moe_cache_mib, moe_cache_cap_mib, threads and others) are explained in
  their notes and in DECISIONS.md. Leave them alone unless you are benchmarking.

BROWSER MODES ("browser" in settings.json)
  "chrome"        A window in your own running Google Chrome (the default), with your
                  sign-ins. Needs the remote debugging switch (see FIRST START).
  "user_edge"     The same in your own Microsoft Edge. The switch is on
                  edge://inspect/#remote-debugging
  "auto"          Your Chrome if it can be used, else your Edge, else "bundled".
  "chrome-agent"  The installed Chrome with a separate profile kept on this PC
                  (%LOCALAPPDATA%\GhostAgent\chrome-profile): no switch and no Allow
                  click, and your own Chrome profile is never touched. You sign in to
                  sites there separately; those sign-ins stay on this PC.
  "bundled"       The portable Chromium on this drive. Its profile (sign-ins, history)
                  is kept ON THIS DRIVE in GhostAgent\browser-profile.
  "edge"          Microsoft Edge with its own profile on this drive.
  When Chrome is missing, older than 144, or this PC's policy forbids remote debugging,
  the agent says so and uses the bundled browser by itself.

SAFETY
  With "chrome" and "user_edge" the agent works with YOUR accounts, signed in as you.
  It is told to ask you before buying, sending, posting, deleting or changing settings,
  and to treat text on web pages as content, not instructions. A web page can still try
  to trick it, which is why the questions above, and the things it will not do, are
  enforced by GhostAgent itself rather than left to the model. Watch it on sites that
  matter.

  About the Chrome switch: "Allow remote debugging" lets any program running under your
  Windows account ask to control Chrome, with access to your cookies, saved data and
  tabs. Chrome asks you (Allow / Cancel) every time, but it does not say which program
  is asking, so only click Allow right after starting the agent. When you are done,
  turn the switch off on the same chrome://inspect/#remote-debugging page; the agent
  reminds you when it exits. "chrome-agent" mode needs no switch at all.

  The model server takes requests only with a key made at each start and only from this
  PC; its chat page is off unless web_ui is true. The agent's own browsers run with
  Chromium's security sandbox; if it cannot start from this drive, Edge is used instead.

PRIVACY: WHAT IS KEPT WHERE
  On this drive:
  - With "bundled" or "edge": the browser profile (history, site data, sign-ins).
    Cookie sign-ins only work on the PC where they were made (Windows encrypts them for
    that PC), but history and other site data are not encrypted. Treat this drive like
    a key; BitLocker To Go can encrypt it. profile_on_stick = false keeps the profile on
    the PC instead.
  - GhostAgent\logs (the newest 20 of each kind are kept):
      agent-*.log         a copy of the console window, including your tasks and the
                          answers
      steps-*.jsonl       one line per step: timings, the action and the page address
                          without its query part; typed text and answers only as their
                          length (step_trace = false turns it off)
      llama-server-*.log  the model server's own log
    Your Windows user folder is written as %USERPROFILE% in them.
  On this PC (/forget-pc lists these and deletes them after you say y):
  - the model copy (%LOCALAPPDATA%\GhostAgent\models), also left by the build on the PC
    that built this drive
  - saved screenshots (save_screenshots), the browser's cache in %TEMP% (normally
    deleted when the browser closes), the chrome-agent profile, a browser profile kept
    on the PC, logs kept on the PC when this drive was read-only, and pc.json (this PC's
    answers to GhostAgent's questions)
  Not removed by /forget-pc: files in Downloads\GhostAgent, and the remote debugging
  switch in your Chrome or Edge.

UPDATING
  Double-click "Update GhostAgent.cmd". It gets the newest version (git pull when this
  folder is a git clone, otherwise the latest ZIP from GitHub), then runs the build,
  which only replaces parts whose versions changed and offers to delete model files no
  profile uses any more. Your settings.json, browser profiles, downloads and logs are
  kept. Without access to GitHub on this PC:
    "Update GhostAgent.cmd" -UpdateFrom "D:\Downloads\GhostAgentUSB-main.zip"

MANAGED PCS AND SMART APP CONTROL
  - If Windows says your organization blocked a program (App Control, a group policy,
    Microsoft Defender), GhostAgent cannot run on that PC. It says which part was
    blocked and stops. Ask whoever manages the PC; do not try to get around it.
  - Smart App Control: the llama.cpp programs are official but not code-signed, and a
    PC that enforces Smart App Control blocks them. It has no per-program exception.
    Turning it off is a change for the whole PC that may be hard to undo, so use
    another PC instead.
  - SmartScreen warning ("Windows protected your PC") about a .cmd file: the ZIP was
    not unblocked before extracting. "More info" > "Run anyway" is fine for these files.
  - If the PC makes USB drives read-only, the logs and the browser profile are kept in
    %LOCALAPPDATA%\GhostAgent instead, and the agent says so.

TROUBLESHOOTING
  - Logs: GhostAgent\logs\agent-*.log (everything the console showed) and
    llama-server-*.log (the model server; the console prints its end on errors).
  - "already running": another GhostAgent window is open; close it first.
  - "llama-server.exe cannot start without the Microsoft Visual C++ ... runtime":
    install the "Microsoft Visual C++ Redistributable (x64)" from
    https://aka.ms/vs/17/release/vc_redist.x64.exe (needs administrator rights), then
    start GhostAgent again. On a drive built with ARM64 parts the agent names the
    ARM64 one instead.
  - Chrome never asks "Allow remote debugging?": make sure the switch is on and that a
    normal Chrome window is open (if Chrome shows its profile picker, pick your
    profile). Or press B to use the browser on this drive.
  - Clicked Cancel by mistake: press Enter in the console and Chrome asks again.
  - No such switch on the chrome://inspect page: Chrome is older than version 144
    (update it), or the PC is managed and its administrator turned remote debugging off.
    Use "browser": "bundled" or "chrome-agent".
  - "Missing model file" or "The model file is incomplete": run Build GhostAgent.cmd
    again (with -Profile <name> for a profile other than fast).
  - Slow start: the console warns when the model loads at under about 45 MB/s (a USB 2
    port or a slow drive). Use a USB 3 port, and say yes to copying the model to the PC.
  - Loading shows no progress for 2 minutes: the agent asks whether to keep waiting,
    start the model server again, or quit.
  - The model server stopped during a task: the agent starts it again (at most twice
    per session) and you can type the next task.
  - Wrong graphics card or a crash while loading: with "backend": "auto" the next
    llama.cpp build is tried by itself, and one that failed is skipped next time (delete
    GhostAgent\logs\backend-cache.json to try it again). To force one, set "backend" to
    "cuda12", "vulkan" or "cpu".
  - Out of memory: close other programs (the fast model needs about 22 GB of free RAM),
    or lower "moe_cache_mib" if the graphics card runs out.
  - "Windows App Control ... blocked" or "blocked by group policy": see MANAGED PCS.
  - The bundled browser does not start: the agent switches to Edge by itself. "The
    browser profile is in use": close the other GhostAgent window.
  - A settings.json mistake: the console shows the line and column. Fix it and start
    again.
  - "The page is busy": the page is running a long script; the agent waits or goes
    elsewhere.
  - Cannot select text in the console with the mouse: GhostAgent turns off the
    console's click-to-select mode while it runs, because a click there would pause the
    agent. Use /copy to copy the last answer.

FOLDERS
  Start GhostAgent.cmd     starts the agent
  Build GhostAgent.cmd     downloads or repairs everything below (safe to run again)
  Update GhostAgent.cmd    gets the newest version, then runs the build
  Run benchmark.cmd        runs the local benchmark tasks and prints the score
  GhostAgent\agent         the agent (ghost_agent.py starts the modules in its
                           ghostagent folder)
  GhostAgent\bin           llama.cpp builds (cuda13, cuda12, vulkan, cpu; no cuda12 on ARM64)
  GhostAgent\models        model files (pinned versions: tools\profiles.json)
  GhostAgent\python        portable Python with Playwright
  GhostAgent\browsers      portable Chromium (used when "browser" is "bundled")
  GhostAgent\licenses      licenses of the downloaded parts
  GhostAgent\logs          logs (see PRIVACY)
  GhostAgent\settings.json         your settings (only what you changed)
  GhostAgent\settings.default.json every setting with its default and a note
