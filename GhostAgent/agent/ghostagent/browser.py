"""The browser the agent drives (Playwright): starting or attaching to it, observations, and every
tool's action."""

import base64
import contextlib
import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

from . import paths, runstate
from .browsercache import (CACHE_LOCK_FILE, FolderLock, browser_cache_dir, empty_folder, prune_browser_caches,
                           remove_browser_cache)
from .console import c, clean, first_line, info, one_line, warn
from .guards import (ERROR_SCHEMES, HEADLESS_UNSAFE_DEFAULT_SWITCHES, INTERNAL_SCHEMES, PASSWORD_REFUSAL,
                     RISKY_REFUSAL, ToolRefused, UNSAFE_DEFAULT_SWITCHES, host_is_local, host_listed, normalise_host,
                     playwright_feature_switches, risky_words, untrusted, url_scheme, vet_url)
from .keys import BadKeys, PASTE_REFUSAL, activates, is_paste, parse_keys, types_text
from .modelclient import StepCancelled, ctrl_c_stops_task
from .pagehelpers import (BUSY_RESULT, PageBusy, READER_JS_SECONDS, STALL_ALERT, StallDetector, action_signature,
                          mark_of_the_web, observation_footer, observation_header, page_js, page_title, reader_slice,
                          safe_download_name, size_text, title_is_address, unique_path, with_alert)
from .pagejs import (FIELD_VALUE_JS, FOCUSED_FIELD_JS, FOCUSED_FRAME_JS, FOCUSED_LABEL_JS, FOCUSED_SECRET_JS, LABEL_JS,
                     METRICS_JS, OVERLAY_JS, QUIET_JS, READER_JS, SCROLL_PROBE_JS, SECRET_AT_JS, SELECT_JS,
                     SELECT_OPTIONS_JS, SIGNATURE_JS)
from .paths import ROOT, download_folder, hide_home, log_only, pc_data_dir, screenshots_dir
from .questions import _keyboard, ask_console, ask_yes_no
from .runstate import STOP
from .screenshot import BIG_SCREENSHOT_PIXELS, MIN_IMAGE_TILES, snap_viewport_size
from .tools import canonical_args
from .userbrowser import (BrowserGone, REFUSED_RE, USER_BROWSERS, UseBundled, attach_problem, lend_clipboard,
                          pick_auto_browser)
from .winproc import CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW, blocked_message


def _bypass_proxy_for_localhost():
    """Make sure local addresses never go through a proxy configured in the environment
    (Playwright's driver reads NO_PROXY for its own connections, for example to Chrome's DevTools)."""
    current = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    entries = [e.strip() for e in current.split(",") if e.strip()]
    for host in ("127.0.0.1", "localhost"):
        if host not in entries:
            entries.append(host)
    os.environ["NO_PROXY"] = ",".join(entries)


def _scrub_playwright_environment():
    """Playwright's driver is a Node.js program that inherits this process's environment. On a PC
    that sets NODE_OPTIONS it could load other code; other PLAYWRIGHT_* variables, or PWDEBUG (which
    opens Playwright's inspector and pauses), change how it runs. Only the browsers folder on this
    drive is kept."""
    for key in list(os.environ):
        name = key.upper()
        if name in ("NODE_OPTIONS", "PWDEBUG") or (name.startswith("PLAYWRIGHT_") and name != "PLAYWRIGHT_BROWSERS_PATH"):
            os.environ.pop(key, None)
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(ROOT / "browsers")


def _detach_playwright_driver_from_console():
    """Ctrl+C in the console goes to every process attached to it. Start Playwright's driver (and
    so the browser) without a console, so Ctrl+C only stops the current task."""
    if os.name != "nt":
        return
    import asyncio
    original = asyncio.create_subprocess_exec
    if getattr(original, "_ghost_patched", False):
        return

    async def create_subprocess_exec(*args, **kwargs):
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        return await original(*args, **kwargs)

    create_subprocess_exec._ghost_patched = True
    asyncio.create_subprocess_exec = create_subprocess_exec


class Browser:
    """What the agent drives. "chrome" / "user_edge": its own window in the user's running Google
    Chrome or Microsoft Edge, with their profile, sign-ins and extensions; their other windows and
    tabs are left alone. "auto": Chrome if it can be used, else Edge, else the bundled browser.
    "chrome-agent": the installed Chrome with a profile of the agent's own on this PC.
    "bundled" / "edge": a separate browser with its own profile on this drive."""

    MODES = ("chrome", "user_edge", "auto", "chrome-agent", "bundled", "edge")

    def __init__(self, settings, headless=False):
        self.s = settings
        self.mode = str(settings.get("browser") or "chrome").strip().lower()
        if self.mode not in self.MODES:
            warn(f'Unknown "browser" setting "{settings.get("browser")}"; using "chrome".')
            self.mode = "chrome"
        # Tests only (--headless-browser): the bundled Chromium without a window, with a throwaway
        # profile, so the agent can be exercised end to end without anything showing on screen.
        self.headless = bool(headless)
        if self.headless:
            self.mode = "bundled"
        self._temp_profile = None
        # Size of the page area. Updated from every screenshot, since the user's Chrome window can
        # be any size; model coordinates are scaled to it.
        self.w, self.h, _ = self._target_size()
        self.pw = None
        self.browser = None   # set when attached to the user's Chrome or Edge
        self.ctx = None
        self.page = None
        self.own = []         # the agent's tabs in the user's browser
        self.cdp = {}         # page -> its CDP session (focus emulation, window state, tab title)
        # What happened in the browser that the model has not been told yet (a dialog, a new tab, a
        # download). Told in the next tool result or, if it happened while the model was thinking,
        # in the next observation; then cleared.
        self.events = []
        # Stall check (see StallDetector): an alert for the next observation once actions keep
        # changing nothing.
        self.stall = StallDetector()
        self._alert = None
        self._scroll_probe = None     # scroll offsets from before a scroll_web, compared after it
        self._busy = False            # the current page did not answer a short script in time
        self._size_warned = False     # the "big window" warning, once per task
        self._fit_logged = False      # "the screen is too small for whole tiles", once per session
        # New tabs: one that opens within 2 s of the agent's own click, key press or typing becomes the
        # current tab; one a page opens by itself (a timer, a script) is only listed, so a page cannot
        # take the agent somewhere else behind its back.
        self._last_gesture = float("-inf")
        self._opening_tab = False     # the agent itself is opening a tab (switch_tab_web, a fresh tab)
        self._closing_tab = False     # the agent itself is closing a tab (close_tab_web)
        self._openers = {}            # tab -> the tab that opened it (a pop-up that closes returns there)
        self._by_page = set()         # tabs a page opened by itself, not switched to
        self._in_task = False         # a task is running: file pickers are caught only then
        self._choosers = set()        # tabs whose file pickers are being caught
        # Downloads in the agent's own browser that are still running: Download -> (name, file in progress).
        self._downloads = {}
        self._download_session = None  # chrome / user_edge: the CDP session that reports downloads
        self._cdp_downloads = {}       # chrome / user_edge: download id -> file name
        # settle_mode adaptive: per tab, the main-frame navigations under way and the requests in flight.
        self._net = {}
        # For the step trace: the last screenshot and page address, and how long the last action
        # and the wait after it took.
        self.last_shot = {"png_bytes": None, "png_w": None, "png_h": None}
        self.last_url = None
        self.last_timings = {"execute": None, "settle": None}
        self.last_ok = True           # the last action was done (see execute)
        # Questions for the user (a page's confirm(), a click on Delete) need someone at the console
        # who can see the browser; the headless test browser has nobody, so the answer is always no.
        self.can_ask = not self.headless
        # The current task's text: the risky-click and local-address checks read it.
        self.task_text = ""
        self._approved = set()        # (label, host) the user allowed during this task
        self._start_hosts = set()     # hosts of the start pages the user configured
        self._task_hosts = set()      # hosts open when the current task began (the user chose them)
        self._local = {}              # host -> on this PC or the local network (looked up once)
        self._agent_navigating = False  # True while the agent itself leaves a page (beforeunload)
        self._cache_dir = None        # the browser's disk cache in TEMP, removed on close
        self._cache_lock = None       # its FolderLock, while this agent's browser uses it
        self.target = None            # the UserBrowser attached to (chrome / user_edge)
        self._exe = None
        self._was_attached = False    # connected to the user's browser at some point in this session

    @property
    def attached(self):
        """The agent works in the user's own browser (Chrome or Edge), next to their own tabs."""
        return self.mode in ("chrome", "user_edge")

    @property
    def user_browser_name(self):
        target = self.target or USER_BROWSERS.get(self.mode)
        return target.name if target else "Chrome"

    @property
    def prompt_note(self):
        if self.attached:
            return (f"\n- You work in your own window of the user's everyday {self.user_browser_name}, where they may "
                    "already be signed in. Only the tabs listed in the observation are yours; the user's other tabs "
                    "are private.")
        return ""

    @property
    def quit_note(self):
        """Said when the session ends after the agent used the user's browser: the switch the user
        turned on stays on until they turn it off."""
        if not self._was_attached or self.target is None:
            return None
        return (f"Remote debugging stays allowed in your {self.target.name} until you turn it off at "
                f"{self.target.inspect_url} ({self.target.name} asks you before any program connects).")

    @property
    def windowed(self):
        """The page area is whatever size the browser window has (an installed browser's window),
        rather than a fixed viewport (the bundled browser and Edge with the agent's own profile)."""
        return self.attached or self.mode == "chrome-agent"

    def _target_size(self, max_w=None, max_h=None):
        """(width, height, ok): the page area to aim for, viewport_width x viewport_height within
        max_w x max_h, in whole 32-pixel tiles when snap_viewport is on (see snap_viewport_size)."""
        w, h = int(self.s["viewport_width"]), int(self.s["viewport_height"])
        if self.s.get("snap_viewport"):
            return snap_viewport_size(w, h, max_w, max_h)
        return (min(w, max_w) if max_w else w), (min(h, max_h) if max_h else h), True

    def start(self, start_url=None):
        """Open the browser at start_url (default: the start_url setting)."""
        _detach_playwright_driver_from_console()
        _bypass_proxy_for_localhost()
        _scrub_playwright_environment()
        try:
            from playwright.sync_api import sync_playwright
            self.pw = sync_playwright().start()
        except Exception as e:
            # An organization rule or Defender can block the browser control's program (node.exe) or
            # one of its DLLs: said plainly instead of with a traceback.
            msg = blocked_message(e, "the browser control that comes with GhostAgent (Playwright, in GhostAgent\\python)")
            if msg:
                raise SystemExit(msg) from None
            raise
        if self.mode == "auto":
            self.mode = pick_auto_browser()
            info({"chrome": "using your Chrome (browser auto)", "user_edge": "using your Edge (browser auto)"}.get(
                self.mode, "using the browser on this drive (browser auto: neither Chrome nor Edge can be used here)"))
        if self.attached:
            self.target = USER_BROWSERS[self.mode]
            self._exe, problem = attach_problem(self.target)
            if problem:
                warn(problem + " Using the browser on this drive instead.")
                self.mode, self.target = "bundled", None
        url = self._start_page(start_url or self.s["start_url"])
        if self.attached:
            self._attach_or_bundled(url)
        elif self.mode == "chrome-agent":
            self._launch_chrome_agent(url)
        else:
            self._launch_own(url)

    def _attach_or_bundled(self, url=None):
        try:
            self._attach_chrome(url)
        except UseBundled:
            info("using the browser on this drive for this session")
            self.mode, self.target, self.browser, self.own, self.cdp = "bundled", None, None, [], {}
            self._launch_own(url)

    def ensure_connected(self):
        """Between tasks: when the connection to the user's browser was lost (it was closed, or it
        crashed), connect again. Within a task a lost connection ends the task instead (BrowserGone)."""
        if not self.attached or self._connected(check=True):
            return
        warn(f"The connection to {self.user_browser_name} was lost (was it closed?). Reconnecting ...")
        self.browser, self.own, self.cdp = None, [], {}
        self._attach_or_bundled()

    def _start_page(self, url):
        """A start page the user configured (start_url, or a task list's start_url): http(s) or
        about:blank like everything else, but its host may be on this PC or the local network,
        since the user chose it. One that cannot be used is replaced by a blank page."""
        try:
            url = vet_url(url, self.s.get("allow_file_urls"), self.s.get("allow_internal_pages"))
        except ToolRefused as e:
            warn(f'The start page "{one_line(str(url), 100)}" cannot be used ({e}) A blank page opens instead.')
            return "about:blank"
        try:
            host = urllib.parse.urlsplit(url).hostname
        except ValueError:
            host = None
        if host:
            self._start_hosts.add(normalise_host(host))
        return url

    def _attach_chrome(self, url=None):
        """Connect to the user's Chrome or Edge (self.target) and open the agent's window in it."""
        # Ctrl+C while waiting stops the wait at the next check instead of interrupting Playwright
        # (a second Ctrl+C quits at once, for example while the browser's question is open).
        with ctrl_c_stops_task(clear=False):
            self._wait_for_user_browser()
        self._was_attached = True
        contexts = self.browser.contexts
        self.ctx = contexts[0]
        profile = self.target.profile_name()
        detail = f', profile "{clean(profile)}"' if profile and len(contexts) == 1 else ""
        info(f"connected to your {self.target.name} {self.browser.version}{detail}")
        if len(contexts) > 1:
            info(f"{len(contexts)} profiles are open in {self.target.name}; the agent's window opens in the first "
                 "one. Close the other profiles' windows if it should use a different one.")
        # One context-wide handler: dialogs in the agent's tabs follow dialog_policy, the ones in the
        # user's own tabs are left for the user (without any handler Playwright would dismiss them).
        self.ctx.on("dialog", self._on_dialog)
        self._watch_browser_downloads()
        self._new_window(url or self._start_page(self.s["start_url"]))

    def _watch_browser_downloads(self):
        """Chrome / Edge of the user's own: their own download handling stays as it is (where files go,
        Safe Browsing, the download bubble), but the agent is told when one of its tabs starts a
        download and when it ends, so the model knows a click downloaded something."""
        try:
            session = self.browser.new_browser_cdp_session()
            session.on("Browser.downloadWillBegin", self._on_cdp_download_begin)
            session.on("Browser.downloadProgress", self._on_cdp_download_progress)
            # "default" keeps the browser's own behaviour; only the events are switched on.
            session.send("Browser.setDownloadBehavior", {"behavior": "default", "eventsEnabled": True})
            self._download_session = session
        except Exception as e:
            log_only(f"download events are not available in this browser ({first_line(e)})")

    def _agent_frame(self, frame_id):
        """True when a CDP frame id belongs to one of the agent's tabs (its main frame, whose id is the
        tab's target id, or a frame inside it). Downloads in the user's own tabs are none of the
        model's business."""
        tabs = self._tabs()
        for p in tabs:
            if self._target_id(p) == frame_id:
                return True
        for p in tabs:
            session = self._session(p)
            try:
                tree = session.send("Page.getFrameTree")["frameTree"] if session else None
            except Exception:
                tree = None
            stack = [tree] if tree else []
            while stack:
                node = stack.pop()
                if node.get("frame", {}).get("id") == frame_id:
                    return True
                stack.extend(node.get("childFrames") or [])
        return False

    def _on_cdp_download_begin(self, params):
        if not self._agent_frame(params.get("frameId")):
            return
        name = safe_download_name(params.get("suggestedFilename") or "download")
        self._cdp_downloads[params.get("guid")] = name
        self.events.append(f"A download started: {name}. {self.user_browser_name} saves it as it saves your downloads; "
                           "GhostAgent never opens downloaded files.")

    def _on_cdp_download_progress(self, params):
        state = params.get("state")
        if state not in ("completed", "canceled"):
            return
        name = self._cdp_downloads.pop(params.get("guid"), None)
        if name is None:
            return
        self.events.append(f"The download of {name} finished." if state == "completed" else
                           f"The download of {name} was cancelled.")

    @staticmethod
    def _pressed_b(keyboard):
        """True when B was pressed in the console since the last look (other keys are dropped)."""
        try:
            while keyboard.kbhit():
                ch = keyboard.getwch()
                if ch in ("b", "B"):
                    return True
                if ch in ("\x00", "\xe0"):
                    keyboard.getwch()
        except Exception:
            pass
        return False

    def _wait_for_user_browser(self):
        t, exe = self.target, self._exe or self.target.find()
        if not t.running():
            info(f"starting {t.long_name} ...")
            t.start(exe)
            for _ in range(40):
                if t.running() or STOP.is_set():
                    break
                time.sleep(0.5)
            time.sleep(2)  # it writes DevToolsActivePort shortly after starting
        told = False
        tried = prompted = None
        quick_refusals = 0
        errors_shown = set()
        keyboard = _keyboard()
        give_back = None  # puts the user's own clipboard text back once the setup is over
        try:
            while True:
                if STOP.is_set():
                    raise StepCancelled()
                if keyboard is not None and self._pressed_b(keyboard):
                    raise UseBundled()
                endpoint = t.endpoint() if t.running() else None
                if endpoint and endpoint != tried:
                    if endpoint != prompted:
                        prompted = endpoint
                        print(c(f'  {t.name} asks "Allow remote debugging?" - click Allow in the {t.name} window.',
                                "1;33"), flush=True)
                    t0 = time.time()
                    try:
                        # timeout=0: the connection waits until Allow or Cancel is clicked.
                        self.browser = self.pw.chromium.connect_over_cdp(
                            endpoint, timeout=0, no_defaults=True, is_local=True)
                        break
                    except Exception as e:
                        if not REFUSED_RE.search(str(e)):
                            # A file left behind by a browser that was closed or crashed: wait for a new
                            # one. Each different error is shown once, so a real problem is not hidden.
                            tried = endpoint
                            if first_line(e) not in errors_shown:
                                errors_shown.add(first_line(e))
                                warn(f"Could not connect to {t.name}: {first_line(e)}")
                        elif time.time() - t0 < 2:
                            # Refused before anyone could answer: the browser has no window to ask in
                            # (it is starting, shows its profile picker, or runs with all windows closed).
                            quick_refusals += 1
                            if quick_refusals % 3 == 0:
                                warn(f"{t.name} needs an open window to ask in"
                                     + (". Opening one" if quick_refusals % 15 == 3 else "")
                                     + f"; if {t.name} shows its profile picker, pick your profile.")
                                if quick_refusals % 15 == 3:
                                    t.start(exe)
                            time.sleep(2)
                            continue
                        else:
                            warn(f"{t.name} refused the connection (Cancel was clicked).")
                            try:
                                answer = input(f"  Press Enter to ask {t.name} again, or type b and press Enter to use "
                                               "the browser on this drive. ")
                            except EOFError:
                                answer = ""
                            if answer.strip().lower() == "b":
                                raise UseBundled()
                            prompted = None
                            continue
                if not told and (not endpoint or endpoint == tried):
                    told = True
                    give_back = lend_clipboard(t.inspect_url)
                    print()
                    print(c(f"  To let the agent use your {t.name}, turn on remote debugging once (it stays on):", "1;33"))
                    print(f"    1. In {t.name}, open a new tab and go to  {t.inspect_url}"
                          + ("   (copied: paste it with Ctrl+V)" if give_back else ""))
                    print('    2. Turn on "Allow remote debugging for this browser instance".')
                    print(f'    3. Click Allow when {t.name} asks "Allow remote debugging?".')
                    print(c(f"  Waiting for {t.name} ... (press B to use the browser on this drive instead, for this "
                            "session)", "90"), flush=True)
                time.sleep(1)
        finally:
            if give_back:
                give_back()

    def _new_window(self, url):
        """Open the agent's own window in the user's Chrome, with a page area of about
        viewport_width x viewport_height (smaller if the screen is), and make it current."""
        tw, th, _ = self._target_size()
        cdp = self.browser.new_browser_cdp_session()
        try:
            with self.ctx.expect_page(timeout=20000) as new:
                cdp.send("Target.createTarget", {"url": "about:blank", "newWindow": True, "left": 0, "top": 0,
                                                 "width": tw + 16, "height": th + 140})
        finally:
            try:
                cdp.detach()
            except Exception:
                pass
        page = new.value
        self._adopt(page)
        self._fit_window(page)
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
        except Exception:
            pass
        # Again once a page shows: Chrome's "controlled by automated test software" bar may have
        # appeared meanwhile.
        self._fit_window(page)

    def _fit_window(self, page):
        """Size the window so its page area is viewport_width x viewport_height, or as much of that as
        the screen holds; with snap_viewport, whole 32-pixel tiles (see snap_viewport_size). True when
        the window could be sized."""
        session = self._session(page)
        if session is None:
            return False
        js = ("() => [outerWidth - innerWidth, outerHeight - innerHeight, screen.availLeft, screen.availTop,"
              " screen.availWidth, screen.availHeight, innerWidth, innerHeight]")
        try:
            dw, dh, left, top, aw, ah, _, _ = page_js(page, js)
            win = session.send("Browser.getWindowForTarget")
            if win.get("bounds", {}).get("windowState", "normal") != "normal":
                # A maximized (or minimized) window cannot be given a size until it is a normal one again.
                session.send("Browser.setWindowBounds", {"windowId": win["windowId"], "bounds": {"windowState": "normal"}})
            w, h, ok = self._target_size(aw - dw, ah - dh)
            if not ok and not self._fit_logged:
                self._fit_logged = True
                log_only(f"The screen leaves a page area of at most {aw - dw}x{ah - dh}, too small for "
                         f"{MIN_IMAGE_TILES} whole tiles; the model server resamples the screenshots.")
            session.send("Browser.setWindowBounds", {"windowId": win["windowId"], "bounds": {
                "left": left, "top": top, "width": w + dw, "height": h + dh}})
            if self.s.get("snap_viewport") and ok:
                # Window borders and bars can differ from what was measured before the resize: one
                # correction makes the page area exact, since whole tiles are the point.
                iw, ih = page_js(page, "() => [innerWidth, innerHeight]")
                if (iw, ih) != (w, h):
                    session.send("Browser.setWindowBounds", {"windowId": win["windowId"], "bounds": {
                        "width": w + dw + (w - iw), "height": h + dh + (h - ih)}})
            return True
        except Exception:
            return False

    def refit_window(self):
        """Before each task (refit_window_each_task): put the agent's window back to its size. A window
        the user maximized makes every screenshot bigger, and every step slower. Only for windows of
        an installed browser; the bundled browser's page area is fixed anyway."""
        if self.headless or not self.windowed or not self.s.get("refit_window_each_task", True):
            return False
        try:
            return self._fit_window(self._live_page())
        except Exception:
            return False

    def _session(self, page):
        """The tab's own CDP session (made once): browser-side calls such as its title or its window
        need no page JavaScript, so they also answer while the page is busy."""
        session = self.cdp.get(page)
        if session is None and self.ctx is not None:
            try:
                session = self.ctx.new_cdp_session(page)
                self.cdp[page] = session
            except Exception:
                return None
        return session

    def _target_info(self, page):
        session = self._session(page)
        if session is None:
            return {}
        try:
            return session.send("Target.getTargetInfo").get("targetInfo") or {}
        except Exception:
            return {}

    def _target_id(self, page):
        return self._target_info(page).get("targetId")

    def _tab_title(self, page):
        """A tab's title, read on the browser side (CDP), so a busy page cannot hold the step. A page
        without a title gives "" (the browser itself would show its address)."""
        info_ = self._target_info(page)
        if info_:
            title, url = str(info_.get("title") or ""), str(info_.get("url") or page.url)
            return "" if title_is_address(title, url) else title
        try:
            return str(page_js(page, "() => document.title", seconds=1.0) or "")
        except Exception:
            return ""

    def _adopt(self, page, current=True):
        """Take a tab into the agent's tabs in the user's browser (its window, a pop-up of one of its
        tabs, a tab it opened); current=False lists it without switching to it."""
        if page in self.own:
            return
        self.own.append(page)
        page.on("popup", self._on_popup)
        self._watch_tab(page)
        try:
            # Playwright's focus emulation is off in the user's browser (no_defaults); turn it on for
            # the agent's tabs only, so pages behave as focused while the console has the focus.
            session = self.ctx.new_cdp_session(page)
            session.send("Emulation.setFocusEmulationEnabled", {"enabled": True})
            self.cdp[page] = session
        except Exception:
            pass
        if current:
            self.page = page

    def _on_popup(self, page):
        """A tab that one of the agent's tabs opened in the user's browser."""
        self._adopt(page, current=False)
        self._new_tab(page)

    def _gesture(self):
        """The agent just clicked, typed or pressed a key: a tab that opens in the next 2 s is its doing."""
        self._last_gesture = time.monotonic()

    def _new_tab(self, page):
        """A new tab: the current one if the agent's own click or key press opened it (or between
        tasks, where the user is the one using the window); otherwise listed as opened by the page."""
        try:
            opener = page.opener()
        except Exception:
            opener = None
        if opener is not None:
            self._openers[page] = opener
        if self._opening_tab:
            return  # the agent opens it itself and makes it current
        if not self._in_task or time.monotonic() - self._last_gesture <= 2.0:
            self.page = page
            self.events.append("A new tab opened and is now the current tab.")
        else:
            self._by_page.add(page)
            self.events.append("The page opened a new tab by itself; it was not switched to (it is in the tab list, "
                               "marked 'opened by the page').")

    def _watch_tab(self, page):
        """What every one of the agent's tabs reports, in every browser mode."""
        page.on("close", lambda: self._on_tab_closed(page))
        if self._in_task:
            self._catch_file_chooser(page, True)
        if self.s.get("settle_mode") == "adaptive":
            self._watch_network(page)

    def _on_tab_closed(self, page):
        """A tab closed. When it was the current one and the page closed it itself (a sign-in pop-up
        that is done), the tab that opened it becomes current again."""
        if page is not self.page:
            return
        opener = self._openers.get(page)
        if opener is not None and not opener.is_closed():
            self.page = opener
            if not self._closing_tab:
                self.events.append("The current tab closed itself; the tab that opened it is the current tab again.")
        elif not self._closing_tab:
            self.events.append("The current tab closed itself.")

    def _on_file_chooser(self, chooser):
        # Catching it keeps the system's file dialog from opening (it would block the window, and the
        # model cannot see or use it); not choosing a file cancels it.
        self.events.append("A file picker opened; uploading local files is not supported. Ask the user to choose "
                           "the file.")

    def _catch_file_chooser(self, page, on):
        """File pickers in the agent's tabs are caught only while a task runs: between tasks the user
        may use the window to upload something themselves (with a listener, Playwright would swallow
        their file dialog too)."""
        try:
            if on and page not in self._choosers:
                page.on("filechooser", self._on_file_chooser)
                self._choosers.add(page)
            elif not on and page in self._choosers:
                self._choosers.discard(page)
                page.remove_listener("filechooser", self._on_file_chooser)
        except Exception:
            pass

    def _minimized(self, page):
        session = self.cdp.get(page)
        if session is None:
            return False
        try:
            return session.send("Browser.getWindowForTarget")["bounds"].get("windowState") == "minimized"
        except Exception:
            return False

    def _launch(self, opts, channel=None):
        """launch_persistent_context with Chrome's sandbox on, so a page that breaks the browser is
        still shut in. Playwright turns the sandbox off unless asked. When the sandboxed start fails,
        it is retried without the sandbox only if allow_unsandboxed_browser is true; otherwise the
        error goes to the caller."""
        kw = dict(opts, chromium_sandbox=True)
        if channel:
            kw["channel"] = channel
        try:
            return self.pw.chromium.launch_persistent_context(**kw)
        except Exception as e:
            if "already in use" in str(e) or "existing browser session" in str(e):
                raise SystemExit("The browser profile is in use. Close any other GhostAgent window and try again.")
            if not self.s.get("allow_unsandboxed_browser"):
                raise
            warn(f"The browser did not start with its security sandbox ({first_line(e)}). allow_unsandboxed_browser "
                 "is true, so it starts without it: a web page that breaks the browser could then reach this PC.")
            kw["chromium_sandbox"] = False
            return self.pw.chromium.launch_persistent_context(**kw)

    def _launch_own(self, url=None):
        features = playwright_feature_switches()
        if self.headless:
            # Tests and CI: no window, the GPU left alone, and a profile that is deleted on close.
            # Otherwise like the real thing (sandbox and pop-up blocker on), except that the
            # browser's background update requests stay off, so tests never reach the internet.
            self._temp_profile = Path(tempfile.mkdtemp(prefix="ghost-agent-headless-"))
            self.ctx = self._launch(dict(
                user_data_dir=str(self._temp_profile / "profile"), channel="chromium", headless=True,
                viewport={"width": self.w, "height": self.h}, device_scale_factor=1, accept_downloads=True,
                handle_sigint=False, handle_sigterm=False, handle_sighup=False,
                ignore_default_args=HEADLESS_UNSAFE_DEFAULT_SWITCHES,
                args=[f"--disk-cache-dir={self._temp_profile / 'cache'}", "--disable-gpu", "--no-first-run",
                      "--no-default-browser-check", "--mute-audio"] + features))
        else:
            paths.PROFILE.mkdir(parents=True, exist_ok=True)
            # The cache sits in this PC's TEMP folder (fast, and off the stick), one folder per
            # profile, and is removed on close. What a window closed with its X button left goes
            # now, and so do caches no other running agent holds.
            self._cache_dir = browser_cache_dir()
            self._cache_lock = FolderLock.take(self._cache_dir)
            if self._cache_lock is None:
                # Another agent runs with this same profile (external_server allows that) and holds
                # that cache. Its browser has the profile, so this one may end up in Edge with a
                # profile of its own: it gets a cache of its own too, so the two never share one.
                self._cache_dir = browser_cache_dir(f"{paths.PROFILE}|{os.getpid()}")
                self._cache_lock = FolderLock.take(self._cache_dir)
            if self._cache_lock is not None:
                empty_folder(self._cache_dir, keep={CACHE_LOCK_FILE})
            prune_browser_caches(keep=self._cache_dir)
            opts = dict(
                user_data_dir=str(paths.PROFILE), headless=False,
                viewport={"width": self.w, "height": self.h}, device_scale_factor=1,
                accept_downloads=True,
                handle_sigint=False, handle_sigterm=False, handle_sighup=False,
                ignore_default_args=UNSAFE_DEFAULT_SWITCHES,
                args=[f"--disk-cache-dir={self._cache_dir}", "--window-position=0,0", "--no-default-browser-check"]
                     + features,
            )
            if self.mode != "edge":
                try:
                    self.ctx = self._launch(opts)
                except SystemExit:
                    raise
                except Exception as e:
                    warn(f"The browser on this drive did not start ({first_line(e)}); trying Microsoft Edge instead. "
                         "If the reason is its security sandbox (some USB drives), allow_unsandboxed_browser = true "
                         "runs it without the sandbox.")
            if self.ctx is None:
                opts["user_data_dir"] = str(paths.PROFILE) + "-edge"
                try:
                    self.ctx = self._launch(opts, channel="msedge")
                except SystemExit:
                    raise
                except Exception as e:
                    raise SystemExit(f"Microsoft Edge did not start either ({first_line(e)}).") from None
        self._use_context(url)

    def _use_context(self, url=None):
        """The agent's own browser has started (self.ctx): every tab in it is the agent's."""
        self.ctx.on("page", self._on_new_page)
        for p in self.ctx.pages:
            self._wire(p)
        self.page = self.ctx.pages[0] if self.ctx.pages else self._open_tab()
        try:
            self.page.goto(url or self._start_page(self.s["start_url"]), wait_until="domcontentloaded", timeout=20000)
        except Exception:
            pass

    def chrome_agent_options(self, headless=False):
        """launch_persistent_context options for browser "chrome-agent": the installed Google Chrome with a
        profile of its own on this PC (%LOCALAPPDATA%\\GhostAgent\\chrome-profile). Chrome lets a program
        control a profile that is not the user's everyday one without the Allow question, and the user's
        own profile and windows are never touched. Sign-ins made there stay on this PC. Extensions work
        in it (a password manager, for example), so Playwright's switch that turns them off is dropped.
        headless (tests only): no window, and the background update requests stay off as in the
        headless test browser."""
        profile = pc_data_dir() / "chrome-profile"
        profile.mkdir(parents=True, exist_ok=True)
        unsafe = HEADLESS_UNSAFE_DEFAULT_SWITCHES if headless else UNSAFE_DEFAULT_SWITCHES
        return dict(user_data_dir=str(profile), channel="chrome", headless=headless, no_viewport=True,
                    accept_downloads=True, handle_sigint=False, handle_sigterm=False, handle_sighup=False,
                    ignore_default_args=unsafe + ["--disable-extensions"],
                    args=["--window-position=0,0", f"--window-size={self.w + 16},{self.h + 140}",
                          "--no-default-browser-check", "--no-first-run"] + playwright_feature_switches())

    def _launch_chrome_agent(self, url=None):
        if USER_BROWSERS["chrome"].find() is None:
            warn("Google Chrome is not installed on this computer (browser chrome-agent). Using the browser on this "
                 "drive instead.")
            self.mode = "bundled"
            self._launch_own(url)
            return
        try:
            self.ctx = self._launch(self.chrome_agent_options())
        except SystemExit:
            raise
        except Exception as e:
            warn(f"Chrome did not start with the agent's profile ({first_line(e)}). Using the browser on this drive "
                 "instead.")
            self.mode = "bundled"
            self._launch_own(url)
            return
        self._use_context(url)

    def reset_for_task(self, url):
        """Unattended runs: every task starts fresh, in a single tab at its start page."""
        pages = self._tabs()
        page = pages[0] if pages else self._open_tab()
        self.page = page
        self._closing_tab = True
        try:
            for p in pages[1:]:
                try:
                    p.close()
                except Exception:
                    pass
        finally:
            self._closing_tab = False
        self.events = []
        self.stall.reset()
        if url:
            with self._navigating():
                page.goto(self._start_page(url), wait_until="domcontentloaded", timeout=45000)

    @contextlib.contextmanager
    def _navigating(self):
        """The agent itself is leaving the page (go_to_web, back, refresh, closing a tab): a page's
        leave-page warning is accepted during this, and only during this."""
        self._agent_navigating = True
        try:
            yield
        finally:
            self._agent_navigating = False

    def begin_task(self, task):
        """A task starts: remember its text (the checks on risky clicks and on local addresses read
        it) and allow the hosts of the pages open now, which the user chose."""
        self.task_text = str(task or "")
        self._approved = set()
        self._task_hosts = set()
        # What happened before the task (the user's own clicks between tasks) is not news for the
        # model: the first observation shows how things are now.
        self.events = []
        self.stall.reset()
        self._alert = None
        self._size_warned = False
        self._last_gesture = float("-inf")
        self._in_task = True
        try:
            for p in self._tabs():
                self._catch_file_chooser(p, True)
                host = urllib.parse.urlsplit(p.url).hostname
                if host:
                    self._task_hosts.add(normalise_host(host))
        except Exception:
            pass

    def end_task(self):
        """The task is over: the window is the user's again (their own file pickers work)."""
        self._in_task = False
        for p in list(self._choosers):
            self._catch_file_chooser(p, False)

    def _open_tab(self):
        """A new tab the agent opens itself in its own browser: current at once, and not reported as
        a tab that opened."""
        self._opening_tab = True
        try:
            page = self.ctx.new_page()
        finally:
            self._opening_tab = False
        self.page = page
        return page

    def _wire(self, page):
        """Every tab of the agent's own browser: dialogs, downloads, and what _watch_tab adds."""
        page.on("dialog", self._on_dialog)
        page.on("download", self._on_download)
        self._watch_tab(page)

    def _on_new_page(self, page):
        self._wire(page)
        self._new_tab(page)

    def _on_dialog(self, dialog):
        """A page's alert, confirm, prompt or leave-page warning in one of the agent's tabs. A native
        confirm() or prompt() is usually a site's last check before something that cannot be undone
        ("Delete all items?"), so it is never simply accepted: dialog_policy decides, and "ask" puts
        it to the user in the console. While the question waits, the browser waits too."""
        if dialog.page not in self._tabs():
            return  # one of the user's own Chrome tabs: theirs to answer
        kind = dialog.type
        message = (dialog.message or "")[:300]
        answer = None  # the text for a prompt()
        if kind == "alert":
            ok, outcome = True, "accepted"
        elif kind == "beforeunload":
            ok = self._agent_navigating
            if not ok:
                self.events.append("A leave-page warning appeared; the navigation was cancelled.")
            else:
                self.events.append("A leave-page warning appeared and was accepted, since you chose to leave the page "
                                   "(anything unsaved on it is lost).")
            self._answer_dialog(dialog, ok)
            return
        else:
            policy = "dismiss" if runstate.HEADLESS else str(self.s.get("dialog_policy") or "ask").lower()
            if policy == "accept":
                ok, outcome = True, "accepted"
                answer = dialog.default_value if kind == "prompt" else None
            elif policy == "ask" and self.can_ask:
                ok, how = ask_yes_no(f'The page asks: "{clean(message)}". Allow?')
                if ok and kind == "prompt":
                    answer, how = ask_console("Text to send (edit it, then press Enter; Esc cancels):",
                                              prefill=dialog.default_value or "", private=True)
                    ok = answer is not None
                outcome = ("accepted by the user" if ok else
                           "dismissed: nobody answered in time" if how == "timeout" else
                           "dismissed because the task is being stopped" if how == "stopped" else
                           "dismissed: there was nobody at the console to ask" if how == "unattended" else
                           "CANCELLED by the user")
            else:
                # Nobody to ask (an unattended run or no console), or the user chose "dismiss".
                ok, outcome = False, "dismissed"
        self.events.append(f"A {kind} dialog said {untrusted(message)} and was {outcome}.")
        self._answer_dialog(dialog, ok, answer)

    @staticmethod
    def _answer_dialog(dialog, ok, text=None):
        try:
            if not ok:
                dialog.dismiss()
            elif text is not None:
                dialog.accept(text)
            else:
                dialog.accept()
        except Exception:
            pass

    def _download_limit(self):
        return max(1, int(self.s.get("download_max_mb") or 500)) * 2**20

    def _on_download(self, download):
        """A download in the agent's own browser (downloads "save"): kept in the PC's Downloads\\GhostAgent
        folder under a name that replaces nothing, marked as coming from the internet, and never
        opened. One larger than download_max_mb is stopped. Playwright runs this handler on its own,
        so waiting here for the download to finish does not hold up the step."""
        name = safe_download_name(download.suggested_filename)
        if str(self.s.get("downloads") or "save").lower() == "off":
            try:
                download.cancel()
            except Exception:
                pass
            self.events.append(f"A download ({name}) was blocked: downloads are turned off in the settings.")
            return
        limit = self._download_limit()
        try:
            partial = Path(download._impl_obj._artifact.absolute_path + ".crdownload")
        except Exception:
            partial = None  # the size is then checked once the download is complete
        self._downloads[download] = {"name": name, "partial": partial, "told": False}
        try:
            path = download.path()  # waits until the download is complete (or stopped)
        except Exception:
            path = None
        if self._downloads.pop(download, None) is None:
            return  # stopped for its size meanwhile (already reported)
        failure = None
        try:
            failure = download.failure()
        except Exception:
            pass
        if path is None or failure:
            self.events.append(f"The download of {name} failed ({one_line(str(failure or 'no file'), 120)}).")
            return
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        if size > limit:
            self._too_big(download, name)
            return
        try:
            folder = download_folder()
            folder.mkdir(parents=True, exist_ok=True)
            target = unique_path(folder, name)
            download.save_as(str(target))
        except Exception as e:
            self.events.append(f"The download of {name} could not be saved ({first_line(e)}).")
            return
        if not mark_of_the_web(target, download.url):
            warn(f"{hide_home(str(target))} could not be marked as downloaded from the internet (the drive may not "
                 "be NTFS), so Windows will not warn before it is opened.")
        self.events.append(f"Downloaded {target.name} ({size_text(size)}) to {hide_home(str(folder))}. GhostAgent never "
                           "opens downloaded files.")

    def _too_big(self, download, name):
        try:
            download.cancel()
        except Exception:
            pass
        try:
            download.delete()
        except Exception:
            pass
        self.events.append(f"The download of {name} was stopped: it is larger than the download_max_mb limit "
                           f"({self._download_limit() // 2**20} MB).")

    def _check_downloads(self):
        """Between actions: stop a running download that has grown past download_max_mb, and say once
        which ones are still running."""
        limit = self._download_limit()
        for download, d in list(self._downloads.items()):
            try:
                size = d["partial"].stat().st_size if d["partial"] else 0
            except OSError:
                size = 0
            if size > limit:
                if self._downloads.pop(download, None) is not None:
                    self._too_big(download, d["name"])
            elif not d["told"]:
                d["told"] = True
                self.events.append(f"{d['name']} is still downloading ({size_text(size)} so far); you will be told when "
                                   "it is done.")

    def playwright_dead(self):
        """True after a KeyboardInterrupt landed inside a Playwright call (a double Ctrl+C): its
        dispatcher is gone, and any further Playwright call would wait forever."""
        fiber = getattr(getattr(getattr(self.pw, "_impl_obj", None), "_connection", None), "_dispatcher_fiber", None)
        return bool(getattr(fiber, "dead", False))

    def close(self):
        if not self.playwright_dead():
            try:
                if self.browser:
                    self.browser.close()  # only disconnects: the user's Chrome and the agent's window stay open
                elif self.ctx:
                    self.ctx.close()
            except Exception:
                pass
            try:
                if self.pw:
                    self.pw.stop()
            except Exception:
                pass
        if self._temp_profile is not None:
            shutil.rmtree(self._temp_profile, ignore_errors=True)
            self._temp_profile = None
        if self._cache_dir is not None:
            # Copies of everything the browser showed: they should not stay on a PC the stick visited.
            # Only this agent's own (whose lock it holds).
            if self._cache_lock is not None:
                remove_browser_cache(self._cache_dir, self._cache_lock)
                self._cache_lock = None
            self._cache_dir = None

    # -- observation ------------------------------------------------------------------------

    def _connected(self, check=False):
        """The connection to the user's browser is up (always True for the agent's own browser).
        Playwright learns that the browser went away only while it handles messages, which it does
        during a call: check=True makes one small round trip first, so a browser closed while the
        agent was idle is noticed now rather than by the next action failing."""
        if not self.attached:
            return True
        if self.browser is None or not self.browser.is_connected():
            return False
        if check:
            try:
                session = self.browser.new_browser_cdp_session()
                session.send("Browser.getVersion")
                session.detach()
            except Exception:
                pass
        return self.browser.is_connected()

    def _tabs(self):
        """The agent's tabs: in the user's browser only the ones it opened, otherwise all of them."""
        if self.attached:
            self.own = [p for p in self.own if not p.is_closed()]
            tabs = list(self.own)
        else:
            tabs = [p for p in self.ctx.pages if not p.is_closed()]
        # Forget what was kept about tabs that are gone.
        live = set(tabs)
        if any(p not in live for p in self.cdp):
            self.cdp = {p: s for p, s in self.cdp.items() if p in live}
        if any(p not in live for p in self._net):
            self._net = {p: n for p, n in self._net.items() if p in live}
        if any(p not in live and p is not self.page for p in self._openers):
            # (the current tab's opener is kept until _live_page has gone back to it)
            self._openers = {p: o for p, o in self._openers.items() if p in live or p is self.page}
        self._by_page &= live
        self._choosers &= live
        return tabs

    def _live_page(self):
        if not self._connected(check=True):
            # Not reconnected here: that may need the user's Allow click, which must not happen in the
            # middle of a task. ensure_connected() reconnects before the next one.
            raise BrowserGone(f"{self.user_browser_name} was closed; start the next task to reconnect.")
        pages = self._tabs()
        if not pages:
            if self.attached:
                self._new_window(self._start_page(self.s["start_url"]))  # the user closed the agent's window
            else:
                self._open_tab()  # the agent's own doing: not reported as a tab the page opened
            pages = self._tabs()
        if self.page not in pages:
            # The current tab closed: back to the tab that opened it (a sign-in pop-up that is done),
            # else the last one.
            opener = self._openers.get(self.page)
            self.page = opener if opener in pages else pages[-1]
            self._by_page.discard(self.page)
        return self.page

    # -- what the agent may open ------------------------------------------------------------

    def _is_local(self, host):
        h = normalise_host(host)
        if h not in self._local:
            self._local[h] = host_is_local(h)
        return self._local[h]

    def _host_permitted(self, host):
        """Any public host; a host on this PC or the local network only when the task names it, the
        allowed_hosts setting lists it, or the user had it open (a start page, or a tab that was open
        when the task began). That keeps a page's instructions from sending the agent to a router,
        an intranet site or a local service."""
        h = normalise_host(host)
        if not h or h in self._start_hosts or h in self._task_hosts or self._task_names(h):
            return True
        if host_listed(h, self.s.get("allowed_hosts") or []):
            return True
        return not self._is_local(h)

    def _task_names(self, host):
        """True when the task's text names this host as a whole word: '192.168.1.1' does not count
        as named by '192.168.1.10', nor 'intra' by 'intranet'."""
        return bool(re.search(r"(?<![\w.-])" + re.escape(host) + r"(?![\w-]|\.\w)", self.task_text, re.I))

    def _vet(self, raw):
        """vet_url plus the local-network check: the address to open, or ToolRefused."""
        url = vet_url(raw, self.s.get("allow_file_urls"), self.s.get("allow_internal_pages"))
        if url_scheme(url) not in ("http", "https"):
            return url  # about:blank, or a file: or browser page the settings allow
        try:
            host = urllib.parse.urlsplit(url).hostname
        except ValueError:
            raise ToolRefused("That is not a valid web address.") from None
        if host and not self._host_permitted(host):
            raise ToolRefused(f"Opening {host} is blocked: it is on this computer or the local network, and the "
                              "task does not name it.")
        return url

    def _blocked_reason(self, url):
        """Why the agent may not stay on a page with this address, or None when it may."""
        scheme = url_scheme(url)
        if scheme in ("http", "https"):
            try:
                host = urllib.parse.urlsplit(url).hostname
            except ValueError:
                host = None
            if host and not self._host_permitted(host):
                return (f"Opening {host} is blocked: it is on this computer or the local network, and the task "
                        "does not name it.")
            return None
        if scheme in ("about",) or scheme in ERROR_SCHEMES:
            return None
        if scheme == "blob" and url_scheme(url[5:]) in ("http", "https"):
            return self._blocked_reason(url[5:])  # made by a web page, with that page's origin
        if (scheme == "file" and self.s.get("allow_file_urls")) or (scheme in INTERNAL_SCHEMES and self.s.get("allow_internal_pages")):
            return None
        return f"Navigation to {scheme or 'that kind of'}: pages is blocked."

    def _check_tabs(self):
        """Move any of the agent's tabs off a page it may not use (a redirect, a click, a pop-up, or
        the user leaving the agent's window on such a page) to about:blank. Returns what happened, as
        notes for the model."""
        notes = []
        try:
            tabs = self._tabs()
        except Exception:
            return notes
        for p in tabs:
            try:
                why = self._blocked_reason(p.url)
            except Exception:
                continue
            if not why:
                continue
            info("The agent's tab was on a page it may not use; it now shows a blank page. (" + why + ")")
            with self._navigating():
                try:
                    p.goto("about:blank", timeout=10000)
                except Exception:
                    pass
            notes.append(why)
        return notes

    def _focus_chain(self, page):
        """The frames that hold the keyboard focus: the main frame, then the frame whose element has
        the focus there, and so on down. Key presses go to the last one. Only these frames are asked
        about the focus: asking every frame would also wait on frames that have no document yet (a
        lazy or still-loading iframe, an advert) and call the whole page busy. PageBusy when one of
        these frames does not answer: then keys would wait for it."""
        frames = [page.main_frame]
        frame = page.main_frame
        for _ in range(5):
            try:
                el = page_js(frame, FOCUSED_FRAME_JS, handle=True).as_element()
                child = el.content_frame() if el is not None else None
            except PageBusy:
                raise
            except Exception:
                break
            if child is None:
                break
            frames.append(child)
            frame = child
        return frames

    def _focused_secret(self, page):
        """True when the field with the keyboard focus (in any frame) is a password or code field.
        PageBusy when the page does not answer: then nothing may be typed."""
        for frame in self._focus_chain(page):
            try:
                if page_js(frame, FOCUSED_SECRET_JS, True):
                    return True
            except PageBusy:
                raise
            except Exception:
                pass
        return False

    def _focused_field(self, page):
        """(what FOCUSED_FIELD_JS says about the focused element, its frame), looking into the frame
        that has the focus. ({'kind': 'none'}, main frame) when nothing could be read."""
        chain = self._focus_chain(page)
        frame = chain[-1]
        try:
            r = page_js(frame, FOCUSED_FIELD_JS, True)
        except PageBusy:
            raise
        except Exception:
            r = None
        if len(chain) > 1 and (not r or r.get("kind") == "none"):
            # The focus is inside a frame that does not say what it is (its page may handle keys
            # itself): typed, but reported as not checked.
            return {"kind": "frame"}, frame
        return (r, frame) if r else ({"kind": "none"}, page.main_frame)

    def _confirm_risky(self, page, x, y, mode):
        """Ask the user before a click (mode 'click') or an Enter in a form (mode 'submit') whose
        button says Delete, Send, Pay and the like, unless the task asks for exactly that. Raises
        ToolRefused when the user says no (or nobody can be asked), PageBusy when the page does not
        answer (then the button cannot be read, so nothing is done)."""
        if not self.s.get("confirm_risky_actions", True):
            return
        try:
            frame, fx, fy = self._frame_at(page, x, y)
            label = page_js(frame, LABEL_JS, [fx, fy, mode])
        except PageBusy:
            raise
        except Exception:
            return
        self._confirm_risky_label(page, label, mode)

    def _confirm_risky_key(self, page, submits=True):
        """_confirm_risky for Enter or Space pressed with press_keys_web: what the focused element would
        set off (a button or link, or, for a key that submits forms (Enter), the submit button of the
        form around a field)."""
        if not self.s.get("confirm_risky_actions", True):
            return
        for frame in reversed(self._focus_chain(page)):
            try:
                found = page_js(frame, FOCUSED_LABEL_JS, True)
            except PageBusy:
                raise
            except Exception:
                continue
            if found:
                if found[0] == "click" or submits:
                    self._confirm_risky_label(page, found[1], found[0])
                return

    def _confirm_risky_label(self, page, label, mode):
        words = risky_words(label, self.task_text)
        if not words:
            return
        try:
            host = urllib.parse.urlsplit(page.url).hostname or page.url
        except ValueError:
            host = page.url
        key = (one_line(label, 150).lower(), host)
        if key in self._approved:
            return
        action = "click" if mode == "click" else "pressing Enter to submit"
        if self.can_ask:
            ok, how = ask_yes_no(f'Allow {action} "{clean(one_line(label, 100))}" on {clean(host)}?')
        else:
            ok, how = False, "unattended"
        if ok:
            self._approved.add(key)
            return
        if how == "unattended":
            raise ToolRefused(RISKY_REFUSAL + " (Nobody could be asked in this run; the task does not ask for '"
                              + "', '".join(words) + "'.)")
        raise ToolRefused(RISKY_REFUSAL)

    def overlay(self, kind, title="", text=""):
        """The status or answer box in the current page (OVERLAY_JS), or "remove". Skipped on a page
        that is busy: it would only make the step wait."""
        if self.playwright_dead() or not self._connected():
            return
        if self._busy and kind != "remove":
            return
        try:
            page_js(self._live_page(), OVERLAY_JS, [kind, title, text])
        except Exception:
            pass

    def _remove_overlay(self, page):
        """Take the status box off the page. False when the page did not answer in time (it is busy:
        a click or a key press would wait until its script ends)."""
        try:
            page_js(page, OVERLAY_JS, ["remove", "", ""])
            return True
        except PageBusy:
            return False
        except Exception:
            return True  # navigating or closing: not a busy page

    def _screenshot(self, page):
        """(png bytes, None) or (None, why there is none)."""
        if self._busy:
            # The renderer is stuck in a script; a capture may still come once it lets go.
            try:
                return page.screenshot(type="png", scale="css", timeout=5000), None
            except Exception:
                return None, "the page is busy (its scripts are not responding)"
        try:
            return page.screenshot(type="png", scale="css", timeout=10000), None
        except Exception:
            pass
        # Usually a native popup (dropdown, date picker) is open and blocks the capture, or the tab is
        # not the one showing in its window.
        try:
            page.keyboard.press("Escape")
            try:
                page.bring_to_front()
            except Exception:
                pass
            page.wait_for_timeout(500)
            return page.screenshot(type="png", scale="css", timeout=10000), None
        except Exception as e:
            return None, str(e).splitlines()[0] if str(e) else type(e).__name__

    def _tab_list(self, page):
        """[(index, title, url, current, opened_by_page)] for the agent's tabs."""
        out = []
        for i, p in enumerate(self._tabs()):
            out.append((i, self._tab_title(p), p.url, p is page, p in self._by_page))
        return out

    def observation(self, bring_to_front=False):
        page = self._live_page()
        # A page may have moved on by itself while the model was thinking (a timer, a redirect).
        notes = self._check_tabs()
        page = self._live_page()
        if bring_to_front or self._minimized(page):
            try:
                page.bring_to_front()  # also restores a minimized window, which cannot be captured
            except Exception:
                pass
        self._check_downloads()
        self._busy = not self._remove_overlay(page)
        self.last_shot = {"png_bytes": None, "png_w": None, "png_h": None}
        png, problem = self._screenshot(page)
        if png:
            self.w, self.h = struct.unpack(">II", png[16:24])  # PNG header: width, height
            self.last_shot = {"png_bytes": len(png), "png_w": self.w, "png_h": self.h}
            if self.s["save_screenshots"]:
                # On this PC, not the stick: they show signed-in pages. Kept 7 days (prune_screenshots).
                shots = screenshots_dir()
                shots.mkdir(parents=True, exist_ok=True)
                (shots / f"{datetime.now():%Y%m%d-%H%M%S-%f}.png").write_bytes(png)
            if self.w * self.h > BIG_SCREENSHOT_PIXELS and not self._size_warned:
                self._size_warned = True
                if self.windowed:
                    warn(f"The agent's browser window is large ({self.w}x{self.h}): a maximized window makes every "
                         "step slower. Make it smaller, or let GhostAgent size it again at the next task.")
                else:
                    warn(f"The agent's page area is large ({self.w}x{self.h}): a big screenshot makes every step "
                         "slower. viewport_width and viewport_height in settings.json set its size.")
            image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}
        else:
            image = {"type": "text", "text": f"[screenshot failed: {problem}]"}
        if self._busy:
            notes.append("The page is busy (its scripts are not responding).")
        # What happened since the last tool result (a tab opened by the page, a dialog, a download).
        notes += self.events
        self.events = []
        tabs = self._tab_list(page)
        self.last_url = page.url
        if str(self.s.get("observation_format") or "classic") == "header":
            metrics = None
            if not self._busy:
                try:
                    metrics = page_js(page, METRICS_JS)
                except Exception:
                    metrics = None
            title = next((t for _, t, _, current, _ in tabs if current), "")
            head = observation_header(page.url, title, (self.w, self.h) if png else None, metrics,
                                      [(i, t, current, by_page) for i, t, _, current, by_page in tabs], notes)
            parts = [{"type": "text", "text": head}, image, {"type": "text", "text": observation_footer(metrics)}]
        else:
            lines = []
            for i, title, url, current, by_page in tabs:
                mark = " (current)" if current else " (opened by the page)" if by_page else ""
                lines.append(f"  [{i}] {untrusted(page_title(title, 80))} - {url[:200]}{mark}")
            note = ("Note: " + " ".join(notes) + "\n") if notes else ""
            parts = [{"type": "text", "text": "<observation>\n"}, image,
                     {"type": "text", "text": f"\nURL: {page.url}\n{note}Open tabs:\n" + "\n".join(lines) + "\n</observation>"}]
        obs = {"role": "user", "content": parts}
        if self._alert:
            with_alert(obs, self._alert)
            self._alert = None
        return obs

    # -- actions ----------------------------------------------------------------------------

    # Tools that put the mouse or the keyboard on the page: on a busy page they would wait for its
    # script to end, so they are not tried there.
    INPUT_TOOLS = ("click_web", "write", "select_option", "move_mouse_web", "press_keys_web", "scroll_web",
                   "drag_and_drop_web")
    # Tools whose effect on the page is checked (the page's state before and after). ctrl_f_web is
    # not: selecting a match changes nothing the check sees, though it is a result. Nor is
    # close_tab_web: closing another tab leaves the current page as it was, and "did not change"
    # would invite closing again, which would close a different tab.
    CHECKED_TOOLS = INPUT_TOOLS + ("go_to_web", "go_back_web", "refresh_web", "switch_tab_web", "wait_web")
    NO_MOVE = "The page did not move (already at the end, or the content is in a scrollable panel: scroll inside it with x,y)."
    # The same for tool_schema "h", whose scroll_web names the panel with scroll_element.
    NO_MOVE_H = ("The page did not move (already at the end, or the content is in a scrollable panel: scroll inside "
                 "it with scroll_element).")

    @property
    def _no_move(self):
        return self.NO_MOVE_H if str(self.s.get("tool_schema") or "classic") == "h" else self.NO_MOVE

    @staticmethod
    def _frame_at(page, x, y):
        """Follow a viewport point down into iframes. Returns (frame, x, y) in that frame's coordinates."""
        frame = page.main_frame
        for _ in range(5):
            el = page_js(frame, "([x, y]) => { const e = document.elementFromPoint(x, y); "
                                "return e && (e.tagName === 'IFRAME' || e.tagName === 'FRAME') ? e : null; }",
                         [x, y], handle=True).as_element()
            if el is None:
                break
            child = el.content_frame()
            if child is None or not child.url:
                # No frame, or one without a document yet (lazy, or still on its way): it could not
                # answer, so the point counts as on the frame element in this frame.
                break
            left, top = el.evaluate("e => { const r = e.getBoundingClientRect(); return [r.left + e.clientLeft, r.top + e.clientTop]; }")
            frame, x, y = child, x - left, y - top
        return frame, x, y

    def _xy(self, args):
        """The model's 0-1000 point in page pixels, inside the page area (1000 is its last pixel)."""
        x = max(0, min(1000, int(round(float(args["x"])))))
        y = max(0, min(1000, int(round(float(args["y"])))))
        return min(x / 1000 * self.w, self.w - 1), min(y / 1000 * self.h, self.h - 1)

    def _signature(self, page):
        """A fingerprint of the page's state (SIGNATURE_JS), or None when it cannot be read."""
        try:
            sig = page_js(page, SIGNATURE_JS)
        except Exception:
            return None
        return hashlib.sha1(json.dumps(sig, ensure_ascii=False).encode("utf-8", "replace")).hexdigest() if sig else None

    def location(self):
        """(current tab, its address), or None when the browser cannot say. The agent compares it
        before and after each call of a step: once it changes, the rest of the step's calls (aimed at
        what the model saw) are skipped."""
        try:
            page = self._live_page()
            return id(page), page.url
        except Exception:
            return None

    def execute(self, name, args):
        """Run one tool call and return the text the model gets back. The time the action took and
        the time spent waiting for the page to settle afterwards are kept in self.last_timings, and
        self.last_ok says whether the action was done (False for a refusal, a busy page, a field that
        was not there): the agent skips the rest of a step's calls after one that was not."""
        page = self._live_page()
        args = canonical_args(name, args, bool(self.s.get("write_enter_default", False)))
        self.last_timings = {"execute": None, "settle": None}
        self.last_ok = True
        self._scroll_probe = None
        adaptive = str(self.s.get("settle_mode") or "fixed") == "adaptive"
        t0 = time.perf_counter()
        before = None
        try:
            try:
                if name in self.INPUT_TOOLS or not self._busy:
                    # The status box comes off before anything happens, so the action cannot land on
                    # it. This is also the busy check: a page that does not answer would hold the mouse.
                    self._busy = not self._remove_overlay(page)
                    if self._busy and name in self.INPUT_TOOLS:
                        raise PageBusy()
                if name in self.CHECKED_TOOLS and not self._busy:
                    before = self._signature(page)
                if adaptive and name in ("click_web", "write", "select_option", "press_keys_web", "drag_and_drop_web"):
                    self._start_quiet(page)
                result, note = self._act(page, name, args)
            except ToolRefused as e:
                self.last_ok = False
                result, note = str(e), None  # a refusal is the tool's answer, not an error
            except PageBusy:
                self.last_ok = False
                self._busy = True
                result, note = f"{BUSY_RESULT} Current URL: {page.url}", None
        finally:
            self.last_timings["execute"] = round(time.perf_counter() - t0, 3)
        if result is not None:
            # Answered without changing the page: nothing to wait for.
            self._check_downloads()
            if self.events:
                result += " " + " ".join(self.events)
                self.events = []
            return result

        t1 = time.perf_counter()
        try:
            if not STOP.is_set():
                self._settle(name, args, adaptive)
        finally:
            self.last_timings["settle"] = round(time.perf_counter() - t1, 3)
        # Where the action led: a redirect, a click or a pop-up may have reached a page the agent
        # may not use (another scheme, or an address on the local network the task does not name).
        self.events += self._check_tabs()
        self._check_downloads()
        live = self._live_page()
        changed = None
        after = None
        if live is not page:
            changed = True
        elif before is not None:
            after = self._signature(live)
            if after is not None:
                changed = after != before
        if self._scroll_probe is not None:
            moved = self._scroll_moved()
            if moved is False:
                note = (note + " " if note else "") + self._no_move
            elif moved:
                changed = True  # a panel scrolled: the page's own position may be the same
                after = None    # ... so its fingerprint cannot tell old from new: count the scroll as new
        out = "Executed."
        if note:
            out += " " + note
        if changed is False and self._no_move not in out:
            out += " The page did not change."
        if name != "wait_web" or changed:
            n = self.stall.record(action_signature(name, args), after, changed, before)
            if n:
                self._alert = STALL_ALERT.format(n=n)
        if self.events:
            out += " " + " ".join(self.events)
            self.events = []
        return out + f" Current URL: {live.url}"

    # -- settling: waiting for the page after an action ---------------------------------------

    def _settle(self, name, args, adaptive):
        if not adaptive:
            # settle_mode "fixed": the same pause after every action, then the page's HTML loaded.
            self._pause(float(self.s["settle_seconds"]) * 1000)
            try:
                self._live_page().wait_for_load_state("domcontentloaded", timeout=10000)
            except Exception:
                pass
            return
        # settle_mode "adaptive": as long as this kind of action needs, ending early once the page is quiet.
        cap = float(self.s["settle_seconds"])
        if name in ("wait_web", "ctrl_f_web", "reader_mode"):
            return  # waited already, or nothing moves
        if name in ("scroll_web", "move_mouse_web"):
            self._pause(300)
        elif name in ("go_to_web", "go_back_web", "refresh_web"):
            self._wait_quiet(0.0, 1.5)  # the HTML has loaded already (wait_until="domcontentloaded")
        else:
            self._wait_quiet(0.25, max(0.25, cap))

    def _pause(self, ms):
        """Wait in the current tab (Playwright handles the browser's events meanwhile). The tab may
        close meanwhile (a sign-in pop-up that is done): that ends this pause, not the step."""
        try:
            self._live_page().wait_for_timeout(ms)
        except Exception:
            if self.playwright_dead() or not self._connected():
                raise

    def _start_quiet(self, page):
        """settle_mode adaptive: start watching the page's DOM for changes (QUIET_JS), before the action."""
        try:
            page_js(page, QUIET_JS, "start", seconds=1.0)
        except Exception:
            pass

    def _quiet_ms(self, page, seconds=1.0):
        """Milliseconds since the page's DOM last changed, or None when the page did not answer."""
        try:
            return float(page_js(page, QUIET_JS, None, seconds=seconds) or 0)
        except Exception:
            return None

    def _watch_network(self, page):
        """settle_mode adaptive: keep count of the tab's requests in flight and of its page loads
        under way (a click that starts loading another page)."""
        if page in self._net:
            return
        net = self._net[page] = {"flight": {}, "nav": set(), "last": time.monotonic()}

        def started(request):
            now = time.monotonic()
            net["flight"][request] = now
            net["last"] = now
            try:
                if request.is_navigation_request() and request.frame == page.main_frame:
                    net["nav"].add(request)
            except Exception:
                pass

        def ended(request):
            net["flight"].pop(request, None)
            net["nav"].discard(request)
            net["last"] = time.monotonic()

        try:
            page.on("request", started)
            page.on("requestfinished", ended)
            page.on("requestfailed", ended)
        except Exception:
            pass

    def _loading(self, page):
        """A page load of this tab is under way (its request has not finished)."""
        net = self._net.get(page)
        if not net:
            return False
        now = time.monotonic()
        for request in list(net["nav"]):
            if now - net["flight"].get(request, now) > 30:
                net["nav"].discard(request)  # never finished: not worth waiting for
        return bool(net["nav"])

    def _network_quiet(self, page, seconds=0.5):
        """No request of this tab has been in flight for `seconds` (a request open for more than 10 s,
        such as a live update channel, is not counted)."""
        net = self._net.get(page)
        if not net:
            return True
        now = time.monotonic()
        busy = [t for t in net["flight"].values() if now - t < 10]
        return not busy and now - net["last"] >= seconds

    def _wait_quiet(self, minimum, maximum):
        """Wait at least `minimum` and at most `maximum` seconds, ending once no page load is under way
        and the DOM has not changed for 250 ms."""
        start = time.monotonic()
        while not STOP.is_set():
            elapsed = time.monotonic() - start
            if elapsed >= maximum:
                break
            page = self._live_page()
            if not self._loading(page):
                quiet = self._quiet_ms(page, seconds=max(0.1, min(1.0, maximum - elapsed)))
                if quiet is not None and quiet >= 250 and elapsed >= minimum:
                    break
            self._pause(min(50.0, max(1.0, (maximum - (time.monotonic() - start)) * 1000)))
        # A page that started loading meanwhile: its HTML should be there for the screenshot.
        left = maximum - (time.monotonic() - start)
        if left > 0:
            try:
                self._live_page().wait_for_load_state("domcontentloaded", timeout=left * 1000)
            except Exception:
                pass

    def _scroll_state(self, page, px, py):
        """Before a scroll: remember the scroll offsets the wheel at (px, py) can change, in the page
        and in the frame under the point."""
        probes = [(page.main_frame, px, py)]
        try:
            frame, fx, fy = self._frame_at(page, px, py)
            if frame != page.main_frame:
                probes.append((frame, fx, fy))
        except PageBusy:
            raise
        except Exception:
            pass
        states = []
        for frame, x, y in probes:
            try:
                states.append(page_js(frame, SCROLL_PROBE_JS, [x, y, True]))
            except PageBusy:
                raise
            except Exception:
                states.append(None)
        self._scroll_probe = (page, [f for f, _, _ in probes], states)

    def _scroll_moved(self):
        """After a scroll: True when something scrolled, False when nothing did, None when unknown.
        Smooth scrolling takes a moment, so 'nothing' is looked at again for up to half a second."""
        page, frames, before = self._scroll_probe
        self._scroll_probe = None
        if page is not self.page or any(s is None for s in before):
            return None
        deadline = time.monotonic() + 0.5
        while True:
            now = []
            for frame in frames:
                try:
                    now.append(page_js(frame, SCROLL_PROBE_JS, [None, None, False], seconds=1.0))
                except Exception:
                    now.append(None)
            if any(s is None for s in now):
                return None
            if now != before:
                return True
            if time.monotonic() >= deadline or STOP.is_set():
                return False
            self._pause(100)

    # -- the actions themselves ---------------------------------------------------------------

    def _act(self, page, name, args):
        """The action itself. Returns (final text, None) when the page was not changed and there is
        nothing to wait for, else (None, a note for the model)."""
        note = ""
        if name == "click_web":
            px, py = self._xy(args)
            left = (args.get("button") or "left") == "left"
            # A native <select> opens an OS popup that screenshots cannot capture (and that blocks
            # them), so hand the options to the model instead of opening it. When the page cannot be
            # asked (it is navigating), it is just clicked.
            options = None
            if left:
                try:
                    frame, fx, fy = self._frame_at(page, px, py)
                    options = page_js(frame, SELECT_OPTIONS_JS, [fx, fy])
                except PageBusy:
                    raise
                except Exception:
                    options = None
            if options is not None:
                self.last_ok = False  # not clicked: what the model meant to do next did not happen
                return ("That is a dropdown menu, so it was not opened. Its options are: "
                        + " | ".join(options) + ". Use select_option with the same coordinates to choose one."
                        + f" Current URL: {page.url}"), None
            if left:
                self._confirm_risky(page, px, py, "click")
            self._gesture()
            # Held for 50 ms like a real click: some sites ignore a press and release in the same instant.
            page.mouse.click(px, py, button=args.get("button") or "left",
                             click_count=2 if args.get("double_click") else 1, delay=50)
            self._gesture()
        elif name == "write":
            return None, self._write(page, args)
        elif name == "select_option":
            px, py = self._xy(args)
            frame, fx, fy = self._frame_at(page, px, py)
            res = page_js(frame, SELECT_JS, [fx, fy, str(args.get("option", ""))]) or {}
            note = res.get("msg", "")
            if res.get("ok") is False:
                # No such dropdown or option: nothing was chosen, so the step's later calls (say, a
                # click on Submit) must not go ahead as if it had been.
                self.last_ok = False
        elif name == "move_mouse_web":
            page.mouse.move(*self._xy(args))
        elif name == "press_keys_web":
            try:
                presses = parse_keys(args.get("keys"))
            except BadKeys as e:
                raise ToolRefused(str(e)) from None
            if any(is_paste(k) for k in presses):
                raise ToolRefused(PASTE_REFUSAL)
            # Checked press by press, since an earlier press (Tab) can move the focus: typing a password
            # key by key is still typing it, and Enter or Space on a Delete button is still a click on it.
            secret = None  # whether the focused field is a password field; looked up again after a non-text key
            for combo in presses:
                if types_text(combo):
                    if secret is None:
                        secret = self._focused_secret(page)
                    if secret:
                        raise ToolRefused(PASSWORD_REFUSAL)
                else:
                    secret = None
                if activates(combo):
                    self._confirm_risky_key(page, submits=not types_text(combo))  # Space types in a field
                self._gesture()
                page.keyboard.press(combo)
            self._gesture()
        elif name == "scroll_web":
            if args.get("x") is not None and args.get("y") is not None:
                px, py = self._xy(args)
            else:
                px, py = self.w / 2, self.h / 2
            self._scroll_state(page, px, py)
            page.mouse.move(px, py)
            d = args.get("direction", "down")
            dist = (self.h if d in ("up", "down") else self.w) * 0.8
            try:
                if args.get("num_pixels") is not None:
                    dist = max(1.0, min(20000.0, abs(float(args["num_pixels"]))))
            except (TypeError, ValueError):
                pass
            dx = dist if d == "right" else -dist if d == "left" else 0
            dy = dist if d == "down" else -dist if d == "up" else 0
            page.mouse.wheel(dx, dy)
        elif name == "drag_and_drop_web":
            try:
                x1, y1 = self._xy({"x": args["x1"], "y": args["y1"]})
                x2, y2 = self._xy({"x": args["x2"], "y": args["y2"]})
            except (KeyError, TypeError, ValueError):
                raise ToolRefused("drag_and_drop_web needs x1, y1, x2 and y2 (integers from 0 to 1000); "
                                  "nothing was dragged.") from None
            # Released on a button, a drag is a click on it (a drag that ends where it started is
            # just a click): the same question as a click on Delete, Send or Pay.
            self._confirm_risky(page, x2, y2, "click")
            # Pressed, moved in a few steps (pages that track the pointer need the moves in between,
            # and the browser starts a drag only once the pointer has travelled), then released.
            self._gesture()
            page.mouse.move(x1, y1)
            page.mouse.down()
            page.mouse.move(x2, y2, steps=5)
            page.mouse.up()
            self._gesture()
        elif name == "go_to_web":
            url = self._vet(args.get("url", ""))
            with self._navigating():
                page.goto(url, wait_until="domcontentloaded", timeout=45000)
        elif name == "go_back_web":
            with self._navigating():
                page.go_back(wait_until="domcontentloaded", timeout=30000)
        elif name == "refresh_web":
            with self._navigating():
                page.reload(wait_until="domcontentloaded", timeout=45000)
        elif name == "switch_tab_web":
            if args.get("url") and args.get("tab_id") is None:
                url = self._vet(args["url"])
                if self.attached:
                    # Opened from the agent's tab, so the new tab lands in the agent's window, not
                    # in whichever of the user's windows was used last.
                    self._opening_tab = True
                    try:
                        with page.expect_popup(timeout=15000) as popup:
                            page_js(page, "(u) => { window.open(u, '_blank', 'noopener'); return 1; }", url)
                    finally:
                        self._opening_tab = False
                    new = popup.value
                    self._adopt(new)
                    self.page = new
                    new.wait_for_load_state("domcontentloaded", timeout=45000)
                else:
                    new = self._open_tab()
                    new.goto(url, wait_until="domcontentloaded", timeout=45000)
            else:
                idx = self._tab_number(args.get("tab_id", 0))
                pages = self._tabs()
                if not 0 <= idx < len(pages):
                    self.last_ok = False
                    return f"There is no tab {idx}. Open tabs: 0 to {len(pages) - 1}.", None
                self.page = pages[idx]
                self._by_page.discard(self.page)
                try:
                    self.page.bring_to_front()
                except Exception:
                    pass
        elif name == "close_tab_web":
            pages = self._tabs()
            idx = self._tab_number(args["tab_id"]) if args.get("tab_id") is not None else pages.index(page)
            if not 0 <= idx < len(pages):
                self.last_ok = False
                return f"There is no tab {idx}.", None
            closing = pages[idx]
            with self._navigating():
                if len(pages) == 1 and self.attached:
                    closing.goto("about:blank")  # closing it would close the agent's window
                else:
                    if len(pages) == 1:
                        self._open_tab()
                    self._closing_tab = True
                    try:
                        closing.close()
                    finally:
                        self._closing_tab = False
            # Closing another tab leaves the current one as it is; closing the current one goes back
            # to the tab that opened it, else to the last one (_live_page).
            self._live_page()
        elif name == "ctrl_f_web":
            note = self._find(page, str(args.get("query", "")))
        elif name == "reader_mode":
            text = page_js(page, READER_JS, seconds=READER_JS_SECONDS) or ""
            chunk, a, b, total = reader_slice(text, args.get("start"), args.get("end"))
            footer = f"[Showing chars {a}-{b} of {total} total]"
            if b < total:
                footer += f" For the next part, call reader_mode with start={b}."
            # Marked as page text: what a page says is data for the task, not instructions.
            return (f"Page: {untrusted(page_title(self._tab_title(page)))} ({page.url})\n\n"
                    f"{untrusted(chr(10) + chunk + chr(10))}\n{footer}"), None
        elif name == "wait_web":
            self._wait(page, args)
        else:
            self.last_ok = False
            return f"Unknown tool {name}.", None
        return None, note

    @staticmethod
    def _tab_number(value):
        """A tab id as the model wrote it (7, "7", "[7]") as a number, or a refusal naming the problem."""
        try:
            return int(float(str(value).strip().strip("[]")))
        except (TypeError, ValueError):
            raise ToolRefused(f"{json.dumps(value)} is not a tab id; use the number in brackets in the list "
                              "of tabs.") from None

    def _write(self, page, args):
        """write: click the field, check that a text field has the focus, type, and read it back."""
        px, py = self._xy(args)
        text = str(args.get("text", ""))
        # Passwords and one-time codes are the user's to type: a page's instructions could otherwise
        # have the agent enter credentials it saw somewhere into a form of the page's.
        frame, fx, fy = self._frame_at(page, px, py)
        if page_js(frame, SECRET_AT_JS, [fx, fy]):
            raise ToolRefused(PASSWORD_REFUSAL)
        # A line break in the text is typed as Enter, so it submits the form just like enter (press_enter).
        if args.get("enter") or re.search(r"[\r\n]", text):
            self._confirm_risky(page, px, py, "submit")
        self._gesture()
        page.mouse.click(px, py, delay=50)
        self._gesture()
        field, field_frame = self._focused_field(page)
        if field.get("secret"):  # the click moved the focus to such a field after all
            raise ToolRefused(PASSWORD_REFUSAL)
        kind = field.get("kind")
        if kind == "frame":
            kind = "editor"  # the focus is in a frame that did not say what it is: typed, not checked
        if kind not in ("field", "editor"):
            self.last_ok = False
            return (f"Warning: no text field had focus at ({args.get('x')},{args.get('y')}); nothing was typed."
                    + (" The field there is read-only." if kind == "readonly" else ""))
        if args.get("overwrite", True):
            page.keyboard.press("Control+A")
            page.keyboard.press("Backspace")
        page.keyboard.type(text)
        self._gesture()
        name = re.sub(r"[^\w .,:@#/-]", "", str(field.get("name") or ""))[:40].strip()
        what = field.get("type") or field.get("tag") or "field"
        desc = f'the {what} field "{name}"' if name else f"the {what} field"
        if kind == "editor":
            report = f"Typed into {desc}; not checked (this kind of editor cannot be read back)."
        else:
            try:
                value = page_js(field_frame, FIELD_VALUE_JS)
            except Exception:
                value = None
            if value is None:
                report = f"Typed into {desc}; not checked (it could not be read back)."
            elif field.get("type") == "password":
                report = f"Typed into {desc}; it now holds {len(value)} characters."
            elif not str(value).strip():
                report = f"Typed into {desc}; it is now empty."
            else:
                shown = one_line(str(value), 80)
                if text.strip() and text.strip()[:80] in str(value):
                    report = f'Typed into {desc}; it now reads "{shown}".'
                else:
                    # Not just what was typed: what the field holds is the page's doing, so it is
                    # marked as page text.
                    report = f"Typed into {desc}; it now reads {untrusted(shown)}."
        if args.get("enter"):
            page.keyboard.press("Enter")
            self._gesture()
        return report

    def _find(self, page, query):
        """ctrl_f_web: the first match in the page or in one of its frames from the same site, selected
        and scrolled into view."""
        main = page.main_frame
        try:
            origin = urllib.parse.urlsplit(page.url)[:2]
        except ValueError:
            origin = None
        for frame in page.frames:
            if frame != main:
                try:
                    same = urllib.parse.urlsplit(frame.url)[:2] == origin or frame.url in ("about:blank", "about:srcdoc")
                except ValueError:
                    same = False
                if not same:
                    continue
            try:
                hit = page_js(frame, "(q) => window.find(q, false, false, true)", query)
            except PageBusy:
                if frame == main:
                    raise
                continue
            except Exception:
                continue
            if hit:
                if frame != main:
                    try:
                        frame.frame_element().scroll_into_view_if_needed(timeout=2000)
                    except Exception:
                        pass
                return "Match found and scrolled into view."
        return "No match on this page."

    def _wait(self, page, args):
        """wait_web. In short slices, so Ctrl+C (STOP) ends a long wait within a quarter of a second.
        settle_mode adaptive ends it early once the page has been quiet (no requests for 500 ms and an
        unchanged DOM), but never before min(1 s, the time asked for)."""
        seconds = min(60.0, max(0.0, float(args.get("seconds", 2) or 0)))
        start = time.monotonic()
        end, least = start + seconds, start + min(1.0, seconds)
        adaptive = str(self.s.get("settle_mode") or "fixed") == "adaptive"
        if adaptive:
            self._start_quiet(page)
        while not STOP.is_set():
            now = time.monotonic()
            if now >= end:
                break
            if adaptive and now >= least:
                live = self._live_page()
                if not self._loading(live) and self._network_quiet(live, 0.5):
                    quiet = self._quiet_ms(live, seconds=0.25)
                    if quiet is not None and quiet >= 250:
                        break
            self._pause(min(250.0, max(1.0, (end - time.monotonic()) * 1000)))
