"""llama-server: the port, the GPU layout, starting and watching the server while it loads, its
log, restarts, and an external server (external_server)."""

import ctypes
import math
import os
import re
import secrets
import signal
import socket
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from . import paths, runstate
from .backends import backend_is_auto, gpu_key, nvidia_smi_gpus, pick_backend, remember_bad_backend
from .console import LiveLine, c, clean, console_only, err, first_line, info, warn
from .gguf import DEFAULT_MOE_RESERVE_MIB, VISION_RESERVE_MIB, model_layout, moe_reserve_mib
from .modelcopy import CopyCancelled, Progress, _size_of, decide_model_copy, eta_text, model_files, record_load_speed
from .paths import BIN, hide_home, hide_home_in_file, pc_data_dir
from .preflight import context_size, gb, is_dense, model_alias, model_path, preserves_reasoning
from .questions import _keyboard, ask_console
from .screenshot import MAX_IMAGE_TILES, MIN_IMAGE_TILES
from .serverflags import LOCKDOWN_SERVER_FLAGS, flat_args, keep_supported_flags, server_flags
from .winproc import (CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW, blocked_message, default_threads, dll_not_found,
                      has_console, keep_pc_awake, process_cpu_seconds, process_read_bytes, vc_runtime_advice,
                      vc_runtime_missing)


# --------------------------------------------------------------------------------------------
# llama-server
# --------------------------------------------------------------------------------------------

# Calls to llama-server on this computer never go through a proxy: with a system or environment
# proxy set, urllib would otherwise send even 127.0.0.1 to it, and loading would never finish.
LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def port_is_free(port, timeout=None):
    """Nothing listens on this port of 127.0.0.1 (a connect test). Windows retries a connection that is
    refused for about two seconds before giving up; a listener on this PC answers at once, so a short
    `timeout` gives the same answer without that wait."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if timeout is not None:
            s.settimeout(timeout)
        try:
            return s.connect_ex(("127.0.0.1", port)) != 0
        except OSError:
            return True  # the timeout passed: nothing accepted the connection


def port_bindable(port):
    """A server could listen on this port of 127.0.0.1. Ports that Hyper-V, WSL or Docker reserved have
    no listener, so they look free to a connect test, but binding them is refused."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


PORT_TRIES = 50
PORT_CONNECT_TIMEOUT = 0.5  # seconds; a listener on 127.0.0.1 accepts in well under a millisecond


def server_healthy(base, timeout=2):
    """True when the model server's /health answers 200 (never through a proxy)."""
    try:
        with LOCAL.open(base + "/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def gpu_layout_args(device, settings, auto_fit=False, emulate_free_mib=None, reserve_mib=None, dense=False,
                    mmproj_mib=0):
    """(llama-server arguments for the GPU layout, True when the explicit MoE layout is used).

    Mixture-of-experts models (the fast profiles): with enough free VRAM, every expert stays in system
    RAM (--cpu-moe) and the VRAM left after `reserve_mib` (the rest of the model, the vision projector,
    the KV cache and compute buffers; see moe_reserve_mib) and fit_target_mib goes to the expert cache,
    which measured faster than a static split. With less (or auto_fit), llama.cpp fits the layers
    itself. Dense models (the accurate profile) always let llama.cpp fit the layers (--fit-target).
    llama.cpp's fit does not count the vision projector, so on every --fit-target path the margin
    grows by the projector's size (mmproj_mib) and its buffers (VISION_RESERVE_MIB).

    emulate_free_mib (benchmarks: --emulate-free-vram-mib) lays the model out as if the GPU had at
    most that much free VRAM, so a big desktop card reproduces a small laptop card's layout. When
    llama.cpp fits the layout itself (--fit-target), the difference is added to the margin it must
    leave free instead."""
    if not device:
        return [], False
    real_free = int(device["free"])
    free = min(real_free, int(emulate_free_mib)) if emulate_free_mib is not None else real_free
    args = ["--device", device["id"]]
    reserve = DEFAULT_MOE_RESERVE_MIB if reserve_mib is None else int(reserve_mib)
    if not dense and free >= reserve + 900 and not auto_fit:
        args += ["-ngl", "99", "--cpu-moe", "--fit", "off"]
        cache = settings["moe_cache_mib"]
        if cache in ("auto", None, ""):
            cap = int(settings.get("moe_cache_cap_mib", 4096))
            cache = min(cap, free - reserve - int(settings["fit_target_mib"]))
        if int(cache) >= 256:
            args += ["--moe-cache-mib", str(int(cache))]
        return args, True
    # The projector and its buffers are on the GPU whichever way the layers are fitted (measured on
    # the 27B: about 1.0-1.1 GB above the fit budget), so the MoE fallback leaves room for them too.
    margin = int(settings["fit_target_mib"]) + (real_free - free) + int(mmproj_mib) + VISION_RESERVE_MIB
    return args + ["--fit-target", str(margin)], False


def parse_affinity_mask(text):
    """A CPU mask written in hex ('ff', '0xF0F'), as an int; ValueError when it is not one."""
    mask = int(str(text).strip().lower().removeprefix("0x"), 16)
    if mask <= 0:
        raise ValueError("the mask selects no CPU")
    return mask


def set_process_affinity(proc, mask):
    """Keep a child process on the logical CPUs in `mask` (benchmarks: --affinity-mask). Bits for
    CPUs this PC does not have are dropped. Returns the mask applied, or None."""
    if os.name != "nt" or proc is None:
        return None
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                           ctypes.POINTER(ctypes.c_size_t)]
    k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    handle = wintypes.HANDLE(int(proc._handle))
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if not k32.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
        warn(f"Could not read the CPU affinity of llama-server (error {ctypes.get_last_error()}).")
        return None
    usable = mask & system_mask.value
    if not usable:
        warn(f"The affinity mask {mask:#x} selects none of this PC's CPUs ({system_mask.value:#x}); not applied.")
        return None
    if not k32.SetProcessAffinityMask(handle, usable):
        warn(f"Could not set the CPU affinity of llama-server (error {ctypes.get_last_error()}).")
        return None
    return usable


def server_environment(api_key, extra=None):
    """The environment for llama-server: this process's, minus every LLAMA_* and AIP_* variable,
    plus the run's API key. llama-server reads its options from LLAMA_* variables as well (for
    example LLAMA_ARG_TOOLS turns on its file and shell tools, LLAMA_SERVER_SLOTS_DEBUG puts prompts
    in /slots) and AIP_* moves it to another port, so a PC that happens to set them must not change
    the server behind the agent's back. The key travels here rather than on the command line, so it
    never shows in the process list or in the log. `extra`: variables a setting asks for."""
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("LLAMA_", "AIP_"))}
    if api_key:
        env["LLAMA_API_KEY"] = api_key
    env.update(extra or {})
    return env


# Flags in extra_server_args that loosen the server's lock-down, or replace the agent's own key.
LOOSENING_SERVER_FLAGS = ("--api-key", "--api-key-file", "--cors-origins", "--cors-credentials", "--tools", "--agent",
                          "-ag", "--mcp-servers-config", "--mcp-servers-json", "--ui", "--webui", "--slots",
                          "--ui-mcp-proxy", "--webui-mcp-proxy")

# llama-server writes these twice for every screenshot. They are expected (image positions jump),
# not errors, and would bury the lines that matter when a log tail is shown. The log keeps them.
NOISY_SERVER_LINES = ("find_slot: non-consecutive token position",)


def flag_value(args, flag):
    """The value given to `flag` in a list of command-line arguments ('--flag v' or '--flag=v')."""
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def loggable_command(args):
    """The command line for the log: no API key, and the home folder written as %USERPROFILE%."""
    shown, hide_next = [], False
    for a in args:
        if hide_next:
            shown.append("<hidden>")
            hide_next = False
        elif a == "--api-key":
            shown.append(a)
            hide_next = True
        elif a.startswith("--api-key="):
            shown.append("--api-key=<hidden>")
        else:
            shown.append(a)
    return hide_home(subprocess.list2cmdline(shown))


MAX_SERVER_RESTARTS = 2      # per session: a server that keeps crashing is not restarted forever
SLOW_LOAD_HINT_MB_S = 45     # below this for 20 s while loading: probably a USB 2 port or a slow stick
SLOW_LOAD_HINT_SECONDS = 20
STALL_SECONDS = 120          # no reads, no log lines and no CPU time for this long: ask what to do
UNATTENDED_STALL_SECONDS = 600


class LoadWatch:
    """Watches llama-server while it loads: progress and speed of its file reads, a hint when they are
    slow, and a stall (nothing read, nothing logged, no CPU time used) that lasts too long."""

    def __init__(self, total, now):
        self.total = total
        self.samples = []            # (time, bytes read) over the last few seconds
        self.first = None            # (time, bytes read) at the first sample
        self.read_all = None         # (time, bytes read) once the model files had been read
        self.slow_since = None
        self.hinted = False
        self.last_activity = now
        self.last_state = None

    def update(self, now, read, log_size, cpu):
        """Feed one sample (bytes read, log size, CPU seconds; each may be None). Returns ["slow"]
        the first time reading has been slow for SLOW_LOAD_HINT_SECONDS, else []."""
        state = (read, log_size, None if cpu is None else round(cpu, 1))
        if state != self.last_state:
            self.last_state = state
            self.last_activity = now
        if read is None:
            return []
        if self.first is None:
            self.first = (now, read)
        self.samples = [(t, r) for t, r in self.samples if now - t <= 5.0] + [(now, read)]
        if self.read_all is None and self.total and read >= self.total:
            self.read_all = (now, read)
        rate = self.rate()
        # Only while the model is being read (the first seconds go to starting up), and only a slow
        # read, not a pause: a load that stops altogether is the stall check's business.
        if self.read_all is None and read - self.first[1] >= 64 * 2**20 and rate:
            if rate >= SLOW_LOAD_HINT_MB_S * 1e6:
                self.slow_since = None
            else:
                self.slow_since = now if self.slow_since is None else self.slow_since
                if not self.hinted and now - self.slow_since >= SLOW_LOAD_HINT_SECONDS:
                    self.hinted = True
                    return ["slow"]
        return []

    def stalled_for(self, now):
        return now - self.last_activity

    def rate(self):
        """Bytes per second over the last few seconds, or None before there are two samples."""
        if len(self.samples) < 2:
            return None
        (t0, r0), (t1, r1) = self.samples[0], self.samples[-1]
        return (r1 - r0) / (t1 - t0) if t1 > t0 else None

    def average_rate(self):
        """Bytes per second from the first sample until the model had been read (or until now), when
        at least 1 GiB was read in that time; else None (the load had mostly happened unwatched)."""
        end = self.read_all or (self.samples[-1] if self.samples else None)
        if self.first is None or end is None:
            return None
        n, t = end[1] - self.first[1], end[0] - self.first[0]
        return n / t if n >= 2**30 and t > 0 else None

    def line(self, elapsed):
        if not self.total or not self.samples:
            return f"loading the model into memory ... {int(elapsed)}s"
        if self.read_all is None:
            done = min(self.samples[-1][1], self.total)
            text = f"loading the model: {gb(done)} of {gb(self.total)} GB"
            rate = self.rate()
            if rate and rate > 1e6:
                text += f" ({rate / 1e6:.0f} MB/s, {eta_text((self.total - done) / rate)} left)"
            return text
        return f"loading the model: all read, getting it ready ... {int(elapsed)}s"


class LlamaServer:
    def __init__(self, settings):
        self.settings = settings
        self.proc = None
        self.log_path = None
        self.port = int(settings["port"])
        self.base = f"http://127.0.0.1:{self.port}"
        self.choice = None
        self.explicit_layout = False
        self.api_key = None
        self.emulate_free_vram_mib = None  # benchmarks: --emulate-free-vram-mib
        self.affinity_mask = None          # benchmarks: --affinity-mask
        self.restarts = 0                  # restarts after a crash in this session
        self.quit_requested = False        # the user chose Quit while a load stalled
        self.copy_to_pc = None             # decided once per run (decide_model_copy)
        self.files = None                  # the model files to load (the PC's copy or the drive's)
        self.load_bytes = 0
        self.failed = []                   # (backend, device) that could not load the model in this run
        self.unattended = False            # run_tasks: nobody to ask when the load stalls
        self._graceful = False             # this start can be stopped with Ctrl+Break (graceful_server_stop)
        self._auto_fit = False             # how the last start laid the model out (restart() repeats it)
        self._launcher = None              # begin(): the copy and the start, in the background
        self._launch_error = None
        self._progress = None
        self._notes = []
        self._quiet = False
        self._noticed = set()
        self._cancel = threading.Event()

    # -- messages that may come from the background start -------------------------------------

    def _say(self, kind, text):
        if self._quiet:
            self._notes.append((kind, text))
        else:
            (warn if kind == "warn" else info)(text)

    def _flush_notes(self):
        notes, self._notes = self._notes, []
        for kind, text in notes:
            (warn if kind == "warn" else info)(text)

    # -- choosing the build ---------------------------------------------------------------------

    def _check_vc_runtime(self):
        """Before the first build runs: say plainly when the Visual C++ runtime llama.cpp needs looks
        missing. Only a warning, because a folder on the PATH may still hold it; a build that then
        fails for want of it stops with the same advice (list_devices, wait_ready)."""
        if ("vc",) in self._noticed:
            return
        self._noticed.add(("vc",))
        missing = vc_runtime_missing(BIN / "cpu")
        if missing:
            self._say("warn", vc_runtime_advice(missing))

    def _choose_backend(self, after=None):
        self._check_vc_runtime()
        self.choice = pick_backend(self.settings, after=after)
        backend, exe, device = self.choice
        if device:
            info(f"GPU: {device['name']} ({device['total'] // 1024} GB, {device['free'] // 1024} GB free) via {backend}")
            if self.emulate_free_vram_mib is not None:
                info(f"laying the model out as if only {min(device['free'], self.emulate_free_vram_mib)} MiB "
                     "of VRAM were free (--emulate-free-vram-mib)")
        elif backend == "cpu" or self.settings.get("backend") == "cpu" or backend_is_auto(self.settings):
            # (Without a CPU build on the drive, a GPU build runs on the CPU instead. A forced GPU build
            # that found no GPU has said so itself.)
            if self.failed:
                warn("The GPU builds could not run the model, so it runs on the CPU only: this works, but each step "
                     "will be slow.")
            else:
                warn("No supported GPU found. Running on the CPU only: this works, but each step will be slow.")

    def can_fall_back(self):
        """Another build may still work: backend "auto" (or a forced build that is not on the drive), and
        this one is not the CPU build (nor a GPU build already running on the CPU, for want of a CPU
        build)."""
        return (backend_is_auto(self.settings) and self.choice is not None and self.choice[0] != "cpu"
                and self.choice[2] is not None and not self.quit_requested)

    def next_backend(self):
        """After a build could not load the model: move to the next build in BACKENDS that sees a GPU,
        else the CPU build. Returns the new build's name."""
        self.failed.append((self.choice[0], self.choice[2]))
        self.explicit_layout = False
        self._choose_backend(after=self.choice[0])
        return self.choice[0]

    def can_restart(self):
        """After a crash: the server may be started again (MAX_SERVER_RESTARTS per session)."""
        return self.restarts < MAX_SERVER_RESTARTS and not self.quit_requested and self.choice is not None

    def restart(self):
        """Start llama-server again after it crashed, with the same build and layout. Seconds until it
        is ready, or None."""
        self.restarts += 1
        self._end_process()
        self.start(auto_fit=self._auto_fit)
        return self.wait_ready()

    def remember_failed_backends(self):
        """Once another build has loaded the model, the ones that failed on this GPU and driver are
        remembered, so the next start skips their slow failed load (logs\\backend-cache.json). Only when
        the build that worked used the GPU too: when only the CPU build could load the model, the GPU may
        just have been short of free memory at the time (a game, another program), and remembering that
        would keep every later start on the slow CPU."""
        if self.choice is None or not self.choice[2]:
            self.failed = []
            return
        for backend, device in self.failed:
            if device:
                remember_bad_backend(gpu_key(device, nvidia_smi_gpus()), backend)
        self.failed = []

    # -- starting -------------------------------------------------------------------------------

    def prepare(self, interactive=True):
        """What must happen before the start, in the foreground: whether the model is copied to this
        PC (which may ask), and which build runs it (which says which GPU)."""
        if self.copy_to_pc is None:
            self.copy_to_pc = decide_model_copy(self.settings, interactive)
        if self.choice is None:
            self._choose_backend()

    def begin(self, interactive=True):
        """Start loading as early as possible. The copy to this PC (if any) and the start of
        llama-server run in the background, so the browser (Chrome's Allow click) can start meanwhile;
        wait_ready() then shows how far they got."""
        self.prepare(interactive)
        if self.copy_to_pc and self.files is None:
            files = [model_path(self.settings, "model"), model_path(self.settings, "mmproj")]
            for f in files:
                if not f.exists():
                    raise SystemExit(f"Missing model file: {f}")
            self._progress = Progress(sum(f.stat().st_size for f in files), "copying the model to this PC")
            self._cancel = threading.Event()
            self._launch_error = None
            self._launcher = threading.Thread(target=self._background_start, name="model-start", daemon=True)
            self._launcher.start()
        else:
            self.start()
        return self.choice[0]

    def _background_start(self):
        try:
            with keep_pc_awake():
                notes = []
                self.files = model_files(self.settings, copy=True, progress=self._progress, notes=notes,
                                         cancel=self._cancel)
                self._notes += [("warn", n) for n in notes]
            if not self._cancel.is_set():
                self.start(quiet=True)
        except CopyCancelled:
            pass
        except BaseException as e:  # a blocked program (SystemExit) or anything else: shown by wait_ready
            self._launch_error = e

    def start(self, auto_fit=False, quiet=False):
        """Start llama-server. auto_fit=True leaves the GPU/CPU split entirely to llama.cpp."""
        s = self.settings
        self._quiet = quiet
        self._auto_fit = auto_fit
        try:
            return self._start(s, auto_fit)
        finally:
            self._quiet = False

    def _start(self, s, auto_fit):
        if self.files is None:
            if self.copy_to_pc is None:
                self.copy_to_pc = s.get("copy_model_to_pc") is True
            self.files = model_files(s, copy=self.copy_to_pc)
        model, mmproj = self.files
        if self.choice is None:
            self._choose_backend()
        backend, exe, device = self.choice

        # The bind test finds ports Windows reserved; the connect test finds a program that listens on
        # all addresses (binding 127.0.0.1 alone still works then). Without a short timeout the connect
        # test would wait about two seconds on every free port.
        first = self.port
        for _ in range(PORT_TRIES):
            if port_bindable(self.port) and port_is_free(self.port, timeout=PORT_CONNECT_TIMEOUT):
                break
            self.port += 1
        else:
            raise SystemExit(f"No port between {first} and {self.port - 1} on 127.0.0.1 can be used for the model "
                             "server. Set another \"port\" in settings.json.")
        self.base = f"http://127.0.0.1:{self.port}"

        # A new key for every start: only this process knows it, so no web page, other program or
        # other user of this PC can use the model or take its single slot.
        self.api_key = secrets.token_urlsafe(32)
        extra = [str(a) for a in s["extra_server_args"]]
        for flag in dict.fromkeys(a.split("=", 1)[0] for a in extra):
            if flag in LOOSENING_SERVER_FLAGS and ("extra", flag) not in self._noticed:
                self._noticed.add(("extra", flag))
                self._say("warn", f"extra_server_args contains {flag}, which loosens how the model server is locked "
                                  "down or replaces the agent's own setting for it.")
        own_key = flag_value(extra, "--api-key")
        if own_key is None and flag_value(extra, "--api-key-file"):
            key_file = Path(flag_value(extra, "--api-key-file"))
            if not key_file.is_absolute():
                key_file = exe.parent / key_file  # llama-server runs in its own folder and reads it from there
            try:
                own_key = next(line.strip() for line in key_file.read_text(
                    encoding="utf-8-sig").splitlines() if line.strip() and not line.lstrip().startswith("#"))
            except (OSError, StopIteration):
                self._say("warn", "Could not read the key from --api-key-file, so the agent cannot use the model server.")
        if own_key:
            self.api_key = own_key.split(",")[0].strip()  # the server takes the command line's key over ours

        dense = is_dense(s)
        ctx = context_size(s)
        mmproj_bytes = _size_of(mmproj)
        groups = [
            ("-m", [model]),
            ("--mmproj", [mmproj]),
            ("--alias", [model_alias(s)]),
            ("--host", ["127.0.0.1"]),
            ("--port", [self.port]),
            ("-c", [ctx]),
            ("-np", ["1"]),
            ("-fa", ["on"]),
            ("-ctk", ["q8_0"]),
            ("-ctv", ["q8_0"]),
            # Prompt cache. Holo4 is hybrid: 30 Gated DeltaNet layers and 10 attention layers. The
            # DeltaNet state cannot be rolled back token by token, so llama-server can only resume a
            # prompt from a saved checkpoint at or before the first token that changed, and it drops
            # every later checkpoint (b11669 tools/server/server-context.cpp:3844-3904 and
            # :4094-4173). With its default spacing of 8192 tokens between checkpoints, the only one
            # before an evicted screenshot is the one at the start of the task, so each step read the
            # whole history again. Checkpoints every checkpoint_min_step tokens keep one close behind
            # the change. The RAM prompt cache (--cache-ram) only ever stores states older than the
            # live slot, which can never be resumed further for a single agent: off, and with it the
            # idle-slot saving that needs it.
            ("--ctx-checkpoints", [int(s["ctx_checkpoints"])]),
            ("--checkpoint-min-step", [int(s["checkpoint_min_step"])]),
            ("--cache-ram", [int(s["cache_ram_mib"])]),
        ]
        if int(s["cache_ram_mib"]) == 0:
            groups.append(("--no-cache-idle-slots", []))
        groups.append(("--reasoning-format", ["deepseek"]))
        if not dense and not preserves_reasoning(s):
            # The 35B's template shows no past reasoning by default anyway. The 27B's template renders
            # a think block for every past turn (empty when the agent sends no reasoning back), which
            # is how H Company serves it, so the server's default (preserve) stays for dense models.
            # With preserve_reasoning on, the server's default is what the agent needs.
            groups.append(("--no-reasoning-preserve", []))
        groups += [
            ("--load-mode", ["none"]),
            ("--image-min-tokens", [MIN_IMAGE_TILES]),
            ("--image-max-tokens", [MAX_IMAGE_TILES]),  # optional: a build without it starts without the cap
            # One micro-batch per screenshot: the CPU-resident experts are streamed to the GPU once
            # per micro-batch while reading the prompt, so bigger batches mean fewer transfers.
            ("-b", ["2048"]),
            ("-ub", ["2048"]),
            # Locked down (the key comes with the environment, see server_environment): browsers
            # may call the server only from pages on localhost, and /slots is off. The web chat
            # page is off too unless the web_ui setting asks for it.
            ("--cors-origins", ["localhost"]),
            ("--no-slots", []),
        ]
        if not s["web_ui"]:
            groups.append(("--no-webui", []))
        reserve = None
        if device and not dense:
            layout = model_layout(model)
            if layout is not None:
                reserve = moe_reserve_mib(layout, mmproj_bytes, ctx)
        layout_args, self.explicit_layout = gpu_layout_args(
            device, s, auto_fit, self.emulate_free_vram_mib, reserve_mib=reserve, dense=dense,
            mmproj_mib=math.ceil(mmproj_bytes / 2**20))
        groups += _groups_of(layout_args)
        if not device and backend != "cpu":
            groups.append(("--device", ["none"]))  # a GPU build without a usable GPU: nothing offloaded to an iGPU
        if s.get("moe_cache_cpu_share") != "" and not dense and self.explicit_layout:
            groups.append(("--moe-cache-cpu-share", [s["moe_cache_cpu_share"]]))
        groups.append(("-t", [int(s["threads"]) or default_threads()]))

        supported = server_flags(exe)
        groups, dropped, missing = keep_supported_flags(groups, supported)
        new = [f for f in dropped if ("drop", backend, f) not in self._noticed]
        if new:
            self._noticed.update(("drop", backend, f) for f in new)
            self._say("info", f"this llama.cpp build ({backend}) does not know {', '.join(new)}; started without "
                              f"{'it' if len(new) == 1 else 'them'}")
            weaker = [f for f in new if f in LOCKDOWN_SERVER_FLAGS]
            if weaker:
                self._say("warn", f"Without {', '.join(weaker)} the model server is locked down less tightly (its "
                                  "key still protects it). Run Build GhostAgent.cmd again for the llama.cpp build "
                                  "GhostAgent was made for.")
        new = [f for f in missing if ("miss", backend, f) not in self._noticed]
        if new:
            self._noticed.update(("miss", backend, f) for f in new)
            self._say("warn", f"This llama.cpp build ({backend}) does not list {', '.join(new)}, which the agent needs; "
                              "the model server may not start. Run Build GhostAgent.cmd again.")
        args = [str(exe)] + flat_args(groups) + extra

        env_extra = {}
        if s.get("op_offload_min_batch") != "":
            # Prompt micro-batches smaller than this run the CPU-resident experts on the CPU instead
            # of uploading them to the GPU (llama.cpp's default is 32 tokens).
            env_extra["GGML_OP_OFFLOAD_MIN_BATCH"] = str(int(s["op_offload_min_batch"]))
        # Ctrl+Break reaches only a process on this window's console. Without a console here (a test
        # or a service), a child without CREATE_NO_WINDOW would open a console window of its own.
        graceful = bool(s.get("graceful_server_stop")) and has_console()

        paths.LOGS.mkdir(parents=True, exist_ok=True)
        self.log_path = paths.LOGS / f"llama-server-{datetime.now():%Y%m%d-%H%M%S}.log"
        log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        log.write("command: " + loggable_command(args) + "\n\n")
        log.flush()
        # A separate process group keeps Ctrl+C (stop the current task) away from the server. Without
        # a console of its own (graceful_server_stop) it shares this window's console, which is what
        # lets stop() send it Ctrl+Break.
        flags = CREATE_NEW_PROCESS_GROUP | (0 if graceful else CREATE_NO_WINDOW)
        try:
            self.proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, cwd=str(exe.parent),
                                         env=server_environment(self.api_key, env_extra), creationflags=flags)
        except OSError as e:
            log.close()
            msg = blocked_message(e, f"{exe.name} ({backend} build), an unsigned program from the official llama.cpp release")
            raise SystemExit(msg or f"Could not start {exe} ({first_line(e)}).") from None
        finally:
            if not log.closed:
                log.close()  # the server has its own handle to it
        self._graceful = graceful
        self.load_bytes = sum(_size_of(f) for f in (model, mmproj))
        if self.affinity_mask is not None:
            applied = set_process_affinity(self.proc, self.affinity_mask)
            if applied is not None:
                self._say("info", f"llama-server runs on the CPUs in mask {applied:#x}")
        if s["web_ui"]:
            # On the console only: the key is not written to any log.
            console_only(c(f"  The model's chat page is at {self.base} (web_ui). It asks for an API key; "
                           f"for this run it is: {self.api_key}", "90"))
        return backend

    # -- waiting --------------------------------------------------------------------------------

    def _wait_for_launcher(self):
        """begin()'s background work: show the copy's progress until llama-server has been started.
        False when nothing was started."""
        shown = False
        live = LiveLine()
        while self._launcher.is_alive():
            if self._progress is not None and self._progress.done:
                live.show(self._progress.line())
                shown = True
            self._launcher.join(0.5)
        if self._progress is not None and (shown or self._progress.done > self._progress.skipped):
            live.close(self._progress.line())  # how it ended, also for a quick copy
        self._launcher = None
        self._flush_notes()
        if self._launch_error is not None:
            e, self._launch_error = self._launch_error, None
            raise e
        return self.proc is not None

    def wait_ready(self):
        """Seconds until ready, or None if the server exited while loading (or the user chose Quit
        when the load stalled). Shows how much of the model has been read, how fast, and how long it
        should take; hints at a slow USB port; asks what to do when nothing moves for 2 minutes."""
        if self._launcher is not None and not self._wait_for_launcher():
            return None
        t0 = time.time()
        watch = LoadWatch(self.load_bytes, time.monotonic())
        last = None
        live = LiveLine()
        stall_limit = STALL_SECONDS if self._can_ask() else UNATTENDED_STALL_SECONDS
        from_pc = bool(self.files) and _is_under(self.files[0], pc_data_dir())
        while True:
            code = self.proc.poll()
            if code is not None:
                live.close()
                err("The model server stopped while loading. Last lines of its log:")
                for line in self.tail(25):
                    print("    " + clean(line))
                if dll_not_found(code):
                    exe = Path(self.choice[1]) if self.choice else BIN / "cpu" / "llama-server.exe"
                    missing = vc_runtime_missing(exe.parent)
                    if missing:
                        raise SystemExit(vc_runtime_advice(missing))  # every build needs it: no point trying another
                    err(f"Windows could not find a DLL that {exe.parent.name}\\llama-server.exe needs. Run Build "
                        "GhostAgent.cmd again to repair the llama.cpp files.")
                return None
            if server_healthy(self.base):
                live.close()
                if not from_pc:
                    record_load_speed(watch.average_rate())
                return time.time() - t0
            now = time.monotonic()
            events = watch.update(now, process_read_bytes(self.proc), _size_of(self.log_path),
                                  process_cpu_seconds(self.proc))
            text = watch.line(time.time() - t0)
            if text != last:
                last = text
                live.show(text)
            if "slow" in events and not from_pc:
                live.close()
                warn(f"The model loads at only {watch.rate() / 1e6:.0f} MB/s. That looks like a USB 2 port or a slow "
                     "stick: a USB 3 port (often blue, or marked SS) is much faster, and copy_model_to_pc = true in "
                     "settings.json keeps a copy on this PC, so later starts take seconds.")
                last = None
            if watch.stalled_for(now) >= stall_limit:
                live.close()
                what = self._stalled(stall_limit)
                if what == "quit":
                    return None
                if what == "restart":
                    self._end_process()
                    self.start(auto_fit=self._auto_fit)  # the same layout as the start that stalled
                    t0 = time.time()
                    watch = LoadWatch(self.load_bytes, time.monotonic())
                else:
                    watch.last_activity = time.monotonic()
                last = None
            time.sleep(0.5)

    def _can_ask(self):
        return not self.unattended and not runstate.HEADLESS and _keyboard() is not None

    def _stalled(self, seconds):
        """The load showed no sign of life for `seconds`: "wait", "restart" or "quit"."""
        err(f"The model server has shown no progress for {int(seconds // 60)} minutes (nothing read, nothing "
            "logged, no CPU time used). Last lines of its log:")
        for line in self.tail(15):
            print("    " + clean(line))
        if not self._can_ask():
            err("Giving up on this start of the model server.")
            self.quit_requested = True
            self._end_process()
            return "quit"
        answer, how = ask_console("Stop waiting? r = start the model server again, q = quit, "
                                  "Enter = keep waiting [r/q/Enter]")
        answer = (answer or "").strip().lower() if how == "answered" else ""
        if answer in ("q", "quit"):
            self.quit_requested = True
            self._end_process()
            return "quit"
        if answer in ("r", "retry", "restart"):
            info("starting the model server again ...")
            return "restart"
        info("waiting for the model server ...")
        return "wait"

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def describe(self):
        return f"server {self.base}, log {self.log_path.name if self.log_path else '-'}"

    def tail(self, n):
        """The last n lines of the server's log, without the expected warnings that would bury the
        real problem (the file itself keeps every line)."""
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return []
        return [line for line in lines if not any(s in line for s in NOISY_SERVER_LINES)][-n:]

    def _end_process(self):
        if self.proc and self.proc.poll() is None:
            if self._graceful:
                # Ctrl+Break first: llama-server can then write its closing statistics (the MoE-cache
                # numbers with -lv 4) before it exits. Tuning runs only; terminate() follows if not.
                try:
                    os.kill(self.proc.pid, signal.CTRL_BREAK_EVENT)
                    self.proc.wait(15)
                except Exception:
                    pass
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(10)
                except Exception:
                    self.proc.kill()
        if self.log_path:
            # llama-server writes the model's full path into its log, which names the Windows account
            # when the model was copied to this PC. The log stays on the stick, so that goes now.
            hide_home_in_file(self.log_path)

    def stop(self):
        self._cancel.set()  # a copy still running in the background ends at its next chunk
        if self._launcher is not None and self._launcher is not threading.current_thread():
            self._launcher.join(30)
        self._end_process()


def _is_under(path, folder):
    try:
        return Path(path).resolve().is_relative_to(Path(folder).resolve())
    except (OSError, ValueError):
        return False


def _groups_of(args):
    """[(flag, [values])] for a flat argument list (a value is anything that is not a flag)."""
    groups = []
    for a in args:
        if a.startswith("-") and not re.fullmatch(r"-\d+(\.\d+)?", a):
            groups.append((a, []))
        else:
            groups[-1][1].append(a)
    return groups


class ExternalServer:
    """A model server that is already running (the external_server setting): the agent starts and
    stops nothing and needs no model files. For tests (tools\\bench\\fake_llama.py) and benchmarks
    (a llama-server started by hand)."""

    def __init__(self, url, api_key=None):
        self.base = url.rstrip("/")
        self.proc = None
        self.log_path = None
        # Only the key configured for it (external_server_key): the key made for a llama-server the
        # agent starts is never sent anywhere else.
        self.api_key = api_key or None
        self.explicit_layout = False
        self.restarts = 0
        self.quit_requested = False
        self.unattended = False

    def prepare(self, interactive=True):
        pass

    def begin(self, interactive=True):
        return "external"

    def start(self, auto_fit=False, quiet=False):
        return "external"

    def can_fall_back(self):
        return False

    def can_restart(self):
        return False  # the agent did not start it, so it does not restart it either

    def wait_ready(self, timeout=600, give_up_unreachable=5.0):
        """Seconds until /health answers 200, or None. A server that is still loading (it answers,
        but not yet with 200) gets `timeout` seconds; when nothing accepts connections at all, the
        wait ends after `give_up_unreachable` seconds."""
        u = urllib.parse.urlsplit(self.base)
        host, port = u.hostname or "127.0.0.1", u.port or (443 if u.scheme == "https" else 80)
        t0 = time.time()
        reachable_at = t0
        while time.time() - t0 < timeout:
            if server_healthy(self.base):
                return time.time() - t0
            try:
                socket.create_connection((host, port), timeout=1).close()
                reachable_at = time.time()
            except OSError:
                if time.time() - reachable_at > give_up_unreachable:
                    break
            time.sleep(0.5)
        err(f"No model server answers at {self.base}/health (the external_server setting).")
        return None

    def alive(self):
        return server_healthy(self.base)

    def describe(self):
        return f"external server {self.base}"

    def tail(self, n):
        return []

    def stop(self):
        pass
