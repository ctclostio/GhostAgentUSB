"""Preflight: the active model profile and its files, and the checks that would otherwise fail late
(missing model files, too little RAM)."""

import ctypes
import math
import os
from pathlib import Path

from .console import warn
from .paths import ROOT
from .settings import load_profiles


# --------------------------------------------------------------------------------------------
# Preflight: problems that would otherwise show up late, after the Chrome step or the model load.
# --------------------------------------------------------------------------------------------

# RAM the model needs (GB) per profile, for a tools\profiles.json without "ram_gb" (the file's own
# values are the same). Unknown models: their size plus 1 GB.
RAM_NEED_GB = {"fast": 22, "fast-q8": 40, "accurate": 19}
RAM_HEADROOM_GB = 9  # Windows, the browser and everything else that is usually open


def pinned_files():
    """{file name: (size in bytes, profile name)} for every model file in tools\\profiles.json."""
    out = {}
    try:
        for name, prof in load_profiles()["profiles"].items():
            for part in ("model", "mmproj"):
                f = prof.get(part) or {}
                if f.get("name") and f.get("size"):
                    out.setdefault(f["name"], (int(f["size"]), name if part == "model" else None))
    except (ValueError, AttributeError, TypeError):
        pass
    return out


def pinned_sha256(name):
    """The SHA-256 tools\\profiles.json pins for a model file name, or None."""
    for prof in load_profiles()["profiles"].values():
        for part in ("model", "mmproj"):
            f = prof.get(part) or {}
            if isinstance(f, dict) and str(f.get("name", "")).lower() == name.lower() and f.get("sha256"):
                return str(f["sha256"]).lower()
    return None


def active_profile(settings):
    """(name, profile) of the model in use. A "model" setting that names a file some profile pins
    selects that profile (its layout and context then fit the file); otherwise model_profile does."""
    doc = load_profiles()
    profiles = doc["profiles"]
    model = str(settings.get("model") or "").strip()
    if model:
        file_name = Path(model).name.lower()
        for name, prof in profiles.items():
            if str((prof.get("model") or {}).get("name", "")).lower() == file_name:
                return name, prof
    name = str(settings.get("model_profile") or doc["default"])
    if name not in profiles:
        name = doc["default"]
    return name, profiles[name]


def model_path(settings, part="model"):
    """The model ("model") or vision projector ("mmproj") file: the setting when it is set, else the
    active profile's file in the models folder."""
    value = str(settings.get(part) or "").strip()
    if not value:
        _, prof = active_profile(settings)
        value = "models/" + str((prof.get(part) or {}).get("name") or "")
    return ROOT / value


def context_size(settings):
    """Tokens of context for the model: the ctx_size setting, or (0) the active profile's size."""
    ctx = int(settings.get("ctx_size") or 0)
    if ctx > 0:
        return ctx
    return int(active_profile(settings)[1].get("ctx_size") or 65536)


def model_alias(settings):
    """The name llama-server serves the model under (--alias), which requests also send."""
    return str(active_profile(settings)[1].get("alias") or "holo4")


def is_dense(settings):
    """True for a dense model (every parameter used for every token), such as Holo4-27B."""
    return str(active_profile(settings)[1].get("arch") or "moe").lower() == "dense"


def profile_flag(prof, key):
    """A true/false field of a profile in tools\\profiles.json, or None when it is missing or not a
    true/false value (the caller then infers it)."""
    value = prof.get(key)
    return value if isinstance(value, bool) else None


def takes_reasoning_effort(settings):
    """True when the model's chat template knows reasoning_effort (Holo4-27B's does; the 35B's template
    rejects values it does not know). The profile's "takes_reasoning_effort" says so; without it, dense
    models do."""
    prof = active_profile(settings)[1]
    flag = profile_flag(prof, "takes_reasoning_effort")
    return flag if flag is not None else str(prof.get("arch") or "").lower() == "dense"


def preserves_reasoning(settings):
    """True when the model's reasoning from earlier steps goes back into its context (the
    preserve_reasoning setting). "auto" follows the profile's "preserve_reasoning"; without it, the
    dense 27B keeps it (its chat template is built to show past reasoning, which is how H Company
    serves it), and the fast profiles do not until a benchmark shows it helps them."""
    value = settings.get("preserve_reasoning", "auto")
    if isinstance(value, bool):
        return value
    prof = active_profile(settings)[1]
    flag = profile_flag(prof, "preserve_reasoning")
    return flag if flag is not None else str(prof.get("arch") or "").lower() == "dense"


def gb(n):
    return f"{n / 1e9:.1f}"


def check_model_files(settings):
    """Stop with one clear line when a model file is missing or has the wrong size (an interrupted
    download or copy), instead of letting llama.cpp fail on it after a long load."""
    pins = pinned_files()
    for key in ("model", "mmproj"):
        path = model_path(settings, key)
        if not path.is_file():
            name, _ = active_profile(settings)
            how = ("Run Build GhostAgent.cmd to download it." if name == load_profiles()["default"] else
                   f'Run "Build GhostAgent.cmd" -Profile {name} to download it (model_profile is "{name}").')
            raise SystemExit(f"Missing model file: {path}\n  {how}")
        want = (pins.get(path.name) or (None, None))[0]
        size = path.stat().st_size
        if want and size < want:
            raise SystemExit(f"The model file is incomplete ({gb(size)} of {gb(want)} GB): {path.name}. "
                             "Run Build GhostAgent.cmd again.")
        if want and size != want:
            raise SystemExit(f"The model file {path.name} is not the expected one ({gb(size)} GB instead of "
                             f"{gb(want)} GB). Run Build GhostAgent.cmd again.")


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def memory_status():
    """{'total', 'avail', 'commit_avail'} in bytes (GlobalMemoryStatusEx), or None."""
    if os.name != "nt":
        return None
    m = _MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    try:
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None
    except Exception:
        return None
    return {"total": m.ullTotalPhys, "avail": m.ullAvailPhys, "commit_avail": m.ullAvailPageFile}


def ram_need_gb(settings):
    """GB of free RAM the model needs: its profile's "ram_gb" in tools\\profiles.json, else RAM_NEED_GB,
    else (a model no profile lists) its size plus 1 GB."""
    path = model_path(settings)
    pin = pinned_files().get(path.name)
    if pin and pin[1]:
        ram = (load_profiles()["profiles"].get(pin[1]) or {}).get("ram_gb")
        if isinstance(ram, (int, float)) and not isinstance(ram, bool) and 0 < ram < 10000:
            return math.ceil(ram)
    if pin and pin[1] in RAM_NEED_GB:
        return RAM_NEED_GB[pin[1]]
    try:
        return math.ceil(path.stat().st_size / 1e9) + 1
    except OSError:
        return RAM_NEED_GB["fast"]


def check_ram(settings, interactive=True):
    """Warn before loading when this PC clearly has too little memory for the model. On a PC
    with too little RAM in total, ask whether to go on (unattended runs only warn)."""
    m = memory_status()
    if not m:
        return
    need = ram_need_gb(settings)
    total = m["total"] / 2**30
    if total < need + RAM_HEADROOM_GB:
        warn(f"This PC has {round(total)} GB of RAM; the model needs about {need} GB free. "
             "It will be very slow or fail.")
        if interactive:
            try:
                answer = input("  Continue anyway? [y/N] ").strip().lower()
            except EOFError:
                answer = ""
            if answer not in ("y", "yes"):
                raise SystemExit(0)
    elif min(m["avail"], m["commit_avail"]) < need * 2**30:
        warn(f"Only {min(m['avail'], m['commit_avail']) / 2**30:.0f} GB of memory is free right now, and the model "
             f"needs about {need} GB. Close other programs, or loading may fail or be very slow.")
