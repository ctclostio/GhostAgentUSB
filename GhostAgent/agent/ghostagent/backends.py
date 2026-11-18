"""Which llama.cpp build (CUDA 13, CUDA 12, Vulkan or CPU) and which GPU the model server uses."""

import ctypes
import itertools
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from . import paths
from .console import info, warn
from .paths import BIN
from .settings import BACKENDS
from .winproc import (CREATE_NO_WINDOW, LLAMA_SERVER_WHAT, blocked_message, dll_not_found, vc_runtime_advice,
                      vc_runtime_missing)


# --------------------------------------------------------------------------------------------
# Which llama.cpp build and which GPU
# --------------------------------------------------------------------------------------------

DEVICE_RE = re.compile(r"^\s*((?:CUDA|Vulkan|ROCm|SYCL)\d+):\s*(.+?)\s*\((\d+) MiB,\s*(\d+) MiB free\)", re.M)

# Graphics adapters with less dedicated memory than this share the PC's RAM: integrated GPUs.
INTEGRATED_MAX_MIB = 512
_DXGI_CACHE = None


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
                ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint), ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                ("LuidLow", ctypes.c_ulong), ("LuidHigh", ctypes.c_long), ("Flags", ctypes.c_uint)]


def dxgi_adapters():
    """[{name, vendor, dedicated_mib}] for every graphics adapter Windows knows (DXGI), without the
    software renderer. Only reads what the driver reports: no device is created. [] when unavailable."""
    global _DXGI_CACHE
    if _DXGI_CACHE is not None:
        return _DXGI_CACHE
    found = []
    if os.name == "nt":
        try:
            class GUID(ctypes.Structure):
                _fields_ = [("a", ctypes.c_ulong), ("b", ctypes.c_ushort), ("c", ctypes.c_ushort), ("d", ctypes.c_ubyte * 8)]

            def method(obj, index, *argtypes):
                table = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
                return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(table[index])

            iid = GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
            factory = ctypes.c_void_p()
            if ctypes.WinDLL("dxgi").CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory)) == 0 and factory:
                try:
                    for i in range(16):
                        adapter = ctypes.c_void_p()
                        # IDXGIFactory1::EnumAdapters1 (12), IDXGIAdapter1::GetDesc1 (10), IUnknown::Release (2)
                        if method(factory, 12, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(factory, i, ctypes.byref(adapter)) != 0:
                            break
                        desc = _DXGI_ADAPTER_DESC1()
                        if method(adapter, 10, ctypes.POINTER(_DXGI_ADAPTER_DESC1))(adapter, ctypes.byref(desc)) == 0 \
                                and not desc.Flags & 2:  # DXGI_ADAPTER_FLAG_SOFTWARE
                            found.append({"name": desc.Description, "vendor": desc.VendorId,
                                          "dedicated_mib": desc.DedicatedVideoMemory // 2**20})
                        method(adapter, 2)(adapter)
                finally:
                    method(factory, 2)(factory)
        except Exception:
            found = []
    _DXGI_CACHE = found
    return found


def _gpu_name_key(name):
    return " ".join(re.sub(r"\((?:r|tm)\)|[^a-z0-9]+", " ", str(name).lower()).split())


def is_integrated(device, adapters=None):
    """True for a GPU that shares the PC's RAM. Decided by the dedicated memory Windows reports for the
    adapter (below 512 MiB), not by its name: names such as Arc or Radeon are used for both kinds.
    Fallback for an adapter DXGI does not list: llama.cpp reports all of its memory as free, which only
    shared memory does. CUDA builds list NVIDIA cards only, which are discrete."""
    if device["id"].startswith("CUDA"):
        return False
    key = _gpu_name_key(device["name"])
    for a in (dxgi_adapters() if adapters is None else adapters):
        if _gpu_name_key(a["name"]) == key:
            return a["dedicated_mib"] < INTEGRATED_MAX_MIB
    return int(device["free"]) >= int(device["total"])


def list_devices(exe):
    """The GPUs a llama.cpp build can use ([] when none, or when the build does not run here). Stops
    with a plain message when Windows blocks the program, instead of quietly falling back to the CPU."""
    try:
        out = subprocess.run([str(exe), "--list-devices"], capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=120, creationflags=CREATE_NO_WINDOW, cwd=str(exe.parent))
    except OSError as e:
        msg = blocked_message(e, LLAMA_SERVER_WHAT)
        if msg:
            raise SystemExit(msg) from None
        return []
    except Exception:
        return []
    if dll_not_found(out.returncode):
        missing = vc_runtime_missing(exe.parent)
        if missing:
            raise SystemExit(vc_runtime_advice(missing))  # no build can run: each needs the runtime
        return []  # another DLL of this build (a CUDA one, say): the next build may still work
    text = (out.stdout or "") + (out.stderr or "")
    devices = []
    for m in DEVICE_RE.finditer(text):
        dev = {"id": m[1], "name": m[2], "total": int(m[3]), "free": int(m[4])}
        dev["integrated"] = is_integrated(dev)
        devices.append(dev)
    return devices


_NVIDIA_SMI_CACHE = None


def nvidia_smi_gpus():
    """[{name, compute_cap, driver, driver_major}] from nvidia-smi, [] without an NVIDIA driver. Read
    once per run; nvidia-smi only reads what the driver reports."""
    global _NVIDIA_SMI_CACHE
    if _NVIDIA_SMI_CACHE is not None:
        return _NVIDIA_SMI_CACHE
    gpus = []
    exe = shutil.which("nvidia-smi") or str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe")
    if Path(exe).is_file():
        try:
            out = subprocess.run([exe, "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
                                 creationflags=CREATE_NO_WINDOW)
            gpus = parse_nvidia_smi(out.stdout or "")
        except (OSError, subprocess.SubprocessError, ValueError):
            gpus = []
    _NVIDIA_SMI_CACHE = gpus
    return gpus


def parse_nvidia_smi(text):
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        try:
            cap = float(parts[1])
        except ValueError:
            cap = None  # "[N/A]" on very old drivers
        m = re.match(r"(\d+)", parts[2])
        gpus.append({"name": parts[0], "compute_cap": cap, "driver": parts[2], "driver_major": int(m[1]) if m else None})
    return gpus


# The cuda13 build (CUDA 13.x) needs a driver from the 580 series on, and has GPU code for compute
# capability 7.5 (Turing) and newer only. llama.cpp's --list-devices still lists older cards, which
# then crash while loading, so they go straight to the cuda12 build.
CUDA13_MIN_DRIVER = 580
CUDA13_MIN_COMPUTE = 7.5


def cuda13_unsuitable(gpus):
    """Why the cuda13 build cannot work with these NVIDIA GPUs, or None."""
    if not gpus:
        return None
    caps = [g["compute_cap"] for g in gpus if g["compute_cap"] is not None]
    if caps and max(caps) < CUDA13_MIN_COMPUTE:
        return f"the GPU's compute capability {max(caps)} is below {CUDA13_MIN_COMPUTE}"
    drivers = [g["driver_major"] for g in gpus if g["driver_major"] is not None]
    if drivers and max(drivers) < CUDA13_MIN_DRIVER:
        return f"the NVIDIA driver {max(g['driver'] for g in gpus)} is older than {CUDA13_MIN_DRIVER}"
    return None


def nvidia_gpu_present():
    """True when this PC has an NVIDIA graphics adapter, with or without a working driver (DXGI, then
    the PCI display devices Windows has recorded)."""
    if any(a["vendor"] == 0x10DE for a in dxgi_adapters()):
        return True
    if os.name != "nt":
        return False
    import winreg
    display = "{4d36e968-e325-11ce-bfc1-08002be10318}"
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Enum\PCI") as pci:
            for i in itertools.count():
                try:
                    dev = winreg.EnumKey(pci, i)
                except OSError:
                    break
                if not dev.upper().startswith("VEN_10DE"):
                    continue
                with winreg.OpenKey(pci, dev) as k:
                    for j in itertools.count():
                        try:
                            inst = winreg.EnumKey(k, j)
                        except OSError:
                            break
                        try:
                            with winreg.OpenKey(k, inst) as ik:
                                if str(winreg.QueryValueEx(ik, "ClassGUID")[0]).lower() == display:
                                    return True
                        except OSError:
                            pass
    except OSError:
        pass
    return False


def backend_cache_file():
    """logs\\backend-cache.json: builds that could not run the model with a given GPU and driver."""
    return paths.LOGS / "backend-cache.json"


def gpu_key(device, gpus=None):
    """'GPU name | driver version': what a known-bad build is remembered for. A new driver is a new key."""
    driver = next((g["driver"] for g in (gpus or []) if g["name"] == device["name"]), None)
    return f"{device['name']} | driver {driver}" if driver else device["name"]


def known_bad_backends():
    try:
        data = json.loads(backend_cache_file().read_text(encoding="utf-8"))
        return {k: list(v) for k, v in data.get("bad", {}).items() if isinstance(v, list)}
    except (OSError, ValueError, AttributeError):
        return {}


def remember_bad_backend(key, backend):
    bad = known_bad_backends()
    if backend in bad.get(key, []):
        return
    bad.setdefault(key, []).append(backend)
    doc = {"_note": "llama.cpp builds that could not run the model with this GPU and driver; backend \"auto\" "
                    "skips them. Delete this file to try them again.", "bad": bad}
    try:
        paths.LOGS.mkdir(parents=True, exist_ok=True)
        backend_cache_file().write_text(json.dumps(doc, indent=2), encoding="utf-8")
    except OSError:
        pass


def installed_backends():
    """The llama.cpp builds on this drive, in BACKENDS order (an ARM64 drive has no cuda12 build)."""
    return [name for name in BACKENDS if (BIN / name / "llama-server.exe").is_file()]


def backend_is_auto(settings):
    """True when the builds are tried in BACKENDS order, each one falling back to the next: backend
    "auto", or a forced GPU build that is not on this drive. (A forced cpu build that is missing is
    still CPU only: see pick_backend.)"""
    wanted = settings.get("backend", "auto")
    return wanted == "auto" or (wanted != "cpu" and wanted not in installed_backends())


def pick_backend(settings, after=None):
    """(backend name, exe path, device or None). With backend "auto" the builds are tried in BACKENDS
    order: the first one that sees a usable GPU wins. after: a build that failed to load the model;
    the search goes on with the builds behind it. A build that is not installed is passed over (an
    ARM64 drive has no cuda12 build); a forced GPU build that is missing is treated as "auto"
    (backend_is_auto), and a forced cpu build that is missing is replaced by a GPU build without a GPU."""
    installed = installed_backends()
    if not installed:
        raise SystemExit(f"No llama.cpp build was found in {BIN} (no llama-server.exe). Run Build GhostAgent.cmd again.")
    wanted = settings["backend"]
    auto = backend_is_auto(settings)
    if wanted == "cpu" and "cpu" not in installed:
        # CPU only was asked for: every llama.cpp build has the CPU backend, and "--device none" keeps
        # the GPU out of it (LlamaServer._start).
        if after is None:
            warn(f"There is no cpu build on this drive, so the {installed[-1]} build runs the model on the CPU "
                 "(backend setting).")
        return installed[-1], BIN / installed[-1] / "llama-server.exe", None
    if auto and wanted != "auto" and after is None:
        warn(f"The {wanted} build (backend setting) is not on this drive, so the builds are tried in the usual order: "
             f"{', '.join(installed)}.")
    order = list(BACKENDS) if auto else [wanted]
    if after in order:
        order = order[order.index(after) + 1:]
    gpus = nvidia_smi_gpus() if auto and any(b.startswith("cuda") for b in order) else []
    bad = known_bad_backends() if auto else {}
    cuda_device_seen = False
    for name in order:
        exe = BIN / name / "llama-server.exe"
        if name not in installed:
            continue
        if name == "cpu":
            break
        if auto and name == "cuda13":
            why = cuda13_unsuitable(gpus)
            if why:
                info(f"skipping the cuda13 build: {why}")
                continue
        info(f"checking {name} build ...")
        devices = [d for d in list_devices(exe) if not d["integrated"]]
        if devices:
            best = max(devices, key=lambda d: d["total"])
            cuda_device_seen |= name.startswith("cuda")
            if auto and name in bad.get(gpu_key(best, gpus), []):
                info(f"skipping the {name} build: it could not run the model on this GPU before "
                     "(delete logs\\backend-cache.json to try it again)")
                continue
            return name, exe, best
        if not auto:
            warn(f"The {name} build found no usable GPU; it will run on the CPU.")
            return name, exe, None
    cuda = [b for b in order if b.startswith("cuda") and b in installed]
    if auto and not cuda_device_seen and cuda and nvidia_gpu_present():
        needs = {"cuda13": f"the cuda13 build needs version {CUDA13_MIN_DRIVER} or newer",
                 "cuda12": "the cuda12 build needs version 550 or newer"}  # (only the builds on this drive)
        warn("This PC has an NVIDIA graphics card, but llama.cpp's CUDA builds cannot use it. Update the NVIDIA "
             f"driver: {', '.join(needs[b] for b in cuda)}.")
    if "cpu" in installed:
        return "cpu", BIN / "cpu" / "llama-server.exe", None
    # No CPU build: the last GPU build runs on the CPU just as well (every llama.cpp build has the CPU backend).
    return installed[-1], BIN / installed[-1] / "llama-server.exe", None
