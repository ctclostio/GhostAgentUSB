"""Child processes and Windows: everything the agent starts dies with it, one agent at a time,
programs Windows refuses to run, crash dialogs, sleep, Quick Edit, and how a process is doing."""

import contextlib
import ctypes
import os
import struct
import sys
from pathlib import Path


# --------------------------------------------------------------------------------------------
# Child-process lifetime: everything we start dies with this process, even if the window is
# closed with the X button.
# --------------------------------------------------------------------------------------------

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

_JOB = None
_MUTEX = None


def already_running():
    """True if another copy of the agent is open (a second copy would load the model twice)."""
    global _MUTEX
    if os.name != "nt":
        return False
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = ctypes.c_void_p
    _MUTEX = k32.CreateMutexW(None, False, "Local\\GhostAgent")
    return ctypes.get_last_error() == 183  # ERROR_ALREADY_EXISTS


def bind_children_to_this_process():
    global _JOB
    if os.name != "nt":
        return
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC_LIMIT(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BASIC_LIMIT),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.GetCurrentProcess.restype = wintypes.HANDLE

    job = k32.CreateJobObjectW(None, None)
    if not job:
        return
    limits = EXTENDED_LIMIT()
    # KILL_ON_JOB_CLOSE, plus BREAKAWAY_OK so the user's Chrome can be started outside the job
    # (see start_chrome): it must stay open when the agent exits.
    limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x800
    k32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits))
    if k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        _JOB = job


# --------------------------------------------------------------------------------------------
# Windows: programs it refuses to run, error dialogs, sleep, and how a process is doing
# --------------------------------------------------------------------------------------------

LLAMA_SERVER_WHAT = "llama-server.exe, an unsigned program from the official llama.cpp release"


def blocked_message(e, what):
    """A plain sentence when Windows refused to run something (App Control, a policy, Defender), else
    None. `what` names it, for example LLAMA_SERVER_WHAT. Never a hint on getting around a rule: that
    is for whoever manages the PC to decide."""
    code = getattr(e, "winerror", None)
    if code is None:
        # A blocked DLL inside an import arrives as an ImportError that carries Windows' own wording.
        text = str(e).lower()
        for words, number in (("application control policy", 4551), ("blocked by group policy", 1260),
                              ("contains a virus", 225), ("access is denied", 5)):
            if words in text:
                code = number
                break
    if code is None:
        return None
    if 4550 <= code <= 4560:
        return f"Windows App Control (Smart App Control or your organization's policy) blocked {what}."
    if code == 1260:
        return f"Windows blocked {what}: it is blocked by group policy on this PC."
    if code == 225:
        return f"Microsoft Defender flagged {what} and blocked it."
    if code == 5:
        return f"Windows refused to start {what} (an organization rule may block unsigned programs on USB drives)."
    return None


# llama.cpp's Windows builds need Microsoft's Visual C++ 2015-2022 runtime, which is not on the stick
# (Microsoft's license lets only its own installer put it on a PC). Most PCs have it in System32.
VC_RUNTIME_DLLS = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")
VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
VC_REDIST_URL_ARM64 = "https://aka.ms/vs/17/release/vc_redist.arm64.exe"  # for a drive built with ARM64 parts
STATUS_DLL_NOT_FOUND = 0xC0000135  # how a program ends when Windows cannot find a DLL it needs


def vc_redist():
    """(name, download address) of the Visual C++ Redistributable this drive's programs need. The build
    puts one architecture on a drive, Python and llama.cpp alike, so this Python's tells: an ARM64 drive
    needs the ARM64 one, and an x64 drive needs the x64 one even on an ARM64 PC (where it runs emulated)."""
    if "ARM64" in sys.version.upper():
        return "ARM64", VC_REDIST_URL_ARM64
    return "x64", VC_REDIST_URL


def vc_runtime_missing(folder):
    """The Visual C++ runtime DLLs that are neither in System32 nor in `folder` (next to llama-server.exe),
    by file. [] when all are there (or not on Windows)."""
    if os.name != "nt":
        return []
    system32 = Path(os.environ.get("SystemRoot") or r"C:\Windows") / "System32"
    return [dll for dll in VC_RUNTIME_DLLS if not (system32 / dll).is_file() and not (Path(folder) / dll).is_file()]


def vc_runtime_advice(missing):
    arch, url = vc_redist()
    return ("llama-server.exe cannot start without the Microsoft Visual C++ 2015-2022 runtime, and this PC lacks "
            f"{', '.join(missing) if missing else 'part of it'}. Install the \"Microsoft Visual C++ Redistributable "
            f"({arch})\" from {url} (it needs administrator rights), then start GhostAgent again.")


def dll_not_found(returncode):
    """True when a process ended because Windows could not find a DLL it needs (STATUS_DLL_NOT_FOUND)."""
    return returncode is not None and int(returncode) & 0xFFFFFFFF == STATUS_DLL_NOT_FOUND


def quiet_crash_dialogs():
    """No Windows Error Reporting or 'missing DLL' dialog for this process and the ones it starts (they
    inherit the mode). Such a dialog would keep a crashed llama-server alive and waiting for a click
    nobody sees, so the agent would never notice the crash."""
    if os.name == "nt":
        try:
            ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)  # FAILCRITICALERRORS | NOGPFAULTERRORBOX | NOOPENFILEERRORBOX
        except Exception:
            pass


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextlib.contextmanager
def keep_pc_awake():
    """Keep Windows from going to sleep while a task runs or the model copies: the agent's clicks and
    typing go through the browser, not the keyboard and mouse, so they do not count as use of the PC.
    The request belongs to the calling thread and ends with the block. The display may still turn off."""
    k32 = None
    if os.name == "nt":
        try:
            k32 = ctypes.windll.kernel32
            k32.SetThreadExecutionState.restype = ctypes.c_uint
            k32.SetThreadExecutionState.argtypes = [ctypes.c_uint]
            k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        except Exception:
            k32 = None
    try:
        yield
    finally:
        if k32 is not None:
            try:
                k32.SetThreadExecutionState(ES_CONTINUOUS)
            except Exception:
                pass


def disable_quick_edit():
    """In a classic console window, selecting text with the mouse (QuickEdit) pauses any program that
    prints, so one stray click would freeze a task in the middle of a step. QuickEdit is turned off for
    this window only; right-click > Mark still selects text, and /copy copies the last answer.
    Returns True when the console mode was changed."""
    if os.name != "nt":
        return False
    try:
        k32 = ctypes.windll.kernel32
        k32.GetStdHandle.restype = ctypes.c_void_p
        handle = k32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        if not handle or handle == ctypes.c_void_p(-1).value:
            return False
        mode = ctypes.c_ulong()
        if not k32.GetConsoleMode(ctypes.c_void_p(handle), ctypes.byref(mode)):
            return False  # not a console (redirected input)
        new = (mode.value & ~0x0040) | 0x0080  # ENABLE_QUICK_EDIT_MODE off, ENABLE_EXTENDED_FLAGS on
        if new == mode.value or not k32.SetConsoleMode(ctypes.c_void_p(handle), new):
            return False
        global _SAVED_CONSOLE_MODE
        _SAVED_CONSOLE_MODE = (handle, mode.value)
        return True
    except Exception:
        return False


_SAVED_CONSOLE_MODE = None  # (input handle, mode) from before disable_quick_edit


def restore_quick_edit():
    """Put the console's input mode back as it was before disable_quick_edit, when the agent ends: the
    console window may belong to a command prompt the user keeps using."""
    global _SAVED_CONSOLE_MODE
    saved, _SAVED_CONSOLE_MODE = _SAVED_CONSOLE_MODE, None
    if saved is None or os.name != "nt":
        return
    try:
        ctypes.windll.kernel32.SetConsoleMode(ctypes.c_void_p(saved[0]), saved[1])
    except Exception:
        pass


def has_console():
    """This process has a console (possibly without a window) that a child process can share."""
    if os.name != "nt":
        return False
    try:
        return bool(ctypes.windll.kernel32.GetConsoleCP())
    except Exception:
        return False


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


def process_read_bytes(proc):
    """Bytes a child process has read from files so far (GetProcessIoCounters), or None. llama-server
    reads the model with plain file reads (--load-mode none), so this is how far loading has got."""
    handle = getattr(proc, "_handle", None)
    if os.name != "nt" or not handle:
        return None
    try:
        counters = _IO_COUNTERS()
        k32 = ctypes.windll.kernel32
        k32.GetProcessIoCounters.argtypes = [ctypes.c_void_p, ctypes.POINTER(_IO_COUNTERS)]
        if not k32.GetProcessIoCounters(ctypes.c_void_p(int(handle)), ctypes.byref(counters)):
            return None
        return int(counters.ReadTransferCount)
    except Exception:
        return None


def process_cpu_seconds(proc):
    """CPU time a child process has used (GetProcessTimes), or None."""
    handle = getattr(proc, "_handle", None)
    if os.name != "nt" or not handle:
        return None
    try:
        times = [ctypes.c_ulonglong() for _ in range(4)]
        k32 = ctypes.windll.kernel32
        k32.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_ulonglong)] * 4
        if not k32.GetProcessTimes(ctypes.c_void_p(int(handle)), *(ctypes.byref(t) for t in times)):
            return None
        return (times[2].value + times[3].value) / 1e7  # kernel + user, in 100 ns units
    except Exception:
        return None


def performance_cores():
    """Physical performance cores (P-cores) on a CPU that mixes core types (Intel 12th generation and
    later), or None when all cores are alike or Windows cannot say. Windows gives every core an
    efficiency class; the fastest cores have the highest one."""
    if os.name != "nt":
        return None
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetLogicalProcessorInformationEx.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        size = ctypes.c_ulong(0)
        k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))  # 0 = RelationProcessorCore
        if not size.value:
            return None
        buf = ctypes.create_string_buffer(size.value)
        if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
            return None
        return p_cores_from(buf.raw[:size.value])
    except Exception:
        return None


def p_cores_from(raw):
    """performance_cores() for a SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX buffer (one record per core:
    Relationship, Size, then PROCESSOR_RELATIONSHIP with Flags and EfficiencyClass)."""
    classes, off = [], 0
    while off + 10 <= len(raw):
        relationship, length = struct.unpack_from("<II", raw, off)
        if length <= 0:
            break
        if relationship == 0:
            classes.append(raw[off + 9])
        off += length
    if len(set(classes)) < 2:
        return None
    top = max(classes)
    return sum(1 for c in classes if c == top)


def default_threads():
    """CPU threads for llama-server when the threads setting is 0: on a CPU with performance and
    efficiency cores, one per P-core (at most 8), since a token waits for its slowest thread; otherwise
    8, or fewer on a small CPU."""
    p = performance_cores()
    if p:
        return min(p, 8)
    return min(8, os.cpu_count() or 8)
