"""The llama-server flags the agent may pass, and which of them the installed build knows."""

import json
import re
import subprocess
from pathlib import Path

from . import paths
from .winproc import CREATE_NO_WINDOW, LLAMA_SERVER_WHAT, blocked_message


# --------------------------------------------------------------------------------------------
# Which flags the installed llama-server knows. GhostAgent runs with the llama.cpp build its
# Build script installed (b11669, or b12359 and later); a flag an older build lacks is left out
# with a notice instead of stopping the server with "invalid argument".
# --------------------------------------------------------------------------------------------

# Without these the agent cannot work: a build that lacks one fails the build's smoke test.
REQUIRED_SERVER_FLAGS = ("-m", "--mmproj", "--alias", "--host", "--port", "-c", "-np", "-fa", "-ctk", "-ctv",
                         "--ctx-checkpoints", "--reasoning-format", "--load-mode", "--image-min-tokens", "-b", "-ub",
                         "-t", "--list-devices")
# Passed in some cases (a GPU layout, a profile, a setting), or added after b11669: left out, with a
# notice, when the build does not know them. tools\server-flags.json lists both groups.
OPTIONAL_SERVER_FLAGS = ("--no-reasoning-preserve", "--checkpoint-min-step", "--cache-ram", "--no-cache-idle-slots",
                         "--cors-origins", "--no-slots", "--no-webui", "--device", "-ngl", "--cpu-moe", "--fit",
                         "--moe-cache-mib", "--fit-target", "--moe-cache-cpu-share", "--image-max-tokens")
# Of those, the ones that lock the server down: leaving one out gets a warning, not just a notice
# ("lockdown" in tools\server-flags.json, which the build's smoke test reads too).
LOCKDOWN_SERVER_FLAGS = ("--cors-origins", "--no-slots", "--no-webui")
HELP_FLAG_RE = re.compile(r"(?<![\w-])(--?[A-Za-z][\w.-]*)")


def parse_help_flags(text):
    """Every flag in llama-server's --help text. Only the definition at the start of a line counts (the
    names before the description column), so a flag that is merely mentioned in a note does not."""
    flags = set()
    for line in text.splitlines():
        if not line.startswith("-"):
            continue
        definition = re.split(r"(?<=[^,\s])\s{2,}", line.rstrip(), maxsplit=1)[0]
        flags.update(HELP_FLAG_RE.findall(definition))
    return flags


def build_fingerprint(exe):
    """Size and modification time of llama-server.exe and the DLLs that hold its code (the .exe itself
    is a small launcher that is the same in every release)."""
    parts = []
    for p in (exe, exe.with_name("llama-server-impl.dll"), exe.with_name("llama-common.dll")):
        try:
            st = p.stat()
            parts.append(f"{p.name}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            pass
    return "|".join(parts)


def server_flags(exe):
    """The flags this llama-server build knows, from its --help, or None when they cannot be read (then
    every flag is passed, as before). The answer is kept in logs\\server-flags-cache.json for as long
    as the build's files are unchanged, so --help runs once per build."""
    if not Path(exe).is_file():
        return None
    cache_file = paths.LOGS / "server-flags-cache.json"
    key = exe.parent.name
    fingerprint = build_fingerprint(exe)
    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
        if not isinstance(cache, dict):
            cache = {}
    except (OSError, ValueError):
        cache = {}
    entry = cache.get(key)
    if isinstance(entry, dict) and entry.get("fingerprint") == fingerprint and entry.get("flags"):
        return set(entry["flags"])
    try:
        out = subprocess.run([str(exe), "--help"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                             timeout=60, creationflags=CREATE_NO_WINDOW, cwd=str(exe.parent))
    except OSError as e:
        msg = blocked_message(e, LLAMA_SERVER_WHAT)
        if msg:
            raise SystemExit(msg) from None
        return None
    except Exception:
        return None
    flags = parse_help_flags((out.stdout or "") + (out.stderr or ""))
    if "--port" not in flags or "-m" not in flags:
        return None  # not a help text this code understands
    cache[key] = {"fingerprint": fingerprint, "flags": sorted(flags)}
    try:
        paths.LOGS.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    except OSError:
        pass
    return flags


def keep_supported_flags(groups, supported):
    """(groups the build knows, optional flags left out, required flags it does not list). groups is
    [(flag, [values])]; supported None means unknown, and everything is kept."""
    if supported is None:
        return list(groups), [], []
    kept, dropped, missing = [], [], []
    for flag, values in groups:
        if flag in supported:
            kept.append((flag, values))
        elif flag in OPTIONAL_SERVER_FLAGS:
            dropped.append(flag)
        else:
            kept.append((flag, values))
            missing.append(flag)
    return kept, dropped, missing


def flat_args(groups):
    return [str(x) for flag, values in groups for x in (flag, *values)]
