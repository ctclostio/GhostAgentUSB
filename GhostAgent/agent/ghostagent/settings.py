"""Settings: the defaults, settings.default.json under settings.json, the model profiles in
tools\\profiles.json, and checking every value."""

import copy
import difflib
import json
import math
import os
import re
import shlex
import shutil
from datetime import datetime

from . import paths
from .console import err, info, warn
from .paths import DEFAULTS_FILE, PROFILES_FILE


# --------------------------------------------------------------------------------------------
# Settings: settings.default.json (every setting, with notes) under settings.json (the user's
# changes). Every value is checked: a bad one is reported and replaced by its default alone.
# --------------------------------------------------------------------------------------------

DEFAULT_SETTINGS = {
    # The model: a profile from tools\profiles.json. "model" and "mmproj" empty = that profile's files.
    "model_profile": "fast",
    "model": "",
    "mmproj": "",
    "reasoning_effort": "",  # accurate profile only; "" = its template's default (xhigh)
    "backend": "auto",
    "port": 18931,
    "ctx_size": 0,  # 0 = the profile's context size
    "threads": 0,
    "moe_cache_mib": "auto",
    "moe_cache_cap_mib": 4096,
    "fit_target_mib": 1024,
    # llama-server's prompt cache (see LlamaServer.start for why these matter on Holo4).
    "ctx_checkpoints": 8,
    "checkpoint_min_step": 1536,
    "cache_ram_mib": 0,
    # Tuning knobs for benchmarks; empty = llama.cpp's own default.
    "moe_cache_cpu_share": "",
    "op_offload_min_batch": "",
    "graceful_server_stop": False,
    "extra_server_args": [],
    "external_server": "",
    "start_url": "https://www.bing.com",
    "viewport_width": 1280,
    "viewport_height": 800,
    # Page area in whole 32-pixel tiles, at least 1024 of them (no resampling); measured no gain, so off.
    "snap_viewport": False,
    "refit_window_each_task": True,
    "max_steps": 0,  # 0 = no limit
    "progress_checkin_steps": 0,  # 0 = never
    "image_budget": 3,
    # "block" (several screenshots go at once, down to image_keep_after_evict, so the prompt only grows
    # in between) or "rolling" (the oldest goes every step). block/1 measured faster at equal success.
    "image_eviction": "block",
    "image_keep_after_evict": 1,
    # "classic" or "header" (scroll position and page extent); header measured slower, no more accurate.
    "observation_format": "classic",
    # The instructions ("h", after H Company's Holo4 guidance, or the short "classic") and the tools'
    # names and arguments ("h", H Company's Holo4 web tools, or "classic"). h measured faster, same successes.
    "prompt_version": "h",
    "tool_schema": "h",
    "write_enter_default": False,
    "temperature": 0.8,
    "top_k": 20,
    "top_p": 0.95,
    "min_p": 0.0,
    "max_tokens": 4096,  # 0 = no limit
    "seed": -1,  # -1 = random
    "thinking": True,
    # The model's earlier reasoning in its context: "auto" = per model profile (off for the fast
    # profiles until a benchmark decides, on for accurate), or true / false.
    "preserve_reasoning": "auto",
    "settle_seconds": 1.0,
    # "adaptive" (per action, ends once the page is quiet) or "fixed" (settle_seconds after every action).
    "settle_mode": "adaptive",
    "browser": "chrome",
    "save_screenshots": False,
    "downloads": "save",
    "download_max_mb": 500,
    "copy_model_to_pc": "ask",
    "step_trace": True,
    "trace_messages": False,
    # Safety rails outside the model (see the notes in settings.default.json).
    "dialog_policy": "ask",
    "confirm_risky_actions": True,
    "allowed_hosts": [],
    "allow_file_urls": False,
    "allow_internal_pages": False,
    "allow_unsandboxed_browser": False,
    "profile_on_stick": True,
    "web_ui": False,
    "external_server_key": "",
}

# Tried in this order when "backend" is "auto". The first build that sees a usable GPU wins.
BACKENDS = ["cuda13", "cuda12", "vulkan", "cpu"]

# What settings.json shipped with before the split into two files. A full copy of the old file is
# turned into a list of the user's changes by comparing against these values (not against the
# current defaults), so a default that changes later still reaches users who never touched it.
LEGACY_DEFAULTS = {
    "copy_model_to_pc": False, "start_url": "https://www.bing.com", "max_steps": 0, "image_budget": 3,
    "temperature": 0.8, "thinking": True, "settle_seconds": 1.0, "viewport_width": 1280,
    "viewport_height": 800, "browser": "chrome", "save_screenshots": False, "backend": "auto",
    "moe_cache_mib": "auto", "fit_target_mib": 1024, "threads": 0, "ctx_size": 65536, "port": 18931,
    "extra_server_args": [], "model": "models/Holo4-35B-A3B-Q4_K_M.gguf",
    "mmproj": "models/Holo4-35B-A3B-mmproj-f16.gguf",
}
LEGACY_NOTE_KEYS = {"_copy_model_to_pc", "_max_steps", "_image_budget", "_viewport", "_browser",
                    "_save_screenshots", "_backend", "_moe_cache_mib", "_threads"}
USER_SETTINGS_NOTE = ("Your own settings: only the ones you changed. Every setting, with a note on what it "
                      "does, is in settings.default.json (updates replace that file, so do not edit it). "
                      "To change a setting, copy its line from there to here, change the value, save, "
                      "and restart the agent. Keys starting with _ are notes and are ignored.")

TRUE_WORDS = {"true", "1", "yes", "on"}
FALSE_WORDS = {"false", "0", "no", "off"}

# Allowed values. Rules for settings that do not exist (yet) are simply not used.
SETTING_CHOICES = {
    "backend": ["auto"] + BACKENDS,
    "browser": ["chrome", "user_edge", "auto", "chrome-agent", "bundled", "edge"],
    "reasoning_effort": ["", "xhigh", "medium", "low"],
    "dialog_policy": ["ask", "dismiss", "accept"],
    "image_eviction": ["rolling", "block"],
    "observation_format": ["classic", "header"],
    "prompt_version": ["classic", "h"],
    "tool_schema": ["classic", "h"],
    "settle_mode": ["fixed", "adaptive"],
    "downloads": ["save", "off"],
}
SETTING_MIN = {
    "port": 1, "ctx_size": 0, "threads": 0, "fit_target_mib": 0, "viewport_width": 320,
    "viewport_height": 240, "max_steps": 0, "progress_checkin_steps": 0, "download_max_mb": 1,
    "image_budget": 1, "temperature": 0.0, "top_k": 0,
    "top_p": 0.0, "min_p": 0.0, "max_tokens": 0, "seed": -1, "settle_seconds": 0.0,
    "checkpoint_min_step": 0, "ctx_checkpoints": 0, "cache_ram_mib": 0, "image_keep_after_evict": 1,
    "moe_cache_mib": 0, "moe_cache_cap_mib": 0, "moe_cache_cpu_share": 0.0, "op_offload_min_batch": 1,
}
SETTING_MAX = {"port": 65535, "top_p": 1.0, "min_p": 1.0, "settle_seconds": 60.0, "moe_cache_cpu_share": 1.0}
# llama.cpp keeps the sampling seed in 32 bits (and -1 means a random one), so a bigger seed would be
# cut down to another number without a word.
SEED_MAX = 2**32 - 1


class BadSetting(ValueError):
    pass


# The fast profile, for a drive that lost tools\profiles.json: enough to run the model the build
# installs by default. The file is the real source (shared with the build).
FALLBACK_PROFILES = {"default": "fast", "profiles": {"fast": {
    "label": "Holo4-35B-A3B Q4_K_M (H Company)", "arch": "moe", "ctx_size": 65536, "alias": "holo4-35b-a3b",
    "model": {"name": "Holo4-35B-A3B-Q4_K_M.gguf"}, "mmproj": {"name": "Holo4-35B-A3B-mmproj-f16.gguf"}}}}
_PROFILES_CACHE = {}


def load_profiles():
    """tools\\profiles.json as {"default": name, "profiles": {name: profile}}, read once per change of
    the file. A missing or broken file gives FALLBACK_PROFILES."""
    try:
        stamp = (str(PROFILES_FILE), PROFILES_FILE.stat().st_mtime_ns)
    except OSError:
        return FALLBACK_PROFILES
    if stamp not in _PROFILES_CACHE:
        doc = None
        try:
            raw = json.loads(PROFILES_FILE.read_text(encoding="utf-8-sig"))
            profiles = {k: v for k, v in (raw.get("profiles") or {}).items() if isinstance(v, dict)}
            if profiles:
                default = raw.get("default") if raw.get("default") in profiles else sorted(profiles)[0]
                doc = {"default": default, "profiles": profiles}
        except (OSError, ValueError, AttributeError, TypeError):
            pass
        _PROFILES_CACHE.clear()
        _PROFILES_CACHE[stamp] = doc or FALLBACK_PROFILES
    return _PROFILES_CACHE[stamp]


def profile_names():
    """Model profile names from tools\\profiles.json."""
    return sorted(load_profiles()["profiles"])


def coerce_setting(key, value, default):
    """The value converted to the type of its default, or BadSetting saying what was expected."""
    if key == "moe_cache_mib":
        if value in (None, "") or (isinstance(value, str) and value.strip().lower() == "auto"):
            return "auto"
        if isinstance(value, str) and value.strip().lower() in ("off", "none"):
            return 0
        try:
            return coerce_setting("", value, 0)  # otherwise a whole number of MiB
        except BadSetting:
            raise BadSetting('"auto" or a whole number of MiB (0 = off)') from None
    if key == "copy_model_to_pc":
        if value is None or (isinstance(value, str) and value.strip().lower() in ("", "ask", "auto")):
            return "ask"
        try:
            return coerce_setting("", value, False)
        except BadSetting:
            raise BadSetting('"ask", true or false') from None
    if key == "preserve_reasoning":
        if value is None or (isinstance(value, str) and value.strip().lower() in ("", "auto")):
            return "auto"
        try:
            return coerce_setting("", value, False)
        except BadSetting:
            raise BadSetting('"auto", true or false') from None
    if key in ("moe_cache_cpu_share", "op_offload_min_batch"):
        # Empty means "not passed": llama.cpp then uses its own default.
        if value is None or (isinstance(value, str) and not value.strip()):
            return ""
        whole = key == "op_offload_min_batch"
        try:
            return coerce_setting("", value, 0 if whole else 0.0)
        except BadSetting:
            raise BadSetting("empty, or " + ("a whole number" if whole else "a number from 0 to 1")) from None
    if key == "external_server":
        value = "" if value is None else value
        if not isinstance(value, str):
            raise BadSetting("an address such as http://127.0.0.1:8080, or empty")
        value = value.strip().rstrip("/")
        if value and not re.match(r"^https?://[^/\s]+$", value, re.I):
            raise BadSetting("an address such as http://127.0.0.1:8080, or empty")
        return value
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in TRUE_WORDS | FALSE_WORDS:
            return value.strip().lower() in TRUE_WORDS
        raise BadSetting("true or false")
    if isinstance(default, int):
        if isinstance(value, bool):
            raise BadSetting("a whole number")
        if isinstance(value, int):
            return value
        try:
            number = float(value.strip()) if isinstance(value, str) else float(value)
        except (TypeError, ValueError):
            raise BadSetting("a whole number") from None
        if not number.is_integer():
            raise BadSetting("a whole number")
        return int(number)
    if isinstance(default, float):
        if isinstance(value, bool):
            raise BadSetting("a number")
        try:
            number = float(value.strip()) if isinstance(value, str) else float(value)
        except (TypeError, ValueError):
            raise BadSetting("a number") from None
        if not math.isfinite(number):
            raise BadSetting("a number")
        return number
    if isinstance(default, list):
        if isinstance(value, str):
            # Windows rules: backslashes in paths stay as they are; quotes only group the words.
            return [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t
                    for t in shlex.split(value, posix=False)]
        if isinstance(value, list) and all(isinstance(v, (str, int, float)) and not isinstance(v, bool)
                                           for v in value):
            return [str(v) for v in value]
        if key == "allowed_hosts":
            raise BadSetting("a list such as [\"192.168.1.1\", \"nas.local\"]")
        raise BadSetting("a list such as [\"--flag\", \"value\"]")
    if isinstance(default, str):
        if not isinstance(value, str):
            raise BadSetting("text in quotes")
        return value
    return value


def check_setting(key, value):
    """Range and choice rules on an already converted value. Returns the (normalised) value."""
    choices = SETTING_CHOICES.get(key)
    if key == "model_profile":
        choices = profile_names() or None
    if choices is not None:
        low = value.strip().lower() if isinstance(value, str) else value
        if low not in choices:
            raise BadSetting("one of " + ", ".join(f'"{x}"' for x in choices))
        value = low
    if key == "ctx_size" and 0 < value < 2048:
        raise BadSetting("0 (the model profile's size) or at least 2048")
    if key == "seed" and value != -1 and not 0 <= value <= SEED_MAX:
        raise BadSetting(f"-1 (random) or a whole number from 0 to {SEED_MAX}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        low, high = SETTING_MIN.get(key), SETTING_MAX.get(key)
        if low is not None and value < low:
            raise BadSetting(f"at least {low}" + (f" and at most {high}" if high is not None else ""))
        if high is not None and value > high:
            raise BadSetting(f"at most {high}" + (f" and at least {low}" if low is not None else ""))
    return value


def apply_settings(settings, values, source):
    """Merge a dict of settings into `settings`, checking each one. A bad value is reported and
    that setting keeps its current value; unknown names are reported as possible typos."""
    for key, value in values.items():
        if key.startswith("_"):
            continue
        if key not in DEFAULT_SETTINGS:
            near = difflib.get_close_matches(key, list(DEFAULT_SETTINGS), n=1)
            warn(f'{source}: unknown setting "{key}" is ignored' + (f' (did you mean "{near[0]}"?)' if near else "")
                 + ". The list of settings is in settings.default.json.")
            continue
        try:
            settings[key] = check_setting(key, coerce_setting(key, value, DEFAULT_SETTINGS[key]))
        except BadSetting as e:
            warn(f'{source}: "{key}" is {json.dumps(value)}, but it must be {e}. '
                 f"Using {json.dumps(settings[key])} instead.")
    return settings


def read_settings_file(path, interactive=True):
    """The JSON object in a settings file, or None when it is missing or unreadable. JSON syntax
    errors are shown with their line and column (and wait for Enter when interactive)."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))  # -sig: Notepad and PowerShell 5 add a BOM
    except json.JSONDecodeError as e:
        err(f"{path.name} has a mistake at line {e.lineno}, column {e.colno}: {e.msg}.")
        info("Until it is fixed, the agent uses the default value of every setting in that file.")
        if not interactive:
            raise SystemExit(f"Fix {path} and run again.")
        try:
            input("  Press Enter to continue with the defaults, or close this window and fix the file. ")
        except EOFError:
            pass
        return None
    except (OSError, UnicodeDecodeError) as e:
        warn(f"Could not read {path.name} ({e}); using the defaults for it.")
        return None
    if not isinstance(data, dict):
        warn(f"{path.name} must hold one JSON object {{ ... }}; it is ignored.")
        return None
    return data


def is_legacy_settings(data):
    """True for a full copy of the settings.json that shipped before settings.default.json."""
    note = str(data.get("_note", ""))
    if note.startswith(USER_SETTINGS_NOTE[:40]):
        # Already the new layout, even if the user copied a few settings here together with their
        # notes: converting it again could drop a value they chose on purpose.
        return False
    return (note.startswith("Change a value, save, and restart the agent")
            or len(LEGACY_NOTE_KEYS & set(data)) >= 3)


def same_setting(key, a, b):
    try:
        default = DEFAULT_SETTINGS.get(key, b)
        return coerce_setting(key, a, default) == coerce_setting(key, b, default)
    except BadSetting:
        return False


def write_user_settings(values):
    """Write settings.json (the user's changes only), atomically."""
    doc = {"_note": USER_SETTINGS_NOTE}
    order = [k for k in DEFAULT_SETTINGS if k in values] + [k for k in values if k not in DEFAULT_SETTINGS]
    for k in order:
        doc[k] = values[k]
    tmp = paths.SETTINGS_FILE.with_name(paths.SETTINGS_FILE.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, paths.SETTINGS_FILE)


def migrate_legacy_settings(data):
    """Turn an old full settings.json into a list of the user's changes. The old file is kept as
    settings.backup.json, and every value that differs from what shipped stays, so nothing the user
    chose is lost. Returns the new contents (or the old ones if the file cannot be rewritten)."""
    changed = {k: v for k, v in data.items()
               if not k.startswith("_") and not (k in LEGACY_DEFAULTS and same_setting(k, v, LEGACY_DEFAULTS[k]))}
    backup = paths.SETTINGS_FILE.with_name("settings.backup.json")
    if backup.exists():
        backup = paths.SETTINGS_FILE.with_name(f"settings.backup-{datetime.now():%Y%m%d-%H%M%S}.json")
    try:
        shutil.copy2(paths.SETTINGS_FILE, backup)
        write_user_settings(changed)
    except OSError as e:
        warn(f"Could not update settings.json to the new layout ({e}); it is used as it is.")
        return data
    info("settings.json now holds only the settings you changed"
         + (f" ({', '.join(changed)})" if changed else " (none)")
         + f". All settings and their notes are in settings.default.json; your old file is saved as {backup.name}.")
    return changed


def load_settings(override=None, interactive=True):
    """Defaults (code, then settings.default.json), then the user's settings.json, then an
    override dict (the --settings-override switch). settings.json is created on the first start."""
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    shipped = read_settings_file(DEFAULTS_FILE, interactive)
    if shipped:
        apply_settings(settings, shipped, DEFAULTS_FILE.name)
    user = read_settings_file(paths.SETTINGS_FILE, interactive)
    if user is not None:
        if is_legacy_settings(user):
            user = migrate_legacy_settings(user)
        apply_settings(settings, user, paths.SETTINGS_FILE.name)
    elif not paths.SETTINGS_FILE.exists():
        try:
            write_user_settings({})
        except OSError:
            pass  # a read-only drive: the defaults work without the file
    if override:
        apply_settings(settings, override, "--settings-override")
    check_related_settings(settings)
    return settings


def check_related_settings(settings):
    """Rules that tie two settings together, checked once every layer is in."""
    budget, keep = int(settings.get("image_budget", 3)), int(settings.get("image_keep_after_evict", 2))
    if settings.get("image_eviction") == "block" and budget >= 2 and keep > budget - 1:
        warn(f'"image_keep_after_evict" is {keep}, but with "image_budget" {budget} it must be at most '
             f"{budget - 1}. Using {budget - 1} instead.")
        settings["image_keep_after_evict"] = budget - 1
    return settings
