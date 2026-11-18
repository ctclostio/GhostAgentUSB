"""press_keys_web: key names in Playwright's spelling, and the checks on what a press would do
(paste, type text, set off a button)."""

import json
import re


KEY_ALIASES = {
    "ctrl": "Control", "control": "Control", "cmd": "Control", "command": "Control", "meta": "Meta",
    "win": "Meta", "super": "Meta", "alt": "Alt", "option": "Alt", "shift": "Shift",
    "enter": "Enter", "return": "Enter", "esc": "Escape", "escape": "Escape", "tab": "Tab",
    "space": "Space", "spacebar": "Space", "backspace": "Backspace", "delete": "Delete", "del": "Delete",
    "up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight",
    "arrowup": "ArrowUp", "arrowdown": "ArrowDown", "arrowleft": "ArrowLeft", "arrowright": "ArrowRight",
    "pageup": "PageUp", "pagedown": "PageDown", "pgup": "PageUp", "pgdn": "PageDown",
    "home": "Home", "end": "End", "insert": "Insert",
}


def _key_parts(combo):
    """The keys of one press in Playwright's spelling: 'ctrl+a' is ['Control', 'a'], 'pagedown' is
    ['PageDown'], 'Control++' is ['Control', '+']. Empty parts are skipped. A letter in a chord takes
    the case a real keyboard gives it: upper case with Shift (any Shift key), lower case without.
    Playwright sends exactly the character it is given and does not apply a held Shift to a letter,
    so 'Shift+v' would type a lower-case v, and 'Control+T' would send a 'T' no keyboard sends
    without Shift (checked on tools\\bench\\site\\fixtures\\keys.html)."""
    combo = combo.strip()
    plus_key = combo.endswith("++") or combo == "+"
    parts = [p for p in (combo[:-2] if plus_key else combo).split("+") if p] + (["+"] if plus_key else [])
    shift = any(p.lower() in ("shift", "shiftleft", "shiftright") for p in parts[:-1])
    out = []
    for p in parts:
        low = p.lower()
        if low in KEY_ALIASES:
            out.append(KEY_ALIASES[low])
        elif re.fullmatch(r"f\d{1,2}", low):
            out.append(low.upper())
        elif len(p) == 1:
            out.append(p if len(parts) == 1 else p.upper() if shift else p.lower())
        else:
            out.append(p[0].upper() + p[1:])
    return out


def normalize_key(combo):
    """One press in Playwright's spelling (see _key_parts): 'ctrl+a' is 'Control+a'."""
    return "+".join(_key_parts(combo))


# Key names Playwright knows (its US keyboard layout: key codes and key values). Any single
# character is a key too. Checked before anything is pressed: a name Playwright does not know
# fails in the middle of a chord, after its first keys were already pressed.
KNOWN_KEYS = {k.lower(): k for k in (
    "Control", "Shift", "Alt", "Meta", "ControlOrMeta", "AltGraph", "Enter", "Escape", "Tab", "Space", "Backspace",
    "Delete", "Insert", "Home", "End", "PageUp", "PageDown", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
    "CapsLock", "NumLock", "ScrollLock", "Pause", "PrintScreen", "ContextMenu", "Backquote", "Minus", "Equal",
    "Backslash", "BracketLeft", "BracketRight", "Semicolon", "Quote", "Comma", "Period", "Slash",
    "ShiftLeft", "ShiftRight", "ControlLeft", "ControlRight", "AltLeft", "AltRight", "MetaLeft", "MetaRight",
    "NumpadDivide", "NumpadMultiply", "NumpadSubtract", "NumpadAdd", "NumpadDecimal", "NumpadEnter",
    "AudioVolumeMute", "AudioVolumeDown", "AudioVolumeUp", "MediaTrackNext", "MediaTrackPrevious", "MediaPlayPause",
    *[f"F{n}" for n in range(1, 13)], *[f"Key{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
    *[f"Digit{d}" for d in range(10)], *[f"Numpad{d}" for d in range(10)])}
KEY_NAMES_HELP = ("Use key names such as Enter, Escape, Tab, Backspace, Delete, Space, ArrowUp, ArrowDown, ArrowLeft, "
                  "ArrowRight, PageUp, PageDown, Home, End or F1-F12, or a single character, with Control, Shift, Alt "
                  "or Meta joined by + for a shortcut (for example ctrl + a). Separate several presses with spaces.")


class BadKeys(ValueError):
    pass


def parse_keys(keys):
    """The presses in a press_keys_web argument, each in Playwright's spelling, or BadKeys naming the
    first key that does not exist (then nothing may be pressed). Accepted forms:
      - a JSON list, read as one chord: '["ctrl", "a"]' is Control+a;
      - presses separated by spaces, where spaces around a plus belong to the chord: 'ctrl + c' is
        one press, 'Tab Tab Enter' three;
      - 'Page Down' and 'Page Up' as two words."""
    presses = [parts for parts in (_key_parts(chord) for chord in _chords(keys)) if parts]
    for parts in presses:
        for part in parts:
            if len(part) > 1 and part.lower() not in KNOWN_KEYS:
                raise BadKeys(f'Unknown key "{part}"; nothing was pressed. ' + KEY_NAMES_HELP)
    if not presses:
        raise BadKeys("No keys were given. " + KEY_NAMES_HELP)
    return ["+".join(parts) for parts in presses]


def _chords(keys):
    """The presses in a press_keys_web argument as written (see parse_keys), not yet normalised."""
    if isinstance(keys, (list, tuple)):
        # A model may send the list itself rather than its JSON text: the same one chord.
        keys = json.dumps([str(k) for k in keys])
    text = str(keys if keys is not None else "").strip()
    if text.startswith("["):
        try:
            items = json.loads(text)
        except ValueError:
            items = None
        if isinstance(items, list) and items and all(isinstance(i, str) and i.strip() for i in items):
            return ["+".join(i.strip() for i in items)]
    text = re.sub(r"(?i)\bpage\s+(down|up)\b", lambda m: "Page" + m.group(1).capitalize(), text)
    return re.sub(r"\s*\+\s*", "+", text).split()


def key_presses(keys):
    """The presses a press_keys_web argument stands for, normalised, without checking the names (for
    the checks that look at what would be pressed)."""
    return [p for p in (normalize_key(chord) for chord in _chords(keys)) if p]


PASTE_REFUSAL = "Pasting from the clipboard is disabled; type the text with write instead."


def _key_char(key):
    """A key name in lower case, with Playwright's key codes read as the character they type:
    'KeyV' is 'v', 'Digit1' and 'Numpad1' are '1'. Playwright accepts both spellings, so the checks
    below must too."""
    low = key.lower()
    if re.fullmatch(r"key[a-z]", low):
        return low[3]
    if re.fullmatch(r"(?:digit|numpad)\d", low):
        return low[-1]
    return low


def is_paste(combo):
    """True for the chords that paste the clipboard: Control+V, Control+Shift+V, Meta+V, Shift+Insert
    (in any case and any order of the modifiers, also as Control+KeyV or ControlOrMeta+V). A page could
    otherwise have the agent paste whatever the user last copied, a password or a code, into a form
    of its own."""
    parts = [p.lower() for p in combo.split("+") if p]
    if not parts:
        return False
    key, mods = _key_char(parts[-1]), set(parts[:-1])
    if mods & {"controlormeta", "control-or-meta"}:
        mods.add("control")
    return bool((key == "v" and mods & {"control", "meta"}) or (key == "insert" and "shift" in mods))


# Keys that type nothing. Every other key does (a letter, 'Space', 'Digit1', 'Equal', 'NumpadAdd'...).
NON_TEXT_KEYS = {
    "enter", "numpadenter", "escape", "tab", "backspace", "delete", "insert", "home", "end", "pageup", "pagedown",
    "arrowup", "arrowdown", "arrowleft", "arrowright", "shift", "shiftleft", "shiftright", "control", "controlleft",
    "controlright", "alt", "altleft", "altright", "meta", "metaleft", "metaright", "controlormeta", "capslock",
    "numlock", "scrolllock", "pause", "printscreen", "contextmenu", "clear",
}


def types_text(combo):
    """True when a press types a character: any key except the ones in NON_TEXT_KEYS and F1-F24,
    alone or with Shift ('a', 'Shift+A', 'Space', 'KeyA', 'Digit1', 'Equal')."""
    parts = [p.lower() for p in combo.split("+") if p]
    if not parts or not set(parts[:-1]) <= {"shift"}:
        return False
    key = parts[-1]
    return key not in NON_TEXT_KEYS and not re.fullmatch(r"f\d{1,2}", key)


def activates(combo):
    """True for a press that can set off the focused button, link or form: Enter (also with a
    modifier: Control+Enter sends in many mail and chat sites) or Space."""
    parts = [p.lower() for p in combo.split("+") if p]
    return bool(parts) and (parts[-1] in ("enter", "numpadenter") or (parts[-1] in ("space", " ") and len(parts) == 1))
