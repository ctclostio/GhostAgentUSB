"""The model copy on this PC (copy_model_to_pc): asked once per PC, resumed after an interruption,
checked against its SHA-256 while it copies."""

import ctypes
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from .console import LiveLine, info, warn
from .paths import pc_data_dir
from .preflight import gb, model_path, pinned_sha256
from .questions import ask_console
from .settings import FALLBACK_PROFILES, load_profiles


# --------------------------------------------------------------------------------------------
# The model copy on this PC (copy_model_to_pc): asked once per PC, resumed after an
# interruption, checked against its SHA-256 while it copies
# --------------------------------------------------------------------------------------------

DRIVE_REMOVABLE = 2
SLOW_LOAD_MB_S = 200           # a drive that loaded slower than this gets the copy question too
COPY_FREE_MARGIN = 20 * 2**30  # what must stay free on the PC's disk after the copy
COPY_CHUNK = 64 * 2**20
USB_LOAD_MB_S = 69             # what a USB 3 stick measured; only for the "about N min" in the question


def pc_settings_file():
    return pc_data_dir() / "pc.json"


def read_pc_settings():
    """%LOCALAPPDATA%\\GhostAgent\\pc.json: this PC's answers to GhostAgent's questions, and what it measured."""
    try:
        data = json.loads(pc_settings_file().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_pc_settings(**changes):
    data = read_pc_settings()
    data.update(changes)
    data["_note"] = ("GhostAgent's answers and measurements for this PC. Delete this file (or use /forget-pc) "
                     "to be asked again.")
    path = pc_settings_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def record_load_speed(bytes_per_s):
    """Keep how fast the model loaded from its drive (pc.json load_mb_s): a slow drive gets the question
    about a copy on this PC next time. The lowest speed seen is kept, because a second start often
    reads the model from Windows' file cache, which says nothing about the drive."""
    if not bytes_per_s:
        return
    mb_s = max(1, round(bytes_per_s / 1e6))
    old = read_pc_settings().get("load_mb_s")
    if isinstance(old, (int, float)) and not isinstance(old, bool) and old > 0:
        if old <= mb_s:
            return
    write_pc_settings(load_mb_s=mb_s)


def drive_type(path):
    """GetDriveTypeW for the drive a path is on (2 = removable, 3 = fixed), or None."""
    if os.name != "nt":
        return None
    try:
        drive = Path(path).resolve().drive
        return int(ctypes.windll.kernel32.GetDriveTypeW(drive + "\\")) if drive else None
    except Exception:
        return None


def same_drive(a, b):
    try:
        return Path(a).resolve().drive.lower() == Path(b).resolve().drive.lower()
    except OSError:
        return False


def free_bytes(folder):
    """Free space on the drive of a folder (or of its nearest existing parent)."""
    p = Path(folder)
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return 0


def eta_text(seconds):
    if seconds is None or seconds < 0:
        return ""
    if seconds < 90:
        return f"about {max(5, int(round(seconds / 5.0)) * 5)} s"
    return f"about {int(round(seconds / 60.0))} min"


class Progress:
    """Bytes done out of a total, shown as 'x of y GB, z MB/s, about n min left'. One thread adds,
    another may read: plain numbers, so no lock is needed."""

    def __init__(self, total, verb, on_change=None):
        self.total, self.verb, self.on_change = total, verb, on_change
        self.done = 0
        self.skipped = 0  # already in place, so not part of the speed
        self.t0 = time.monotonic()

    def add(self, n, skipped=False):
        self.done += n
        if skipped:
            self.skipped += n
        if self.on_change:
            self.on_change(self)

    def rate(self):
        elapsed = time.monotonic() - self.t0
        return (self.done - self.skipped) / elapsed if elapsed > 0.5 else 0.0

    def line(self):
        rate = self.rate()
        left = (self.total - self.done) / rate if rate > 0 else None
        text = f"{self.verb}: {gb(self.done)} of {gb(self.total)} GB"
        if rate > 0:
            text += f", {rate / 1e6:.0f} MB/s"
        if left is not None and self.done < self.total:
            text += f", {eta_text(left)} left"
        return text


def console_progress(prefix="  "):
    """An on_change for Progress that redraws one console line at most twice a second (a LiveLine:
    plain lines when the output is not a console). Its .line is ended once the progress is complete."""
    state = {"at": 0.0}
    line = LiveLine(prefix)

    def show(progress):
        now = time.monotonic()
        if progress.done >= progress.total:
            if not state.get("closed"):
                state["closed"] = True
                line.close(progress.line())  # how it ended: shown either way
        elif now - state["at"] >= 0.5:
            state["at"] = now
            line.show(progress.line())
    show.line = line
    return show


def read_verified(folder):
    """{file name: {size, mtime_ns, sha256}} from <folder>\\.verified, the record the Build script writes
    of files that passed their SHA-256 check (the agent writes the same record for its copies)."""
    try:
        data = json.loads((Path(folder) / ".verified").read_text(encoding="utf-8-sig"))
        files = data.get("files") or {}
        return {k: v for k, v in files.items() if isinstance(v, dict)}
    except (OSError, ValueError, AttributeError):
        return {}


def write_verified(folder, path, sha256):
    folder = Path(folder)
    files = {name: rec for name, rec in read_verified(folder).items() if (folder / name).exists()}
    st = Path(path).stat()
    files[Path(path).name] = {"name": Path(path).name, "size": st.st_size, "mtime": st.st_mtime_ns / 1e9,
                              "mtime_ns": st.st_mtime_ns, "sha256": sha256}
    doc = {"note": "Files that passed their SHA-256 check (written by tools\\Build-GhostAgent.ps1 and by GhostAgent). "
                   "While size and mtime_ns (nanoseconds since 1970, UTC) still match, a file is not hashed again.",
           "files": files}
    try:
        (folder / ".verified").write_text(json.dumps(doc, indent=2), encoding="ascii")
    except OSError:
        pass


def expected_sha256(src):
    """The SHA-256 a model file must have: models\\SHA256SUMS.txt next to it, else tools\\profiles.json."""
    try:
        for line in (Path(src).parent / "SHA256SUMS.txt").read_text(encoding="utf-8-sig").splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and parts[1].lstrip("*").strip().lower() == Path(src).name.lower():
                return parts[0].lower()
    except OSError:
        pass
    return pinned_sha256(Path(src).name)


class CopyCancelled(Exception):
    pass


def _hash_into(path, length, digest, progress, cancel):
    with open(path, "rb") as f:
        left = length
        while left > 0:
            if cancel is not None and cancel.is_set():
                raise CopyCancelled()
            chunk = f.read(min(COPY_CHUNK, left))
            if not chunk:
                break
            digest.update(chunk)
            left -= len(chunk)
            if progress:
                progress.add(len(chunk))


def copy_model_file(src, dest_dir, progress=None, notes=None, cancel=None):
    """The path to load `src` from: its copy in dest_dir, made or finished now, or `src` itself when
    the copy cannot be made. A partial copy (<name>.part) is resumed. Every byte is hashed on the way
    and checked against the pinned SHA-256; a good copy is recorded in .verified, so later starts
    compare only its size and time. Problems are added to `notes` as sentences for the user."""
    notes = notes if notes is not None else []
    src, dest_dir = Path(src), Path(dest_dir)
    dst = dest_dir / src.name
    size = src.stat().st_size
    if _same_file(src, dst):
        # The "model" setting already names the copy on this PC: there is nothing to copy, and a
        # failed check below must never delete the only copy there is.
        if progress:
            progress.add(size, skipped=True)
        return src
    want = expected_sha256(src)
    if dst.is_file() and dst.stat().st_size == size:
        st = dst.stat()
        rec = read_verified(dest_dir).get(src.name) or {}
        if int(rec.get("size", -1)) == size and int(rec.get("mtime_ns", -1)) == st.st_mtime_ns \
                and (not want or str(rec.get("sha256", "")).lower() == want):
            if progress:
                progress.add(size, skipped=True)
            return dst
        if not want:
            if progress:
                progress.add(size, skipped=True)
            return dst  # nothing to check it against (a model the user brought): the size decides
        # A copy made before copies were checked (or one that changed): hashed once, then remembered.
        digest = hashlib.sha256()
        _hash_into(dst, size, digest, progress, cancel)
        if digest.hexdigest() == want:
            write_verified(dest_dir, dst, want)
            return dst
        notes.append(f"The copy of {src.name} on this PC was damaged; it is copied again.")
        if progress:
            progress.done -= size
        dst.unlink()
    part = dest_dir / (src.name + ".part")
    for attempt in (1, 2):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        if free_bytes(dest_dir) < size - have + 2 * 2**30:
            if part.exists():
                part.unlink()  # a partial copy that cannot be finished only takes space
            notes.append(f"Not enough free disk space to copy {src.name} to {dest_dir}; it loads from the USB drive.")
            if progress:
                progress.add(size, skipped=True)
            return src
        digest = hashlib.sha256()
        start = progress.done if progress else 0
        if have:
            _hash_into(part, have, digest, progress, cancel)  # the bytes already there count too
        with open(src, "rb") as fi, open(part, "ab") as fo:
            fi.seek(have)
            while True:
                if cancel is not None and cancel.is_set():
                    raise CopyCancelled()
                chunk = fi.read(COPY_CHUNK)
                if not chunk:
                    break
                fo.write(chunk)
                digest.update(chunk)
                if progress:
                    progress.add(len(chunk))
        got = digest.hexdigest()
        if want and got != want:
            part.unlink()
            if progress:
                progress.done = start
            if have and attempt == 1:
                continue  # the part from an earlier run may be the damaged one: once more from the start
            notes.append(f"The model on the USB drive is damaged ({src.name} does not match its SHA-256). Run "
                         "Build GhostAgent.cmd again to repair it; until then it loads from the USB drive.")
            if progress:
                progress.add(size, skipped=True)
            return src
        part.replace(dst)
        if want:
            write_verified(dest_dir, dst, got)
        return dst
    return src


def model_cache_dir():
    return pc_data_dir() / "models"


def prune_model_cache(folder, keep, notes=None):
    """Delete model files in the PC's copy folder that no profile uses any more (and partial copies of
    them), with a notice. `keep`: file names in use now. Returns the names deleted."""
    notes = notes if notes is not None else []
    folder = Path(folder)
    doc = load_profiles()
    if doc is FALLBACK_PROFILES:
        # Without tools\profiles.json the agent knows only the fast profile: copies of the other
        # profiles' models (tens of GB to copy again) are not "unused", so nothing is deleted.
        return []
    wanted = {n.lower() for n in keep}
    for prof in doc["profiles"].values():
        for part in ("model", "mmproj"):
            name = (prof.get(part) or {}).get("name") if isinstance(prof.get(part), dict) else None
            if name:
                wanted.add(str(name).lower())
    removed = []
    for f in list(folder.glob("*.gguf")) + list(folder.glob("*.part")):
        base = f.name[:-5] if f.name.lower().endswith(".part") else f.name
        if base.lower() in wanted:
            continue
        try:
            f.unlink()
        except OSError:
            continue
        removed.append(f.name)
        try:
            folder.joinpath(base + ".layout.json").unlink()
        except OSError:
            pass
    if removed:
        notes.append(f"Deleted {', '.join(removed)} from {folder}: no model profile uses them any more.")
    return removed


def decide_model_copy(settings, interactive=True):
    """Whether to load the model from a copy on this PC. copy_model_to_pc true or false decides; "ask"
    asks once per PC (the answer is kept in pc.json) when the model is on a removable drive, or loaded
    slowly from its drive before, and this PC's disk has room for it with 20 GB to spare."""
    value = settings.get("copy_model_to_pc", "ask")
    if isinstance(value, bool):
        return value
    pc = read_pc_settings()
    if isinstance(pc.get("copy_model_to_pc"), bool):
        return pc["copy_model_to_pc"]
    files = [model_path(settings, "model"), model_path(settings, "mmproj")]
    try:
        total = sum(f.stat().st_size for f in files)
    except OSError:
        return False
    dest = model_cache_dir()
    if same_drive(files[0], dest):
        return False  # already on this PC's disk
    if all(_size_of(dest / f.name) == _size_of(f) for f in files):
        return True  # a copy is there already (the Build script leaves one on the PC it ran on): nothing to ask
    rate = pc.get("load_mb_s")
    slow = isinstance(rate, (int, float)) and not isinstance(rate, bool) and 0 < rate < SLOW_LOAD_MB_S
    if not (drive_type(files[0]) == DRIVE_REMOVABLE or slow):
        return False
    if free_bytes(dest) < total + COPY_FREE_MARGIN:
        return False
    if not interactive:
        return False
    minutes = max(1, round(total / ((rate if slow else USB_LOAD_MB_S) * 1e6) / 60))
    answer, how = ask_console(f"Copy the {gb(total)} GB model to this PC, so the agent starts in 10-25 s instead of "
                              f"about {minutes} min? [Y/n]")
    if how != "answered":
        return False  # no answer: nothing copied, and asked again next time
    yes = answer.strip().lower() in ("", "y", "yes")
    write_pc_settings(copy_model_to_pc=yes)
    info(("The model will be copied to " + str(dest) if yes else "The model stays on the drive")
         + ". The answer is kept for this PC (copy_model_to_pc in settings.json overrides it; /forget-pc removes it).")
    return yes


def model_files(settings, copy=None, progress=None, notes=None, cancel=None):
    """[model, projector] paths to load. With copy (copy_model_to_pc), each file is copied once to this
    PC's disk and loaded from there: a USB drive reads slowly, a local SSD loads the model in seconds.
    Without a progress object the copy shows its own progress line."""
    files = [model_path(settings, "model"), model_path(settings, "mmproj")]
    for f in files:
        if not f.exists():
            raise SystemExit(f"Missing model file: {f}")
    if copy is None:
        copy = settings.get("copy_model_to_pc") is True
    if not copy:
        return files
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    dest = model_cache_dir()
    old = base / "HoloAgent" / "models"  # where the model was copied before the rename to GhostAgent
    if old.is_dir() and not dest.exists():
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            old.rename(dest)
            old.parent.rmdir()
        except OSError:
            pass
    dest.mkdir(parents=True, exist_ok=True)
    own_line = progress is None
    if own_line:
        progress = Progress(sum(f.stat().st_size for f in files), "copying the model to this PC", console_progress())
    notes = [] if notes is None else notes
    shown = len(notes)
    local = [copy_model_file(src, dest, progress, notes, cancel) for src in files]
    prune_model_cache(dest, [f.name for f in files], notes)
    if own_line:
        progress.on_change.line.close()  # (already ended when the copy completed)
        for note in notes[shown:]:
            warn(note)
    return local


def _size_of(path):
    try:
        return Path(path).stat().st_size
    except (OSError, TypeError):
        return 0


def _same_file(a, b):
    try:
        return Path(a).exists() and Path(b).exists() and os.path.samefile(a, b)
    except OSError:
        return False
