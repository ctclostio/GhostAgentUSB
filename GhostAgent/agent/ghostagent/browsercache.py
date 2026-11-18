"""The agent's own browser's disk cache in this PC's TEMP folder: one folder per browser profile, a
lock file held while that browser runs, and removing the caches no running agent holds."""

import hashlib
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

from . import paths


BROWSER_CACHE_PREFIX = "ghost-agent-browser-cache"
# Inside each cache folder: held open and locked while that agent's browser runs, so no other agent
# deletes the folder. Windows lets go of the lock when the process ends, however it ends.
CACHE_LOCK_FILE = "ghost-agent.lock"
# A cache folder without a lock file (one from before the lock, or one being set up at this very
# moment) is left alone until it is this old (seconds).
UNLOCKED_CACHE_AGE = 600


def temp_folder():
    return Path(os.environ.get("TEMP") or os.environ.get("TMP") or tempfile.gettempdir())


def browser_cache_dir(profile=None):
    """The agent's own browser's disk cache, kept in this PC's TEMP folder (fast, and off the stick):
    %TEMP%\\ghost-agent-browser-cache-<id>, one folder per browser profile, so two agents with other
    data folders (a test or benchmark next to a real run) never share or delete each other's cache."""
    where = os.path.normcase(os.path.abspath(str(profile or paths.PROFILE)))
    return temp_folder() / f"{BROWSER_CACHE_PREFIX}-{hashlib.sha256(where.encode('utf-8')).hexdigest()[:12]}"


def _folder_key(folder):
    return os.path.normcase(os.path.abspath(str(folder)))


class FolderLock:
    """The lock file in a browser cache folder, held open and locked (FolderLock.take)."""

    held = set()  # the folders whose lock this process holds (another lock on them fails here too)

    def __init__(self, f, folder):
        self.f = f
        self.key = _folder_key(folder)
        FolderLock.held.add(self.key)

    @classmethod
    def holds(cls, folder):
        """True when this process holds the lock of `folder`."""
        return _folder_key(folder) in cls.held

    @classmethod
    def take(cls, folder):
        """The lock of `folder` (created when missing), or None when another agent holds it."""
        try:
            Path(folder).mkdir(parents=True, exist_ok=True)
            f = open(Path(folder) / CACHE_LOCK_FILE, "a+b")
        except OSError:
            return None
        try:
            import msvcrt
        except ImportError:
            return cls(f, folder)  # not Windows: no lock, the folder is simply used
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            f.close()
            return None
        return cls(f, folder)

    def release(self):
        if self.f is None:
            return
        try:
            import msvcrt
            self.f.seek(0)
            msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
        except (ImportError, OSError, ValueError):
            pass
        try:
            self.f.close()
        except OSError:
            pass
        self.f = None
        FolderLock.held.discard(self.key)


def empty_folder(folder, keep=()):
    """Delete everything in a folder except the names in `keep` (what cannot be deleted stays)."""
    try:
        children = list(Path(folder).iterdir())
    except OSError:
        return
    for p in children:
        if p.name in keep:
            continue
        try:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink()
        except OSError:
            pass


def remove_browser_cache(folder, lock):
    """Delete a browser cache whose lock is held: its contents first, then the lock and the folder.
    When something cannot be deleted yet (a file the browser has not let go of), the lock file stays
    with it, so the next start deletes the rest at once instead of after UNLOCKED_CACHE_AGE."""
    empty_folder(folder, keep={CACHE_LOCK_FILE})
    lock.release()
    try:
        if any(p.name != CACHE_LOCK_FILE for p in Path(folder).iterdir()):
            return
        (Path(folder) / CACHE_LOCK_FILE).unlink()
        Path(folder).rmdir()
    except OSError:
        pass  # another agent took the folder over in between: it is that agent's now


def is_browser_cache_name(name):
    return name == BROWSER_CACHE_PREFIX or re.fullmatch(re.escape(BROWSER_CACHE_PREFIX) + r"-[0-9a-f]{12}", name)


def browser_cache_folders(keep=None):
    """The agents' browser cache folders in TEMP (real folders with one of their names), except `keep`."""
    keep = _folder_key(keep) if keep else None
    try:
        return [p for p in temp_folder().iterdir()
                if is_browser_cache_name(p.name) and p.is_dir() and not p.is_symlink()
                and not (hasattr(p, "is_junction") and p.is_junction()) and _folder_key(p) != keep]
    except OSError:
        return []


def held_by_another_agent(folder):
    """True when another running agent holds the lock of a browser cache."""
    if FolderLock.holds(folder) or not (Path(folder) / CACHE_LOCK_FILE).exists():
        return False
    lock = FolderLock.take(folder)
    if lock is None:
        return True
    lock.release()
    return False


def browser_cache_is_stale(folder):
    """True when no running agent holds a browser cache: its lock is free, or it has no lock file and is
    older than UNLOCKED_CACHE_AGE (one from before the lock; a younger one may be being set up)."""
    if FolderLock.holds(folder):
        return False
    if (Path(folder) / CACHE_LOCK_FILE).exists():
        return not held_by_another_agent(folder)
    try:
        return time.time() - Path(folder).stat().st_mtime >= UNLOCKED_CACHE_AGE
    except OSError:
        return False


def browser_caches_to_forget():
    """[(what, folder)] for /forget-pc: this agent's browser cache (also while its browser runs) and the
    caches earlier runs left that no running agent holds, such as one from another data folder or from
    when the stick had another drive letter. Caches other running agents use are not listed."""
    own = _folder_key(browser_cache_dir())
    found = []
    for folder in browser_cache_folders():
        if held_by_another_agent(folder):
            continue
        if FolderLock.holds(folder) or _folder_key(folder) == own:
            found.append(("the browser's cache", folder))
        elif browser_cache_is_stale(folder):
            found.append(("a browser cache left by an earlier run", folder))
    return found


def delete_browser_cache(folder):
    """Delete a browser cache that no running agent holds (see browser_cache_is_stale). True when it
    is gone."""
    folder = Path(folder)
    if (folder / CACHE_LOCK_FILE).exists():
        lock = FolderLock.take(folder)
        if lock is None:
            return False
        remove_browser_cache(folder, lock)
    elif browser_cache_is_stale(folder):
        shutil.rmtree(folder, ignore_errors=True)
    return not folder.exists()


def prune_browser_caches(keep=None):
    """Delete the browser caches in TEMP that no running agent holds: ones left by a window closed
    with its X button, or by a crash. A cache whose lock another agent holds is never touched, and
    `keep` (this agent's own) neither. Returns how many were deleted."""
    return sum(delete_browser_cache(folder) for folder in browser_cache_folders(keep))
