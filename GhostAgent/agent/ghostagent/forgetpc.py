"""/forget-pc: what GhostAgent may have left on this PC, and removing it."""

import shutil

from . import paths
from .browsercache import (FolderLock, _folder_key, browser_cache_dir, browser_caches_to_forget, delete_browser_cache,
                           is_browser_cache_name, prune_browser_caches, temp_folder)
from .console import c, info, warn
from .paths import pc_data_dir, screenshots_dir


# --------------------------------------------------------------------------------------------
# /forget-pc: what GhostAgent may have left on this PC, and removing it
# --------------------------------------------------------------------------------------------

def pc_leftovers(settings=None):
    """[(what, path)] for everything GhostAgent keeps on this PC that exists right now."""
    base = pc_data_dir()
    items = [
        ("the copy of the model", base / "models"),
        ("the copy of the model from before the rename", base.parent / "HoloAgent"),
        ("saved screenshots", screenshots_dir()),
        *browser_caches_to_forget(),
        ("the browser profile kept on this PC", base / "browser-profile"),
        ("the Edge profile kept on this PC", base / "browser-profile-edge"),
        ("the agent's Chrome profile on this PC (browser chrome-agent)", base / "chrome-profile"),
        ("logs kept on this PC (read-only drive)", base / "logs"),
        ("downloads kept on this PC (read-only drive)", base / "downloads"),
        ("this PC's answers to GhostAgent's questions", base / "pc.json"),
    ]
    return [(what, path) for what, path in items if path.exists()]


def remove_path(path):
    """Delete a file or folder. True when it is gone afterwards."""
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
    except OSError:
        pass
    return not path.exists()


def forget_pc(settings, later):
    """The /forget-pc command: list what GhostAgent keeps on this PC, and after a yes, delete it.
    What is still in use (the model the server has open, the running browser's cache, this run's
    log) is added to `later` and deleted when GhostAgent closes."""
    items = pc_leftovers(settings)
    if not items:
        info("GhostAgent keeps nothing on this PC right now.")
        return
    print(c("  GhostAgent keeps these on this PC:", "1;33"))
    for what, path in items:
        print(f"    {what}: {path}")
    try:
        answer = input(c("  Delete them? [y/N] ", "1;33")).strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
        print()
    if answer not in ("y", "yes"):
        info("Nothing was deleted.")
        return
    own_cache = _folder_key(browser_cache_dir())
    for what, path in items:
        if (is_browser_cache_name(path.name) and path.parent == temp_folder() and not FolderLock.holds(path)
                and _folder_key(path) != own_cache):
            # A cache an earlier run left goes only while no agent holds it (one may have started since),
            # and is never left for later: by then it may be another agent's.
            if delete_browser_cache(path):
                info(f"deleted {what}")
            else:
                info(f"{what} is now in use by another GhostAgent window, so it was left alone")
            continue
        if remove_path(path):
            info(f"deleted {what}")
        else:
            later.append((what, path))
            info(f"{what} is in use; it is deleted when GhostAgent closes")
    prune_browser_caches(keep=browser_cache_dir())  # caches left by earlier runs that no agent holds
    base = pc_data_dir()
    try:
        base.rmdir()  # only when empty
    except OSError:
        pass


def finish_forget_pc(later):
    """At exit, after the browser and the model server have stopped: delete what /forget-pc could
    not delete while it was in use."""
    if not later:
        return
    if paths._CONSOLE_LOG is not None and any(paths._CONSOLE_LOG.path.is_relative_to(p) for _, p in later if p.is_dir()):
        paths._CONSOLE_LOG.close()  # this run's log is among them: it can only go once it is closed
        paths._CONSOLE_LOG = None
    left = [(what, path) for what, path in later if not remove_path(path)]
    try:
        pc_data_dir().rmdir()
    except OSError:
        pass
    for what, path in left:
        warn(f"Could not delete {what} ({path}); delete it by hand once GhostAgent is closed.")
