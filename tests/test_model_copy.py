"""The model copy on this PC (copy_model_to_pc): asked once per PC, made in the background while the
browser starts, checked against its SHA-256 while it copies, resumed after an interruption, and
cleaned up when no profile uses a file any more.

Every test points LOCALAPPDATA at its own temporary folder and copies small fake model files, with
their sums in a temporary SHA256SUMS.txt; the real model and the real %LOCALAPPDATA% are never used."""

import hashlib
import json
import os
import threading
import time
import unittest
import unittest.mock as mock
from pathlib import Path

import helpers
from agent_support import AGENT, patch_agent, quiet, scratch_paths, settings


class Proc:
    _handle = 0
    pid = 4

    def __init__(self):
        self.code = None

    def poll(self):
        return self.code

    def terminate(self):
        self.code = 1

    def kill(self):
        self.code = 1

    def wait(self, timeout=None):
        return self.code


class CopyCase(unittest.TestCase):
    def setUp(self):
        scratch_paths(self)
        self.base = helpers.temp_dir(self)
        self.local = self.base / "LocalAppData"
        self.stick = self.base / "stick" / "models"
        self.stick.mkdir(parents=True)
        patcher = mock.patch.dict(os.environ, {"LOCALAPPDATA": str(self.local)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.dest = AGENT.model_cache_dir()

    def model(self, name="Holo4-35B-A3B-Q4_K_M.gguf", size=300_000, seed=1, listed=True):
        """A fake model file on the 'stick', with its sum in SHA256SUMS.txt next to it."""
        data = bytes((seed * 7 + i * 13) % 251 for i in range(size))
        path = self.stick / name
        path.write_bytes(data)
        if listed:
            sums = self.stick / "SHA256SUMS.txt"
            lines = [l for l in (sums.read_text(encoding="utf-8").splitlines() if sums.exists() else [])
                     if not l.endswith(name)]
            sums.write_text("\n".join(lines + [f"{hashlib.sha256(data).hexdigest()} *{name}"]) + "\n", encoding="utf-8")
        return path

    def copy(self, src, **kw):
        notes = []
        with patch_agent("COPY_CHUNK", 64 * 1024):
            got = AGENT.copy_model_file(src, self.dest, notes=notes, **kw)
        return got, notes


class CopyFileTests(CopyCase):
    def test_a_fresh_copy_is_checked_and_recorded(self):
        src = self.model()
        self.dest.mkdir(parents=True)
        progress = AGENT.Progress(src.stat().st_size, "copying")
        got, notes = self.copy(src, progress=progress)
        self.assertEqual(got, self.dest / src.name)
        self.assertEqual(got.read_bytes(), src.read_bytes())
        self.assertEqual(notes, [])
        self.assertEqual(progress.done, src.stat().st_size)
        rec = AGENT.read_verified(self.dest)[src.name]
        self.assertEqual((rec["size"], rec["mtime_ns"]), (got.stat().st_size, got.stat().st_mtime_ns))
        self.assertEqual(rec["sha256"], hashlib.sha256(src.read_bytes()).hexdigest())
        self.assertFalse((self.dest / (src.name + ".part")).exists())
        # the next start compares size and time only
        with patch_agent("_hash_into", side_effect=AssertionError("hashed again")):
            self.assertEqual(self.copy(src)[0], got)

    def test_an_interrupted_copy_is_resumed(self):
        src = self.model()
        self.dest.mkdir(parents=True)
        part = self.dest / (src.name + ".part")
        part.write_bytes(src.read_bytes()[:100_000])
        read_from = []
        real_open = open

        def spy_open(path, mode="r", *a, **kw):
            f = real_open(path, mode, *a, **kw)
            if Path(path) == src:
                real_seek = f.seek
                f.seek = lambda pos, *x: read_from.append(pos) or real_seek(pos, *x)
            return f

        with mock.patch("builtins.open", spy_open):
            got, notes = self.copy(src)
        self.assertEqual(read_from, [100_000])  # only the rest was read from the stick
        self.assertEqual(got.read_bytes(), src.read_bytes())
        self.assertEqual(notes, [])

    def test_a_damaged_partial_copy_is_made_again_from_the_start(self):
        src = self.model()
        self.dest.mkdir(parents=True)
        part = self.dest / (src.name + ".part")
        broken = bytearray(src.read_bytes()[:150_000])
        broken[1234] ^= 0xFF
        part.write_bytes(bytes(broken))
        got, notes = self.copy(src)
        self.assertEqual(got.read_bytes(), src.read_bytes())
        self.assertEqual(notes, [])
        # a part longer than the file is no part of it
        got.unlink()
        part.write_bytes(src.read_bytes() + b"extra")
        got, notes = self.copy(src)
        self.assertEqual(got.read_bytes(), src.read_bytes())

    def test_a_damaged_model_on_the_stick_is_loaded_from_there_with_a_note(self):
        src = self.model()
        data = bytearray(src.read_bytes())
        data[777] ^= 0x01
        src.write_bytes(bytes(data))  # no longer what SHA256SUMS.txt says
        self.dest.mkdir(parents=True)
        got, notes = self.copy(src)
        self.assertEqual(got, src)
        self.assertEqual(len(notes), 1)
        self.assertIn("The model on the USB drive is damaged", notes[0])
        self.assertIn("Build GhostAgent.cmd", notes[0])
        self.assertEqual(list(self.dest.iterdir()), [])  # neither a copy nor a part is left

    def test_the_pin_in_profiles_is_used_without_sums_file(self):
        src = self.model(listed=False)
        with patch_agent("pinned_sha256", return_value="0" * 64):
            self.assertEqual(AGENT.expected_sha256(src), "0" * 64)
            self.dest.mkdir(parents=True)
            got, notes = self.copy(src)
        self.assertEqual(got, src)  # the bytes do not match the pin
        self.assertIn("damaged", notes[0])

    def test_not_enough_space_for_one_file(self):
        src = self.model()
        self.dest.mkdir(parents=True)
        part = self.dest / (src.name + ".part")
        part.write_bytes(src.read_bytes()[:1000])
        with patch_agent("free_bytes", return_value=10):
            got, notes = self.copy(src)
        self.assertEqual(got, src)
        self.assertIn("Not enough free disk space", notes[0])
        self.assertFalse(part.exists())  # a part that cannot be finished only takes space

    def test_an_old_unchecked_copy_is_checked_once(self):
        src = self.model()
        self.dest.mkdir(parents=True)
        (self.dest / src.name).write_bytes(src.read_bytes())  # copied before copies were checked
        got, notes = self.copy(src)
        self.assertEqual((got, notes), (self.dest / src.name, []))
        self.assertIn(src.name, AGENT.read_verified(self.dest))
        # a damaged old copy of the right size is replaced
        data = bytearray(src.read_bytes())
        data[5] ^= 0x10
        (self.dest / src.name).write_bytes(bytes(data))
        (self.dest / ".verified").unlink()
        got, notes = self.copy(src)
        self.assertEqual(got.read_bytes(), src.read_bytes())
        self.assertIn("was damaged; it is copied again", notes[0])

    def test_the_builds_record_is_understood(self):
        src = self.model()
        self.dest.mkdir(parents=True)
        dst = self.dest / src.name
        dst.write_bytes(src.read_bytes())
        st = dst.stat()
        doc = {"note": "Written by tools\\Build-GhostAgent.ps1", "files": {src.name: {
            "name": src.name, "size": st.st_size, "mtime": st.st_mtime_ns / 1e9, "mtime_ns": st.st_mtime_ns,
            "sha256": hashlib.sha256(src.read_bytes()).hexdigest().upper()}}}
        (self.dest / ".verified").write_text(json.dumps(doc), encoding="ascii")
        with patch_agent("_hash_into", side_effect=AssertionError("hashed again")):
            self.assertEqual(self.copy(src)[0], dst)

    def test_cancel_stops_the_copy(self):
        src = self.model()
        self.dest.mkdir(parents=True)
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(AGENT.CopyCancelled):
            self.copy(src, cancel=cancel)

    def test_unused_model_files_are_deleted(self):
        self.dest.mkdir(parents=True)
        for name in ("Holo4-35B-A3B-Q4_K_M.gguf", "Holo4-27B-Q4_K_M.gguf", "old-model.gguf", "old-model.gguf.part",
                     "Holo4-35B-A3B-Q4_K_M.part", "old-model.gguf.layout.json"):
            (self.dest / name).write_bytes(b"x")
        notes = []
        removed = AGENT.prune_model_cache(self.dest, ["Holo4-35B-A3B-Q4_K_M.gguf"], notes)
        self.assertEqual(sorted(removed), ["Holo4-35B-A3B-Q4_K_M.part", "old-model.gguf", "old-model.gguf.part"])
        left = sorted(p.name for p in self.dest.iterdir())
        self.assertEqual(left, ["Holo4-27B-Q4_K_M.gguf", "Holo4-35B-A3B-Q4_K_M.gguf"])  # profiles still use these
        self.assertIn("no model profile uses them any more", notes[0])

    def test_a_model_setting_that_names_the_copy_itself_is_never_deleted(self):
        """copy_model_to_pc true while the model setting already points into the PC's copy folder: the
        source is the destination, so a failed check must not delete the only copy."""
        self.dest.mkdir(parents=True)
        data = bytes(range(256)) * 100
        src = self.dest / "Holo4-35B-A3B-Q4_K_M.gguf"
        src.write_bytes(data)
        with patch_agent("expected_sha256", return_value="0" * 64):  # does not match
            got, notes = self.copy(src)
        self.assertEqual(got, src)
        self.assertEqual(src.read_bytes(), data)
        self.assertEqual(notes, [])

    def test_nothing_is_pruned_without_the_profiles_file(self):
        """Without tools\\profiles.json only the fast profile is known; the other profiles' copies stay."""
        self.dest.mkdir(parents=True)
        for name in ("Holo4-35B-A3B-Q4_K_M.gguf", "Holo4-27B-Q4_K_M.gguf"):
            (self.dest / name).write_bytes(b"x")
        with patch_agent("PROFILES_FILE", self.base / "missing" / "profiles.json"):
            self.assertIs(AGENT.load_profiles(), AGENT.FALLBACK_PROFILES)
            self.assertEqual(AGENT.prune_model_cache(self.dest, ["Holo4-35B-A3B-Q4_K_M.gguf"]), [])
        self.assertTrue((self.dest / "Holo4-27B-Q4_K_M.gguf").exists())

    def test_progress_line(self):
        p = AGENT.Progress(22_200_000_000, "copying the model to this PC")
        p.t0 -= 10
        p.add(680_000_000)
        self.assertEqual(p.line(), "copying the model to this PC: 0.7 of 22.2 GB, 68 MB/s, about 5 min left")
        q = AGENT.Progress(100, "copying")
        q.add(100, skipped=True)
        self.assertEqual(q.rate(), 0.0)  # already in place: not part of the speed


class DecideTests(CopyCase):
    def setUp(self):
        super().setUp()
        self.files = [self.model(), self.model("Holo4-35B-A3B-mmproj-f16.gguf", 50_000, 2)]
        self.s = settings(model=str(self.files[0]), mmproj=str(self.files[1]))

    def decide(self, answer=None, interactive=True, removable=True, free=10**13, **over):
        asked = []

        def ask(question, **kw):
            asked.append(question)
            return answer if answer is not None else (None, "timeout")

        with patch_agent("drive_type", return_value=2 if removable else 3), \
                patch_agent("same_drive", return_value=False), \
                patch_agent("free_bytes", return_value=free), \
                patch_agent("ask_console", ask), quiet():
            got = AGENT.decide_model_copy(dict(self.s, **over), interactive)
        return got, asked

    def test_the_setting_decides_when_it_is_true_or_false(self):
        self.assertEqual(self.decide(copy_model_to_pc=True), (True, []))
        self.assertEqual(self.decide(copy_model_to_pc=False), (False, []))

    def test_asked_once_per_pc(self):
        got, asked = self.decide(("", "answered"))  # Enter = yes
        self.assertTrue(got)
        self.assertIn("Copy the 0.0 GB model to this PC", asked[0])
        self.assertTrue(AGENT.read_pc_settings()["copy_model_to_pc"])
        self.assertEqual(self.decide(("n", "answered")), (True, []))  # not asked again
        AGENT.pc_settings_file().unlink()
        self.assertEqual(self.decide(("n", "answered"))[0], False)
        self.assertIs(AGENT.read_pc_settings()["copy_model_to_pc"], False)
        self.assertEqual(self.decide(("", "answered")), (False, []))

    def test_no_answer_is_not_remembered(self):
        got, asked = self.decide()
        self.assertEqual((got, len(asked)), (False, 1))
        self.assertNotIn("copy_model_to_pc", AGENT.read_pc_settings())
        self.assertEqual(self.decide(interactive=False), (False, []))

    def test_only_asked_when_it_helps_and_fits(self):
        self.assertEqual(self.decide(("", "answered"), removable=False), (False, []))  # a fixed drive, no slow load seen
        self.assertEqual(self.decide(("", "answered"), free=10**9), (False, []))  # less than the model plus 20 GB
        AGENT.write_pc_settings(load_mb_s=60)  # loaded slowly from that drive before
        got, asked = self.decide(("y", "answered"), removable=False)
        self.assertTrue(got)
        self.assertEqual(len(asked), 1)

    def test_a_copy_that_is_there_already_is_used(self):
        self.dest.mkdir(parents=True)
        for f in self.files:
            (self.dest / f.name).write_bytes(f.read_bytes())
        self.assertEqual(self.decide(), (True, []))


class BeginTests(CopyCase):
    def test_the_copy_runs_in_the_background_and_then_the_server_starts(self):
        files = [self.model(size=400_000), self.model("Holo4-35B-A3B-mmproj-f16.gguf", 50_000, 2)]
        s = settings(model=str(files[0]), mmproj=str(files[1]), copy_model_to_pc=True)
        folder = scratch_paths(self)
        exe = folder / "bin" / "cpu" / "llama-server.exe"
        exe.parent.mkdir(parents=True)
        launched, awake = [], []
        server = AGENT.LlamaServer(s)

        class Awake:
            def __enter__(self):
                awake.append(threading.current_thread().name)

            def __exit__(self, *a):
                return False

        with patch_agent("pick_backend", return_value=("cpu", exe, None)), \
                patch_agent("COPY_CHUNK", 64 * 1024), \
                patch_agent("keep_pc_awake", Awake), \
                mock.patch.object(AGENT.subprocess, "Popen", lambda args, **kw: launched.append(args) or Proc()), \
                patch_agent("port_is_free", return_value=True), \
                patch_agent("server_healthy", return_value=True), quiet() as out:
            self.assertEqual(server.begin(interactive=False), "cpu")
            self.assertIsNotNone(server.wait_ready())
        self.assertEqual(awake, ["model-start"])  # the PC stays awake while it copies
        args = launched[0]
        self.assertEqual(Path(args[args.index("-m") + 1]), self.dest / files[0].name)
        self.assertEqual(Path(args[args.index("--mmproj") + 1]), self.dest / files[1].name)
        self.assertIn("copying the model to this PC", out.getvalue())
        self.assertEqual((self.dest / files[0].name).read_bytes(), files[0].read_bytes())

    def test_stop_cancels_a_copy_in_progress(self):
        files = [self.model(size=2_000_000), self.model("Holo4-35B-A3B-mmproj-f16.gguf", 50_000, 2)]
        s = settings(model=str(files[0]), mmproj=str(files[1]), copy_model_to_pc=True)
        folder = scratch_paths(self)
        exe = folder / "bin" / "cpu" / "llama-server.exe"
        exe.parent.mkdir(parents=True)
        launched = []
        server = AGENT.LlamaServer(s)
        real_hash = AGENT._hash_into
        started = threading.Event()

        def slow_add(self_progress, n, skipped=False):
            started.set()
            time.sleep(0.05)
            self_progress.done += n

        with patch_agent("pick_backend", return_value=("cpu", exe, None)), \
                patch_agent("COPY_CHUNK", 16 * 1024), \
                mock.patch.object(AGENT.Progress, "add", slow_add), \
                mock.patch.object(AGENT.subprocess, "Popen", lambda args, **kw: launched.append(args) or Proc()), \
                patch_agent("_hash_into", real_hash), quiet():
            server.begin(interactive=False)
            self.assertTrue(started.wait(10))
            server.stop()
        self.assertEqual(launched, [])  # nothing was started after the stop
        self.assertFalse((self.dest / files[0].name).exists())
        self.assertTrue((self.dest / (files[0].name + ".part")).exists())  # resumed next time


if __name__ == "__main__":
    unittest.main()
