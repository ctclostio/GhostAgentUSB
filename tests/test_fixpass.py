"""Regression tests for the fix pass after 0.2.0: the VRAM margin on every --fit-target path, the image
token cap, trace addresses, the seed range, the short form of the home folder, the per-agent browser
cache, Ctrl+C during a console question, the Visual C++ runtime check, profile fields, model
checksums, plain output when not on a console, prompt-cache safe shortening, key chords, tab titles
and llama.cpp builds missing from the drive. (The bench server's HEAD requests and the fake model's
chat count are tested in test_infra.py, the build script's changes in test_build_script.py.)

No GPU, no model and no network beyond 127.0.0.1."""

import contextlib
import json
import re
import signal
import types
import unittest
import unittest.mock as mock
from pathlib import Path

import helpers
from agent_support import AGENT, StubBrowser, call, make_agent, quiet, reset_stop, scratch_paths, settings
from security_support import SiteBrowserCase, use_keyboard

DEVICE = {"id": "CUDA0", "name": "NVIDIA GeForce RTX 4090", "total": 24564, "free": 22000, "integrated": False}


class Proc:
    """Popen's stand-in: a process that runs until terminated."""
    _handle = 0
    pid = 4

    def __init__(self, code=None):
        self.code = code

    def poll(self):
        return self.code

    def terminate(self):
        self.code = 1

    def kill(self):
        self.code = 1

    def wait(self, timeout=None):
        return self.code


def start_server(test, s, device=DEVICE, backend="cuda13", supported=None, server=None, files=None):
    """LlamaServer.start with the build, the model files and Popen replaced: (args, server, printed)."""
    folder = scratch_paths(test)
    exe = folder / "bin" / backend / "llama-server.exe"
    exe.parent.mkdir(parents=True, exist_ok=True)
    launched = []
    server = server or AGENT.LlamaServer(s)
    with mock.patch.object(AGENT, "pick_backend", return_value=(backend, exe, device)), \
            mock.patch.object(AGENT, "model_files", return_value=files or [Path("m.gguf"), Path("p.gguf")]), \
            mock.patch.object(AGENT, "server_flags", return_value=supported), \
            mock.patch.object(AGENT, "model_layout", return_value=None), \
            mock.patch.object(AGENT, "vc_runtime_missing", return_value=[]), \
            mock.patch.object(AGENT.subprocess, "Popen", lambda args, **kw: launched.append(args) or Proc()), \
            mock.patch.object(AGENT, "port_is_free", return_value=True), \
            mock.patch.object(AGENT, "port_bindable", return_value=True), quiet() as out:
        server.start()
    return launched[-1], server, out.getvalue()


class FitTargetMarginTests(unittest.TestCase):
    """llama.cpp's --fit never counts the vision projector, so every --fit-target path leaves room for it."""

    def projector_margin(self, mmproj_mib):
        return mmproj_mib + AGENT.VISION_RESERVE_MIB

    def test_moe_low_vram_fallback_counts_the_projector(self):
        args, explicit = AGENT.gpu_layout_args(dict(DEVICE, free=5000), settings(), reserve_mib=4326, mmproj_mib=858)
        self.assertFalse(explicit)
        self.assertEqual(args, ["--device", "CUDA0", "--fit-target", str(1024 + self.projector_margin(858))])

    def test_moe_auto_fit_retry_counts_the_projector(self):
        args, explicit = AGENT.gpu_layout_args(DEVICE, settings(), auto_fit=True, mmproj_mib=858)
        self.assertFalse(explicit)
        self.assertEqual(args[-2:], ["--fit-target", str(1024 + self.projector_margin(858))])
        # With the laptop emulation the emulated difference is left free as well.
        args, _ = AGENT.gpu_layout_args(DEVICE, settings(), auto_fit=True, emulate_free_mib=7070, mmproj_mib=858)
        self.assertEqual(args[-1], str(1024 + (22000 - 7070) + self.projector_margin(858)))

    def test_dense_path_is_unchanged(self):
        args, explicit = AGENT.gpu_layout_args(DEVICE, settings(), dense=True, mmproj_mib=885)
        self.assertFalse(explicit)
        self.assertEqual(args[-1], str(1024 + self.projector_margin(885)))

    def test_explicit_moe_layout_has_no_fit_target(self):
        # The explicit layout's reserve already counts the projector (moe_reserve_mib).
        args, explicit = AGENT.gpu_layout_args(DEVICE, settings(), reserve_mib=4326, mmproj_mib=858)
        self.assertTrue(explicit)
        self.assertNotIn("--fit-target", args)


class ImageMaxTokensTests(unittest.TestCase):
    def test_the_cap_is_passed_next_to_the_minimum(self):
        args, _, _ = start_server(self, settings())
        i = args.index("--image-min-tokens")
        self.assertEqual(args[i:i + 4], ["--image-min-tokens", "1024", "--image-max-tokens", "1280"])
        self.assertEqual((AGENT.MIN_IMAGE_TILES, AGENT.MAX_IMAGE_TILES), (1024, 1280))
        # 1280 tiles of 32x32 pixels is the big-screenshot warning's 1.3 megapixels.
        self.assertAlmostEqual(AGENT.MAX_IMAGE_TILES * AGENT.TILE_PX ** 2 / AGENT.BIG_SCREENSHOT_PIXELS, 1.0, delta=0.02)

    def test_a_build_without_it_starts_without_it(self):
        known = set(AGENT.REQUIRED_SERVER_FLAGS) | set(AGENT.OPTIONAL_SERVER_FLAGS)
        args, _, out = start_server(self, settings(), supported=known - {"--image-max-tokens"})
        self.assertNotIn("--image-max-tokens", args)
        self.assertIn("--image-min-tokens", args)
        self.assertIn("does not know --image-max-tokens; started without it", out)
        self.assertNotIn("locked down less tightly", out)  # not a lock-down flag

    def test_it_is_optional_everywhere(self):
        doc = json.loads((helpers.REPO / "tools" / "server-flags.json").read_text(encoding="utf-8"))
        self.assertIn("--image-max-tokens", doc["optional"])
        self.assertIn("--image-max-tokens", doc["_optional_when"])
        self.assertIn("--image-max-tokens", AGENT.OPTIONAL_SERVER_FLAGS)
        self.assertNotIn("--image-max-tokens", AGENT.REQUIRED_SERVER_FLAGS)


class FlagListTests(unittest.TestCase):
    def test_agent_and_file_agree_on_the_lockdown_flags(self):
        doc = json.loads((helpers.REPO / "tools" / "server-flags.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["lockdown"], list(AGENT.LOCKDOWN_SERVER_FLAGS))
        # --no-webui is passed by default but optional everywhere: web_ui leaves it out, and a build
        # without it starts without it (with a warning, as a lock-down flag).
        self.assertIn("--no-webui", AGENT.OPTIONAL_SERVER_FLAGS)
        self.assertNotIn("--no-webui", doc["required"])

    def test_a_build_without_no_webui_warns(self):
        known = set(AGENT.REQUIRED_SERVER_FLAGS) | set(AGENT.OPTIONAL_SERVER_FLAGS)
        args, _, out = start_server(self, settings(), supported=known - {"--no-webui"})
        self.assertNotIn("--no-webui", args)
        self.assertIn("locked down less tightly", out)
        args, _, _ = start_server(self, settings(web_ui=True))
        self.assertNotIn("--no-webui", args)


class TraceUrlTests(unittest.TestCase):
    def test_ipv6_hosts_keep_their_brackets(self):
        u = AGENT.trace_url
        self.assertEqual(u("http://[::1]:8080/a?q=secret"), "http://[::1]:8080/a")
        self.assertEqual(u("https://[2001:DB8::1]/path#frag"), "https://[2001:db8::1]/path")
        self.assertEqual(u("http://user:pw@[::1]/x"), "http://[::1]/x")
        # In an error line too, and IPv4 and names are as before.
        self.assertEqual(AGENT.trace_error("net::ERR_FAILED at http://[::1]:8080/a?q=secret"),
                         "net::ERR_FAILED at http://[::1]:8080/a")
        self.assertEqual(u("http://127.0.0.1:8765/a?q=1"), "http://127.0.0.1:8765/a")
        self.assertEqual(u("https://Example.test/b"), "https://example.test/b")


class SeedRangeTests(unittest.TestCase):
    def apply(self, value):
        s = settings()
        with quiet() as out:
            AGENT.apply_settings(s, {"seed": value}, "settings.json")
        return s["seed"], out.getvalue()

    def test_seed_is_minus_one_or_32_bits(self):
        for good in (-1, 0, 1, 42, 2**32 - 1, "4294967295"):
            value, out = self.apply(good)
            self.assertEqual((value, out), (int(good), ""), good)
        for bad in (-2, 2**32, 2**40, "4294967296"):
            value, out = self.apply(bad)
            self.assertEqual(value, -1, bad)
            self.assertIn("-1 (random) or a whole number from 0 to 4294967295", out)

    def test_the_command_line_override_is_checked_too(self):
        scratch_paths(self)
        with quiet() as out:
            s = AGENT.load_settings(override={"seed": 2**33}, interactive=False)
        self.assertEqual(s["seed"], -1)
        self.assertIn('--settings-override: "seed" is 8589934592', out.getvalue())
        with quiet():
            self.assertEqual(AGENT.load_settings(override={"seed": 7}, interactive=False)["seed"], 7)

    def test_replay_seed_is_checked(self):
        import replay
        for bad in ("-5", str(2**32)):
            with quiet() as out, self.assertRaises(SystemExit) as cm:
                replay.main(["--replay-dir", ".", "--seed", bad])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("--seed must be -1 (random)", out.getvalue())


class HideShortHomeTests(unittest.TestCase):
    def setUp(self):
        AGENT._HOME_FORMS.clear()
        self.addCleanup(AGENT._HOME_FORMS.clear)

    def test_the_short_form_of_the_home_folder_is_hidden_too(self):
        long_home = "C:\\Users\\Long Name Here"
        short_home = "C:\\Users\\LONGNA~1"
        asked = []

        def fake_short(path):
            asked.append(path)
            return short_home if path.lower() == long_home.lower() else path

        with mock.patch.object(AGENT.Path, "home", return_value=Path(long_home)), \
                mock.patch.object(AGENT, "short_path", fake_short):
            h = AGENT.hide_home
            self.assertEqual(h(f"{long_home}\\AppData\\x.gguf"), "%USERPROFILE%\\AppData\\x.gguf")
            self.assertEqual(h(f"loading {short_home}\\AppData\\Local\\m.gguf"), "loading %USERPROFILE%\\AppData\\Local\\m.gguf")
            self.assertEqual(h("c:/users/longna~1/x"), "%USERPROFILE%/x")
            self.assertEqual(h(f'"{short_home}"'), '"%USERPROFILE%"')
            self.assertEqual(h(short_home + "2\\x"), short_home + "2\\x")  # only that exact folder
            self.assertEqual(h("C:\\Users\\LONGNA~2\\x"), "C:\\Users\\LONGNA~2\\x")
        self.assertEqual(asked, [long_home])  # worked out once per home folder

    def test_same_short_and_long_form_is_harmless(self):
        home = "C:\\Users\\bob"
        with mock.patch.object(AGENT.Path, "home", return_value=Path(home)), \
                mock.patch.object(AGENT, "short_path", lambda p: p):
            self.assertEqual(AGENT.hide_home(home + "\\x"), "%USERPROFILE%\\x")
            self.assertEqual(AGENT.hide_home("C:\\Users\\bobby\\x"), "C:\\Users\\bobby\\x")

    def test_short_path_of_a_real_folder(self):
        folder = helpers.temp_dir(self, prefix="a folder with a long name ")
        short = AGENT.short_path(folder)
        self.assertEqual(Path(short).resolve(), Path(folder).resolve())  # the same folder either way
        missing = str(folder / "not there")
        self.assertEqual(AGENT.short_path(missing), missing)


class PerAgentBrowserCacheTests(unittest.TestCase):
    """Two agents with other data folders (external_server and --data-dir) each have their own browser
    cache in TEMP, and neither deletes the other's."""

    def setUp(self):
        scratch_paths(self)
        self.temp = helpers.temp_dir(self)
        patcher = mock.patch.dict(AGENT.os.environ, {"TEMP": str(self.temp), "TMP": str(self.temp)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def launch(self, profile):
        from test_security_browser import FakePlaywright
        with mock.patch.object(AGENT, "PROFILE", profile):
            b = AGENT.Browser(settings(browser="bundled"))
            self.addCleanup(b.close)
            b.pw = FakePlaywright()
            with quiet():
                b._launch_own("https://example.com/")
        cache = Path(b.pw.calls[0]["args"][0].split("=", 1)[1])
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "Cache_Data").mkdir(exist_ok=True)
        (cache / "Cache_Data" / "data_1").write_bytes(b"page bytes")  # what the browser writes while it runs
        return b, cache

    def test_one_cache_per_profile(self):
        a = AGENT.browser_cache_dir(self.temp / "one" / "browser-profile")
        b = AGENT.browser_cache_dir(self.temp / "two" / "browser-profile")
        self.assertNotEqual(a, b)
        self.assertEqual(a, AGENT.browser_cache_dir(str(self.temp / "ONE" / "browser-profile")))  # Windows: case
        for folder in (a, b):
            self.assertEqual(folder.parent, self.temp)
            self.assertRegex(folder.name, r"^ghost-agent-browser-cache-[0-9a-f]{12}$")

    def test_concurrent_agents_keep_each_others_cache(self):
        first, cache1 = self.launch(self.temp / "one" / "browser-profile")
        second, cache2 = self.launch(self.temp / "two" / "browser-profile")
        self.assertNotEqual(cache1, cache2)
        self.assertTrue((cache1 / "Cache_Data" / "data_1").exists())  # the second start left it alone
        second.close()
        self.assertFalse(cache2.exists())
        self.assertTrue((cache1 / "Cache_Data" / "data_1").exists())  # the second close too
        first.close()
        self.assertFalse(cache1.exists())

    def test_a_second_agent_on_the_same_profile_gets_a_cache_of_its_own(self):
        # external_server lets two agents run with one data folder. The second one's browser finds the
        # profile in use and may fall back to Edge: it must neither share nor delete the first one's cache.
        profile = self.temp / "one" / "browser-profile"
        first, cache = self.launch(profile)
        second, other = self.launch(profile)
        self.assertNotEqual(cache, other)
        self.assertEqual(other.parent, self.temp)
        self.assertTrue(AGENT.is_browser_cache_name(other.name))  # so a crash's leftovers are pruned too
        self.assertIsNotNone(second._cache_lock)
        self.assertTrue((cache / "Cache_Data" / "data_1").exists())
        second.close()
        self.assertFalse(other.exists())
        self.assertTrue((cache / "Cache_Data" / "data_1").exists())
        first.close()
        self.assertFalse(cache.exists())

    def test_what_cannot_be_deleted_yet_keeps_its_lock_file(self):
        folder = self.temp / "ghost-agent-browser-cache-0123456789ab"
        lock = AGENT.FolderLock.take(folder)
        self.assertTrue(AGENT.FolderLock.holds(folder))
        busy = folder / "Cache_Data" / "data_1"
        busy.parent.mkdir()
        busy.write_bytes(b"x")
        with open(busy, "rb"):  # a file the browser has not let go of yet
            AGENT.remove_browser_cache(folder, lock)
            self.assertFalse(AGENT.FolderLock.holds(folder))
            self.assertTrue((folder / AGENT.CACHE_LOCK_FILE).exists())
        # So the next start deletes the rest at once, instead of after UNLOCKED_CACHE_AGE.
        self.assertEqual(AGENT.prune_browser_caches(), 1)
        self.assertFalse(folder.exists())

    def test_stale_caches_go_and_held_ones_stay(self):
        def make(name, lock=False):
            folder = self.temp / name
            (folder / "Cache_Data").mkdir(parents=True)
            (folder / "Cache_Data" / "data_0").write_bytes(b"x")
            if lock:
                (folder / AGENT.CACHE_LOCK_FILE).write_bytes(b"")
            return folder

        crashed = make("ghost-agent-browser-cache-0123456789ab", lock=True)   # left by a crash: no holder
        held = make("ghost-agent-browser-cache-ba9876543210", lock=True)      # another running agent
        holder = AGENT.FolderLock.take(held)
        self.assertIsNotNone(holder)
        self.addCleanup(holder.release)
        self.assertIsNone(AGENT.FolderLock.take(held))  # a second taker is refused
        legacy = make("ghost-agent-browser-cache")                            # from 0.2.0, no lock file
        old = AGENT.time.time() - AGENT.UNLOCKED_CACHE_AGE - 60
        AGENT.os.utime(legacy, (old, old))
        young = make("ghost-agent-browser-cache-00000000000f")                # being set up right now
        unrelated = make("ghost-agent-browser-cache-notes")                   # not a name the agent uses
        AGENT.os.utime(unrelated, (old, old))
        mine = make("ghost-agent-browser-cache-aaaaaaaaaaaa", lock=True)

        self.assertEqual(AGENT.prune_browser_caches(keep=mine), 2)
        self.assertFalse(crashed.exists())
        self.assertFalse(legacy.exists())
        for kept in (held, young, unrelated, mine):
            self.assertTrue((kept / "Cache_Data" / "data_0").exists(), kept)
        holder.release()
        self.assertEqual(AGENT.prune_browser_caches(keep=mine), 1)  # no longer held
        self.assertFalse(held.exists())

    def test_forget_pc_finds_caches_that_earlier_runs_left(self):
        import builtins
        self.enterContext(helpers.isolated_env(TEMP=self.temp, TMP=self.temp))

        def make(name):
            folder = self.temp / name
            (folder / "Cache_Data").mkdir(parents=True)
            (folder / "Cache_Data" / "data_0").write_bytes(b"x")
            (folder / AGENT.CACHE_LOCK_FILE).write_bytes(b"")
            return folder

        # A crash under another drive letter (another profile path) left one; another agent uses one.
        left = make("ghost-agent-browser-cache-0123456789ab")
        held = make("ghost-agent-browser-cache-ba9876543210")
        holder = AGENT.FolderLock.take(held)
        self.addCleanup(holder.release)
        AGENT.FolderLock.held.discard(holder.key)  # as if another process held it
        self.assertFalse((self.temp / AGENT.browser_cache_dir().name).exists())  # this agent's own: none

        listed = AGENT.pc_leftovers()
        self.assertIn(("a browser cache left by an earlier run", left), listed)
        self.assertNotIn(held, [p for _, p in listed])
        later = []
        with quiet() as out, mock.patch.object(builtins, "input", lambda prompt="": "y"):
            AGENT.forget_pc(settings(), later)
        self.assertNotIn("keeps nothing", out.getvalue())  # before the fix it said so, and kept the cache
        self.assertFalse(left.exists())
        self.assertTrue((held / "Cache_Data" / "data_0").exists())
        self.assertEqual(later, [])

    def test_launch_clears_stale_caches(self):
        stale = self.temp / "ghost-agent-browser-cache-0123456789ab"
        stale.mkdir()
        (stale / AGENT.CACHE_LOCK_FILE).write_bytes(b"")
        b, cache = self.launch(self.temp / "one" / "browser-profile")
        self.assertFalse(stale.exists())
        self.assertTrue(cache.exists())


class VcRuntimeTests(unittest.TestCase):
    """llama-server needs the Visual C++ runtime, which is not on the stick: missing DLLs get plain advice."""

    def setUp(self):
        self.folder = scratch_paths(self)
        self.windows = helpers.temp_dir(self)
        (self.windows / "System32").mkdir()
        patcher = mock.patch.dict(AGENT.os.environ, {"SystemRoot": str(self.windows)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.exe = self.folder / "bin" / "cpu" / "llama-server.exe"
        self.exe.parent.mkdir(parents=True)
        self.exe.write_bytes(b"MZ")

    def test_missing_dlls_are_found_by_file(self):
        self.assertEqual(AGENT.vc_runtime_missing(self.exe.parent), list(AGENT.VC_RUNTIME_DLLS))
        (self.windows / "System32" / "msvcp140.dll").write_bytes(b"x")
        (self.exe.parent / "vcruntime140.dll").write_bytes(b"x")  # next to the exe counts too
        self.assertEqual(AGENT.vc_runtime_missing(self.exe.parent), ["vcruntime140_1.dll"])
        (self.windows / "System32" / "vcruntime140_1.dll").write_bytes(b"x")
        self.assertEqual(AGENT.vc_runtime_missing(self.exe.parent), [])

    def test_the_advice(self):
        text = AGENT.vc_runtime_advice(["msvcp140.dll"])
        self.assertIn('"Microsoft Visual C++ Redistributable (x64)"', text)
        self.assertIn("https://aka.ms/vs/17/release/vc_redist.x64.exe", text)
        self.assertIn("administrator rights", text)
        self.assertIn("msvcp140.dll", text)
        self.assertTrue(AGENT.dll_not_found(0xC0000135))
        self.assertTrue(AGENT.dll_not_found(-1073741515))  # the same code, as a signed number
        self.assertFalse(AGENT.dll_not_found(1))
        self.assertFalse(AGENT.dll_not_found(None))

    def test_an_arm64_drive_gets_the_arm64_redistributable(self):
        # The build puts ARM64 Python and ARM64 llama.cpp builds on a drive built on an ARM PC; those need
        # the ARM64 runtime. An x64 drive needs the x64 one, also when it runs emulated on an ARM PC.
        with mock.patch.object(AGENT.sys, "version", "3.14.8 (tags/v3.14.8) [MSC v.1944 64 bit (ARM64)]"):
            text = AGENT.vc_runtime_advice(["msvcp140.dll"])
        self.assertIn('"Microsoft Visual C++ Redistributable (ARM64)"', text)
        self.assertIn("https://aka.ms/vs/17/release/vc_redist.arm64.exe", text)
        with mock.patch.object(AGENT.sys, "version", "3.14.8 (tags/v3.14.8) [MSC v.1944 64 bit (AMD64)]"):
            self.assertEqual(AGENT.vc_redist(), ("x64", "https://aka.ms/vs/17/release/vc_redist.x64.exe"))

    def test_a_warning_before_the_first_build_runs(self):
        server = AGENT.LlamaServer(settings())
        with mock.patch.object(AGENT, "BIN", self.folder / "bin"), \
                mock.patch.object(AGENT, "pick_backend", return_value=("cpu", self.exe, None)), quiet() as out:
            server._choose_backend()
            server._choose_backend()
        self.assertEqual(out.getvalue().count("vc_redist.x64.exe"), 1)  # said once
        for dll in AGENT.VC_RUNTIME_DLLS:
            (self.windows / "System32" / dll).write_bytes(b"x")
        server = AGENT.LlamaServer(settings())
        with mock.patch.object(AGENT, "BIN", self.folder / "bin"), \
                mock.patch.object(AGENT, "pick_backend", return_value=("cpu", self.exe, None)), quiet() as out:
            server._choose_backend()
        self.assertNotIn("Visual C++", out.getvalue())

    def test_list_devices_stops_with_the_advice(self):
        done = AGENT.subprocess.CompletedProcess([], 0xC0000135, stdout="", stderr="")
        with mock.patch.object(AGENT.subprocess, "run", return_value=done), self.assertRaises(SystemExit) as cm:
            AGENT.list_devices(self.exe)
        self.assertIn("vc_redist.x64.exe", str(cm.exception))
        # With the runtime there, a missing DLL is the build's own (a CUDA one): the next build is tried.
        for dll in AGENT.VC_RUNTIME_DLLS:
            (self.windows / "System32" / dll).write_bytes(b"x")
        with mock.patch.object(AGENT.subprocess, "run", return_value=done):
            self.assertEqual(AGENT.list_devices(self.exe), [])

    def wait_for_dead_server(self):
        server = AGENT.LlamaServer(settings())
        server.choice = ("cpu", self.exe, None)
        server.proc = Proc(code=0xC0000135)
        server.log_path = self.folder / "logs" / "llama-server-x.log"
        server.log_path.parent.mkdir(parents=True, exist_ok=True)
        server.log_path.write_text("command: llama-server.exe\n", encoding="utf-8")
        with quiet() as out:
            return server.wait_ready(), out

    def test_a_server_that_dies_for_want_of_the_runtime(self):
        with self.assertRaises(SystemExit) as cm:
            self.wait_for_dead_server()
        self.assertIn('Install the "Microsoft Visual C++ Redistributable (x64)"', str(cm.exception))
        for dll in AGENT.VC_RUNTIME_DLLS:
            (self.windows / "System32" / dll).write_bytes(b"x")
        secs, out = self.wait_for_dead_server()
        self.assertIsNone(secs)
        self.assertIn("could not find a DLL that cpu\\llama-server.exe needs", out.getvalue())


class ProfileFieldTests(unittest.TestCase):
    """tools\\profiles.json says what the agent used to infer: takes_reasoning_effort, preserve_reasoning, ram_gb."""

    def test_the_file_says_what_the_agent_inferred(self):
        doc = json.loads((helpers.REPO / "tools" / "profiles.json").read_text(encoding="utf-8"))
        for name, prof in doc["profiles"].items():
            dense = prof["arch"] == "dense"
            self.assertIs(prof["takes_reasoning_effort"], dense, name)  # true only for accurate
            self.assertIs(prof["preserve_reasoning"], dense, name)      # "auto": accurate on, the fast ones off
            self.assertEqual(prof["ram_gb"], AGENT.RAM_NEED_GB[name], name)
        self.assertEqual([n for n, p in doc["profiles"].items() if p["takes_reasoning_effort"]], ["accurate"])

    def test_the_agent_reads_the_fields(self):
        def with_profile(prof):
            return mock.patch.object(AGENT, "active_profile", return_value=("x", prof))

        cases = [({"arch": "dense"}, True), ({"arch": "moe"}, False),            # inferred
                 ({"arch": "dense", "takes_reasoning_effort": False}, False),    # said: wins
                 ({"arch": "moe", "takes_reasoning_effort": True}, True),
                 ({"arch": "moe", "takes_reasoning_effort": "yes"}, False)]      # not true/false: inferred
        for prof, want in cases:
            with with_profile(prof):
                self.assertIs(AGENT.takes_reasoning_effort(settings()), want, prof)
        cases = [({"arch": "dense"}, True), ({"arch": "moe"}, False),
                 ({"arch": "dense", "preserve_reasoning": False}, False),
                 ({"arch": "moe", "preserve_reasoning": True}, True),
                 ({"arch": "dense", "preserve_reasoning": "auto"}, True)]
        for prof, want in cases:
            with with_profile(prof):
                self.assertIs(AGENT.preserves_reasoning(settings()), want, prof)
                self.assertFalse(AGENT.preserves_reasoning(settings(preserve_reasoning=False)))  # the setting wins

    def test_the_real_profiles(self):
        self.assertTrue(AGENT.takes_reasoning_effort(settings(model_profile="accurate")))
        self.assertFalse(AGENT.takes_reasoning_effort(settings(model_profile="fast-q8")))
        self.assertTrue(AGENT.preserves_reasoning(settings(model_profile="accurate")))
        self.assertFalse(AGENT.preserves_reasoning(settings(model_profile="fast")))

    def test_ram_comes_from_the_profile(self):
        folder = helpers.temp_dir(self)
        doc = json.loads((helpers.REPO / "tools" / "profiles.json").read_text(encoding="utf-8"))
        doc["profiles"]["fast"]["ram_gb"] = 30
        del doc["profiles"]["accurate"]["ram_gb"]          # missing: the agent's own table
        doc["profiles"]["fast-q8"]["ram_gb"] = "lots"      # not a number: the agent's own table
        path = folder / "profiles.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with mock.patch.object(AGENT, "PROFILES_FILE", path):
            self.assertEqual(AGENT.ram_need_gb(settings(model_profile="fast")), 30)
            self.assertEqual(AGENT.ram_need_gb(settings(model_profile="accurate")), AGENT.RAM_NEED_GB["accurate"])
            self.assertEqual(AGENT.ram_need_gb(settings(model_profile="fast-q8")), AGENT.RAM_NEED_GB["fast-q8"])
        self.assertEqual(AGENT.ram_need_gb(settings(model_profile="fast")), 22)  # the real file


class ModelChecksumFileTests(unittest.TestCase):
    """GhostAgent\\models\\SHA256SUMS.txt, SOURCE.txt and tools\\profiles.json describe the same files."""

    MODELS = helpers.REPO / "GhostAgent" / "models"

    def profiles(self):
        return json.loads((helpers.REPO / "tools" / "profiles.json").read_text(encoding="utf-8"))["profiles"]

    def test_sha256sums_lists_every_file_of_every_profile(self):
        pinned = {}
        for name, prof in self.profiles().items():
            for part in ("model", "mmproj"):
                f = prof[part]
                self.assertEqual(pinned.setdefault(f["name"], f["sha256"].lower()), f["sha256"].lower(),
                                 f"{f['name']} is pinned to two hashes")
        raw = (self.MODELS / "SHA256SUMS.txt").read_bytes()
        self.assertEqual(raw.count(b"\r\n"), raw.count(b"\n"), "SHA256SUMS.txt keeps CRLF line ends")
        listed = {}
        for line in raw.decode("ascii").splitlines():
            m = re.fullmatch(r"([0-9a-f]{64}) \*(\S+)", line)
            self.assertIsNotNone(m, line)
            self.assertNotIn(m[2], listed, f"{m[2]} is listed twice")
            listed[m[2]] = m[1]
        self.assertEqual(listed, pinned)

    def test_the_agent_finds_each_sum_there(self):
        for name, sha in ((f["name"], f["sha256"]) for p in self.profiles().values() for f in (p["model"], p["mmproj"])):
            self.assertEqual(AGENT.expected_sha256(self.MODELS / name), sha.lower())

    def test_source_txt_names_every_profile_and_its_source(self):
        text = (self.MODELS / "SOURCE.txt").read_text(encoding="ascii")
        squash = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())  # noqa: E731
        lines = {line.split(":", 1)[0].strip(): line for line in text.splitlines() if line.startswith("  ")}
        self.assertEqual(set(lines), set(self.profiles()))
        for name, prof in self.profiles().items():
            line = lines[name]
            self.assertIn(prof["model"]["url"].split("/resolve/")[0], line, f"{name}: the model's repository")
            mmproj_repo, mmproj_file = prof["mmproj"]["url"].split("/resolve/")[0], prof["mmproj"]["url"].rsplit("/", 1)[1]
            self.assertTrue(mmproj_repo in line or mmproj_file in line, f"{name}: the projector's source")
            self.assertIn(squash(prof["license"]), squash(line), f"{name}: license")


class CtrlCDuringQuestionTests(unittest.TestCase):
    """A console question puts its own Ctrl+C handler over the task's; two presses within 3 seconds
    still quit, whichever handler gets each one."""

    def setUp(self):
        reset_stop(self)
        scratch_paths(self)
        self.clock = [1000.0]
        fake_time = types.SimpleNamespace(monotonic=lambda: self.clock[0], sleep=lambda s: None)
        patcher = mock.patch.object(AGENT, "time", fake_time)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def press():
        """Ctrl+C, delivered to whichever handler is installed right now."""
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)

    def test_press_in_the_question_then_in_the_task_quits(self):
        use_keyboard(self, self.press)
        with quiet() as out, AGENT.ctrl_c_stops_task():
            self.assertEqual(AGENT.ask_console("Allow this?"), (None, "stopped"))
            self.assertTrue(AGENT.STOP.is_set())
            self.clock[0] += 2.0
            with self.assertRaises(AGENT.ForcedStop):
                self.press()  # the task's handler: before the fix this took a third press
        self.assertIn("press Ctrl+C again within 3 s to quit GhostAgent now", out.getvalue())

    def test_ctrl_c_typed_as_a_key_counts_as_the_first_press(self):
        use_keyboard(self, "\x03")
        with quiet(), AGENT.ctrl_c_stops_task():
            self.assertEqual(AGENT.ask_console("Allow this?"), (None, "stopped"))
            self.clock[0] += 1.0
            with self.assertRaises(AGENT.ForcedStop):
                self.press()

    def test_a_late_second_press_only_stops(self):
        use_keyboard(self, self.press)
        with quiet(), AGENT.ctrl_c_stops_task():
            AGENT.ask_console("Allow this?")
            self.clock[0] += AGENT.FORCE_QUIT_SECONDS + 0.5
            self.press()  # more than 3 seconds later: just another first press
            self.assertTrue(AGENT.STOP.is_set())
            self.clock[0] += 1.0
            with self.assertRaises(AGENT.ForcedStop):
                self.press()

    def test_a_new_task_forgets_the_last_press(self):
        with quiet():
            with AGENT.ctrl_c_stops_task():
                self.press()
            with AGENT.ctrl_c_stops_task():  # the next task, a moment later
                AGENT.STOP.set()             # (stopped for another reason)
                self.clock[0] += 0.5
                self.press()                 # a first press, not a second one
                self.assertTrue(AGENT.STOP.is_set())


class _Tty:
    """A stream that says whether it is a console."""

    def __init__(self, answer=True):
        self.answer = answer

    def isatty(self):
        return self.answer

    def reconfigure(self, **kwargs):
        pass


class PlainOutputTests(unittest.TestCase):
    """Output that is not a console (piped, or redirected to a file as unattended runs often are) gets
    no colour codes and no lines redrawn in place with '\\r', only plain lines."""

    def setUp(self):
        self.clock = [500.0]
        fake_time = types.SimpleNamespace(monotonic=lambda: self.clock[0], time=lambda: self.clock[0],
                                          sleep=lambda s: None)
        for patcher in (mock.patch.object(AGENT, "time", fake_time),
                        mock.patch.object(AGENT, "PLAIN_OUTPUT", AGENT.PLAIN_OUTPUT)):  # restored afterwards
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_only_a_console_counts_as_one(self):
        self.assertTrue(AGENT.output_is_console(_Tty()))
        self.assertFalse(AGENT.output_is_console(_Tty(False)))
        self.assertFalse(AGENT.output_is_console(object()))  # no isatty at all

        class Closed:
            def isatty(self):
                raise ValueError("I/O operation on closed file")
        self.assertFalse(AGENT.output_is_console(Closed()))

    def test_start_up_picks_plain_output_for_a_pipe(self):
        with mock.patch.object(AGENT.os, "system") as system:
            with mock.patch.object(AGENT.sys, "stdout", _Tty(False)), mock.patch.object(AGENT.sys, "stderr", _Tty(False)):
                AGENT._enable_ansi()
            self.assertTrue(AGENT.PLAIN_OUTPUT)
            system.assert_not_called()  # (the call that turns on colour in a Windows console)
            with mock.patch.object(AGENT.sys, "stdout", _Tty()), mock.patch.object(AGENT.sys, "stderr", _Tty()):
                AGENT._enable_ansi()
            self.assertFalse(AGENT.PLAIN_OUTPUT)

    def test_no_colour_codes_in_plain_output(self):
        AGENT.PLAIN_OUTPUT = True
        self.assertEqual(AGENT.c("hello", "33"), "hello")
        with quiet() as out:
            AGENT.info("a note")
            AGENT.warn("careful")
            AGENT.err("broken")
            AGENT.head("[1/1] task")
        self.assertEqual(out.getvalue(), "  a note\n  ! careful\n  x broken\n[1/1] task\n")
        AGENT.PLAIN_OUTPUT = False
        self.assertEqual(AGENT.c("hello", "33"), "\x1b[33mhello\x1b[0m")

    def test_a_live_line_writes_plain_lines_now_and_then(self):
        AGENT.PLAIN_OUTPUT = True
        with quiet() as out:
            live = AGENT.LiveLine()
            live.show("loading 10%")
            self.clock[0] += 5
            live.show("loading 20%")  # too soon for another line
            self.clock[0] += AGENT.PLAIN_STATUS_SECONDS
            live.show("loading 60%")
            live.close()              # nothing to end
            live.close("loaded in 21 s")
        self.assertEqual(out.getvalue(), "  loading 10%\n  loading 60%\n  loaded in 21 s\n")

    def test_a_step_status_line_writes_nothing_without_a_console(self):
        AGENT.PLAIN_OUTPUT = True
        with quiet() as out:
            live = AGENT.LiveLine(every=None)
            for _ in range(3):
                live.show("step 1: thinking ...")
                self.clock[0] += 60
            live.clear(80)
        self.assertEqual(out.getvalue(), "")

    def test_a_live_line_redraws_in_place_on_a_console(self):
        AGENT.PLAIN_OUTPUT = False
        with quiet() as out:
            live = AGENT.LiveLine()
            live.show("a")
            live.show("b")
            live.close()
            live.close()  # once is enough
        self.assertEqual(out.getvalue(), "\r  a   \r  b   \n")
        with quiet() as out:
            live = AGENT.LiveLine(every=None)
            live.show("step 1")
            live.clear(10)
        self.assertEqual(out.getvalue(), "\r  step 1   \r" + " " * 10 + "\r")

    def test_copy_progress_ends_with_one_plain_line(self):
        AGENT.PLAIN_OUTPUT = True
        with quiet() as out:
            show = AGENT.console_progress()
            progress = AGENT.Progress(4 * 10**9, "copying", show)
            for _ in range(4):
                self.clock[0] += 1
                progress.add(10**9)
            show.line.close()
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 2, lines)  # the first progress line and how it ended, no more
        self.assertTrue(lines[-1].startswith("  copying: 4.0 of 4.0 GB"), lines)
        self.assertNotIn("\r", out.getvalue())

    def test_an_unattended_run_writes_no_escape_codes_or_redrawn_lines(self):
        import os
        import subprocess
        from agent_support import AGENT_PY
        site = helpers.start_bench_server()
        self.addCleanup(site.stop)
        answer = {"name": "answer", "arguments": {"content": "Done."}}
        fake = helpers.start_fake_llama(self, script=[{"reasoning": "Nothing to do.", "chunk_delay": 0.05,
                                                       "tool_calls": [answer]}])
        d = helpers.temp_dir(self)
        tasks = d / "tasks.json"
        tasks.write_text(json.dumps([{"name": "plain", "task": "Say done.",
                                      "start_url": site.url("/search/index.html")}]), encoding="utf-8")
        override = {"external_server": fake.base, "settle_seconds": 0.1}
        env = dict(os.environ, TEMP=str(d), TMP=str(d))
        r = subprocess.run([str(helpers.PYTHON), "-I", str(AGENT_PY), "--headless-browser", "--data-dir", str(d),
                            "--tasks-file", str(tasks), "--results", str(d / "results.jsonl"),
                            "--settings-override", json.dumps(override)],
                           capture_output=True, timeout=180, creationflags=helpers.CREATE_NO_WINDOW,
                           cwd=str(helpers.REPO), env=env)  # bytes: text mode would turn a lone '\r' into '\n'
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out.decode("utf-8", "replace"))
        self.assertIn(b"Answer: Done.", out)
        self.assertNotIn(b"\x1b", out)
        self.assertNotIn(b"\r", out.replace(b"\r\n", b""))


def first_change(before, after):
    """The index of the first message of `before` that `after` does not repeat unchanged: the server
    reuses its cache up to there (len(before) when `after` only adds to the end)."""
    return next((i for i, (a, b) in enumerate(zip(before, after)) if a != b), len(before))


class BlockEvictionShorteningTests(unittest.TestCase):
    """A step that evicts screenshots also shortens long tool results from older steps, since its prompt
    changes anyway. That must not change the prompt any further back than the eviction itself does:
    with image_eviction "block" the server would read those steps again."""

    LONG = "Page text: " + "word " * 200

    def setUp(self):
        reset_stop(self)

    def run_loop(self, mode, shorten=True, steps=12, budget=3):
        script = [{"reasoning": "go", "tool_calls": [call("scroll_web", direction="down")]}] * (steps - 1)
        script.append({"tool_calls": [call("answer", content="done")]})
        browser = StubBrowser(results={"scroll_web": self.LONG})
        agent, fake, _ = make_agent(self, script=script, browser=browser, image_eviction=mode, image_keep_after_evict=2,
                                    image_budget=budget)
        off = mock.patch.object(AGENT, "shorten_old_results", lambda *a, **k: 0)
        with quiet(), (contextlib.nullcontext() if shorten else off):
            self.assertEqual(agent.run("Scroll a while."), "done")
        bodies = [r["body"]["messages"] for r in fake.chat_requests()]
        return [first_change(a, b) for a, b in zip(bodies, bodies[1:])], bodies

    def test_shortening_changes_nothing_before_the_eviction(self):
        for mode in ("block", "rolling"):
            with self.subTest(mode=mode):
                firsts, bodies = self.run_loop(mode)
                self.assertEqual(firsts, self.run_loop(mode, shorten=False)[0])
                cut = [m for m in bodies[-1] if m["role"] == "tool" and m["content"].endswith(AGENT.TRUNCATED)]
                self.assertTrue(cut, "old results are still shortened")

    def test_an_image_budget_of_two_still_shortens(self):
        # With 2 screenshots the first one an eviction removes is only 2 steps back, so results 3 steps
        # old are always in front of it: with the fixed 3 steps nothing at all was shortened any more.
        self.assertEqual([AGENT.old_result_steps(b) for b in (1, 2, 3, 4, 8)], [2, 2, 3, 3, 3])
        for mode in ("rolling", "block"):
            with self.subTest(mode=mode):
                firsts, bodies = self.run_loop(mode, budget=2)
                self.assertEqual(firsts, self.run_loop(mode, shorten=False, budget=2)[0])
                results = [m["content"] for m in bodies[-1] if m["role"] == "tool"]
                cut = [r for r in results if r.endswith(AGENT.TRUNCATED)]
                self.assertGreaterEqual(len(cut), len(results) // 2 - 1, mode)
                self.assertFalse(results[-1].endswith(AGENT.TRUNCATED))  # the newest result stays whole

    def test_only_from_the_first_change_on(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
        for n in range(1, 8):
            msgs.append({"role": "user", "content": [{"type": "text", "text": f"<observation>{n}</observation>"}]})
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{n}", "type": "function",
                         "function": {"name": "scroll_web", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": f"c{n}", "content": self.LONG})
        since = AGENT.step_starts(msgs)[2]  # the third step's observation
        self.assertEqual(AGENT.shorten_old_results(msgs, since=since), 2)  # steps 3 and 4 (5-7 are recent)
        cut = [i for i, m in enumerate(msgs) if m["role"] == "tool" and m["content"].endswith(AGENT.TRUNCATED)]
        self.assertTrue(cut and min(cut) > since, cut)
        self.assertEqual(AGENT.shorten_old_results(msgs, everywhere=True, since=0), 2)  # steps 1 and 2


class KeyCaseTests(SiteBrowserCase):
    """A letter in a chord gets the case a real keyboard gives it: upper case with any Shift key, lower
    case without. Playwright sends exactly the character it is given and does not apply a held Shift
    to a letter, so the case decides what the page receives."""

    def test_the_case_follows_shift(self):
        for combo, want in (("Shift+V", "Shift+V"), ("shift+v", "Shift+V"), ("ctrl+shift+t", "Control+Shift+T"),
                            ("Shift+Control+t", "Shift+Control+T"), ("ShiftLeft+v", "ShiftLeft+V"),
                            ("ShiftRight+V", "ShiftRight+V"), ("Control+T", "Control+t"), ("ctrl+A", "Control+a"),
                            ("V", "V"), ("v", "v")):
            self.assertEqual(AGENT.normalize_key(combo), want, combo)

    def test_what_the_keys_page_receives(self):
        b = self.start("/fixtures/keys.html")
        b.page.focus("#target")
        for keys, key, mods in (("shift+v", "V", ["shift"]), ("Control+Shift+T", "T", ["ctrl", "shift"]),
                                ("ctrl+A", "a", ["ctrl"]), ("ShiftRight+v", "V", ["shift"])):
            b.page.evaluate("window.keys = []")
            self.act(b, "press_keys_web", keys=keys)
            down = [k for k in b.page.evaluate("window.keys") if k["type"] == "keydown"]
            self.assertEqual((down[-1]["key"], down[-1]["mods"]), (key, mods), keys)
        self.assertEqual(b.page.input_value("#target"), "V")  # V, then select all and V again


class TabTitleTests(SiteBrowserCase):
    """A page without a title is reported by the browser under its address, which the agent drops;
    a real title that is only the start of the address stays."""

    def test_which_titles_are_only_the_address(self):
        address = AGENT.title_is_address
        self.assertTrue(address("127.0.0.1:8765/api/items", "http://127.0.0.1:8765/api/items"))
        self.assertTrue(address("127.0.0.1:8765", "http://127.0.0.1:8765/"))  # a bare host loses its slash
        self.assertTrue(address("127.0.0.1:8765/x?q=a b", "http://127.0.0.1:8765/x?q=a%20b"))  # unescaped
        self.assertTrue(address("about:blank", "about:blank"))
        # Chromium drops only the http:// scheme; https keeps it, and a bare host still loses its slash.
        self.assertTrue(address("https://example.test", "https://example.test/"))
        self.assertTrue(address("https://example.test/a b", "https://example.test/a%20b"))
        self.assertFalse(address("example.test", "https://example.test/x"))
        self.assertFalse(address("localhost", "http://localhost:8080/"))  # a page titled localhost
        self.assertFalse(address("127.0.0.1", "http://127.0.0.1:8765/search/index.html"))
        self.assertFalse(address("127.0.0.1:8765/dir", "http://127.0.0.1:8765/dir/"))  # only a bare host's slash
        self.assertFalse(address("", "http://127.0.0.1:8765/"))
        self.assertFalse(address("Search", "http://127.0.0.1:8765/search/index.html"))

    def test_titles_in_the_browser(self):
        b = self.start("/search/index.html")
        host = f"127.0.0.1:{self.site.port}"
        for title in (host.split(":")[0], host, "Search"):
            b.page.evaluate("t => document.title = t", title)
            b.page.wait_for_function("t => document.title === t", arg=title)
            self.assertEqual(b._tab_title(b.page), title)
        b.page.goto(self.site.url("/api/items?q=a%20b"))  # JSON: no title of its own
        self.assertEqual(b._tab_title(b.page), "")


class MissingBackendTests(unittest.TestCase):
    """A drive built for ARM64 has no cuda12 build (llama.cpp publishes none), and a damaged or partial
    build may lack others: picking a build passes over what is not there."""

    def setUp(self):
        self.folder = scratch_paths(self)
        self.bin = self.folder / "bin"
        self.install("cuda13", "vulkan", "cpu")  # what the build puts on an ARM64 drive
        self.seen = {"cuda13": [DEVICE], "vulkan": [dict(DEVICE, id="Vulkan0")]}
        self.listed = []
        self.gpus = [{"name": DEVICE["name"], "compute_cap": 8.9, "driver": "621.01", "driver_major": 621}]
        for name, value in (("BIN", self.bin), ("list_devices", self.list_devices),
                            ("nvidia_smi_gpus", lambda: self.gpus), ("nvidia_gpu_present", lambda: True)):
            patcher = mock.patch.object(AGENT, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def install(self, *names):
        for name in names:
            (self.bin / name).mkdir(parents=True, exist_ok=True)
            (self.bin / name / "llama-server.exe").write_bytes(b"MZ")

    def remove(self, name):
        (self.bin / name / "llama-server.exe").unlink()

    def list_devices(self, exe):
        self.listed.append(exe.parent.name)
        return [dict(d) for d in self.seen.get(exe.parent.name, [])]

    def pick(self, after=None, **over):
        with quiet() as out:
            backend, exe, device = AGENT.pick_backend(settings(**over), after=after)
        self.assertTrue(exe.is_file(), exe)  # never a build that is not there
        return backend, device, out.getvalue()

    def test_the_usual_order_without_cuda12(self):
        self.assertEqual(self.pick()[0], "cuda13")
        self.assertEqual(self.pick(after="cuda13")[0], "vulkan")  # cuda13 could not load the model
        self.assertEqual(self.pick(after="vulkan")[0], "cpu")
        self.assertNotIn("cuda12", self.listed)

    def test_an_old_driver_goes_to_vulkan_and_names_only_the_cuda13_build(self):
        self.gpus[0].update(driver="560.94", driver_major=560)
        backend, device, out = self.pick()
        self.assertEqual((backend, device["id"]), ("vulkan", "Vulkan0"))
        self.seen = {}  # and when Vulkan sees no GPU either
        backend, device, out = self.pick()
        self.assertEqual((backend, device), ("cpu", None))
        self.assertIn("Update the NVIDIA driver: the cuda13 build needs version 580 or newer.", out)
        self.assertNotIn("cuda12", out)

    def test_a_forced_build_that_is_not_there(self):
        backend, device, out = self.pick(backend="cuda12")
        self.assertEqual(backend, "cuda13")  # tried in the usual order instead of silently on the CPU
        self.assertIn("The cuda12 build (backend setting) is not on this drive", out)
        self.assertIn("cuda13, vulkan, cpu", out)

    def test_without_a_cpu_build_a_gpu_build_runs_on_the_cpu(self):
        self.remove("cpu")
        self.seen = {}
        backend, device, out = self.pick()
        self.assertEqual((backend, device), ("vulkan", None))
        server = AGENT.LlamaServer(settings())
        with mock.patch.object(AGENT, "pick_backend", return_value=("vulkan", self.bin / "vulkan" / "llama-server.exe", None)), \
                mock.patch.object(AGENT, "vc_runtime_missing", return_value=[]), quiet() as out:
            server._choose_backend()
        self.assertIn("Running on the CPU only", out.getvalue())
        self.assertFalse(server.can_fall_back())  # already on the CPU: no build left to try

    def test_a_forced_build_that_is_not_there_still_falls_back(self):
        # It is tried as "auto", so the server falls back as with "auto" when a build cannot load the model.
        server = AGENT.LlamaServer(settings(backend="cuda12"))
        with mock.patch.object(AGENT, "vc_runtime_missing", return_value=[]), quiet() as out:
            server._choose_backend()
            self.assertEqual(server.choice[0], "cuda13")
            self.assertTrue(server.can_fall_back())  # before the fix: False, so no other build was tried
            self.assertEqual(server.next_backend(), "vulkan")
        self.assertEqual(out.getvalue().count("is not on this drive"), 1)  # said once, not at every fallback

    def test_a_forced_cpu_build_that_is_not_there_stays_on_the_cpu(self):
        self.remove("cpu")
        backend, device, out = self.pick(backend="cpu")
        self.assertEqual((backend, device), ("vulkan", None))  # not cuda13 on the GPU, as "auto" would pick
        self.assertIn("There is no cpu build on this drive, so the vulkan build runs the model on the CPU", out)
        self.assertEqual(self.listed, [])  # no build was even asked for its GPUs
        self.assertFalse(AGENT.backend_is_auto(settings(backend="cpu")))
        args, server, printed = start_server(self, settings(backend="cpu"), device=None, backend="vulkan")
        self.assertEqual(args[args.index("--device") + 1], "none")  # the GPU stays out of it
        self.assertIn("Running on the CPU only", printed)
        self.assertFalse(server.can_fall_back())

    def test_no_build_at_all_says_what_to_do(self):
        for name in ("cuda13", "vulkan", "cpu"):
            self.remove(name)
        with self.assertRaises(SystemExit) as cm:
            self.pick()
        self.assertIn("Run Build GhostAgent.cmd again", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
