"""CS-13: settings are checked value by value, the defaults live in settings.default.json and the
user's changes in settings.json, and setup problems (model files, RAM, a read-only drive) are
reported before anything slow or interactive happens."""

import json
import os
import shutil
import subprocess
import unittest
import unittest.mock as mock
from pathlib import Path

import helpers
from agent_support import AGENT, patch_agent, quiet, settings

# The settings.json that shipped before settings.default.json existed (version 0.2), as a user who
# chose copy_model_to_pc and the bundled browser would have it.
LEGACY_FILE = r'''{
  "_note": "Change a value, save, and restart the agent. Keys starting with _ are notes and are ignored.",

  "copy_model_to_pc": true,
  "_copy_model_to_pc": "true = copy the 22 GB model once to this PC (%LOCALAPPDATA%\\GhostAgent). Loading from a USB stick takes ~5 min; from the PC's SSD, seconds.",

  "start_url": "https://www.bing.com",
  "max_steps": 0,
  "_max_steps": "Steps (one action each) before the agent gives up on a task. 0 = no limit: it keeps going until it finishes or you press Ctrl+C.",
  "image_budget": 3,
  "_image_budget": "Screenshots kept in the model's context. H Company recommends at most 3; 2 is a little faster.",
  "temperature": 0.8,
  "thinking": true,
  "settle_seconds": 1.0,
  "viewport_width": 1280,
  "viewport_height": 800,
  "_viewport": "Page area of the agent's window. Bigger screenshots are slower to read.",
  "browser": "bundled",
  "_browser": "chrome = a window in your own running Google Chrome (your sign-ins). bundled = the separate Chromium on this drive. edge = Microsoft Edge with its own profile on this drive.",
  "save_screenshots": false,
  "_save_screenshots": "true = keep every screenshot the model sees in logs\\screenshots (for troubleshooting).",

  "backend": "auto",
  "_backend": "auto tries cuda13, cuda12, vulkan, then cpu. Set one of those names to force it.",
  "moe_cache_mib": "auto",
  "_moe_cache_mib": "VRAM (MiB) for caching model experts on the GPU. auto = free VRAM minus about 5.6 GB. 0 = off.",
  "fit_target_mib": 1024,
  "threads": 0,
  "_threads": "CPU threads for the model. 0 = 8 (or fewer on small CPUs).",
  "ctx_size": 65536,
  "port": 18931,
  "extra_server_args": [],

  "model": "models/Holo4-35B-A3B-Q4_K_M.gguf",
  "mmproj": "models/Holo4-35B-A3B-mmproj-f16.gguf"
}
'''


class SettingsCase(unittest.TestCase):
    def setUp(self):
        self.dir = helpers.temp_dir(self)
        self.user = self.dir / "settings.json"
        self.defaults = self.dir / "settings.default.json"
        shutil.copy(AGENT.DEFAULTS_FILE, self.defaults)
        for name, value in (("SETTINGS_FILE", self.user), ("DEFAULTS_FILE", self.defaults)):
            p = patch_agent(name, value)
            p.start()
            self.addCleanup(p.stop)

    def load(self, user=None, raw=None, encoding="utf-8", override=None, interactive=False, answers=("",)):
        """Write settings.json (a dict or raw text), load it, and return (settings, printed text)."""
        if user is not None:
            raw = json.dumps(user, indent=2)
        if raw is not None:
            self.user.write_text(raw, encoding=encoding)
        replies = list(answers)
        with quiet() as out, mock.patch("builtins.input", side_effect=lambda prompt="": replies.pop(0)) as asked:
            s = AGENT.load_settings(override, interactive=interactive)
        self.asked = asked
        return s, out.getvalue()


class SettingsValueTests(SettingsCase):
    def test_a_file_with_a_bom_loads(self):
        s, out = self.load(raw='{"max_steps": 25, "browser": "bundled"}', encoding="utf-8-sig")
        self.assertEqual(self.user.read_bytes()[:3], b"\xef\xbb\xbf")
        self.assertEqual((s["max_steps"], s["browser"]), (25, "bundled"))
        self.assertNotIn("!", out)

    def test_true_and_false_written_as_text(self):
        s, out = self.load({"copy_model_to_pc": "false", "save_screenshots": "False", "thinking": "off",
                            "step_trace": "no", "trace_messages": "0"})
        for key in ("copy_model_to_pc", "save_screenshots", "thinking", "step_trace", "trace_messages"):
            self.assertIs(s[key], False, key)
        s, _ = self.load({"copy_model_to_pc": "true", "save_screenshots": "YES", "thinking": "on", "trace_messages": 1})
        for key in ("copy_model_to_pc", "save_screenshots", "thinking", "trace_messages"):
            self.assertIs(s[key], True, key)
        s, out = self.load({"thinking": "maybe"})
        self.assertIs(s["thinking"], True)
        self.assertIn('"thinking"', out)
        self.assertIn("true or false", out)

    def test_extra_server_args_as_one_string(self):
        s, _ = self.load({"extra_server_args": "--foo bar"})
        self.assertEqual(s["extra_server_args"], ["--foo", "bar"])
        s, _ = self.load({"extra_server_args": r'--lora "C:\My Models\a b.gguf" -v'})
        self.assertEqual(s["extra_server_args"], ["--lora", r"C:\My Models\a b.gguf", "-v"])
        s, _ = self.load({"extra_server_args": ["--n-cpu-moe", 30]})
        self.assertEqual(s["extra_server_args"], ["--n-cpu-moe", "30"])

    def test_unknown_backend_warns_and_keeps_auto(self):
        s, out = self.load({"backend": "cuda", "max_steps": 12})
        self.assertEqual(s["backend"], "auto")
        self.assertEqual(s["max_steps"], 12)  # only the bad value is replaced
        self.assertIn('"backend" is "cuda"', out)
        self.assertIn('"cuda13"', out)
        s, out = self.load({"backend": "Vulkan"})
        self.assertEqual(s["backend"], "vulkan")

    def test_image_budget_zero_warns_and_uses_three(self):
        s, out = self.load({"image_budget": 0})
        self.assertEqual(s["image_budget"], 3)
        self.assertIn('"image_budget" is 0', out)
        self.assertIn("at least 1", out)

    def test_a_typo_is_reported(self):
        s, out = self.load({"max_step": 5, "_max_step": "a note is fine"})
        self.assertNotIn("max_step", s)
        self.assertEqual(s["max_steps"], 0)
        self.assertIn('unknown setting "max_step"', out)
        self.assertIn('did you mean "max_steps"', out)
        self.assertNotIn("_max_step", out)

    def test_numbers(self):
        s, out = self.load({"max_steps": "40", "temperature": "0.7", "port": 18000.0, "settle_seconds": 2})
        self.assertEqual((s["max_steps"], s["temperature"], s["port"], s["settle_seconds"]), (40, 0.7, 18000, 2.0))
        self.assertIsInstance(s["settle_seconds"], float)
        self.assertEqual(out.strip(), "")
        s, out = self.load({"max_steps": "4.5", "port": 70000, "temperature": True, "top_p": 1.5,
                            "max_tokens": -1, "seed": -2, "ctx_size": "big"})
        for key in ("max_steps", "port", "temperature", "top_p", "max_tokens", "seed", "ctx_size"):
            self.assertEqual(s[key], AGENT.DEFAULT_SETTINGS[key], key)
            self.assertIn(f'"{key}"', out)

    def test_moe_cache_mib(self):
        for value, want in (("auto", "auto"), (2048, 2048), ("2048", 2048), ("off", 0), (0, 0), ("", "auto")):
            s, out = self.load({"moe_cache_mib": value})
            self.assertEqual(s["moe_cache_mib"], want, value)
            self.assertEqual(out.strip(), "", value)
        for value in (-5, "lots", 1.5, True):
            s, out = self.load({"moe_cache_mib": value})
            self.assertEqual(s["moe_cache_mib"], "auto", value)
            self.assertIn('"moe_cache_mib"', out)

    def test_text_settings_and_null(self):
        s, out = self.load({"start_url": 5, "browser": None, "model": "models/x.gguf"})
        self.assertEqual(s["start_url"], "https://www.bing.com")
        self.assertEqual(s["browser"], "chrome")
        self.assertEqual(s["model"], "models/x.gguf")
        self.assertIn('"start_url" is 5', out)
        self.assertIn('"browser" is null', out)

    def test_external_server(self):
        s, out = self.load({"external_server": "http://127.0.0.1:8080/"})
        self.assertEqual(s["external_server"], "http://127.0.0.1:8080")
        s, out = self.load({"external_server": "localhost:8080"})
        self.assertEqual(s["external_server"], "")
        self.assertIn('"external_server"', out)

    def test_rules_for_settings_added_later(self):
        """The checks for settings other changes add (dialog policy, image eviction, model profile,
        reasoning effort, checkpoint spacing) work as soon as those settings exist."""
        later = {"dialog_policy": "ask", "image_eviction": "rolling", "model_profile": "fast",
                 "reasoning_effort": "", "checkpoint_min_step": 1536}
        with mock.patch.dict(AGENT.DEFAULT_SETTINGS, later):
            s, out = self.load({"dialog_policy": "always", "image_eviction": "Block", "model_profile": "tiny",
                                "reasoning_effort": "max", "checkpoint_min_step": -1})
            self.assertEqual(s["dialog_policy"], "ask")
            self.assertEqual(s["image_eviction"], "block")
            self.assertEqual(s["model_profile"], "fast")
            self.assertEqual(s["reasoning_effort"], "")
            self.assertEqual(s["checkpoint_min_step"], 1536)
            for key in ("dialog_policy", "model_profile", "reasoning_effort", "checkpoint_min_step"):
                self.assertIn(f'"{key}"', out)
            self.assertIn('"accurate"', out)  # the profile names come from tools\profiles.json
            s, out = self.load({"model_profile": "accurate", "reasoning_effort": "medium"})
            self.assertEqual((s["model_profile"], s["reasoning_effort"]), ("accurate", "medium"))

    def test_override_comes_last(self):
        s, out = self.load({"max_steps": 10, "browser": "bundled"}, override={"max_steps": "30", "nonsense": 1})
        self.assertEqual((s["max_steps"], s["browser"]), (30, "bundled"))
        self.assertIn('--settings-override: unknown setting "nonsense"', out)

    def test_defaults_file_is_a_layer(self):
        """settings.default.json is read too (the code's table is the fallback)."""
        doc = json.loads(self.defaults.read_text(encoding="utf-8"))
        doc["settle_seconds"] = 2.5
        self.defaults.write_text(json.dumps(doc), encoding="utf-8")
        s, _ = self.load({})
        self.assertEqual(s["settle_seconds"], 2.5)
        self.defaults.unlink()
        s, _ = self.load({"max_steps": 3})
        self.assertEqual((s["settle_seconds"], s["max_steps"]), (1.0, 3))

    def test_json_mistake_shows_line_and_column_and_waits(self):
        raw = '{\n  "max_steps": 5,\n  "browser" "bundled"\n}\n'
        s, out = self.load(raw=raw, interactive=True)
        self.assertIn("settings.json has a mistake at line 3, column 13", out)
        self.assertEqual(self.asked.call_count, 1)
        self.assertIn("Press Enter", self.asked.call_args[0][0])
        self.assertEqual(s, settings())
        self.assertEqual(self.user.read_text(encoding="utf-8"), raw)  # never overwritten
        with self.assertRaises(SystemExit):  # unattended runs stop instead of running with the wrong settings
            self.load(raw=raw, interactive=False)

    def test_not_an_object(self):
        s, out = self.load(raw='["browser", "bundled"]')
        self.assertEqual(s, settings())
        self.assertIn("one JSON object", out)


class SettingsFilesTests(SettingsCase):
    def test_defaults_file_matches_the_code(self):
        doc = json.loads(AGENT.DEFAULTS_FILE.read_text(encoding="utf-8"))
        values = {k: v for k, v in doc.items() if not k.startswith("_")}
        self.assertEqual(values, AGENT.DEFAULT_SETTINGS)
        for key in values:
            self.assertTrue(str(doc.get("_" + key, "")).strip(), f"settings.default.json has no note for {key}")
        notes = {k[1:] for k in doc if k.startswith("_") and k != "_note"}
        self.assertEqual(notes - set(values), set(), "notes for settings that do not exist")

    def test_the_benchmarked_defaults(self):
        # Chosen on the local benchmark (tools\bench): the h prompt and tools, block eviction keeping
        # one screenshot, adaptive settle. The shipped settings.default.json says the same.
        chosen = {"prompt_version": "h", "tool_schema": "h", "image_eviction": "block",
                  "image_keep_after_evict": 1, "settle_mode": "adaptive"}
        shipped = json.loads((helpers.REPO / "GhostAgent" / "settings.default.json").read_text(encoding="utf-8"))
        for key, value in chosen.items():
            self.assertEqual(AGENT.DEFAULT_SETTINGS[key], value, key)
            self.assertEqual(shipped[key], AGENT.DEFAULT_SETTINGS[key], key)
        s, out = self.load()
        self.assertEqual({k: s[k] for k in chosen}, chosen)
        self.assertEqual(out.strip(), "")  # keep 1 fits the default image_budget: no warning

    def test_defaults_file_is_tracked_and_settings_json_is_not(self):
        ignore = (helpers.REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("/GhostAgent/settings.json", ignore)
        self.assertTrue((helpers.REPO / "GhostAgent" / "settings.default.json").is_file())
        if shutil.which("git") and (helpers.REPO / ".git").exists():
            r = subprocess.run(["git", "check-ignore", "-q", "--no-index", "GhostAgent/settings.json"],
                               cwd=str(helpers.REPO), creationflags=helpers.CREATE_NO_WINDOW)
            self.assertEqual(r.returncode, 0)
            r = subprocess.run(["git", "check-ignore", "-q", "--no-index", "GhostAgent/settings.default.json"],
                               cwd=str(helpers.REPO), creationflags=helpers.CREATE_NO_WINDOW)
            self.assertEqual(r.returncode, 1)

    def test_first_start_creates_a_settings_file_with_only_a_note(self):
        self.assertFalse(self.user.exists())
        s, out = self.load()
        self.assertEqual(s, settings())
        doc = json.loads(self.user.read_text(encoding="utf-8"))
        self.assertEqual(list(doc), ["_note"])
        self.assertIn("settings.default.json", doc["_note"])
        s2, out2 = self.load()  # the second start reads it without a word
        self.assertEqual(s2, settings())
        self.assertEqual(out2.strip(), "")

    def test_users_file_is_left_as_it_is(self):
        raw = '{\n  "browser": "bundled",\n  "copy_model_to_pc": true\n}\n'
        s, out = self.load(raw=raw)
        self.assertEqual((s["browser"], s["copy_model_to_pc"]), ("bundled", True))
        self.assertEqual(self.user.read_text(encoding="utf-8"), raw)
        self.assertFalse(list(self.dir.glob("settings.backup*")))

    def test_old_full_settings_file_becomes_a_list_of_changes(self):
        s, out = self.load(raw=LEGACY_FILE)
        self.assertEqual((s["browser"], s["copy_model_to_pc"]), ("bundled", True))
        self.assertEqual({k: v for k, v in s.items() if v != AGENT.DEFAULT_SETTINGS[k]},
                         {"browser": "bundled", "copy_model_to_pc": True})
        doc = json.loads(self.user.read_text(encoding="utf-8"))
        self.assertEqual({k: v for k, v in doc.items() if not k.startswith("_")},
                         {"copy_model_to_pc": True, "browser": "bundled"})
        backup = self.dir / "settings.backup.json"
        self.assertEqual(backup.read_text(encoding="utf-8"), LEGACY_FILE)  # the old file, untouched
        self.assertIn("settings.backup.json", out)
        self.assertIn("copy_model_to_pc, browser", out)
        s2, out2 = self.load()  # converted once only
        self.assertEqual((s2["browser"], s2["copy_model_to_pc"]), ("bundled", True))
        self.assertEqual(out2.strip(), "")

    def test_conversion_never_loses_a_value(self):
        doc = json.loads(LEGACY_FILE)
        doc.update({"backend": "cuda", "my_own_key": 7, "max_steps": "40", "thinking": "true"})
        (self.dir / "settings.backup.json").write_text("{}", encoding="utf-8")  # an earlier backup stays
        s, out = self.load(doc)
        kept = json.loads(self.user.read_text(encoding="utf-8"))
        self.assertEqual(kept["backend"], "cuda")      # wrong, but the user's: reported, not dropped
        self.assertEqual(kept["my_own_key"], 7)
        self.assertEqual(kept["max_steps"], "40")
        self.assertNotIn("thinking", kept)             # "true" is what shipped
        self.assertEqual(s["max_steps"], 40)
        self.assertEqual((self.dir / "settings.backup.json").read_text(encoding="utf-8"), "{}")
        backups = list(self.dir.glob("settings.backup-*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(json.loads(backups[0].read_text(encoding="utf-8")), doc)

    def test_conversion_that_cannot_write_keeps_the_old_file(self):
        with patch_agent("write_user_settings", side_effect=PermissionError("read-only")):
            s, out = self.load(raw=LEGACY_FILE)
        self.assertEqual((s["browser"], s["copy_model_to_pc"]), ("bundled", True))
        self.assertEqual(self.user.read_text(encoding="utf-8"), LEGACY_FILE)
        self.assertIn("used as it is", out)

    def test_read_only_folder_needs_no_settings_file(self):
        with patch_agent("write_user_settings", side_effect=PermissionError("read-only")):
            s, out = self.load()
        self.assertEqual(s, settings())
        self.assertEqual(out.strip(), "")


class PreflightModelTests(unittest.TestCase):
    def setUp(self):
        self.dir = helpers.temp_dir(self)

    def model_settings(self, model_bytes, mmproj_bytes, names=("Holo4-35B-A3B-Q4_K_M.gguf", "Holo4-35B-A3B-mmproj-f16.gguf")):
        paths = []
        for name, size in zip(names, (model_bytes, mmproj_bytes)):
            p = self.dir / name
            if size is not None:
                p.write_bytes(b"\0" * size)
            paths.append(str(p))  # absolute: ROOT / an absolute path is that path
        return settings(model=paths[0], mmproj=paths[1])

    def test_short_model_file_is_called_incomplete(self):
        s = self.model_settings(1_000_000, 10)
        with self.assertRaises(SystemExit) as cm:
            AGENT.check_model_files(s)
        msg = str(cm.exception.code)
        self.assertIn("The model file is incomplete (0.0 of 21.3 GB)", msg)
        self.assertIn("Run Build GhostAgent.cmd again", msg)
        self.assertEqual(AGENT.gb(12_100_000_000), "12.1")

    def test_mmproj_and_other_sizes(self):
        pins = {"m.gguf": (100, "fast"), "p.gguf": (50, None)}
        with patch_agent("pinned_files", return_value=pins):
            AGENT.check_model_files(self.model_settings(100, 50, ("m.gguf", "p.gguf")))  # sizes match: fine
            with self.assertRaises(SystemExit) as cm:
                AGENT.check_model_files(self.model_settings(100, 20, ("m.gguf", "p.gguf")))
            self.assertIn("p.gguf", str(cm.exception.code))
            with self.assertRaises(SystemExit) as cm:
                AGENT.check_model_files(self.model_settings(150, 50, ("m.gguf", "p.gguf")))
            self.assertIn("not the expected one", str(cm.exception.code))
        # A file no profile lists (a model the user brought) is only checked for existence.
        AGENT.check_model_files(self.model_settings(10, 10, ("own.gguf", "own-mmproj.gguf")))

    def test_missing_model_file(self):
        with self.assertRaises(SystemExit) as cm:
            AGENT.check_model_files(self.model_settings(None, 10))
        self.assertIn("Missing model file", str(cm.exception.code))

    def test_profiles_give_sizes_and_ram_needs(self):
        pins = AGENT.pinned_files()
        self.assertEqual(pins["Holo4-35B-A3B-Q4_K_M.gguf"], (21297870912, "fast"))
        self.assertEqual(pins["Holo4-35B-A3B-mmproj-f16.gguf"][0], 899282976)
        self.assertEqual(AGENT.ram_need_gb(settings(model="models/Holo4-35B-A3B-Q4_K_M.gguf")), 22)
        self.assertEqual(AGENT.ram_need_gb(settings(model="models/Holo4-35B-A3B-Q8_0.gguf")), 40)
        self.assertEqual(AGENT.ram_need_gb(settings(model="models/Holo4-27B-Q4_K_M.gguf")), 19)
        own = self.dir / "own.gguf"
        own.write_bytes(b"\0" * 1000)
        self.assertEqual(AGENT.ram_need_gb(settings(model=str(own))), 2)


GiB = 2**30


class PreflightRamTests(unittest.TestCase):
    def check(self, total_gib, avail_gib, commit_gib=None, interactive=True, answer=""):
        status = {"total": total_gib * GiB, "avail": avail_gib * GiB, "commit_avail": (commit_gib or avail_gib) * GiB}
        with quiet() as out, patch_agent("memory_status", return_value=status), \
                mock.patch("builtins.input", return_value=answer) as asked:
            try:
                AGENT.check_ram(settings(), interactive=interactive)
                exited = None
            except SystemExit as e:
                exited = e.code
        return out.getvalue(), asked, exited

    def test_sixteen_gb_pc_asks_before_loading(self):
        out, asked, exited = self.check(15.8, 9)
        self.assertIn("This PC has 16 GB of RAM; the model needs about 22 GB free", out)
        self.assertEqual(asked.call_count, 1)
        self.assertIn("[y/N]", asked.call_args[0][0])
        self.assertEqual(exited, 0)  # no answer means no
        out, asked, exited = self.check(15.8, 9, answer="y")
        self.assertIsNone(exited)

    def test_unattended_runs_only_warn(self):
        out, asked, exited = self.check(15.8, 9, interactive=False)
        self.assertIn("16 GB of RAM", out)
        self.assertEqual(asked.call_count, 0)
        self.assertIsNone(exited)

    def test_little_free_memory_warns(self):
        out, asked, exited = self.check(63.7, 14.2, 40)
        self.assertIn("Only 14 GB of memory is free right now", out)
        self.assertEqual(asked.call_count, 0)
        out, asked, exited = self.check(63.7, 40, 18)  # commit charge (RAM + page file) counts too
        self.assertIn("Only 18 GB", out)
        out, asked, exited = self.check(63.7, 40, 40)
        self.assertEqual(out.strip(), "")

    def test_real_memory_status(self):
        m = AGENT.memory_status()
        self.assertGreater(m["total"], 1 * GiB)
        self.assertGreater(m["avail"], 0)
        self.assertLessEqual(m["avail"], m["total"])


class ReadOnlyDriveTests(unittest.TestCase):
    def test_logs_move_to_localappdata_when_the_drive_is_read_only(self):
        stick = helpers.temp_dir(self, "stick-")
        user = os.environ.get("USERNAME")
        if not user or not shutil.which("icacls"):
            self.skipTest("needs icacls and USERNAME")
        r = subprocess.run(["icacls", str(stick), "/deny", f"{user}:(OI)(CI)(W)"], capture_output=True, text=True,
                           creationflags=helpers.CREATE_NO_WINDOW)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.addCleanup(subprocess.run, ["icacls", str(stick), "/remove:d", user], capture_output=True,
                        creationflags=helpers.CREATE_NO_WINDOW)
        with self.assertRaises(OSError):
            (stick / "probe").write_bytes(b"x")
        with helpers.isolated_env() as base:
            with patch_agent("LOGS", stick / "logs"), patch_agent("PROFILE", stick / "browser-profile"), \
                    patch_agent("DOWNLOADS", stick / "downloads"):
                notice = AGENT.ensure_writable_data()
                local = Path(os.environ["LOCALAPPDATA"]) / "GhostAgent"
                self.assertIn("cannot be written to", notice)
                self.assertIn(str(local), notice)
                self.assertEqual(AGENT.LOGS, local / "logs")
                self.assertEqual(AGENT.PROFILE, local / "browser-profile")
                self.assertEqual(AGENT.DOWNLOADS, stick / "downloads")  # downloads go to the PC's Downloads folder anyway
                self.assertTrue(AGENT.LOGS.is_dir())
                left = list((stick / "logs").iterdir()) if (stick / "logs").exists() else []
                self.assertEqual(left, [])  # nothing was written on the read-only drive

    def test_writable_drive_stays(self):
        stick = helpers.temp_dir(self, "stick-")
        with patch_agent("LOGS", stick / "logs"):
            self.assertIsNone(AGENT.ensure_writable_data())
            self.assertEqual(AGENT.LOGS, stick / "logs")
            self.assertEqual(list((stick / "logs").iterdir()), [])  # the probe file is gone again


if __name__ == "__main__":
    unittest.main()
