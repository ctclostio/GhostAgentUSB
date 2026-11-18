"""Tests for tools\\Build-GhostAgent.ps1 and "Update GhostAgent.cmd" that need no network and no GPU.

The build script is dot-sourced into Windows PowerShell 5.1 (the PowerShell a double-click uses),
which defines its functions without running a build. Each check prints a "RESULT name value" line;
all checks run in one PowerShell process to keep the suite fast. Downloads, copies and volume
queries are replaced by small stand-ins where a test needs them.

Meant to live in tests\\ (it imports tests\\helpers.py).
"""

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

from helpers import CREATE_NO_WINDOW, REPO, temp_dir

BUILD = REPO / "tools" / "Build-GhostAgent.ps1"
UPDATE_CMD = REPO / "Update GhostAgent.cmd"
POWERSHELL = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
CMD = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "cmd.exe"


def ps_env(tmp):
    env = dict(os.environ)
    env["PSModulePath"] = ""  # as "Build GhostAgent.cmd" does: a PowerShell 7 module path breaks 5.1
    env["TEMP"] = env["TMP"] = str(tmp)
    return env


CHECKS = r"""
$ErrorActionPreference = 'Stop'
. $env:GA_BUILD
function Report($name, $value) { Write-Output ("RESULT $name " + ($value | ConvertTo-Json -Compress)) }
$t = $env:GA_TMP

# --- model files that no profile lists ----------------------------------------------------------
$d = Join-Path $t 'stale'
New-Item -ItemType Directory -Force -Path $d | Out-Null
foreach ($n in 'a.gguf', 'a.gguf.part', 'old.gguf', 'old.gguf.part', 'notes.txt') { Set-Content -LiteralPath (Join-Path $d $n) -Value 'x' }
Report stale (@(Get-StaleModelFiles $d @{ 'a.gguf' = $true } | ForEach-Object { $_.Name } | Sort-Object) -join ',')

# --- small sizes ---------------------------------------------------------------------------------
Report size_small (Format-Size 300000)
Report size_tiny (Format-Size 1)
Report size_model (Format-Size 21297870912)

# --- FAT32 refusal --------------------------------------------------------------------------------
try { Assert-LargeFileSupport @{ Root = 'E:\'; FileSystem = 'FAT32' } 21297870912 'The model'; Report fat32 '' }
catch { Report fat32 $_.Exception.Message }
try { Assert-LargeFileSupport @{ Root = 'E:\'; FileSystem = 'exFAT' } 21297870912 'The model'; Report exfat 'ok' }
catch { Report exfat $_.Exception.Message }

# --- help flags are case-sensitive ----------------------------------------------------------------
$k = Get-HelpFlags @('-C,    --cpu-mask M          CPU affinity mask', '-t,    --threads N           threads (default: -1)', 'usage: llama-server -c 4096')
Report flags_c ($k -ccontains '-c')
Report flags_C ($k -ccontains '-C')
Report flags_t ($k -ccontains '-t')
Report flags_threads ($k -ccontains '--threads')

# --- Visual C++ runtime check ---------------------------------------------------------------------
$Sys32 = Join-Path $t 'no-system32'
New-Item -ItemType Directory -Force -Path $Sys32 | Out-Null
$vc = Join-Path $t 'vc'
New-Item -ItemType Directory -Force -Path $vc | Out-Null
Report vc_missing (Test-VcRuntime $vc)
foreach ($n in 'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll') { Set-Content -LiteralPath (Join-Path $vc $n) -Value '' }
Report vc_present (Test-VcRuntime $vc)

# --- .verified round trip -------------------------------------------------------------------------
$vd = Join-Path $t 'verified'
New-Item -ItemType Directory -Force -Path $vd | Out-Null
$fa = Join-Path $vd 'a.gguf'; Set-Content -LiteralPath $fa -Value 'aaaa' -Encoding Ascii -NoNewline
$fb = Join-Path $vd 'b.gguf'; Set-Content -LiteralPath $fb -Value 'bbbb' -Encoding Ascii -NoNewline
$pa = @{ Name = 'a.gguf'; Size = [int64]4; Sha256 = ('ab' * 32) }
$pb = @{ Name = 'b.gguf'; Size = [int64]4; Sha256 = ('cd' * 32) }
Write-Verified $fa $pa
Write-Verified $fb $pb
Report verified_a (Test-Verified $fa $pa)
(Get-Item -LiteralPath $fb).LastWriteTime = (Get-Date).AddDays(-1)
Report verified_b_touched (Test-Verified $fb $pb)
Report verified_other_pin (Test-Verified $fa $pb)

# --- staging: fresh free space, earlier partial counts ------------------------------------------
function Test-SameVolume($A, $B) { return $false }
function Get-VolumeInfo($Path) { return @{ Root = 'Z:\'; FileSystem = 'NTFS'; FreeBytes = [int64]2000000500; Id = 'z' } }
$NoStaging = $false
$script:VolumeList = @('stale')
$r1 = Get-StageDir 'Q:\models' 1000 'f.gguf'
Report stage_no_room ($null -eq $r1)
Report stage_volume_list_reset ($null -eq $script:VolumeList)
$stageDir = Join-Path $env:LOCALAPPDATA 'GhostAgent\models'
Set-Content -LiteralPath (Join-Path $stageDir 'f.gguf.part') -Value ('x' * 600) -Encoding Ascii -NoNewline
Report stage_with_partial ([string](Get-StageDir 'Q:\models' 1000 'f.gguf'))
Remove-Item -LiteralPath (Join-Path $stageDir 'f.gguf.part')
# A complete staged copy is used even when the PC's disk is nearly full.
Set-Content -LiteralPath (Join-Path $stageDir 'g.gguf') -Value ('x' * 1000) -Encoding Ascii -NoNewline
function Get-VolumeInfo($Path) { return @{ Root = 'Z:\'; FileSystem = 'NTFS'; FreeBytes = [int64]5; Id = 'z' } }
Report stage_complete_copy ([string](Get-StageDir 'Q:\models' 1000 'g.gguf'))
Remove-Item -LiteralPath (Join-Path $stageDir 'g.gguf')

# --- a download already under way on the target drive is continued -------------------------------
$fakeStage = Join-Path $t 'stage'
New-Item -ItemType Directory -Force -Path $fakeStage | Out-Null
function Get-StageDir($Models, $Need, $Name) { return $fakeStage }
$script:downloads = @()
function Invoke-ResumableDownload([string[]]$Urls, [string]$Part, [int64]$Size, [string]$Label) {
    $script:downloads += $Part
    Set-Content -LiteralPath $Part -Value ('y' * $Size) -Encoding Ascii -NoNewline
}
function Get-Sha256([string]$Path, [string]$Label) { return ('ef' * 32) }
function Copy-WithRobocopy([string]$Src, [string]$DestDir) { Copy-Item -LiteralPath $Src -Destination $DestDir }
$pin = @{ Name = 'f.gguf'; Size = [int64]50; Sha256 = ('ef' * 32); Urls = @('https://example.invalid/f') }
$m1 = Join-Path $t 'models1'
New-Item -ItemType Directory -Force -Path $m1 | Out-Null
Set-Content -LiteralPath (Join-Path $m1 'f.gguf.part') -Value ('y' * 10) -Encoding Ascii -NoNewline
Install-ModelFile $pin $m1 @()
Report direct_part_continued (@($script:downloads) -join '|')
Report direct_done (Test-Verified (Join-Path $m1 'f.gguf') $pin)
$script:downloads = @()
$m2 = Join-Path $t 'models2'
New-Item -ItemType Directory -Force -Path $m2 | Out-Null
Install-ModelFile $pin $m2 @()
Report staged_download (@($script:downloads) -join '|')
Report staged_done (Test-Verified (Join-Path $m2 'f.gguf') $pin)

# --- the update uses the same proxy rules as the build -------------------------------------------
foreach ($n in 'HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY') { Remove-Item "Env:$n" -ErrorAction SilentlyContinue }
function Get-SystemProxy { return 'http://proxy.test:3128' }
$Proxy = $null
$u = Initialize-Proxy
Report proxy_url $u
Report proxy_curl ($script:CurlProxyArgs -join ' ')
Report proxy_update_uses_it ((Get-Command Invoke-Update).Definition -match 'Initialize-Proxy')

# --- what the smoke test says about optional flags a build lacks ----------------------------------
$lock = @('--cors-origins', '--no-slots', '--no-webui')
$n1 = @(Get-OptionalFlagNotes 'bin\cuda13' @('--moe-cache-cpu-share') $lock)
Report optflag_one_kind (@($n1 | ForEach-Object { $_.Kind }) -join ',')
Report optflag_one_text $n1[0].Text
$n2 = @(Get-OptionalFlagNotes 'bin\cpu' @('--moe-cache-cpu-share', '--image-max-tokens', '--no-slots') $lock)
Report optflag_mixed_kinds (@($n2 | ForEach-Object { $_.Kind }) -join ',')
Report optflag_mixed_texts (@($n2 | ForEach-Object { $_.Text }) -join ' | ')
Report optflag_none (@(Get-OptionalFlagNotes 'bin\cpu' @() $lock).Count)
Report optflag_old_wording ((Get-Command Invoke-SmokeTest).Definition -match 'those cases would fail')

# --- the dry run names where a model really goes --------------------------------------------------
function Test-SameVolume($A, $B) { return $true }
Report dest_same_drive (Get-ModelDestNote 'C:\stick\GhostAgent\models')
function Test-SameVolume($A, $B) { return $false }
Report dest_other_drive (Get-ModelDestNote 'E:\GhostAgent\models')
$NoStaging = $true
Report dest_no_staging (Get-ModelDestNote 'E:\GhostAgent\models')
$NoStaging = $false

# --- an update mirrors the agent's package and keeps everything else ------------------------------
$us = Join-Path $t 'update-src'
$ud = Join-Path $t 'update-dst'
foreach ($d in "$us\GhostAgent\agent\ghostagent", "$ud\GhostAgent\agent\ghostagent", "$ud\GhostAgent\models") {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
Set-Content -LiteralPath "$us\GhostAgent\agent\ghost_agent.py" -Value 'new'
Set-Content -LiteralPath "$us\GhostAgent\agent\ghostagent\cli.py" -Value 'new'
Set-Content -LiteralPath "$ud\GhostAgent\agent\ghostagent\cli.py" -Value 'old'
Set-Content -LiteralPath "$ud\GhostAgent\agent\ghostagent\gone.py" -Value 'a module the new version no longer has'
Set-Content -LiteralPath "$ud\GhostAgent\models\m.gguf" -Value 'model'
$savedRoot = $Root
$Root = $ud
Copy-UpdateFiles $us (Join-Path $ud 'GhostAgent\settings.json') | Out-Null
$Root = $savedRoot
Report update_module_replaced ((Get-Content -LiteralPath "$ud\GhostAgent\agent\ghostagent\cli.py") -join '')
Report update_stale_module_gone (-not (Test-Path -LiteralPath "$ud\GhostAgent\agent\ghostagent\gone.py"))
Report update_model_kept (Test-Path -LiteralPath "$ud\GhostAgent\models\m.gguf")
"""


class BuildScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = temp_dir(prefix="build-")
        cls.tmp = tmp
        script = tmp / "checks.ps1"
        script.write_text(CHECKS.replace("\n", "\r\n"), encoding="ascii")
        env = ps_env(tmp)
        env["GA_BUILD"] = str(BUILD)
        env["GA_TMP"] = str(tmp)
        r = subprocess.run([str(POWERSHELL), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                           capture_output=True, text=True, timeout=180, env=env, creationflags=CREATE_NO_WINDOW,
                           stdin=subprocess.DEVNULL)
        cls.output = r.stdout + r.stderr
        cls.results = {}
        for line in r.stdout.splitlines():
            if line.startswith("RESULT "):
                _, name, value = line.split(" ", 2)
                cls.results[name] = json.loads(value)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def result(self, name):
        if name not in self.results:
            self.fail(f"the PowerShell checks did not report {name}:\n{self.output[-3000:]}")
        return self.results[name]

    def test_unfinished_download_of_a_listed_model_is_not_stale(self):
        self.assertEqual(self.result("stale"), "old.gguf,old.gguf.part")

    def test_small_sizes_are_not_shown_as_zero(self):
        self.assertEqual(self.result("size_small"), "300 KB")
        self.assertEqual(self.result("size_tiny"), "1 KB")
        self.assertEqual(self.result("size_model"), "21.3 GB")

    def test_fat32_is_refused_with_advice_and_exfat_is_not(self):
        msg = self.result("fat32")
        self.assertIn("FAT32", msg)
        self.assertIn("NTFS", msg)
        self.assertIn("21.3 GB", msg)
        self.assertEqual(self.result("exfat"), "ok")

    def test_help_flags_are_case_sensitive(self):
        self.assertFalse(self.result("flags_c"))
        self.assertTrue(self.result("flags_C"))
        self.assertTrue(self.result("flags_t"))
        self.assertTrue(self.result("flags_threads"))

    def test_missing_visual_cpp_runtime_is_detected(self):
        self.assertFalse(self.result("vc_missing"))
        self.assertTrue(self.result("vc_present"))

    def test_verified_record_matches_python_stat(self):
        self.assertTrue(self.result("verified_a"))
        self.assertFalse(self.result("verified_b_touched"))
        self.assertFalse(self.result("verified_other_pin"))
        # The agent compares mtime_ns with os.stat(); both must give the same number.
        doc = json.loads((self.tmp / "verified" / ".verified").read_text(encoding="ascii"))
        rec = doc["files"]["a.gguf"]
        st = os.stat(self.tmp / "verified" / "a.gguf")
        self.assertEqual(rec["mtime_ns"], st.st_mtime_ns)
        self.assertEqual(rec["size"], st.st_size)

    def test_staging_reads_free_space_fresh_and_counts_earlier_partial(self):
        self.assertTrue(self.result("stage_no_room"))
        self.assertTrue(self.result("stage_volume_list_reset"))
        self.assertTrue(self.result("stage_with_partial").lower().endswith(r"ghostagent\models"))
        self.assertTrue(self.result("stage_complete_copy").lower().endswith(r"ghostagent\models"))

    def test_partial_download_on_target_is_continued_not_restarted(self):
        self.assertEqual(self.result("direct_part_continued"), str(self.tmp / "models1" / "f.gguf.part"))
        self.assertTrue(self.result("direct_done"))
        self.assertEqual(self.result("staged_download"), str(self.tmp / "stage" / "f.gguf.part"))
        self.assertTrue(self.result("staged_done"))

    def test_update_uses_windows_proxy_setting(self):
        self.assertEqual(self.result("proxy_url"), "http://proxy.test:3128")
        self.assertEqual(self.result("proxy_curl"), "--proxy http://proxy.test:3128")
        self.assertTrue(self.result("proxy_update_uses_it"))
        # The same update step mirrors the agent's package (GhostAgent\agent\ghostagent): a module the
        # new version no longer has does not stay behind, and nothing outside the package is deleted.
        self.assertEqual(self.result("update_module_replaced"), "new")
        self.assertTrue(self.result("update_stale_module_gone"))
        self.assertTrue(self.result("update_model_kept"))

    def test_missing_optional_flags_are_only_a_note(self):
        # The agent leaves out optional flags a build lacks, so nothing "would fail".
        self.assertEqual(self.result("optflag_one_kind"), "Note")
        self.assertEqual(self.result("optflag_one_text"),
                         "bin\\cuda13: llama-server does not accept --moe-cache-cpu-share; optional, GhostAgent runs without it")
        self.assertFalse(self.result("optflag_old_wording"))
        # A lock-down flag gets a warning of its own.
        self.assertEqual(self.result("optflag_mixed_kinds"), "Note,Warn")
        texts = self.result("optflag_mixed_texts")
        self.assertIn("--moe-cache-cpu-share --image-max-tokens; optional, GhostAgent runs without them", texts)
        self.assertIn("--no-slots; GhostAgent runs without it, but its model server is then locked down less tightly", texts)
        self.assertEqual(self.result("optflag_none"), 0)

    def test_dry_run_names_where_the_model_goes(self):
        self.assertIn("straight to the target drive", self.result("dest_same_drive"))
        self.assertNotIn("staged", self.result("dest_same_drive"))
        self.assertIn("staged in %LOCALAPPDATA%\\GhostAgent\\models", self.result("dest_other_drive"))
        self.assertIn("straight to the target drive", self.result("dest_no_staging"))


class PinTests(unittest.TestCase):
    def test_llama_default_tag_stays_b11669_until_benchmark(self):
        text = BUILD.read_text(encoding="ascii")
        self.assertIn("[string]$LlamaTag = 'b11669'", text)

    def test_requirements_lock_is_fully_hashed(self):
        lines = (REPO / "tools" / "requirements.lock").read_text(encoding="ascii").splitlines()
        reqs = [l for l in lines if l and not l.startswith((" ", "#"))]
        self.assertTrue(any(r.startswith("playwright==1.65.1") for r in reqs))
        hashes = [l.strip() for l in lines if l.strip().startswith("--hash=sha256:")]
        self.assertEqual(len(hashes), 2 * len(reqs))
        for h in hashes:
            self.assertRegex(h, r"^--hash=sha256:[0-9a-f]{64}( \\)?$")

    def test_server_flags_file_shape(self):
        doc = json.loads((REPO / "tools" / "server-flags.json").read_text(encoding="utf-8"))
        flags = doc["required"] + doc["optional"]
        self.assertEqual(len(flags), len(set(flags)))
        for f in flags:
            self.assertRegex(f, r"^--?[A-Za-z][A-Za-z0-9-]*$")
        # The lock-down flags are optional (the agent starts without them, with a warning), and so is
        # --no-webui, which the web_ui setting leaves out.
        self.assertTrue(set(doc["lockdown"]) <= set(doc["optional"]))
        self.assertIn("--no-webui", doc["lockdown"])

    def test_the_build_reads_its_flag_lists_from_the_file(self):
        # One list for the agent, the build and the file: the smoke test has no flag list of its own.
        text = BUILD.read_text(encoding="ascii")
        self.assertIn("$flags.required", text)
        self.assertIn("$flags.optional", text)
        self.assertIn("$flags.lockdown", text)
        for flag in ("--no-webui", "--cors-origins", "--moe-cache-cpu-share", "--image-max-tokens"):
            self.assertNotIn(f"'{flag}'", text)


class UpdateCmdTests(unittest.TestCase):
    def test_quoted_argument_with_ampersand_reaches_both_steps(self):
        # A stand-in build script that prints the arguments it receives, in a folder whose name
        # holds the characters cmd.exe treats specially.
        root = temp_dir(self, prefix="upd-") / "Ghost & Co (x)"
        (root / "tools").mkdir(parents=True)
        shutil.copy(UPDATE_CMD, root / UPDATE_CMD.name)
        (root / "tools" / "Build-GhostAgent.ps1").write_text(
            "param([Parameter(ValueFromRemainingArguments=$true)]$a)\r\n"
            "foreach ($x in $a) { Write-Output \"ARG <$x>\" }\r\n"
            "Write-Output 'END'\r\nexit 0\r\n", encoding="ascii")
        cmdline = f'"{CMD}" /d /s /c ""{root / UPDATE_CMD.name}" -UpdateFrom "D:\\A&B (1).zip" -Profile fast"'
        r = subprocess.run(cmdline, capture_output=True, text=True, timeout=120, env=ps_env(root.parent),
                           creationflags=CREATE_NO_WINDOW, stdin=subprocess.DEVNULL)
        out = r.stdout + r.stderr
        self.assertNotIn("is not recognized", out)
        runs = [chunk for chunk in out.split("END") if "ARG" in chunk]
        self.assertEqual(len(runs), 2, out)
        update, build = runs
        self.assertIn("ARG <-Update>", update)
        self.assertIn("ARG <-OfferCleanup>", build)
        for run in runs:
            self.assertIn("ARG <D:\\A&B (1).zip>", run)
            self.assertIn("ARG <fast>", run)
        self.assertFalse((root.parent / "GhostAgent-Update.cmd").exists(), "the copy in %TEMP% was left behind")

    def test_the_temp_copy_goes_and_a_failure_still_shows(self):
        root = temp_dir(self, prefix="upd-") / "GhostAgent"
        (root / "tools").mkdir(parents=True)
        shutil.copy(UPDATE_CMD, root / UPDATE_CMD.name)
        (root / "tools" / "Build-GhostAgent.ps1").write_text("Write-Output 'UPDATE RAN'\r\nexit 3\r\n", encoding="ascii")
        temp = root.parent / "temp"
        temp.mkdir()
        r = subprocess.run(f'"{CMD}" /d /s /c ""{root / UPDATE_CMD.name}""', capture_output=True, text=True,
                           timeout=120, env=ps_env(temp), creationflags=CREATE_NO_WINDOW, stdin=subprocess.DEVNULL)
        out = r.stdout + r.stderr
        self.assertIn("UPDATE RAN", out)
        self.assertIn("The update stopped", out)  # the failure is still reported before the copy goes
        self.assertNotIn("cannot find", out.lower())
        self.assertEqual(list(temp.iterdir()), [], "the copy in %TEMP% was left behind")
        self.assertTrue((root / UPDATE_CMD.name).exists())  # only the copy deletes itself


if __name__ == "__main__":
    unittest.main()
