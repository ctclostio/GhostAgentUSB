<#
.SYNOPSIS
  Fills GhostAgent\ with everything the agent needs: the Holo4 model, llama.cpp, portable Python
  with Playwright, and a portable Chromium. Run it through "Build GhostAgent.cmd" (or through
  "Update GhostAgent.cmd" to bring an existing copy up to date).

.DESCRIPTION
  Every part comes from its official source at a pinned version and is checked against a pinned
  SHA-256 sum before it is used:
    - the llama.cpp release archives (see -LlamaTag)
    - the Python embeddable package and pip
    - the Playwright and greenlet wheels (tools\requirements.lock, installed with --require-hashes)
    - the Chrome for Testing archive that the pinned Playwright expects
    - the model files (tools\profiles.json, each pinned to a Hugging Face revision)

  It is safe to run again. Each installed part carries a stamp file (.component) that names the
  pinned archives it came from: a part whose stamp matches is skipped, any other part is replaced
  as a whole. Downloads resume where they stopped. A model file is hashed once; later runs only
  compare its size and modification time with models\.verified.

.PARAMETER ModelProfile
  (-Profile) Which model to set up: fast (the default in tools\profiles.json, Holo4-35B-A3B
  Q4_K_M), fast-q8, accurate (Holo4-27B, non-commercial license) or all.

.PARAMETER LlamaTag
  The llama.cpp release to install: b11669 (default) or b12359 (v1.0.0). Run the build again with
  the other tag to switch; only the llama.cpp folders are replaced.

.PARAMETER ModelFrom
  A folder that already holds the model files (for example an older copy of the stick). A file
  there is hard-linked when it is on the same drive (instant), otherwise copied, then checked.

.PARAMETER MoveModel
  With -ModelFrom on the same drive: move the files instead of hard-linking them.

.PARAMETER SkipModel
  Set up everything except the model.

.PARAMETER SkipChromium
  Do not install the portable Chromium. The "bundled" browser mode then does not work; Chrome
  and Edge modes still do.

.PARAMETER NoStaging
  Download the model straight to the target drive instead of first to this PC's disk.

.PARAMETER KeepDownloads
  Keep the downloaded archives in %TEMP%\GhostAgent-build instead of deleting them.

.PARAMETER Proxy
  A proxy for every download, for example http://proxy.example.com:8080. Without it, the proxy
  in HTTPS_PROXY / HTTP_PROXY or the Windows proxy setting is used.

.PARAMETER Arch
  x64 or arm64 parts. Detected automatically; set it to list the other architecture with -DryRun.

.PARAMETER DryRun
  Print what would be downloaded (URL, size, SHA-256, destination) and change nothing.

.PARAMETER OfferCleanup
  Offer to delete model files in GhostAgent\models that no profile uses ("Update GhostAgent.cmd"
  passes this).

.PARAMETER Update
  Fetch the newest GhostAgent files (git pull in a git clone, otherwise the latest ZIP from
  GitHub) and copy them over this folder, keeping settings.json, the browser profiles, downloads
  and logs. It does not build; "Update GhostAgent.cmd" runs the build afterwards.

.PARAMETER UpdateFrom
  With -Update: take the new files from this ZIP file, folder or https:// URL instead.
#>
[CmdletBinding()]
param(
    [Alias('Profile')]
    [string]$ModelProfile,
    [ValidateSet('b11669', 'b12359')]
    [string]$LlamaTag = 'b11669',
    [string]$ModelFrom,
    [switch]$MoveModel,
    [switch]$SkipModel,
    [switch]$SkipChromium,
    [switch]$NoStaging,
    [switch]$KeepDownloads,
    [string]$Proxy,
    [ValidateSet('auto', 'x64', 'arm64')]
    [string]$Arch = 'auto',
    [switch]$DryRun,
    [switch]$OfferCleanup,
    [switch]$Update,
    [string]$UpdateFrom
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Root  = Split-Path -Parent $PSScriptRoot
$Ghost = Join-Path $Root 'GhostAgent'
$Tools = $PSScriptRoot
$Cache = Join-Path $env:TEMP 'GhostAgent-build'
$Repo  = 'ctclostio/GhostAgentUSB'

# Windows' own tools, by full path, so that another curl.exe or tar.exe earlier in PATH is never used.
$Sys32    = Join-Path $env:SystemRoot 'System32'
$Curl     = Join-Path $Sys32 'curl.exe'
$Tar      = Join-Path $Sys32 'tar.exe'
$Robocopy = Join-Path $Sys32 'Robocopy.exe'
$CertUtil = Join-Path $Sys32 'certutil.exe'
$CmdExe   = Join-Path $Sys32 'cmd.exe'

# App Control policies can run this script in Constrained Language Mode, where .NET calls are
# refused. The script sticks to cmdlets and Windows tools, and checks this before its few .NET calls.
$FullLanguage = $ExecutionContext.SessionState.LanguageMode -eq 'FullLanguage'

# --- pinned versions --------------------------------------------------------------------------

# llama.cpp release assets: tag -> architecture -> folder under bin\ -> archives unpacked into it,
# each as "name size sha256". Sizes and sums are the ones the GitHub release API publishes.
# b11669 stays the default until the GPU benchmark accepts b12359 (v1.0.0). Both stay selectable,
# so either one is the rollback for the other. CUDA 13 moved from 13.4 to 13.5 between them, which
# changes the asset names too; the CUDA 13.5 build needs NVIDIA driver 580 or newer (the agent falls
# back to the CUDA 12 build on older drivers).
$LlamaReleases = @{
    'b11669' = @{
        x64 = [ordered]@{
            cuda13 = 'llama-b11669-bin-win-cuda-13.4-x64.zip 154582227 3b92d5c064f5032e603e531d3ec8961f3bd6613e5dfad047399a92dec8e95dbc',
                     'cudart-llama-bin-win-cuda-13.4-x64.zip 423535356 738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668'
            cuda12 = 'llama-b11669-bin-win-cuda-12.4-x64.zip 266582698 c2da1f73ed176ae093ac5aceb978a7b167a16113f6d74c6789eeaee2135cb8dc',
                     'cudart-llama-bin-win-cuda-12.4-x64.zip 391443627 8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6'
            vulkan = 'llama-b11669-bin-win-vulkan-x64.zip 32182350 2c07574e7bde71321cb785282b20cc821e951132149eeceeea32ddebdbf0598e'
            cpu    = 'llama-b11669-bin-win-cpu-x64.zip 19537886 37b3efcf962847ca869745fd4ccf7f5e7e8dacfeff9fa3523b4ca8cdc66d632b'
        }
        arm64 = [ordered]@{
            cuda13 = 'llama-b11669-bin-win-cuda-13.4-arm64.zip 146809734 6759d52197bef2955807c53778dc045110ac4e8677c4f113edad9fe5f8e9e97f',
                     'cudart-llama-bin-win-cuda-13.4-arm64.zip 153262407 642dcde8805b3e3165ca710a5443b3b4044b27d96bd3ee3132473988c9bcb774'
            vulkan = 'llama-b11669-bin-win-vulkan-arm64.zip 24817965 963d3b185362cadca3de643409f916a5ce95dd0f7fd8617ba9ee5d8833ac1ef3'
            cpu    = 'llama-b11669-bin-win-cpu-arm64.zip 12442177 9cff39ffed17edcd03a7d876073f74d7b48cd43a1e5eafdd59ee094185ad871b'
        }
    }
    'b12359' = @{
        x64 = [ordered]@{
            cuda13 = 'llama-b12359-bin-win-cuda-13.5-x64.zip 179546112 1de9ca1ff408d6e348ea0007c6880ad2393311a02e7799bcde26dc16daef03dd',
                     'cudart-llama-bin-win-cuda-13.5-x64.zip 426314199 c2bf4cc202bfbaaa7a0b75f5082457c3811326947bb216059b72839d844e9cba'
            cuda12 = 'llama-b12359-bin-win-cuda-12.4-x64.zip 318976332 e9c0abf442e8339c06728366c1ad2c711332f9939c0c00603fb75d85fd44c914',
                     'cudart-llama-bin-win-cuda-12.4-x64.zip 391443627 8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6'
            vulkan = 'llama-b12359-bin-win-vulkan-x64.zip 38090643 e8d18c6e4210cca87e7307c2ea78363ac8d158e4e1397358ec96ab70b53a0d5e'
            cpu    = 'llama-b12359-bin-win-cpu-x64.zip 25097509 837c1655e87df3b66989f4be33a792a1399c785bf259038dc75c900096683e69'
        }
        arm64 = [ordered]@{
            cuda13 = 'llama-b12359-bin-win-cuda-13.5-arm64.zip 167147417 f40398808e74b88c3ccee904824667ef9825692843533e7fc22881ab2e4de9da',
                     'cudart-llama-bin-win-cuda-13.5-arm64.zip 278821062 f89e7b2457f627d052dd99f42205ddfd01964454809754cdc649cffc74320dca'
            vulkan = 'llama-b12359-bin-win-vulkan-arm64.zip 26027698 62cd50089cffdd825524a91c5a8819041eb75d8e2beacd6eca68194dba50574f'
            cpu    = 'llama-b12359-bin-win-cpu-arm64.zip 13455326 d27564c754478d2832789dfcebfc5ace018e36a224c41d810b7cbc1ba8243ec0'
        }
    }
}
$LlamaBase = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaTag"
# Rough size of each unpacked backend (MB), for the free-space check.
$LlamaInstallMB = @{ cuda13 = 900; cuda12 = 1300; vulkan = 120; cpu = 80 }

$PythonVersion  = '3.14.8'
$PythonBase     = "https://www.python.org/ftp/python/$PythonVersion"
$PythonArchives = @{
    x64   = 'python-3.14.8-embed-amd64.zip 12601679 a93abe456ab01bd96d7a085b3cdb6566b3063f4241360d114142fbdb07f0a310'
    arm64 = 'python-3.14.8-embed-arm64.zip 11923092 155be84ccb57c6331cf0e39001c78a1dfac3be62f403f3ff5f2e29b80dda7ebe'
}
$PipWheel = 'pip-26.3-py3-none-any.whl 1950951 f3b60b20ffaadd4415ed7957212a38ac63fa4c5fa375bd0f4a797777940fad13'
$PipUrl   = 'https://files.pythonhosted.org/packages/07/25/ee8c38ae72d66b844ce8eb9711ba905dba21ca1a8a0c960057c41cd23337/pip-26.3-py3-none-any.whl'
$Lock     = Join-Path $Tools 'requirements.lock'

# The Chrome for Testing build that the pinned Playwright expects (its driver\package\browsers.json;
# the build stops if the two disagree). It is downloaded and checked here because Playwright's own
# installer does not verify what it downloads. There is no Windows ARM64 build, so ARM64 PCs get the
# x64 one (it runs under emulation). chrome.exe and chrome.dll sums are for the fallback through
# "playwright install".
$Chromium = @{
    Revision  = '1254'
    Version   = '158.0.8127.18'
    Archive   = 'chrome-win64-158.0.8127.18.zip 215820284 b796e8a6231463d1a150ee679fd7cb0168b3728162d2d19bf7bbd867d60d55e2'
    Urls      = @('https://storage.googleapis.com/chrome-for-testing-public/158.0.8127.18/win64/chrome-win64.zip',
                  'https://cdn.playwright.dev/builds/cft/158.0.8127.18/win64/chrome-win64.zip')
    ChromeExe = 'dafaf388c1194eb610fc5706ba986eb38e5ebedb1fae44f39569e20940a3bae2'
    ChromeDll = 'c6b849fa1d8e4b2ddd8a8322e767b0baf00eaa3ccb3cfbe681d86daded6019ed'
}

# --- helpers ----------------------------------------------------------------------------------

function Step($msg) { Write-Host ''; Write-Host "== $msg" -ForegroundColor Cyan }
function Note($msg) { Write-Host "   $msg" -ForegroundColor DarkGray }
function Warn($msg) { Write-Host "   $msg" -ForegroundColor Yellow }

# Sizes in decimal GB/MB, the unit drives are sold in.
function Format-Size($Bytes) {
    $b = [double]$Bytes
    if ($b -ge 1e9) { return '{0:N1} GB' -f ($b / 1e9) }
    if ($b -ge 1e6) { return '{0:N0} MB' -f ($b / 1e6) }
    $kb = [int64]($b / 1e3)
    if ($kb -lt 1 -and $b -gt 0) { $kb = 1 }
    return '{0:N0} KB' -f $kb
}

# "name size sha256" -> download item
function ConvertTo-Pin([string]$Spec, [string[]]$Urls) {
    $f = $Spec -split '\s+'
    return @{ Name = $f[0]; Size = [int64]$f[1]; Sha256 = $f[2].ToLower(); Urls = $Urls }
}

function Get-GhostVersion {
    $f = Join-Path $Root 'VERSION'
    # "$(...)" rather than [string](...): in Windows PowerShell 5.1 an empty result cast to [string]
    # stays $null, and .Trim() on it fails.
    $v = ''
    if (Test-Path -LiteralPath $f) { $v = "$(Get-Content -LiteralPath $f -TotalCount 1)".Trim() }
    if ($v) { return $v }
    return 'unknown'
}

# Runs a program and returns its exit code and output (stdout and stderr) as text lines.
# Windows PowerShell turns stderr lines into errors, so errors are not fatal inside this function.
function Invoke-Captured([string]$Exe, [string[]]$Arguments) {
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $Exe @Arguments 2>&1 | ForEach-Object { "$_" }
        return @{ ExitCode = $LASTEXITCODE; Output = @($out); Failed = $null }
    } catch {
        return @{ ExitCode = -1; Output = @(); Failed = $_.Exception.Message }
    }
}

# Quotes one argument for a command line. A trailing backslash would escape the closing quote, so
# a drive root such as D:\ is passed as "D:\.".
function Format-Arg([string]$Value) {
    if ($Value.EndsWith('\')) { $Value += '.' }
    return '"' + $Value + '"'
}

# Deletes a folder. cmd's "rd /s" removes a junction or symbolic link itself instead of following
# it into its target, which Windows PowerShell's Remove-Item -Recurse can do. The path goes through
# an environment variable so that characters such as & in it stay literal.
function Remove-Dir([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $ErrorActionPreference = 'Continue'
    $env:GHOSTAGENT_RD = $Path
    & $CmdExe /d /c rd /s /q '"%GHOSTAGENT_RD%"' 2>$null | Out-Null
    Remove-Item Env:GHOSTAGENT_RD -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $Path) {
        throw "Could not delete $Path. Close GhostAgent (and anything else using files in it), then run the build again."
    }
}

# SHA-256 of a file. Three ways, because none works everywhere:
#  1. .NET, in 64 MB chunks with progress. Works even when Windows PowerShell inherits PowerShell 7's
#     module path (which hides Get-FileHash), but not in Constrained Language Mode.
#  2. Get-FileHash, which signed system modules can still run in Constrained Language Mode.
#  3. certutil -hashfile, which needs neither.
function Get-Sha256([string]$Path, [string]$Label) {
    $size = [int64](Get-Item -LiteralPath $Path).Length
    if (-not $Label) { $Label = Split-Path -Leaf $Path }
    if ($FullLanguage) {
        try { return Get-Sha256Net $Path $Label $size } catch { Note "hashing through .NET failed ($($_.Exception.Message)); trying another way" }
    }
    if ($size -ge 256MB) { Note "checking $Label ($(Format-Size $size), no progress shown in this mode)" }
    try {
        $h = Get-FileHash -LiteralPath $Path -Algorithm SHA256 -ErrorAction Stop
        if ($h.Hash) { return ([string]$h.Hash).ToLower() }
    } catch { }
    $r = Invoke-Captured $CertUtil @('-hashfile', $Path, 'SHA256')
    if ($r.ExitCode -eq 0) {
        foreach ($line in $r.Output) {
            $hex = $line -replace '\s', ''
            if ($hex -match '^[0-9a-fA-F]{64}$') { return $hex.ToLower() }
        }
    }
    throw "Could not compute the SHA-256 of $Path."
}

function Get-Sha256Net([string]$Path, [string]$Label, [int64]$Size) {
    $sha = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::Open($Path, 'Open', 'Read', 'ReadWrite')
    $buf = New-Object byte[] (64MB)
    $done = [int64]0
    $t0 = Get-Date
    $show = $Size -ge 256MB
    try {
        while (($n = $stream.Read($buf, 0, $buf.Length)) -gt 0) {
            [void]$sha.TransformBlock($buf, 0, $n, $null, 0)
            $done += $n
            if ($show) {
                $sec = ((Get-Date) - $t0).TotalSeconds
                $rate = if ($sec -gt 0) { $done / 1e6 / $sec } else { 0 }
                Write-Host -NoNewline ("`r   checking {0}: {1,3}%  ({2:N0} MB/s) " -f $Label, [int64](100 * $done / $Size - 0.5), $rate) -ForegroundColor DarkGray
            }
        }
        [void]$sha.TransformFinalBlock($buf, 0, 0)
        if ($show) { Write-Host '' }
        return ([BitConverter]::ToString($sha.Hash) -replace '-', '').ToLower()
    } finally { $stream.Dispose(); $sha.Dispose() }
}

# The volume that holds $Path: root, file system, free bytes and an id to compare volumes. Works
# for drive letters and for volumes mounted in a folder; returns $null for network paths.
function Get-VolumeInfo([string]$Path) {
    $full = $Path
    try { $full = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).ProviderPath } catch { }
    if (-not $full.EndsWith('\')) { $full += '\' }
    $low = $full.ToLower()
    $best = $null
    try {
        if ($null -eq $script:VolumeList) { $script:VolumeList = @(Get-CimInstance Win32_Volume -ErrorAction Stop) }
        foreach ($v in $script:VolumeList) {
            $name = [string]$v.Name
            if (-not $name -or $name.StartsWith('\\')) { continue }
            if ($low.StartsWith($name.ToLower()) -and (-not $best -or $name.Length -gt $best.Root.Length)) {
                $best = @{ Root = $name; FileSystem = [string]$v.FileSystem; FreeBytes = [int64]$v.FreeSpace; Id = [string]$v.DeviceID }
            }
        }
    } catch { }
    if ($best) { return $best }
    if ($full -match '^([A-Za-z]:)\\') {
        try {
            $d = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$($Matches[1])'" -ErrorAction Stop
            if ($d) { return @{ Root = "$($d.DeviceID)\"; FileSystem = [string]$d.FileSystem; FreeBytes = [int64]$d.FreeSpace; Id = [string]$d.DeviceID } }
        } catch { }
    }
    return $null
}

function Reset-VolumeList { $script:VolumeList = $null }

function Test-SameVolume([string]$A, [string]$B) {
    $va = Get-VolumeInfo $A
    $vb = Get-VolumeInfo $B
    return [bool]($va -and $vb -and $va.Id -eq $vb.Id)
}

# FAT and FAT32 cannot hold a file of 4 GB or more, and every model file is far larger. This runs
# before anything is downloaded, so the build does not fail at 4 GB (and again on every rerun).
function Assert-LargeFileSupport($Volume, [int64]$LargestFile, [string]$What) {
    if (-not $Volume -or -not $Volume.FileSystem) { return }
    $fs = ([string]$Volume.FileSystem).ToUpper()
    if ($fs -ne 'FAT' -and $fs -ne 'FAT32') { return }
    $where = ([string]$Volume.Root).TrimEnd('\')
    throw ("$where is formatted $fs, which cannot hold files larger than 4 GB. $What is $(Format-Size $LargestFile). " +
           "Back up the stick, reformat it as NTFS (or exFAT), and run again. Formatting erases everything on it. " +
           "(-SkipModel sets up everything except the model on this drive.)")
}

# Proxy from the Windows setting (Internet Options), for curl and pip, which only read environment
# variables. Returns e.g. 'http://proxy:8080', or $null when no fixed proxy is set.
function Get-SystemProxy([string]$Key = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings') {
    try { $p = Get-ItemProperty -LiteralPath $Key -ErrorAction Stop } catch { return $null }
    if (-not $p.ProxyEnable -or -not $p.ProxyServer) { return $null }
    $server = ([string]$p.ProxyServer).Trim()
    $pick = $null
    if ($server -match '=') {
        # Per-protocol form: "http=host:port;https=host:port;socks=host:port"
        foreach ($scheme in 'https', 'http') {
            foreach ($part in $server -split ';') {
                if (-not $pick -and $part.Trim() -match "^$scheme=(.+)$") { $pick = $Matches[1].Trim() }
            }
        }
    } else { $pick = $server }
    if (-not $pick) { return $null }
    if ($pick -notmatch '^[A-Za-z][A-Za-z0-9+.-]*://') { $pick = "http://$pick" }
    return $pick
}

# Proxy for every download (the build and the update): -Proxy, else the environment (curl and pip
# read it themselves), else the Windows setting. Sets curl's arguments; returns the proxy given to
# pip, or $null.
function Initialize-Proxy {
    $script:CurlProxyArgs = @()
    $proxyUrl = $null
    if ($Proxy) { $proxyUrl = $Proxy; Note "using the proxy $Proxy" }
    elseif (Test-ProxyEnv) { Note "using the proxy from the $(Test-ProxyEnv) environment variable" }
    else {
        $proxyUrl = Get-SystemProxy
        if ($proxyUrl) { Note "using the Windows proxy setting: $proxyUrl" }
        else {
            $ie = Get-ItemProperty -LiteralPath 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' -ErrorAction SilentlyContinue
            if ($ie -and $ie.AutoConfigURL) { Note 'Windows uses a proxy script, which the build cannot read. If downloads fail, run it with -Proxy http://host:port.' }
        }
    }
    if ($proxyUrl) {
        $script:CurlProxyArgs = @('--proxy', $proxyUrl)
        # For pip and for Playwright's own downloader (the Chromium fallback).
        $env:HTTPS_PROXY = $proxyUrl
        $env:HTTP_PROXY = $proxyUrl
    }
    return $proxyUrl
}

# Environment variable names are not case-sensitive on Windows, so HTTPS_PROXY also finds https_proxy.
function Test-ProxyEnv {
    foreach ($n in 'HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY') {
        if ((Test-Path "Env:$n") -and (Get-Item "Env:$n").Value) { return $n }
    }
    return $null
}

# Downloads $Urls (mirrors, tried in order) to $Part, resuming whatever is already in $Part. curl's
# own --retry does not cover a dropped connection, and it can throw away the bytes of a failed try,
# so this loop restarts curl itself up to 30 times with a growing pause; each try resumes (-C -)
# from the bytes on disk. A connection slower than 100 KB/s for a minute is dropped and resumed.
function Invoke-ResumableDownload([string[]]$Urls, [string]$Part, [int64]$Size, [string]$Label) {
    $maxTries = 30
    $wait = 5
    $mirror = 0
    for ($try = 1; $try -le $maxTries; $try++) {
        $have = [int64]0
        if (Test-Path -LiteralPath $Part) { $have = [int64](Get-Item -LiteralPath $Part).Length }
        if ($Size -gt 0 -and $have -gt $Size) { Remove-Item -LiteralPath $Part -Force; $have = 0 }
        if ($Size -gt 0 -and $have -eq $Size) { return }
        $url = $Urls[$mirror]
        if ($have -gt 0) { Note "resuming $Label at $(Format-Size $have) (try $try of $maxTries)" }
        elseif ($try -gt 1) { Note "downloading $Label again (try $try of $maxTries)" }
        $curlArgs = @('-q', '-L', '--fail', '--proto', '=https', '--proto-redir', '=https', '-C', '-',
                      '--retry', '3', '--speed-limit', '102400', '--speed-time', '60', '--connect-timeout', '30',
                      '--ssl-revoke-best-effort', '-w', '%{http_code}', '-o', $Part) + $CurlProxyArgs + @($url)
        $out = & $Curl @curlArgs
        $rc = $LASTEXITCODE
        # A curl that was stopped prints nothing at all, so $out can be empty.
        $http = "$(@($out) | Select-Object -Last 1)".Trim()
        if ($rc -eq 0) {
            if ($Size -le 0) { return }
            $have = [int64](Get-Item -LiteralPath $Part).Length
            if ($have -eq $Size) { return }
            if ($have -gt $Size) { Remove-Item -LiteralPath $Part -Force }
            Note "$Label is incomplete ($(Format-Size $have) of $(Format-Size $Size))"
        } elseif ($rc -eq 23) {
            throw "Could not write $Part. The drive may be full, or unable to hold a file this large."
        } elseif ($rc -eq 22 -and $http -eq '416') {
            # The server cannot continue from this point: the partial file does not belong to it.
            Remove-Item -LiteralPath $Part -Force -ErrorAction SilentlyContinue
            continue
        } elseif ($rc -eq 33) {
            Note 'the server cannot resume downloads; starting this file from the beginning'
            Remove-Item -LiteralPath $Part -Force -ErrorAction SilentlyContinue
            continue
        } elseif ($rc -eq 22 -and $http -match '^(400|401|403|404|410|451)$') {
            if ($mirror + 1 -lt $Urls.Count) {
                $mirror++
                Note "HTTP $http from $url; trying $($Urls[$mirror])"
                continue
            }
            throw "Downloading $Label failed: the server answered HTTP $http ($url)."
        }
        if ($try -lt $maxTries) {
            # A curl that was cut off leaves its progress line unfinished; start on a fresh line.
            Write-Host ''
            Note "the download stopped (curl exit code $rc); trying again in $wait s"
            Start-Sleep -Seconds $wait
            $wait = $wait * 2
            if ($wait -gt 60) { $wait = 60 }
        }
    }
    throw "Downloading $Label did not finish after $maxTries tries. Check the internet connection and run the build again: it continues where it stopped."
}

# Downloads one pinned archive into the cache (resuming) and checks it. A cached file that does not
# match is replaced. A fresh download that does not match is deleted and the build stops: fetching
# the same file again would not help.
function Get-Download($Item, [string]$Dest) {
    $name = $Item.Name
    if (Test-Path -LiteralPath $Dest) {
        if ([int64](Get-Item -LiteralPath $Dest).Length -eq $Item.Size -and (Get-Sha256 $Dest $name) -eq $Item.Sha256) {
            Note "${name}: already downloaded"
            return
        }
        Remove-Item -LiteralPath $Dest -Force
    }
    $part = "$Dest.part"
    Note "downloading $name ($(Format-Size $Item.Size))"
    Invoke-ResumableDownload $Item.Urls $part $Item.Size $name
    $sum = Get-Sha256 $part $name
    if ($sum -ne $Item.Sha256) {
        Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
        throw ("The server's file differs from the pinned version: $name`n" +
               "  expected SHA-256 $($Item.Sha256)`n  received SHA-256 $sum`n" +
               'Nothing from it was installed. If this happens again, the pinned version needs updating.')
    }
    Move-Item -LiteralPath $part -Destination $Dest -Force
}

function Expand-To([string]$Archive, [string]$Dir) {
    New-Item -ItemType Directory -Force -Path $Dir | Out-Null
    & $Tar -xf $Archive -C $Dir
    if ($LASTEXITCODE -ne 0) { throw "Could not unpack $Archive (tar exit code $LASTEXITCODE)." }
}

# --- component stamps -------------------------------------------------------------------------
# Each installed part has a .component file naming the pinned archives (and their sums) it was
# unpacked from. It is written last, so a part interrupted halfway never looks finished.

function Test-Stamp([string]$Dir, [string]$Text) {
    $f = Join-Path $Dir '.component'
    if (-not (Test-Path -LiteralPath $f)) { return $false }
    $have = Get-Content -LiteralPath $f -Raw
    if (-not $have) { return $false }
    return ($have -replace '\r', '').Trim() -eq ($Text -replace '\r', '').Trim()
}

function Write-Stamp([string]$Dir, [string]$Text) {
    Set-Content -LiteralPath (Join-Path $Dir '.component') -Value $Text -Encoding Ascii
}

# Downloads the archives, unpacks them into <dir>.tmp, runs $AfterUnpack on it, stamps it, and only
# then swaps it in for <dir>. An interrupted run leaves either the old part or no part, never half.
function Install-Component([string]$Dir, $Items, [string]$Stamp, [scriptblock]$AfterUnpack) {
    $tmp = "$Dir.tmp"
    $old = "$Dir.old"
    foreach ($item in $Items) { Get-Download $item (Join-Path $Cache $item.Name) }
    Remove-Dir $tmp
    Remove-Dir $old
    foreach ($item in $Items) { Expand-To (Join-Path $Cache $item.Name) $tmp }
    if ($AfterUnpack) { & $AfterUnpack $tmp }
    Write-Stamp $tmp $Stamp
    if (Test-Path -LiteralPath $Dir) {
        # Renaming fails as a whole while a program in the folder is running, so a part that is in
        # use is never left half-deleted.
        try { Rename-Item -LiteralPath $Dir -NewName (Split-Path -Leaf $old) }
        catch { throw "Could not replace $Dir because it is in use. Close GhostAgent, then run the build again." }
    }
    Rename-Item -LiteralPath $tmp -NewName (Split-Path -Leaf $Dir)
    Remove-Item -LiteralPath (Join-Path $old '.component') -Force -ErrorAction SilentlyContinue
    try { Remove-Dir $old } catch { Note "could not delete $old yet; the next run removes it" }
}

# llama-server needs only llama-server.exe, its -impl.dll and the shared DLLs. The other tools
# (llama-cli, bench, quantize, rpc-server and so on) are about 27 unsigned programs per backend that
# GhostAgent never runs, so they are not put on the stick.
function Remove-UnusedTools([string]$Dir) {
    foreach ($f in @(Get-ChildItem -LiteralPath $Dir -File)) {
        $n = $f.Name.ToLower()
        $drop = ($n.EndsWith('.exe') -and $n -ne 'llama-server.exe') -or ($n.EndsWith('-impl.dll') -and $n -ne 'llama-server-impl.dll')
        if ($drop) { Remove-Item -LiteralPath $f.FullName -Force }
    }
}

# --- model files ------------------------------------------------------------------------------
# models\.verified records each model file that passed its SHA-256 check (name, size, modification
# time, sum). While size and time still match, the file is not hashed again: that is minutes per
# file on a USB stick.

function Read-Verified([string]$Dir) {
    $map = @{}
    $f = Join-Path $Dir '.verified'
    if (Test-Path -LiteralPath $f) {
        try {
            $j = Get-Content -LiteralPath $f -Raw | ConvertFrom-Json
            foreach ($p in $j.files.PSObject.Properties) { $map[$p.Name] = $p.Value }
        } catch { }
    }
    return $map
}

# Size and modification time in nanoseconds since 1970 (the same number Python's os.stat gives).
function Get-FileStamp([string]$Path) {
    $i = Get-Item -LiteralPath $Path
    return @{ Size = [int64]$i.Length; MtimeNs = ([int64]$i.LastWriteTimeUtc.Ticks - 621355968000000000) * 100 }
}

function Test-Verified([string]$Path, $Pin) {
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $rec = (Read-Verified (Split-Path -Parent $Path))[(Split-Path -Leaf $Path)]
    if (-not $rec) { return $false }
    $s = Get-FileStamp $Path
    return ([int64]$rec.size -eq $s.Size) -and ($s.Size -eq $Pin.Size) -and ([int64]$rec.mtime_ns -eq $s.MtimeNs) -and
           ([string]$rec.sha256 -eq $Pin.Sha256)
}

function Write-Verified([string]$Path, $Pin) {
    $dir = Split-Path -Parent $Path
    $map = Read-Verified $dir
    $s = Get-FileStamp $Path
    $map[(Split-Path -Leaf $Path)] = @{ name = (Split-Path -Leaf $Path); size = $s.Size; mtime = $s.MtimeNs / 1e9; mtime_ns = $s.MtimeNs; sha256 = $Pin.Sha256 }
    $files = @{}
    foreach ($k in @($map.Keys)) { if (Test-Path -LiteralPath (Join-Path $dir $k)) { $files[$k] = $map[$k] } }
    $doc = @{ note = 'Written by tools\Build-GhostAgent.ps1 after each file passed its SHA-256 check. While size and mtime_ns (nanoseconds since 1970, UTC) still match, the file is not hashed again.'; files = $files }
    Set-Content -LiteralPath (Join-Path $dir '.verified') -Value ($doc | ConvertTo-Json -Depth 5) -Encoding Ascii
}

# Copies one file into $DestDir with robocopy: it shows progress, uses unbuffered I/O for big files
# (/J) and retries on errors. Exit codes 0-7 mean success.
function Copy-WithRobocopy([string]$Src, [string]$DestDir) {
    $srcDir = Split-Path -Parent $Src
    $leaf = Split-Path -Leaf $Src
    # Run it on the console directly (not through PowerShell's pipeline) so its progress shows.
    $argLine = '{0} {1} {2} /J /R:5 /W:5 /COPY:DT /NJH /NJS /NDL /NC /NS' -f (Format-Arg $srcDir), (Format-Arg $DestDir), (Format-Arg $leaf)
    $proc = Start-Process -FilePath $Robocopy -ArgumentList $argLine -NoNewWindow -Wait -PassThru
    $rc = $proc.ExitCode
    if ($rc -ge 8) { throw "Copying $leaf to $DestDir failed (robocopy exit code $rc)." }
    if (-not (Test-Path -LiteralPath (Join-Path $DestDir $leaf))) { throw "Copying $leaf to $DestDir failed." }
}

# Takes a model file from a folder that already has it. Returns $true when $Dest is in place and good.
function Copy-ModelFrom([string]$Src, [string]$Dest, $Pin, [bool]$Explicit) {
    $name = $Pin.Name
    $from = Split-Path -Parent $Src
    $srcVerified = Test-Verified $Src $Pin
    $how = 'copied'
    if (Test-SameVolume $Src (Split-Path -Parent $Dest)) {
        try {
            if ($Explicit -and $MoveModel) {
                Move-Item -LiteralPath $Src -Destination $Dest
                $how = 'moved'
            } else {
                New-Item -ItemType HardLink -Path $Dest -Value $Src | Out-Null
                $how = 'linked'
            }
        } catch { Note "could not link $name ($($_.Exception.Message)); copying it instead" }
    }
    if ($how -eq 'copied') { Note "copying $name from $from"; Copy-WithRobocopy $Src (Split-Path -Parent $Dest) }
    elseif ($how -eq 'linked') { Note "${name}: hard-linked from $from (same drive, nothing copied)" }
    else { Note "${name}: moved from $from" }
    if ($how -ne 'copied' -and $srcVerified) {
        Write-Verified $Dest $Pin
        Note "${name}: OK (checked earlier in $from)"
        return $true
    }
    $sum = Get-Sha256 $Dest $name
    if ($sum -eq $Pin.Sha256) { Write-Verified $Dest $Pin; Note "${name}: OK"; return $true }
    if ($how -eq 'moved') { Move-Item -LiteralPath $Dest -Destination $Src } else { Remove-Item -LiteralPath $Dest -Force }
    if ($Explicit) { throw "$name in $from does not match its pinned SHA-256 (it is damaged or a different version)." }
    Warn "$name in $from does not match its pinned SHA-256; not using it"
    return $false
}

# Where a downloaded model is checked before it goes to the stick: this PC's disk, which is much
# faster to write and hash than a USB stick and does not depend on the stick staying plugged in.
# The same folder is the agent's copy_model_to_pc cache, so the agent starts fast on this PC.
function Get-StageDir([string]$Models, [int64]$Need, [string]$Name) {
    if ($NoStaging -or -not $env:LOCALAPPDATA) { return $null }
    $stage = Join-Path $env:LOCALAPPDATA 'GhostAgent\models'
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    if (Test-SameVolume $stage $Models) { return $null }
    # Free space changes as files are staged (with -Profile all, one after another), so read it fresh.
    Reset-VolumeList
    $vol = Get-VolumeInfo $stage
    if (-not $vol) { return $null }
    if ($vol.FileSystem -match '^FAT') { return $null }
    # A whole copy staged by an earlier build (for another stick, say) needs no room at all; bytes an
    # unfinished one already has count as space this file has.
    $full = Join-Path $stage $Name
    if ((Test-Path -LiteralPath $full) -and [int64](Get-Item -LiteralPath $full).Length -eq $Need) { return $stage }
    $partial = "$full.part"
    if (Test-Path -LiteralPath $partial) { $Need -= [int64](Get-Item -LiteralPath $partial).Length }
    if ($vol.FreeBytes -lt $Need + 2e9) {
        Note "not enough free space on $($vol.Root) to stage the download there; downloading straight to the target drive"
        return $null
    }
    return $stage
}

# Where -DryRun says a model goes: staged on this PC's disk first only when that is another drive
# than the target (see Get-StageDir, which a dry run must not call: it creates the folder).
function Get-ModelDestNote([string]$Models) {
    if ($NoStaging -or -not $env:LOCALAPPDATA) { return 'models (downloaded straight to the target drive)' }
    $stage = Join-Path $env:LOCALAPPDATA 'GhostAgent\models'
    if (Test-SameVolume $stage $Models) { return 'models (downloaded straight to the target drive, which is this PC''s own disk)' }
    return 'models (staged in %LOCALAPPDATA%\GhostAgent\models when this PC has room)'
}

function Install-ModelFile($Pin, [string]$Models, [string[]]$Candidates) {
    $name = $Pin.Name
    $dest = Join-Path $Models $name
    if (Test-Verified $dest $Pin) { Note "${name}: checked earlier, unchanged"; return }
    if (Test-Path -LiteralPath $dest) {
        $len = [int64](Get-Item -LiteralPath $dest).Length
        if ($len -eq $Pin.Size) {
            if ((Get-Sha256 $dest $name) -eq $Pin.Sha256) { Write-Verified $dest $Pin; Note "${name}: OK"; return }
            Warn "$name here does not match its pinned SHA-256 (a damaged copy); replacing it"
        } else {
            Warn "$name here has the wrong size ($(Format-Size $len), expected $(Format-Size $Pin.Size)); replacing it"
        }
        Remove-Item -LiteralPath $dest -Force
    }

    # A folder that already has the file: -ModelFrom first, then an older HoloAgent copy.
    for ($i = 0; $i -lt $Candidates.Count; $i++) {
        $src = Join-Path $Candidates[$i] $name
        $explicit = ($i -eq 0 -and [bool]$script:ModelFromDir)
        if (-not (Test-Path -LiteralPath $src -PathType Leaf)) {
            if ($explicit) { Note "$name is not in $($Candidates[$i]); downloading it" }
            continue
        }
        if ([int64](Get-Item -LiteralPath $src).Length -ne $Pin.Size) { Note "$src has a different size; not using it"; continue }
        if (Copy-ModelFrom $src $dest $Pin $explicit) { return }
    }

    $part = "$dest.part"
    $stage = Get-StageDir $Models $Pin.Size $name
    if ($stage -and (Test-Path -LiteralPath $part)) {
        # An earlier run was downloading straight to the target drive. Continue that download unless
        # this PC's disk already holds more of the file, rather than start again from zero.
        $staged = Join-Path $stage $name
        $stagedHave = [int64]0
        if (Test-Path -LiteralPath $staged) { $stagedHave = [int64](Get-Item -LiteralPath $staged).Length }
        elseif (Test-Path -LiteralPath "$staged.part") { $stagedHave = [int64](Get-Item -LiteralPath "$staged.part").Length }
        if ([int64](Get-Item -LiteralPath $part).Length -gt $stagedHave) {
            Note "continuing the earlier download of $name on the target drive"
            $stage = $null
        }
    }
    if ($stage) {
        $staged = Join-Path $stage $name
        if (-not (Test-Verified $staged $Pin)) {
            if ((Test-Path -LiteralPath $staged) -and [int64](Get-Item -LiteralPath $staged).Length -eq $Pin.Size -and
                (Get-Sha256 $staged $name) -eq $Pin.Sha256) {
                Write-Verified $staged $Pin
            } else {
                if (Test-Path -LiteralPath $staged) { Remove-Item -LiteralPath $staged -Force }
                Note "downloading $name ($(Format-Size $Pin.Size)) to $stage first"
                Invoke-ResumableDownload $Pin.Urls "$staged.part" $Pin.Size $name
                $sum = Get-Sha256 "$staged.part" $name
                if ($sum -ne $Pin.Sha256) {
                    Remove-Item -LiteralPath "$staged.part" -Force
                    throw ("The server's file differs from the pinned version: $name`n  expected SHA-256 $($Pin.Sha256)`n" +
                           "  received SHA-256 $sum`nThe download was deleted. tools\profiles.json pins this file; it needs updating " +
                           'if the model was changed on purpose. Run the build again only if you think the download was damaged on the way.')
                }
                Move-Item -LiteralPath "$staged.part" -Destination $staged -Force
                Write-Verified $staged $Pin
            }
        }
        # A partial download on the stick from an earlier, unstaged run is no longer needed.
        if (Test-Path -LiteralPath $part) { Remove-Item -LiteralPath $part -Force }
        Note "copying $name to $Models"
        Copy-WithRobocopy $staged $Models
        if ((Get-Sha256 $dest "$name (copy)") -ne $Pin.Sha256) {
            Remove-Item -LiteralPath $dest -Force
            throw ("The copy of $name on the target drive does not match the checked file on this PC ($staged). " +
                   'The drive may be failing, or it was unplugged while copying. Run the build again to copy it once more.')
        }
        Write-Verified $dest $Pin
        Note "${name}: OK"
        return
    }

    Note "downloading $name ($(Format-Size $Pin.Size))"
    Invoke-ResumableDownload $Pin.Urls $part $Pin.Size $name
    $sum = Get-Sha256 $part $name
    if ($sum -ne $Pin.Sha256) {
        Remove-Item -LiteralPath $part -Force
        throw ("The server's file differs from the pinned version: $name`n  expected SHA-256 $($Pin.Sha256)`n" +
               "  received SHA-256 $sum`nThe download was deleted. tools\profiles.json pins this file; it needs updating " +
               'if the model was changed on purpose. Run the build again only if you think the download was damaged on the way.')
    }
    Move-Item -LiteralPath $part -Destination $dest -Force
    Write-Verified $dest $Pin
    Note "${name}: OK"
}

# Model files in $Dir that no profile lists. An unfinished download (.part) of a file a profile
# does list is not one of them: a later build with that profile continues it.
function Get-StaleModelFiles([string]$Dir, $Listed) {
    return @(Get-ChildItem -LiteralPath $Dir -File -ErrorAction SilentlyContinue | Where-Object {
        ($_.Name -like '*.gguf' -or $_.Name -like '*.gguf.part') -and -not $Listed.ContainsKey(($_.Name -replace '\.part$', '')) })
}

# --- keep the PC awake ------------------------------------------------------------------------
# A long download stops when a laptop goes to sleep. A small helper process asks Windows to stay
# awake (SetThreadExecutionState) and ends by itself as soon as this script's process exits.

$script:AwakeHelper = $null
function Start-KeepAwake([string]$PythonDir) {
    $pyw = Join-Path $PythonDir 'pythonw.exe'
    if (-not (Test-Path -LiteralPath $pyw)) { return }
    $code = Join-Path $Cache 'keep-awake.py'
    Set-Content -LiteralPath $code -Encoding Ascii -Value @(
        'import ctypes, sys',
        'k = ctypes.windll.kernel32',
        'k.SetThreadExecutionState(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED',
        'h = k.OpenProcess(0x00100000, False, int(sys.argv[1]))  # SYNCHRONIZE on the build',
        'if h:',
        '    k.WaitForSingleObject(h, 0xFFFFFFFF)')
    try {
        $script:AwakeHelper = Start-Process -FilePath $pyw -ArgumentList @('-I', "`"$code`"", "$PID") -PassThru -WindowStyle Hidden
        Note 'keeping this PC awake while the build runs'
    } catch { Note "could not start the keep-awake helper ($($_.Exception.Message))" }
}

function Stop-KeepAwake {
    if ($script:AwakeHelper) {
        Stop-Process -Id $script:AwakeHelper.Id -Force -ErrorAction SilentlyContinue
        $script:AwakeHelper = $null
    }
}

# --- update -----------------------------------------------------------------------------------

function Find-SourceRoot([string]$Dir) {
    if (Test-Path -LiteralPath (Join-Path $Dir 'tools\Build-GhostAgent.ps1')) { return $Dir }
    foreach ($d in @(Get-ChildItem -LiteralPath $Dir -Directory)) {
        if (Test-Path -LiteralPath (Join-Path $d.FullName 'tools\Build-GhostAgent.ps1')) { return $d.FullName }
    }
    throw 'The update does not look like GhostAgent: it has no tools\Build-GhostAgent.ps1.'
}

function Get-LatestZipUrl {
    $json = Join-Path $Cache 'update\release.json'
    $out = & $Curl -q -s -L --proto =https --connect-timeout 30 -o $json -w '%{http_code}' @CurlProxyArgs "https://api.github.com/repos/$Repo/releases/latest"
    if ($LASTEXITCODE -eq 0 -and "$out".Trim() -eq '200') {
        $rel = Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
        if ($rel.zipball_url) { Note "latest release: $($rel.tag_name)"; return [string]$rel.zipball_url }
    }
    Note 'no published release found; using the newest files on the main branch'
    return "https://github.com/$Repo/archive/refs/heads/main.zip"
}

function Get-UpdateSource {
    $work = Join-Path $Cache 'update'
    Remove-Dir $work
    New-Item -ItemType Directory -Force -Path $work | Out-Null
    $zip = Join-Path $work 'GhostAgent-update.zip'
    if ($UpdateFrom -match '^https://') {
        Invoke-ResumableDownload @($UpdateFrom) $zip 0 'the update'
    } elseif ($UpdateFrom) {
        if (-not (Test-Path -LiteralPath $UpdateFrom)) { throw "-UpdateFrom: not found: $UpdateFrom" }
        $full = (Resolve-Path -LiteralPath $UpdateFrom).ProviderPath
        if (Test-Path -LiteralPath $full -PathType Container) { return Find-SourceRoot $full }
        Unblock-File -LiteralPath $full -ErrorAction SilentlyContinue
        $zip = $full
    } else {
        $url = Get-LatestZipUrl
        Note "downloading $url"
        Invoke-ResumableDownload @($url) $zip 0 'the latest GhostAgent'
    }
    $x = Join-Path $work 'files'
    Expand-To $zip $x
    return Find-SourceRoot $x
}

function Update-WithGit([string]$Git, [string]$Settings) {
    $ErrorActionPreference = 'Continue'
    Note 'this folder is a git clone: running git pull'
    $backup = $null
    if (Test-Path -LiteralPath $Settings) {
        $backup = Join-Path $Cache 'settings.json.before-update'
        Copy-Item -LiteralPath $Settings -Destination $backup -Force
        # Local edits to a tracked settings.json make git pull refuse to run. Put the tracked
        # version back for the pull; the user's copy is restored below whatever happens.
        & $Git -C $Root ls-files --error-unmatch -- 'GhostAgent/settings.json' 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { & $Git -C $Root checkout -- 'GhostAgent/settings.json' 2>$null | Out-Null }
    }
    try {
        & $Git -C $Root pull --ff-only | Out-Host
        $rc = $LASTEXITCODE
    } finally {
        if ($backup) { Copy-Item -LiteralPath $backup -Destination $Settings -Force }
    }
    if ($rc -ne 0) { throw "git pull failed (exit code $rc, see above). Nothing else was changed, and settings.json is as it was." }
}

function Copy-UpdateFiles([string]$Src, [string]$Settings) {
    if (-not (Test-Path -LiteralPath (Join-Path $Src 'GhostAgent\agent\ghost_agent.py'))) {
        throw "The update in $Src does not look like GhostAgent: it has no GhostAgent\agent\ghost_agent.py."
    }
    $srcFull = (Resolve-Path -LiteralPath $Src).ProviderPath.TrimEnd('\')
    if ($srcFull -eq $Root.TrimEnd('\')) { throw '-UpdateFrom points at this folder itself.' }
    Note "copying the new files from $srcFull"
    Get-ChildItem -LiteralPath $srcFull -Recurse -File -ErrorAction SilentlyContinue | Unblock-File -ErrorAction SilentlyContinue
    # Never overwritten: the user's settings, the browser profiles (sign-ins), downloads and logs.
    # Nothing is deleted either (no /PURGE), so model files and downloaded parts stay; only the agent's
    # own package is mirrored (below).
    & $Robocopy $srcFull $Root /E /IS /IT /R:2 /W:2 /XF settings.json /XD 'browser-profile*' downloads logs .git /NJH /NJS /NDL /NFL /NP | Out-Host
    $rc = $LASTEXITCODE
    if ($rc -ge 8) { throw "Copying the new files failed (robocopy exit code $rc)." }
    # The agent's package (GhostAgent\agent\ghostagent) is mirrored, so a module the new version no
    # longer has does not stay behind next to the new ones. An update from a release before the package
    # has no such folder and leaves it as it is (that release's ghost_agent.py does not use it).
    $pkg = 'GhostAgent\agent\ghostagent'
    if (Test-Path -LiteralPath (Join-Path $srcFull $pkg) -PathType Container) {
        & $Robocopy (Join-Path $srcFull $pkg) (Join-Path $Root $pkg) /MIR /IS /IT /R:2 /W:2 /NJH /NJS /NDL /NFL /NP | Out-Host
        $rc = $LASTEXITCODE
        if ($rc -ge 8) { throw "Copying the agent's modules failed (robocopy exit code $rc)." }
    }
    $newSettings = Join-Path $srcFull 'GhostAgent\settings.json'
    if (-not (Test-Path -LiteralPath $Settings) -and (Test-Path -LiteralPath $newSettings)) {
        Copy-Item -LiteralPath $newSettings -Destination $Settings
    }
}

function Invoke-Update {
    $old = Get-GhostVersion
    Write-Host "GhostAgent update ($Root, version $old)" -ForegroundColor White
    New-Item -ItemType Directory -Force -Path $Cache | Out-Null
    $settings = Join-Path $Ghost 'settings.json'
    Initialize-Proxy | Out-Null
    $git = Get-Command git.exe -ErrorAction SilentlyContinue
    if (-not $UpdateFrom -and $git -and (Test-Path -LiteralPath (Join-Path $Root '.git'))) {
        Update-WithGit $git.Source $settings
    } else {
        Copy-UpdateFiles (Get-UpdateSource) $settings
        if (-not $KeepDownloads) { Remove-Dir (Join-Path $Cache 'update') }
    }
    # Files from a downloaded ZIP carry the download mark; clear it so nothing prompts later.
    Get-ChildItem -LiteralPath $Root -Recurse -File -ErrorAction SilentlyContinue | Unblock-File -ErrorAction SilentlyContinue
    Write-Host ''
    Write-Host "GhostAgent files updated: version $old -> $(Get-GhostVersion). Your settings.json and browser profiles were kept." -ForegroundColor Green
}

# --- smoke test -------------------------------------------------------------------------------

# Flags in the option column of llama-server --help (lines that start with a dash). A list, compared
# with -ccontains: flags are case-sensitive (-c is the context size, -C a CPU mask), PowerShell's
# hashtables are not.
function Get-HelpFlags($Lines) {
    $flags = @()
    foreach ($line in $Lines) {
        if (-not $line.StartsWith('-')) { continue }
        foreach ($m in (Select-String -InputObject $line -Pattern '(?<![\w-])(--?[A-Za-z0-9][A-Za-z0-9_-]*)' -AllMatches -CaseSensitive).Matches) {
            $flags += $m.Groups[1].Value
        }
    }
    return ,$flags
}

# llama.cpp's Windows builds need the Microsoft Visual C++ runtime, which they do not include. It is
# usually installed with Windows or with other programs, but not always.
function Test-VcRuntime([string]$Dir) {
    foreach ($dll in 'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll') {
        if (-not (Test-Path -LiteralPath (Join-Path $Dir $dll)) -and -not (Test-Path -LiteralPath (Join-Path $Sys32 $dll))) { return $false }
    }
    return $true
}

# A part that turned out broken loses its stamp, so the next run installs it again. (A program that
# Windows refuses to start keeps its stamp: downloading it again would not change that.)
function Clear-Stamp([string]$Dir) {
    Remove-Item -LiteralPath (Join-Path $Dir '.component') -Force -ErrorAction SilentlyContinue
}

# Plain words for the Windows status codes a program that cannot start exits with.
function Get-ExitHint([int64]$Code) {
    switch ($Code) {
        -1073741515 { return ' (a DLL it needs is missing)' }
        -1073741701 { return ' (a DLL is damaged or built for another kind of processor)' }
        -1073741795 { return ' (this processor lacks an instruction the program uses)' }
        -1073741819 { return ' (it crashed)' }
    }
    return ''
}

# What the smoke test says about optional flags a build lacks (tools\server-flags.json). The agent
# leaves such a flag out with a notice, so it is only a note; a lock-down flag gets a warning, since
# the model server is then locked down less tightly (its key still protects it).
function Get-OptionalFlagNotes([string]$Where, [string[]]$Missing, [string[]]$Lockdown) {
    $out = @()
    $plain = @($Missing | Where-Object { $Lockdown -cnotcontains $_ })
    $weaker = @($Missing | Where-Object { $Lockdown -ccontains $_ })
    if ($plain.Count) {
        $out += @{ Kind = 'Note'; Text = "${Where}: llama-server does not accept $($plain -join ' '); optional, GhostAgent runs without $(if ($plain.Count -eq 1) { 'it' } else { 'them' })" }
    }
    if ($weaker.Count) {
        $out += @{ Kind = 'Warn'; Text = ("${Where}: llama-server does not accept $($weaker -join ' '); GhostAgent runs without " +
                                          "$(if ($weaker.Count -eq 1) { 'it' } else { 'them' }), but its model server is then locked down less tightly") }
    }
    return $out
}

# Runs every installed part once, so a broken unpack, a missing DLL or a program that Windows
# blocks on this PC (App Control, Smart App Control) shows up now instead of at the first start.
function Invoke-SmokeTest([string[]]$Backends, [bool]$WithBrowser) {
    Step 'Smoke test'
    $fail = @()
    $vcNoted = $false
    $flagFile = Join-Path $Tools 'server-flags.json'
    $flags = Get-Content -LiteralPath $flagFile -Raw | ConvertFrom-Json
    $build = $LlamaTag.Substring(1)
    foreach ($b in $Backends) {
        $dir = Join-Path $Ghost "bin\$b"
        $exe = Join-Path $dir 'llama-server.exe'
        if (-not (Test-Path -LiteralPath $exe)) { $fail += "bin\$b\llama-server.exe is missing"; Clear-Stamp $dir; continue }
        $v = Invoke-Captured $exe @('--version')
        if ($v.Failed) { $fail += "bin\${b}: Windows did not run llama-server.exe: $($v.Failed)"; continue }
        $verLine = @($v.Output | Where-Object { $_ -match 'version' }) | Select-Object -First 1
        if ($v.ExitCode -eq -1073741515 -and -not (Test-VcRuntime $dir)) {
            # The stamp stays: installing llama.cpp again would not add the missing runtime.
            if (-not $vcNoted) {
                $fail += ('llama-server.exe cannot start: this PC lacks the Microsoft Visual C++ runtime (msvcp140.dll, ' +
                          'vcruntime140.dll) that llama.cpp needs. Install the "Microsoft Visual C++ Redistributable" ' +
                          'from Microsoft (x64, or ARM64 on an ARM PC), then run the build again.')
                $vcNoted = $true
            }
            continue
        }
        if ($v.ExitCode -ne 0 -or -not $verLine) {
            $fail += ("bin\${b}: llama-server.exe --version failed (exit code $($v.ExitCode)$(Get-ExitHint $v.ExitCode)). " +
                      "$(($v.Output | Select-Object -Last 3) -join ' ')").TrimEnd()
            Clear-Stamp $dir
            continue
        }
        if ($verLine -notmatch "\b$build\b") { $fail += "bin\${b}: expected llama.cpp build $build, found: $verLine"; Clear-Stamp $dir }
        $h = Invoke-Captured $exe @('--help')
        $known = Get-HelpFlags $h.Output
        $missReq = @($flags.required | Where-Object { $known -cnotcontains $_ })
        $missOpt = @($flags.optional | Where-Object { $known -cnotcontains $_ })
        if ($missReq.Count) {
            $fail += ("bin\${b}: llama-server does not accept flags GhostAgent always passes: $($missReq -join ' '). " +
                      'This llama.cpp version and tools\server-flags.json disagree; running the build again does not change that.')
        }
        foreach ($line in (Get-OptionalFlagNotes "bin\$b" $missOpt @($flags.lockdown))) {
            if ($line.Kind -eq 'Warn') { Warn $line.Text } else { Note $line.Text }
        }
        $total = @($flags.required).Count + @($flags.optional).Count
        if (-not $missReq.Count) {
            $accepted = if ($missOpt.Count) { "{0} of {1}" -f ($total - $missOpt.Count), $total } else { "all $total" }
            Note ("bin\{0}: {1}; {2} GhostAgent flags accepted" -f $b, $verLine.Trim(), $accepted)
        }
    }

    $py = Join-Path $Ghost 'python\python.exe'
    $smoke = Join-Path $Cache 'smoke_test.py'
    Set-Content -LiteralPath $smoke -Encoding Ascii -Value @(
        'import sys, importlib.metadata',
        'import playwright.sync_api',
        'print("playwright", importlib.metadata.version("playwright"))',
        'if "--browser" in sys.argv:',
        '    from playwright.sync_api import sync_playwright',
        '    try:',
        '        with sync_playwright() as p:',
        '            try:',
        '                b = p.chromium.launch(channel="chromium", headless=True, chromium_sandbox=True)',
        '                sandbox = "on"',
        '            except Exception as e:',
        '                print("sandboxed launch failed:", str(e).strip().splitlines()[0])',
        '                b = p.chromium.launch(channel="chromium", headless=True)',
        '                sandbox = "off"',
        '            page = b.new_page()',
        '            page.goto("about:blank")',
        '            print("chromium", b.version, "sandbox", sandbox)',
        '            b.close()',
        '    except Exception as e:',
        '        print("chromium failed:", " ".join(str(e).split())[:300])',
        '        sys.exit(3)')
    $env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $Ghost 'browsers'
    $pyArgs = @('-I', $smoke)
    if ($WithBrowser) { $pyArgs += '--browser' }
    $r = Invoke-Captured $py $pyArgs
    foreach ($line in $r.Output) {
        if ($line -match '^playwright ') { Note "Python $PythonVersion, $line" }
        elseif ($line -match '^chromium (\S+) sandbox (\w+)') {
            Note "bundled Chromium $($Matches[1]) starts (headless), renderer sandbox $($Matches[2])"
            if ($Matches[2] -ne 'on') { Warn 'the bundled Chromium could not start with its sandbox from this drive' }
        } elseif ($line -match '^sandboxed launch failed: (.*)') { Warn "sandboxed Chromium launch failed: $($Matches[1])" }
    }
    if ($r.Failed) { $fail += "Windows did not run python.exe: $($r.Failed)" }
    elseif ($r.ExitCode -eq 3) {
        $fail += "The bundled Chromium did not start: $((@($r.Output | Where-Object { $_ -match '^chromium failed' }) | Select-Object -First 1) -replace '^chromium failed: ', '')"
        Clear-Stamp (Join-Path $Ghost "browsers\chromium-$($Chromium.Revision)")
    } elseif ($r.ExitCode -ne 0) {
        $fail += "Python/Playwright check failed (exit code $($r.ExitCode)$(Get-ExitHint $r.ExitCode)): $(($r.Output | Select-Object -Last 4) -join ' | ')"
        # -1073741515 means "a DLL was not found": Python itself is incomplete. Otherwise reinstall the packages.
        if ($r.ExitCode -eq -1073741515) { Clear-Stamp (Join-Path $Ghost 'python') }
        else { Clear-Stamp (Join-Path $Ghost 'python\Lib\site-packages') }
    }
    return ,$fail
}

# --- the build --------------------------------------------------------------------------------

function Get-HostArch {
    $a = "$env:PROCESSOR_ARCHITEW6432 $env:PROCESSOR_ARCHITECTURE"
    if ($a -match 'ARM64') { return 'arm64' }
    # x64 PowerShell under emulation on an ARM64 PC reports AMD64 in both variables.
    try { if ((Get-CimInstance Win32_Processor -ErrorAction Stop | Select-Object -First 1).Architecture -eq 12) { return 'arm64' } } catch { }
    if ($a -match 'AMD64') { return 'x64' }
    throw 'GhostAgent needs 64-bit Windows (x64 or ARM64).'
}

function Invoke-Build {
    $version = Get-GhostVersion
    $archKey = if ($Arch -eq 'auto') { Get-HostArch } else { $Arch }

    # Model profiles (tools\profiles.json, shared with the agent).
    $profilesDoc = Get-Content -LiteralPath (Join-Path $Tools 'profiles.json') -Raw | ConvertFrom-Json
    $allProfiles = @($profilesDoc.profiles.PSObject.Properties | ForEach-Object { $_.Name })
    $want = if ($ModelProfile) { $ModelProfile } else { [string]$profilesDoc.default }
    if ($want -eq 'all') { $chosen = $allProfiles }
    elseif ($allProfiles -contains $want) { $chosen = @($want) }
    else { throw "Unknown profile '$want'. Choose one of: $($allProfiles -join ', '), all." }
    $modelPins = @()
    $seen = @{}
    foreach ($p in $chosen) {
        foreach ($kind in 'model', 'mmproj') {
            $f = $profilesDoc.profiles.$p.$kind
            if ($seen.ContainsKey([string]$f.name)) { continue }
            $seen[[string]$f.name] = $true
            $modelPins += @{ Name = [string]$f.name; Size = [int64]$f.size; Sha256 = ([string]$f.sha256).ToLower(); Urls = @([string]$f.url) }
        }
    }
    $listedModels = @{}
    foreach ($p in $allProfiles) {
        foreach ($kind in 'model', 'mmproj') { $listedModels[[string]$profilesDoc.profiles.$p.$kind.name] = $true }
    }

    # Pinned parts for this architecture.
    $builds = $LlamaReleases[$LlamaTag][$archKey]
    $pyPin  = ConvertTo-Pin $PythonArchives[$archKey] @("$PythonBase/$(($PythonArchives[$archKey] -split '\s+')[0])")
    $pipPin = ConvertTo-Pin $PipWheel @($PipUrl)
    $crPin  = ConvertTo-Pin $Chromium.Archive $Chromium.Urls
    $lockLines = @(Get-Content -LiteralPath $Lock | Where-Object { $_.Trim() -and -not $_.Trim().StartsWith('#') } | ForEach-Object { $_.Trim() })
    $pwVersion = ''
    foreach ($l in $lockLines) { if ($l -match '^playwright==([0-9.]+)') { $pwVersion = $Matches[1] } }
    $verParts = $PythonVersion -split '\.'
    $pyTag = "python$($verParts[0])$($verParts[1])"
    $pthLines = @("$pyTag.zip", '.', 'Lib\site-packages', '..\agent', '', 'import site')

    $models = Join-Path $Ghost 'models'
    $py = Join-Path $Ghost 'python'
    $site = Join-Path $py 'Lib\site-packages'
    $browserDir = Join-Path $Ghost "browsers\chromium-$($Chromium.Revision)"
    $stamps = @{
        python   = "python $PythonVersion $archKey`n$($pyPin.Name) $($pyPin.Sha256)`npth $($pthLines -join ' | ')"
        packages = "packages $archKey`n$($lockLines -join "`n")"
        chromium = "chromium $($Chromium.Version) revision $($Chromium.Revision)`n$($crPin.Name) $($crPin.Sha256)`nmarkers INSTALLATION_COMPLETE DEPENDENCIES_VALIDATED"
    }
    function Get-LlamaItems([string]$Name) {
        $items = @()
        foreach ($spec in @($builds[$Name])) { $items += ,(ConvertTo-Pin $spec @("$LlamaBase/$(($spec -split '\s+')[0])")) }
        return ,$items
    }
    function Get-LlamaStamp([string]$Name) {
        $lines = @("llama.cpp $LlamaTag $archKey $Name", 'tools pruned')
        foreach ($i in (Get-LlamaItems $Name)) { $lines += "$($i.Name) $($i.Sha256)" }
        return $lines -join "`n"
    }

    Write-Host "GhostAgent $version build -> $Ghost" -ForegroundColor White
    $modelLabel = if ($SkipModel) { 'none (-SkipModel)' } else { ($chosen -join ', ') }
    Note "llama.cpp $LlamaTag, $archKey parts, model profile: $modelLabel"
    if ($archKey -eq 'arm64') { Warn 'ARM64 parts are untested: there was no ARM64 hardware to try them on. Please report how it goes.' }

    $proxyUrl = Initialize-Proxy

    foreach ($tool in $Curl, $Tar, $Robocopy) {
        if (-not (Test-Path -LiteralPath $tool)) { throw "$tool is missing (it ships with Windows 10 1803 and newer)." }
    }

    # --- free space and file system ---------------------------------------------------------
    New-Item -ItemType Directory -Force -Path $Ghost | Out-Null
    $script:ModelFromDir = $null
    if ($ModelFrom) {
        if (-not (Test-Path -LiteralPath $ModelFrom -PathType Container)) { throw "-ModelFrom: folder not found: $ModelFrom" }
        $script:ModelFromDir = (Resolve-Path -LiteralPath $ModelFrom).ProviderPath
    }
    $candidates = @()
    if ($script:ModelFromDir) { $candidates += $script:ModelFromDir }
    # An older copy under the previous name (HoloAgent) next to this folder: on the same drive its
    # model is hard-linked, so nothing is downloaded or copied.
    foreach ($d in (Join-Path $Root 'HoloAgent\models'), (Join-Path (Split-Path -Parent $Root) 'HoloAgent\models')) {
        if ((Test-Path -LiteralPath $d -PathType Container) -and $candidates -notcontains $d) { $candidates += $d }
    }

    $needTarget = [int64]0
    $needCache = [int64]0
    $toFetch = @()
    function Add-CacheNeed($Item) {
        $f = Join-Path $Cache $Item.Name
        if ((Test-Path -LiteralPath $f) -and [int64](Get-Item -LiteralPath $f).Length -eq $Item.Size) { return [int64]0 }
        $have = [int64]0
        if (Test-Path -LiteralPath "$f.part") { $have = [int64](Get-Item -LiteralPath "$f.part").Length }
        if ($have -ge $Item.Size) { return [int64]0 }
        return [int64]($Item.Size - $have)
    }
    foreach ($name in $builds.Keys) {
        if (Test-Stamp (Join-Path $Ghost "bin\$name") (Get-LlamaStamp $name)) { continue }
        $needTarget += [int64]$LlamaInstallMB[$name] * 1000000
        foreach ($i in (Get-LlamaItems $name)) { $needCache += Add-CacheNeed $i; $toFetch += @{ Item = $i; Dest = "bin\$name" } }
    }
    if (-not (Test-Stamp $py $stamps.python)) {
        $needTarget += 30e6; $needCache += Add-CacheNeed $pyPin; $toFetch += @{ Item = $pyPin; Dest = 'python' }
    }
    if (-not (Test-Stamp $site $stamps.packages)) {
        $needTarget += 130e6; $needCache += 50e6; $toFetch += @{ Item = $pipPin; Dest = 'pip (runs from the cache, not installed)' }
    }
    if (-not $SkipChromium -and -not (Test-Stamp $browserDir $stamps.chromium)) {
        $needTarget += 480e6; $needCache += Add-CacheNeed $crPin; $toFetch += @{ Item = $crPin; Dest = "browsers\chromium-$($Chromium.Revision)" }
    }
    if (-not $SkipModel) {
        foreach ($m in $modelPins) {
            $dest = Join-Path $models $m.Name
            if (Test-Verified $dest $m) { continue }
            if ((Test-Path -LiteralPath $dest) -and [int64](Get-Item -LiteralPath $dest).Length -eq $m.Size) { continue }
            $linkable = $false
            foreach ($c in $candidates) {
                $src = Join-Path $c $m.Name
                if ((Test-Path -LiteralPath $src) -and [int64](Get-Item -LiteralPath $src).Length -eq $m.Size -and (Test-SameVolume $src $Ghost)) { $linkable = $true }
            }
            if ($linkable) { continue }
            $have = [int64]0
            if (Test-Path -LiteralPath "$dest.part") { $have = [int64](Get-Item -LiteralPath "$dest.part").Length }
            $needTarget += $m.Size - $have
            $toFetch += @{ Item = $m; Dest = (Get-ModelDestNote $models) }
        }
    }

    $target = Get-VolumeInfo $Ghost
    if (-not $SkipModel) {
        $largest = [int64]0
        foreach ($m in $modelPins) { if ($m.Size -gt $largest) { $largest = $m.Size } }
        $what = if ($modelPins.Count -gt 2) { 'The largest model file' } else { 'The model' }
        Assert-LargeFileSupport $target $largest $what
    }

    if ($DryRun) {
        Step 'Dry run: nothing is downloaded or changed'
        if (-not $toFetch.Count) { Note 'everything is already in place' }
        foreach ($t in $toFetch) {
            Write-Host ("   {0}  ({1})" -f $t.Item.Name, (Format-Size $t.Item.Size))
            Note "url:    $($t.Item.Urls -join '  or  ')"
            Note "sha256: $($t.Item.Sha256)"
            Note "into:   $($t.Dest)"
        }
        Write-Host "   playwright $pwVersion and its dependencies: pip installs tools\requirements.lock (wheels pinned by SHA-256)"
        if ($target) { Note "target drive $($target.Root) ($($target.FileSystem)), free $(Format-Size $target.FreeBytes)" }
        Note "about $(Format-Size $needTarget) to add on the target drive, $(Format-Size $needCache) of archives to download into $Cache"
        return 0
    }

    New-Item -ItemType Directory -Force -Path $Cache | Out-Null
    $cacheVol = Get-VolumeInfo $Cache
    if ($target) {
        $need = $needTarget
        if ($cacheVol -and $cacheVol.Id -eq $target.Id) { $need += $needCache }
        Note "free space on $($target.Root): $(Format-Size $target.FreeBytes) (needs about $(Format-Size $need))"
        if ($target.FreeBytes -lt $need) {
            throw "Not enough free space on $($target.Root): $(Format-Size $target.FreeBytes) free, about $(Format-Size $need) needed."
        }
    }
    if ($cacheVol -and (-not $target -or $cacheVol.Id -ne $target.Id) -and $cacheVol.FreeBytes -lt $needCache + 1e9) {
        throw ("Not enough free space on $($cacheVol.Root) for the downloads ($Cache): $(Format-Size $cacheVol.FreeBytes) free, " +
               "about $(Format-Size ($needCache + 1e9)) needed. Free some space, or point TEMP at a drive with more room.")
    }

    # --- Python + Playwright ----------------------------------------------------------------
    Step "Portable Python $PythonVersion with Playwright $pwVersion"
    if (Test-Stamp $py $stamps.python) { Note 'python: already in place' }
    else {
        $pth = "$pyTag._pth"
        Install-Component $py @($pyPin) $stamps.python {
            param($dir)
            # Search path of the embedded Python: its own files, Lib\site-packages and the agent folder.
            if (-not (Test-Path -LiteralPath (Join-Path $dir $pth))) { throw "The Python package has no $pth; its layout changed." }
            Set-Content -LiteralPath (Join-Path $dir $pth) -Value $pthLines -Encoding Ascii
        }
        Note 'python: ready'
    }
    $pyExe = Join-Path $py 'python.exe'
    if (Test-Stamp $site $stamps.packages) { Note "Playwright ${pwVersion}: already installed" }
    else {
        # The embedded Python has no pip. Run pip straight from its (pinned) wheel, without installing it.
        $whl = Join-Path $Cache $pipPin.Name
        Get-Download $pipPin $whl
        $pipDir = Join-Path $Cache 'pip'
        Remove-Dir $pipDir
        Expand-To $whl $pipDir
        # A clean folder, so no package from an earlier version is left behind.
        Remove-Dir $site
        New-Item -ItemType Directory -Force -Path $site | Out-Null
        $runPip = 'import sys; sys.path.insert(0, sys.argv[1]); from pip._internal.cli.main import main; sys.exit(main(sys.argv[2:]))'
        # --isolated ignores pip settings and PIP_* variables on this PC, --require-hashes and --no-deps
        # install exactly the wheels in requirements.lock, and --only-binary never builds from source.
        $pipArgs = @('-I', '-c', $runPip, $pipDir, '--isolated', 'install', '--disable-pip-version-check',
                     '--no-warn-script-location', '--no-cache-dir', '--require-hashes', '--no-deps',
                     '--only-binary=:all:', '--upgrade', '--index-url', 'https://pypi.org/simple',
                     '--target', $site, '-r', $Lock)
        if ($proxyUrl) { $pipArgs += @('--proxy', $proxyUrl) }
        & $pyExe @pipArgs | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'Installing Playwright failed (see the pip messages above).' }
        Write-Stamp $site $stamps.packages
        Note "Playwright ${pwVersion}: installed"
    }

    Start-KeepAwake $py

    # --- llama.cpp --------------------------------------------------------------------------
    Step "llama.cpp $LlamaTag ($(@($builds.Keys) -join ', '))"
    foreach ($name in $builds.Keys) {
        $dir = Join-Path $Ghost "bin\$name"
        $stamp = Get-LlamaStamp $name
        if (Test-Stamp $dir $stamp) { Note "bin\${name}: already in place"; continue }
        if (Test-Path -LiteralPath $dir) { Note "bin\${name}: replacing it (different or unfinished version)" }
        Install-Component $dir (Get-LlamaItems $name) $stamp { param($d) Remove-UnusedTools $d }
        Note "bin\${name}: ready"
    }
    Set-Content -LiteralPath (Join-Path $Ghost 'bin\VERSION.txt') -Value $LlamaTag -Encoding Ascii

    # --- Chromium ---------------------------------------------------------------------------
    if ($SkipChromium) {
        Step 'Portable Chromium: skipped (-SkipChromium)'
        Note 'the "bundled" browser mode will not work; "chrome" and "edge" modes still do'
    } else {
        Step "Portable Chromium $($Chromium.Version) for Playwright"
        # The pinned Chromium must be the one this Playwright drives.
        $bj = Join-Path $site 'playwright\driver\package\browsers.json'
        $expected = @((Get-Content -LiteralPath $bj -Raw | ConvertFrom-Json).browsers | Where-Object { $_.name -eq 'chromium' })[0]
        if ([string]$expected.revision -ne $Chromium.Revision -or [string]$expected.browserVersion -ne $Chromium.Version) {
            throw ("Playwright $pwVersion expects Chromium $($expected.browserVersion) (revision $($expected.revision)), but the build pins " +
                   "$($Chromium.Version) (revision $($Chromium.Revision)). Update the Chromium pin in tools\Build-GhostAgent.ps1.")
        }
        if (Test-Stamp $browserDir $stamps.chromium) { Note "chromium-$($Chromium.Revision): already in place" }
        else {
            try {
                Install-Component $browserDir @($crPin) $stamps.chromium {
                    param($dir)
                    # The markers Playwright's installer leaves behind; without them it reinstalls.
                    foreach ($m in 'INSTALLATION_COMPLETE', 'DEPENDENCIES_VALIDATED') { Set-Content -LiteralPath (Join-Path $dir $m) -Value '' -Encoding Ascii }
                }
            } catch {
                if ($_.Exception.Message -match 'differs from the pinned version') { throw }
                Warn "the direct Chromium download failed ($($_.Exception.Message)); trying Playwright's installer"
                $env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $Ghost 'browsers'
                & $pyExe -I -m playwright install chromium --no-shell | Out-Host
                if ($LASTEXITCODE -ne 0) { throw 'Installing Chromium failed.' }
                $cw = Join-Path $browserDir 'chrome-win64'
                foreach ($pair in @(@('chrome.exe', $Chromium.ChromeExe), @('chrome.dll', $Chromium.ChromeDll))) {
                    $f = Join-Path $cw $pair[0]
                    if (-not (Test-Path -LiteralPath $f) -or (Get-Sha256 $f) -ne $pair[1]) {
                        Remove-Dir $browserDir
                        throw "The Chromium that Playwright installed does not match the pinned $($pair[0]); it was removed."
                    }
                }
                Write-Stamp $browserDir $stamps.chromium
            }
            Note "chromium-$($Chromium.Revision): ready"
        }
        # Older Chromium revisions, and tools the agent never uses (ffmpeg records video; winldd is
        # only needed by Playwright's dependency check for a folder layout this Chromium does not use).
        foreach ($d in @(Get-ChildItem -LiteralPath (Join-Path $Ghost 'browsers') -Directory -ErrorAction SilentlyContinue)) {
            if ($d.Name -match '^(chromium|chromium_headless_shell|ffmpeg|winldd)-\d+(\.tmp|\.old)?$' -and $d.FullName -ne $browserDir) {
                Note "removing browsers\$($d.Name) (not used by this version)"
                Remove-Dir $d.FullName
            }
        }
    }

    # --- model ------------------------------------------------------------------------------
    if ($SkipModel) {
        Step 'Model: skipped (-SkipModel)'
    } else {
        $total = [int64]0
        foreach ($m in $modelPins) { $total += $m.Size }
        Step "Model: $($chosen -join ', ') ($(Format-Size $total))"
        New-Item -ItemType Directory -Force -Path $models | Out-Null
        foreach ($m in $modelPins) { Install-ModelFile $m $models $candidates }
        if ($chosen -notcontains [string]$profilesDoc.default) {
            Note "To use it, set ""model_profile"": ""$($chosen[0])"" in GhostAgent\settings.json."
        }
    }

    # Model files that no profile uses any more (for example after a model update).
    $stale = @(Get-StaleModelFiles $models $listedModels)
    if ($stale.Count) {
        Step 'Model files no profile uses'
        foreach ($s in $stale) { Note "$($s.Name) ($(Format-Size $s.Length))" }
        if ($OfferCleanup) {
            $answer = Read-Host '   Delete them? [y/N]'
            if ($answer -match '^\s*y') {
                foreach ($s in $stale) { Remove-Item -LiteralPath $s.FullName -Force }
                Note 'deleted'
            } else { Note 'kept' }
        } else { Note 'They can be deleted by hand, or run "Update GhostAgent.cmd" to be asked.' }
    }

    # --- smoke test and summary -------------------------------------------------------------
    $failures = Invoke-SmokeTest @($builds.Keys) (-not $SkipChromium -and (Test-Path -LiteralPath $browserDir))

    Stop-KeepAwake
    if (-not $KeepDownloads) { Remove-Dir $Cache }
    $size = (Get-ChildItem -LiteralPath $Ghost -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    Write-Host ''
    if ($failures.Count) {
        Write-Host 'Everything was downloaded, but the smoke test found problems:' -ForegroundColor Red
        foreach ($f in $failures) { Write-Host "  - $f" -ForegroundColor Red }
        Write-Host ('If Windows blocked a program (App Control, Smart App Control or antivirus), the agent cannot use it on this PC. ' +
                    'Otherwise run the build again; it replaces only what is missing or different.')
        return 1
    }
    Write-Host ("Done. GhostAgent $version is {0} (llama.cpp $LlamaTag)." -f (Format-Size $size)) -ForegroundColor Green
    if ($env:LOCALAPPDATA -and (Test-Path -LiteralPath (Join-Path $env:LOCALAPPDATA 'GhostAgent\models\.verified'))) {
        Note "A checked copy of the model stays in $env:LOCALAPPDATA\GhostAgent\models, so the agent loads it fast on this PC (copy_model_to_pc). If you will not run the agent on this PC, delete that folder."
    }
    Write-Host 'Start the agent with "Start GhostAgent.cmd". README.txt explains the first-time Chrome setting.'
    return 0
}

# --- main -------------------------------------------------------------------------------------

# Loaded with ". .\Build-GhostAgent.ps1" (by tests): define the functions above, run nothing.
if ($MyInvocation.InvocationName -eq '.') { return }

# Only the pinned sources are used: no download-host overrides for Playwright and no pip settings
# from this PC's environment.
foreach ($n in 'PLAYWRIGHT_DOWNLOAD_HOST', 'PLAYWRIGHT_CHROMIUM_DOWNLOAD_HOST', 'PLAYWRIGHT_BROWSERS_PATH') {
    Remove-Item "Env:$n" -ErrorAction SilentlyContinue
}
Get-ChildItem Env: | Where-Object { $_.Name -like 'PIP_*' } | ForEach-Object { Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue }
$script:CurlProxyArgs = @()

$rc = 0
try {
    if ($Update) { Invoke-Update }
    else { $rc = Invoke-Build }
} catch {
    Write-Host ''
    Write-Host $_.Exception.Message -ForegroundColor Red
    Note "(at line $($_.InvocationInfo.ScriptLineNumber) of tools\Build-GhostAgent.ps1)"
    $rc = 1
} finally {
    Stop-KeepAwake
}
exit $rc
