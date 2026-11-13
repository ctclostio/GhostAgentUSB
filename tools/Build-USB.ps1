<#
.SYNOPSIS
  Fills HoloAgent\ with everything the agent needs (about 23 GB): the Holo4 model, llama.cpp,
  portable Python with Playwright, and a portable Chromium. Run it through "Build USB.cmd".

.DESCRIPTION
  Every part comes from its official source at a pinned version and is checked against a SHA-256
  sum. Parts that are already in place are skipped, so it is safe to run again after an
  interrupted download (the model download resumes where it stopped).

.PARAMETER ModelFrom
  A folder that already holds Holo4-35B-A3B-Q4_K_M.gguf and Holo4-35B-A3B-mmproj-f16.gguf (for
  example an older copy of the stick). They are copied from there instead of downloaded.

.PARAMETER SkipModel
  Set up everything except the model.

.PARAMETER KeepDownloads
  Keep the downloaded archives in %TEMP%\GhostAgentUSB-build instead of deleting them.
#>
[CmdletBinding()]
param(
    [string]$ModelFrom,
    [switch]$SkipModel,
    [switch]$KeepDownloads
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Root  = Split-Path -Parent $PSScriptRoot
$Holo  = Join-Path $Root 'HoloAgent'
$Cache = Join-Path $env:TEMP 'GhostAgentUSB-build'

# --- pinned versions --------------------------------------------------------------------------

$LlamaTag   = 'b11669'
$LlamaBase  = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaTag"
$PythonUrl  = 'https://www.python.org/ftp/python/3.14.8/python-3.14.8-embed-amd64.zip'
$PipUrl     = 'https://files.pythonhosted.org/packages/07/25/ee8c38ae72d66b844ce8eb9711ba905dba21ca1a8a0c960057c41cd23337/pip-26.3-py3-none-any.whl'
$Playwright = 'playwright==1.63.0'
$ModelBase  = 'https://huggingface.co/Hcompany/Holo4-35B-A3B-GGUF/resolve/main'

# llama.cpp builds: folder under bin\ -> archives unpacked into it
$Builds = [ordered]@{
    cuda13 = @("llama-$LlamaTag-bin-win-cuda-13.4-x64.zip", 'cudart-llama-bin-win-cuda-13.4-x64.zip')
    cuda12 = @("llama-$LlamaTag-bin-win-cuda-12.4-x64.zip", 'cudart-llama-bin-win-cuda-12.4-x64.zip')
    vulkan = @("llama-$LlamaTag-bin-win-vulkan-x64.zip")
    cpu    = @("llama-$LlamaTag-bin-win-cpu-x64.zip")
}

# Model files: name on Hugging Face -> name in HoloAgent\models
$ModelFiles = [ordered]@{
    'q4_k_m.gguf'     = 'Holo4-35B-A3B-Q4_K_M.gguf'
    'mmproj.f16.gguf' = 'Holo4-35B-A3B-mmproj-f16.gguf'
}

# SHA-256 of every download
$Sha = @{
    "llama-$LlamaTag-bin-win-cuda-13.4-x64.zip" = '3b92d5c064f5032e603e531d3ec8961f3bd6613e5dfad047399a92dec8e95dbc'
    'cudart-llama-bin-win-cuda-13.4-x64.zip'    = '738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668'
    "llama-$LlamaTag-bin-win-cuda-12.4-x64.zip" = 'c2da1f73ed176ae093ac5aceb978a7b167a16113f6d74c6789eeaee2135cb8dc'
    'cudart-llama-bin-win-cuda-12.4-x64.zip'    = '8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6'
    "llama-$LlamaTag-bin-win-vulkan-x64.zip"    = '2c07574e7bde71321cb785282b20cc821e951132149eeceeea32ddebdbf0598e'
    "llama-$LlamaTag-bin-win-cpu-x64.zip"       = '37b3efcf962847ca869745fd4ccf7f5e7e8dacfeff9fa3523b4ca8cdc66d632b'
    'python-3.14.8-embed-amd64.zip'             = 'a93abe456ab01bd96d7a085b3cdb6566b3063f4241360d114142fbdb07f0a310'
    'pip-26.3-py3-none-any.whl'                 = 'f3b60b20ffaadd4415ed7957212a38ac63fa4c5fa375bd0f4a797777940fad13'
}
# The model sums live next to the model, in HoloAgent\models\SHA256SUMS.txt
foreach ($line in Get-Content (Join-Path $Holo 'models\SHA256SUMS.txt')) {
    if ($line -match '^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$') { $Sha[$Matches[2]] = $Matches[1].ToLower() }
}

# --- helpers ----------------------------------------------------------------------------------

function Step($msg) { Write-Host ''; Write-Host "== $msg" -ForegroundColor Cyan }
function Note($msg) { Write-Host "   $msg" -ForegroundColor DarkGray }

function Get-Sha256($path) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower() }

function Test-Sum($path) {
    $name = Split-Path -Leaf $path
    if (-not $Sha.ContainsKey($name)) { throw "No SHA-256 sum known for $name." }
    return (Test-Path -LiteralPath $path) -and ((Get-Sha256 $path) -eq $Sha[$name])
}

# Downloads $url to $dest (resuming a partial download) and checks its sum.
function Get-Download($url, $dest) {
    if (Test-Sum $dest) { Note "$(Split-Path -Leaf $dest): already downloaded"; return }
    $part = "$dest.part"
    Note "downloading $(Split-Path -Leaf $dest)"
    for ($try = 1; $try -le 2; $try++) {
        & curl.exe -L --fail --retry 5 --retry-delay 5 -C - -o $part $url
        if ($LASTEXITCODE -ne 0) { throw "Download failed (curl exit code $LASTEXITCODE): $url" }
        Move-Item -LiteralPath $part -Destination $dest -Force
        if (Test-Sum $dest) { return }
        # A stale .part from an earlier, different file: start over once.
        Remove-Item -LiteralPath $dest -Force
        Note 'checksum did not match; downloading again from scratch'
    }
    throw "Checksum mismatch for $(Split-Path -Leaf $dest). The download may be corrupt; run again."
}

function Expand-To($archive, $dir) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    & tar.exe -xf $archive -C $dir
    if ($LASTEXITCODE -ne 0) { throw "Could not unpack $archive" }
}

# --- checks -----------------------------------------------------------------------------------

if (-not [Environment]::Is64BitOperatingSystem) { throw 'GhostAgentUSB needs 64-bit Windows.' }
foreach ($tool in 'curl.exe', 'tar.exe') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool is missing (it ships with Windows 10 1803 and newer)." }
}
$needGB = 3 + $(if ($SkipModel) { 0 } else { 21 })
$drive = New-Object System.IO.DriveInfo ([IO.Path]::GetPathRoot($Holo))
$freeGB = [math]::Floor($drive.AvailableFreeSpace / 1GB)
Write-Host "GhostAgentUSB build -> $Holo" -ForegroundColor White
Note "free space on $($drive.Name): $freeGB GB (needs about $needGB GB)"
if ($freeGB -lt $needGB) { throw "Not enough free space on $($drive.Name): $freeGB GB free, about $needGB GB needed." }
New-Item -ItemType Directory -Force -Path $Cache | Out-Null

# --- llama.cpp --------------------------------------------------------------------------------

Step "llama.cpp $LlamaTag (CUDA 13, CUDA 12, Vulkan, CPU)"
foreach ($dir in $Builds.Keys) {
    $target = Join-Path $Holo "bin\$dir"
    if (Test-Path (Join-Path $target 'llama-server.exe')) { Note "bin\${dir}: already in place"; continue }
    foreach ($zip in $Builds[$dir]) {
        $file = Join-Path $Cache $zip
        Get-Download "$LlamaBase/$zip" $file
        Expand-To $file $target
    }
    Note "bin\${dir}: ready"
}

# --- Python + Playwright ----------------------------------------------------------------------

Step 'Portable Python 3.14 with Playwright'
$py = Join-Path $Holo 'python'
if (-not (Test-Path (Join-Path $py 'python.exe'))) {
    $file = Join-Path $Cache 'python-3.14.8-embed-amd64.zip'
    Get-Download $PythonUrl $file
    Expand-To $file $py
}
# Search path of the embedded Python: its own files, Lib\site-packages and the agent folder.
$pth = "python314.zip`r`n.`r`nLib\site-packages`r`n..\agent`r`n`r`nimport site`r`n"
[IO.File]::WriteAllText((Join-Path $py 'python314._pth'), $pth, [Text.Encoding]::ASCII)

$site = Join-Path $py 'Lib\site-packages'
if (Test-Path (Join-Path $site 'playwright-1.63.0.dist-info')) {
    Note 'Playwright 1.63.0: already installed'
} else {
    # The embedded Python has no pip. Run pip straight from its unpacked wheel, without installing it.
    $whl = Join-Path $Cache 'pip-26.3-py3-none-any.whl'
    Get-Download $PipUrl $whl
    $pipDir = Join-Path $Cache 'pip'
    if (Test-Path $pipDir) { Remove-Item -Recurse -Force $pipDir }
    Expand-To $whl $pipDir
    $runPip = 'import sys; sys.path.insert(0, sys.argv[1]); from pip._internal.cli.main import main; sys.exit(main(sys.argv[2:]))'
    & (Join-Path $py 'python.exe') -c $runPip $pipDir install --disable-pip-version-check --no-warn-script-location `
        --only-binary=:all: --target $site $Playwright
    if ($LASTEXITCODE -ne 0) { throw 'Installing Playwright failed.' }
    Note 'Playwright 1.63.0: installed'
}

# --- Chromium ---------------------------------------------------------------------------------

Step 'Portable Chromium for Playwright'
$env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $Holo 'browsers'
& (Join-Path $py 'python.exe') -m playwright install chromium --no-shell
if ($LASTEXITCODE -ne 0) { throw 'Installing Chromium failed.' }
Note 'browsers: ready'

# --- model ------------------------------------------------------------------------------------

if ($SkipModel) {
    Step 'Model: skipped (-SkipModel)'
} else {
    Step 'Holo4-35B-A3B model (about 21 GB)'
    $models = Join-Path $Holo 'models'
    foreach ($src in $ModelFiles.Keys) {
        $name = $ModelFiles[$src]
        $dest = Join-Path $models $name
        if ($ModelFrom) {
            $from = Join-Path $ModelFrom $name
            if (-not (Test-Path -LiteralPath $from)) { throw "Not found: $from" }
            if (-not (Test-Sum $dest)) {
                Note "copying $name from $ModelFrom"
                Copy-Item -LiteralPath $from -Destination $dest -Force
            }
        } else {
            Get-Download "$ModelBase/$src" $dest
        }
        Note "checking $name"
        if (-not (Test-Sum $dest)) { throw "Checksum mismatch for $name." }
        Note "${name}: OK"
    }
}

# --- done -------------------------------------------------------------------------------------

if (-not $KeepDownloads) { Remove-Item -Recurse -Force $Cache -ErrorAction SilentlyContinue }
$total = (Get-ChildItem -LiteralPath $Holo -Recurse -Force -File | Measure-Object Length -Sum).Sum
Write-Host ''
Write-Host ("Done. HoloAgent is {0:N1} GB." -f ($total / 1GB)) -ForegroundColor Green
Write-Host 'Start the agent with "Start Holo Browser Agent.cmd". README.txt explains the first-time Chrome setting.'
