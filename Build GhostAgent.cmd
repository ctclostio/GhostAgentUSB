@echo off
setlocal
rem Downloads everything the agent needs into the GhostAgent folder next to this file:
rem the Holo4 model (about 22 GB), llama.cpp, portable Python with Playwright and a portable Chromium.
rem Every part is pinned to a version and checked against its SHA-256 sum.
rem Safe to run again: parts already in place are skipped and an interrupted download resumes.
rem Options:  -Profile accurate                        which model: fast (default), fast-q8, accurate or all
rem           -ModelFrom "D:\old\GhostAgent\models"   take the model from an older copy instead of downloading
rem           -SkipModel                               everything except the model
rem           -SkipChromium                            no portable Chromium (bundled browser mode)
rem           -LlamaTag b12359                         llama.cpp v1.0.0 instead of b11669 (run again to switch back)
rem           -KeepDownloads                           keep the archives in %TEMP%\GhostAgent-build
rem           -Proxy http://host:port                  download through this proxy
rem           -DryRun                                  only list what would be downloaded
if not exist "%~dp0tools\Build-GhostAgent.ps1" (
  echo Cannot find tools\Build-GhostAgent.ps1 next to this file.
  echo Extract the whole ZIP first ^(right-click it, Extract All^), then run "Build GhostAgent.cmd"
  echo from the extracted folder.
  echo.
  pause
  exit /b 1
)
rem A PowerShell 7 module path breaks Windows PowerShell; an empty one makes it use its own default.
set "PSModulePath="
rem Files extracted from a downloaded ZIP carry the download mark, so SmartScreen and Smart App
rem Control warn about or block the .cmd and .ps1 files. Clear it once for the whole folder.
set "GHOSTAGENT_ROOT=%~dp0."
powershell.exe -NoProfile -Command "Get-ChildItem -LiteralPath $env:GHOSTAGENT_ROOT -Recurse -File -ErrorAction SilentlyContinue | Unblock-File -ErrorAction SilentlyContinue"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\Build-GhostAgent.ps1" %*
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo The build stopped (see the message above). Once that is sorted out, run it again: it continues where it left off.
pause
exit /b %RC%
