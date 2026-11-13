@echo off
setlocal
rem Downloads everything the agent needs into the GhostAgent folder next to this file (about 23 GB):
rem the Holo4 model, llama.cpp, portable Python with Playwright and a portable Chromium.
rem Safe to run again: finished parts are skipped and an interrupted model download resumes.
rem Options:  -ModelFrom "D:\old\GhostAgent\models"   copy the model from an older copy instead of downloading
rem           -SkipModel                               everything except the model
rem A PowerShell 7 module path breaks Windows PowerShell; an empty one makes it use its own default.
set "PSModulePath="
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\Build-GhostAgent.ps1" %*
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo The build stopped with an error (see above). Run it again to continue where it left off.
pause
exit /b %RC%
