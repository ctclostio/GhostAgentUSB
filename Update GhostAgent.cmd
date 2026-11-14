@echo off
setlocal
rem Updates GhostAgent: gets the newest files (git pull when this folder is a git clone, otherwise
rem the latest ZIP from GitHub), copies them over this folder and then runs the build, which only
rem replaces downloaded parts whose pinned versions changed.
rem Kept as they are: GhostAgent\settings.json, the browser profiles (sign-ins), downloads and logs.
rem Options are passed on to the build, for example  -Profile accurate  or  -LlamaTag b12359.
rem   -UpdateFrom "D:\Downloads\GhostAgentUSB-main.zip"   update from a ZIP or folder you already have
if /i "%~1"=="--from-temp" goto :run
if not exist "%~dp0tools\Build-GhostAgent.ps1" (
  echo Cannot find tools\Build-GhostAgent.ps1 next to this file.
  echo Extract the whole ZIP first, then run "Update GhostAgent.cmd" from the extracted folder.
  echo.
  pause
  exit /b 1
)
rem The update replaces this very file, and cmd.exe reads a batch file bit by bit while running it.
rem So a copy in %TEMP% does the work. This block is read in full before it runs, so nothing below
rem is read from the replaced file.
set "GHOSTAGENT_ROOT=%~dp0."
rem No quotes around this assignment: the arguments keep their own quotes, so a quoted path that
rem contains & or ( ) stays one literal argument.
set GHOSTAGENT_ARGS=%*
(
  type "%~f0" > "%TEMP%\GhostAgent-Update.cmd"
  call "%TEMP%\GhostAgent-Update.cmd" --from-temp
  exit /b
)

:run
rem A PowerShell 7 module path breaks Windows PowerShell; an empty one makes it use its own default.
set "PSModulePath="
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%GHOSTAGENT_ROOT%\tools\Build-GhostAgent.ps1" -Update %GHOSTAGENT_ARGS%
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto :done
echo.
rem The build that runs now is the new one that the update just put in place.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%GHOSTAGENT_ROOT%\tools\Build-GhostAgent.ps1" -OfferCleanup %GHOSTAGENT_ARGS%
set "RC=%ERRORLEVEL%"
:done
echo.
if not "%RC%"=="0" echo The update stopped (see the message above). Once that is sorted out, run it again: it continues where it left off.
pause
exit /b %RC%
