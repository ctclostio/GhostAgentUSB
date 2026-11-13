@echo off
rem Starts GhostAgent from this drive (any drive letter).
rem It opens in its own window so Ctrl+C there only stops the current task.
set "GHOST=%~dp0GhostAgent"
if not exist "%GHOST%\python\python.exe" (
  echo Cannot find "%GHOST%\python\python.exe".
  echo Keep this file next to the GhostAgent folder, and run "Build GhostAgent.cmd" once first.
  pause
  exit /b 1
)
start "GhostAgent" /d "%GHOST%" "%GHOST%\python\python.exe" "%GHOST%\agent\ghost_agent.py" %*
