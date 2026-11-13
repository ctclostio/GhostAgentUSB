@echo off
rem Starts the Holo4 local browser agent from this drive (any drive letter).
rem It opens in its own window so Ctrl+C there only stops the current task.
set "HOLO=%~dp0HoloAgent"
if not exist "%HOLO%\python\python.exe" (
  echo Cannot find "%HOLO%\python\python.exe".
  echo Keep this file next to the HoloAgent folder.
  pause
  exit /b 1
)
start "Holo Browser Agent" /d "%HOLO%" "%HOLO%\python\python.exe" "%HOLO%\agent\holo_agent.py" %*
