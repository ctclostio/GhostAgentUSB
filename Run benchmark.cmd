@echo off
rem Runs the 12 local benchmark tasks once with the bundled browser and prints the score.
rem Extra options go to the agent, e.g.: "Run benchmark.cmd" --settings-override my-settings.json
rem Results: GhostAgent\logs\bench-*-results.jsonl and bench-*-score.json
set "GHOST=%~dp0GhostAgent"
if not exist "%GHOST%\python\python.exe" (
  echo Cannot find "%GHOST%\python\python.exe".
  echo Keep this file next to the GhostAgent folder, and run "Build GhostAgent.cmd" once first.
  pause
  exit /b 1
)
"%GHOST%\python\python.exe" -I "%~dp0tools\bench\run_suite.py" %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
