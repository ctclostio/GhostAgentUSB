@echo off
rem Downloads everything the agent needs into the HoloAgent folder next to this file (about 23 GB):
rem the Holo4 model, llama.cpp, portable Python with Playwright and a portable Chromium.
rem Safe to run again: finished parts are skipped and an interrupted model download resumes.
rem Options:  -ModelFrom "D:\old\HoloAgent\models"   copy the model from an older copy instead of downloading
rem           -SkipModel                              everything except the model
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\Build-USB.ps1" %*
echo.
pause
