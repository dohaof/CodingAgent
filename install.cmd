@echo off
rem Double-clickable setup for cagent and its desktop client.
rem
rem Deliberately thin: all it does is find a Python and hand over to
rem tools\install.py, because the real work needs error handling that batch
rem cannot express. The final pause is what makes the output readable when this
rem was double-clicked rather than run from a prompt.
setlocal
cd /d "%~dp0"

set "PY="
for %%C in (py python python3) do (
  if not defined PY (
    where %%C >nul 2>nul && set "PY=%%C"
  )
)

if not defined PY (
  echo.
  echo   Python was not found on PATH.
  echo   Install Python 3.11 or newer from https://www.python.org/downloads/
  echo   ^(tick "Add python.exe to PATH" in the installer^) and run this again.
  echo.
  pause
  exit /b 1
)

%PY% "tools\install.py" %*
set "CODE=%ERRORLEVEL%"

echo.
pause
exit /b %CODE%
