@echo off
setlocal
cd /d "%~dp0.."
chcp 65001 >nul
set "PYTHONUTF8=1"

if not exist ".venv\Scripts\python.exe" (
  echo [SETUP] Creating the Python 3.12 environment...
  py -3.12 -m venv .venv
  if errorlevel 1 goto :failed
)

echo [SETUP] Checking pinned dependencies...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :failed

echo [VOICES] Generating the six voice samples...
".venv\Scripts\python.exe" -m app voices audition --project roman-urine-laundry --open-output
if errorlevel 1 goto :failed

echo.
echo [DONE] Voice samples were generated.
pause
exit /b 0

:failed
echo.
echo [FAILED] Read the error above.
pause
exit /b 1
