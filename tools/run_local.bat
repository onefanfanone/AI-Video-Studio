@echo off
setlocal
cd /d "%~dp0.."
chcp 65001 >nul
set "PYTHONUTF8=1"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] The main environment is missing. Run the root run.bat first.
  goto :failed
)

echo [LOCAL] Building with raw assets only. No visual API or ComfyUI call will be made.
".venv\Scripts\python.exe" -m app build --project roman-urine-laundry --visual-mode local --open-output
if errorlevel 1 goto :failed

echo.
echo [DONE] Local asset build completed.
pause
exit /b 0

:failed
echo.
echo [FAILED] Read the error above.
pause
exit /b 1
