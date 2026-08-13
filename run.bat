@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"

if /i "%~1"=="--self-test" goto :self_test

set "STUDIO_BOOTSTRAP=%LOCALAPPDATA%\AI-Video-Studio\bootstrap"
set "BUNDLED_PYTHON=%STUDIO_BOOTSTRAP%\python\python.exe"
set "BUNDLED_FFMPEG=%~dp0bootstrap\ffmpeg\bin"
if not exist "%BUNDLED_FFMPEG%\ffmpeg.exe" set "BUNDLED_FFMPEG=%STUDIO_BOOTSTRAP%\ffmpeg\bin"
if exist "%BUNDLED_FFMPEG%\ffmpeg.exe" set "PATH=%BUNDLED_FFMPEG%;%PATH%"

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [ERROR] FFmpeg was not found in PATH.
  goto :failed
)

where ffprobe >nul 2>nul
if errorlevel 1 (
  echo [ERROR] FFprobe was not found in PATH.
  goto :failed
)

if not exist ".venv\Scripts\python.exe" (
  echo [SETUP] Creating the Python 3.12 environment...
  if not exist "%BUNDLED_PYTHON%" if exist "%~dp0bootstrap\python-3.12.10-amd64.exe" (
    echo [SETUP] Installing the bundled Python runtime for the current user...
    if not exist "%STUDIO_BOOTSTRAP%" mkdir "%STUDIO_BOOTSTRAP%"
    "%~dp0bootstrap\python-3.12.10-amd64.exe" /quiet InstallAllUsers=0 TargetDir="%STUDIO_BOOTSTRAP%\python" Include_pip=1 Include_launcher=0 PrependPath=0 Shortcuts=0
    if errorlevel 1 goto :failed
  )
  if exist "%BUNDLED_PYTHON%" (
    "%BUNDLED_PYTHON%" -m venv .venv
  ) else (
    py -3.12 -m venv .venv
  )
  if errorlevel 1 goto :failed
)

echo [SETUP] Checking pinned dependencies...
if exist "%~dp0wheelhouse" (
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --no-index --find-links "%~dp0wheelhouse" -r requirements.txt
) else (
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
)
if errorlevel 1 goto :failed

echo [STUDIO] Opening the local AI-Video Studio console...
".venv\Scripts\python.exe" -m app studio --open-output
if errorlevel 1 goto :failed

echo.
echo [DONE] AI-Video Studio and the requested video task finished.
pause
exit /b 0

:self_test
echo [SELF-TEST] Batch parser is working.
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [SELF-TEST] FAIL: FFmpeg was not found.
  exit /b 1
)
where ffprobe >nul 2>nul
if errorlevel 1 (
  echo [SELF-TEST] FAIL: FFprobe was not found.
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo [SELF-TEST] FAIL: .venv\Scripts\python.exe was not found.
  exit /b 1
)
".venv\Scripts\python.exe" -c "import app; print('[SELF-TEST] Python and app import are working.')"
if errorlevel 1 exit /b 1
echo [SELF-TEST] PASS
exit /b 0

:failed
echo.
echo [FAILED] Read the error above. Fix it, then run this same run.bat again.
pause
exit /b 1
