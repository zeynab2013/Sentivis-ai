@echo off
setlocal
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo Starting Sentivis AI from %CD%
python -u -m app.main
set EXITCODE=%ERRORLEVEL%
if not %EXITCODE%==0 (
  echo.
  echo Sentivis exited with code %EXITCODE%. Check logs\application.log
)
exit /b %EXITCODE%
