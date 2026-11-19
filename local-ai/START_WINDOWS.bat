@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python 3.11 is required. Install Python from https://www.python.org/downloads/
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe (
  py -3.11 -m venv .venv
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
if errorlevel 1 (
  pause
  exit /b 1
)
pip install -r requirements.txt
if errorlevel 1 (
  pause
  exit /b 1
)
echo ClipForge Local AI Engine is starting at http://127.0.0.1:8765
python -m uvicorn worker:app --host 127.0.0.1 --port 8765
pause
