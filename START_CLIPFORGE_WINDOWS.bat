@echo off
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo Node.js is required. Install the current LTS release from https://nodejs.org/
  pause
  exit /b 1
)
if not exist node_modules (
  echo Installing ClipForge Studio dependencies...
  call npm install
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
start "ClipForge Local AI Engine" "%ComSpec%" /k "cd /d ""%~dp0local-ai"" && START_WINDOWS.bat"
start "ClipForge Studio" "%ComSpec%" /k "cd /d ""%~dp0"" && npm run dev"
timeout /t 6 /nobreak >nul
start "" "http://localhost:3000"
