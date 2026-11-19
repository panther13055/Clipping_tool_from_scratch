@echo off
cd /d "%~dp0"
start "ClipForge Local AI Engine" "%ComSpec%" /k "cd /d ""%~dp0local-ai"" && START_WINDOWS.bat"
start "ClipForge Studio" "%ComSpec%" /k "cd /d ""%~dp0"" && START_APP_WINDOWS.bat"
timeout /t 8 /nobreak >nul
start "" "http://localhost:3000"
