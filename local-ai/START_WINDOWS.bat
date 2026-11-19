@echo off
cd /d "%~dp0"
py -3 -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn worker:app --host 127.0.0.1 --port 8765
pause
