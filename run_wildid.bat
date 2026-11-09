@echo off
title WildID - AI-Assisted Wildlife Monitoring System
cd /d "%~dp0"
if not exist venv (
  python -m venv venv
)
call venv\Scripts\activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
