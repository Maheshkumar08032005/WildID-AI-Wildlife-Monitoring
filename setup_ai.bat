@echo off
title WildID - Install Local AI
cd /d "%~dp0"
if not exist venv (
  echo Creating virtual environment...
  python -m venv venv
)
call venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers==4.44.2 safetensors sentencepiece
echo.
echo AI packages installed. CLIP model weights download on first server start.
echo This requires internet once. Later inference runs locally.
pause
