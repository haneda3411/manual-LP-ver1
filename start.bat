@echo off
cd /d "%~dp0backend"
start "マニュアルツール" python app.py
timeout /t 3 /nobreak > nul
start "" "%~dp0tool.html"
