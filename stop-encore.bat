@echo off
REM  Terminate every pod on the account and stop billing.
title Encore - stopping
cd /d "%~dp0"
backend\.venv\Scripts\python.exe cloud\serve_on_runpod.py down
echo.
backend\.venv\Scripts\python.exe cloud\serve_on_runpod.py status
pause
