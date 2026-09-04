@echo off
REM  What is running, and what it has cost so far.
title Encore - status
cd /d "%~dp0"
backend\.venv\Scripts\python.exe cloud\serve_on_runpod.py status
pause
