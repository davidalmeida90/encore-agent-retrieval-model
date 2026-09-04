@echo off
REM ============================================================
REM  Encore: rent the GPU, start the app, open the browser.
REM  Double-click this. Everything else is automatic.
REM
REM  Keep this window open while you use the app. It shows the
REM  phases and the running cost.
REM
REM  CLOSING THIS WINDOW STOPS THE GPU and stops billing.
REM  So does Ctrl+C. That is deliberate: a GPU bills whether or
REM  not anyone is watching it.
REM ============================================================
title Encore - starting
cd /d "%~dp0"
backend\.venv\Scripts\python.exe cloud\launch.py %*
echo.
echo ============================================================
echo  GPU stopped, billing stopped.
echo  Run status-encore.bat to confirm nothing is running.
echo ============================================================
pause
