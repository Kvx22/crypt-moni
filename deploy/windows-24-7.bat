@echo off
rem Keeps the scanner + dashboard running on Windows; restarts it if it ever exits.
rem Double-click it, or add it to Task Scheduler ("At log on") for 24/7. Dashboard: http://127.0.0.1:8765
cd /d "%~dp0.."
:loop
py run.py up
echo Scanner stopped, restarting in 10 seconds (close this window to stop for good)...
timeout /t 10 /nobreak >nul
goto loop
