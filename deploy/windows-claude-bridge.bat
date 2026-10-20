@echo off
rem Runs the Claude file bridge for the Robinhood board; restarts it if it ever exits.
rem Double-click it, or add it to Task Scheduler ("At log on"). It writes data\claude_inbox.json every 30 seconds.
cd /d "%~dp0.."
:loop
python claude_bridge.py --config config.robinhood.json loop
echo Bridge stopped, restarting in 10 seconds (close this window to stop for good)...
timeout /t 10 /nobreak >nul
goto loop
