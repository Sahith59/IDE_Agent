@echo off
echo =============================================
echo   Stopping IDE Expert Agent (Nexus) - Windows
echo =============================================

taskkill /f /im ollama.exe >nul 2>&1
echo Ollama server stopped.

taskkill /f /im python.exe >nul 2>&1
echo Nexus Chat closed.

echo All processes stopped. Safe to unplug SSD.
pause
