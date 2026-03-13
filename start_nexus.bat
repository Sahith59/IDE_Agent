@echo off
echo =============================================
echo   IDE Expert Agent (Nexus) - Windows
echo =============================================

:: Detect SSD drive letter automatically
set SSD=%~d0
echo Detected Project Root at: %SSD%\IDE_Expert_Project

:: Important Directories
set VENV_DIR=%SSD%\IDE_Expert_Project\win\venv
set APP_DIR=%SSD%\IDE_Expert_Project\app

:: Point to your existing models directory on the parent SSD layout
set OLLAMA_MODELS=%SSD%\Ollama_Models

:: Set up the local Ollama binaries inside our project
set OLLAMA_HOME=%SSD%\IDE_Expert_Project\ollama\win

:: 1. Start Ollama server in background
echo Starting local Ollama server...
if exist "%OLLAMA_HOME%\ollama.exe" (
    start "Ollama" /min "%OLLAMA_HOME%\ollama.exe" serve
) else (
    echo Warning: No local Ollama binary found at %OLLAMA_HOME%\ollama.exe.
    echo Assuming Ollama is running globally on the host Windows machine.
)
timeout /t 3 /nobreak > nul

:: 2. Activate Python Virtual Environment
echo Starting Python Environment...
call "%VENV_DIR%\Scripts\activate.bat"

:: 3. Launch the Chat CLI
echo Launching Nexus...
python "%APP_DIR%\cli.py"

:: --- Cleanup after CLI exits ---
echo Shutting down...
taskkill /f /im ollama.exe >nul 2>&1
echo Ollama server stopped. You can safely unplug your SSD.
pause
