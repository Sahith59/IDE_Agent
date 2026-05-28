#!/bin/bash
echo "============================================="
echo "       IDE Expert Agent (Nexus) - macOS      "
echo "============================================="

# Optional: pass a project directory as the first argument.
#   ./start_nexus.sh /path/to/my/project
# If omitted, Nexus runs in the current working directory.
if [ -n "$1" ] && [ -d "$1" ]; then
    cd "$1" || exit 1
    echo "Project directory: $1"
else
    echo "Project directory: $(pwd)"
fi

# Nexus install root (where this script lives)
SSD=$(dirname "$(realpath "$0")")

# Important directories
export VENV_DIR="/Volumes/Sahith_SSD/IDE_Expert_Project/mac/venv"
export APP_DIR="$SSD/app"

# Point to the SSD model store — avoids downloading anything to Mac
export OLLAMA_MODELS="/Volumes/Sahith_SSD/Ollama_Models"

# 1. Start Ollama server with SSD models path
echo "Starting Ollama server (using SSD models)..."
pkill -x ollama 2>/dev/null
sleep 2
OLLAMA_MODELS="$OLLAMA_MODELS" /opt/homebrew/bin/ollama serve > /tmp/nexus_ollama.log 2>&1 &
OLLAMA_PID=$!
sleep 3

# 2. Activate python virtual environment
echo "Starting Python Environment..."
source "$VENV_DIR/bin/activate"

# 3. Launch the Chat CLI (CWD is the active project directory)
echo "Launching Nexus..."
python "$APP_DIR/cli.py"

# --- Cleanup happens after the CLI exits ---
echo "Shutting down..."
if [ ! -z "$OLLAMA_PID" ]; then
    kill $OLLAMA_PID 2>/dev/null
    echo "Ollama server stopped."
fi
echo "You can safely unplug your SSD."
