#!/bin/bash
echo "============================================="
echo "       IDE Expert Agent (Nexus) - macOS      "
echo "============================================="

# Keep track of where the script is run from (the SSD root path for your project)
SSD=$(dirname "$(realpath "$0")")
echo "Detected Project Root at: $SSD"

# Important directories 
export VENV_DIR="$SSD/mac/venv"
export APP_DIR="$SSD/app"

# Point to your existing models directory on the parent SSD layout
export OLLAMA_MODELS="/Volumes/Sahith_SSD/Ollama_Models"

# Set up the local Ollama binaries inside our project
export OLLAMA_HOME="$SSD/ollama/mac"

# 1. Start Ollama server in the background
echo "Starting local Ollama server..."
# Using the Host machine's Ollama or downloaded local Ollama if applicable
# For this setup, assuming Ollama exists in the system or you followed the guide to place it in ollama/mac/
if [ -f "$OLLAMA_HOME/ollama" ]; then
    "$OLLAMA_HOME/ollama" serve &
    OLLAMA_PID=$!
else
    echo "Warning: No local Ollama binary found at $OLLAMA_HOME/ollama."
    echo "Assuming Ollama is running globally on the host Mac."
    # We still want to let it run, it'll connect to localhost:11434
    OLLAMA_PID=""
fi

# Give Ollama a few seconds to boot up completely
sleep 3

# 2. Activate python virtual environment
echo "Starting Python Environment..."
source "$VENV_DIR/bin/activate"

# 3. Launch the Chat CLI
echo "Launching Nexus..."
python "$APP_DIR/cli.py"

# --- Cleanup happens after the CLI exits ---
echo "Shutting down..."
if [ ! -z "$OLLAMA_PID" ]; then
    kill $OLLAMA_PID 2>/dev/null
    echo "Ollama server stopped."
fi
echo "You can safely unplug your SSD."
