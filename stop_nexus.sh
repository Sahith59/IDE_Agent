#!/bin/bash
echo "============================================="
echo "   Stopping IDE Expert Agent (Nexus) - macOS "
echo "============================================="

# Kill Ollama
pkill -f ollama
echo "Ollama server stopped."

# Kill Python processes running cli.py
pkill -f "python.*cli.py"
echo "Nexus CLI stopped."

echo "All processes stopped. Safe to unplug SSD."
