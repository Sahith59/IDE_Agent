# Nexus IDE Expert Agent - Commands Guide

Welcome to the IDE Expert Agent (Nexus)! This guide outlines all the commands and features you need to navigate the CLI.

## Startup & Menus

When you launch Nexus using `./start_nexus.sh` (or `.bat` on Windows), you are greeted with a session management menu.

### Session Menu Commands
When prompted to **"Select an option:"**, you can enter:
- **`[Enter]`**: Press the Return/Enter key without typing anything to start a brand new, clean chat session.
- **`[Number]`** (e.g., `0`, `1`): Type the number of an existing session to load it and continue from where you left off.
- **`d [Number]`** (e.g., `d 0`): Deletes the specified chat session permanently.
- **`r [Number] [New Name]`** (e.g., `r 0 debugging_auth`): Renames the specified session so you can easily identify it later. Names can include spaces and will be saved safely.

### Model Selection
After selecting your chat session, you will be prompted to pick an AI model.
- Enter the number corresponding to the model you want to use (e.g., `1` for `qwen2.5:14b`).
- Press **`[Enter]`** to default to `llama3`.
*Note: Do not select `nomic-embed-text` here, as it is an embedding model meant for databases, not chatting!*

## In-Chat Commands

Once the chat starts, the prompt will say **`You:`**. Type your questions naturally. 

To exit or perform actions during the chat:
- **`exit`** or **`quit`**: Safely saves your conversation and exits the application.
- **`history`**: *(Currently informative)* Will remind you of previous conversation turns in a future update if implemented as a direct command, but right now the chat remembers everything automatically!

## Safe Shutdown
Always exit the chat by typing `exit` or `quit` before closing the terminal window. This ensures your latest messages are written to your external SSD safely. The local Ollama server will cleanly shut itself down in the background.
