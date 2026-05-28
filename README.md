# Nexus — Offline IDE Expert Agent

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-orange.svg)](https://python.langchain.com/docs/langgraph)
[![ChromaDB](https://img.shields.io/badge/Database-ChromaDB-green.svg)](https://docs.trychroma.com/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-lightgrey.svg)](https://ollama.com/)
[![Qwen](https://img.shields.io/badge/Model-Qwen_2.5_14b-purple.svg)](https://huggingface.co/Qwen)
[![macOS](https://img.shields.io/badge/macOS-Only-black.svg?logo=apple)](https://www.apple.com/macos/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

Nexus is a fully local, privacy-first, retrieval-augmented generation (RAG) agent that functions as an offline expert team member for your codebase. It runs entirely inside your terminal — no cloud API calls, no data leaving your machine.

---

## Feature Overview

| Category | Feature |
|---|---|
| **UI / UX** | Animated NEXUS logo splash, orange-red theme, rich markdown responses |
| **Navigation** | Session browser, session history, `/new`, `/switch`, `/sessions` |
| **Context** | Pinned files (`/context add`), workspace tree injection, git-aware context |
| **File Access** | `@filename`, `@/absolute/path`, `@~/home/path`, quoted paths with spaces |
| **Web** | `@web:query` inline web search (DuckDuckGo, offline-safe) |
| **Agent Mode** | `@agent <intent>` — multi-turn file edit loop with unified diff preview |
| **Plugins** | Drop `.py` files in `plugins/` — auto-registers as `/command` |
| **PR Docs** | `/pr` — generate `.docx` summaries from GitHub PR URLs or screenshots |
| **Export** | `/export` — save full session to `~/Desktop/nexus_<session>_<date>.md` |
| **Diagnostics** | `/diagnose <error>` — greps workspace, builds targeted debug prompt |
| **Session Tags** | `/tag <label>`, `/search <keyword>` — organize and find past sessions |
| **Interrupt** | ESC key mid-response to stop generation instantly |
| **Post-response** | `[c] copy · [r] run · Enter` after every AI response |

---

## Architecture

### 1. LangGraph Semantic Orchestration

Every query is vectorized by `mxbai-embed-large-v1` and classified into one of five categories: `Code`, `Architecture`, `Business Logic`, `Ops`, or `Tribal Knowledge`. A `StateGraph` DAG routes generic queries directly to the LLM (bypassing the database) and domain queries through the Ensemble Retrieval node.

### 2. Dual-Index Hybrid RAG Pipeline

- **Dense (ChromaDB):** `nomic-embed-text` embeddings for semantic similarity
- **Sparse (BM25):** Inverted-index tokenization for exact identifier / error-code matches
- **Ensemble Retrieval:** Fused, deduplicated context window fed to the LLM

### 3. Max Marginal Relevance (MMR)

Retrieval uses MMR (`fetch_k=20`, `k=4`) to guarantee the four retrieved chunks are both highly relevant *and* maximally diverse — preventing the LLM from receiving redundant paragraphs.

### 4. Multimodal Vision Ingestion

`ingest.py` parses `.docx`, `.pptx`, and `.pdf` files, extracts embedded diagrams, and processes them through `llama3.2-vision:11b`. The generated semantic descriptions are embedded into the same databases as text content.

### 5. Workspace Awareness & File Injection

On startup, Nexus maps the active project directory (`os.getcwd()`) and injects the folder tree into every prompt. The `@filename` and `@/absolute/path` syntax reads raw source directly into the context window at zero latency — no chunking, no retrieval delay.

### 6. Agent Mode (`@agent`)

`@agent <intent>` launches a multi-turn agentic loop: Nexus scouts relevant files, proposes changes as `FILE:` blocks, displays a unified diff, and applies writes only after explicit user approval.

### 7. Plugin System

Any `.py` file in `plugins/` that exports a `PLUGIN` dict with `name`, `description`, and `run` keys is automatically registered as a `/name` slash command. The `run(arg, ctx)` function receives the argument string and a context dict with `session`, `llm`, `console`, and `root_dir`.

---

## Setup

### Prerequisites

- macOS (Apple Silicon recommended)
- [Ollama](https://ollama.ai/) installed locally
- Python 3.11+

### Install Dependencies

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Pull Required Models

```bash
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
ollama pull llama3.2-vision:11b   # only needed for vision ingestion
```

### Optional: SSD Model Store

If your Ollama models live on an external drive, set `OLLAMA_MODELS` before starting:

```bash
export OLLAMA_MODELS="/Volumes/YourDrive/Ollama_Models"
```

`start_nexus.sh` handles this automatically when configured.

---

## Usage

### 1. Ingest Your Documentation

Drop `.pdf`, `.docx`, `.xlsx`, `.pptx`, or `.md` files into `data/raw_docs/`, then:

```bash
python ingest.py
```

### 2. Start Nexus

```bash
# Launch in current directory (Nexus maps it as the active project)
./start_nexus.sh

# Or point at a specific project
./start_nexus.sh /path/to/your/project
```

---

## Commands Reference

### Session

| Command | Description |
|---|---|
| `/new` | Start a new session |
| `/sessions` | Browse and switch between past sessions |
| `/switch <id>` | Jump directly to a session by ID |
| `/export` | Save current session to a Markdown file on Desktop |
| `/tag <label>` | Tag the current session |
| `/search <keyword>` | Full-text search across all sessions |

### Context & Files

| Command | Description |
|---|---|
| `/context add <path>` | Pin a file — injected into every prompt |
| `/context show` | List pinned files |
| `/context rm <path>` | Unpin a file |
| `/context clear` | Clear all pinned files |
| `@filename` | Inject a workspace file inline |
| `@/absolute/path` | Inject any file from anywhere on disk |
| `@~/path` | Inject a file relative to home directory |
| `@web:query` | Run a DuckDuckGo search and inject results |

### Agent & Plugins

| Command | Description |
|---|---|
| `@agent <intent>` | Launch agentic file-edit loop |
| `/pr` | Generate a `.docx` summary from a GitHub PR URL or screenshot |
| `/<plugin>` | Run any installed plugin |

### Utilities

| Command | Description |
|---|---|
| `/diagnose <error>` | Debug an error using workspace symbol search |
| `/model <name>` | Switch the active Ollama model |
| `/clear` | Clear screen and start fresh |
| `/help` | Show command quick-reference |
| `/instructions` | Full paginated onboarding guide |
| `ESC` | Interrupt generation mid-response |

---

## Writing a Plugin

Create `plugins/myplugin.py`:

```python
def run(arg, ctx):
    console = ctx["console"]
    session = ctx["session"]
    console.print(f"Hello from myplugin! Arg: {arg}")

PLUGIN = {
    "name":        "myplugin",
    "description": "One-line description shown in /help",
    "run":         run,
}
```

Restart Nexus — `/myplugin` is now a live command.

---

## Project Structure

```
IDE_Agent/
├── app/
│   └── cli.py              # Main CLI application (v3.0.0)
├── data/
│   ├── raw_docs/           # Drop ingestion files here
│   └── history/            # Session JSON files
├── plugins/
│   └── hello.py            # Sample plugin
├── pr_generator/
│   └── generator.py        # GitHub PR → .docx documentation tool
├── ingest.py               # RAG ingestion pipeline
├── start_nexus.sh          # macOS launcher
├── requirements.txt        # Python dependencies
└── COMMANDS_GUIDE.md       # Full command reference
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama (`qwen2.5:14b`, `llama3`, `llama3.2-vision:11b`) |
| Embeddings | `nomic-embed-text`, `mxbai-embed-large-v1` |
| Orchestration | LangGraph `StateGraph` |
| Vector DB | ChromaDB |
| Keyword Search | BM25 (rank-bm25) |
| Semantic Routing | `semantic-router` + HuggingFace Transformers |
| CLI UI | Rich (panels, markdown, live, spinner, rules) |
| Web Search | ddgs (DuckDuckGo) |
| PDF Parsing | PyMuPDF (fitz) |
| Document Parsing | python-docx, python-pptx, openpyxl, docx2txt |

---

## License

MIT
