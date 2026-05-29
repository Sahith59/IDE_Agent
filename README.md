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

## Install

### Option 1 — pip (recommended, installs a permanent `nexus` command)

```bash
pip install ollama-term
```

Then run from any project directory:

```bash
cd /path/to/your/project
nexus
```

First launch runs a **setup wizard** that asks for your Ollama models path and lets you pick your default model. Config is saved to `~/.nexus/config.json` and never asked again.

### Option 2 — Development / from source

```bash
git clone https://github.com/Sahith59/IDE_Agent.git
cd IDE_Agent
pip install -e .
nexus
```

---

## Prerequisites

- [Ollama](https://ollama.ai/) installed and running
- Python 3.11+
- macOS or Linux

### Pull required models

```bash
ollama pull qwen2.5:14b          # default chat model
ollama pull nomic-embed-text     # embeddings (for RAG)
ollama pull llama3.2-vision:11b  # vision ingestion (optional)
```

### External model store (SSD / NAS)

If your models live on an external drive, either:
- Set `OLLAMA_MODELS=/Volumes/Drive/Ollama_Models` before running `nexus`, **or**
- Enter the path when the setup wizard asks on first launch

---

## Ingest Your Documentation

Drop `.pdf`, `.docx`, `.xlsx`, `.pptx`, or `.md` files into `~/.nexus/raw_docs/`, then:

```bash
nexus-ingest

# Or point at any directory:
nexus-ingest /path/to/your/docs
```

## Start Nexus

```bash
# Launch in current directory (Nexus maps it as the active project)
nexus

# Or with start_nexus.sh (for SSD model setup on macOS):
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

Create `~/.nexus/plugins/myplugin.py`:

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

Restart Nexus — `/myplugin` is now a live command. Built-in sample plugin (`/hello`) is included with the package.

---

## Data Directory

All persistent data lives in `~/.nexus/` — survives upgrades and reinstalls:

```
~/.nexus/
├── config.json     # ollama path, default model
├── history/        # saved sessions
├── chroma_db/      # vector database (after nexus-ingest)
├── bm25_index.pkl  # keyword index (after nexus-ingest)
├── raw_docs/       # drop files here before nexus-ingest
├── hf_cache/       # HuggingFace embedding model cache
├── plugins/        # your custom plugins
└── docs/           # /pr generated Word documents
```

## Project Structure

```
IDE_Agent/
├── nexusai/                # Installable Python package
│   ├── __init__.py
│   ├── cli.py              # Main CLI
│   ├── config.py           # ~/.nexus path management
│   ├── setup_wizard.py     # First-run wizard
│   ├── ingest.py           # RAG pipeline
│   ├── pr_generator.py     # GitHub PR → .docx tool
│   └── plugins/
│       └── hello.py        # Built-in sample plugin
├── pyproject.toml          # Package config + entry points
├── start_nexus.sh          # SSD/external model launcher (macOS)
├── requirements.txt        # Direct dependencies
└── README.md
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
