# Nexus - Offline IDE Expert Agent

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-orange.svg)](https://python.langchain.com/docs/langgraph)
[![ChromaDB](https://img.shields.io/badge/Database-ChromaDB-green.svg)](https://docs.trychroma.com/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-lightgrey.svg)](https://ollama.com/)
[![Qwen](https://img.shields.io/badge/Model-Qwen_2.5_14b-purple.svg)](https://huggingface.co/Qwen)
[![macOS](https://img.shields.io/badge/macOS-Only-black.svg?logo=apple)](https://www.apple.com/macos/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

Nexus is a fully local, privacy-first, retrieval-augmented generation (RAG) agent that functions as an offline expert team member for your application. Operating entirely within a secure, terminal-based environment, Nexus ingests your source code, architectural documentation, git history, and markdown notes to provide context-aware technical assistance without ever transmitting data to external servers.

---

## Architectural Highlights

### 1. LangGraph Semantic Orchestration
The application control flow is managed by a `langgraph.graph.StateGraph` Directed Acyclic Graph (DAG). 
- Every incoming user query is vectorized by a lightweight HuggingFace encoder (`mxbai-embed-large-v1`) in under 0.2 seconds.
- The intent is mathematically clustered into one of five distinct enterprise categories: `Code`, `Architecture`, `Business Logic`, `Ops`, or `Tribal Knowledge`.
- The graph conditionally routes the execution; generic conversational queries completely bypass the database for instant responses, while domain-specific questions trigger the Ensemble Retrieval node.

### 2. Dual-Index Hybrid RAG Pipeline
Nexus relies on a dual-ingestion architecture to eliminate the classic weaknesses of pure vector databases:
- **Dense Vector Search (ChromaDB):** Converts documents into mathematical embeddings using `nomic-embed-text` for deep semantic understanding and conceptual similarity.
- **Sparse Keyword Search (BM25):** Tokenizes documents into an inverted index (`bm25_index.pkl`) to guarantee exact-match retrieval for alphanumeric identifiers, error codes, and unique acronyms.
- **Ensemble Retrieval:** At inference, Nexus fuses both databases and removes duplicate chunks to provide the optimal, citation-backed context window to the language model.

### 3. Max Marginal Relevance (MMR)
To prevent the language model from receiving multiple redundant paragraphs describing the exact same concept, the retrieval node strictly utilizes Max Marginal Relevance (`fetch_k=20`, `k=4`). This mathematically forces the retrieved chunks to be both highly relevant to the query and highly diverse from one another.

### 4. Multimodal Vision Ingestion
Documentation rarely consists of exclusively plain text. Nexus is engineered to parse standard `.docx`, `.pptx`, and `.pdf` files, physically extract embedded flowcharts, diagrams, and images, and process them through a local Vision LLM (`llama3.2-vision:11b`). The generated semantic descriptions of the diagrams are then embedded directly into the databases alongside the textual content.

### 5. Continuous Offline Learning
Nexus supports instantaneous "drop-in" knowledge updates. By natively parsing Markdown (`.md`) files and Excel (`.xlsx`) spreadsheets into localized metadata chunks, developers can continuously drop meeting notes, deployment logs, or architecture decision records into the data directory for rapid re-indexing.

### 6. Zero-Latency Workspace Awareness & File Injection
To function as a true modular IDE assistant, Nexus executes a lightning-fast heuristic upon startup to map the user's current project directory (`os.getcwd()`), invisibly injecting the active repository's folder structure into the context window. Furthermore, developers can target local scripts using the `@filename` syntax (e.g., "Find the bug in `@auth.py`"). Nexus intercepts the token via regex, physically reads the raw source code, and directly injects it into the LLM stream. This guarantees 0.0s latency codebase inclusion and strictly bypasses the fragmentation risks of traditional document chunking.

---

## Technical Stack & Hardware Optimization

- **Core Application:** Python, Rich (CLI UI engine)
- **AI Orchestration:** LangGraph, LangChain
- **Language Models (Local):** Ollama (`qwen2.5:14b`, `llama3`, `llama3.2-vision:11b`)
- **Embeddings:** `nomic-embed-text`, `mixedbread-ai/mxbai-embed-large-v1`
- **Databases:** Local ChromaDB, localized Python Pickles (BM25)
- **Memory Optimization:** Designed specifically for Apple Silicon hardware efficiency. Heavy Machine Learning libraries (PyTorch, Transformers, Pandas) are exclusively lazy-loaded dynamically when required by a specific execution path, completely eliminating CLI startup compilation delays.

---

## Setup and Installation

### Prerequisites
- macOS (Apple Silicon highly recommended for acceptable inference latency).
- [Ollama](https://ollama.ai/) installed locally.

### Usage
1. **Drop your files:** Place any `.pdf`, `.docx`, `.xlsx`, `.pptx`, or `.md` documents into the `data/raw_docs/` folder.
2. **Ingest Documentation:** Run the indexing pipeline to compile the databases.
   ```bash
   ./mac/venv/bin/python3 ingest.py
   ```
3. **Start the Agent:** Launch the interactive Nexus CLI.
   ```bash
   ./start_nexus.sh
   ```
