# Nexus: Offline IDE Expert Agent 🤖💻

Nexus is a fully local, privacy-first, retrieval-augmented generation (RAG) agent that functions as an all-knowing senior team member for your application. It acts as an offline terminal-based assistant that allows you to chat securely with your own product documentation, diagrams, code, and flowcharts.

## 🌟 Key Features

*   **100% Offline & Private:** Powered by a local Ollama LLM (`qwen2.5:14b` / `llama3`). No data ever leaves your SSD.
*   **Multimodal RAG Pipeline:** Ingests not just text, but embedded flowcharts and images from `.docx`, `.pptx`, and `.pdf` files using Vision Models.
*   **Enterprise Semantic Router:** Integrates `semantic-router` with the `1.3GB mxbai-embed-large-v1` embedding model to accurately route queries. Generic coding questions instantly bypass the vector database, while expert knowledge questions trigger deep document retrieval.
*   **Zero Hallucination Strict Mode:** When querying documentation, Nexus is rigidly prompted to answer *only* based on retrieved ChromaDB context.
*   **Blazing Fast Startup:** Employs Lazy Loading architectures so heavy ML libraries (like PyTorch and Transformers) don't lock up macOS during CLI startup.

## 🛠️ Tech Stack
*   **Core UI:** Python, Rich (for beautiful CLI, Markdown tables, and live streaming)
*   **Language Models:** Ollama (`qwen2.5:14b` for RAG, `llama3` for Chat, `llama3.2-vision:11b` for images)
*   **Embeddings & Routing:** `nomic-embed-text` & `mixedbread-ai/mxbai-embed-large-v1`
*   **Vector Database:** Local ChromaDB
*   **Document Parsing:** LangChain, PyMuPDF, `python-docx`, `python-pptx`, Pandas

## 🚀 How to Run

1.  **Drop your files:** Place any `.pdf`, `.docx`, `.xlsx`, or `.pptx` documents into the `data/raw_docs/` folder.
2.  **Ingest:** Run the ingestion pipeline to extract text/images and embed them into ChromaDB:
    ```bash
    ./mac/venv/bin/python3 ingest.py
    ```
3.  **Chat:** Start the interactive console and select your model.
    ```bash
    ./start_nexus.sh
    ```

## 🧠 The Agent Personalities

Thanks to the Semantic Router, Nexus wears two hats depending on what you ask:
*   **The Engineer:** Ask it to *"write a binary search script"*, and it will act as a brilliant software engineer, writing code freely and rapidly without database overhead.
*   **The Architect:** Ask it *"what happened between IDEA and PEARS?"*, and it will freeze its knowledge, boot up ChromaDB, pull the relevant paragraphs from your business documents, and answer cautiously and accurately.
