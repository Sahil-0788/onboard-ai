# onboard-ai

A fully offline AI assistant that lets you ask natural-language questions about any codebase - no internet required, no code ever leaves your machine.

Point it at a folder, and it reads, understands, and lets you query the codebase in plain English - "where is user login handled," "what does this file do" - getting grounded answers that cite the exact file involved.

## Why offline matters

Most AI coding assistants (Copilot, Cursor, Claude Code) send your code to a cloud API. That's a dealbreaker for many companies with strict IP/data policies, and it means your tool is useless without internet. `onboard-ai` runs entirely on your own laptop - the embedding model, the vector database, and the LLM all run locally via [Ollama](https://ollama.com). Zero API calls, zero cost per query, zero code leaving your device.

## How it works

This is a Retrieval-Augmented Generation (RAG) pipeline, built from scratch:

1. **Chunking** - walks the target folder, reads every code file, and splits each one into smaller chunks (skipping junk folders like `node_modules`, `.git`, `venv`).
2. **Embedding** - each chunk is converted into a vector embedding using `nomic-embed-text`, running locally via Ollama. Filenames are embedded alongside the code itself, so filename-specific questions retrieve reliably.
3. **Storage & retrieval** - embeddings are stored in a local [ChromaDB](https://www.trychroma.com/) vector database. A question is embedded the same way, and the most semantically similar chunks are retrieved - not keyword matching, actual meaning-based search.
4. **Generation** - the retrieved chunks are fed to a local LLM (`phi3`, via Ollama) along with the question, producing a concise, grounded answer that cites the specific file(s) involved.

```
folder → chunk → embed → store (ChromaDB) → search → LLM answer (phi3)
```

## Tech stack

- **Python** - core pipeline
- **Streamlit** - web interface
- **Ollama** - runs local models (`phi3` for generation, `nomic-embed-text` for embeddings)
- **ChromaDB** - local vector database

## Running it locally

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) installed.

```bash
# Pull the required models
ollama pull phi3
ollama pull nomic-embed-text

# Set up the environment
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Make sure `ollama serve` is running in the background before starting the app.

## Status

Actively in development. Currently a desktop web app (Streamlit); planned upgrade to a packaged native desktop app (Tauri).

### Known limitations
- Single codebase at a time (multi-folder indexing without ID conflicts in progress)
- Re-indexing clears and rebuilds the index for that folder rather than incrementally updating changed files only

## Roadmap

- [ ] Support indexing multiple codebases without ID collisions
- [ ] Incremental re-indexing (only changed files)
- [ ] Show which source files were used to generate each answer
- [ ] Package as a native desktop app (Tauri) with an installable `.exe`
