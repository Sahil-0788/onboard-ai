# onboard-ai

A fully offline AI assistant that lets you ask natural-language questions about any codebase - no internet required, no code ever leaves your machine.

Point it at a folder, and it reads, understands, and lets you query the codebase in plain English - "where is user login handled," "what does this file do" - getting grounded answers that cite the exact files involved.

## Why offline matters

Most AI coding assistants (Copilot, Cursor, Claude Code) send your code to a cloud API. That's a dealbreaker for many companies with strict IP/data policies, and it means your tool is useless without internet. `onboard-ai` runs entirely on your own laptop - the embedding model, the vector database, and the LLM all run locally via [Ollama](https://ollama.com). Zero API calls, zero cost per query, zero code leaving your device.

## How it works

This is a Retrieval-Augmented Generation (RAG) pipeline, built from scratch:

1. **Chunking** - walks the target folder, reads every code file, and splits each one into smaller chunks (skipping junk folders like `node_modules`, `.git`, `venv`).
2. **Embedding** - each chunk is converted into a vector embedding using `nomic-embed-text`, running locally via Ollama. Filenames are embedded alongside the code itself, so filename-specific questions retrieve reliably.
3. **Storage & retrieval** - embeddings are stored in a local [ChromaDB](https://www.trychroma.com/) vector database. A question is embedded the same way and the most semantically similar chunks are retrieved (meaning-based search, not keyword matching). If the question names a file, that file is read directly instead.
4. **Generation** - the retrieved chunks are fed to a local LLM (`phi3`, via Ollama) along with the question and recent chat history, producing a short answer that cites the files involved.

```
folder -> chunk -> embed -> store (ChromaDB) -> search or read file -> LLM answer (phi3)
```

Extras built into the pipeline:
- Typo correction for file names ("pattrn3.cpp" is read as "pattern3.cpp")
- Follow-up questions ("what is num used for in that?") stay on the file being discussed
- Chats are saved and can be reopened

## Tech stack

- **Python** - core pipeline
- **FastAPI** - local backend API (`core.py` holds the logic, `api.py` exposes it)
- **Ollama** - runs local models (`phi3` for generation, `nomic-embed-text` for embeddings)
- **ChromaDB** - local vector database
- **React + Tauri** - desktop UI (in progress)

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
```

Run these in two separate terminals.

Terminal 1 (Ollama):
```bash
ollama serve
```

Terminal 2 (backend):
```bash
venv\Scripts\activate
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Then open `http://127.0.0.1:8000/docs` for an interactive page where you can try every endpoint.

### API endpoints

| Endpoint | What it does |
|---|---|
| `GET /health` | Checks that Ollama and both models are available |
| `POST /index` | Indexes a folder and creates a new chat |
| `GET /sessions` | Lists saved chats, newest first |
| `GET /sessions/{id}` | Returns one chat with its messages |
| `POST /sessions` | Starts a new chat on an already indexed folder |
| `POST /sessions/{id}/ask` | Asks a question; returns the answer, sources, and any typo correction |
| `POST /reset` | Deletes all indexes and chats |

`app.py` is the original Streamlit UI. It is deprecated and still uses the old logic; the React desktop UI will replace it.

## Status

Actively in development. The backend works end to end; the desktop UI and installer are next.

### Known limitations
- **Answer quality is limited by the small local model.** `phi3` (3.8B) sometimes gets details wrong or contradicts itself. Evaluating a code-focused model is planned.
- Indexing is one blocking call with no progress bar yet, and is slow on large folders.
- With many folders indexed, a question about a small one can come back empty (the folder filter runs after the search).
- Data is stored next to the code; a packaged app needs a per-user data folder.
- Re-indexing clears and rebuilds the index for that folder rather than updating only changed files.

## Roadmap

- [x] FastAPI backend with saved chats and follow-up questions
- [x] Show which source files were used for each answer
- [ ] Improve answer quality (cleanup of model output, test code-focused models)
- [ ] Filter by folder inside ChromaDB; progress reporting for indexing
- [ ] React UI
- [ ] Package as a native desktop app (Tauri) with an installable `.exe`
- [ ] First-run setup that installs Ollama and downloads the models
- [ ] Incremental re-indexing (only changed files)