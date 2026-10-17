from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import core
import sessions

app = FastAPI(title="onboard-ai backend")

# The server only listens on 127.0.0.1 (this laptop), so allowing any origin
# just lets the desktop UI talk to it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_DOWN = "Can't connect to Ollama. Make sure Ollama is running, then try again."


class FolderRequest(BaseModel):
    folder: str


class AskRequest(BaseModel):
    question: str


def _require_ollama():
    if not core.ollama_running():
        raise HTTPException(status_code=503, detail=OLLAMA_DOWN)


@app.get("/health")
def health():
    return core.ollama_status()


@app.get("/sessions")
def list_sessions():
    items = sorted(sessions.list_sessions(), key=lambda kv: int(kv[0]), reverse=True)
    return [
        {
            "id": sid,
            "folder": data["folder"],
            "created_at": data["created_at"],
            "message_count": len(data["messages"]),
        }
        for sid, data in items
    ]


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    data = sessions.get_session(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="That chat no longer exists.")
    return {"id": session_id, **data}


@app.post("/sessions")
def new_session(req: FolderRequest):
    try:
        return {"session_id": core.start_session(req.folder)}
    except core.CoreError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/sessions/{session_id}/status")
def session_status(session_id: str):
    """Has the code changed since this chat's folder was indexed?"""
    try:
        return core.get_status(session_id)
    except core.CoreError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/sessions/{session_id}/refresh")
def refresh(session_id: str):
    """Re-indexes only the changed, new and deleted files. Poll GET /index/{job_id}."""
    _require_ollama()
    try:
        return core.start_refresh_job(session_id)
    except core.CoreError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong while refreshing: {e}")


@app.post("/index")
def index(req: FolderRequest):
    """Starts indexing in the background and returns a job_id. Poll GET /index/{job_id}."""
    _require_ollama()
    try:
        return core.start_index_job(req.folder)
    except core.CoreError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong while indexing: {e}")


@app.get("/index/{job_id}")
def index_status(job_id: str):
    job = core.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail="Unknown job. It may have finished long ago or the server was restarted.",
        )
    return job


@app.post("/sessions/{session_id}/ask")
def ask(session_id: str, req: AskRequest):
    _require_ollama()
    try:
        return core.answer_question(session_id, req.question)
    except core.CoreError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong while answering: {e}")


@app.post("/reset")
def reset():
    try:
        core.reset_all()
        return {"ok": True}
    except core.CoreError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Couldn't reset: {e}")