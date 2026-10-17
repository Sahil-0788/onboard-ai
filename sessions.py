import json
import os
import threading
import time

import paths

SESSIONS_FILE = paths.SESSIONS_PATH

# The background indexing thread and a question can both write chats at the same moment,
# so every read-modify-write happens under this lock.
_LOCK = threading.RLock()


def load_sessions():
    if not os.path.exists(SESSIONS_FILE):
        return {}
    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_sessions(sessions):
    """Writes to a temp file first, then swaps it in, so a crash can't leave a half-written file."""
    tmp = SESSIONS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)
    os.replace(tmp, SESSIONS_FILE)


def create_session(folder):
    with _LOCK:
        sessions = load_sessions()
        session_id = str(int(time.time() * 1000))
        sessions[session_id] = {
            "folder": folder,
            "created_at": time.strftime("%Y-%m-%d %H:%M"),
            "messages": [],
        }
        save_sessions(sessions)
        return session_id


def add_message(session_id, question, answer, focus=None):
    """focus = full paths of the file(s) this exchange was about, so follow-ups can reuse them."""
    with _LOCK:
        sessions = load_sessions()
        if session_id in sessions:
            sessions[session_id]["messages"].append(
                {"question": question, "answer": answer, "focus": focus or []}
            )
            save_sessions(sessions)


def get_session(session_id):
    return load_sessions().get(session_id)


def list_sessions():
    sessions = load_sessions()
    return sorted(sessions.items(), key=lambda kv: kv[1]["created_at"], reverse=True)


def list_distinct_folders():
    sessions = load_sessions()
    return sorted(set(s["folder"] for s in sessions.values()))


def clear_all():
    with _LOCK:
        if os.path.exists(SESSIONS_FILE):
            os.remove(SESSIONS_FILE)