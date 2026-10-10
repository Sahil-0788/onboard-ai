import json
import os
import time

SESSIONS_FILE = "chat_sessions.json"


def load_sessions():
    if not os.path.exists(SESSIONS_FILE):
        return {}
    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_sessions(sessions):
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)


def create_session(folder):
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