import os


def _base_dir():
    """Where onboard-ai keeps its data: the vector database and the saved chats.
    Set the ONBOARD_AI_DATA environment variable to use a different folder (handy for tests)."""
    override = os.environ.get("ONBOARD_AI_DATA")
    if override:
        base = override
    else:
        root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        base = os.path.join(root, "onboard-ai")
    os.makedirs(base, exist_ok=True)
    return base


DATA_DIR = _base_dir()
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")
SESSIONS_PATH = os.path.join(DATA_DIR, "chat_sessions.json")