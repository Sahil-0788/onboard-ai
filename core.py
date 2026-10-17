import os
import re
import time
import uuid
import difflib
import threading
import ollama

import sessions
import vectorstore
from chunker import find_code_files, chunk_file
from embedder import get_embedding, get_embeddings
from vectorstore import (
    add_chunks, search, clear_folder, delete_files, list_indexed_files, in_folder, get_file_chunks,
)

# The model that writes answers. Change this one line to switch models.
ANSWER_MODEL = "qwen2.5-coder:3b"

# Words that suggest a question refers back to the previous answer ("what is X used for in that").
REFERENCE_WORDS = {"that", "it", "those", "these", "them", "they", "above", "earlier", "previous"}
# Words that suggest a broad search question instead ("which file handles ...").
SEARCH_WORDS = {"file", "files", "folder", "project", "codebase", "which", "where", "any"}

# File types where "//" starts a comment.
C_STYLE_EXTENSIONS = {".c", ".cpp", ".java", ".js", ".ts"}

MAX_CONTEXT_CHARS = 6500          # keeps the prompt inside a 4096-token window
MAX_HISTORY_ANSWER_CHARS = 500
MAX_SENTENCES = 4                 # hard cap on answer length (small models ignore the prompt rule)
MAX_SENTENCES_EXAMPLE = 6

EMBED_BATCH_SIZE = 16             # chunks embedded per Ollama call while indexing
MAX_JOBS_KEPT = 50

# Indexing jobs live in memory only: they are forgotten when the server restarts.
_JOBS = {}
_JOBS_LOCK = threading.Lock()


class CoreError(Exception):
    """An error with a message that is safe to show to the user."""


def ollama_status():
    try:
        resp = ollama.list()
        names = [(m.model or "") for m in resp.models]
    except Exception:
        return {"ollama_running": False, "answer_model": False, "embedding_model": False}
    return {
        "ollama_running": True,
        "answer_model": any(n.startswith(ANSWER_MODEL) for n in names),
        "embedding_model": any(n.startswith("nomic-embed-text") for n in names),
    }


def ollama_running():
    return ollama_status()["ollama_running"]


# ---------------------------------------------------------------- change detection

def _stamp(path):
    """(modified time, size) of a file. Used to notice edits made after indexing."""
    st = os.stat(path)
    return (st.st_mtime, st.st_size)


def _scan_changes(folder):
    """Compares what was indexed with what is on disk now."""
    result = {"changed": [], "added": [], "removed": [], "folder_missing": False}
    if not os.path.isdir(folder):
        result["folder_missing"] = True
        return result

    stored = vectorstore.file_stamps(folder)
    on_disk = {}
    for path in find_code_files(folder):
        try:
            on_disk[path] = _stamp(path)
        except OSError:
            continue

    for path, stamp in stored.items():
        if path not in on_disk:
            result["removed"].append(path)
        elif not _same_stamp(stamp, on_disk[path]):
            result["changed"].append(path)

    for path in on_disk:
        if path not in stored:
            try:
                # An empty file never produces chunks, so it can never be "indexed". Ignore it.
                if chunk_file(path):
                    result["added"].append(path)
            except OSError:
                continue

    for key in ("changed", "added", "removed"):
        result[key].sort()
    return result


def _stale_summary(scan):
    """Short description of what changed, or None if the index is up to date."""
    if scan["folder_missing"]:
        return {"folder_missing": True, "changed": 0, "added": 0, "removed": 0, "examples": []}
    changed, added, removed = scan["changed"], scan["added"], scan["removed"]
    if not (changed or added or removed):
        return None
    names = [os.path.basename(p) for p in changed + added + removed]
    return {
        "folder_missing": False,
        "changed": len(changed),
        "added": len(added),
        "removed": len(removed),
        "examples": names[:5],
    }


def get_status(session_id):
    session = sessions.get_session(session_id)
    if not session:
        raise CoreError("That chat no longer exists.")
    stale = _stale_summary(_scan_changes(session["folder"]))
    return {"up_to_date": stale is None, "stale": stale}


# ---------------------------------------------------------------- indexing jobs

def _update_job(job_id, **fields):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job:
            job.update(fields)


def _finish_job(job_id, **fields):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job:
            job["seconds"] = round(time.time() - job["started"], 1)
            job.update(fields)


def _trim_jobs():
    """Called while holding the lock. Forget the oldest finished jobs."""
    if len(_JOBS) <= MAX_JOBS_KEPT:
        return
    finished = [jid for jid, j in _JOBS.items() if j["status"] != "running"]
    for jid in finished[: len(_JOBS) - MAX_JOBS_KEPT]:
        del _JOBS[jid]


def _register_job(folder, kind, total, files):
    """Creates a job record. Refuses if the same folder is already being worked on."""
    job_id = uuid.uuid4().hex[:10]
    with _JOBS_LOCK:
        for job in _JOBS.values():
            if job["folder"] == folder and job["status"] == "running":
                raise CoreError("That folder is already being processed. Wait for it to finish.")
        _JOBS[job_id] = {
            "job_id": job_id,
            "kind": kind,              # index | refresh
            "folder": folder,
            "status": "running",       # running | done | error
            "phase": "embedding",      # embedding | saving | done
            "done": 0,
            "total": total,
            "files": files,
            "result": None,
            "error": None,
            "seconds": None,
            "started": time.time(),
        }
        _trim_jobs()
    return job_id


def _embed_batch(batch):
    """Embeds a batch of chunks. If the batch call fails, retries one by one so a single
    bad chunk only skips itself. Returns a list the same length as batch (None = failed)."""
    try:
        vectors = get_embeddings([c["text"] for c in batch])
        if len(vectors) == len(batch):
            return vectors
    except Exception:
        pass
    out = []
    for chunk in batch:
        try:
            out.append(get_embedding(chunk["text"]))
        except Exception:
            out.append(None)
    return out


def _run_job(job_id, folder, all_chunks, stamps, mode, info, remove_paths=None):
    """Runs in a background thread. mode 'index' replaces the whole folder's index;
    mode 'refresh' replaces only the files in remove_paths and adds the new chunks."""
    try:
        valid_chunks, embeddings, skipped = [], [], 0
        total = len(all_chunks)
        for i in range(0, total, EMBED_BATCH_SIZE):
            batch = all_chunks[i : i + EMBED_BATCH_SIZE]
            for chunk, vector in zip(batch, _embed_batch(batch)):
                if vector is None:
                    skipped += 1
                else:
                    valid_chunks.append(chunk)
                    embeddings.append(vector)
            _update_job(job_id, done=min(i + len(batch), total))

        if mode == "index" and not valid_chunks:
            raise CoreError("No chunks could be processed. The files may be too large or unusual to embed.")

        # Old data is removed only now, after the new embeddings exist, so a failed run
        # never wipes a working index.
        _update_job(job_id, phase="saving")
        if mode == "index":
            clear_folder(folder)
        else:
            delete_files(folder, remove_paths or [])
        if valid_chunks:
            add_chunks(valid_chunks, embeddings, folder, stamps)

        result = dict(info)
        if mode == "index":
            result["session_id"] = sessions.create_session(folder)
        result["chunks"] = len(valid_chunks)
        result["skipped"] = skipped
        _finish_job(job_id, status="done", phase="done", result=result)
    except CoreError as e:
        _finish_job(job_id, status="error", error=str(e))
    except Exception as e:
        _finish_job(job_id, status="error", error=f"Something went wrong while indexing: {e}")


def start_index_job(folder):
    """Checks the folder and reads its files right away (so mistakes fail fast with a clear
    message), then embeds in the background. Returns a job id to poll with get_job."""
    folder = folder.strip().strip('"')
    if not folder:
        raise CoreError("Please enter a folder path first.")
    if not os.path.isdir(folder):
        raise CoreError("That folder doesn't exist. Double-check the path and try again.")
    folder = os.path.normpath(folder)

    files = find_code_files(folder)
    if not files:
        raise CoreError("No supported code files found in that folder.")

    stamps, all_chunks = {}, []
    for file_path in files:
        try:
            stamps[file_path] = _stamp(file_path)   # taken BEFORE reading, so a later edit is noticed
        except OSError:
            continue
        all_chunks.extend(chunk_file(file_path))
    if not all_chunks:
        raise CoreError("Files were found, but no readable content was extracted from them.")

    job_id = _register_job(folder, "index", len(all_chunks), len(files))
    threading.Thread(
        target=_run_job,
        args=(job_id, folder, all_chunks, stamps, "index", {"folder": folder, "files": len(files)}),
        daemon=True,
    ).start()
    return {"job_id": job_id, "folder": folder, "files": len(files), "chunks": len(all_chunks)}


def start_refresh_job(session_id):
    """Re-indexes only what changed since the folder was indexed. Keeps the same chat."""
    session = sessions.get_session(session_id)
    if not session:
        raise CoreError("That chat no longer exists.")
    folder = session["folder"]

    scan = _scan_changes(folder)
    if scan["folder_missing"]:
        raise CoreError("The folder no longer exists on this computer, so it can't be refreshed.")
    changed, added, removed = scan["changed"], scan["added"], scan["removed"]
    if not (changed or added or removed):
        raise CoreError("Nothing has changed since this folder was indexed.")

    stamps, new_chunks = {}, []
    for path in changed + added:
        try:
            stamps[path] = _stamp(path)
        except OSError:
            continue
        new_chunks.extend(chunk_file(path))

    info = {
        "session_id": session_id,
        "folder": folder,
        "changed": len(changed),
        "added": len(added),
        "removed": len(removed),
    }
    job_id = _register_job(folder, "refresh", len(new_chunks), len(changed) + len(added) + len(removed))
    threading.Thread(
        target=_run_job,
        args=(job_id, folder, new_chunks, stamps, "refresh", info, changed + removed),
        daemon=True,
    ).start()
    return {
        "job_id": job_id,
        "folder": folder,
        "changed": len(changed),
        "added": len(added),
        "removed": len(removed),
        "chunks": len(new_chunks),
    }


def get_job(job_id):
    """Progress of an indexing job, or None if the id is unknown (or the server restarted)."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        view = {k: v for k, v in job.items() if k != "started"}
        if job["status"] == "running":
            view["seconds"] = round(time.time() - job["started"], 1)
    total = view["total"] or 1
    view["percent"] = 100 if view["status"] == "done" else min(99, int(view["done"] * 100 / total))
    return view


# ---------------------------------------------------------------- chats and answers

def start_session(folder):
    folder = os.path.normpath(folder.strip().strip('"'))
    if folder not in sessions.list_distinct_folders():
        raise CoreError("That folder hasn't been indexed yet.")
    return sessions.create_session(folder)


def _folder_files(folder):
    return sorted(list_indexed_files(folder))


def _candidates(folder, files):
    """Names a user might type: file names (with and without extension) and the folder's own name.
    The folder name is split into single words ("C Programs" -> "c", "programs"), because a
    name containing a space can never match one typed word and caused bad corrections."""
    names = set(re.split(r"[\s_\-]+", os.path.basename(folder).lower()))
    names.discard("")
    for path in files:
        rel = os.path.relpath(path, folder)
        for part in rel.replace("\\", "/").split("/"):
            part = part.lower()
            if part:
                names.add(part)                        # e.g. pattern3.cpp
                names.add(os.path.splitext(part)[0])   # e.g. pattern3
    return names


def _related(word, name):
    """True if the word is just a shorter or longer form of a name (pattern vs pattern1)."""
    stem = re.sub(r"[\d_\-]+$", "", name)
    return bool(stem) and (word.startswith(stem) or stem.startswith(word))


def _correct_typos(question, candidates):
    words, corrected = [], False
    for word in question.split():
        clean = word.strip(".,?!\"'").lower()
        if len(clean) > 3 and clean not in candidates and not any(_related(clean, c) for c in candidates):
            match = difflib.get_close_matches(clean, candidates, n=1, cutoff=0.8)
            if match:
                words.append(match[0])
                corrected = True
                continue
        words.append(word)
    return (" ".join(words) if corrected else question), corrected


def _mentioned_files(question, files):
    """Files the question names directly: a full file name, or a distinctive name without extension."""
    by_name, by_stem = {}, {}
    for path in files:
        base = os.path.basename(path).lower()
        by_name.setdefault(base, []).append(path)
        by_stem.setdefault(os.path.splitext(base)[0], []).append(path)

    found = []
    for word in question.split():
        token = word.strip(".,?!\"'()[]:;").lower()
        if not token:
            continue
        matches = by_name.get(token)
        if matches is None and (len(token) >= 6 or any(c.isdigit() for c in token)):
            matches = by_stem.get(token)
        # A name shared by many files (like index.html) is ambiguous, so don't guess.
        if matches and len(matches) <= 2:
            for m in matches:
                if m not in found:
                    found.append(m)
    return found[:3]


def _strip_commented_code(text):
    """Remove '//' comments that look like disabled code (they end in ; { or }),
    so the model doesn't mistake them for real code. Prose comments are kept."""
    out = []
    for line in text.split("\n"):
        m = re.search(r"(?<!:)//(.*)$", line)
        # Skip matches inside a string: an odd number of quotes before the '//' means we're in one.
        if m and line[: m.start()].count('"') % 2 == 0 and m.group(1).rstrip().endswith((";", "{", "}")):
            line = line[: m.start()].rstrip()
            if not line.strip():
                continue
        out.append(line)
    return "\n".join(out)


def _cut_leaked_tail(answer):
    """Small models sometimes add a made-up ALL-CAPS heading or 'Certainly!' after the real answer."""
    pattern = r"\n\s*\n(?=\s*(?:Certainly\b|[A-Z][A-Z0-9 ,.'\-?:]{15,}))"
    return re.split(pattern, answer)[0].strip()


def _limit_sentences(text, max_sentences):
    """Keep only the first few sentences. A split needs whitespace after the punctuation,
    so file names like pattern3.cpp are not broken apart."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= max_sentences:
        return text
    return " ".join(sentences[:max_sentences]).strip()


def _trim_to_sentence(text):
    """If the answer was cut off by the token limit, drop the unfinished last sentence."""
    if not text or text[-1] in ".!?":
        return text
    cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "), text.rfind(".\n"))
    if cut > len(text) * 0.4:
        return text[: cut + 1].strip()
    return text


def answer_question(session_id, question, model=None):
    model = model or ANSWER_MODEL
    question = question.strip()
    if not question:
        raise CoreError("Please type a question first.")

    session = sessions.get_session(session_id)
    if not session:
        raise CoreError("That chat no longer exists.")
    folder = session["folder"]
    history = session["messages"]

    files = _folder_files(folder)
    if not files:
        raise CoreError("This folder needs to be indexed again before you can ask about it.")

    # Tell the caller if the code changed since indexing (the answer still uses the old index).
    try:
        stale = _stale_summary(_scan_changes(folder))
    except Exception:
        stale = None

    candidates = _candidates(folder, files)
    effective_question, corrected = _correct_typos(question, candidates)

    # Which file(s) is this question about?
    q_words = {w.strip(".,?!\"'").lower() for w in effective_question.split()}
    mentioned = _mentioned_files(effective_question, files)
    refers_back = (
        bool(history)
        and bool(q_words & REFERENCE_WORDS)
        and not (q_words & SEARCH_WORDS)
        and len(q_words) <= 14
    )
    previous_focus = [p for p in history[-1].get("focus", []) if p in files] if history else []

    focus = list(mentioned)
    if refers_back:
        for p in previous_focus:
            if p not in focus:
                focus.append(p)
    focus = focus[:3]

    if focus:
        # Read the named file(s) directly instead of guessing with search.
        per_file = 3 if len(focus) == 1 else 2
        docs, metas = [], []
        for p in focus:
            d, m = get_file_chunks(p, per_file)
            docs.extend(d)
            metas.extend(m)
    else:
        retrieval_text = effective_question
        if refers_back and history:
            retrieval_text = f"{history[-1]['question']} {effective_question}"
        results = search(get_embedding(retrieval_text), top_k=5, folder=folder)
        docs, metas = results["documents"][0], results["metadatas"][0]

    if not docs:
        raise CoreError("No relevant code found for that question.")

    pieces, used_metas, total = [], [], 0
    for meta, text in zip(metas, docs):
        if os.path.splitext(meta["file"])[1].lower() in C_STYLE_EXTENSIONS:
            text = _strip_commented_code(text)
        # Show a short path relative to the indexed folder, not the full C:\Users\... path.
        shown_name = os.path.relpath(meta["file"], folder)
        piece = f"File: {shown_name} (lines {meta['start_line']}-{meta['end_line']})\n{text}"
        if pieces and total + len(piece) > MAX_CONTEXT_CHARS:
            break
        pieces.append(piece[:MAX_CONTEXT_CHARS])
        used_metas.append(meta)
        total += len(piece)
    context = "\n\n---\n\n".join(pieces)
    sources = sorted({os.path.basename(m["file"]) for m in used_metas})

    recent = history[-3:]
    history_text = "\n\n".join(
        f"Q: {h['question']}\nA: {h['answer'][:MAX_HISTORY_ANSWER_CHARS]}" for h in recent
    )
    history_block = (
        f"\nRECENT CONVERSATION (for context on follow-up questions):\n{history_text}\n"
        if history_text else ""
    )
    focus_line = (
        f"The question is about: {', '.join(os.path.basename(p) for p in focus)}.\n" if focus else ""
    )

    wants_example = any(
        k in effective_question.lower()
        for k in ["example", "sample output", "sample run", "what would it print", "show output"]
    )
    if wants_example:
        example_instruction = (
            "The user is asking for an example, so you may include one — but ONLY if you can "
            "work it out correctly and precisely from the actual code logic shown. Double check "
            "your arithmetic/logic before including it."
        )
    else:
        example_instruction = (
            "Do NOT include any example input, output, sample run, or walkthrough with specific "
            "numbers. Describe what the code does in general terms only, with no invented examples."
        )

    prompt = f"""You are a helpful assistant answering questions about a codebase.
Use ONLY the code snippets below to answer the question.
If the answer isn't in the snippets, say so honestly. Do not guess about code that is not shown.
Mention the specific file name(s) naturally in your answer, as part of the sentence.
Refer to a file by the short name shown after "File:", never by a full drive path.
If the question refers back to something discussed earlier ("that", "it", "the one you mentioned"),
use the recent conversation below to understand what it refers to.
{focus_line}
Keep your answer SHORT: maximum 2 short paragraphs, 3-4 sentences total.
Do not use numbered lists, bullet points, or headers. Just plain, concise prose.
Focus only on what the question actually asks — do not describe every line of code.
Be precise and specific: mention actual variable names, exact loop bounds, and exact
conditions from the code, rather than vague general descriptions.
{example_instruction}
{history_block}
CODE SNIPPETS:
{context}

QUESTION: {effective_question}
"""

    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={"num_predict": 200, "stop": ["QUESTION:", "CODE SNIPPETS:", "RECENT CONVERSATION"]},
    )
    answer = response["response"].strip()
    answer = re.split(r"\n\s*QUESTION:", answer)[0].strip()   # safety net if the model leaks it anyway
    answer = _cut_leaked_tail(answer)
    answer = re.sub(r"^File:\s*", "", answer)                 # drop a stray "File:" copied from the prompt
    if not wants_example:
        answer = re.sub(r"```.*?```", "", answer, flags=re.DOTALL).strip()
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    answer = _trim_to_sentence(answer)
    answer = _limit_sentences(answer, MAX_SENTENCES_EXAMPLE if wants_example else MAX_SENTENCES)

    sessions.add_message(session_id, question, answer, focus=focus)
    return {
        "answer": answer,
        "interpreted_as": effective_question if corrected else None,
        "sources": sources,
        "stale": stale,
    }


def reset_all():
    """Deletes every index and every chat. Refuses while an indexing job is running."""
    with _JOBS_LOCK:
        if any(j["status"] == "running" for j in _JOBS.values()):
            raise CoreError("Indexing is still running. Wait for it to finish, then reset.")
    vectorstore.reset_index()
    sessions.clear_all()

def _same_stamp(stored, on_disk):
    """True if a stored (mtime, size) matches the file on disk. Times are compared with a
    tolerance of 1 ms because the stored float can differ in its last digits after
    going through the database (an exact == flagged untouched files as changed)."""
    s_time, s_size = stored
    d_time, d_size = on_disk
    if s_time is None or s_size is None:
        return False   # indexed before stamps existed
    return s_size == d_size and abs(s_time - d_time) < 0.001