import os
import re
import difflib
import shutil
import ollama

import sessions
import vectorstore
from chunker import find_code_files, chunk_file
from embedder import get_embedding
from vectorstore import add_chunks, search, clear_folder, list_indexed_files, in_folder, get_file_chunks

# Words that suggest a question refers back to the previous answer ("what is X used for in that").
REFERENCE_WORDS = {"that", "it", "those", "these", "them", "they", "above", "earlier", "previous"}
# Words that suggest a broad search question instead ("which file handles ...").
SEARCH_WORDS = {"file", "files", "folder", "project", "codebase", "which", "where", "any"}

MAX_CONTEXT_CHARS = 6500          # keeps the prompt inside phi3's 4096-token window
MAX_HISTORY_ANSWER_CHARS = 500


class CoreError(Exception):
    """An error with a message that is safe to show to the user."""


def ollama_status():
    try:
        resp = ollama.list()
        names = [(m.model or "") for m in resp.models]
    except Exception:
        return {"ollama_running": False, "phi3": False, "embedding_model": False}
    return {
        "ollama_running": True,
        "phi3": any(n.startswith("phi3") for n in names),
        "embedding_model": any(n.startswith("nomic-embed-text") for n in names),
    }


def ollama_running():
    return ollama_status()["ollama_running"]


def index_folder(folder):
    folder = folder.strip().strip('"')
    if not folder:
        raise CoreError("Please enter a folder path first.")
    if not os.path.isdir(folder):
        raise CoreError("That folder doesn't exist. Double-check the path and try again.")
    folder = os.path.normpath(folder)

    files = find_code_files(folder)
    if not files:
        raise CoreError("No supported code files found in that folder.")

    all_chunks = []
    for file_path in files:
        all_chunks.extend(chunk_file(file_path))
    if not all_chunks:
        raise CoreError("Files were found, but no readable content was extracted from them.")

    valid_chunks, embeddings, skipped = [], [], 0
    for chunk in all_chunks:
        try:
            embeddings.append(get_embedding(chunk["text"]))
            valid_chunks.append(chunk)
        except Exception:
            skipped += 1
    if not valid_chunks:
        raise CoreError("No chunks could be processed. The files may be too large or unusual to embed.")

    # Only now remove the old data for this folder, so a failed run never wipes a working index.
    clear_folder(folder)
    add_chunks(valid_chunks, embeddings)
    session_id = sessions.create_session(folder)

    return {
        "session_id": session_id,
        "folder": folder,
        "files": len(files),
        "chunks": len(valid_chunks),
        "skipped": skipped,
    }


def start_session(folder):
    folder = os.path.normpath(folder.strip().strip('"'))
    if folder not in sessions.list_distinct_folders():
        raise CoreError("That folder hasn't been indexed yet.")
    return sessions.create_session(folder)


def _folder_files(folder):
    return sorted(p for p in list_indexed_files() if in_folder(p, folder))


def _candidates(folder, files):
    """Names a user might type: file names (with and without extension) and the folder's own name."""
    names = {os.path.basename(folder).lower()}
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


def _trim_to_sentence(text):
    """If the answer was cut off by the token limit, drop the unfinished last sentence."""
    if not text or text[-1] in ".!?":
        return text
    cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "), text.rfind(".\n"))
    if cut > len(text) * 0.4:
        return text[: cut + 1].strip()
    return text


def answer_question(session_id, question):
    question = question.strip()
    if not question:
        raise CoreError("Please type a question first.")

    session = sessions.get_session(session_id)
    if not session:
        raise CoreError("That chat no longer exists.")
    folder = session["folder"]
    history = session["messages"]

    files = _folder_files(folder)
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
        piece = f"File: {meta['file']} (lines {meta['start_line']}-{meta['end_line']})\n{text}"
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
        model="phi3",
        prompt=prompt,
        options={"num_predict": 200, "stop": ["QUESTION:", "CODE SNIPPETS:", "RECENT CONVERSATION"]},
    )
    answer = response["response"].strip()
    answer = re.split(r"\n\s*QUESTION:", answer)[0].strip()   # safety net if the model leaks it anyway
    if not wants_example:
        answer = re.sub(r"```.*?```", "", answer, flags=re.DOTALL).strip()
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    answer = _trim_to_sentence(answer)

    sessions.add_message(session_id, question, answer, focus=focus)
    return {
        "answer": answer,
        "interpreted_as": effective_question if corrected else None,
        "sources": sources,
    }


def reset_all():
    shutil.rmtree("./chroma_db", ignore_errors=True)
    if os.path.exists(sessions.SESSIONS_FILE):
        os.remove(sessions.SESSIONS_FILE)
    vectorstore.client = vectorstore.chromadb.PersistentClient(path="./chroma_db")
    vectorstore.collection = vectorstore.client.get_or_create_collection(name="codebase")