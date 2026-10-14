import os
import time

import core
import sessions

FOLDER = os.path.normpath(r"C:\Users\Sahil Uday Bhat\Desktop\DSA CPP")
MODELS = ["phi3", "qwen2.5-coder:3b"]

# Asked in this order, so the follow-ups have something to refer back to.
QUESTIONS = [
    "explain pattern3.cpp",                                   # one file
    "what is the variable ch used for in that",               # follow-up
    "how is it different from pattern5.cpp",                  # comparison
    "which file prints a pyramid",                            # broad search
    "explain pattrn3.cpp",                                    # typo
    "does pattern3.cpp use recursion",                        # not present
    "give an example output of pattern3.cpp when n is 3",     # example (correct: A B C / D E F / G H I)
    "what does pattern10-pyramid.cpp do",                     # another single file
]

if not core.ollama_running():
    raise SystemExit("Ollama is not running. Start `ollama serve` first.")
if FOLDER not in sessions.list_distinct_folders():
    raise SystemExit("DSA CPP has no chat yet. Index it first with POST /index.")

created = []
lines = []


def out(text=""):
    print(text)
    lines.append(text)


try:
    for model in MODELS:
        out("=" * 70)
        out(f"MODEL: {model}")
        out("=" * 70)
        sid = sessions.create_session(FOLDER)   # fresh chat per model so histories don't mix
        created.append(sid)
        for q in QUESTIONS:
            start = time.time()
            try:
                r = core.answer_question(sid, q, model=model)
                body = r["answer"]
                note = f"sources: {r['sources']}  interpreted_as: {r['interpreted_as']}"
            except Exception as e:
                body, note = f"ERROR: {e}", ""
            secs = time.time() - start
            out(f"\nQ: {q}   [{secs:.1f}s]")
            out(f"A: {body}")
            if note:
                out(note)
        out()
finally:
    # Remove the throwaway chats so they don't clutter your chat list.
    data = sessions.load_sessions()
    for sid in created:
        data.pop(sid, None)
    sessions.save_sessions(data)

with open("compare_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nSaved to compare_results.txt")