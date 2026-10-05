import streamlit as st
from chunker import find_code_files, chunk_file
from embedder import get_embedding
from vectorstore import add_chunks, search, clear_folder, list_indexed_files
import sessions
import re
import ollama
import os
import difflib

st.set_page_config(page_title="Codebase Q&A", page_icon="💬")
st.title("Ask your codebase")


def check_ollama_running():
    try:
        ollama.list()
        return True
    except Exception:
        return False


if not check_ollama_running():
    st.error(
        "Can't connect to Ollama. Make sure Ollama is running "
        "(open a terminal and run `ollama serve`), then refresh this page."
    )
    st.stop()

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

with st.sidebar:
    st.header("Codebase Q&A")

    if st.button("➕ New codebase", use_container_width=True):
        st.session_state.current_session_id = None
        st.session_state.show_index_form = True
        st.rerun()

    known_folders = sessions.list_distinct_folders()
    if known_folders:
        chosen = st.selectbox("Start new chat on existing folder", ["-- choose --"] + known_folders)
        if chosen != "-- choose --" and st.button("Start new chat", use_container_width=True):
            new_id = sessions.create_session(chosen)
            st.session_state.current_session_id = new_id
            st.session_state.show_index_form = False
            st.rerun()

    st.divider()
    st.subheader("Your chats")
    past_sessions = sessions.list_sessions()
    for session_id, data in past_sessions:
        label = f"{data['folder'].split(chr(92))[-1]}"
        sublabel = data['created_at']
        if st.button(f"💬 {label}", key=f"open_{session_id}", use_container_width=True, help=data['folder']):
            st.session_state.current_session_id = session_id
            st.session_state.show_index_form = False
            st.rerun()

    st.divider()
    with st.expander("Advanced"):
        if st.button("Reset everything"):
            import shutil
            import vectorstore
            try:
                shutil.rmtree("./chroma_db", ignore_errors=True)
                if os.path.exists("chat_sessions.json"):
                    os.remove("chat_sessions.json")
                vectorstore.client = vectorstore.chromadb.PersistentClient(path="./chroma_db")
                vectorstore.collection = vectorstore.client.get_or_create_collection(name="codebase")
                st.session_state.current_session_id = None
                st.success("Everything cleared.")
            except Exception as e:
                st.error(f"Couldn't reset: {e}")

if "show_index_form" not in st.session_state:
    st.session_state.show_index_form = not sessions.list_sessions()

if st.session_state.show_index_form or not st.session_state.current_session_id:
    st.subheader("Index a new codebase")
    folder = st.text_input("Codebase folder path").strip().strip('"')

    if st.button("Index this codebase"):
        if not folder:
            st.warning("Please enter a folder path first.")
        elif not os.path.isdir(folder):
            st.error("That folder doesn't exist. Double-check the path and try again.")
        else:
            try:
                with st.spinner("Reading and understanding your codebase..."):
                    clear_folder(folder)
                    files = find_code_files(folder)

                    if not files:
                        st.warning("No supported code files found in that folder.")
                    else:
                        all_chunks = []
                        for file_path in files:
                            all_chunks.extend(chunk_file(file_path))

                        if not all_chunks:
                            st.warning("Files were found, but no readable content was extracted from them.")
                        else:
                            all_embeddings = []
                            valid_chunks = []
                            skipped_count = 0

                            for chunk in all_chunks:
                                try:
                                    embedding = get_embedding(chunk["text"])
                                    all_embeddings.append(embedding)
                                    valid_chunks.append(chunk)
                                except Exception:
                                    skipped_count += 1

                            all_chunks = valid_chunks

                            if not all_chunks:
                                st.error("No chunks could be processed.")
                            else:
                                add_chunks(all_chunks, all_embeddings)
                                new_id = sessions.create_session(folder)
                                st.session_state.current_session_id = new_id
                                st.session_state.show_index_form = False

                                msg = f"Indexed {len(all_chunks)} chunks from {len(files)} files."
                                if skipped_count > 0:
                                    msg += f" ({skipped_count} chunks skipped.)"
                                st.success(msg)
                                st.rerun()
            except Exception as e:
                st.error(f"Something went wrong while indexing: {e}")

if st.session_state.current_session_id:
    session = sessions.get_session(st.session_state.current_session_id)
    if session:
        current_folder = session["folder"]
        st.subheader(f"Chat: {current_folder}")

        for entry in session["messages"]:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                st.markdown(entry["answer"])

        question = st.text_input("Ask a question about this codebase", key="question_input")

        if st.button("Get answer") and question:
            try:
                with st.spinner("Thinking..."):
                    indexed_files = [f for f in list_indexed_files() if f.startswith(current_folder)]
                    candidates = set()
                    for path in indexed_files:
                        for part in path.replace("\\", "/").split("/"):
                            name = os.path.splitext(part)[0].lower()
                            if name:
                                candidates.add(name)

                    effective_question_words = []
                    corrected_any = False
                    for word in question.split():
                        clean_word = word.strip(".,?!").lower()
                        if len(clean_word) > 3 and clean_word not in candidates:
                            match = difflib.get_close_matches(clean_word, candidates, n=1, cutoff=0.75)
                            if match:
                                effective_question_words.append(match[0])
                                corrected_any = True
                                continue
                        effective_question_words.append(word)

                    effective_question = " ".join(effective_question_words) if corrected_any else question

                    if corrected_any:
                        st.caption(f"Interpreting this as: \"{effective_question}\"")

                    question_embedding = get_embedding(effective_question)
                    results = search(question_embedding, top_k=5, folder=current_folder)

                    if not results["documents"][0]:
                        st.warning("No relevant code found for that question.")
                    else:
                        context_pieces = []
                        for i in range(len(results["documents"][0])):
                            meta = results["metadatas"][0][i]
                            text = results["documents"][0][i]
                            context_pieces.append(
                                f"File: {meta['file']} (lines {meta['start_line']}-{meta['end_line']})\n{text}"
                            )
                        context = "\n\n---\n\n".join(context_pieces)

                        # Include the last 3 exchanges so follow-up questions
                        # ("what about X in that", "can you elaborate") work.
                        recent_history = session["messages"][-3:]
                        history_text = ""
                        if recent_history:
                            history_lines = []
                            for h in recent_history:
                                history_lines.append(f"Q: {h['question']}\nA: {h['answer']}")
                            history_text = "\n\n".join(history_lines)

                        wants_example = any(
                            keyword in effective_question.lower()
                            for keyword in ["example", "sample output", "sample run", "what would it print", "show output"]
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

                        history_block = (
                            f"\nRECENT CONVERSATION (for context on follow-up questions):\n{history_text}\n"
                            if history_text else ""
                        )

                        prompt = f"""You are a helpful assistant answering questions about a codebase.
Use ONLY the code snippets below to answer the question.
If the answer isn't in the snippets, say so honestly.
Mention the specific file name(s) naturally in your answer, as part of the sentence.
If the question refers back to something discussed earlier ("that", "it", "the one you mentioned"),
use the recent conversation below to understand what it refers to.

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
                            options={"num_predict": 200}
                        )

                        answer_text = response["response"].strip()
                        if not wants_example:
                            answer_text = re.sub(r"```.*?```", "", answer_text, flags=re.DOTALL).strip()

                        sessions.add_message(st.session_state.current_session_id, question, answer_text)
                        st.rerun()
            except Exception as e:
                st.error(f"Something went wrong while answering: {e}")