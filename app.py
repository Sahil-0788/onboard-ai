import streamlit as st
from chunker import find_code_files, chunk_file
from embedder import get_embedding
from vectorstore import add_chunks, search, clear_folder
import re
import ollama
import os

st.set_page_config(page_title="Codebase Q&A", page_icon="💬")
st.title("Ask your codebase")

if "indexed" not in st.session_state:
    st.session_state.indexed = False


def check_ollama_running():
    """Returns True if Ollama is reachable, False otherwise."""
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
    st.stop()  # halts the app here, nothing below runs

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
                        for chunk in all_chunks:
                            all_embeddings.append(get_embedding(chunk["text"]))

                        add_chunks(all_chunks, all_embeddings)
                        st.session_state.indexed = True
                        st.success(f"Indexed {len(all_chunks)} chunks from {len(files)} files.")
        except Exception as e:
            st.error(f"Something went wrong while indexing: {e}")

if st.session_state.indexed:
    question = st.text_input("Ask a question about this codebase")

    if st.button("Get answer") and question:
        try:
            with st.spinner("Thinking..."):
                question_embedding = get_embedding(question)
                results = search(question_embedding, top_k=5)

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

                    wants_example = any(
                        keyword in question.lower()
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

                    prompt = f"""You are a helpful assistant answering questions about a codebase.
Use ONLY the code snippets below to answer the question.
If the answer isn't in the snippets, say so honestly.
Mention the specific file name(s) naturally in your answer, as part of the sentence.

Keep your answer SHORT: maximum 2 short paragraphs, 3-4 sentences total.
Do not use numbered lists, bullet points, or headers. Just plain, concise prose.
Focus only on what the question actually asks — do not describe every line of code.
Be precise and specific: mention actual variable names, exact loop bounds, and exact
conditions from the code, rather than vague general descriptions.
{example_instruction}

CODE SNIPPETS:
{context}

QUESTION: {question}
"""
                    response = ollama.generate(
                        model="phi3",
                        prompt=prompt,
                        options={"num_predict": 200}
                    )

                    answer_text = response["response"].strip()
                    if not wants_example:
                        answer_text = re.sub(r"```.*?```", "", answer_text, flags=re.DOTALL).strip()

                    st.markdown(answer_text)
        except Exception as e:
            st.error(f"Something went wrong while answering: {e}")