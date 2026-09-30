import streamlit as st
from chunker import find_code_files, chunk_file
from embedder import get_embedding
from vectorstore import add_chunks, search, clear_folder
import ollama

st.set_page_config(page_title="Codebase Q&A", page_icon="💬")
st.title("Ask your codebase")

if "indexed" not in st.session_state:
    st.session_state.indexed = False

folder = st.text_input("Codebase folder path").strip().strip('"')

if st.button("Index this codebase"):
    if not folder:
        st.warning("Please enter a folder path first.")
    else:
        with st.spinner("Reading and understanding your codebase..."):
            clear_folder(folder)
            files = find_code_files(folder)
            all_chunks = []
            for file_path in files:
                all_chunks.extend(chunk_file(file_path))

            all_embeddings = []
            for chunk in all_chunks:
                all_embeddings.append(get_embedding(chunk["text"]))

            add_chunks(all_chunks, all_embeddings)
            st.session_state.indexed = True

        st.success(f"Indexed {len(all_chunks)} chunks from {len(files)} files.")

if st.session_state.indexed:
    question = st.text_input("Ask a question about this codebase")

    if st.button("Get answer") and question:
        with st.spinner("Thinking..."):
            question_embedding = get_embedding(question)
            results = search(question_embedding, top_k=5)

            context_pieces = []
            for i in range(len(results["documents"][0])):
                meta = results["metadatas"][0][i]
                text = results["documents"][0][i]
                context_pieces.append(
                    f"File: {meta['file']} (lines {meta['start_line']}-{meta['end_line']})\n{text}"
                )
            context = "\n\n---\n\n".join(context_pieces)

            prompt = f"""You are a helpful assistant answering questions about a codebase.
Use ONLY the code snippets below to answer the question.
If the answer isn't in the snippets, say so honestly.
Mention the specific file name(s) naturally in your answer, as part of the sentence.

Keep your answer SHORT: maximum 2 short paragraphs, 3-4 sentences total.
Do not use numbered lists, bullet points, or headers. Just plain, concise prose.
Focus only on what the question actually asks — do not describe every line of code.
Do NOT include any example input, output, sample run, or walkthrough with specific
numbers. Describe what the code does in general terms only, with no invented examples.

CODE SNIPPETS:
{context}

QUESTION: {question}
"""
            response = ollama.generate(model="phi3", prompt=prompt)

        st.markdown(response["response"].strip())