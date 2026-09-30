from chunker import find_code_files, chunk_file
from embedder import get_embedding
from vectorstore import add_chunks, search
import ollama


def clean_input(prompt_text):
    """
    Gets user input and strips extra whitespace + surrounding quotes,
    so it doesn't matter if they paste a path with or without quotes.
    """
    return input(prompt_text).strip().strip('"')


folder = clean_input("Enter the full path to the codebase folder: ")

# Step 1-2: find files and chunk them
files = find_code_files(folder)
all_chunks = []
for file_path in files:
    all_chunks.extend(chunk_file(file_path))

print(f"Found {len(all_chunks)} chunks. Generating embeddings (this may take a bit)...")

# Step 3: embed every chunk
all_embeddings = []
for chunk in all_chunks:
    embedding = get_embedding(chunk["text"])
    all_embeddings.append(embedding)

# Step 4: store everything in ChromaDB
add_chunks(all_chunks, all_embeddings)
print("All chunks stored in the database.")

# Step 5: ask a question and get a real AI answer
question = clean_input("\nAsk a question about this codebase: ")
question_embedding = get_embedding(question)

results = search(question_embedding, top_k=3)

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

Write 2-4 short paragraphs, not a long list of individual bullet points.
Group related ideas together into the same paragraph (e.g. all the setup/initialization
details in one paragraph, the main loop logic in another, the overall purpose in a final
paragraph). Do not use headers or labels like "Answer:".

CODE SNIPPETS:
{context}

QUESTION: {question}
"""

response = ollama.generate(model="phi3", prompt=prompt)

print()
print(response["response"].strip())