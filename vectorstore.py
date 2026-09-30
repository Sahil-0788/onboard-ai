import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="codebase")


def add_chunks(chunks, embeddings):
    """
    Saves a batch of chunks into ChromaDB, along with their embeddings.
    """
    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        ids.append(str(i))
        documents.append(chunk["text"])
        metadatas.append({
            "file": chunk["file"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def clear_folder(folder_path):
    """
    Deletes all previously stored chunks that came from this specific folder,
    so re-indexing the same folder doesn't create duplicates.
    """
    existing = collection.get()

    ids_to_delete = []
    for i, metadata in enumerate(existing["metadatas"]):
        if metadata["file"].startswith(folder_path):
            ids_to_delete.append(existing["ids"][i])

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)


def search(query_embedding, top_k=3):
    """
    Given an embedding (e.g. of a question), finds the top_k
    most similar chunks stored in the database.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return results