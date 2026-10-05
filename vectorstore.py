import chromadb
import os

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
        unique_id = f"{chunk['file']}::{chunk['start_line']}-{chunk['end_line']}"
        ids.append(unique_id)
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


def search(query_embedding, top_k=3, folder=None):
    if folder:
        raw = collection.query(query_embeddings=[query_embedding], n_results=top_k * 5)
        filtered_docs, filtered_metas = [], []
        for doc, meta in zip(raw["documents"][0], raw["metadatas"][0]):
            if meta["file"].startswith(folder):
                filtered_docs.append(doc)
                filtered_metas.append(meta)
            if len(filtered_docs) >= top_k:
                break
        return {"documents": [filtered_docs], "metadatas": [filtered_metas]}
    else:
        return collection.query(query_embeddings=[query_embedding], n_results=top_k)

def list_indexed_files():
    """Returns the set of all file paths currently stored in the database."""
    try:
        existing = collection.get()
        return set(m["file"] for m in existing["metadatas"])
    except Exception:
        return set()

def list_indexed_folders():
    """
    Returns the set of distinct top-level folders currently indexed,
    derived from the stored file paths.
    """
    files = list_indexed_files()
    folders = set()
    for f in files:
        # Walk up from the file to find a reasonable "project folder" guess:
        # take everything except the filename itself.
        folder = os.path.dirname(f)
        folders.add(folder)
    return folders