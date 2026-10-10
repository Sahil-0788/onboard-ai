import os
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="codebase")


def in_folder(file_path, folder):
    """True if file_path is the folder itself or sits inside it (not just a name prefix)."""
    folder = folder.rstrip("\\/")
    return file_path == folder or file_path.startswith(folder + os.sep)


def add_chunks(chunks, embeddings):
    ids, documents, metadatas = [], [], []
    for chunk in chunks:
        unique_id = f"{chunk['file']}::{chunk['start_line']}-{chunk['end_line']}"
        ids.append(unique_id)
        documents.append(chunk["text"])
        metadatas.append({
            "file": chunk["file"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
        })
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def clear_folder(folder_path):
    existing = collection.get()
    ids_to_delete = []
    for i, metadata in enumerate(existing["metadatas"]):
        if in_folder(metadata["file"], folder_path):
            ids_to_delete.append(existing["ids"][i])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)


def list_indexed_files():
    try:
        existing = collection.get()
        return set(m["file"] for m in existing["metadatas"])
    except Exception:
        return set()


def get_file_chunks(file_path, limit=3):
    """Returns (documents, metadatas) for the first `limit` chunks of one file, in line order."""
    try:
        res = collection.get(where={"file": file_path})
    except Exception:
        return [], []
    pairs = sorted(zip(res["metadatas"], res["documents"]), key=lambda p: p[0]["start_line"])[:limit]
    return [p[1] for p in pairs], [p[0] for p in pairs]


def search(query_embedding, top_k=3, folder=None):
    if folder:
        raw = collection.query(query_embeddings=[query_embedding], n_results=top_k * 5)
        filtered_docs, filtered_metas = [], []
        for doc, meta in zip(raw["documents"][0], raw["metadatas"][0]):
            if in_folder(meta["file"], folder):
                filtered_docs.append(doc)
                filtered_metas.append(meta)
            if len(filtered_docs) >= top_k:
                break
        return {"documents": [filtered_docs], "metadatas": [filtered_metas]}
    return collection.query(query_embeddings=[query_embedding], n_results=top_k)