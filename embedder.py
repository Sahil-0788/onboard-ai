import ollama

EMBEDDING_MODEL = "nomic-embed-text"


def get_embedding(text):
    """Embeds one piece of text (used for questions)."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    return response["embeddings"][0]


def get_embeddings(texts):
    """Embeds many texts in one call. Returns a list of vectors in the same order."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
    return response["embeddings"]