import ollama

def get_embedding(text):
    """
    Sends a piece of text to the local embedding model (nomic-embed-text)
    running via Ollama, and gets back a list of numbers representing
    that text's meaning.
    """
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


if __name__ == "__main__":
    # Quick test — embed two related sentences and one unrelated one,
    # so we can confirm similar meaning = similar numbers
    text1 = "function to add two numbers"
    text2 = "method that sums two integers"
    text3 = "sorting an array using bubble sort"

    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    emb3 = get_embedding(text3)

    print(f"Embedding length: {len(emb1)} numbers")
    print(f"First 5 numbers of embedding 1: {emb1[:5]}")