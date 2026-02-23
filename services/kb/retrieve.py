"""
This module retrieves relevant KB chunks for a given query.

It is used at runtime when a ticket arrives.
The agent calls this to fetch relevant knowledge.
"""

import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


# ---------------------------------------------------------
# Connect to existing Chroma collection
# ---------------------------------------------------------
def get_collection():
    """
    Connects to the persistent Chroma collection.
    Returns the collection object.
    """

    load_dotenv()

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")

    client = chromadb.PersistentClient(path=persist_dir)

    embedding_function = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )

    collection = client.get_or_create_collection(
        name="support_kb",
        embedding_function=embedding_function
    )

    return collection


# ---------------------------------------------------------
# Retrieve relevant documents
# ---------------------------------------------------------
def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Performs semantic search.

    Parameters:
        query: The user ticket text
        top_k: Number of results to return

    Returns:
        List of relevant KB chunks
    """

    collection = get_collection()

    # Perform vector similarity search
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    output = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        output.append({
            "text": doc,
            "source": meta.get("source"),
            "chunk_id": meta.get("chunk_id"),
            "distance": dist
        })

    return output


# Test retrieval
if __name__ == "__main__":
    query = "My delivery is delayed and I want refund."
    hits = retrieve(query)

    for i, hit in enumerate(hits, 1):
        print(f"\n--- Result {i} ---")
        print("Source:", hit["source"])
        print(hit["text"])
