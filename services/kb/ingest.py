"""
This script builds (or rebuilds) the Knowledge Base vector store.

It:
1. Reads documents from kb_docs/
2. Splits them into smaller chunks
3. Converts chunks into embeddings (vectors)
4. Stores them inside Chroma (vector database)

You run this script when:
- You first set up the project
- You update KB documents
"""

# Standard library imports
import os
from pathlib import Path

# Load environment variables from .env
from dotenv import load_dotenv

# LangChain components for embeddings + chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Chroma database
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


# ---------------------------------------------------------
# STEP 1: Load all KB documents from kb_docs folder
# ---------------------------------------------------------
def load_kb_docs(kb_dir: str) -> list[Document]:
    """
    Reads all .md and .txt files from the given directory.

    Returns:
        A list of LangChain Document objects.
        Each Document contains:
            - page_content (the actual text)
            - metadata (like file name)
    """

    docs = []
    base = Path(kb_dir)

    for file_path in base.glob("**/*"):
        # Skip directories
        if file_path.is_dir():
            continue

        # Only allow markdown and text files
        if file_path.suffix.lower() not in [".md", ".txt"]:
            continue

        # Read file content
        text = file_path.read_text(encoding="utf-8", errors="ignore")

        # Store as Document object
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name  # Keep track of source file
                }
            )
        )

    return docs


# ---------------------------------------------------------
# STEP 2: Split documents into smaller chunks
# ---------------------------------------------------------
def chunk_docs(docs: list[Document]) -> list[Document]:
    """
    Splits large documents into smaller pieces.

    Why?
    - Improves retrieval accuracy
    - Prevents large embeddings
    - Allows fine-grained search

    We use overlap to preserve context between chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # Each chunk ~800 characters
        chunk_overlap=120    # 120 character overlap between chunks
    )

    chunked_docs = []

    for doc in docs:
        chunks = splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):
            chunked_docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i  # Track chunk index
                    }
                )
            )

    return chunked_docs


# ---------------------------------------------------------
# STEP 3: Store chunks into Chroma vector DB
# ---------------------------------------------------------
def ingest_to_chroma(
    docs: list[Document],
    persist_dir: str,
    collection_name: str = "support_kb",
):
    """
    Converts text chunks into embeddings and stores them in Chroma.

    Parameters:
        docs: List of chunked Document objects
        persist_dir: Directory where Chroma stores data
        collection_name: Name of vector collection
    """

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")

    # Ensure directory exists
    os.makedirs(persist_dir, exist_ok=True)

    # Create persistent Chroma client
    client = chromadb.PersistentClient(path=persist_dir)

    # Create embedding function (OpenAI model)
    embedding_function = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )

    # Delete old collection (MVP simplicity)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    # Prepare lists for bulk insert
    ids = []
    texts = []
    metadatas = []

    for idx, doc in enumerate(docs):
        ids.append(f"kb_{idx}")
        texts.append(doc.page_content)
        metadatas.append(doc.metadata)

    # Store everything in Chroma
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas
    )

    print(f"Ingested {len(docs)} chunks into collection '{collection_name}'")


# ---------------------------------------------------------
# Entry point (runs if script executed directly)
# ---------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()

    kb_dir = "./kb_docs"
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

    raw_docs = load_kb_docs(kb_dir)
    chunked_docs = chunk_docs(raw_docs)

    ingest_to_chroma(chunked_docs, persist_dir)
