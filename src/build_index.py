"""
File: build_index.py
Author: Derrick Chen
Date: 2025-12-04
Description:
Builds an index of document chunks and their embeddings for retrieval.
"""
import json
import os
from typing import Dict, List

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents(data_dir: str) -> List[Dict]:
    """Read UTF-8 text documents in filename order."""
    docs = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith(".txt"):
            path = os.path.join(data_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append({"filename": fname, "text": text})
    return docs


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 100) -> List[str]:
    """Split text into word chunks with overlap between successive chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap
    return chunks


def main():
    """Embed corpus chunks and save their vectors and metadata."""
    import torch
    from sentence_transformers import SentenceTransformer

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading documents from {DATA_DIR}...")
    docs = load_documents(DATA_DIR)
    if not docs:
        raise RuntimeError("No .txt files found in data/ directory.")

    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "doc": doc["filename"],
                    "chunk_id": i,
                    "text": chunk,
                }
            )

    print(f"Total chunks: {len(all_chunks)}")

    texts = [c["text"] for c in all_chunks]

    print(f"Loading embedding model: {MODEL_NAME}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)

    print("Encoding chunks...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    embeddings_path = os.path.join(OUTPUT_DIR, "embeddings.npy")
    chunks_path = os.path.join(OUTPUT_DIR, "chunks.json")

    np.save(embeddings_path, embeddings)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved embeddings to {embeddings_path}")
    print(f"Saved chunk metadata to {chunks_path}")
    print("Index build complete.")


if __name__ == "__main__":
    main()
