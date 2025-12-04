"""
File: query_rag.py
Author: Derrick Chen
Date: 2025-12-04
Description:
Queries a retrieval-augmented generation (RAG) system using a built index.
"""
import os
import json
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")

EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR, "embeddings.npy")
CHUNKS_PATH = os.path.join(ARTIFACTS_DIR, "chunks.json")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"  # small instruction-following model


def load_index():
    """
    Docstring for load_index
    Loads the embeddings and chunk metadata from the artifacts directory.
    """
    if not (os.path.exists(EMBEDDINGS_PATH) and os.path.exists(CHUNKS_PATH)):
        raise RuntimeError("Index not found. Run build_index.py first.")
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return embeddings, chunks


def retrieve_top_k(
    query: str,
    embedder: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: List[dict],
    k: int = 3,
):
    """
    Retrieve the top-k most similar chunks to the query.
    :param query: The user query string
    :param embedder: The embedding model
    :param embeddings: The array of chunk embeddings
    :param chunks: The list of chunk metadata
    :param k: Number of top chunks to retrieve
    """
    query_vec = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_vec, embeddings)[0]
    top_idx = sims.argsort()[-k:][::-1]
    top_chunks = [chunks[i] for i in top_idx]
    return top_chunks


def build_context(chunks: List[dict], max_chars: int = 1500) -> str:
    """
    Concatenate retrieved chunks into a context string with a rough char limit.
    :param chunks: List of chunk metadata
    :param max_chars: Maximum number of characters in the context
    """
    pieces = []
    total = 0
    for c in chunks:
        t = c["text"].strip()
        if total + len(t) > max_chars:
            break
        pieces.append(f"From {c['doc']} (chunk {c['chunk_id']}):\n{t}")
        total += len(t)
    return "\n\n".join(pieces)


def main():
    """
    Docstring for main
    Main function to query the RAG system
    1. Load the index (embeddings and chunks)
    2. Load the embedding model and LLM
    3. Prompt user for a question
    4. Retrieve relevant chunks
    5. Build context and generate answer using LLM
    6. Display answer and retrieved context
    """
    print("Loading index...")
    embeddings, chunks = load_index()

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"Loading LLM: {LLM_MODEL_NAME}")
    qa_pipeline = pipeline("text2text-generation", model=LLM_MODEL_NAME)

    print("Ready. Type your question (or 'exit' to quit).")
    while True:
        query = input("\nQuestion: ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            break

        print("Retrieving relevant context...")
        top_chunks = retrieve_top_k(query, embedder, embeddings, chunks, k=3)
        context = build_context(top_chunks)

        prompt = (
            "You are an AI assistant. Use the context below to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

        print("\nGenerating answer...\n")
        result = qa_pipeline(prompt, max_length=256, do_sample=False)[0]["generated_text"]
        print("Answer:", result)

        print("\n--- Retrieved Context (for transparency) ---")
        print(context)


if __name__ == "__main__":
    main()
