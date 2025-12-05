"""
File: query_rag.py 
Author: Derrick Chen
Date: 2025-12-04
Description:
  Query a simple Retrieval-Augmented Generation (RAG) index and return a
  strictly short (1–2 sentences) paraphrased answer.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline

INDEX_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "artifacts")
EMBEDDER_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"  # drop-in upsize possible

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int

def load_index(index_dir: str) -> Tuple[np.ndarray, List[Chunk]]:
    emb_path = os.path.join(index_dir, "embeddings.npy")
    meta_path = os.path.join(index_dir, "chunks.json")
    if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(
            f"Index not found in '{index_dir}'. Expected embeddings.npy and chunks.json"
        )

    embeddings = np.load(emb_path).astype(np.float32)
    with open(meta_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    chunks = [Chunk(text=i.get("text", ""), source=i.get("source", "unknown"), chunk_id=int(i.get("chunk_id", idx))) for idx, i in enumerate(raw)]

    return embeddings, chunks


def device_hint() -> str:
    return "cpu"

def retrieve(
    query: str,
    embedder: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: Sequence[Chunk],
    k: int = 2,
    min_similarity: float = 0.30,
) -> Tuple[List[Chunk], np.ndarray]:
    """Return a filtered top-k set and the full similarity vector."""
    query_vec = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_vec, embeddings)[0]

    # Top-k indices sorted by similarity (desc)
    top_idx = sims.argsort()[-max(1, k):][::-1]

    # If best match below threshold, return empty
    if sims[top_idx[0]] < min_similarity:
        return [], sims

    # Keep chunks within 90% of best and above absolute threshold
    best = sims[top_idx[0]]
    filtered_idx = [i for i in top_idx if sims[i] >= 0.9 * best and sims[i] >= min_similarity]
    return [chunks[i] for i in filtered_idx], sims


def build_prompt(context: str, query: str) -> str:
    return (
        "Use ONLY the context to answer. If the context doesn’t contain the answer, "
        "reply exactly: I don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer rules:\n"
        "• Write 1–2 sentences total (<=28 words each).\n"
        "• Paraphrase; do not copy 5+ consecutive words from the context.\n"
        "• Be specific and avoid hedging.\n\n"
        "Answer:"
    )


def truncate_sentences(text: str, n: int = 2, max_words: int = 28) -> str:
    # Split into sentences (simple, robust enough for short outputs)
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: List[str] = []
    for s in parts:
        words = s.strip().split()
        if not words:
            continue
        if len(words) > max_words:
            words = words[:max_words]
            if words[-1].endswith((".", "!", "?")):
                pass
            else:
                words[-1] = words[-1].rstrip(",;") + "."
        kept.append(" ".join(words))
        if len(kept) >= n:
            break
    # Guarantee 1 sentence minimum if model produced nothing usable
    if not kept:
        return "I don't know."
    return " " .join(kept)

def run_query(
    query: str,
    *,
    index_dir: str,
    k: int,
    min_similarity: float,
    max_words: int,
    style: str,
    show_context: bool,
    embedder_name: str,
    llm_name: str,
) -> str:
    print("Loading index...")
    embeddings, chunks = load_index(index_dir)

    print(f"Loading embedding model: {embedder_name}")
    embedder = SentenceTransformer(embedder_name)

    print(f"Loading LLM: {llm_name}")
    textgen = pipeline("text2text-generation", model=llm_name)

    print(f"Device set to use {device_hint()}")
    print("Retrieving relevant context...\n")

    k_eff = 1 if re.match(r"\s*what\s+is|\s*who\s+is", query, flags=re.I) else k

    top_chunks, sims = retrieve(query, embedder, embeddings, chunks, k=k_eff, min_similarity=min_similarity)

    if not top_chunks:
        context = ""
    else:
        context = "\n\n".join(c.text for c in top_chunks)

    print("Generating answer...\n")
    prompt = build_prompt(context, query)

    raw = textgen(prompt, max_new_tokens=128, num_beams=4, do_sample=False)[0]["generated_text"]

    answer = truncate_sentences(raw, n=2 if style == "short" else 3, max_words=max_words)

    print(f"Answer: {answer}\n")

    if show_context and top_chunks:
        print("--- Retrieved Context (for transparency) ---")
        for c in top_chunks:
            print(f"From {c.source} (chunk {c.chunk_id}):\n{c.text}\n")
    elif show_context:
        print("--- Retrieved Context (for transparency) ---\n<no sufficiently similar chunks>\n")

    return answer

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Query a simple RAG index with concise answers.")
    p.add_argument("--question", "-q", type=str, help="Your question to ask the RAG system.")
    p.add_argument("--index-dir", type=str, default=INDEX_DIR_DEFAULT, help="Directory with embeddings.npy and chunks.json")
    p.add_argument("--k", type=int, default=2, help="Top-k chunks to start from (will auto-tighten for definition-style prompts)")
    p.add_argument("--min-similarity", type=float, default=0.30, help="Absolute similarity threshold for accepting matches")
    p.add_argument("--max-words", type=int, default=28, help="Max words per sentence in final output")
    p.add_argument("--style", choices=["short", "medium"], default="short", help="Answer length style (short=1–2 sentences, medium=up to 3)")
    p.add_argument("--no-context", action="store_true", help="Do not print the retrieved context block")
    p.add_argument("--embedder-name", type=str, default=EMBEDDER_DEFAULT, help="SentenceTransformer model name")
    p.add_argument("--llm-name", type=str, default=LLM_MODEL_NAME, help="HF text2text model name")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    # Interactive fallback if no --question provided
    if not args.question:
        print("Ready. Type your question (or 'exit' to quit).\n")
        while True:
            try:
                q = input("Question: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not q or q.lower() in {"exit", "quit"}:
                break
            try:
                run_query(
                    q,
                    index_dir=args.index_dir,
                    k=args.k,
                    min_similarity=args.min_similarity,
                    max_words=args.max_words,
                    style=args.style,
                    show_context=not args.no_context,
                    embedder_name=args.embedder_name,
                    llm_name=args.llm_name,
                )
            except Exception as e:
                print(f"Error: {e}")
        return

    # One-shot mode
    try:
        run_query(
            args.question,
            index_dir=args.index_dir,
            k=args.k,
            min_similarity=args.min_similarity,
            max_words=args.max_words,
            style=args.style,
            show_context=not args.no_context,
            embedder_name=args.embedder_name,
            llm_name=args.llm_name,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
