"""Evaluate source retrieval against the checked-in corpus and vector index."""

import argparse
import json
from pathlib import Path

from src.query_rag import (
    EMBEDDER_DEFAULT,
    INDEX_DIR_DEFAULT,
    device_hint,
    load_embedder,
    load_index,
    retrieve,
)

QUESTIONS_DEFAULT = Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_DEFAULT)
    parser.add_argument("--index-dir", default=INDEX_DIR_DEFAULT)
    parser.add_argument("--embedder-name", default=EMBEDDER_DEFAULT)
    args = parser.parse_args()

    with args.questions.open(encoding="utf-8") as file:
        examples = [json.loads(line) for line in file if line.strip()]
    if not examples:
        parser.error("questions file is empty")

    embeddings, chunks = load_index(args.index_dir)
    embedder = load_embedder(args.embedder_name, device_hint())
    hits = {1: 0, 3: 0}
    for example in examples:
        top, _ = retrieve(example["question"], embedder, embeddings, chunks, k=max(hits))
        for k in hits:
            hits[k] += any(chunk.source == example["source"] for chunk in top[:k])
    for k, count in hits.items():
        print(f"hit@{k}: {count}/{len(examples)} ({count / len(examples):.1%})")


if __name__ == "__main__":
    main()
