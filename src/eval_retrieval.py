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
    parser.add_argument("--min-similarity", type=float, default=0.25)
    args = parser.parse_args()

    with args.questions.open(encoding="utf-8") as file:
        examples = [json.loads(line) for line in file if line.strip()]
    if not examples:
        parser.error("questions file is empty")

    embeddings, chunks = load_index(args.index_dir)
    embedder = load_embedder(args.embedder_name, device_hint())
    hits = {1: 0, 3: 0}
    answerable = 0
    abstained = 0
    for example in examples:
        top, _ = retrieve(
            example["question"], embedder, embeddings, chunks,
            k=max(hits), min_similarity=args.min_similarity,
        )
        if example["source"] is None:
            abstained += int(not top)
            continue
        answerable += 1
        for k in hits:
            hits[k] += any(chunk.source == example["source"] for chunk in top[:k])
    if answerable:
        for k, count in hits.items():
            print(f"hit@{k}: {count}/{answerable} ({count / answerable:.1%})")
    unanswerable = len(examples) - answerable
    if unanswerable:
        print(f"correct abstentions: {abstained}/{unanswerable} ({abstained / unanswerable:.1%})")


if __name__ == "__main__":
    main()
