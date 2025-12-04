# mini-rag — Minimal RAG demo

A compact Retrieval-Augmented Generation (RAG) example that embeds local `.txt` files and answers questions using a small instruction-tuned LLM.

## Quickstart

Prerequisites: Python 3.8+, internet access (models are downloaded on first run).

1. Create and activate a virtual environment (PowerShell):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Build the retrieval index (creates `artifacts/`):

```
python -m src.build_index
```

3. Run the interactive RAG query loop:

```
python -m src.query_rag
```

## Example

Query:

```
What is the purpose of this repository?
```

Expected (example) answer:

```
This repository demonstrates a minimal RAG pipeline: it embeds local text documents, retrieves relevant chunks for a user query, and uses a small instruction-tuned LLM to generate a concise answer based on the retrieved context.
```

## Files

- `data/` — source `.txt` documents used to build the index.
- `artifacts/` — generated `embeddings.npy` and `chunks.json` (output of `build_index`).
- `src/build_index.py` — builds document chunks and embeddings.
- `src/query_rag.py` — interactive retrieval + generation loop.
- `requirements.txt` — Python dependencies.

## Notes

- If `data/` contains no `.txt` files, `build_index` will raise an error.
- Models are downloaded automatically (first run may take time and require internet).
- Activate the virtual environment with the appropriate command for your shell.

Minimal. Reproducible. Ready for local experimentation.

## Project structure

```
mini-rag/
├── data/
│   ├── sample_1.txt
│   └── sample_2.txt
├── artifacts/
│   ├── embeddings.npy
│   └── chunks.json
├── src/
│   ├── build_index.py
│   └── query_rag.py
├── requirements.txt
└── README.md
```

Files shown are current as of this repository snapshot. The `.git/` directory is present but omitted above.
