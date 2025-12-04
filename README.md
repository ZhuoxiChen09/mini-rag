# LLM RAG Demo — Retrieval-Augmented Question Answering

This project is a minimal **Retrieval-Augmented Generation (RAG)** demo using:

- `sentence-transformers` for semantic search  
- A small instruction-tuned LLM (`google/flan-t5-base`) for answer generation  

It builds an embedding index over local `.txt` documents and lets you ask natural language questions, retrieving relevant chunks and generating an answer based on them.

---

## Features

- Embeds local `.txt` files using `all-MiniLM-L6-v2`
- Chunks long documents into overlapping segments
- Performs cosine-similarity-based retrieval
- Uses a small open-source LLM to answer questions using retrieved context
- Runs fully locally (no external API keys required)

---

## Structure

```text
llm-rag/
│
├── data/                 # Source documents (.txt)
# mini-rag — Minimal RAG demo

A compact Retrieval-Augmented Generation (RAG) example that embeds local `.txt` files and answers questions using a small instruction-tuned LLM.

## Quickstart

Prerequisites: Python 3.8+, internet access (models are downloaded on first run).

1. Create and activate a virtual environment (PowerShell):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirement.txt
```

2. Build the retrieval index (creates `artifacts/`):

```
python -m src.build_index
```

3. Run the interactive RAG query loop:

```
python -m src.query_rag
```

## Files

- `data/` — place source `.txt` documents here.
- `artifacts/` — generated `embeddings.npy` and `chunks.json` (output of the index build).
- `src/build_index.py` — builds document chunks and embeddings.
- `src/query_rag.py` — interactive retrieval + generation loop.
- `requirement.txt` — Python dependencies.

## Notes

- If `data/` contains no `.txt` files, `build_index` will raise an error.
- Models are downloaded automatically (first run may take time and require internet).
- For other shells, activate the virtual environment with the appropriate command.

Minimal. Reproducible. Ready for local experimentation.
