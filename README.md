# mini-rag

A compact Retrieval-Augmented Generation (RAG) demonstration that embeds local text files and answers questions using instruction-tuned language models.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Internet access (models are downloaded on first run)

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

## Example Usage

Query:
```
What is the purpose of this repository?
```

Expected Output:
```
This repository demonstrates a minimal RAG pipeline: it embeds local text documents, retrieves relevant chunks for a user query, and uses a small instruction-tuned LLM to generate a concise answer based on the retrieved context.
```

## Retrieval & Answering Behavior

- The retriever uses cosine similarity over chunk embeddings. A minimum similarity threshold is applied (default: `0.15`) — if no chunk meets this threshold the system will respond with `I don't know` to avoid hallucination.
- The LLM generation is constrained for brevity: answers are post-processed to produce 1–2 short sentences (word-truncated if necessary) to keep responses concise and easier to read.
- You can adjust the threshold and output behavior by editing `src/query_rag.py` (see `retrieve_top_k(..., min_similarity=...)` and the `generate_answer` post-processing logic).

## Project Structure

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

## File Descriptions

- `data/` — Source text documents used to build the retrieval index
- `artifacts/` — Generated embeddings and chunk metadata
- `src/build_index.py` — Builds document chunks and embeddings
- `src/query_rag.py` — Interactive retrieval and generation loop
- `requirements.txt` — Python dependencies

## Notes

- Text files must be present in `data/` directory for index building
- Models are downloaded automatically on first run (requires internet access)
- Activate the virtual environment before running commands

## Quick Test Commands

Run a single question non-interactively (useful for testing):
```
python -m src.query_rag --question "What is a large language model?" --k 3
```

Rebuild the index after changing files in `data/`:
```
python -m src.build_index
```
