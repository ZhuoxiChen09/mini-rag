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
