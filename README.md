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
│   ├── doc1.txt
│   └── doc2.txt
│
├── artifacts/            # Saved embeddings and chunk metadata (generated)
│   ├── embeddings.npy
│   └── chunks.json
│
├── src/
│   ├── build_index.py    # Build embeddings index from data/
│   └── query_rag.py      # Interactive RAG-style Q&A
│
├── requirements.txt
└── README.md
