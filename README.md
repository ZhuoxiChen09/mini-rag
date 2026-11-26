# mini-rag

This repository is a **personal playground** for experimenting with Retrieval-Augmented Generation (RAG) techniques on small local text corpora. It is intended for learning and prototyping only; the implementation is simplified, and the retrieval and answer quality are **not tuned for production or mission-critical use**.

The code provides a minimal end‑to‑end RAG pipeline that:
- embeds local text files,
- retrieves semantically similar chunks for a given query, and
- uses a small instruction‑tuned language model to generate a concise answer from the retrieved context.

## Getting Started

### Prerequisites

- Python 3.10–3.11
- Internet access (models are downloaded on first run)

1. Create a virtual environment and install the dependencies.

   Bash:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

   PowerShell:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
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
What is a large language model?
```

Expected Output:
```
machine learning systems trained to predict the next token in a sequence of text
```

Query:
```
How does self-attention work?
```

Expected Output:
```
Self-attention lets each token attend to every other token in the sequence, computing weighted combinations of token representations
```

Example outputs from a local run; exact wording may vary by model version.

## Retrieval & Answering Behavior

- The retriever uses cosine similarity over chunk embeddings. A minimum similarity threshold is applied (default: `0.25`) — if no chunk meets this threshold the system will respond with `I don't know` before calling the generator.
- The LLM generation is constrained for brevity: answers are post-processed to produce 1–2 short sentences (word-truncated if necessary) to keep responses concise and easier to read.
- Context truncation: Long contexts are automatically truncated to fit within the model's sequence length limit (512 tokens), prioritizing the most relevant chunks.
- Source attribution: Retrieved chunks display their source filename and chunk ID for transparency.
- You can adjust the threshold and output behavior via command-line arguments or by editing `src/query_rag.py` (see `retrieve(..., min_similarity=...)` and the `truncate_sentences` post-processing logic).

## Tests

```bash
python -m pip install -r requirements-dev.txt
ruff check .
pytest -q
```

The unit tests use synthetic vectors and do not download models. Run
`python -m src.eval_retrieval` to evaluate source retrieval on ten labeled questions and measure retrieval abstention on four questions outside the corpus. The evaluator prints source hit@1 and hit@3 on answerable questions and the fraction of out-of-corpus questions for which no chunk was returned. It does not grade generated answers. Use `--min-similarity` to compare thresholds on the same questions. This command downloads the embedding model on first run.

## Technical Notes

- **Module invocation:** Use `python -m src.query_rag` (underscore). Running `python -m src.query.rag` will raise `ModuleNotFoundError` because the module name uses an underscore rather than a dotted package path.
- **Models used:** Embeddings: `sentence-transformers/all-MiniLM-L6-v2`. LLM: `google/flan-t5-base`. Both models are downloaded automatically on first run (internet required).
- **Device fallback:** The scripts detect available devices and will fall back to CPU if a GPU is not available (you will see `Device set to use cpu`). The example runs used CPU.
- **Retrieval behavior:** If no chunk passes the `min_similarity` threshold the tool returns `I don't know`. A retrieved chunk can still be irrelevant or insufficient, so an answer can be wrong even when the cutoff is met.
- **Context length:** The system automatically truncates contexts to ~240 words to fit within the model's 512-token limit, ensuring reliable generation without sequence length errors.

## Project Structure

```
mini-rag/
├── data/
│   ├── sample_1.txt
│   ├── sample_2.txt
│   └── sample_3.txt
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

If you see unexpected `I don't know` answers, try the following:

```
# 1) Ensure relevant text files are present in `data/`
python -m src.build_index

# 2) Re-run the query loop
python -m src.query_rag

# 3) Lower the similarity threshold via command-line argument
python -m src.query_rag --question "Your question" --min-similarity 0.20

# 4) Or adjust the default threshold in `src/query_rag.py`
```

## Command-Line Options

The `query_rag.py` script supports several options:

- `--question`, `-q`: Ask a single question (non-interactive mode)
- `--k`: Number of top chunks to retrieve (default: 2)
- `--min-similarity`: Minimum similarity threshold (default: 0.25)
- `--max-words`: Maximum words per sentence (default: 28)
- `--style`: Answer length style - `short` (1-2 sentences) or `medium` (up to 3 sentences)
- `--no-context`: Hide the retrieved context in output
- `--index-dir`: Custom path to artifacts directory
