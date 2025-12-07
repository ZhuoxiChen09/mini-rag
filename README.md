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

## Retrieval & Answering Behavior

- The retriever uses cosine similarity over chunk embeddings. A minimum similarity threshold is applied (default: `0.25`) — if no chunk meets this threshold the system will respond with `I don't know` to avoid hallucination.
- The LLM generation is constrained for brevity: answers are post-processed to produce 1–2 short sentences (word-truncated if necessary) to keep responses concise and easier to read.
- Context truncation: Long contexts are automatically truncated to fit within the model's sequence length limit (512 tokens), prioritizing the most relevant chunks.
- Source attribution: Retrieved chunks display their source filename and chunk ID for transparency.
- You can adjust the threshold and output behavior via command-line arguments or by editing `src/query_rag.py` (see `retrieve(..., min_similarity=...)` and the `truncate_sentences` post-processing logic).

## Recent Improvements

- **Simplified prompt engineering:** Optimized prompts for better performance with small language models (flan-t5-base)
- **Improved answer extraction:** Better filtering of prompt artifacts and instruction leakage
- **Context truncation:** Automatic handling of long contexts to prevent sequence length errors
- **Fixed source metadata:** Chunks now correctly display their source filenames
- **Lowered similarity threshold:** Default threshold reduced from 0.30 to 0.25 for better recall

## Technical Notes

- **Module invocation:** Use `python -m src.query_rag` (underscore). Running `python -m src.query.rag` will raise `ModuleNotFoundError` because the module name uses an underscore rather than a dotted package path.
- **Models used:** Embeddings: `sentence-transformers/all-MiniLM-L6-v2`. LLM: `google/flan-t5-base`. Both models are downloaded automatically on first run (internet required).
- **Device fallback:** The scripts detect available devices and will fall back to CPU if a GPU is not available (you will see `Device set to use cpu`). The example runs used CPU.
- **Retrieval behavior:** If no chunk passes the `min_similarity` threshold the tool returns `I don't know`. This prevents ungrounded LLM hallucination but means specific queries may produce no answer until you add relevant documents to `data/` or lower the threshold.
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
