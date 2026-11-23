import json
import sys
from unittest.mock import Mock

import numpy as np
import pytest

from src import query_rag
from src.build_index import chunk_text, load_documents
from src.query_rag import Chunk, build_prompt, load_index, retrieve, truncate_sentences


def test_chunk_sizes_and_overlap():
    assert chunk_text("one two three four five", chunk_size=3, overlap=1) == [
        "one two three", "three four five", "five",
    ]
    assert chunk_text("") == []


def test_load_documents_is_sorted(tmp_path):
    (tmp_path / "z.txt").write_text("last", encoding="utf-8")
    (tmp_path / "a.txt").write_text("first", encoding="utf-8")
    assert [doc["filename"] for doc in load_documents(tmp_path)] == ["a.txt", "z.txt"]


def test_truncate_sentences():
    assert truncate_sentences('Answer: "These are words. So are these."', n=1) == (
        "These are words."
    )
    assert truncate_sentences("These are a lot of words.", max_words=3) == "These are a."
    assert truncate_sentences(" ") == "I don't know."


def test_build_prompt():
    prompt = build_prompt("A source sentence.", "What does it say?")
    assert prompt.startswith("Question: What does it say?\n\nContext: A source sentence.")
    assert "I don't know." in prompt
    assert prompt.endswith("Answer:")


def test_load_index(tmp_path):
    with pytest.raises(FileNotFoundError, match="embeddings.npy and chunks.json"):
        load_index(tmp_path)
    np.save(tmp_path / "embeddings.npy", np.array([[1, 0]]))
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path)
    (tmp_path / "chunks.json").write_text(
        json.dumps([{"text": "a", "doc": "sample.txt"}]), encoding="utf-8"
    )
    embeddings, chunks = load_index(tmp_path)
    np.testing.assert_array_equal(embeddings, [[1, 0]])
    assert embeddings.dtype == np.float32
    assert chunks == [Chunk("a", "sample.txt", 0)]


class FakeEmbedder:
    def encode(self, queries, *, convert_to_numpy):
        assert queries == ["sample"]
        assert convert_to_numpy is True
        return np.array([[1.0, 0.0]])


def test_retrieve_applies_relative_and_absolute_filters():
    embeddings = np.array([[1.0, 0.0], [0.96, 0.28], [0.8, 0.6]])
    chunks = [Chunk(str(i), "sample.txt", i) for i in range(3)]
    embedder = FakeEmbedder()
    results, scores = retrieve("sample", embedder, embeddings, chunks, k=3)
    assert results == chunks[:2]
    np.testing.assert_allclose(scores, [1.0, 0.96, 0.8])
    results, _ = retrieve("sample", embedder, embeddings[2:], chunks[2:], min_similarity=0.9)
    assert results == []


def test_interactive_queries_reuse_resources(monkeypatch):
    resources = object()
    loader = Mock(return_value=resources)
    answer = Mock()
    questions = iter(["first question", "second question", "exit"])
    monkeypatch.setattr(sys, "argv", ["query_rag"])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))
    monkeypatch.setattr(query_rag, "load_query_resources", loader)
    monkeypatch.setattr(query_rag, "run_query", answer)
    query_rag.main()
    assert loader.call_count == 1
    assert [call.args[0] for call in answer.call_args_list] == [
        "first question", "second question"
    ]
    assert all(call.kwargs["resources"] is resources for call in answer.call_args_list)
