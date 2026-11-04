from __future__ import annotations

import numpy as np
import pytest

from app.retriever import Chunk, Retriever


class FakeEncoder:
    """Deterministic encoder: no model download or network access in tests."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []
        self.options: list[dict] = []

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append(list(sentences))
        self.options.append(dict(kwargs))
        return np.asarray([self.vectors[text] for text in sentences], dtype=np.float32)


def test_search_ranks_by_semantic_vector_and_preserves_source_metadata() -> None:
    encoder = FakeEncoder(
        {
            "Quarterly revenue increased": [1.0, 0.0],
            "Hospital emergency department": [0.0, 1.0],
            "Where can I get urgent medical treatment?": [0.02, 1.0],
        }
    )
    retriever = Retriever(encoder=encoder)
    retriever.rebuild(
        [
            Chunk("Quarterly revenue increased", "annual-report.pdf", 4),
            Chunk("Hospital emergency department", "employee-handbook.pdf", 12),
        ]
    )

    results = retriever.search("Where can I get urgent medical treatment?", k=2)

    assert results[0][0].source == "employee-handbook.pdf"
    assert results[0][0].page == 12
    assert results[0][1] > results[1][1]
    # Indexing and querying go through the exact same configured encoder.
    assert len(encoder.calls) == 2
    assert encoder.options[0]["normalize_embeddings"] is True
    assert encoder.options[1]["normalize_embeddings"] is True


def test_search_limits_results_to_corpus_size() -> None:
    encoder = FakeEncoder({"first": [1.0, 0.0], "second": [0.0, 1.0], "query": [1.0, 0.0]})
    retriever = Retriever(encoder=encoder)
    retriever.rebuild([Chunk("first", "a.pdf", 1), Chunk("second", "b.pdf", 2)])

    assert len(retriever.search("query", k=10)) == 2


def test_empty_index_returns_no_results_without_loading_model() -> None:
    retriever = Retriever()
    retriever.rebuild([])

    assert retriever.search("a valid query") == []
    assert retriever._encoder is None


def test_search_rejects_empty_query() -> None:
    retriever = Retriever(encoder=FakeEncoder({}))

    with pytest.raises(ValueError, match="non-empty"):
        retriever.search("   ")


@pytest.mark.parametrize("bad_k", [0, -1, True, 1.5])
def test_search_rejects_invalid_k(bad_k: object) -> None:
    retriever = Retriever(encoder=FakeEncoder({}))

    with pytest.raises(ValueError, match="positive integer"):
        retriever.search("valid query", k=bad_k)  # type: ignore[arg-type]


def test_rebuild_rejects_empty_chunk_text() -> None:
    retriever = Retriever(encoder=FakeEncoder({}))

    with pytest.raises(ValueError, match="non-empty text"):
        retriever.rebuild([Chunk("   ", "empty.pdf", 1)])


def test_embedding_shape_mismatch_is_reported() -> None:
    class ChangingDimensionEncoder:
        def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
            if sentences == ["document"]:
                return np.asarray([[1.0, 0.0]], dtype=np.float32)
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    retriever = Retriever(encoder=ChangingDimensionEncoder())
    retriever.rebuild([Chunk("document", "source.pdf", 1)])

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        retriever.search("query")


def test_non_finite_embeddings_are_rejected() -> None:
    class InvalidEncoder:
        def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
            return np.full((len(sentences), 2), np.nan, dtype=np.float32)

    retriever = Retriever(encoder=InvalidEncoder())

    with pytest.raises(ValueError, match="NaN or infinite"):
        retriever.rebuild([Chunk("document", "source.pdf", 1)])


def test_numpy_fallback_works_without_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.retriever as retriever_module

    monkeypatch.setattr(retriever_module, "faiss", None)
    encoder = FakeEncoder(
        {
            "alpha document": [1.0, 0.0],
            "beta document": [0.0, 1.0],
            "alpha question": [1.0, 0.0],
        }
    )
    retriever = Retriever(encoder=encoder)
    retriever.rebuild(
        [Chunk("alpha document", "alpha.pdf", 1), Chunk("beta document", "beta.pdf", 2)]
    )

    results = retriever.search("alpha question", k=1)

    assert len(results) == 1
    assert results[0][0].source == "alpha.pdf"
    assert results[0][1] == pytest.approx(1.0)
