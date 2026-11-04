"""Semantic retrieval for the enterprise RAG application.

Both document chunks and user questions are embedded with the same
Sentence Transformers model. Vectors are L2-normalized so inner-product
search is equivalent to cosine-similarity search.
"""
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from .config import settings

try:
    import faiss
except ImportError:  # NumPy remains a supported fallback for small corpora.
    faiss = None


class TextEncoder(Protocol):
    """Minimal interface implemented by SentenceTransformer and test doubles."""

    def encode(self, sentences: list[str], **kwargs: Any) -> Any:
        ...


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None = None


class Retriever:
    """Embed, index, and retrieve source-aware document chunks."""

    def __init__(
        self,
        encoder: TextEncoder | None = None,
        model_name: str | None = None,
    ) -> None:
        # Load the model only when the first indexing/search operation is made.
        # This keeps application imports and unit tests lightweight.
        self._encoder = encoder
        self.model_name = model_name or settings.embedding_model
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None
        self.index: Any | None = None

    def _get_encoder(self) -> TextEncoder:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Semantic retrieval requires sentence-transformers. "
                    "Install backend/requirements.txt before using the retriever."
                ) from exc

            self._encoder = SentenceTransformer(self.model_name)

        return self._encoder

    @staticmethod
    def _normalise_vectors(vectors: Any, expected_rows: int) -> np.ndarray:
        """Return finite, contiguous float32 vectors with unit L2 norm."""
        matrix = np.asarray(vectors, dtype=np.float32)

        if matrix.ndim == 1 and expected_rows == 1:
            matrix = matrix.reshape(1, -1)

        if (
            matrix.ndim != 2
            or matrix.shape[0] != expected_rows
            or matrix.shape[1] == 0
        ):
            raise ValueError(
                "The embedding model returned an invalid shape: "
                f"expected ({expected_rows}, embedding_dimension), "
                f"received {matrix.shape}."
            )

        if not np.isfinite(matrix).all():
            raise ValueError("The embedding model returned NaN or infinite values.")

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Leave zero vectors as zero rather than dividing by zero.
        matrix = matrix / np.maximum(norms, 1e-12)
        return np.ascontiguousarray(matrix, dtype=np.float32)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Encode text using one shared model and return normalized vectors."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Texts to embed must be non-empty strings.")

        raw_vectors = self._get_encoder().encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self._normalise_vectors(raw_vectors, expected_rows=len(texts))

    def rebuild(self, chunks: Sequence[Chunk]) -> None:
        """Replace the current index with embeddings for the supplied chunks."""
        new_chunks = list(chunks)
        if not new_chunks:
            self.chunks = []
            self.matrix = np.empty((0, 0), dtype=np.float32)
            self.index = None
            return

        if any(not isinstance(chunk.text, str) or not chunk.text.strip()
               for chunk in new_chunks):
            raise ValueError("Every chunk must contain non-empty text.")

        # Build into local variables first so a failed model call does not leave
        # the retriever with a half-updated index.
        new_matrix = self.embed([chunk.text for chunk in new_chunks])
        new_index = None

        if faiss is not None:
            new_index = faiss.IndexFlatIP(new_matrix.shape[1])
            new_index.add(new_matrix)

        self.chunks = new_chunks
        self.matrix = new_matrix
        self.index = new_index

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        """Return the top-k chunks ranked by cosine similarity."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer.")

        if not self.chunks or self.matrix is None:
            return []

        query_vector = self.embed([query])
        if query_vector.shape[1] != self.matrix.shape[1]:
            raise ValueError(
                "Embedding dimension mismatch between indexed documents "
                f"({self.matrix.shape[1]}) and query ({query_vector.shape[1]}). "
                "Use the same embedding model and rebuild the index."
            )

        limit = min(k, len(self.chunks))
        if self.index is not None:
            scores, ids = self.index.search(query_vector, limit)
            return [
                (self.chunks[int(chunk_id)], float(scores[0][position]))
                for position, chunk_id in enumerate(ids[0])
                if chunk_id >= 0
            ]

        # Fallback for environments where FAISS is not installed.
        scores = (self.matrix @ query_vector[0]).ravel()
        ranked_ids = np.argsort(-scores, kind="stable")[:limit]
        return [
            (self.chunks[int(chunk_id)], float(scores[chunk_id]))
            for chunk_id in ranked_ids
        ]
