import pytest

from app.ingest import chunk_text


def test_chunk_text_normalizes_whitespace_and_returns_content():
    chunks = chunk_text("  alpha   beta\n gamma  ", size=12, overlap=3)

    assert chunks
    assert all(chunk == chunk.strip() for chunk in chunks)
    assert " ".join(chunks).replace("  ", " ").startswith("alpha")


def test_chunk_text_respects_size_and_makes_progress():
    text = "one two three four five six seven eight nine ten eleven twelve"
    chunks = chunk_text(text, size=16, overlap=4)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 16 for chunk in chunks)
    assert chunks[0] != chunks[1]


def test_chunk_text_returns_empty_for_blank_text():
    assert chunk_text(" \n\t ", size=20, overlap=2) == []


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11), (True, 0)],
)
def test_chunk_text_rejects_invalid_bounds(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some content", size=size, overlap=overlap)


def test_chunk_text_rejects_non_string_input():
    with pytest.raises(TypeError):
        chunk_text(None, size=20, overlap=2)
