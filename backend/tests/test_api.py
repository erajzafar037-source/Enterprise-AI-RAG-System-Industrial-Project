from fastapi.testclient import TestClient

import app.main as main


class FakeRetriever:
    def search(self, question, k=4):
        class Item:
            text = "The support policy allows 30 days of leave."
            source = "handbook.pdf"
            page = 3
        return [(Item(), 0.93)]


class FakeCache:
    def get(self, question, top_k=4, index_version="local"):
        return None
    def set(self, question, value, ttl=900, top_k=4, index_version="local"):
        return None


class FakeLLM:
    async def stream(self, question, docs):
        yield "The policy allows 30 days of leave [1]."


def test_health_and_metrics():
    client = TestClient(main.app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/health/live").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_query_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr(main, "retriever", FakeRetriever())
    monkeypatch.setattr(main, "cache", FakeCache())
    monkeypatch.setattr(main, "llm", FakeLLM())
    response = TestClient(main.app).post(
        "/query", json={"question": "How much leave is allowed?", "top_k": 2}
    )
    assert response.status_code == 200
    assert "30 days" in response.json()["answer"]
    assert response.json()["sources"][0]["source"] == "handbook.pdf"


def test_query_stream_emits_tokens_and_sources(monkeypatch):
    monkeypatch.setattr(main, "retriever", FakeRetriever())
    monkeypatch.setattr(main, "cache", FakeCache())
    monkeypatch.setattr(main, "llm", FakeLLM())
    response = TestClient(main.app).post(
        "/query/stream", json={"question": "How much leave is allowed?"}
    )
    assert response.status_code == 200
    assert "event-stream" in response.headers["content-type"]
    assert '"type": "token"' in response.text
    assert '"type": "sources"' in response.text


def test_upload_rejects_non_pdf():
    response = TestClient(main.app).post(
        "/documents", files={"file": ("notes.txt", b"not a PDF", "text/plain")}
    )
    assert response.status_code == 415
