# Production RAG AI System

Production-oriented enterprise Retrieval-Augmented Generation reference implementation demonstrating AI engineering, MLOps and cloud-native delivery.

## Architecture

User -> React -> FastAPI -> Redis cache -> Retriever/FAISS -> LLM -> streaming response

Platform: PDF ingestion, Prometheus metrics, evaluation, GitHub Actions CI, Docker and Kubernetes.

## Features

- PDF upload and page-aware chunking
- Semantic retrieval using Sentence Transformers embeddings
- FAISS inner-product search over L2-normalized vectors (cosine similarity); NumPy fallback
- Configurable embedding model via EMBEDDING_MODEL
- OpenAI streaming when OPENAI_API_KEY is configured
- Key-free deterministic local demo mode
- Redis response caching with graceful fallback
- Prometheus metrics at /metrics
- Offline evaluation harness
- Docker Compose
- Kubernetes Deployment, Service and HPA
- React recruiter demo UI

## Semantic retrieval

The retriever encodes document chunks and incoming questions with the same Sentence Transformers model. The default is `sentence-transformers/all-MiniLM-L6-v2`. Model loading is lazy: the model is downloaded from Hugging Face the first time the application indexes or searches non-empty content. Set `EMBEDDING_MODEL` in the environment to use a different compatible model.

FAISS uses inner-product search over normalized vectors, which is equivalent to cosine similarity. If FAISS is unavailable, the retriever falls back to NumPy similarity search. Rebuild the document index whenever you change the embedding model; vectors from different models must not be mixed.

## Quick start

1. Copy `backend/.env.example` to `.env` and configure the settings you need.
2. Run: `docker compose up --build`.
3. API: `http://localhost:8000`.
4. Health: `GET /health`.
5. Metrics: `GET /metrics`.

The first semantic indexing/query operation downloads the configured embedding model. Allow extra startup time and network access the first time. The model cache should be persisted for repeat container runs in a production deployment.

Frontend: `cd frontend && npm install && npm run dev`.

## Tests

Run the backend test suite from the repository root:

```bash
cd backend
pytest -q
```

Retrieval unit tests use an injected deterministic encoder, so they do not download a model or call an external API.

## API

POST `/documents` accepts a PDF.
POST `/query` returns a grounded answer and sources.
POST `/query/stream` returns Server-Sent Events for token streaming.

## Monitoring

Run the stack in `monitoring/`. Track request rate, latency, cache hits/misses, indexed chunks and LLM errors.

## Evaluation

Run: `python evaluation/evaluate.py evaluation/predictions.example.jsonl`.

This starts with transparent term coverage. A mature production system should add retrieval recall@k, groundedness, answer relevance, human review and an LLM judge.

## Kubernetes

`k8s/deployment.yaml` contains two replicas, readiness/liveness probes and an HPA. Replace the example image and secret before a real cluster deployment.

## CV mapping

Python/FastAPI -> `backend/app`
LLM/RAG -> `rag.py`, `retriever.py`, `llm.py`
PDF ingestion -> `ingest.py`
Redis -> `cache.py`
Kubernetes -> `k8s`
Monitoring -> `metrics.py` and `monitoring/`
Evaluation -> `evaluation/`
CI/CD -> `.github/workflows/ci.yml`
Cloud-ready -> Docker and Kubernetes

Portfolio honesty: this repository is Kubernetes-ready and includes a key-free local demo. Add a public cloud URL only after you have actually deployed it.

Author: Eraj Zafar
GitHub: https://github.com/erajzafar037-source
LinkedIn: https://www.linkedin.com/in/eraj-zafar-945a4248/
