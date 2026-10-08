# System Design Notes

## Request path
1. React sends a query to FastAPI.
2. Redis is checked using a normalized SHA-256 cache key.
3. Retriever selects top-k chunks.
4. LLM receives retrieved context and grounding instructions.
5. Tokens are streamed over SSE.
6. Sources are returned with the final event.

## Reliability
- Redis failure does not block retrieval/generation.
- Kubernetes probes remove unhealthy pods from service.
- HPA provides horizontal scaling.
- CI runs tests before Docker build.

## Evaluation
Production RAG needs separate retrieval and generation measurements. This repository starts with transparent offline term coverage and can be extended with recall@k, groundedness, answer relevance and human review.
