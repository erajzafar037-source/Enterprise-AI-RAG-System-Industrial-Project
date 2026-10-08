from prometheus_client import Counter, Histogram, Gauge
REQUESTS=Counter("rag_requests_total","Total RAG API requests",["endpoint","status"])
LATENCY=Histogram("rag_request_latency_seconds","RAG request latency",["endpoint"])
CACHE_HITS=Counter("rag_cache_hits_total","Redis cache hits")
CACHE_MISSES=Counter("rag_cache_misses_total","Redis cache misses")
DOCUMENTS=Gauge("rag_documents_indexed","Number of indexed document chunks")
LLM_ERRORS=Counter("rag_llm_errors_total","LLM generation errors")
