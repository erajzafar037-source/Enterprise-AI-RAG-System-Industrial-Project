# Security and operational limitations

This repository is a portfolio reference implementation, not a hardened multi-tenant document platform.

## Controls implemented
- Upload checks for PDF extension, PDF magic header and configured maximum bytes.
- Temporary files are removed after parsing/indexing.
- Query input has length validation and top-k bounds.
- Redis is best-effort; a cache outage does not prevent query execution.
- Cache keys include top-k and a per-process document-index revision, preventing reuse after re-indexing.
- Retrieved document content is explicitly treated as untrusted evidence in the LLM system prompt.
- API errors are returned as generic client messages; stack traces remain in server logs.

## Limitations to address before any real deployment
- No authentication, authorization, tenant or document-level access controls.
- In-memory FAISS/NumPy index is not durable and is not shared across replicas.
- No malware scanning, encrypted object storage, retention policy or audit trail.
- No prompt-injection classifier or output citation verifier.
- Mock mode is an evidence preview only; it does not generate an LLM answer.
- Metrics are process-local; production monitoring should use a persistent backend and service-level objectives.

Do not submit confidential, personal, regulated or customer documents to this demo. A production deployment needs a threat model, identity and access controls, isolation, durable storage, audit logging and security review.
