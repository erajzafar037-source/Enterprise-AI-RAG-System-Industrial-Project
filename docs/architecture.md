# Architecture Diagram

The recruiter-ready PNG architecture diagram is included in the downloadable project bundle. The GitHub connector used for this build can write source files but cannot attach binary PNG assets directly.

```mermaid
flowchart LR
U[User] --> F[React Frontend]
F --> A[FastAPI]
A --> C[Redis Cache]
A --> R[Retriever / FAISS]
R --> L[LLM]
D[PDF Documents] --> I[PDF Ingestion]
I --> R
A --> M[Prometheus]
M --> G[Grafana]
A --> E[Evaluation]
A --> K[Kubernetes]
```
