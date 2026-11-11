"""FastAPI application for PDF ingestion and source-grounded RAG queries."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.cache import Cache
from app.config import settings
from app.ingest import extract_pdf
from app.llm import LLM
from app.metrics import DOCUMENTS, LATENCY, REQUESTS
from app.retriever import Chunk, Retriever
from app.schemas import QueryRequest

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="PDF-backed enterprise knowledge assistant with source citations.",
)
retriever = Retriever()
llm = LLM()
cache = Cache()
indexed_chunks: list[Chunk] = []
index_version = uuid.uuid4().hex


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    started = time.perf_counter()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        path = request.url.path
        REQUESTS.labels(endpoint=path, status=status).inc()
        LATENCY.labels(endpoint=path).observe(time.perf_counter() - started)


@app.get("/health")
@app.get("/health/live")
def health() -> dict:
    return {"status": "ok", "documents_indexed": len(indexed_chunks)}


@app.get("/health/ready")
def readiness() -> dict:
    return {"status": "ready", "documents_indexed": len(indexed_chunks)}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> dict:
    global index_version
    filename = Path(file.filename or "upload.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF documents are supported.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"PDF exceeds the {settings.max_upload_mb} MB limit.")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="Uploaded file is not a valid PDF.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        chunks = await asyncio.to_thread(extract_pdf, temp_path)
        for chunk in chunks:
            chunk.source = filename
        if not chunks:
            raise HTTPException(status_code=422, detail="PDF contains no extractable text.")

        next_index = [*indexed_chunks, *chunks]
        await asyncio.to_thread(retriever.rebuild, next_index)
        indexed_chunks[:] = next_index
        index_version = uuid.uuid4().hex
        DOCUMENTS.set(len(indexed_chunks))
        return {
            "filename": filename,
            "pages_with_text": len({chunk.page for chunk in chunks if chunk.page is not None}),
            "chunks_added": len(chunks),
            "total_indexed_chunks": len(indexed_chunks),
            "index_version": index_version,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("PDF ingestion failed")
        raise HTTPException(status_code=422, detail="PDF could not be parsed or indexed.") from exc
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                logger.warning("Could not remove temporary upload file")


async def _query_context(question: str, top_k: int) -> tuple[list[Chunk], list[dict], dict | None]:
    cached = await asyncio.to_thread(cache.get, question, top_k, index_version)
    if cached:
        return [], cached.get("sources", []), cached

    hits = await asyncio.to_thread(retriever.search, question, top_k)
    chunks = [chunk for chunk, _score in hits]
    sources = []
    seen = set()
    for chunk, score in hits:
        identity = (chunk.source, chunk.page)
        if identity in seen:
            continue
        seen.add(identity)
        sources.append({
            "source": chunk.source,
            "page": chunk.page,
            "score": round(float(score), 4),
        })
    return chunks, sources, None


async def _generate(question: str, chunks: list[Chunk], sources: list[dict], top_k: int) -> dict:
    parts = []
    async for token in llm.stream(question, chunks):
        parts.append(token)
    result = {"answer": "".join(parts).strip(), "sources": sources, "cached": False}
    await asyncio.to_thread(
        cache.set, question, result, top_k=top_k, index_version=index_version
    )
    return result


@app.post("/query")
async def query(payload: QueryRequest) -> dict:
    top_k = payload.top_k or settings.top_k
    try:
        chunks, sources, cached = await _query_context(payload.question, top_k)
        if cached is not None:
            return {**cached, "cached": True}
        return await _generate(payload.question, chunks, sources, top_k)
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=502, detail="RAG query could not be completed.") from exc


@app.post("/query/stream")
async def query_stream(payload: QueryRequest) -> StreamingResponse:
    top_k = payload.top_k or settings.top_k
    try:
        chunks, sources, cached = await _query_context(payload.question, top_k)
    except Exception as exc:
        logger.exception("RAG retrieval failed")
        raise HTTPException(status_code=502, detail="RAG retrieval could not be completed.") from exc

    async def events():
        if cached is not None:
            answer = cached.get("answer", "")
            yield f"data: {json.dumps({'type': 'answer', 'text': answer})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            yield "data: [DONE]\n\n"
            return
        parts = []
        try:
            async for token in llm.stream(payload.question, chunks):
                parts.append(token)
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
            answer = "".join(parts).strip()
            result = {"answer": answer, "sources": sources, "cached": False}
            await asyncio.to_thread(
                cache.set, payload.question, result, top_k=top_k, index_version=index_version
            )
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception:
            logger.exception("LLM streaming failed")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Answer generation failed.'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
