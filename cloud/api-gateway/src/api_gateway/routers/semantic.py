"""Semantic router — prescriptive LLM queries via RAG pipeline."""

from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class SemanticAskRequest(BaseModel):
    device_id: str
    class_name: str
    confidence: float
    vector: list[float]
    bim_model_id: str
    additional_context: str | None = None


@router.post("/semantic/ask")
async def semantic_ask(request: SemanticAskRequest):
    """Prescriptive LLM query via RAG: returns streaming SSE response."""

    async def event_stream():
        # TODO: Forward to semantic layer gRPC streaming service
        yield "data: {\"token\": \"Processing query...\", \"is_final\": false}\n\n"
        yield "data: {\"token\": \"\", \"is_final\": true}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/semantic/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    asset_tags: str = Form(""),
):
    """Upload a document for RAG ingestion (PDF, DOCX, TXT)."""
    file_content = await file.read()

    # TODO: Forward to semantic layer ingestion pipeline
    return {
        "status": "queued",
        "file_name": file.filename,
        "file_size": len(file_content),
        "asset_tags": [t.strip() for t in asset_tags.split(",") if t.strip()],
    }
