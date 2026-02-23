"""Semantic Layer gRPC servicer — handles RAG and LLM RPCs."""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncGenerator

import structlog

from semantic_layer.llm_orchestrator import LLMOrchestrator
from semantic_layer.document_ingestion import DocumentIngestion
from semantic_layer.vector_search import VectorSearch
from semantic_layer.guardrails import Guardrails

if TYPE_CHECKING:
    from semantic_layer.config import Settings

logger = structlog.get_logger(__name__)


class SemanticServicer:
    """gRPC servicer for semantic query operations.

    Implements SemanticService RPCs:
    - Ask: streaming prescriptive LLM response
    - Search: vector search without LLM
    - Ingest: add document to vector store
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._orchestrator = LLMOrchestrator(settings)
        self._ingestion = DocumentIngestion(settings)
        self._search = VectorSearch(settings)
        self._guardrails = Guardrails()

    async def ask(
        self,
        device_id: str,
        class_name: str,
        confidence: float,
        doa_vector: list[float],
        spatial_hits: list[dict] | None = None,
        additional_context: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a prescriptive LLM response through RAG pipeline.

        Yields:
            Token strings as they are generated.
        """
        # Input guardrails
        input_check = self._guardrails.validate_input(
            f"{class_name} {confidence} {additional_context or ''}"
        )
        if not input_check.passed:
            yield f"[GUARDRAIL] {input_check.reason}"
            return

        buffer = ""
        async for token in self._orchestrator.generate_prescriptive_response(
            device_id=device_id,
            class_name=class_name,
            confidence=confidence,
            doa_vector=doa_vector,
            spatial_hits=spatial_hits,
            additional_context=additional_context,
        ):
            buffer += token
            yield token

        # Output guardrails (post-hoc logging)
        output_check = self._guardrails.validate_output(buffer)
        if not output_check.passed:
            logger.warning("output_guardrail_failed", reason=output_check.reason)

    async def search(
        self,
        query: str,
        top_k: int = 8,
        asset_filter: list[str] | None = None,
    ) -> list[dict]:
        """Perform hybrid search without LLM generation."""
        results = await self._search.search(query, top_k, asset_filter)
        return [
            {
                "content": r.content,
                "source": r.source,
                "page": r.page,
                "score": r.score,
            }
            for r in results
        ]

    async def ingest(
        self,
        content: str,
        source: str,
        asset_tags: list[str] | None = None,
    ) -> dict:
        """Ingest text content into the vector store."""
        count = await self._ingestion.ingest_text(content, source, asset_tags)
        return {"status": "ingested", "chunks": count, "source": source}
