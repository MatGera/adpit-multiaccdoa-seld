"""LLM orchestrator: prescriptive response generation with RAG context."""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncGenerator

import structlog

from semantic_layer.vector_search import VectorSearch, SearchResult
from semantic_layer.prompt_templates import PromptTemplates

if TYPE_CHECKING:
    from semantic_layer.config import Settings

logger = structlog.get_logger(__name__)


class LLMOrchestrator:
    """Orchestrates RAG pipeline: context retrieval → prompt assembly → LLM generation."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._search = VectorSearch(settings)
        self._templates = PromptTemplates()

    async def generate_prescriptive_response(
        self,
        device_id: str,
        class_name: str,
        confidence: float,
        doa_vector: list[float],
        spatial_hits: list[dict] | None = None,
        additional_context: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming prescriptive response.

        Pipeline:
        1. Build query from event context
        2. Retrieve relevant documents via hybrid search
        3. Assemble prompt with Jinja2 template
        4. Stream LLM response

        Yields:
            Token strings as they are generated.
        """
        # 1. Build search query
        query = self._templates.build_search_query(
            class_name=class_name,
            confidence=confidence,
            spatial_hits=spatial_hits,
        )

        # 2. Retrieve context
        search_results = await self._search.search(query, top_k=self._settings.top_k)

        # 3. Assemble prompt
        prompt = self._templates.build_prescriptive_prompt(
            device_id=device_id,
            class_name=class_name,
            confidence=confidence,
            doa_vector=doa_vector,
            spatial_hits=spatial_hits or [],
            retrieved_chunks=[
                {"content": r.content, "source": r.source, "page": r.page}
                for r in search_results
            ],
            additional_context=additional_context,
        )

        # 4. Stream LLM response
        if self._settings.llm_provider == "anthropic":
            async for token in self._stream_anthropic(prompt):
                yield token
        else:
            async for token in self._stream_ollama(prompt):
                yield token

    async def _stream_anthropic(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream response from Anthropic Claude API."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._settings.anthropic_api_key)

        async with client.messages.stream(
            model=self._settings.anthropic_model,
            max_tokens=self._settings.max_tokens,
            temperature=self._settings.temperature,
            messages=[{"role": "user", "content": prompt}],
            system=self._templates.system_prompt(),
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def _stream_ollama(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream response from local Ollama instance."""
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self._settings.ollama_host}/api/generate",
                json={
                    "model": self._settings.ollama_model,
                    "prompt": prompt,
                    "system": self._templates.system_prompt(),
                    "stream": True,
                    "options": {
                        "temperature": self._settings.temperature,
                        "num_predict": self._settings.max_tokens,
                    },
                },
            ) as response:
                import json

                async for line in response.aiter_lines():
                    if line.strip():
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done"):
                            break
