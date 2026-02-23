"""Embedding service: generate and store vector embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

import structlog

if TYPE_CHECKING:
    from semantic_layer.config import Settings

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """Generates text embeddings using sentence-transformers."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> None:
        """Lazy-load the embedding model."""
        if self._model is None:
            logger.info("loading_embedding_model", model=self._settings.embedding_model)
            self._model = SentenceTransformer(self._settings.embedding_model)
            logger.info("embedding_model_loaded",
                         dim=self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        self._load_model()
        assert self._model is not None

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings)

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.embed([text])[0]
