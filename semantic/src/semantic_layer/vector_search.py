"""Hybrid vector search: pgvector cosine + BM25/tsvector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import text

import structlog

from seld_common.db import DatabaseManager
from semantic_layer.embeddings import EmbeddingService

if TYPE_CHECKING:
    from semantic_layer.config import Settings

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """A single search result from the vector store."""
    chunk_id: int
    content: str
    source: str
    page: int | None
    score: float
    asset_tags: list[str]


class VectorSearch:
    """Hybrid search combining pgvector cosine similarity and BM25 text search."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._db = DatabaseManager(settings.database_url)
        self._embedder = EmbeddingService(settings)

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        asset_filter: list[str] | None = None,
    ) -> list[SearchResult]:
        """Perform hybrid search: vector similarity + BM25 text.

        Args:
            query: Natural language query.
            top_k: Number of results to return.
            asset_filter: Optional list of asset tags to filter by.

        Returns:
            Ranked list of SearchResult.
        """
        k = top_k or self._settings.top_k
        query_embedding = self._embedder.embed_single(query)

        # Vector search results
        vector_results = await self._vector_search(query_embedding, k * 2, asset_filter)

        # BM25/tsvector search results
        text_results = await self._text_search(query, k * 2, asset_filter)

        # Reciprocal Rank Fusion (RRF)
        fused = self._rrf_fusion(vector_results, text_results)

        return fused[:k]

    async def _vector_search(
        self,
        embedding: np.ndarray,
        limit: int,
        asset_filter: list[str] | None = None,
    ) -> list[SearchResult]:
        """Cosine similarity search using pgvector."""
        params: dict = {
            "embedding": embedding.tolist(),
            "limit": limit,
            "threshold": self._settings.similarity_threshold,
        }

        asset_clause = ""
        if asset_filter:
            asset_clause = "AND asset_tags && :asset_tags"
            params["asset_tags"] = asset_filter
'''
SELECT text of the chunk, the source, the page number, tags and the cosine similarity (from 1=identical to 0=completely different, subtracting 1)
 between the vector saved in the chunk and the query vector,
FROM the table document_embeddings
WHERE the cosine similarity >= the threshold(a defoult value of 0.5) AND even one of the tags of the user query is present in the tags of the chunk
ORDER BY decrescent cosine similarity 
LIMIT the maximum number of lines form python
'''
        async with self._db.session() as session:
            result = await session.execute(
                text(f"""
                    SELECT idcontent, source, page_number,                     
                           1 - (embedding <=> :embedding::vector) AS score,
                           asset_tags
                    FROM document_embeddings
                    WHERE 1 - (embedding <=> :embedding::vector) >= :threshold
                    {asset_clause}
                    ORDER BY embedding <=> :embedding::vector
                    LIMIT :limit
                """), 
                params,
            )
            rows = result.fetchall()

        return [
            SearchResult(
                chunk_id=r.id,
                content=r.content,
                source=r.source,
                page=r.page_number,
                score=float(r.score),
                asset_tags=r.asset_tags or [],
            )
            for r in rows
        ]

    async def _text_search(
        self,
        query: str,
        limit: int,
        asset_filter: list[str] | None = None,
    ) -> list[SearchResult]:
        """BM25-style full-text search using PostgreSQL tsvector."""
        params: dict = {
            "query": query,
            "limit": limit,
        }

        asset_clause = ""
        if asset_filter:
            asset_clause = "AND asset_tags && :asset_tags"
            params["asset_tags"] = asset_filter

        async with self._db.session() as session:
            result = await session.execute(
                text(f"""
                    SELECT content, source, page_number,
                           ts_rank(
                               content_tsv,
                               plainto_tsquery('english', :query)
                           ) AS score,
                           asset_tags
                    FROM document_embeddings
                    WHERE content_tsv @@
                          plainto_tsquery('english', :query)
                    {asset_clause}
                    ORDER BY score DESC
                    LIMIT :limit
                """),
                params,
            )
            rows = result.fetchall()

        return [
            SearchResult(
                content=r.content,
                source=r.source,
                page=r.page_number,
                score=float(r.score),
                asset_tags=r.asset_tags or [],
            )
            for r in rows
        ]

    def _rrf_fusion(
        self,
        vector_results: list[SearchResult],
        text_results: list[SearchResult],
        k: int = 60,
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion of vector and text search results.

        RRF score = w_v / (k + rank_v) + w_t / (k + rank_t)
        """
        w_v = self._settings.vector_weight
        w_t = self._settings.bm25_weight

        # Build score map by content hash
        scores: dict[str, tuple[float, SearchResult]] = {}

        for rank, result in enumerate(vector_results):
            key = str(result.chunk_id)
            rrf = w_v / (k + rank + 1)
            if key in scores:
                scores[key] = (scores[key][0] + rrf, result)
            else:
                scores[key] = (rrf, result)

        for rank, result in enumerate(text_results):
            key = str(result.chunk_id)
            rrf = w_t / (k + rank + 1)
            if key in scores:
                scores[key] = (scores[key][0] + rrf, scores[key][1])
            else:
                scores[key] = (rrf, result)

        # Sort by fused score
        fused = sorted(scores.values(), key=lambda x: x[0], reverse=True)

        return [
            SearchResult(
                content=result.content,
                source=result.source,
                page=result.page,
                score=score,
                asset_tags=result.asset_tags,
            )
            for score, result in fused
        ]
