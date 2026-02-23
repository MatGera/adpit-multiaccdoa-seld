"""Document ingestion pipeline: parse, chunk, embed, store."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import text

import structlog

from seld_common.db import DatabaseManager
from semantic_layer.embeddings import EmbeddingService

if TYPE_CHECKING:
    from semantic_layer.config import Settings

logger = structlog.get_logger(__name__)


class DocumentIngestion:
    """Ingests documents into the vector store for RAG retrieval."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._db = DatabaseManager(settings.database_url)
        self._embedder = EmbeddingService(settings)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def ingest_text(
        self,
        content: str,
        source: str,
        asset_tags: list[str] | None = None,
    ) -> int:
        """Ingest raw text content: split, embed, store.

        Args:
            content: Raw text content.
            source: Source identifier (file name, URL, etc).
            asset_tags: Optional tags linking to BIM assets.

        Returns:
            Number of chunks stored.
        """
        chunks = self._splitter.split_text(content)
        if not chunks:
            return 0

        embeddings = self._embedder.embed(chunks)

        async with self._db.session() as session:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                content_hash = hashlib.sha256(chunk.encode()).hexdigest()

                await session.execute(
                    text("""
                        INSERT INTO document_embeddings
                            (id, content, source, page_number, embedding,
                             content_hash, asset_tags, chunk_index)
                        VALUES
                            (:id, :content, :source, :page, :embedding,
                             :content_hash, :asset_tags, :chunk_index)
                        ON CONFLICT (content_hash) DO NOTHING
                    """),
                    {
                        "id": str(uuid.uuid4()),
                        "content": chunk,
                        "source": source,
                        "page": None,
                        "embedding": embedding.tolist(),
                        "content_hash": content_hash,
                        "asset_tags": asset_tags or [],
                        "chunk_index": i,
                    },
                )

        logger.info("document_ingested", source=source, chunks=len(chunks))
        return len(chunks)

    async def ingest_pdf(self, file_path: str, asset_tags: list[str] | None = None) -> int:
        """Ingest a PDF file."""
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        total_chunks = 0

        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if not page_text or not page_text.strip():
                continue

            chunks = self._splitter.split_text(page_text)
            if not chunks:
                continue

            embeddings = self._embedder.embed(chunks)

            async with self._db.session() as session:
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    content_hash = hashlib.sha256(chunk.encode()).hexdigest()

                    await session.execute(
                        text("""
                            INSERT INTO document_embeddings
                                (id, content, source, page_number, embedding,
                                 content_hash, asset_tags, chunk_index)
                            VALUES
                                (:id, :content, :source, :page, :embedding,
                                 :content_hash, :asset_tags, :chunk_index)
                            ON CONFLICT (content_hash) DO NOTHING
                        """),
                        {
                            "id": str(uuid.uuid4()),
                            "content": chunk,
                            "source": Path(file_path).name,
                            "page": page_num,
                            "embedding": embedding.tolist(),
                            "content_hash": content_hash,
                            "asset_tags": asset_tags or [],
                            "chunk_index": i,
                        },
                    )

            total_chunks += len(chunks)

        logger.info("pdf_ingested", file=file_path, total_chunks=total_chunks)
        return total_chunks

    async def ingest_docx(self, file_path: str, asset_tags: list[str] | None = None) -> int:
        """Ingest a DOCX file."""
        from docx import Document

        doc = Document(file_path)
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return await self.ingest_text(full_text, Path(file_path).name, asset_tags)
