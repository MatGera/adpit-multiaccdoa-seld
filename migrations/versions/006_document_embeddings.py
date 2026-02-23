"""006 - Document embeddings for RAG (pgvector).

Revision ID: 006
Revises: 005
Create Date: 2026-02-23
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Documents table
    op.create_table(
        "documents",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("file_name", sa.String(512), nullable=False),
        sa.Column("file_type", sa.String(16), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger, nullable=True),
        sa.Column("asset_tags", sa.ARRAY(sa.String), nullable=True),
        sa.Column("metadata", JSONB, nullable=True),
        sa.Column("num_chunks", sa.Integer, nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="processing"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Document chunks with embeddings
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "document_id",
            sa.String(64),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("source_page", sa.Integer, nullable=True),
        sa.Column("metadata", JSONB, nullable=True),
        # pgvector embedding column (1536 dimensions for text-embedding-3-large)
        # Use raw SQL for vector type
    )

    # Add vector column via raw SQL (pgvector extension)
    op.execute("ALTER TABLE document_chunks ADD COLUMN embedding vector(1536)")

    # Add tsvector column for full-text search
    op.execute(
        "ALTER TABLE document_chunks ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
    )

    # Indexes
    op.create_index("idx_chunks_document", "document_chunks", ["document_id"])

    # HNSW index for vector similarity search
    op.execute(
        "CREATE INDEX idx_chunks_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )

    # GIN index for full-text search
    op.execute(
        "CREATE INDEX idx_chunks_content_tsv ON document_chunks "
        "USING GIN (content_tsv)"
    )


def downgrade() -> None:
    op.drop_table("document_chunks")
    op.drop_table("documents")
