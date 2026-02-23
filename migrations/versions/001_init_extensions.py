"""001 - Initialize PostgreSQL extensions.

Revision ID: 001
Create Date: 2026-02-23
"""

from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")        # pgvector
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")   # TimescaleDB
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")       # Trigram similarity
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gin")     # GIN indexes


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS btree_gin")
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
    op.execute("DROP EXTENSION IF EXISTS timescaledb")
    op.execute("DROP EXTENSION IF EXISTS vector")
