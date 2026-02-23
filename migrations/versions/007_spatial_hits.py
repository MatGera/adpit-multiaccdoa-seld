"""007 - Spatial hits log.

Revision ID: 007
Revises: 006
Create Date: 2026-02-23
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Spatial hits log (time-series of resolved asset hits)
    op.create_table(
        "spatial_hits",
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("device_id", sa.String(64), nullable=False),
        sa.Column("bim_model_id", sa.String(64), nullable=False),
        sa.Column("class_name", sa.String(128), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        # DOA vector (device local frame)
        sa.Column("vector_x", sa.Float, nullable=False),
        sa.Column("vector_y", sa.Float, nullable=False),
        sa.Column("vector_z", sa.Float, nullable=False),
        # Resolved asset
        sa.Column("asset_id", sa.String(64), nullable=True),
        sa.Column("asset_name", sa.String(255), nullable=True),
        sa.Column("ifc_type", sa.String(128), nullable=True),
        # Hit point in BIM coordinates
        sa.Column("hit_x", sa.Float, nullable=True),
        sa.Column("hit_y", sa.Float, nullable=True),
        sa.Column("hit_z", sa.Float, nullable=True),
        sa.Column("distance", sa.Float, nullable=True),
        # Triangulation metadata (if multi-sensor)
        sa.Column("triangulation_residual", sa.Float, nullable=True),
        sa.Column("contributing_devices", sa.ARRAY(sa.String), nullable=True),
        # LLM response reference
        sa.Column("llm_response_id", sa.String(64), nullable=True),
    )

    # Convert to hypertable
    op.execute(
        "SELECT create_hypertable('spatial_hits', 'time', "
        "chunk_time_interval => INTERVAL '1 day')"
    )

    op.create_index("idx_spatial_hits_device", "spatial_hits", ["device_id", "time"])
    op.create_index("idx_spatial_hits_asset", "spatial_hits", ["asset_id", "time"])

    # LLM responses log
    op.create_table(
        "llm_responses",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("spatial_hit_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("device_id", sa.String(64), nullable=False),
        sa.Column("asset_id", sa.String(64), nullable=True),
        sa.Column("class_name", sa.String(128), nullable=False),
        sa.Column("response_text", sa.Text, nullable=False),
        sa.Column("citations", JSONB, nullable=True),
        sa.Column("recommended_actions", JSONB, nullable=True),
        sa.Column("severity", sa.String(32), nullable=False),
        sa.Column("llm_model", sa.String(128), nullable=True),
        sa.Column("prompt_tokens", sa.Integer, nullable=True),
        sa.Column("completion_tokens", sa.Integer, nullable=True),
        sa.Column("latency_ms", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("llm_responses")
    op.drop_table("spatial_hits")
