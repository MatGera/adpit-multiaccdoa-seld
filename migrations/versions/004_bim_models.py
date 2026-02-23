"""004 - BIM model registry.

Revision ID: 004
Revises: 003
Create Date: 2026-02-23
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "bim_models",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("ifc_file_path", sa.String(512), nullable=False),
        sa.Column("glb_file_path", sa.String(512), nullable=True),
        sa.Column("ifc_schema", sa.String(32), nullable=True),  # IFC2X3, IFC4, IFC4X3
        sa.Column("metadata", JSONB, nullable=True),
        sa.Column("bounding_box_min", JSONB, nullable=True),  # {x, y, z}
        sa.Column("bounding_box_max", JSONB, nullable=True),  # {x, y, z}
        sa.Column("num_elements", sa.Integer, nullable=True),
        sa.Column("file_size_bytes", sa.BigInteger, nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="processing"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # BIM assets (extracted from IFC elements)
    op.create_table(
        "bim_assets",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("bim_model_id", sa.String(64), sa.ForeignKey("bim_models.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ifc_guid", sa.String(64), nullable=False),
        sa.Column("ifc_type", sa.String(128), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("properties", JSONB, nullable=True),
        sa.Column("bounding_box_min", JSONB, nullable=True),
        sa.Column("bounding_box_max", JSONB, nullable=True),
        sa.Column("centroid", JSONB, nullable=True),  # {x, y, z}
    )

    op.create_index("idx_bim_assets_model", "bim_assets", ["bim_model_id"])
    op.create_index("idx_bim_assets_type", "bim_assets", ["ifc_type"])
    op.create_unique_constraint("uq_bim_assets_model_guid", "bim_assets", ["bim_model_id", "ifc_guid"])


def downgrade() -> None:
    op.drop_table("bim_assets")
    op.drop_table("bim_models")
