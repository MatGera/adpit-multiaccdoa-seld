"""005 - Calibration matrices (roto-translation per device).

Revision ID: 005
Revises: 004
Create Date: 2026-02-23
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "calibration_matrices",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "device_id",
            sa.String(64),
            sa.ForeignKey("devices.device_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "bim_model_id",
            sa.String(64),
            sa.ForeignKey("bim_models.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # 4x4 homogeneous transformation matrix stored as 16-element array (row-major)
        sa.Column("matrix", ARRAY(sa.Float), nullable=False),
        # Origin in BIM coordinates (extracted from matrix for quick spatial queries)
        sa.Column("origin_x", sa.Float, nullable=False),
        sa.Column("origin_y", sa.Float, nullable=False),
        sa.Column("origin_z", sa.Float, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_unique_constraint(
        "uq_calibration_device_model",
        "calibration_matrices",
        ["device_id", "bim_model_id"],
    )


def downgrade() -> None:
    op.drop_table("calibration_matrices")
