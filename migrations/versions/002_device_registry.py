"""002 - Device registry tables.

Revision ID: 002
Revises: 001
Create Date: 2026-02-23
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "devices",
        sa.Column("device_id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "hardware_type",
            sa.Enum("industrial_capacitive", "infrastructure_piezoelectric", name="hardware_type_enum"),
            nullable=False,
        ),
        sa.Column("num_channels", sa.Integer, nullable=False),
        sa.Column("sample_rate", sa.Integer, nullable=False, server_default="48000"),
        sa.Column("frame_length_ms", sa.Integer, nullable=False, server_default="100"),
        sa.Column("confidence_threshold", sa.Float, nullable=False, server_default="0.5"),
        sa.Column("model_version", sa.String(64), nullable=True),
        sa.Column("firmware_version", sa.String(64), nullable=True),
        sa.Column("location", JSONB, nullable=True),
        sa.Column("mqtt_topic_prefix", sa.String(128), nullable=False, server_default="dt/edge"),
        sa.Column("ota_enabled", sa.Boolean, nullable=False, server_default="true"),
        sa.Column(
            "status",
            sa.Enum("online", "offline", "degraded", "maintenance", "provisioning", name="device_status_enum"),
            nullable=False,
            server_default="provisioning",
        ),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Client certificates table for mTLS
    op.create_table(
        "device_certificates",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("device_id", sa.String(64), sa.ForeignKey("devices.device_id", ondelete="CASCADE"), nullable=False),
        sa.Column("cert_pem", sa.Text, nullable=False),
        sa.Column("serial_number", sa.String(128), unique=True, nullable=False),
        sa.Column("issued_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked", sa.Boolean, server_default="false"),
    )

    op.create_index("idx_device_certificates_device_id", "device_certificates", ["device_id"])


def downgrade() -> None:
    op.drop_table("device_certificates")
    op.drop_table("devices")
    op.execute("DROP TYPE IF EXISTS hardware_type_enum")
    op.execute("DROP TYPE IF EXISTS device_status_enum")
