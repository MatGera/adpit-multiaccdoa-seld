"""003 - Prediction hypertable (TimescaleDB).

Revision ID: 003
Revises: 002
Create Date: 2026-02-23
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, ARRAY

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Predictions time-series table
    op.create_table(
        "predictions",
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("device_id", sa.String(64), nullable=False),
        sa.Column("frame_idx", sa.BigInteger, nullable=False),
        sa.Column("class_name", sa.String(128), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("vector_x", sa.Float, nullable=False),
        sa.Column("vector_y", sa.Float, nullable=False),
        sa.Column("vector_z", sa.Float, nullable=False),
    )

    # Convert to TimescaleDB hypertable
    op.execute(
        "SELECT create_hypertable('predictions', 'time', "
        "chunk_time_interval => INTERVAL '1 day')"
    )

    # Indexes for common queries
    op.create_index("idx_predictions_device_time", "predictions", ["device_id", "time"])
    op.create_index("idx_predictions_class", "predictions", ["class_name", "time"])

    # Device telemetry time-series
    op.create_table(
        "device_telemetry",
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("device_id", sa.String(64), nullable=False),
        sa.Column("cpu_temp", sa.Float),
        sa.Column("gpu_temp", sa.Float),
        sa.Column("mem_used_mb", sa.Integer),
        sa.Column("inference_ms", sa.Float),
    )

    op.execute(
        "SELECT create_hypertable('device_telemetry', 'time', "
        "chunk_time_interval => INTERVAL '1 day')"
    )

    op.create_index("idx_telemetry_device_time", "device_telemetry", ["device_id", "time"])

    # Continuous aggregate for prediction rates (1 minute buckets)
    op.execute("""
        CREATE MATERIALIZED VIEW prediction_rates_1m
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 minute', time) AS bucket,
            device_id,
            class_name,
            count(*) AS prediction_count,
            avg(confidence) AS avg_confidence
        FROM predictions
        GROUP BY bucket, device_id, class_name
        WITH NO DATA
    """)

    op.execute("""
        SELECT add_continuous_aggregate_policy('prediction_rates_1m',
            start_offset => INTERVAL '1 hour',
            end_offset => INTERVAL '1 minute',
            schedule_interval => INTERVAL '1 minute')
    """)


def downgrade() -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS prediction_rates_1m CASCADE")
    op.drop_table("device_telemetry")
    op.drop_table("predictions")
