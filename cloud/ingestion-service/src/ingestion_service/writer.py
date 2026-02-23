"""TimescaleDB writer: batch-inserts prediction rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

import structlog

from seld_common.db import DatabaseManager
from seld_common.schemas import EdgePredictionPayload

if TYPE_CHECKING:
    from ingestion_service.config import Settings

logger = structlog.get_logger(__name__)


class TimescaleWriter:
    """Batch-inserts prediction rows into the TimescaleDB hypertable."""

    def __init__(self, settings: Settings) -> None:
        self._db = DatabaseManager(settings.database_url)

    async def write_batch(self, payloads: list[EdgePredictionPayload]) -> int:
        """Insert a batch of prediction payloads into the predictions hypertable.

        Each payload can contain multiple predictions; each prediction becomes
        a separate row in the hypertable.

        Returns:
            Number of rows inserted.
        """
        rows = []
        for payload in payloads:
            for pred in payload.predictions:
                vector = pred.vector
                rows.append({
                    "time": payload.timestamp,
                    "device_id": payload.device_id,
                    "frame_idx": payload.frame_idx,
                    "class_name": pred.class_name,
                    "confidence": pred.confidence,
                    "vector_x": vector[0],
                    "vector_y": vector[1],
                    "vector_z": vector[2],
                    "inference_ms": payload.telemetry.inference_ms if payload.telemetry else None,
                })

        if not rows:
            return 0

        async with self._db.session() as session:
            await session.execute(
                text("""
                    INSERT INTO predictions
                        (time, device_id, frame_idx, class_name, confidence,
                         vector_x, vector_y, vector_z, inference_ms)
                    VALUES
                        (:time, :device_id, :frame_idx, :class_name, :confidence,
                         :vector_x, :vector_y, :vector_z, :inference_ms)
                """),
                rows,
            )

        logger.debug("rows_inserted", count=len(rows))
        return len(rows)

    async def close(self) -> None:
        await self._db.close()
