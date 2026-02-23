"""Kafka consumer: reads prediction payloads and batches them for writing."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from aiokafka import AIOKafkaConsumer
import structlog

from seld_common.schemas import EdgePredictionPayload
from ingestion_service.writer import TimescaleWriter

if TYPE_CHECKING:
    from ingestion_service.config import Settings

logger = structlog.get_logger(__name__)


class PredictionConsumer:
    """Consumes prediction events from Kafka, batches, and writes to TimescaleDB."""

    def __init__(self, settings: Settings, writer: TimescaleWriter) -> None:
        self._settings = settings
        self._writer = writer
        self._consumer: AIOKafkaConsumer | None = None
        self._task: asyncio.Task | None = None
        self._batch: list[EdgePredictionPayload] = []

    async def start(self) -> None:
        """Start the Kafka consumer and the processing loop."""
        self._consumer = AIOKafkaConsumer(
            self._settings.kafka_topic,
            bootstrap_servers=self._settings.kafka_bootstrap_servers,
            group_id=self._settings.kafka_group_id,
            auto_offset_reset=self._settings.kafka_auto_offset_reset,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            enable_auto_commit=False,
        )
        await self._consumer.start()
        logger.info("kafka_consumer_started", topic=self._settings.kafka_topic)
        self._task = asyncio.create_task(self._consume_loop())

    async def stop(self) -> None:
        """Stop the consumer gracefully."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._consumer:
            await self._consumer.stop()
            logger.info("kafka_consumer_stopped")

    async def _consume_loop(self) -> None:
        """Main consumer loop: batch messages and flush periodically."""
        assert self._consumer is not None

        try:
            async for message in self._consumer:
                try:
                    payload = EdgePredictionPayload.model_validate(message.value)
                    self._batch.append(payload)

                    if len(self._batch) >= self._settings.batch_size:
                        await self._flush_batch()
                        await self._consumer.commit()
                except Exception as e:
                    logger.error("message_parse_error",
                                 error=str(e),
                                 offset=message.offset,
                                 partition=message.partition)
        except asyncio.CancelledError:
            if self._batch:
                await self._flush_batch()
            raise

    async def _flush_batch(self) -> None:
        """Write the current batch to TimescaleDB."""
        if not self._batch:
            return

        batch = self._batch.copy()
        self._batch.clear()

        try:
            count = await self._writer.write_batch(batch)
            logger.info("batch_flushed", count=count)
        except Exception as e:
            logger.error("batch_write_error", error=str(e), batch_size=len(batch))
            raise
