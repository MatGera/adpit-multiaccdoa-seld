"""Ingestion service entry point — Kafka consumer to TimescaleDB writer."""

from __future__ import annotations

import asyncio
import signal
import json

import structlog

from ingestion_service.config import Settings
from ingestion_service.consumer import PredictionConsumer
from ingestion_service.writer import TimescaleWriter

logger = structlog.get_logger(__name__)


async def main() -> None:
    settings = Settings()

    writer = TimescaleWriter(settings)
    consumer = PredictionConsumer(settings, writer)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("shutdown_signal_received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    logger.info("ingestion_service_starting",
                kafka=settings.kafka_bootstrap_servers,
                topic=settings.kafka_topic)

    try:
        await consumer.start()
        await stop_event.wait()
    finally:
        await consumer.stop()
        await writer.close()
        logger.info("ingestion_service_stopped")


if __name__ == "__main__":
    asyncio.run(main())
