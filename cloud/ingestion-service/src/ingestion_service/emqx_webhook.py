"""EMQX webhook handler: alternative ingestion path for MQTT→Kafka bridging.

When EMQX is configured to use webhook actions, it POSTs prediction payloads
to this handler, which then publishes them to Kafka for the normal ingestion
pipeline to process.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from aiokafka import AIOKafkaProducer
import structlog

from seld_common.schemas import EdgePredictionPayload

if TYPE_CHECKING:
    from ingestion_service.config import Settings

logger = structlog.get_logger(__name__)


class EMQXWebhookBridge:
    """Receives EMQX webhook POSTs and publishes to Kafka."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        """Start the Kafka producer."""
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        await self._producer.start()
        logger.info("emqx_webhook_bridge_started")

    async def stop(self) -> None:
        """Stop the Kafka producer."""
        if self._producer:
            await self._producer.stop()

    async def handle_webhook(self, raw_payload: dict) -> dict:
        """Process an EMQX webhook payload and forward to Kafka.

        Args:
            raw_payload: The raw webhook body from EMQX.

        Returns:
            Status dict with processing result.
        """
        assert self._producer is not None

        try:
            # EMQX wraps the MQTT payload in its webhook format
            mqtt_payload = raw_payload.get("payload", raw_payload)
            if isinstance(mqtt_payload, str):
                mqtt_payload = json.loads(mqtt_payload)

            # Validate the prediction payload
            prediction = EdgePredictionPayload.model_validate(mqtt_payload)

            # Publish to Kafka
            await self._producer.send_and_wait(
                self._settings.kafka_topic,
                prediction.model_dump(mode="json"),
                key=prediction.device_id.encode("utf-8"),
            )

            logger.debug("webhook_forwarded",
                         device_id=prediction.device_id,
                         predictions=len(prediction.predictions))

            return {"status": "ok", "device_id": prediction.device_id}

        except Exception as e:
            logger.error("webhook_processing_error", error=str(e))
            return {"status": "error", "error": str(e)}
