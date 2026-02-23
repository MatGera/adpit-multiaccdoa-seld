"""MQTT publisher for edge predictions and telemetry."""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlparse

import structlog

from edge_agent.config import EdgeConfig
from edge_agent.inference.postprocess import Prediction
from seld_common.mqtt_client import AsyncMQTTClient

logger = structlog.get_logger(__name__)


class MQTTPublisher:
    """Publishes SELD predictions and telemetry to MQTT broker (NanoMQ)."""

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config

        parsed = urlparse(config.mqtt_broker_url)
        self._client = AsyncMQTTClient(
            client_id=f"edge-{config.device_id}",
            broker_host=parsed.hostname or "localhost",
            broker_port=parsed.port or 1883,
        )

    async def connect(self) -> None:
        """Connect to the MQTT broker."""
        await self._client.connect()

    async def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        await self._client.disconnect()

    async def publish_predictions(
        self,
        predictions: list[Prediction],
        frame_idx: int,
        inference_ms: float,
        health: dict | None = None,
    ) -> None:
        """Publish prediction payload to MQTT.

        Args:
            predictions: List of decoded predictions.
            frame_idx: Current frame index.
            inference_ms: Inference latency in milliseconds.
            health: Optional health telemetry dict.
        """
        payload = {
            "device_id": self.config.device_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_idx": frame_idx,
            "predictions": [
                {
                    "class": p.class_name,
                    "confidence": p.confidence,
                    "vector": list(p.vector),
                }
                for p in predictions
            ],
        }

        if health:
            payload["telemetry"] = {
                "inference_ms": round(inference_ms, 2),
                **health,
            }

        await self._client.publish(
            topic=self.config.prediction_topic,
            payload=payload,
            qos=1,
        )
