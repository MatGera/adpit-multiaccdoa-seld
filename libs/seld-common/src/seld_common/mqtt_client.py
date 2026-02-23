"""Async MQTT 5.0 client wrapper using gmqtt."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

from gmqtt import Client as MQTTClient
from gmqtt.mqtt.constants import MQTTv50

import structlog

logger = structlog.get_logger(__name__)


class AsyncMQTTClient:
    """Thin async wrapper around gmqtt for MQTT 5.0 communication."""

    def __init__(
        self,
        client_id: str,
        broker_host: str,
        broker_port: int = 1883,
        username: str | None = None,
        password: str | None = None,
        tls: bool = False,
    ) -> None:
        self._client_id = client_id
        self._broker_host = broker_host
        self._broker_port = broker_port
        self._username = username
        self._password = password
        self._tls = tls
        self._client: MQTTClient | None = None
        self._message_handlers: dict[str, Callable[..., Any]] = {}
        self._connected = asyncio.Event()

    async def connect(self) -> None:
        """Connect to the MQTT broker."""
        self._client = MQTTClient(self._client_id)

        if self._username and self._password:
            self._client.set_auth_credentials(self._username, self._password)

        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect

        await self._client.connect(
            self._broker_host,
            self._broker_port,
            version=MQTTv50,
        )
        await self._connected.wait()
        logger.info(
            "mqtt_connected",
            broker=f"{self._broker_host}:{self._broker_port}",
            client_id=self._client_id,
        )

    async def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        if self._client:
            await self._client.disconnect()
            logger.info("mqtt_disconnected", client_id=self._client_id)

    async def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        qos: int = 1,
        retain: bool = False,
    ) -> None:
        """Publish a JSON payload to a topic."""
        if not self._client:
            raise RuntimeError("MQTT client not connected")

        message = json.dumps(payload, default=str)
        self._client.publish(topic, message, qos=qos, retain=retain)
        logger.debug("mqtt_published", topic=topic, size=len(message))

    async def subscribe(
        self,
        topic: str,
        handler: Callable[..., Any],
        qos: int = 1,
    ) -> None:
        """Subscribe to a topic with a message handler."""
        if not self._client:
            raise RuntimeError("MQTT client not connected")

        self._message_handlers[topic] = handler
        self._client.subscribe(topic, qos=qos)
        logger.info("mqtt_subscribed", topic=topic)

    def _on_connect(self, client: Any, flags: Any, rc: int, properties: Any) -> None:
        self._connected.set()

    def _on_disconnect(self, client: Any, packet: Any, exc: Exception | None = None) -> None:
        self._connected.clear()
        logger.warning("mqtt_disconnected_unexpected", exc=str(exc) if exc else None)

    async def _on_message(
        self, client: Any, topic: str, payload: bytes, qos: int, properties: Any
    ) -> None:
        try:
            data = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.error("mqtt_invalid_payload", topic=topic)
            return

        # Find matching handler
        for pattern, handler in self._message_handlers.items():
            if self._topic_matches(pattern, topic):
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(topic, data)
                    else:
                        handler(topic, data)
                except Exception:
                    logger.exception("mqtt_handler_error", topic=topic)
                return

        logger.debug("mqtt_unhandled_message", topic=topic)

    @staticmethod
    def _topic_matches(pattern: str, topic: str) -> bool:
        """Simple MQTT topic pattern matching with + and # wildcards."""
        pattern_parts = pattern.split("/")
        topic_parts = topic.split("/")

        for i, p in enumerate(pattern_parts):
            if p == "#":
                return True
            if i >= len(topic_parts):
                return False
            if p != "+" and p != topic_parts[i]:
                return False

        return len(pattern_parts) == len(topic_parts)
