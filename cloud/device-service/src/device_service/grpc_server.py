"""Device gRPC servicer — handles device CRUD and provisioning RPCs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sqlalchemy import text

import structlog

from seld_common.db import DatabaseManager
from seld_common.schemas import DeviceConfig, DeviceHealth, DeviceStatusEnum

if TYPE_CHECKING:
    from device_service.config import Settings

logger = structlog.get_logger(__name__)


class DeviceServicer:
    """gRPC servicer for device registry operations.

    Implements the DeviceService RPCs defined in device.proto:
    - RegisterDevice
    - GetDevice
    - ListDevices
    - UpdateDevice
    - DeleteDevice
    - ProvisionCertificate
    - GetDeviceHealth
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._db = DatabaseManager(settings.database_url)

    async def register_device(self, device_config: dict) -> dict:
        """Register a new edge device in the registry."""
        config = DeviceConfig.model_validate(device_config)

        async with self._db.session() as session:
            # Check for duplicate
            result = await session.execute(
                text("SELECT device_id FROM devices WHERE device_id = :id"),
                {"id": config.device_id},
            )
            if result.fetchone():
                return {"error": "Device already registered", "device_id": config.device_id}

            await session.execute(
                text("""
                    INSERT INTO devices (device_id, name, hardware_type, num_channels,
                                         sample_rate, frame_length_ms, confidence_threshold,
                                         model_version, location, mqtt_topic_prefix,
                                         ota_enabled, status)
                    VALUES (:device_id, :name, :hardware_type, :num_channels,
                            :sample_rate, :frame_length_ms, :confidence_threshold,
                            :model_version, :location, :mqtt_topic_prefix,
                            :ota_enabled, :status)
                """),
                {
                    "device_id": config.device_id,
                    "name": config.name,
                    "hardware_type": config.hardware_type.value,
                    "num_channels": config.num_channels,
                    "sample_rate": config.sample_rate,
                    "frame_length_ms": config.frame_length_ms,
                    "confidence_threshold": config.confidence_threshold,
                    "model_version": config.model_version,
                    "location": json.dumps(config.location.model_dump()) if config.location else None,
                    "mqtt_topic_prefix": config.mqtt_topic_prefix,
                    "ota_enabled": config.ota_enabled,
                    "status": DeviceStatusEnum.PROVISIONING.value,
                },
            )

        logger.info("device_registered", device_id=config.device_id)
        return {"status": "created", "device_id": config.device_id}

    async def get_device(self, device_id: str) -> dict | None:
        """Fetch a device by ID."""
        async with self._db.session() as session:
            result = await session.execute(
                text("SELECT * FROM devices WHERE device_id = :id"),
                {"id": device_id},
            )
            row = result.fetchone()
            if not row:
                return None
            return dict(row._mapping)

    async def list_devices(self, status_filter: str | None = None) -> list[dict]:
        """List all devices, optionally filtered by status."""
        query = "SELECT * FROM devices"
        params: dict = {}
        if status_filter:
            query += " WHERE status = :status"
            params["status"] = status_filter
        query += " ORDER BY device_id"

        async with self._db.session() as session:
            result = await session.execute(text(query), params)
            return [dict(row._mapping) for row in result.fetchall()]

    async def update_device(self, device_id: str, updates: dict) -> dict:
        """Update specific fields of a device."""
        allowed = {
            "name", "confidence_threshold", "model_version",
            "location", "ota_enabled", "status",
        }
        filtered = {k: v for k, v in updates.items() if k in allowed}
        if not filtered:
            return {"error": "No valid fields to update"}

        set_clauses = [f"{k} = :{k}" for k in filtered]
        filtered["device_id"] = device_id

        async with self._db.session() as session:
            await session.execute(
                text(f"UPDATE devices SET {', '.join(set_clauses)} WHERE device_id = :device_id"),
                filtered,
            )

        logger.info("device_updated", device_id=device_id, fields=list(filtered.keys()))
        return {"status": "updated", "device_id": device_id}

    async def delete_device(self, device_id: str) -> dict:
        """Remove a device from the registry."""
        async with self._db.session() as session:
            await session.execute(
                text("DELETE FROM devices WHERE device_id = :id"),
                {"id": device_id},
            )
        logger.info("device_deleted", device_id=device_id)
        return {"status": "deleted", "device_id": device_id}
