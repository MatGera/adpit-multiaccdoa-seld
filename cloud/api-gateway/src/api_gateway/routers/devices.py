"""Devices router — CRUD operations for device registry."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api_gateway.deps import get_db_session
from seld_common.schemas import DeviceConfig, DeviceHealth

router = APIRouter()


@router.get("/devices")
async def list_devices(
    status: str | None = None,
    session: AsyncSession = Depends(get_db_session),
):
    """List all registered devices with optional status filter."""
    query = "SELECT * FROM devices"
    params = {}
    if status:
        query += " WHERE status = :status"
        params["status"] = status
    query += " ORDER BY device_id"

    result = await session.execute(text(query), params)
    rows = result.fetchall()

    return {
        "devices": [dict(row._mapping) for row in rows],
        "total": len(rows),
    }


@router.post("/devices", status_code=201)
async def register_device(
    device: DeviceConfig,
    session: AsyncSession = Depends(get_db_session),
):
    """Register a new edge device."""
    result = await session.execute(
        text("SELECT device_id FROM devices WHERE device_id = :id"),
        {"id": device.device_id},
    )
    if result.fetchone():
        raise HTTPException(status_code=409, detail="Device already registered")

    await session.execute(
        text("""
            INSERT INTO devices (device_id, name, hardware_type, num_channels,
                                 sample_rate, frame_length_ms, confidence_threshold,
                                 model_version, location, mqtt_topic_prefix, ota_enabled)
            VALUES (:device_id, :name, :hardware_type, :num_channels,
                    :sample_rate, :frame_length_ms, :confidence_threshold,
                    :model_version, :location, :mqtt_topic_prefix, :ota_enabled)
        """),
        {
            "device_id": device.device_id,
            "name": device.name,
            "hardware_type": device.hardware_type.value,
            "num_channels": device.num_channels,
            "sample_rate": device.sample_rate,
            "frame_length_ms": device.frame_length_ms,
            "confidence_threshold": device.confidence_threshold,
            "model_version": device.model_version,
            "location": device.location.model_dump() if device.location else None,
            "mqtt_topic_prefix": device.mqtt_topic_prefix,
            "ota_enabled": device.ota_enabled,
        },
    )
    return {"status": "created", "device_id": device.device_id}


@router.patch("/devices/{device_id}")
async def update_device(
    device_id: str,
    updates: dict,
    session: AsyncSession = Depends(get_db_session),
):
    """Update device configuration."""
    allowed_fields = {
        "name", "confidence_threshold", "model_version",
        "location", "ota_enabled", "status",
    }
    filtered = {k: v for k, v in updates.items() if k in allowed_fields}

    if not filtered:
        raise HTTPException(status_code=400, detail="No valid fields to update")

    set_clauses = [f"{k} = :{k}" for k in filtered]
    filtered["device_id"] = device_id

    await session.execute(
        text(f"UPDATE devices SET {', '.join(set_clauses)} WHERE device_id = :device_id"),
        filtered,
    )
    return {"status": "updated", "device_id": device_id}
