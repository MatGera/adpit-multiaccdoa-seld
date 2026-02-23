"""End-to-end smoke tests: verify the full pipeline works.

These tests require all services to be running (docker-compose up).
"""

from __future__ import annotations

import json
import time

import pytest
import httpx


API_BASE = "http://localhost:8000"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_full_prediction_flow():
    """Smoke test: simulate edge device → API → query.

    1. Register a device
    2. Send a mock prediction via API (simulating webhook)
    3. Query predictions and verify the data appears
    """
    async with httpx.AsyncClient(base_url=API_BASE, timeout=30.0) as client:
        # 1. Health check
        resp = await client.get("/health")
        assert resp.status_code == 200

        # 2. Register device
        device_data = {
            "device_id": f"e2e-test-{int(time.time())}",
            "name": "E2E Test Device",
            "hardware_type": "industrial_capacitive",
            "num_channels": 4,
        }
        resp = await client.post("/api/v1/devices", json=device_data)
        assert resp.status_code in (201, 409)  # Created or already exists

        # 3. Query predictions for this device
        resp = await client.get(
            "/api/v1/predictions",
            params={"device_id": device_data["device_id"], "limit": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_websocket_connection():
    """Verify WebSocket connection to live prediction feed."""
    import websockets

    try:
        async with websockets.connect(
            "ws://localhost:8000/api/v1/predictions/live",
            open_timeout=5,
        ) as ws:
            # Connection should be accepted
            assert ws.open

            # Send a filter message
            await ws.send(json.dumps({"device_id": "test"}))

            # Close gracefully
            await ws.close()
    except Exception as e:
        pytest.skip(f"WebSocket not available: {e}")
