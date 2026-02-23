"""Integration tests for API Gateway endpoints."""

from __future__ import annotations

import pytest
import httpx


@pytest.fixture
def api_client(api_base_url: str):
    """Create an HTTP client for the API Gateway."""
    return httpx.AsyncClient(base_url=api_base_url, timeout=10.0)


@pytest.mark.asyncio
async def test_health_check(api_client: httpx.AsyncClient):
    """API Gateway health endpoint should respond with 200."""
    response = await api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "api-gateway"


@pytest.mark.asyncio
async def test_list_devices(api_client: httpx.AsyncClient):
    """GET /api/v1/devices should return a valid response."""
    response = await api_client.get("/api/v1/devices")
    assert response.status_code == 200
    data = response.json()
    assert "devices" in data
    assert isinstance(data["devices"], list)


@pytest.mark.asyncio
async def test_query_predictions(api_client: httpx.AsyncClient):
    """GET /api/v1/predictions should return a valid response."""
    response = await api_client.get("/api/v1/predictions", params={"limit": 10})
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "count" in data


@pytest.mark.asyncio
async def test_spatial_query(api_client: httpx.AsyncClient):
    """POST /api/v1/spatial/query should accept valid requests."""
    response = await api_client.post(
        "/api/v1/spatial/query",
        json={
            "device_id": "test-device-001",
            "direction": {"x": 0.5, "y": -0.3, "z": 0.81},
            "bim_model_id": "test-model",
        },
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_triangulation_requires_min_observations(api_client: httpx.AsyncClient):
    """POST /api/v1/spatial/triangulate should reject <2 observations."""
    response = await api_client.post(
        "/api/v1/spatial/triangulate",
        json={
            "observations": [{"device_id": "d1", "direction": {"x": 1, "y": 0, "z": 0}}],
            "bim_model_id": "test-model",
        },
    )
    assert response.status_code == 400
