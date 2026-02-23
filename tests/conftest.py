"""Shared test fixtures and configuration."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_prediction_payload() -> dict:
    """Sample edge prediction payload for testing."""
    return {
        "device_id": "test-device-001",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frame_idx": 42,
        "predictions": [
            {
                "class": "machinery_impact",
                "confidence": 0.87,
                "vector": [0.5, -0.3, 0.81],
            },
            {
                "class": "alarm",
                "confidence": 0.65,
                "vector": [-0.2, 0.9, 0.39],
            },
        ],
        "telemetry": {
            "inference_ms": 12.5,
            "cpu_temp": 55.0,
            "gpu_temp": 62.0,
            "mem_used_mb": 2048,
        },
    }


@pytest.fixture
def sample_device_config() -> dict:
    """Sample device configuration for testing."""
    return {
        "device_id": "test-device-001",
        "name": "Test Microphone Array",
        "hardware_type": "industrial_capacitive",
        "num_channels": 4,
        "sample_rate": 48000,
        "frame_length_ms": 100,
        "confidence_threshold": 0.5,
        "model_version": "v1.0.0-test",
        "location": {
            "building": "Building A",
            "floor": "Floor 1",
            "zone": "Zone A1",
        },
        "mqtt_topic_prefix": "dt/edge",
        "ota_enabled": True,
    }


@pytest.fixture
def api_base_url() -> str:
    """Base URL for API Gateway."""
    return os.getenv("API_BASE_URL", "http://localhost:8000")
