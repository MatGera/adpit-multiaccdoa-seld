"""Unit tests for Pydantic schemas."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from seld_common.schemas import (
    DOAVector,
    PredictionItem,
    EdgePredictionPayload,
    DeviceConfig,
    DeviceHealth,
    HardwareTypeEnum,
    DeviceStatusEnum,
    SpatialHit,
    Point3D,
)


class TestDOAVector:
    def test_create_valid(self):
        v = DOAVector(x=0.5, y=-0.3, z=0.81)
        assert v.x == 0.5
        assert v.as_tuple() == (0.5, -0.3, 0.81)


class TestPredictionItem:
    def test_create_with_alias(self):
        p = PredictionItem(**{"class": "alarm", "confidence": 0.9, "vector": (0.1, 0.2, 0.3)})
        assert p.class_name == "alarm"
        assert p.confidence == 0.9

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            PredictionItem(**{"class": "x", "confidence": 1.5, "vector": (0, 0, 0)})

        with pytest.raises(Exception):
            PredictionItem(**{"class": "x", "confidence": -0.1, "vector": (0, 0, 0)})


class TestEdgePredictionPayload:
    def test_full_payload(self, sample_prediction_payload):
        payload = EdgePredictionPayload.model_validate(sample_prediction_payload)
        assert payload.device_id == "test-device-001"
        assert len(payload.predictions) == 2
        assert payload.telemetry is not None
        assert payload.telemetry.inference_ms == 12.5

    def test_minimal_payload(self):
        payload = EdgePredictionPayload(
            device_id="dev-001",
            timestamp=datetime.now(timezone.utc),
            frame_idx=0,
            predictions=[],
        )
        assert payload.telemetry is None
        assert payload.predictions == []


class TestDeviceConfig:
    def test_valid_config(self, sample_device_config):
        config = DeviceConfig.model_validate(sample_device_config)
        assert config.device_id == "test-device-001"
        assert config.hardware_type == HardwareTypeEnum.INDUSTRIAL_CAPACITIVE
        assert config.sample_rate == 48000

    def test_default_values(self):
        config = DeviceConfig(
            device_id="d1",
            name="Test",
            hardware_type=HardwareTypeEnum.INDUSTRIAL_CAPACITIVE,
            num_channels=4,
        )
        assert config.confidence_threshold == 0.5
        assert config.ota_enabled is True


class TestSpatialTypes:
    def test_spatial_hit(self):
        hit = SpatialHit(
            asset_id="wall-001",
            asset_name="North Wall",
            ifc_type="IfcWall",
            hit_point=Point3D(x=1.0, y=2.0, z=3.0),
            distance=5.5,
            confidence=0.95,
        )
        assert hit.distance == 5.5
        assert hit.hit_point.x == 1.0
