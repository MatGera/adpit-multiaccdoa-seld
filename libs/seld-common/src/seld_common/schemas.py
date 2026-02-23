"""Pydantic models mirroring @seld/shared-types Zod schemas."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


# --- DOA Vector ---
class DOAVector(BaseModel):
    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


# --- Prediction Item ---
class PredictionItem(BaseModel):
    class_name: str = Field(alias="class")
    confidence: float = Field(ge=0.0, le=1.0)
    vector: tuple[float, float, float]

    model_config = {"populate_by_name": True}


# --- Telemetry ---
class Telemetry(BaseModel):
    inference_ms: float
    cpu_temp: float
    gpu_temp: float
    mem_used_mb: int


# --- Edge Prediction Payload ---
class EdgePredictionPayload(BaseModel):
    device_id: str
    timestamp: datetime
    frame_idx: int = Field(ge=0)
    predictions: list[PredictionItem]
    telemetry: Telemetry | None = None


# --- Device Status ---
class DeviceStatusEnum(StrEnum):
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    PROVISIONING = "provisioning"


# --- Hardware Type ---
class HardwareTypeEnum(StrEnum):
    INDUSTRIAL_CAPACITIVE = "industrial_capacitive"
    INFRASTRUCTURE_PIEZOELECTRIC = "infrastructure_piezoelectric"


# --- Device Config ---
class DeviceLocation(BaseModel):
    building: str | None = None
    floor: str | None = None
    zone: str | None = None
    coordinates: DOAVector | None = None


class DeviceConfig(BaseModel):
    device_id: str
    name: str
    hardware_type: HardwareTypeEnum
    num_channels: int = Field(description="Must be 4 or 8")
    sample_rate: int = 48000
    frame_length_ms: int = 100
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    model_version: str | None = None
    location: DeviceLocation | None = None
    mqtt_topic_prefix: str = "dt/edge"
    ota_enabled: bool = True


# --- Device Health ---
class DeviceHealth(BaseModel):
    device_id: str
    status: DeviceStatusEnum
    last_seen: datetime | None = None
    uptime_seconds: float = 0
    cpu_temp: float = 0
    gpu_temp: float = 0
    mem_used_mb: int = 0
    mem_total_mb: int = 0
    inference_latency_ms: float = 0
    predictions_per_minute: float = 0
    model_version: str | None = None
    firmware_version: str | None = None


# --- Spatial Types ---
class Point3D(BaseModel):
    x: float
    y: float
    z: float


class SpatialHit(BaseModel):
    asset_id: str
    asset_name: str
    ifc_type: str
    hit_point: Point3D
    distance: float
    confidence: float = Field(ge=0.0, le=1.0)


class RaycastResponse(BaseModel):
    ray_origin: Point3D
    ray_direction: Point3D
    hits: list[SpatialHit]


class TriangulationResult(BaseModel):
    estimated_point: Point3D
    residual_error: float
    nearest_asset: SpatialHit | None = None
    contributing_devices: list[str]


# --- Semantic Types ---
class RetrievedChunk(BaseModel):
    content: str
    source: str
    page: int | None = None
    score: float


class Citation(BaseModel):
    source: str
    page: int | None = None
    quote: str


class SeverityEnum(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class LLMResponse(BaseModel):
    response_text: str
    citations: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0)
    asset_id: str | None = None
    recommended_actions: list[str]
    severity: SeverityEnum
