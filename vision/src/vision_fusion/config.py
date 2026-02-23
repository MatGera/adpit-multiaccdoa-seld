"""Vision fusion configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    # Redis for state caching
    redis_url: str = "redis://localhost:6379/1"

    # YOLO model
    yolo_model: str = "yolo11n.pt"
    yolo_confidence: float = 0.4
    yolo_iou: float = 0.5
    yolo_device: str = "0"  # GPU id or 'cpu'

    # Tracking
    tracker_type: str = "botsort"  # botsort | bytetrack
    track_buffer: int = 30  # frames to keep lost tracks

    # Camera
    camera_sources: str = ""  # comma-separated RTSP URLs or video paths
    fps_limit: int = 15  # Max frames processed per second

    # Homography
    homography_cache_ttl: int = 3600  # seconds

    # Mobile mapping
    point_cloud_storage: str = "/data/point_clouds"
