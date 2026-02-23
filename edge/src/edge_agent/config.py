"""Edge agent configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class EdgeConfig(BaseSettings):
    """Configuration for the edge agent, loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    # Device identity
    device_id: str = "ARRAY_01"

    # MQTT
    mqtt_broker_url: str = "mqtt://localhost:1883"
    mqtt_bridge_url: str = "mqtts://emqx.cloud:8883"
    mqtt_topic_prefix: str = "dt/edge"

    # Audio capture
    audio_device: str = "hw:1,0"
    sample_rate: int = 48000
    frame_length_ms: int = 100
    num_channels: int = 4
    mock_audio: bool = False

    # Inference
    tensorrt_engine_path: str = "/models/seld_fp16.engine"
    confidence_threshold: float = 0.5
    num_classes: int = 13
    num_tracks: int = 3

    # Cloud API
    cloud_api_url: str = "https://api.example.com"
    ota_check_interval_s: int = 300

    # Class names (STARSS23 default)
    class_names: list[str] = [
        "female_speech", "male_speech", "clapping", "telephone",
        "laughter", "domestic_sounds", "walk_footsteps", "door_open_close",
        "music", "musical_instrument", "water_tap", "bell", "knock",
    ]

    @property
    def frame_samples(self) -> int:
        """Number of samples per frame."""
        return int(self.sample_rate * self.frame_length_ms / 1000)

    @property
    def prediction_topic(self) -> str:
        return f"{self.mqtt_topic_prefix}/{self.device_id}/predictions"

    @property
    def telemetry_topic(self) -> str:
        return f"{self.mqtt_topic_prefix}/{self.device_id}/telemetry"
