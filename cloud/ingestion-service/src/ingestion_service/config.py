"""Ingestion service configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    # Kafka
    kafka_bootstrap_servers: str = "localhost:19092"
    kafka_topic: str = "seld.predictions.raw"
    kafka_group_id: str = "ingestion-service"
    kafka_auto_offset_reset: str = "latest"

    # Database
    database_url: str = "postgresql+asyncpg://seld:seld_dev_password@localhost:5432/seld_db"

    # Batch settings
    batch_size: int = 100
    batch_timeout_ms: int = 500

    # WebSocket notification (API Gateway)
    api_gateway_ws_url: str = "http://localhost:8000"
    notify_api_gateway: bool = True
