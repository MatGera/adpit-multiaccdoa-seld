"""API Gateway configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://seld:seld_dev_password@localhost:5432/seld_db"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # gRPC backends
    spatial_grpc_host: str = "localhost:50051"
    semantic_grpc_host: str = "localhost:50052"
    device_grpc_host: str = "localhost:50053"

    # MQTT / Kafka
    emqx_host: str = "localhost"
    kafka_bootstrap_servers: str = "localhost:19092"

    # Auth
    jwt_secret: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    oauth2_issuer: str = ""

    # CORS
    cors_origins: list[str] = ["http://localhost:3000"]

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
